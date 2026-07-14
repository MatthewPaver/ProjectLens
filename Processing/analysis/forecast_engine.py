#!/usr/bin/env python3
"""Validated schedule-slippage forecasting for ProjectLens.

The engine forecasts the next reported slippage, then converts that value back to
an end date. Models compete on rolling-origin validation windows. ProjectLens uses
the best model for each task and keeps a simple last-value model as the benchmark.
There are no synthetic or randomly generated forecasts in this module.
"""

from __future__ import annotations

import logging
import math
import re

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive, RandomWalkWithDrift


MODEL_LABELS = {
    "Naive": "Last reported position",
    "RWD": "Random walk with drift",
    "AutoETS": "Automatic exponential smoothing",
    "AutoARIMA": "Automatic ARIMA",
}
MODEL_COLUMNS = list(MODEL_LABELS)
MIN_HISTORY = 7
MAX_VALIDATION_WINDOWS = 5


def _phase_number(value: object) -> float:
    match = re.search(r"(\d+)(?!.*\d)", str(value))
    return float(match.group(1)) if match else np.nan


def _prepare_series(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"task_id", "update_phase"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    prepared = df.copy()
    prepared["ds"] = prepared["update_phase"].map(_phase_number)
    if prepared["ds"].isna().any():
        prepared["ds"] = prepared.groupby("task_id").cumcount() + 1

    if "slip_days" in prepared.columns:
        prepared["y"] = pd.to_numeric(prepared["slip_days"], errors="coerce")
    else:
        actual_col = next(
            (name for name in ("actual_finish", "end_date") if name in prepared.columns),
            None,
        )
        baseline_col = next(
            (name for name in ("baseline_end", "baseline_end_date") if name in prepared.columns),
            None,
        )
        if not actual_col or not baseline_col:
            raise ValueError("forecasting needs slip_days or actual and baseline finish dates")
        prepared["y"] = (
            pd.to_datetime(prepared[actual_col], errors="coerce")
            - pd.to_datetime(prepared[baseline_col], errors="coerce")
        ).dt.total_seconds() / 86_400

    prepared = (
        prepared.dropna(subset=["task_id", "ds", "y"])
        .sort_values(["task_id", "ds"])
        .drop_duplicates(["task_id", "ds"], keep="last")
    )
    prepared["ds"] = prepared["ds"].astype(int)
    context = prepared.copy()
    series = prepared.rename(columns={"task_id": "unique_id"})[
        ["unique_id", "ds", "y"]
    ]
    return series, context


def _normalise_index(frame: pd.DataFrame, id_name: str = "task_id") -> pd.DataFrame:
    result = frame.reset_index()
    if "unique_id" in result.columns:
        return result.rename(columns={"unique_id": id_name})
    if "index" in result.columns:
        return result.rename(columns={"index": id_name})
    if id_name not in result.columns:
        result.insert(0, id_name, "portfolio")
    return result


def _model_scorecard(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_COLUMNS:
        valid = cv[["y", model]].dropna()
        errors = valid[model] - valid["y"]
        lo_col, hi_col = f"{model}-lo-80", f"{model}-hi-80"
        coverage = np.nan
        if lo_col in cv and hi_col in cv:
            bounded = cv[["y", lo_col, hi_col]].dropna()
            if not bounded.empty:
                coverage = ((bounded.y >= bounded[lo_col]) & (bounded.y <= bounded[hi_col])).mean() * 100
        rows.append(
            {
                "model_type": MODEL_LABELS[model],
                "model_key": model,
                "mae_days": round(float(errors.abs().mean()), 2),
                "median_error_days": round(float(errors.abs().median()), 2),
                "bias_days": round(float(errors.mean()), 2),
                "p80_error_days": round(float(errors.abs().quantile(0.8)), 2),
                "interval_coverage_80": round(float(coverage), 1) if pd.notna(coverage) else np.nan,
                "validation_predictions": int(len(valid)),
                "tasks_evaluated": int(cv.loc[valid.index, "task_id"].nunique()),
            }
        )
    scorecard = pd.DataFrame(rows).sort_values(["mae_days", "bias_days"], key=lambda s: s.abs())
    scorecard["champion"] = False
    if not scorecard.empty:
        scorecard.loc[scorecard.index[0], "champion"] = True
    scorecard["validation_method"] = "Rolling origin, one update ahead"
    return scorecard.reset_index(drop=True)


def _confidence(mae: float, history: int, error_tolerance_days: float = 14.0) -> float:
    """Map observed absolute error and history depth to an inspectable score.

    The fixed day tolerance prevents a very late task from appearing more certain
    merely because its error is small relative to a large existing delay.
    """

    accuracy = math.exp(-mae / error_tolerance_days)
    history_factor = min(1.0, history / 12)
    return round(max(0.25, min(0.95, 0.25 + 0.70 * accuracy * history_factor)), 3)


def _date_from_slip(context: pd.DataFrame, task_id: str, slip: float) -> pd.Timestamp:
    task = context[context.task_id == task_id].sort_values("ds")
    for column in ("baseline_end", "baseline_end_date"):
        if column in task.columns:
            baseline = pd.to_datetime(task[column], errors="coerce").dropna()
            if not baseline.empty:
                return (baseline.iloc[-1] + pd.to_timedelta(slip, unit="D")).normalize()
    for column in ("actual_finish", "end_date"):
        if column in task.columns:
            finish = pd.to_datetime(task[column], errors="coerce").dropna()
            if not finish.empty:
                latest_slip = float(task.y.iloc[-1])
                return (finish.iloc[-1] + pd.to_timedelta(slip - latest_slip, unit="D")).normalize()
    return pd.NaT


def run_forecasting(df: pd.DataFrame, project_name: str) -> tuple[pd.DataFrame, list]:
    """Forecast every task and attach the portfolio model scorecard to ``attrs``."""

    logger = logging.getLogger(__name__)
    if df is None or df.empty:
        return pd.DataFrame(), []

    try:
        series, context = _prepare_series(df)
    except ValueError as exc:
        logger.error("[%s] Forecasting input rejected: %s", project_name, exc)
        return pd.DataFrame(), list(df.get("task_id", pd.Series(dtype=str)).dropna().unique())

    counts = series.groupby("unique_id").size()
    valid_ids = counts[counts >= MIN_HISTORY].index
    failed_ids = counts[counts < MIN_HISTORY].index.tolist()
    model_input = series[series.unique_id.isin(valid_ids)].copy()
    if model_input.empty:
        return pd.DataFrame(), failed_ids

    windows = min(MAX_VALIDATION_WINDOWS, int(counts.loc[valid_ids].min()) - 2)
    models = [
        Naive(),
        RandomWalkWithDrift(),
        AutoETS(season_length=1),
        AutoARIMA(season_length=1),
    ]
    engine = StatsForecast(models=models, freq=1, n_jobs=1)

    try:
        cv = _normalise_index(
            engine.cross_validation(
                df=model_input,
                h=1,
                n_windows=windows,
                step_size=1,
                level=[80],
            )
        )
        scorecard = _model_scorecard(cv)
        forecasts = _normalise_index(engine.forecast(df=model_input, h=1, level=[80]))
    except Exception as exc:
        logger.exception("[%s] Validated model suite failed: %s", project_name, exc)
        return pd.DataFrame(), list(counts.index)

    task_errors: dict[str, pd.DataFrame] = {}
    for task_id, task_cv in cv.groupby("task_id"):
        rows = []
        for model in MODEL_COLUMNS:
            error = (task_cv[model] - task_cv.y).dropna()
            if not error.empty:
                rows.append(
                    {
                        "model": model,
                        "mae": float(error.abs().mean()),
                        "p80": float(error.abs().quantile(0.8)),
                        "bias": float(error.mean()),
                        "windows": len(error),
                    }
                )
        task_errors[str(task_id)] = pd.DataFrame(rows).sort_values(["mae", "bias"], key=lambda s: s.abs())

    context_latest = context.sort_values("ds").drop_duplicates("task_id", keep="last").set_index("task_id")
    results = []
    for _, row in forecasts.iterrows():
        task_id = str(row.task_id)
        ranking = task_errors[task_id]
        winner = ranking.iloc[0]
        model = str(winner.model)
        predicted_slip = float(row[model])
        band = max(1.0, float(winner.p80))
        latest = context_latest.loc[task_id]
        latest_slip = float(latest.y)
        confidence = _confidence(float(winner.mae), int(counts.loc[task_id]))
        direction = predicted_slip - latest_slip
        if abs(direction) < 0.5:
            movement = "stable"
        elif direction > 0:
            movement = f"worsening by {direction:.1f} days"
        else:
            movement = f"recovering by {abs(direction):.1f} days"
        results.append(
            {
                "task_id": task_id,
                "task_name": latest.get("task_name", f"Task {task_id}"),
                "update_phase": f"after {latest.update_phase}",
                "predicted_end_date": _date_from_slip(context, task_id, predicted_slip),
                "prediction_low_date": _date_from_slip(context, task_id, predicted_slip - band),
                "prediction_high_date": _date_from_slip(context, task_id, predicted_slip + band),
                "predicted_slip_days": round(predicted_slip, 1),
                "forecast_confidence": confidence,
                "model_type": MODEL_LABELS[model],
                "low_confidence_flag": confidence < 0.7,
                "forecast_mae_days": round(float(winner.mae), 2),
                "forecast_p80_error_days": round(band, 2),
                "validation_windows": int(winner.windows),
                "confidence_reason": (
                    f"{int(winner.windows)} rolling tests; typical error {float(winner.mae):.1f} days; "
                    f"next update is {movement}"
                ),
            }
        )

    output = pd.DataFrame(results).sort_values(
        ["low_confidence_flag", "predicted_slip_days"], ascending=[False, False]
    )
    output.attrs["model_evaluation"] = scorecard
    logger.info(
        "[%s] Produced %s validated forecasts. Portfolio champion: %s (MAE %.2f days).",
        project_name,
        len(output),
        scorecard.iloc[0].model_type,
        scorecard.iloc[0].mae_days,
    )
    return output.reset_index(drop=True), failed_ids
