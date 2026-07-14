import pandas as pd

from Processing.analysis.forecast_engine import run_forecasting


def _history(task_count=3, updates=15):
    rows = []
    patterns = {
        0: [update // 2 for update in range(updates)],
        1: [5] * updates,
        2: [max(8, 14 - update // 2) for update in range(updates)],
    }
    for task_index in range(task_count):
        for update in range(updates):
            rows.append(
                {
                    "task_id": f"T-{task_index + 1}",
                    "task_name": f"Task {task_index + 1}",
                    "update_phase": f"Update_{update + 1:03d}",
                    "baseline_end": "2026-09-30",
                    "slip_days": patterns[task_index][update],
                    "is_critical": task_index == 0,
                }
            )
    return pd.DataFrame(rows)


def test_real_forecast_is_validated_and_deterministic():
    forecasts, failed = run_forecasting(_history(), "Test")

    assert failed == []
    assert len(forecasts) == 3
    assert forecasts.predicted_end_date.notna().all()
    assert forecasts.forecast_confidence.between(0, 1).all()
    assert not forecasts.model_type.str.contains("synthetic|demo", case=False).any()
    assert (forecasts.validation_windows == 5).all()
    assert (forecasts.forecast_p80_error_days >= 1).all()

    scorecard = forecasts.attrs["model_evaluation"]
    assert len(scorecard) == 4
    assert scorecard.champion.sum() == 1
    assert scorecard.validation_predictions.min() == 15
    assert scorecard.iloc[0].mae_days <= scorecard.iloc[-1].mae_days


def test_sparse_history_fails_closed_without_fake_forecasts():
    forecasts, failed = run_forecasting(_history(task_count=1, updates=4), "Sparse")

    assert forecasts.empty
    assert failed == ["T-1"]
