"""Guards against fabricated numbers in pipeline outputs.

ProjectLens reports findings from the schedule's own evidence. These tests pin
that rule for the legacy pipeline: no change point is invented when none is
detected, milestone deviation is computed from baseline dates or left as NA,
and no shipped module draws unseeded random numbers.
"""

import ast
from pathlib import Path

import pandas as pd

from Processing.analysis.changepoint_detector import (
    CHANGEPOINT_OUTPUT_COLUMNS,
    detect_change_points,
)
from Processing.output.output_writer import milestone_deviation_percentage, write_outputs

ROOT = Path(__file__).resolve().parents[2]

# Seeded Monte Carlo over user-supplied triangular estimates: the same inputs and
# seed always give the same result, and nothing is presented as observed data.
SEEDED_RANDOM_ALLOWED = {"Processing/analysis/decision_support.py"}


def _flat_series(task_id: str, value: int, points: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "task_id": [task_id] * points,
        "task_name": [f"Task {task_id}"] * points,
        "update_phase": [f"update_{i}" for i in range(points)],
        "slip_days": [value] * points,
    })


def test_no_change_points_detected_means_none_reported():
    df = pd.concat([_flat_series("A", 0), _flat_series("B", 3)], ignore_index=True)

    result = detect_change_points(df, project_name="Flat")

    assert result.empty
    assert list(result.columns) == CHANGEPOINT_OUTPUT_COLUMNS


def test_change_points_only_carry_observed_slip_values():
    df = pd.DataFrame({
        "task_id": ["C"] * 8,
        "task_name": ["Task C"] * 8,
        "update_phase": [f"update_{i}" for i in range(8)],
        "slip_days": [0, 0, 0, 0, 30, 30, 30, 30],
    })

    first = detect_change_points(df.copy(), project_name="Step")
    second = detect_change_points(df.copy(), project_name="Step")

    pd.testing.assert_frame_equal(first, second)
    assert set(first["slip_days"]).issubset(set(df["slip_days"]))


def test_deviation_is_slip_over_baseline_duration():
    milestones = pd.DataFrame({
        "task_id": ["M1", "M2", "M3", "M4"],
        "slip_days": [10, -5, 4, None],
        "baseline_start_date": ["2026-01-01", "2026-01-01", "2026-03-01", "2026-01-01"],
        "baseline_end_date": ["2026-01-21", "2026-01-11", "2026-03-01", "2026-01-31"],
    })

    result = milestone_deviation_percentage(milestones, cleaned_df=None)

    assert result.iloc[0] == 50.0   # 10 / 20 days
    assert result.iloc[1] == -50.0  # 5 days early on a 10-day baseline
    assert pd.isna(result.iloc[2])  # zero-duration milestone: no percentage
    assert pd.isna(result.iloc[3])  # no slip recorded: no percentage


def test_deviation_uses_latest_cleaned_baseline_and_is_na_without_one():
    milestones = pd.DataFrame({
        "task_id": ["M1", "M2"],
        "slip_days": [6, 6],
        "baseline_end_date": ["2026-02-01", "2026-02-01"],
    })
    cleaned = pd.DataFrame({
        "task_id": ["M1", "M1"],
        "update_phase": ["update_1", "update_2"],
        "baseline_start_date": ["2026-01-01", "2026-01-02"],
    })

    result = milestone_deviation_percentage(milestones, cleaned)

    assert result.iloc[0] == 20.0  # 6 / 30 days, from the latest update
    assert pd.isna(result.iloc[1])  # M2 has no baseline start anywhere


def test_written_milestone_csv_is_deterministic_and_never_invents_deviation(tmp_path):
    milestones = pd.DataFrame({
        "task_id": ["M1", "M2"],
        "task_name": ["Design freeze", "Handover"],
        "actual_finish": ["2026-01-31", "2026-02-10"],
        "baseline_end_date": ["2026-01-21", "2026-02-10"],
        "update_phase": ["update_1", "update_1"],
    })
    slippages = pd.DataFrame({
        "task_id": ["M1", "M2"],
        "task_name": ["Design freeze", "Handover"],
        "update_phase": ["update_1", "update_1"],
        "slip_days": [10, 0],
        "severity_score": [5.0, 0.0],
        "change_type": ["Slipped", "On time"],
    })
    cleaned = pd.DataFrame({
        "task_id": ["M1"],
        "task_name": ["Design freeze"],
        "update_phase": ["update_1"],
        "baseline_start_date": ["2026-01-01"],
    })
    results = {"milestones": milestones, "slippages": slippages}

    write_outputs(str(tmp_path / "run1"), "Demo", cleaned, results)
    write_outputs(str(tmp_path / "run2"), "Demo", cleaned, results)

    first = (tmp_path / "run1" / "milestone_analysis.csv").read_text()
    second = (tmp_path / "run2" / "milestone_analysis.csv").read_text()
    assert first == second

    written = pd.read_csv(tmp_path / "run1" / "milestone_analysis.csv")
    assert written.loc[0, "deviation_percentage"] == 50.0
    assert pd.isna(written.loc[1, "deviation_percentage"])  # no baseline start for M2


def _uses_random(tree: ast.AST) -> list[str]:
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name in {"random", "secrets"}]
        elif isinstance(node, ast.ImportFrom) and node.module in {"random", "numpy.random"}:
            hits.append(node.module)
        elif isinstance(node, ast.Attribute) and node.attr == "random":
            hits.append(ast.unparse(node))
        elif isinstance(node, ast.Name) and node.id == "random":
            hits.append(node.id)
    return hits


def test_shipped_python_modules_do_not_draw_random_numbers():
    shipped = [
        p for p in (ROOT / "Processing").rglob("*.py")
        if "tests" not in p.relative_to(ROOT).parts
    ]
    assert shipped, "expected Processing modules to scan"

    offenders = {}
    for path in shipped:
        rel = path.relative_to(ROOT).as_posix()
        if rel in SEEDED_RANDOM_ALLOWED:
            continue
        hits = _uses_random(ast.parse(path.read_text(encoding="utf-8")))
        if hits:
            offenders[rel] = hits

    assert offenders == {}


def test_seeded_simulation_is_reproducible():
    from Processing.analysis.decision_support import RiskDriver, simulate_completion_risk

    drivers = [RiskDriver("R1", "Late design", "design", 0.5, 2, 5, 10)]
    assert simulate_completion_risk(drivers, seed=7) == simulate_completion_risk(drivers, seed=7)
