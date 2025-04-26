from Processing.analysis.slippage_analysis import analyse_slippages
import pandas as pd

def test_slippage():
    df = pd.DataFrame({
        "task_id": ["t1", "t2"],
        "baseline_end": pd.to_datetime(["2023-01-01", "2023-01-10"]),
        "end_date": pd.to_datetime(["2023-01-05", "2023-01-08"]),
        "percent_complete": [20, 80],
        "project_name": ["TestProject"] * 2,
        "update_phase": ["update_1"] * 2
    })
    result = analyse_slippages(df)
    assert "slip_days" in result.columns
    assert "change_type" in result.columns
