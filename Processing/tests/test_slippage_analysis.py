from Processing.analysis.slippage_analysis import run_slippage_analysis
import pandas as pd

def test_slippage():
    """Tests basic slippage calculation and output columns.
    
    Provides a minimal DataFrame with known baseline and actual end dates
    and verifies that the `analyse_slippages` function runs and produces
    the expected 'slip_days' and 'change_type' columns with plausible values.
    """
    df = pd.DataFrame({
        # Task 1: 4 days slippage
        # Task 2: -2 days slippage (finished early)
        "task_id": ["t1", "t1", "t2"], 
        "task_name": ["Task 1", "Task 1", "Task 2"],
        "baseline_end_date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-10"]),
        "actual_finish": pd.to_datetime(["2023-01-03", "2023-01-05", "2023-01-08"]),
        "update_phase": ["update_1", "update_2", "update_1"],
        "project_name": ["TestProject"] * 3,
        "is_critical": [False, False, False],
    })
    # Expected slip_days: [2, 4, -2]
    # Expected change_in_slip: [NaN, 2, NaN]
    # Expected change_type: [Initial, Slipped Further, Initial]
    result = run_slippage_analysis(df, project_name="TestProject")
    
    assert isinstance(result, pd.DataFrame)
    assert "slip_days" in result.columns
    assert "change_in_slip" in result.columns
    assert "change_type" in result.columns
    
    # Check specific calculated values for correctness
    assert result.loc[result['task_id'] == 't1', 'slip_days'].tolist() == [2, 4]
    assert result.loc[result['task_id'] == 't2', 'slip_days'].iloc[0] == -2
    assert pd.isna(result.loc[result['task_id'] == 't1', 'change_in_slip'].iloc[0])
    assert result.loc[result['task_id'] == 't1', 'change_in_slip'].iloc[1] == 2 # 4 - 2 = 2
    # Verify change_type based on slip changes
    assert result.loc[(result['task_id'] == 't1') & (result['update_phase'] == 'update_1'), 'change_type'].iloc[0] == 'Initial'
    assert result.loc[(result['task_id'] == 't1') & (result['update_phase'] == 'update_2'), 'change_type'].iloc[0] == 'Slipped Further'
    assert result.loc[result['task_id'] == 't2', 'change_type'].iloc[0] == 'Initial' # Finished early, but still first record
