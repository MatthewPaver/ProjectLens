import pandas as pd
from datetime import datetime, timedelta
from Processing.analysis.changepoint_detector import detect_changepoints

def test_change_detection():
    """Tests basic execution of the change point detection function.
    
    Creates a simple DataFrame with a time series (using 'end_date') 
    and checks if the `detect_changepoints` function runs without errors 
    and returns a DataFrame with the expected columns for detected points.
    Note: This test does not validate the *accuracy* of the detection, 
    only the function's execution and output format.
    """
    df = pd.DataFrame({
        "task_id": ["c"] * 6,
        "task_name": ["Task C"] * 6,
        "slip_days": [0, 0, 0, 10, 11, 10],
        "actual_finish": [datetime.today() + timedelta(days=i * 7) for i in range(6)],
        "update_phase": [f"update_{i}" for i in range(6)],
        "project_name": ["Demo"] * 6
    })
    result = detect_changepoints(df, project_name="Demo")
    
    assert isinstance(result, pd.DataFrame)
    
    expected_cols = ['project_name', 'task_id', 'task_name', 'update_phase', 'slip_days']
    if not result.empty:
        assert all(col in result.columns for col in expected_cols)
