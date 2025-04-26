import pandas as pd
from datetime import datetime, timedelta
from Processing.analysis.changepoint_detector import detect_changepoints

def test_change_detection():
    df = pd.DataFrame({
        "task_id": ["c"] * 6,
        "end_date": [datetime.today() + timedelta(days=i * 7) for i in range(6)],
        "update_phase": [f"update_{i}" for i in range(6)],
        "project_name": ["Demo"] * 6
    })
    result = detect_changepoints(df)
    assert isinstance(result, pd.DataFrame)
