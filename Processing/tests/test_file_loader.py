import pandas as pd

from Processing.ingestion.file_loader import load_project_files


def test_load_sample_project_fixtures(tmp_path):
    """Load CSV and Excel inputs from a temporary project folder."""
    fixture_dir = tmp_path / "sample_project"
    fixture_dir.mkdir()

    csv_frame = pd.DataFrame(
        {
            "task_code": ["TKA", "TKB"],
            "task_name": ["Task A", "Task B"],
            "update_phase": ["update_1", "update_1"],
        }
    )
    excel_frame = pd.DataFrame(
        {
            "task_code": ["TKC", "TKD"],
            "some_value": [10, 20],
            "update_phase": ["update_2", "update_2"],
        }
    )
    csv_frame.to_csv(fixture_dir / "sample.csv", index=False)
    excel_frame.to_excel(fixture_dir / "sample.xlsx", index=False)

    df = load_project_files(str(fixture_dir))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "file_name" in df.columns
    assert "task_code" in df.columns
    assert "some_value" in df.columns
    assert "sample.csv" in df["file_name"].unique()
    assert "sample.xlsx" in df["file_name"].unique()
    assert df.loc[df["file_name"] == "sample.csv", "task_code"].iloc[0] == "TKA"
    assert df.loc[df["file_name"] == "sample.xlsx", "some_value"].iloc[0] == 10
