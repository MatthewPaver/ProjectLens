import pandas as pd
import unittest
from unittest.mock import patch, MagicMock
from Processing.core.data_cleaning import clean_dataframe

def test_cleaning_logic():
    """Tests the core logic of `clean_dataframe` using mocks.
    
    This test focuses on verifying the interaction with a mocked SchemaManager
    and checking the post-standardisation logic within `clean_dataframe`,
    such as boolean conversion and status inference.
    It does *not* test the `SchemaManager` itself, only that `clean_dataframe`
    calls its methods as expected and processes the returned (mocked) data.
    """
    # Create raw input data for the test
    raw_data = {
        "Task Name": ["A", "B"],
        "End Date": ["2023-01-01", "2023-01-10"],
        "Baseline End Date": ["2022-12-31", "2023-01-05"],
        "Percent Complete": ["50%", "100"] # Mix types to simulate real data
    }
    df_input = pd.DataFrame(raw_data)

    # Mock the SchemaManager class used within clean_dataframe
    with patch('Processing.core.data_cleaning.SchemaManager') as MockSchemaManager:
        # Configure the mock instance that will be created inside clean_dataframe
        mock_instance = MagicMock() # Create an instance mock
        MockSchemaManager.return_value = mock_instance # When SchemaManager() is called, return our mock instance

        # --- Configure return values of the mock instance's methods --- 
        # 1. Simulate standardise_columns: returns a DataFrame with standardised names
        #    (We'll simulate renaming based on common patterns)
        df_standardised = pd.DataFrame({
            "task_name": ["A", "B"],
            "actual_finish": ["2023-01-01", "2023-01-10"], # Assume End Date -> actual_finish
            "baseline_end_date": ["2022-12-31", "2023-01-05"],
            "percent_complete": ["50%", "100"]
        })
        mock_instance.standardise_columns.return_value = df_standardised

        # 2. Simulate convert_data_types: returns df with types converted (esp. percent_complete)
        df_converted = df_standardised.copy() # Start from the standardised structure
        # Simulate converting percent_complete to numeric, handling potential errors like '%'
        df_converted['percent_complete'] = pd.to_numeric(df_converted['percent_complete'].astype(str).str.replace('%', ''), errors='coerce')
        mock_instance.convert_data_types.return_value = df_converted

        # 3. Simulate enforce_not_null: For this test, assume it just returns the df
        mock_instance.enforce_not_null.return_value = df_converted
        
        # --- Run the function with the fully configured mock ---
        cleaned = clean_dataframe(df_input, schema_type="tasks", project_name="UnitTest")

        # --- Assertions ---
        # Assert that methods on the mock instance were called
        mock_instance.standardise_columns.assert_called_once()
        mock_instance.convert_data_types.assert_called_once()
        mock_instance.enforce_not_null.assert_called_once()

        # Assertions on the final cleaned DataFrame
        assert isinstance(cleaned, pd.DataFrame)
        # Check for columns expected after standardisation (based on mock setup)
        assert "task_name" in cleaned.columns
        assert "actual_finish" in cleaned.columns
        assert "baseline_end_date" in cleaned.columns
        assert "percent_complete" in cleaned.columns
        # Check for columns added later in clean_dataframe
        assert "task_id" in cleaned.columns # Added by later steps in clean_dataframe
        assert "status" in cleaned.columns # Column where the error occurred
        assert cleaned["task_id"].notna().all()
        assert cleaned["status"].notna().all()


def _run_clean_with_standardised(df_standardised):
    """Run clean_dataframe with SchemaManager stubbed to return df_standardised."""
    with patch('Processing.core.data_cleaning.SchemaManager') as MockSchemaManager:
        mock_instance = MagicMock()
        MockSchemaManager.return_value = mock_instance
        mock_instance.required = []
        mock_instance.standardise_columns.return_value = df_standardised.copy()
        mock_instance.convert_data_types.side_effect = lambda df: df
        mock_instance.enforce_not_null.side_effect = lambda df: df
        return clean_dataframe(df_standardised.copy(), schema_type="tasks", project_name="UnitTest")


def test_real_percent_complete_passes_through_unchanged():
    """Real progress values must never be replaced, rebalanced or jittered.

    Uses a mostly-complete schedule (>80% at 1.0) and a mostly-zero schedule
    (>80% at 0.0): the shapes that the old cleaning code used to overwrite
    with random values.
    """
    mostly_complete = [1.0] * 9 + [0.4]
    mostly_zero = [0.0] * 9 + [0.25]
    for values in (mostly_complete, mostly_zero):
        df = pd.DataFrame({
            "task_code": [f"T{i}" for i in range(len(values))],
            "task_name": [f"Task {i}" for i in range(len(values))],
            "status": ["In Progress"] * len(values),
            "percent_complete": values,
        })
        for _ in range(2):  # deterministic across runs
            cleaned = _run_clean_with_standardised(df)
            assert cleaned["percent_complete"].tolist() == values
            # Source status is kept, not recomputed from progress.
            assert cleaned["status"].tolist() == ["In Progress"] * len(values)


def test_percent_complete_units_normalised_and_unknowns_left_missing():
    df = pd.DataFrame({
        "task_code": ["A", "B", "C", "D", "E"],
        "task_name": ["A", "B", "C", "D", "E"],
        "status": ["In Progress", "In Progress", "Complete", "Not Started", "In Progress"],
        "percent_complete": ["50%", "100", None, None, None],
    })
    cleaned = _run_clean_with_standardised(df)
    pc = cleaned["percent_complete"].tolist()
    assert pc[0] == 0.5
    assert pc[1] == 1.0
    assert pc[2] == 1.0  # derived only from an unambiguous "Complete" status
    assert pc[3] == 0.0  # derived only from an unambiguous "Not Started" status
    assert pd.isna(pc[4])  # in progress with no value: unknown, not invented
