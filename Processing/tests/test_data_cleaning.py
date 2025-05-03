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
        assert "task_id" in cleaned.columns # Added by later steps in clean_dataframe
        assert "status" in cleaned.columns # Column where the error occurred
        assert "severity_score" in cleaned.columns
        # Check that the status inference worked (requires numeric percent_complete)
        assert cleaned.loc[0, 'status'] == 'in_progress'
        assert cleaned.loc[1, 'status'] == 'complete'
