import unittest
import pandas as pd
import logging
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the function to test
from Processing.core.data_pipeline import process_project
from Processing.core.schema_manager import SchemaManager

# Mock data for input files
SAMPLE_CSV_CONTENT = """Task Code,Task Name,Baseline Finish,Actual Finish
TKA,Task A,2023-01-10,2023-01-12
TKB,Task B,2023-01-15,2023-01-14"""

class TestPipelineIntegration(unittest.TestCase):
    """Test the pipeline integration with mocks for heavy analysis."""

    def setUp(self):
        """Set up temporary directories and files for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, "Data", "input", "TestProject")
        self.output_base_dir = os.path.join(self.test_dir, "Data", "output")
        self.archive_dir = os.path.join(self.test_dir, "Data", "archive")
        self.schema_dir = os.path.join(self.test_dir, "Data", "schemas") # Need schema dir

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_base_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        os.makedirs(os.path.join(self.archive_dir, 'success'), exist_ok=True)
        os.makedirs(os.path.join(self.archive_dir, 'failed'), exist_ok=True)
        os.makedirs(self.schema_dir, exist_ok=True) # Create schema dir

        # Create dummy input file
        with open(os.path.join(self.input_dir, "update_1.csv"), "w") as f:
            f.write(SAMPLE_CSV_CONTENT)

        # Create a dummy schema file (replace with actual minimal schema if needed)
        # For this test, SchemaManager might be mocked anyway, but good practice
        dummy_schema_path = os.path.join(self.schema_dir, "tasks_schema.json")
        with open(dummy_schema_path, "w") as f:
            f.write('{"schema_type": "tasks", "columns": {}}') # Minimal valid JSON

        # Mock SchemaManager to avoid dependency on exact schema file content for this test
        self.schema_manager_patch = patch('Processing.core.data_pipeline.SchemaManager')
        self.MockSchemaManager = self.schema_manager_patch.start()
        self.mock_schema_instance = MagicMock()
        # Configure mock methods needed by data_cleaning called within process_project
        self.mock_schema_instance.standardise_columns.side_effect = lambda df: df.rename(columns={"Task Code": "task_id", "Task Name": "task_name", "Baseline Finish": "baseline_end_date", "Actual Finish": "actual_finish"})
        self.mock_schema_instance.convert_data_types.side_effect = lambda df: df # Passthrough
        self.mock_schema_instance.enforce_not_null.side_effect = lambda df: df # Passthrough
        self.MockSchemaManager.return_value = self.mock_schema_instance


    def tearDown(self):
        """Clean up temporary directories and stop patches."""
        shutil.rmtree(self.test_dir)
        self.schema_manager_patch.stop()


    # Patch the analysis functions and output writer within the data_pipeline module
    @patch('Processing.core.data_pipeline.run_slippage_analysis')
    @patch('Processing.core.data_pipeline.run_forecasting')
    @patch('Processing.core.data_pipeline.detect_changepoints')
    @patch('Processing.core.data_pipeline.analyse_milestones')
    @patch('Processing.core.data_pipeline.generate_recommendations')
    @patch('Processing.core.data_pipeline.write_outputs')
    @patch('Processing.core.data_pipeline.archive_project') # Also mock archiving
    def test_pipeline_flow_and_basic_output(self,
                                           mock_archive,
                                           mock_write_outputs,
                                           mock_generate_recommendations,
                                           mock_analyse_milestones,
                                           mock_detect_changepoints,
                                           mock_run_forecasting,
                                           mock_run_slippage_analysis):
        """Tests the main pipeline flow, mocking analysis steps."""

        # Configure mock return values for analysis functions
        # Return simple DataFrames or structures expected by write_outputs
        mock_run_slippage_analysis.return_value = pd.DataFrame({'task_id': ['TKA', 'TKB'], 'slip_days': [2, -1]})
        mock_run_forecasting.return_value = pd.DataFrame({'task_id': ['TKA'], 'predicted_end_date': [pd.Timestamp('2023-01-13')]})
        mock_detect_changepoints.return_value = pd.DataFrame({'task_id': ['TKA'], 'update_phase': ['update_1']})
        mock_analyse_milestones.return_value = pd.DataFrame({'task_id': ['TKA'], 'is_milestone': [True]}) # Assuming milestone analysis returns a DF
        mock_generate_recommendations.return_value = [] # Return empty list or list of dicts

        # --- Execute the pipeline for the test project ---
        # Use the mocked SchemaManager instance
        output_path = process_project(
            project_folder_path=self.input_dir,
            schema_manager=self.mock_schema_instance, # Pass the instance
            base_output_dir=self.output_base_dir,
            archive_dir=self.archive_dir
        )

        # --- Assertions ---
        # 1. Check that the main analysis functions were called
        mock_run_slippage_analysis.assert_called_once()
        mock_run_forecasting.assert_called_once()
        mock_detect_changepoints.assert_called_once()
        mock_analyse_milestones.assert_called_once()
        mock_generate_recommendations.assert_called_once()

        # 2. Check that write_outputs was called
        mock_write_outputs.assert_called_once()

        # 3. Check the arguments passed to write_outputs (optional but good)
        args, kwargs = mock_write_outputs.call_args
        self.assertEqual(kwargs['project_name'], "TestProject")
        self.assertTrue(isinstance(kwargs['cleaned_df'], pd.DataFrame))
        self.assertFalse(kwargs['cleaned_df'].empty)
        self.assertIn('task_id', kwargs['cleaned_df'].columns) # Check cleaning happened
        self.assertEqual(kwargs['analysis_results']['slippages'], mock_run_slippage_analysis.return_value)
        self.assertEqual(kwargs['analysis_results']['forecasts'], mock_run_forecasting.return_value)
        # ... check other analysis results if needed

        # 4. Check that archiving was called (assuming success)
        mock_archive.assert_called_once()
        self.assertEqual(mock_archive.call_args[0][0], self.input_dir) # Check correct source dir
        self.assertTrue(mock_archive.call_args[0][1].startswith(os.path.join(self.archive_dir, 'success'))) # Check correct target base

        # 5. Check that an output path was returned and exists (basic check)
        # Note: write_outputs itself is mocked, so files aren't *actually* written
        # unless we remove the mock_write_outputs patch and test file creation.
        # For this test focusing on flow, checking the call is sufficient.
        self.assertIsNotNone(output_path)
        self.assertEqual(output_path, os.path.join(self.output_base_dir, "TestProject"))


if __name__ == '__main__':
    unittest.main()
