import os
import pandas as pd
from Processing.ingestion.file_loader import load_project_files

def test_load_sample_project_fixtures():
    """Tests loading of multiple file types (CSV, XLS) from a fixture directory.
    
    Relies on the presence of a 'fixtures/sample_project' directory relative
    to this test file, containing 'sample.csv' and 'sample.xls'.
    It checks if the function correctly loads and concatenates data from these
    files, adding the 'file_name' column and handling basic loading.
    
    Future Improvement: Consider using pytest fixtures for setup/teardown 
    of the test directory and files.
    """
    # Determine the absolute path to the test fixtures directory
    tests_dir = os.path.dirname(__file__) # Gets the directory of the current test file
    fixture_dir = os.path.abspath(os.path.join(tests_dir, "fixtures", "sample_project"))
    
    print(f"DEBUG: Loading test fixtures from: {fixture_dir}") # Add print for debugging
    assert os.path.isdir(fixture_dir), f"Fixture directory not found: {fixture_dir}"
    
    # Ensure dummy files exist (optional sanity check)
    assert os.path.exists(os.path.join(fixture_dir, "sample.csv")), "Test file sample.csv missing"
    assert os.path.exists(os.path.join(fixture_dir, "sample.xls")), "Test file sample.xls missing"

    # Load files from the fixture directory
    df = load_project_files(fixture_dir)
    
    # Assertions based on the expected content of the dummy files
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Loaded DataFrame should not be empty"
    # Check columns from both dummy files merged (adjust based on actual loader logic)
    # Assuming load_project_files adds 'filename' and 'update_phase'
    assert "file_name" in df.columns
    assert "update_phase" in df.columns 
    # Check for columns from the dummy files (case/content might vary based on pandas read_*)
    assert 'task_code' in df.columns # Specific column from sample.csv
    assert 'some_value' in df.columns # Specific column from sample.xls
    assert len(df) == 4 # Expecting 2 rows from CSV + 2 rows from XLS
    
    # Check if filenames were added correctly
    assert "sample.csv" in df["file_name"].unique()
    assert "sample.xls" in df["file_name"].unique()
    
    # Check a specific value from one of the files
    assert df.loc[df['file_name'] == 'sample.csv', 'task_code'].iloc[0] == 'TKA'
    assert df.loc[df['file_name'] == 'sample.xls', 'some_value'].iloc[0] == 10
    # assert 'colx' in df.columns # From sample.xls (pandas might lowercase)
    # assert len(df) > 0 # Check that some rows were loaded
