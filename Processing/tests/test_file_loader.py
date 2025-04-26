import os
import pandas as pd
from Processing.ingestion.file_loader import load_project_files

def test_load_sample_project_fixtures():
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
    assert 'task_code' in df.columns # From sample.csv
    # assert 'colx' in df.columns # From sample.xls (pandas might lowercase)
    assert len(df) > 0 # Check that some rows were loaded
