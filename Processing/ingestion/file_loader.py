import os
import pandas as pd
import logging

# Get the logger instance (assuming it's configured elsewhere, e.g., in main_runner)
logger = logging.getLogger(__name__)

def load_project_files(folder_path):
    """Loads project data from CSV, XLS, or XLSX files within a specified folder.

    Iterates through files in the given directory, attempting to load data from 
    supported file types (csv, xls, xlsx). It intelligently skips temporary Excel files
    (starting with '~') and combines data from all valid files found into a single 
    pandas DataFrame.

    Error handling is included for file reading and parsing issues. If a file cannot 
    be processed, a warning is logged, and the process continues with the next file.

    Args:
        folder_path (str): The absolute or relative path to the directory containing 
                           the project data files.

    Returns:
        pd.DataFrame: A single DataFrame containing the combined data from all 
                      successfully loaded files. Includes a 'file_name' column 
                      indicating the source file for each row. Returns an empty 
                      DataFrame if the folder doesn't exist, contains no supported 
                      files, or if all files fail to load.
    """
    logger = logging.getLogger(__name__) 
    all_data = []
    project_name = os.path.basename(folder_path) # Extract project name for logging
    logger.info(f"[{project_name}] Starting file loading from folder: {folder_path}")

    # Check if the provided path is a valid directory.
    if not os.path.isdir(folder_path):
        logger.error(f"[{project_name}] Invalid folder path provided: {folder_path}. Cannot load files.")
        return pd.DataFrame() # Return empty DataFrame if path is invalid.

    # List all files in the directory.
    try:
        files = os.listdir(folder_path)
        logger.debug(f"[{project_name}] Found {len(files)} items in folder: {files}")
    except OSError as e:
        logger.error(f"[{project_name}] Cannot access or list files in folder: {folder_path}. Error: {e}")
        return pd.DataFrame()
        
    loaded_file_count = 0
    skipped_file_count = 0
    # Iterate through each file found in the directory.
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        # Skip temporary Excel files (often created when a file is open).
        if filename.startswith('~'):
            logger.debug(f"[{project_name}] Skipping temporary file: {filename}")
            skipped_file_count += 1
            continue
            
        # Process based on file extension.
        if filename.endswith('.csv'):
            logger.debug(f"[{project_name}] Attempting to load CSV file: {filename}")
            try:
                # Attempt to read with common encodings; add more if needed.
                # This step can fail if the CSV is malformed (e.g., incorrect delimiters, quoting issues).
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    logger.warning(f"[{project_name}] UTF-8 decoding failed for {filename}. Trying latin1.")
                    df = pd.read_csv(file_path, encoding='latin1')
                    
                df['file_name'] = filename # Add source file name
                all_data.append(df)
                loaded_file_count += 1
                logger.debug(f"[{project_name}] Successfully loaded CSV: {filename} (Shape: {df.shape})")
            except Exception as e:
                # Log errors encountered during CSV reading/parsing.
                logger.warning(f"[{project_name}] Failed to load CSV file '{filename}': {e}")
                skipped_file_count += 1
                
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            logger.debug(f"[{project_name}] Attempting to load Excel file: {filename}")
            try:
                # Load the first sheet by default using appropriate engine.
                # Requires openpyxl for .xlsx and xlrd for .xls.
                # This can fail if the file is corrupted or not a valid Excel format.
                engine = 'openpyxl' if filename.endswith('.xlsx') else 'xlrd'
                df = pd.read_excel(file_path, engine=engine, sheet_name=0) # Read first sheet
                df['file_name'] = filename # Add source file name
                all_data.append(df)
                loaded_file_count += 1
                logger.debug(f"[{project_name}] Successfully loaded Excel: {filename} (Shape: {df.shape})")
            except Exception as e:
                # Log errors encountered during Excel reading/parsing.
                logger.warning(f"[{project_name}] Failed to load Excel file '{filename}': {e}")
                skipped_file_count += 1
        else:
            # Log files that are skipped due to unsupported extensions.
            logger.debug(f"[{project_name}] Skipping unsupported file type: {filename}")
            skipped_file_count += 1

    # Check if any data was successfully loaded.
    if not all_data:
        logger.warning(f"[{project_name}] No valid data files found or loaded from folder: {folder_path}")
        return pd.DataFrame() # Return empty DataFrame if nothing was loaded.

    # Concatenate all loaded DataFrames into one.
    try:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"[{project_name}] Successfully loaded and combined data from {loaded_file_count} files. Total rows: {len(combined_df)}. Skipped {skipped_file_count} files.")
        return combined_df
    except Exception as e:
        # Log errors during the concatenation step (e.g., schema mismatch if not handled earlier).
        logger.error(f"[{project_name}] Error concatenating data from loaded files: {e}")
        # Return empty DataFrame if concatenation fails, as the combined data is unusable.
        return pd.DataFrame()
