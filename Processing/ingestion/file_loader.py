import os
import pandas as pd
import logging

# Get the logger instance (assuming it's configured elsewhere, e.g., in data_pipeline or main_runner)
logger = logging.getLogger(__name__)

def load_project_files(project_path):
    all_files = []
    update_phase = 0

    # Log the start of the loading process for this project path
    logger.info(f"Starting file loading for project path: {project_path}")

    files_in_dir = sorted(os.listdir(project_path))
    logger.info(f"Found {len(files_in_dir)} items in directory. Processing files ending with .csv, .xls, .xlsx.")

    for file in files_in_dir:
        if not file.endswith((".csv", ".xls", ".xlsx")):
            logger.debug(f"Skipping non-data file: {file}")
            continue

        file_path = os.path.join(project_path, file)
        logger.info(f"Attempting to load file: {file_path}")

        try:
            if file.endswith(".xls"):
                # Use xlrd engine for older .xls format
                df = pd.read_excel(file_path, engine="xlrd")
            elif file.endswith(".xlsx"):
                # Use openpyxl engine for newer .xlsx format
                df = pd.read_excel(file_path, engine="openpyxl")
            else: # Handle .csv
                df = pd.read_csv(file_path)

            logger.info(f"Successfully loaded file: {file_path}")

            df["update_phase"] = f"update_{update_phase + 1}"
            df["file_name"] = file
            all_files.append(df)
            update_phase += 1

        except Exception as e:
            logger.error(f"Could not load {file}: {e}", exc_info=True)

    if not all_files:
        logger.error(f"No valid data files were successfully loaded from {project_path}")
        raise ValueError(f"No valid data files found in {project_path}")

    logger.info(f"Successfully loaded {len(all_files)} files. Concatenating into a single DataFrame.")
    combined_df = pd.concat(all_files, ignore_index=True)
    
    logger.debug("Stripping whitespace from combined DataFrame column names.")
    combined_df.columns = [str(col).strip() for col in combined_df.columns]

    logger.info(f"Finished loading and concatenating files for {project_path}. Total rows: {len(combined_df)}")
    return combined_df
