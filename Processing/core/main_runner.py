#!/usr/bin/env python3
"""Main Runner Script for the ProjectLens Processing Pipeline.

This script serves as the primary entry point for the data processing logic when
run *within* the properly configured virtual environment (typically set up and 
executed by `Processing/main.py`).

Its main responsibilities include:
1.  Setting up project-wide path configurations (Input, Output, Archive, Logs) 
    relative to the project root.
2.  Configuring file-based logging to `Processing/logs/pipeline.log`.
3.  Initialising the `SchemaManager` with the desired schema (e.g., "tasks").
4.  Scanning the configured `Data/input` directory for project subfolders.
5.  Iterating through each found project folder and calling the main 
    `data_pipeline.process_project` function to execute the full 
    load-clean-analyse-output-archive sequence for that project.
6.  Logging the overall success and duration of the pipeline run.

This script is not intended to be run directly by the user but rather invoked
by the environment setup script (`main.py`).
"""
import logging
import os
import sys
from datetime import datetime
import time

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core components (use relative paths for robustness if possible)
from Processing.core.config_loader import resolve_path
from Processing.core.data_pipeline import process_project 
from Processing.core.schema_manager import SchemaManager

# Configuration
# Base directory setup
BASE_DIR = project_root # Use calculated project root

# Use resolve_path correctly now it's imported
INPUT_DIR = resolve_path("Data/input") 
ARCHIVE_DIR = resolve_path("Data/archive")
SCHEMA_DIR = resolve_path("Data/schemas")
LOG_DIR = resolve_path("Processing/logs")

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(SCHEMA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_DIR, 'success'), exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_DIR, 'failed'), exist_ok=True)

# Logging Setup
log_file_path_resolved = os.path.join(LOG_DIR, 'pipeline.log')
print(f"Attempting to configure logging to: {log_file_path_resolved}") 

try:
    # Explicit File Handler Setup
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')
    log_handler = logging.FileHandler(log_file_path_resolved, mode='w', encoding='utf-8')
    log_handler.setFormatter(log_formatter)

    # Get the root logger and add the handler
    root_logger = logging.getLogger() 
    root_logger.setLevel(logging.DEBUG) # Set level on the root logger
    # Remove existing handlers (if any) before adding the new one to avoid duplication
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(log_handler)
    print(f"Explicit FileHandler configured for: {log_file_path_resolved}")
except Exception as e:
    # If basicConfig fails, print error and exit - logging is critical for debugging
    print(f"CRITICAL: Failed to configure logging to file '{log_file_path_resolved}'. Error: {e}")
    sys.exit(f"Logging setup failed: {e}")

# Main Function
def main():
    """Main function to orchestrate the data processing pipeline."""
    logging.info("--- Main runner started ---")
    # Record start time
    total_start_time = time.time()
    # Track overall success across all projects
    all_projects_successful = True
    
    # Initialise SchemaManager
    try:
        schema_manager = SchemaManager(schema_type="tasks") 
        logging.info(f"SchemaManager initialised for schema type 'tasks'.")
    except FileNotFoundError as e:
        logging.error(f"CRITICAL: Schema file not found during SchemaManager init: {e}", exc_info=True)
        print(f"CRITICAL: Schema file not found: {e}. Cannot proceed.")
        return
    except Exception as e:
        logging.error(f"Failed to initialise SchemaManager: {e}", exc_info=True)
        print(f"CRITICAL: Failed to initialise SchemaManager. Cannot proceed.")
        return # Stop execution if SchemaManager fails

    # Get project folders
    try:
        logging.info(f"Resolved input directory: {INPUT_DIR}")
        # List only directories in the input folder
        project_folders = [d for d in os.listdir(INPUT_DIR) 
                           if os.path.isdir(os.path.join(INPUT_DIR, d)) and not d.startswith('.')]
        logging.info(f"Found projects in input directory: {project_folders}")
        if not project_folders:
             print(f"No project folders found in the input directory '{INPUT_DIR}' to process.")
             print(f"Please ensure data exists in subfolders within '{INPUT_DIR}'.")
             print(f"You can generate synthetic data using: python Processing/tests/generate_synthetic_data.py")
             logging.warning(f"No project folders found in {INPUT_DIR}. Stopping.")
             return
        print(f"Found {len(project_folders)} project(s) to process.")
        logging.info(f"Found {len(project_folders)} project(s) to process: {project_folders}")
    except FileNotFoundError:
        logging.error(f"Input directory not found: {INPUT_DIR}")
        print(f"ERROR: Input directory not found at {INPUT_DIR}")
        return
    except Exception as e:
        logging.error(f"Error listing project folders: {e}", exc_info=True)
        print(f"ERROR: Could not list project folders in {INPUT_DIR}")
        return

    # Process each project folder
    for project_folder_name in project_folders:
        logging.info(f"--- Starting loop for project: {project_folder_name} ---")
        print(f"Processing: {project_folder_name}")
        project_folder_path = os.path.join(INPUT_DIR, project_folder_name) # Get full path
        
        try:
            # Pass project_folder_path and schema_manager
            output_dir = process_project(project_folder_path=project_folder_path, schema_manager=schema_manager)
            
            if output_dir:
                logging.info(f"Successfully processed {project_folder_name}.")
                # Archiving is now handled within process_project
            else:
                 logging.error(f"Processing failed for {project_folder_name}. See previous logs.")
                 print(f"Processing failed for {project_folder_name}. Check logs.")
                 # Mark overall success as False if any project fails
                 all_projects_successful = False
                 # Archiving to failed is handled within process_project

        except Exception as e:
            # Log unhandled exceptions from process_project call
            logging.error(f"Unhandled exception during process_project call for {project_folder_name}: {e}", exc_info=True)
            print(f"Unexpected error processing {project_folder_name}. Check logs.")
            # Attempt to archive to failed - Note: archive_project is called internally by process_project on error too,
            # but we might try again here if the exception happened before process_project archived it.
            # This outer block no longer needs to call archive_project directly.
            all_projects_successful = False # Mark as failed on unhandled exception
            
    logging.info("--- Main runner finished processing all projects ---")
    print("\nProcessing complete. Check logs for details and potential errors.")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    root_logger.info(f"===== Pipeline Runner Finished. Total time: {total_duration:.2f} seconds. Overall success: {all_projects_successful} =====")

if __name__ == "__main__":
    main()
    # Ensure all logs are flushed and handlers closed properly
    logging.shutdown()
    # Exit with appropriate code
    # The exit code should be handled by the caller (Processing/main.py)
