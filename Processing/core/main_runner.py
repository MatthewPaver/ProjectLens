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

# --- Configuration ---
# Base directory setup
BASE_DIR = project_root # Use calculated project root

# Use resolve_path correctly now it's imported
INPUT_DIR = resolve_path("Data/input") # Removed base_dir as resolve_path uses ROOT_DIR internally
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

# --- Logging Setup ---
# Configure logging
# --- REMOVED: Original log_dir and file_path variables, handled by basicConfig ---
# log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
# os.makedirs(log_dir, exist_ok=True)
# log_file_path = os.path.join(log_dir, 'pipeline.log')

# --- CHANGE: Simplified logging configuration ---
# Use basicConfig for direct file logging, overwrite each time (filemode='w')
# Ensure the path uses LOG_DIR which is resolved using resolve_path
log_file_path_resolved = os.path.join(LOG_DIR, 'pipeline.log')

# --->>> ADDED: Print the exact path before trying to log to it <<<---
print(f"❗ Attempting to configure logging to: {log_file_path_resolved}") 
# --->>> END ADDED <<<---

try:
    logging.basicConfig(
        level=logging.DEBUG, # Log everything from DEBUG upwards
        format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
        filename=log_file_path_resolved, # Use the resolved path
        filemode='w' # Overwrite the log file each time
    )
    print(f"✅ Logging configured to write to: {log_file_path_resolved}")
except Exception as e:
    # If basicConfig fails, print error and exit - logging is critical for debugging
    print(f"❌ CRITICAL: Failed to configure logging to file '{log_file_path_resolved}'. Error: {e}")
    sys.exit(f"Logging setup failed: {e}")
# --- END CHANGE ---

# --- REMOVED: Explicit handlers setup (handled by basicConfig now) ---
# Setup basic configuration for logging to file and console
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
#                     handlers=[
#                         logging.FileHandler(log_file_path, mode='w'),
#                         logging.StreamHandler() # Restore console logging
#                     ])

# Get the root logger
# --- CHANGE: No longer strictly necessary to get logger after basicConfig, but harmless ---
logger = logging.getLogger()
# --- END CHANGE ---

# --- Main Function ---
def main():
    """Main function to orchestrate the data processing pipeline."""
    logging.info("--- Main runner started ---")
    
    # Initialize SchemaManager
    try:
        schema_manager = SchemaManager(schema_type="tasks") 
        logging.info(f"SchemaManager initialized for schema type 'tasks'.")
    except FileNotFoundError as e:
        logging.error(f"CRITICAL: Schema file not found during SchemaManager init: {e}", exc_info=True)
        print(f"❌ CRITICAL: Schema file not found: {e}. Cannot proceed.")
        return
    except Exception as e:
        logging.error(f"Failed to initialize SchemaManager: {e}", exc_info=True)
        print(f"❌ CRITICAL: Failed to initialize SchemaManager. Cannot proceed.")
        return # Stop execution if SchemaManager fails

    # Get project folders
    try:
        logging.info(f"Resolved input directory: {INPUT_DIR}")
        # List only directories in the input folder
        project_folders = [d for d in os.listdir(INPUT_DIR) 
                           if os.path.isdir(os.path.join(INPUT_DIR, d)) and not d.startswith('.')]
        logging.info(f"Found projects in input directory: {project_folders}")
        if not project_folders:
             print(f"ℹ️ No project folders found in the input directory '{INPUT_DIR}' to process.")
             print(f"   Please ensure data exists in subfolders within '{INPUT_DIR}'.")
             print(f"   You can generate synthetic data using: python Processing/tests/generate_synthetic_data.py")
             logging.warning(f"No project folders found in {INPUT_DIR}. Stopping.")
             return
        print(f"🔍 Found {len(project_folders)} project(s) to process.")
        logging.info(f"Found {len(project_folders)} project(s) to process: {project_folders}")
    except FileNotFoundError:
        logging.error(f"Input directory not found: {INPUT_DIR}")
        print(f"❌ ERROR: Input directory not found at {INPUT_DIR}")
        return
    except Exception as e:
        logging.error(f"Error listing project folders: {e}", exc_info=True)
        print(f"❌ ERROR: Could not list project folders in {INPUT_DIR}")
        return

    # Process each project folder
    for project_folder_name in project_folders:
        logging.info(f"--- Starting loop for project: {project_folder_name} ---")
        print(f"📦 Processing: {project_folder_name}")
        project_folder_path = os.path.join(INPUT_DIR, project_folder_name) # Get full path
        
        try:
            # --- CORRECTED CALL --- 
            # Pass project_folder_path and schema_manager
            # Remove debug argument
            output_dir = process_project(project_folder_path=project_folder_path, schema_manager=schema_manager)
            
            if output_dir:
                logging.info(f"Successfully processed {project_folder_name}.")
                # Archiving is now handled within process_project
            else:
                 logging.error(f"Processing failed for {project_folder_name}. See previous logs.")
                 print(f"❌ Processing failed for {project_folder_name}. Check logs.")
                 # Archiving to failed is handled within process_project

        except Exception as e:
            # Log unhandled exceptions from process_project call
            logging.error(f"Unhandled exception during process_project call for {project_folder_name}: {e}", exc_info=True)
            print(f"❌ Unexpected error processing {project_folder_name}. Check logs.")
            # Attempt to archive to failed - Note: archive_project is called internally by process_project on error too,
            # but we might try again here if the exception happened before process_project archived it.
            # This outer block no longer needs to call archive_project directly.
            # try:
            #      archive_project(project_folder_path, success=False) 
            #      logging.warning(f"Moved {project_folder_name} to failed archive due to unhandled processing error.")
            # except Exception as archive_e:
            #      logging.error(f"CRITICAL: Failed to archive {project_folder_name} after unhandled error: {archive_e}")
                 
    logging.info("--- Main runner finished processing all projects ---")
    print("\n✅ Processing complete. Check logs for details and potential errors.")

    # --->>> ADD SLEEP <<<---
    print("--- Sleeping for 5 seconds before exiting ---")
    logger.info("--- Sleeping for 5 seconds before exiting ---")
    time.sleep(5)
    print("--- Exiting ---")
    logger.info("--- Exiting ---")
    # --->>> END SLEEP <<<---

if __name__ == "__main__":
    main()
