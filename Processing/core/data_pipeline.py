import os
import pandas as pd
import logging
import traceback
import shutil # Import shutil for archive_project

from Processing.core.config_loader import resolve_path
from Processing.core.schema_manager import SchemaManager
from Processing.core.data_cleaning import clean_dataframe

from Processing.ingestion.file_loader import load_project_files

# --- Restore analysis imports ---
from Processing.analysis.slippage_analysis import run_slippage_analysis
from Processing.analysis.forecast_engine import run_forecasting
from Processing.analysis.changepoint_detector import detect_change_points
from Processing.analysis.milestone_analysis import analyse_milestones
from Processing.analysis.recommendation_engine import generate_recommendations
# --- End restore ---

from Processing.output.output_writer import write_outputs

# --- Logging Setup --- 
# --->>> REMOVE Logging Configuration from data_pipeline.py <<<---
# # Configure logging (basic setup, customize as needed)
# log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs')) # Use absolute path based on current file
# os.makedirs(log_dir, exist_ok=True)
# log_file = os.path.join(log_dir, 'pipeline.log')
# 
# # Configure basic logging (adjust level and format as needed)
# # Ensure file handler uses 'w' mode to overwrite log each run
# logging.basicConfig(level=logging.DEBUG, 
#                     format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
#                     handlers=[
#                         logging.FileHandler(log_file, mode='w'), 
#                         logging.StreamHandler() # Keep console output
#                     ])
# logger = logging.getLogger(__name__) # Use module-specific logger if preferred
# logger.info(f"--- Logging configured in data_pipeline.py (Level: {logging.getLevelName(logger.getEffectiveLevel())}) ---")
# --- End Logging Setup --- 

# --->>> Get Logger Instance <<<---
# Modules should get the logger instance configured by the main entry point (main_runner.py)
logger = logging.getLogger(__name__) 
# --->>> END Get Logger Instance <<<---

# --- Define Directory Constants --- 
# Use resolve_path to correctly locate the Data directories in the project root
INPUT_DIR   = resolve_path("Data/input")    # Input folder in project Data
OUTPUT_DIR  = resolve_path("Data/output")   # Output folder in project Data
ARCHIVE_DIR = resolve_path("Data/archive")  # Archive folder in project Data
SCHEMA_DIR  = resolve_path("Data/schemas")  # Schemas folder in project Data

# Ensure Output and Archive directories exist (essential for this script)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_DIR, 'success'), exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_DIR, 'failed'), exist_ok=True)
# --- End Directory Constants --- 

# --- Archive Function (defined locally as it's used only here) ---
def archive_project(project_folder_path: str, success: bool):
    """Moves the processed project folder to the success or failed archive directory."""
    project_folder = os.path.basename(project_folder_path)
    target_dir_suffix = 'success' if success else 'failed'
    target_archive_dir = os.path.join(ARCHIVE_DIR, target_dir_suffix)
    # The destination *directory* path
    destination_path = os.path.join(target_archive_dir, project_folder)
    
    logger.info(f"Attempting to archive '{project_folder}' to '{target_archive_dir}' (Success: {success})")
    
    # Ensure the target archive directory exists (redundant if created above, but safe)
    os.makedirs(target_archive_dir, exist_ok=True)
    
    # Check if the source path still exists before moving
    if not os.path.exists(project_folder_path):
        logger.warning(f"Source path {project_folder_path} does not exist. Cannot archive.")
        return
        
    # Ensure the destination doesn't already exist to avoid errors during move
    if os.path.exists(destination_path):
        logger.warning(f"Destination archive path {destination_path} already exists. Removing before archiving.")
        try:
            if os.path.isdir(destination_path):
                 shutil.rmtree(destination_path)
            else:
                 os.remove(destination_path)
            logger.debug(f"Removed existing item at {destination_path}.")
        except Exception as e_rem:
            logger.error(f"Failed to remove existing item at {destination_path}: {e_rem}")
            # Decide if we should proceed or stop
            return # Stop if we can't clear the destination
             
    try:
        shutil.move(project_folder_path, target_archive_dir) # Move into the target success/failed dir
        logger.info(f"Successfully moved project {project_folder} to {target_archive_dir}")
    except Exception as e_mov:
        logger.error(f"Failed to move project {project_folder} to archive directory {target_archive_dir}: {e_mov}", exc_info=True)
# --- End Archive Function ---

# --- Save Function Placeholder --- 
def save_analysis_results(analysis_results: dict, output_dir: str, project_name: str):
    """Saves each analysis DataFrame to a CSV file."""
    logger.info(f"[{project_name}] Saving analysis results to {output_dir}...")
    for name, df in analysis_results.items():
        # Standardize output filenames
        if name == 'slippage': filename = 'slippage_summary.csv'
        elif name == 'forecast': filename = 'forecast_results.csv'
        elif name == 'changepoints': filename = 'changepoints.csv'
        # Add other analysis types here
        # elif name == 'milestones': filename = 'milestone_analysis.csv' 
        # elif name == 'recommendations': filename = 'recommendations.csv'
        else: 
            filename = f'{name}_output.csv' # Generic fallback
            
        file_path = os.path.join(output_dir, filename)
        
        if df is not None and not df.empty:
            try:
                df.to_csv(file_path, index=False)
                logger.info(f"[{project_name}] Saved {name} analysis ({len(df)} rows) to {filename}")
            except Exception as e:
                logger.error(f"[{project_name}] Failed to save {name} analysis to {file_path}: {e}", exc_info=True)
        elif df is None:
             logger.warning(f"[{project_name}] No data to save for {name} analysis (result was None).")
             # Optionally create empty file with headers if df structure known (difficult if None)
        else: # DataFrame is empty
            logger.warning(f"[{project_name}] No data to save for {name} analysis (result DataFrame is empty).")
            # Optionally save empty file with headers
            try:
                 # Attempt to get headers if df is empty but has columns defined
                 pd.DataFrame(columns=df.columns).to_csv(file_path, index=False)
                 logger.info(f"[{project_name}] Saved empty file with headers for {name} to {filename}")
            except Exception as e_empty:
                 logger.error(f"[{project_name}] Failed to save empty file for {name} to {file_path}: {e_empty}")
# --- End Save Function --- 

def process_project(project_folder_path: str, schema_manager: SchemaManager):
    """Loads, cleans, analyses, and archives data for a single project folder."""
    project_folder = os.path.basename(project_folder_path)
    project_name = project_folder
    logger.info(f"Processing project: {project_folder} (Name: {project_name})") # Optionally add name to log
    
    output_project_dir = os.path.join(OUTPUT_DIR, project_folder)
    os.makedirs(output_project_dir, exist_ok=True)

    # Initialize empty dataframes
    raw_df = pd.DataFrame()
    cleaned_df = pd.DataFrame()
    slippages = pd.DataFrame()
    forecasts = pd.DataFrame()
    changepoints = pd.DataFrame()
    milestones = pd.DataFrame()
    recommendations = pd.DataFrame()
    failed_tasks_for_recommendations = [] # Initialize list for failed task IDs
    
    # --- CHANGE: Restore success variable --- 
    success = True

    # --->>> ADD INPUT EXISTENCE CHECK 1 <<< ---
    logger.debug(f"CHECK 1: At start of process_project for '{project_name}'")
    logger.debug(f"  Input Dir '{INPUT_DIR}' exists: {os.path.exists(INPUT_DIR)}")
    logger.debug(f"  Project Path '{project_folder_path}' exists: {os.path.exists(project_folder_path)}")
    # --->>> END CHECK 1 <<< ---

    try:
        logger.info(f"Started processing: {project_folder}")

        # STEP 1: LOAD
        try:
            raw_df = load_project_files(project_folder_path)
            if raw_df.empty:
                 logger.warning(f"No data loaded from {project_folder_path}. Skipping further processing for {project_folder}.")
                 # Archive handled in finally block
                 return None # Indicate failure
            # Log message regardless of debug flag, level controlled by logger config
            logger.info(f"[{project_name}] Loaded {len(raw_df)} rows from {project_folder}")
            logger.debug(f"[{project_name}] Raw DataFrame shape: {raw_df.shape}") # Log raw shape
        except Exception as e:
            logger.error(f"[{project_name}] File loading failed for {project_folder}: {e}")
            # Log traceback regardless of debug flag, level controlled by logger config
            logger.error(traceback.format_exc())
            success = False
            # Archive handled in finally block
            return None # Indicate failure

        # STEP 2: CLEAN
        try:
            # Ensure clean_data uses schema_manager correctly if needed
            cleaned_df = clean_dataframe(
                raw_df,
                schema_type="tasks",
                project_name=project_folder,
                update_phase="auto"
            )
            if cleaned_df.empty:
                 logger.warning(f"Data cleaning resulted in an empty DataFrame for {project_folder}. Schema issues?")
                 success = False # Treat empty cleaned DF as failure for archiving
                 # Continue processing to allow writing empty files if needed, but mark as failed
            # Log message regardless of debug flag
            logger.info(f"[{project_name}] Cleaned data: {len(cleaned_df)} rows for {project_folder}")
            logger.debug(f"[{project_name}] Cleaned DataFrame shape: {cleaned_df.shape}") # Log cleaned shape
        except Exception as e:
            logger.error(f"[{project_name}] Data cleaning failed for {project_folder}: {e}", exc_info=True)
            success = False
            cleaned_df = pd.DataFrame() # Ensure cleaned_df is empty on critical cleaning error

        # --- STEP 3: ANALYSE - Re-enabled --- 
        if not cleaned_df.empty: 
            logger.info(f"[{project_name}] Starting analysis steps for {project_folder}...")
            # --->>> ADD INPUT EXISTENCE CHECK 2 <<< ---
            logger.debug(f"CHECK 2: Before analysis for '{project_name}'")
            logger.debug(f"  Input Dir '{INPUT_DIR}' exists: {os.path.exists(INPUT_DIR)}")
            logger.debug(f"  Project Path '{project_folder_path}' exists: {os.path.exists(project_folder_path)}")
            # --->>> END CHECK 2 <<< ---
            analysis_results = {}
            try:
                # Pass project_name for context in logging/output
                slippages = run_slippage_analysis(cleaned_df.copy(), project_name) # Assign to variable
                logger.info(f"[{project_name}] Slippage analysis complete. Found {len(slippages) if slippages is not None else 0} records.")
                logger.info(f"[{project_name}] Slippage analysis completed. Result shape: {slippages.shape}")
            except Exception as e:
                logger.error(f"[{project_name}] Error during slippage analysis for {project_folder}: {e}", exc_info=True)
                slippages = pd.DataFrame() # Assign empty DF on error

            try:
                # Pass project_name for context
                forecasts, failed_tasks_for_recommendations = run_forecasting(cleaned_df.copy(), project_name)
                logger.info(f"[{project_name}] Forecasting complete. Generated {len(forecasts) if forecasts is not None else 0} forecasts. {len(failed_tasks_for_recommendations)} tasks failed due to insufficient data.")
                logger.info(f"[{project_name}] Forecasting completed. Result shape: {forecasts.shape}. Failed tasks for recs: {len(failed_tasks_for_recommendations)}")
            except Exception as e:
                logger.error(f"[{project_name}] Error during forecasting for {project_folder}: {e}", exc_info=True)
                forecasts = pd.DataFrame() # Assign empty DF on error
                failed_tasks_for_recommendations = [] # Ensure list is empty on major forecast error
                
            try:
                # Pass project_name for context
                changepoints = detect_change_points(cleaned_df.copy(), project_name) # Assign to variable
                logger.info(f"[{project_name}] Change point detection complete. Found {len(changepoints) if changepoints is not None else 0} points.")
                logger.info(f"[{project_name}] Changepoint detection completed. Result shape: {changepoints.shape}")
            except Exception as e:
                logger.error(f"[{project_name}] Error during change point detection for {project_folder}: {e}", exc_info=True)
                changepoints = pd.DataFrame() # Assign empty DF on error

            try:
                # Pass project_name for context
                milestones = analyse_milestones(cleaned_df.copy()) # REMOVED project_name arg
                logger.info(f"[{project_name}] Milestone analysis complete. Analysed {len(milestones) if milestones is not None else 0} milestones.")
                logger.info(f"[{project_name}] Milestone analysis completed. Result shape: {milestones.shape}")
            except Exception as e:
                logger.error(f"[{project_name}] Error during milestone analysis for {project_folder}: {e}", exc_info=True)
                milestones = pd.DataFrame() # Assign empty DF on error
        else:
            logger.warning(f"[{project_name}] Skipping analysis for {project_folder} due to empty cleaned DataFrame.")
            # Ensure analysis dataframes remain empty
            slippages, forecasts, changepoints, milestones = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # --- STEP 4: RECOMMENDATIONS - Re-enabled --- 
        logger.info(f"[{project_name}] Starting recommendation generation for {project_folder}...")
        try:
            # Pass potentially empty analysis dataframes
            recommendations_dict = generate_recommendations(
                df=cleaned_df, 
                slippages=slippages, 
                forecasts=forecasts, 
                changepoints=changepoints, 
                milestones=milestones,
                failed_forecast_tasks=failed_tasks_for_recommendations # Pass the new list
            )
            logger.info(f"[{project_name}] Recommendation generation complete for {project_folder}.")
        except Exception as e:
            logger.error(f"[{project_name}] Error generating recommendations for {project_folder}: {e}", exc_info=True)
            # recommendations remains an empty dict/structure if generation fails
            recommendations_dict = {} # Ensure it's an empty dict on error

        # --->>> ADD INPUT EXISTENCE CHECK 3 <<< ---
        logger.debug(f"CHECK 3: Before output write for '{project_name}'")
        logger.debug(f"  Input Dir '{INPUT_DIR}' exists: {os.path.exists(INPUT_DIR)}")
        logger.debug(f"  Project Path '{project_folder_path}' exists: {os.path.exists(project_folder_path)}")
        # --->>> END CHECK 3 <<< ---

        # --- CHANGE: Restore STEP 5 (Output writing) --- 
        # STEP 5: OUTPUT - Write to project-specific path
        try:
            # Ensure output directory exists before attempting to write or archive
            if output_project_dir:
                os.makedirs(output_project_dir, exist_ok=True)

                # --->>> Add Diagnostic Prints Here <<<---
                print(f"\n--- DEBUG [{project_name}]: Checking DataFrames before write_outputs ---")
                print("Slippages DataFrame:")
                if slippages is not None:
                    print(f"  Shape: {slippages.shape}")
                    print(slippages.head())
                else:
                    print("  DataFrame is None")

                print("Forecasts DataFrame:")
                if forecasts is not None:
                    print(f"  Shape: {forecasts.shape}")
                    print(forecasts.head())
                else:
                    print("  DataFrame is None")
                print("--- END DEBUG ---\n")
                # --->>> End Diagnostic Prints <<<---

                # Call the centralized output writer
                try:
                    logger.info(f"[{project_name}] Attempting to write outputs using output_writer.write_outputs for {project_name}")
                    write_outputs(
                        output_path=output_project_dir,
                        cleaned_df=cleaned_df, # Pass the final cleaned df
                        slippages=slippages,
                        forecasts=forecasts,
                        changepoints=changepoints,
                        milestones=milestones,
                        recommendations=recommendations_dict # Pass recommendations
                    )
                    logger.info(f"[{project_name}] Output writing process completed for {output_project_dir}")

                    # --->>> ADD DIAGNOSTIC FILE CHECK <<<---
                    logger.info(f"--- DEBUG CHECK: Verifying files after write_outputs for {project_name} ---")
                    forecast_file = os.path.join(output_project_dir, 'forecast_results.csv')
                    slippage_file = os.path.join(output_project_dir, 'slippage_summary.csv')
                    logger.info(f"Checking existence of {output_project_dir}: {os.path.exists(output_project_dir)}")
                    logger.info(f"Checking existence of {forecast_file}: {os.path.exists(forecast_file)}")
                    logger.info(f"Checking existence of {slippage_file}: {os.path.exists(slippage_file)}")
                    logger.info(f"--- END DEBUG CHECK ---")
                    # --->>> END DIAGNOSTIC FILE CHECK <<<---

                    success = True # Mark as success only if writing succeeds
                except Exception as write_e:
                    logger.error(f"[{project_name}] Error during output writing for {project_name} in {output_project_dir}: {write_e}", exc_info=True)
                    success = False # Mark as failed if writing fails

                # Archive project based on success flag - THIS CALL IS REDUNDANT DUE TO FINALLY BLOCK
                # archive_project(project_folder_path, success=success) # Removed redundant call
            else:
                # Handle case where output_dir wasn't determined (e.g., initial loading error)
                logger.error(f"[{project_name}] Output directory not set for {project_name}. Archiving as failed.")
                # archive_project(project_folder_path, success=False) # Removed redundant call

            # --- CHANGE: Return the success boolean --- 
            if success:
                logger.info(f"[{project_name}] Successfully processed {project_folder}.")
            else:
                logger.warning(f"[{project_name}] Processing {project_folder} completed with errors.")
            return output_project_dir if success else None

        except Exception as e:
            logger.error(f"[{project_name}] Output writing failed for {project_folder} to {output_project_dir}: {e}")
            # Log traceback regardless of debug flag
            logger.error(traceback.format_exc())
            success = False # Mark as failed if writing output fails

    except Exception as e:
        # Catch any other unexpected errors during the main processing block
        logger.error(f"[{project_name}] Top-level exception during processing {project_folder}: {e}")
        logger.error(traceback.format_exc())
        success = False
        # Ensure archiving still happens in finally block
        # return None # Return removed, success flag handles outcome, finally block archives
    
    finally:
        # Always attempt to archive based on the final success status
        # This ensures archiving happens even if an error occurred mid-process
        logger.info(f"[{project_name}] Finally block: Archiving '{project_folder}' based on success={success}")
        # --->>> TEMPORARILY DISABLED ARCHIVING FOR DEBUGGING <<<---
        # archive_project(project_folder_path, success=success)
        # logger.warning(f"--- ARCHIVING TEMPORARILY DISABLED in finally block for {project_folder} ---") # Original commented out line

        # --->>> RE-ENABLE ARCHIVING <<<--- # Logic from previous restore
        project_name = os.path.basename(project_folder_path) # Ensure project_name is defined in finally context
        source_path = project_folder_path  # Use the passed-in project_folder_path for archiving

        # --->>> FIX: Define destination_dir within finally block <<<---
        target_dir_suffix = 'success' if success else 'failed'
        # Ensure ARCHIVE_DIR is accessible here (it's defined at module level)
        destination_dir = os.path.join(ARCHIVE_DIR, target_dir_suffix)
        destination_path = os.path.join(destination_dir, project_name) # Full path for the project within success/failed
        # --->>> END FIX <<<---

        # --->>> ADD ARCHIVING DEBUG LOGGING <<<---
        logger.debug(f"--- Archiving Debug ---")
        logger.debug(f"Project Name in finally: {project_name}")
        logger.debug(f"Success flag value: {success}")
        logger.debug(f"Calculated source_path for move: {source_path}")
        logger.debug(f"Checking existence of source_path ({source_path})...")
        source_exists = os.path.exists(source_path)
        logger.debug(f"Result of os.path.exists(source_path): {source_exists}")
        if not source_exists:
            logger.warning(f"Source path '{source_path}' reported as non-existent *immediately before* move attempt.")
            # Optionally list contents of INPUT_DIR to see what IS there
            try:
                input_contents = os.listdir(INPUT_DIR)
                logger.debug(f"Contents of INPUT_DIR ({INPUT_DIR}): {input_contents}")
            except Exception as list_e:
                logger.error(f"Could not list contents of INPUT_DIR: {list_e}")
        # --->>> END ARCHIVING DEBUG LOGGING <<<---

        success_archive = os.path.join(ARCHIVE_DIR, 'success', project_name)
        failed_archive = os.path.join(ARCHIVE_DIR, 'failed', project_name)
        os.makedirs(success_archive, exist_ok=True)
        os.makedirs(failed_archive, exist_ok=True)

        # Move the entire project directory from input to the appropriate archive folder
        # source_path = os.path.join(INPUT_DIR, project_name) # Defined above now
        try:
            if source_exists: # Use the checked value
                # --->>> Use destination_dir defined above <<<---
                logger.info(f"[{project_name}] Moving processed project '{project_name}' from '{source_path}' to '{destination_dir}'")
                
                # --- FIX: Remove destination if it exists before moving ---
                # --->>> Use destination_path defined above <<<---
                if os.path.exists(destination_path):
                    logger.warning(f"[{project_name}] Destination archive path '{destination_path}' already exists. Removing it before move.")
                    try:
                        # --->>> Use destination_path defined above <<<---
                        if os.path.isdir(destination_path):
                             shutil.rmtree(destination_path)
                        else:
                             os.remove(destination_path)
                    except OSError as rm_err:
                        logger.error(f"[{project_name}] Failed to remove existing archive item at '{destination_path}': {rm_err}. Skipping archive.", exc_info=True)
                        # Skip the move if removal fails
                        # Consider setting success=False here?
                        # return # Removing return to allow final block to complete fully
                    else:
                         logger.debug(f"Successfully removed existing item at {destination_path}")

                # --- End FIX ---

                # --->>> Use destination_dir defined above <<<---
                shutil.move(source_path, destination_dir) # Move project *into* the success/failed dir
                logger.info(f"[{project_name}] Successfully moved '{project_name}' to archive: {destination_dir}")
            else:
                # Log already handled by the debug check above
                logger.warning(f"[{project_name}] Skipping move of '{project_name}' because source path was not found.")
                # logger.warning(f"Source path '{source_path}' does not exist. Cannot archive '{project_name}'. This might happen if it was already moved or deleted.") # Redundant log
        except Exception as e:
             # --->>> Use destination_dir defined above <<<---
             logger.error(f"[{project_name}] Error archiving project '{project_name}' to '{destination_dir}': {e}", exc_info=True)
        # --->>> END RE-ENABLE <<<---

        # --- ADDED: Archiving logic --- # THIS SECTION SEEMS REDUNDANT NOW
        # logger.info(f"[{project_name}] Attempting to archive {project_folder} (Success: {success})")
        # try:
        #     # --->>> ADD INPUT EXISTENCE CHECK 3 <<< ---
        #     logger.debug(f"CHECK 3: Before archiving '{project_name}'")
        #     logger.debug(f"  Input Dir '{INPUT_DIR}' exists: {os.path.exists(INPUT_DIR)}")
        #     logger.debug(f"  Project Path '{project_folder_path}' exists: {os.path.exists(project_folder_path)}")
        #     # --->>> END CHECK 3 <<< ---
        #     
        #     # Only attempt to archive if the source exists
        #     if os.path.exists(project_folder_path):
        #         # --- REMOVE call to archive_project as logic is now inline --- 
        #         # archive_project(project_folder_path, success)
        #         pass # Logic is handled above now
        #     else:
        #         logger.warning(f"[{project_name}] Cannot archive {project_folder}, source path {project_folder_path} no longer exists.")
        #         # If the source is gone, we probably still want to signal overall failure if success wasn't True
        #         if not success: 
        #             logger.warning(f"[{project_name}] Marking process as failed since source is gone and success was not True.")
        #             success = False # Setting success flag is handled by the main try/except block
        #             
        # except Exception as e_archive:
        #     logger.critical(f"[{project_name}] CRITICAL: Failed to archive project {project_folder} from path {project_folder_path}: {e_archive}", exc_info=True)
        #     success = False # Mark overall success as False if archive fails
        # --- END REDUNDANT SECTION ---

        # Return the output directory path only if processing was successful *overall*
        # The 'success' variable now reflects the status *after* attempting archiving.
        if success:
            logger.info(f"[{project_name}] process_project completed successfully for {project_folder}. Returning output path: {output_project_dir}")
            return output_project_dir
        else:
            logger.warning(f"[{project_name}] process_project finished with errors or failed archiving for {project_folder}. Returning None.")
            return None
