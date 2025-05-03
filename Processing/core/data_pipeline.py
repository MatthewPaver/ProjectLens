#!/usr/bin/env python3
"""Data Processing Pipeline Orchestration Module.

This module defines the main workflow for processing a single project's data.
It orchestrates the sequence of loading, cleaning, analysing, and outputting results.

Key Components:
- Directory Constants: Defines standard paths for input, output, archive, schemas.
- `_archive_project`: Helper function to move processed project folders.
- `process_project`: The core function that takes a project folder path and:
    1. Loads data using `file_loader`.
    2. Cleans data using `data_cleaning` and `SchemaManager`.
    3. Runs analysis modules (`slippage_analysis`, `forecast_engine`, etc.).
    4. Generates recommendations using `recommendation_engine`.
    5. Writes all outputs using `output_writer`.
    6. Archives the input folder based on overall success.

The pipeline is designed to be somewhat fault-tolerant; failures in individual 
analysis modules are logged but typically don't stop processing for the project.
However, critical failures in loading, cleaning, or output writing will mark the
project as failed for archiving.
"""
import os
import pandas as pd
import logging
import traceback
import shutil

from Processing.core.config_loader import resolve_path
from Processing.core.schema_manager import SchemaManager
from Processing.core.data_cleaning import clean_dataframe

from Processing.ingestion.file_loader import load_project_files

# Analysis module imports
from Processing.analysis.slippage_analysis import run_slippage_analysis
from Processing.analysis.forecast_engine import run_forecasting
from Processing.analysis.changepoint_detector import detect_change_points
from Processing.analysis.milestone_analysis import analyse_milestones
from Processing.analysis.recommendation_engine import generate_recommendations

# Output module import
from Processing.output.output_writer import write_outputs

# --- Logger Setup ---
# Get the logger instance configured by the main entry point (main_runner.py)
logger = logging.getLogger(__name__) 

# --- Directory Constants --- 
# Use resolve_path to locate Data directories relative to project root
INPUT_DIR   = resolve_path("Data/input")    
OUTPUT_DIR  = resolve_path("Data/output")   
ARCHIVE_DIR = resolve_path("Data/archive")  
SCHEMA_DIR  = resolve_path("Data/schemas")  

# Ensure Output and Archive directories exist (essential for this script)
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(ARCHIVE_DIR, 'success'), exist_ok=True)
    os.makedirs(os.path.join(ARCHIVE_DIR, 'failed'), exist_ok=True)
except OSError as e:
     logger.critical(f"Failed to create necessary Output/Archive directories: {e}")
     # Consider exiting if these cannot be created
     # sys.exit(1)
     
# --- Archive Function (Local Helper) ---
def _archive_project(project_folder_path: str, success: bool):
    """Moves the processed project folder to the success or failed archive directory.
    
    Handles existing destinations by removing them first.
    
    Args:
        project_folder_path (str): Full path to the project folder in the input directory.
        success (bool): Whether the processing was successful.
    """
    project_folder_name = os.path.basename(project_folder_path)
    target_dir_suffix = 'success' if success else 'failed'
    target_archive_base = os.path.join(ARCHIVE_DIR, target_dir_suffix)
    destination_path = os.path.join(target_archive_base, project_folder_name)
    
    logger.info(f"Archiving '{project_folder_name}' to '{target_archive_base}' (Success: {success})")
    
    # Ensure the specific success/failed directory exists
    os.makedirs(target_archive_base, exist_ok=True)
    
    # Check source exists
    if not os.path.exists(project_folder_path):
        logger.warning(f"Source path for archiving not found: {project_folder_path}. Skipping archive.")
        return
        
    # Remove existing destination if necessary
    if os.path.exists(destination_path):
        logger.warning(f"Destination archive path {destination_path} already exists. Removing before archiving.")
        try:
            if os.path.isdir(destination_path):
                 shutil.rmtree(destination_path)
            else:
                 os.remove(destination_path)
            logger.debug(f"Removed existing item at {destination_path}.")
        except Exception as e_rem:
            logger.error(f"Failed to remove existing item at {destination_path}: {e_rem}. Archiving aborted.")
            return # Stop if we can't clear the destination
             
    # Move the source folder to the target base directory (success/failed)
    try:
        shutil.move(project_folder_path, target_archive_base) 
        logger.info(f"Successfully moved project '{project_folder_name}' to {target_archive_base}")
    except Exception as e_mov:
        logger.error(f"Failed to move project '{project_folder_name}' to archive directory {target_archive_base}: {e_mov}", exc_info=True)

# Note: This module relies on output_writer.write_outputs for saving analysis results

def process_project(project_folder_path: str, schema_manager: SchemaManager) -> str | None:
    """
    Processes a single project's data through the entire analysis pipeline.
    
    This function orchestrates the sequence of operations for one project:
    1.  Loads data from files within the specified project folder.
    2.  Cleans and standardises the loaded data using the provided SchemaManager.
    3.  Executes various analysis modules (slippage, forecasting, changepoints, milestones).
    4.  Generates project recommendations based on analysis results.
    5.  Writes all generated outputs (cleaned data, analysis results, recommendations) 
        to a corresponding folder in the main output directory.
    6.  Archives the original input project folder to 'success' or 'failed' based on 
        whether all steps (including output writing) completed without critical errors.
    
    Individual analysis module failures are logged but do not necessarily stop the entire 
    pipeline for the project; subsequent steps will proceed with potentially incomplete results.
    However, critical failures during loading, cleaning, or output writing will result in 
    the project being marked as failed for archiving.
    
    Args:
        project_folder_path (str): The full path to the specific project folder 
                                   located within the main input directory.
        schema_manager (SchemaManager): A pre-initialised instance of the SchemaManager, 
                                        configured for the appropriate schema (e.g., 'tasks').
        
    Returns:
        str or None: The path to the project's dedicated output directory if processing 
                     and output writing were successful. Returns None if any critical step 
                     failed, leading to the project being archived as 'failed'.
    """
    success = False  # Flag to track overall success for final archiving decision.
    project_name = os.path.basename(project_folder_path)
    # Define the target output directory specific to this project.
    output_project_dir = os.path.join(OUTPUT_DIR, project_name)
    logger.info(f"--- Starting Project Processing: {project_name} --- Path: {project_folder_path} ---")

    # The following ensures archiving always occurs, regardless of errors in processing steps.
    try:
        # STEP 1: LOAD DATA
        # ==================
        logger.info(f"[{project_name}] Phase 1: Loading project data files...")
        try:
            # Call the file loader to read CSV/Excel files from the project folder.
            raw_df = load_project_files(project_folder_path)
            # Check if loading returned a valid, non-empty DataFrame.
            if raw_df is None or raw_df.empty:
                logger.error(f"[{project_name}] Data loading failed or returned no data. Cannot proceed. Archiving as failed.")
                # Critical failure: Archive immediately and return None.
                _archive_project(project_folder_path, success=False)
                return None
            # Log successful load, including number of rows and source files.
            # Assuming 'file_name' column is added by load_project_files.
            file_count = raw_df['file_name'].nunique() if 'file_name' in raw_df.columns else '(unknown number)'
            logger.info(f"[{project_name}] Successfully loaded {len(raw_df)} raw rows from {file_count} files.")
        except Exception as e_load:
            # Catch any unexpected errors during file loading.
            logger.error(f"[{project_name}] CRITICAL ERROR during data loading: {e_load}", exc_info=True)
            # Critical failure: Archive and return None.
            _archive_project(project_folder_path, success=False)
            return None
        
        # STEP 2: CLEAN DATA
        # =================
        logger.info(f"[{project_name}] Phase 2: Cleaning and standardising loaded data...")
        try:
            # Call the data cleaning function, passing the raw data.
            # It will initialise its own SchemaManager internally based on schema_type.
            # Pass project_name for context-specific logging within the cleaning function.
            # Determine the schema type to use (e.g., 'tasks' is common)
            schema_type_for_cleaning = "tasks" # Or potentially load from config if needed
            cleaned_df = clean_dataframe(
                df=raw_df.copy(), # Pass a copy to avoid modifying the raw DataFrame.
                schema_type=schema_type_for_cleaning, # Pass the schema type name
                project_name=project_name
                # Contextual parameters can be extended here if needed
            )
            # Check if cleaning resulted in a valid, non-empty DataFrame.
            if cleaned_df is None or cleaned_df.empty:
                logger.error(f"[{project_name}] Data cleaning resulted in an empty or invalid DataFrame. Cannot proceed. Archiving as failed.")
                # Critical failure: Archive and return None.
                _archive_project(project_folder_path, success=False)
                return None
            logger.info(f"[{project_name}] Successfully cleaned data. Standardised shape: {cleaned_df.shape}")

        except Exception as e_clean:
            # Catch any unexpected errors during data cleaning.
            logger.error(f"[{project_name}] CRITICAL ERROR during data cleaning: {e_clean}", exc_info=True)
            # Critical failure: Archive and return None.
            _archive_project(project_folder_path, success=False)
            return None

        # Store a pristine copy of the cleaned data for the final output writer.
        # Analysis functions receive copies, but we save this original cleaned version for output.
        original_cleaned_df_for_output = cleaned_df.copy()

        # STEP 3: ANALYSIS MODULES
        # ========================
        logger.info(f"[{project_name}] Phase 3: Running analysis modules...")
        # Dictionary to store results from each analysis module.
        analysis_results = {}

        # Run each analysis module in a separate try-except block.
        # This allows the pipeline to continue even if one module fails.
        # Failed modules will typically result in an empty DataFrame or list being stored.

        # 3a. Slippage Analysis
        try:
            logger.debug(f"[{project_name}] Running slippage analysis...")
            slippages = run_slippage_analysis(cleaned_df.copy(), project_name=project_name)
            analysis_results['slippages'] = slippages if slippages is not None else pd.DataFrame()
            logger.info(f"[{project_name}] Slippage analysis complete. Rows: {len(analysis_results['slippages'])}")
        except Exception as e_slip:
            logger.error(f"[{project_name}] Error during slippage analysis: {e_slip}", exc_info=True)
            analysis_results['slippages'] = pd.DataFrame()

        # 3b. Forecasting
        try:
            logger.debug(f"[{project_name}] Running forecasting analysis...")
            forecasts, failed_forecast_tasks = run_forecasting(cleaned_df.copy(), project_name=project_name)
            analysis_results['forecasts'] = forecasts if forecasts is not None else pd.DataFrame()
            analysis_results['failed_forecast_tasks'] = failed_forecast_tasks if failed_forecast_tasks else []
            logger.info(f"[{project_name}] Forecasting complete. Rows: {len(analysis_results['forecasts'])}, Failed tasks: {len(analysis_results['failed_forecast_tasks'])}")
        except Exception as e_forecast:
            logger.error(f"[{project_name}] Error during forecasting analysis: {e_forecast}", exc_info=True)
            analysis_results['forecasts'] = pd.DataFrame()
            analysis_results['failed_forecast_tasks'] = []

        # 3c. Changepoint Detection
        try:
            logger.debug(f"[{project_name}] Running change point detection...")
            slippages_df = analysis_results.get('slippages')
            if slippages_df is not None and not slippages_df.empty and 'slip_days' in slippages_df.columns:
                if 'task_id' in cleaned_df.columns and 'update_phase' in cleaned_df.columns:
                    if 'task_name' in cleaned_df.columns:
                        context_cols = ['task_id', 'task_name', 'update_phase']
                        cleaned_context = cleaned_df[context_cols].drop_duplicates(subset=['task_id', 'update_phase'])
                    else:
                        logger.warning(f"[{project_name}] task_name column missing in cleaned_df. Creating fallback from task_id.")
                        context_cols = ['task_id', 'update_phase']
                        cleaned_context = cleaned_df[context_cols].drop_duplicates(subset=['task_id', 'update_phase'])
                        cleaned_context['task_name'] = cleaned_context['task_id'].apply(lambda x: f"Task {x}")
                    changepoint_input_df = pd.merge(
                        slippages_df,
                        cleaned_context,
                        on=['task_id', 'update_phase'],
                        how='left'
                    )
                    logger.debug(f"[{project_name}] Prepared merged DataFrame for changepoint detection. Rows: {len(changepoint_input_df)}")
                else:
                    logger.warning(f"[{project_name}] Cannot run changepoint: Missing crucial columns in cleaned_df. Using slippages_df directly.")
                    changepoint_input_df = slippages_df.copy()
                    if 'task_name' not in changepoint_input_df.columns and 'task_id' in changepoint_input_df.columns:
                        changepoint_input_df['task_name'] = changepoint_input_df['task_id'].apply(lambda x: f"Task {x}")
            else:
                 logger.warning(f"[{project_name}] Skipping changepoint detection: No valid slippages data found.")
                 changepoint_input_df = pd.DataFrame()
            if not changepoint_input_df.empty:
                 changepoints = detect_change_points(changepoint_input_df, project_name=project_name)
            else:
                 changepoints = pd.DataFrame()
            analysis_results['changepoints'] = changepoints if changepoints is not None else pd.DataFrame()
            logger.info(f"[{project_name}] Change point detection complete. Rows: {len(analysis_results['changepoints'])}")
        except Exception as e_change:
            logger.error(f"[{project_name}] Error during change point detection: {e_change}", exc_info=True)
            analysis_results['changepoints'] = pd.DataFrame()

        # 3d. Milestone Analysis
        try:
            logger.debug(f"[{project_name}] Running milestone analysis...")
            milestones = analyse_milestones(cleaned_df.copy()) 
            analysis_results['milestones'] = milestones if milestones is not None else pd.DataFrame()
            logger.info(f"[{project_name}] Milestone analysis complete. Rows: {len(analysis_results['milestones'])}")
        except Exception as e_milestone:
            logger.error(f"[{project_name}] Error during milestone analysis: {e_milestone}", exc_info=True)
            analysis_results['milestones'] = pd.DataFrame()

        # STEP 4: RECOMMENDATION GENERATION
        logger.info(f"[{project_name}] Phase 4: Generating recommendations...")
        try:
            recommendations_list = generate_recommendations(
                df=cleaned_df,
                slippages=analysis_results.get('slippages'), 
                forecasts=analysis_results.get('forecasts'), 
                changepoints=analysis_results.get('changepoints'), 
                milestones=analysis_results.get('milestones'),
                failed_forecast_tasks=analysis_results.get('failed_forecast_tasks')
            )
            if recommendations_list is None:
                recommendations_list = []
            logger.info(f"[{project_name}] Recommendation generation complete. Count: {len(recommendations_list)}")
        except Exception as e_rec:
            logger.error(f"[{project_name}] Error generating recommendations: {e_rec}", exc_info=True)
            recommendations_list = []

        # STEP 5: WRITE OUTPUTS
        # =====================
        # This step is considered critical. Failure here marks the project processing as failed.
        logger.info(f"[{project_name}] Phase 5: Writing outputs to directory: {output_project_dir}")
        try:
            # Ensure the project-specific output directory exists.
            os.makedirs(output_project_dir, exist_ok=True)
            # Call the output writer function to save all relevant DataFrames and lists.
            
            # Create analysis_results dict with recommendations included
            analysis_results['recommendations'] = recommendations_list
            
            # Call with updated signature matching output_writer.py
            write_outputs(
                output_path=output_project_dir,
                project_name=project_name,
                cleaned_df=original_cleaned_df_for_output,
                analysis_results=analysis_results
            )
            
            logger.info(f"[{project_name}] Output writing completed successfully.")
            # Set success flag to True ONLY if output writing completes without error.
            success = True 
        except Exception as e_write:
            # Catch any errors during file writing.
            logger.error(f"[{project_name}] CRITICAL ERROR during output writing: {e_write}", exc_info=True)
            # Ensure success flag remains False if output writing fails.
            success = False 

    except Exception as e_main:
        # Catch any top-level unexpected errors occurring outside the specific steps above.
        logger.critical(f"[{project_name}] UNHANDLED CRITICAL EXCEPTION during main processing workflow: {e_main}", exc_info=True)
        success = False # Ensure marked as failed if an unexpected error occurs.
    
    finally:
        # STEP 6: ARCHIVE INPUT DATA
        # ==========================
        # This block executes regardless of whether exceptions occurred in the 'try' block.
        # The 'success' flag (set after output writing) determines the archive location.
        logger.info(f"[{project_name}] Phase 6: Archiving input data based on final success status ({success})...")
        _archive_project(project_folder_path, success=success)
        logger.info(f"--- Finished Project Processing: {project_name} --- Outcome: {'Success' if success else 'Failed'} ---")

    # Return the path to the output directory only if the process was successful.
    # Otherwise, return None to indicate failure.
    return output_project_dir if success else None
