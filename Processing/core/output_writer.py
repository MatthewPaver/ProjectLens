import os
import pandas as pd
import logging

# Assume logger is configured elsewhere
logger = logging.getLogger(__name__)

# Define standard headers for empty files (adjust as needed)
# These are now only used if the original logic were restored
HEADERS = {
    "task_cleaned.csv": "project_name,task_code,task_name,baseline_start_date,baseline_end_date,actual_start,actual_finish,duration_orig,remaining_duration,actual_duration,deviation_days,percent_complete,status,is_critical,severity_score,change_type,slip_days\n",
    "slippage_summary.csv": "project_name,task_code,update_phase,current_finish,baseline_finish,slippage_type,slip_days,severity_score,change_type\n",
    "changepoints.csv": "project_name,task_id,change_point_date,change_type,magnitude\n",
    "milestone_analysis.csv": "project_name,task_id,milestone_name,status,variance_days\n",
    "forecast_results.csv": "project_name,task_code,task_name,forecast_date,confidence_interval_lower,confidence_interval_upper,model_type,low_confidence_flag\n",
    "recommendations.csv": "project_name,task_id,recommendation_type,priority,description,created_at\n"
}

def _save_df(df, filepath, filename, file_headers=""):
    """Helper to save DataFrame, creating empty file with headers if df is None/empty."""
    try:
        # --- Restore original Pandas save logic --- 
        if df is not None and not df.empty:
            # Ensure necessary columns from headers exist, add if missing with default None/0
            # Note: HEADERS constant provides expected cols, but this logic is simplified
            # to just save the columns present in the passed DataFrame.
            # For more robust header/column alignment, the schema definition should be the source of truth.
            
            # Log columns being saved
            logger.debug(f"Saving DataFrame for {filename} with columns: {list(df.columns)}")

            # Use ISO 8601 format for dates
            df.to_csv(filepath, index=False, date_format='%Y-%m-%d %H:%M:%S') 
            logger.info(f"Successfully wrote {filename} ({df.shape[0]} rows).")
        else:
            # Handle empty or None DataFrame
            logger.warning(f"Input DataFrame for {filename} is None or empty. Creating file with headers only.")
            # Try to get headers from the HEADERS constant if defined, otherwise write empty file
            headers_to_write = file_headers if file_headers else "" # Use provided headers
            with open(filepath, 'w') as f:
                f.write(headers_to_write) # Write headers (or empty string) for empty file
            logger.info(f"Successfully wrote empty {filename} (with headers if available).")
        # --- End restore --- 

    except Exception as e:
        # Catch errors during file writing
        logger.error(f"Error saving {filename} to {filepath}: {e}", exc_info=True)


def write_outputs(output_path: str, **kwargs):
    """Writes all analysis results to CSV files in the specified output path."""
    logger.info(f"--- Starting write_outputs V2 (SIMPLIFIED SAVE) for {output_path} ---")
    os.makedirs(output_path, exist_ok=True)

    # Get dataframes from kwargs (needed for mapping, even if not saved)
    cleaned_df = kwargs.get('cleaned_df')
    slippages = kwargs.get('slippages')
    forecasts = kwargs.get('forecasts')
    changepoints = kwargs.get('changepoints')
    milestones = kwargs.get('milestones')
    recommendations = kwargs.get('recommendations')

    # Define file mappings (filename -> dataframe, headers)
    # DataFrames are passed to _save_df but ignored by the simplified logic
    files_to_write = {
        "task_cleaned.csv": (cleaned_df, HEADERS["task_cleaned.csv"]),
        "slippage_summary.csv": (slippages, HEADERS["slippage_summary.csv"]),
        "changepoints.csv": (changepoints, HEADERS["changepoints.csv"]),
        "milestone_analysis.csv": (milestones, HEADERS["milestone_analysis.csv"]),
        "forecast_results.csv": (forecasts, HEADERS["forecast_results.csv"]),
        "recommendations.csv": (recommendations, HEADERS["recommendations.csv"])
    }

    for filename, (df, headers) in files_to_write.items():
        filepath = os.path.join(output_path, filename)
        logger.info(f"Preparing to write {filename} (simplified method)...")
        # Pass df and headers, although _save_df currently ignores them
        _save_df(df, filepath, filename, headers)

    logger.info(f"--- Finished write_outputs V2 (SIMPLIFIED SAVE) for {output_path} ---") 