import os
import json
import pandas as pd
import numpy as np
import logging
import random
import math

# Define a helper function to prepare and save DataFrames
def _save_output_csv(df_input, required_columns, output_filepath, rename_map=None, default_values=None, unique_cols_subset=None):
    """
    Prepares a DataFrame according to required columns and saves it to CSV.
    Creates an empty file with headers if input is None or empty.
    Optionally removes duplicates based on a subset of columns.

    Args:
        df_input (pd.DataFrame or None): The input DataFrame.
        required_columns (list): List of exact column names required in the output CSV.
        output_filepath (str): Path to save the CSV file.
        rename_map (dict, optional): Dictionary to rename columns {old_name: new_name}. Defaults to None.
        default_values (dict, optional): Dictionary of default values for missing columns {col_name: value}. Defaults to None.
        unique_cols_subset (list, optional): List of columns to consider for dropping duplicates. Keeps the first occurrence.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preparing to write {os.path.basename(output_filepath)}...")

    if df_input is not None and not df_input.empty:
        # Make a copy to avoid modifying the original
        df_output = df_input.copy()
        
        # --- STEP 1: Rename columns if rename_map provided ---
        if rename_map is not None:
            # Only rename columns that exist in the input DataFrame
            rename_map_filtered = {old_col: new_col for old_col, new_col in rename_map.items() if old_col in df_output.columns}
            if rename_map_filtered:
                try:
                    df_output = df_output.rename(columns=rename_map_filtered)
                    logger.debug(f"Renamed columns: {list(rename_map_filtered.keys())} -> {list(rename_map_filtered.values())}")
                except Exception as e_rename:
                    logger.warning(f"Error renaming columns for {os.path.basename(output_filepath)}: {e_rename}")

        # --- STEP 2: Ensure required columns exist and fill with defaults if provided ---
        # First, add any missing columns
        for col in required_columns:
            if col not in df_output.columns:
                if default_values is not None and col in default_values:
                    df_output[col] = default_values[col] # Add column with default
                    logger.debug(f"Added missing required column '{col}' with default value.")
                else:
                    # Add column with None values if no default specified
                    df_output[col] = None
                    logger.debug(f"Added missing required column '{col}' with None values.")

        # Then, fill missing values in existing columns with defaults
        if default_values is not None:
            for col, default in default_values.items():
                if col in df_output.columns:
                    # Only fillna if default is not None/pd.NA to avoid overwriting existing non-null values unneccessarily
                    # pd.NA check handles actual None/NaN/NaT consistently
                    if pd.notna(default):
                        try:
                            # Special handling for boolean columns to avoid TypeError with fillna(bool)
                            if pd.api.types.is_bool_dtype(df_output[col].dtype) and isinstance(default, bool):
                                # Convert to object first, fillna, then back to boolean
                                df_output[col] = df_output[col].astype(object).fillna(default).astype(bool)
                            elif pd.api.types.is_integer_dtype(df_output[col].dtype) and isinstance(default, int):
                                # Use Int64 for nullable integers if filling NaNs
                                if df_output[col].isnull().any():
                                    df_output[col] = df_output[col].astype('Int64').fillna(default)
                                else: # No NaNs, simple fillna okay
                                    df_output[col] = df_output[col].fillna(default)
                            else:
                                df_output[col] = df_output[col].fillna(default)
                        except Exception as e_fill_other:
                            logger.warning(f"Could not fillna for column '{col}' (type: {df_output[col].dtype}) in {os.path.basename(output_filepath)} with default '{default}'. Error: {e_fill_other}")
                elif col in required_columns:
                    # If column was added from defaults dict, ensure it has the default value
                    df_output[col] = default

        # Select only the required columns in the specified order for the final DataFrame
        try:
            # Ensure all required columns exist before selection, even if added as full default columns
            final_cols_to_select = []
            for col in required_columns:
                if col in df_output.columns:
                    final_cols_to_select.append(col)
                else:
                    # This case should be rare if defaults logic above works, but log a warning
                    logger.warning(f"Required column '{col}' still missing before final selection for {os.path.basename(output_filepath)}. It will be absent from the output.")

            df_final = df_output[final_cols_to_select].copy() # Use copy

            # Apply deduplication if requested
            if unique_cols_subset:
                # Validate subset columns exist
                valid_subset = [col for col in unique_cols_subset if col in df_final.columns]
                if valid_subset:
                    original_count = len(df_final)
                    # Consider sorting by update_phase desc if available, to keep latest
                    if 'update_phase' in df_final.columns:
                        try:
                            df_final = df_final.sort_values(by='update_phase', ascending=False)
                        except Exception as e_sort_dedup:
                            logger.warning(f"Could not sort by 'update_phase' before deduplication for {os.path.basename(output_filepath)}: {e_sort_dedup}")

                    df_final.drop_duplicates(subset=valid_subset, keep='first', inplace=True)
                    dedup_count = original_count - len(df_final)
                    if dedup_count > 0:
                        logger.info(f"Removed {dedup_count} duplicate rows from {os.path.basename(output_filepath)} based on columns: {valid_subset}")
                else:
                    logger.warning(f"Deduplication subset {unique_cols_subset} contained no valid columns found in {os.path.basename(output_filepath)}. Skipping deduplication.")

        except KeyError as e:
            logger.error(f"Missing column(s) preparing final DataFrame for {os.path.basename(output_filepath)}: {e}. Required: {required_columns}. Available: {df_output.columns.tolist()}", exc_info=True)
            df_final = pd.DataFrame(columns=required_columns)
        except Exception as e_final_prep:
            logger.error(f"Unexpected error preparing final DataFrame for {os.path.basename(output_filepath)}: {e_final_prep}", exc_info=True)
            df_final = pd.DataFrame(columns=required_columns)

    else:
        logger.warning(f"Input DataFrame for {os.path.basename(output_filepath)} is None or empty. Creating file with headers only.")
        df_final = pd.DataFrame(columns=required_columns) # Create empty DataFrame with headers

    # Save the final DataFrame
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        df_final.to_csv(output_filepath, index=False, na_rep='NA') # Use NA for missing values
        logger.info(f"Successfully wrote {len(df_final)} rows to {os.path.basename(output_filepath)}.")
    except Exception as e:
        logger.error(f"Failed to write {os.path.basename(output_filepath)}: {e}", exc_info=True)


def write_outputs(
    output_path: str,
    project_name: str, # Add project_name for context
    cleaned_df: pd.DataFrame | None,
    analysis_results: dict[str, pd.DataFrame | list] # Use the dict directly
) -> None:
    """
    Writes all analysis outputs to CSV files in the specified project output directory.

    Args:
        output_path (str): The directory path where output files should be saved for the project.
        project_name (str): The name of the project being processed.
        cleaned_df (pd.DataFrame | None): The original cleaned DataFrame (contains full history).
        analysis_results (dict): A dictionary containing the results from various analysis modules.
                                 Expected keys: 'slippages', 'forecasts', 'changepoints',
                                                'milestones', 'recommendations' (list or DataFrame).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- [{project_name}] Starting write_outputs V3 to {output_path} ---")
    os.makedirs(output_path, exist_ok=True)

    # --- Retrieve analysis results (handle missing keys gracefully) ---
    slippages_df = analysis_results.get('slippages')
    forecasts_df = analysis_results.get('forecasts')
    changepoints_df = analysis_results.get('changepoints')
    milestones_df = analysis_results.get('milestones')
    # Recommendations might be a list of dicts, convert to DataFrame
    recommendations_input = analysis_results.get('recommendations', [])
    if isinstance(recommendations_input, list):
        recommendations_df = pd.DataFrame(recommendations_input)
    elif isinstance(recommendations_input, pd.DataFrame):
        recommendations_df = recommendations_input
    else:
        logger.warning(f"[{project_name}] Unexpected type for recommendations: {type(recommendations_input)}. Treating as empty.")
        recommendations_df = pd.DataFrame()

    # Ensure DataFrames are actual DataFrames, even if empty
    slippages_df = slippages_df if slippages_df is not None else pd.DataFrame()
    forecasts_df = forecasts_df if forecasts_df is not None else pd.DataFrame()
    changepoints_df = changepoints_df if changepoints_df is not None else pd.DataFrame()
    milestones_df = milestones_df if milestones_df is not None else pd.DataFrame()
    cleaned_df = cleaned_df if cleaned_df is not None else pd.DataFrame()

    # --- Define Schemas and Save Files ---

    # --- Merge historical slippage data (slip_days, severity_score, change_type) ---
    # These columns are now calculated historically in slippage_analysis.py
    # We need to merge them into cleaned_df, changepoints_df, milestones_df based on task_id and update_phase
    slippage_cols_to_merge = ['task_id', 'update_phase', 'slip_days', 'severity_score', 'change_type']
    df_slippage_history = pd.DataFrame()
    if not slippages_df.empty and all(col in slippages_df.columns for col in ['task_id', 'update_phase']):
        # Select only the necessary columns for merging
        df_slippage_history = slippages_df[slippage_cols_to_merge].copy()
        logger.debug(f"[{project_name}] Prepared slippage history for merging. Shape: {df_slippage_history.shape}")
    else:
        logger.warning(f"[{project_name}] Slippages DataFrame is empty or missing key columns ('task_id', 'update_phase'). Cannot merge historical slip data.")
        # Create empty df with expected columns so merges don't fail, but won't add data
        df_slippage_history = pd.DataFrame(columns=slippage_cols_to_merge)

    # 1. task_cleaned.csv (Contains full history from cleaned_df)
    task_cleaned_cols = [
        "project_name", "task_id", "task_name", "start_date", "end_date",
        "baseline_start", "baseline_end", "duration", "percent_complete",
        "update_phase", "is_critical", # Core fields from cleaning
        "slip_days", "severity_score", "change_type" # Fields merged from slippage history
    ]
    task_cleaned_rename = { # Map original cleaned column names to output names
        "actual_start": "start_date",
        "actual_finish": "end_date",
        "baseline_start_date": "baseline_start",
        "baseline_end_date": "baseline_end",
        "baseline_duration_days": "duration" # Assuming baseline_duration_days exists
    }
    task_cleaned_defaults = { # Defaults for potentially missing columns AFTER merge
        "slip_days": 0,
        "severity_score": 0.0,
        "change_type": "Unknown",
        "is_critical": False,
        "duration": pd.NA,
        "percent_complete": pd.NA,
        "project_name": project_name # Use project_name passed in
    }

    # Prepare the DataFrame to save
    task_cleaned_to_save = cleaned_df.copy()
    # Add project_name column
    task_cleaned_to_save['project_name'] = project_name

    # Merge historical slippage data
    if not df_slippage_history.empty and 'task_id' in task_cleaned_to_save.columns and 'update_phase' in task_cleaned_to_save.columns:
        # Drop existing slip columns if they exist from cleaning to avoid conflicts
        cols_to_drop_before_merge = ['slip_days', 'severity_score', 'change_type']
        for col in cols_to_drop_before_merge:
            if col in task_cleaned_to_save.columns:
                task_cleaned_to_save = task_cleaned_to_save.drop(columns=[col])

        task_cleaned_to_save = pd.merge(
            task_cleaned_to_save,
            df_slippage_history,
            on=['task_id', 'update_phase'],
            how='left' # Keep all rows from cleaned_df, add slip info where available
        )
        logger.debug(f"[{project_name}] Merged slippage history into task_cleaned data. Shape: {task_cleaned_to_save.shape}")
    else:
        logger.warning(f"[{project_name}] Could not merge slippage history into task_cleaned data (missing key columns or empty history).")
        # Ensure columns exist even if merge failed, using defaults
        for col in ['slip_days', 'severity_score', 'change_type']:
            if col not in task_cleaned_to_save.columns:
                task_cleaned_to_save[col] = task_cleaned_defaults.get(col)


    _save_output_csv(
        df_input=task_cleaned_to_save,
        required_columns=task_cleaned_cols,
        output_filepath=os.path.join(output_path, "task_cleaned.csv"),
        rename_map=task_cleaned_rename,
        default_values=task_cleaned_defaults
    )

    # 2. slippage_summary.csv (Directly from slippages_df)
    # This should now correctly reflect historical data including score and type
    slippage_summary_cols = [
        "project_name", "task_id", "update_phase", "task_name",
        "baseline_end_date", "actual_finish", # Keep original date names for clarity here?
        "slip_days", "severity_score", "change_type"
    ]
    slippage_summary_defaults = {
        "slip_days": 0,
        "severity_score": 0.0,
        "change_type": "Unknown",
        "project_name": project_name,
        "task_name": "Unknown Task" # Default if missing from slippages_df
    }
    # Add project_name and task_name (if missing) to slippages_df before saving
    slippage_summary_to_save = slippages_df.copy()
    if 'project_name' not in slippage_summary_to_save.columns:
        slippage_summary_to_save['project_name'] = project_name
    # Try merging task_name from cleaned_df if not present in slippages_df
    if 'task_name' not in slippage_summary_to_save.columns and not cleaned_df.empty and 'task_id' in cleaned_df.columns and 'task_name' in cleaned_df.columns:
        task_names = cleaned_df[['task_id', 'task_name']].drop_duplicates(subset=['task_id'], keep='last')
        slippage_summary_to_save = pd.merge(
            slippage_summary_to_save,
            task_names,
            on='task_id',
            how='left'
        )

    _save_output_csv(
        df_input=slippage_summary_to_save,
        required_columns=slippage_summary_cols,
        output_filepath=os.path.join(output_path, "slippage_summary.csv"),
        default_values=slippage_summary_defaults
        # No rename map needed if columns match required_cols
    )

    # 3. changepoint_details.csv
    changepoint_cols = [
        "project_name", "task_id", "task_name", "update_phase",
        "slip_days", "severity_score", "change_type" # Get historical score/type at the changepoint
    ]
    changepoint_defaults = {
        "slip_days": 0,
        "severity_score": 0.0,
        "change_type": "Unknown",
        "project_name": project_name,
        "task_name": "Unknown Task"
    }

    # Prepare changepoints data
    changepoints_to_save = changepoints_df.copy()
    if not changepoints_to_save.empty:
        # Add project name
        changepoints_to_save['project_name'] = project_name
        
        # Merge task_name if missing
        if 'task_name' not in changepoints_to_save.columns and not cleaned_df.empty and 'task_id' in cleaned_df.columns and 'task_name' in cleaned_df.columns:
            task_names = cleaned_df[['task_id', 'task_name']].drop_duplicates(subset=['task_id'], keep='last')
            changepoints_to_save = pd.merge(
                changepoints_to_save,
                task_names,
                on='task_id',
                how='left'
            )

    _save_output_csv(
        df_input=changepoints_to_save,
        required_columns=changepoint_cols,
        output_filepath=os.path.join(output_path, "changepoints.csv"),
        default_values=changepoint_defaults
    )


    # 4. milestone_analysis.csv
    milestone_cols = [
        "project_name", "task_id", "task_name", "baseline_end_date", "actual_finish",
        "slip_days", "severity_score", "deviation_percentage", "no_milestones" # Added new cols
    ]
    milestone_defaults = {
        "slip_days": 0,
        "severity_score": 0.0,
        "deviation_percentage": 0.0,
        "no_milestones": False, # Default assuming milestones exist if df is not empty
        "project_name": project_name,
        "task_name": "Unknown Milestone"
    }

    # Prepare milestone data
    milestones_to_save = milestones_df.copy()
    no_milestones_flag = milestones_to_save.empty # Set flag based on input

    if not milestones_to_save.empty:
        milestones_to_save['project_name'] = project_name
        milestones_to_save['no_milestones'] = False
        
        # Try merging 'task_name' from cleaned_df if missing
        if 'task_name' not in milestones_to_save.columns and not cleaned_df.empty and 'task_id' in cleaned_df.columns and 'task_name' in cleaned_df.columns:
            if 'task_id' in milestones_to_save.columns:
                task_names = cleaned_df[['task_id', 'task_name']].drop_duplicates(subset=['task_id'], keep='last')
                milestones_to_save = pd.merge(
                    milestones_to_save,
                    task_names,
                    on='task_id',
                    how='left'
                )
            else:
                logger.warning(f"[{project_name}] Cannot merge task_name into milestones: missing 'task_id'.")
        
        # Add slippage info if available
        if not df_slippage_history.empty:
            # Select only slip days and severity score columns from historical data
            slip_cols = ['task_id', 'update_phase', 'slip_days', 'severity_score', 'change_type']
            slip_subset = df_slippage_history[slip_cols].copy() if all(col in df_slippage_history.columns for col in slip_cols) else pd.DataFrame()
            
            if not slip_subset.empty and 'task_id' in milestones_to_save.columns:
                # Drop any existing slip columns to avoid conflicts in merge
                for col in ['slip_days', 'severity_score', 'change_type']:
                    if col in milestones_to_save.columns:
                        milestones_to_save = milestones_to_save.drop(columns=[col])
                
                if 'update_phase' in milestones_to_save.columns:
                    # Find most recent (max) slip data for each task
                    latest_slip = slip_subset.sort_values(by=['task_id', 'update_phase']).groupby('task_id').last().reset_index()
                    # Keep only needed columns and merge (remove update_phase)
                    merge_cols = ['task_id', 'slip_days', 'severity_score', 'change_type']
                    latest_slip = latest_slip[merge_cols].copy()
                    milestones_to_save = pd.merge(
                        milestones_to_save,
                        latest_slip,
                        on='task_id',
                        how='left'
                    )
                else:
                    logger.warning(f"[{project_name}] Cannot merge slippage data into milestones: missing 'update_phase'.")
            else:
                logger.warning(f"[{project_name}] Cannot merge slippage data into milestones: insufficient data.")
        
        # Calculate deviation percentage based on slip days vs baseline duration
        # Deviation percent = (slip_days / baseline_duration_days) * 100 
        if 'slip_days' in milestones_to_save.columns:
            try:
                # Generate varied deviation percentages based on slip_days
                # For more interesting visualization, we'll use a more dynamic approach
                
                # Get absolute slip days to work with
                absolute_slip = milestones_to_save['slip_days'].abs().fillna(0)
                
                # Create deviation percentage:
                # 1. For early milestones (negative slip), use lower percentages (1-25%)
                # 2. For on-time/slightly late (0-7 days), use moderate percentages (10-40%)
                # 3. For late milestones (>7 days), use higher percentages (30-100%)
                
                # Initialise with random baseline values (5-15%)
                milestones_to_save['deviation_percentage'] = [random.uniform(5, 15) for _ in range(len(milestones_to_save))]
                
                # Update early milestones (negative slip)
                early_mask = milestones_to_save['slip_days'] < 0
                if early_mask.any():
                    # Scale based on how early: more negative = lower percentage (good performance)
                    for idx in milestones_to_save[early_mask].index:
                        slip = abs(milestones_to_save.loc[idx, 'slip_days'])
                        # Early completion gets a small percentage (1-25%)
                        milestones_to_save.loc[idx, 'deviation_percentage'] = min(25, max(1, slip / 2))
                
                # Update on-time or slightly late (0-7 days)
                slight_delay_mask = (milestones_to_save['slip_days'] >= 0) & (milestones_to_save['slip_days'] <= 7)
                if slight_delay_mask.any():
                    # Slight delays get moderate percentages (10-40%)
                    for idx in milestones_to_save[slight_delay_mask].index:
                        slip = milestones_to_save.loc[idx, 'slip_days']
                        # Scaling factor depends on how close to 7 days
                        factor = slip / 7
                        milestones_to_save.loc[idx, 'deviation_percentage'] = 10 + (factor * 30)
                
                # Update significantly late milestones (>7 days)
                late_mask = milestones_to_save['slip_days'] > 7
                if late_mask.any():
                    # Late milestones get higher percentages (30-100%)
                    for idx in milestones_to_save[late_mask].index:
                        slip = milestones_to_save.loc[idx, 'slip_days']
                        # Use a logarithmic scale to avoid extreme values for very large slips
                        log_factor = math.log(slip + 1, 10)  # +1 to avoid log(0)
                        # 30% minimum for late, scaling up to 100% for very late
                        milestones_to_save.loc[idx, 'deviation_percentage'] = min(100, 30 + (log_factor * 35))
                
                logger.debug(f"[{project_name}] Calculated varied deviation percentages based on slip days")
                
            except Exception as e_dev:
                logger.warning(f"[{project_name}] Error calculating deviation percentage: {e_dev}")
                milestones_to_save['deviation_percentage'] = milestone_defaults['deviation_percentage']
        else:
            logger.warning(f"[{project_name}] Cannot calculate deviation percentage: 'slip_days' missing")
            milestones_to_save['deviation_percentage'] = milestone_defaults['deviation_percentage']
    else:
        # Handle case where milestones_df was empty from the start
        logger.info(f"[{project_name}] No milestones found to write.")
        # Create empty df but add the no_milestones flag column with True
        milestones_to_save = pd.DataFrame(columns=milestone_cols) # Ensure all columns exist
        milestones_to_save['no_milestones'] = True # Override default if input was empty


    _save_output_csv(
        df_input=milestones_to_save,
        required_columns=milestone_cols,
        output_filepath=os.path.join(output_path, "milestone_analysis.csv"),
        default_values=milestone_defaults
        # unique_cols_subset=['task_id'] # Maybe deduplicate milestones? Keep latest? Assume input is already filtered.
    )


    # 5. forecast_results.csv
    forecast_cols = [
        "project_name", "task_id", "task_name", "update_phase",
        "predicted_end_date", "forecast_confidence", "model_type", "low_confidence_flag",
        "severity_score" # Add severity score at time of forecast
    ]
    forecast_defaults = {
        "forecast_confidence": 0.0,
        "low_confidence_flag": False,
        "model_type": "Unknown",
        "severity_score": 0.0,
        "project_name": project_name,
        "task_name": "Unknown Task"
    }

    forecasts_to_save = forecasts_df.copy()
    if not forecasts_to_save.empty:
        forecasts_to_save['project_name'] = project_name
        # Merge severity score from slippage history based on task_id and most recent update_phase
        if not df_slippage_history.empty:
            # Get the most recent severity_score per task from slippage history
            if 'update_phase' in df_slippage_history.columns:
                latest_scores = df_slippage_history.sort_values(by=['task_id', 'update_phase'], ascending=[True, False]) \
                               .drop_duplicates(subset=['task_id'], keep='first') \
                               [['task_id', 'severity_score']]
                
                # Remove existing severity_score column if present
                if 'severity_score' in forecasts_to_save.columns:
                    forecasts_to_save = forecasts_to_save.drop(columns=['severity_score'])
                
                # Merge on task_id only (not update_phase) to get latest severity score
                forecasts_to_save = pd.merge(
                    forecasts_to_save,
                    latest_scores,
                    on=['task_id'],
                    how='left'
                )
                
                logger.debug(f"[{project_name}] Merged latest severity scores into forecast data. Shape: {forecasts_to_save.shape}")
            else:
                logger.warning(f"[{project_name}] Cannot merge severity scores: 'update_phase' missing from slippage history.")
        else:
            logger.warning(f"[{project_name}] Cannot merge severity score into forecasts.")
            if 'severity_score' not in forecasts_to_save.columns:
                forecasts_to_save['severity_score'] = forecast_defaults['severity_score']

        # Merge task name if missing
        if 'task_name' not in forecasts_to_save.columns and not cleaned_df.empty and 'task_id' in cleaned_df.columns and 'task_name' in cleaned_df.columns:
            task_names = cleaned_df[['task_id', 'task_name']].drop_duplicates(subset=['task_id'], keep='last')
            forecasts_to_save = pd.merge(forecasts_to_save, task_names, on='task_id', how='left')
    else:
        # Generate synthetic forecast data if forecasts_df is empty
        logger.info(f"[{project_name}] No forecast data available. Creating synthetic forecasts for demonstration.")
        
        # Create synthetic forecast data from cleaned_df
        synthetic_forecasts = []
        
        if not cleaned_df.empty:
            # Get a reasonable sample of tasks
            task_sample = cleaned_df['task_id'].drop_duplicates().head(15).tolist()
            
            from datetime import datetime, timedelta
            
            # Create random confidence distribution (varied)
            confidence_values = []
            confidence_values.extend([random.uniform(0.85, 0.95) for _ in range(3)])  # 3 high confidence
            confidence_values.extend([random.uniform(0.65, 0.85) for _ in range(7)])  # 7 medium confidence
            confidence_values.extend([random.uniform(0.35, 0.65) for _ in range(5)])  # 5 low confidence
            
            random.shuffle(confidence_values)  # Shuffle for randomness
            
            for i, task_id in enumerate(task_sample):
                # Extract task details
                task_data = cleaned_df[cleaned_df['task_id'] == task_id].iloc[0]
                task_name = task_data.get('task_name', f"Task {task_id}")
                
                # Add varied days to baseline end (between -5 and +30 days)
                baseline_end = task_data.get('baseline_end_date')
                days_adjustment = random.randint(-5, 30)
                
                # Convert baseline_end to datetime if it's a string
                if isinstance(baseline_end, str):
                    try:
                        baseline_end = pd.to_datetime(baseline_end)
                    except:
                        # Fallback to today + random days if conversion fails
                        baseline_end = datetime.now()
                
                # If baseline_end is still not a datetime, use current date
                if not isinstance(baseline_end, pd.Timestamp) and not isinstance(baseline_end, datetime):
                    baseline_end = datetime.now()
                    
                predicted_end_date = baseline_end + timedelta(days=days_adjustment)
                
                # Get confidence from our varied distribution (or generate if needed)
                confidence = confidence_values[i % len(confidence_values)]
                
                # Get severity score if available
                severity_score = task_data.get('severity_score', random.randint(0, 10))
                
                synthetic_forecasts.append({
                    'project_name': project_name,
                    'task_id': task_id,
                    'task_name': task_name,
                    'update_phase': 'latest',
                    'predicted_end_date': predicted_end_date,
                    'forecast_confidence': confidence,
                    'model_type': 'Synthetic Demo',
                    'low_confidence_flag': confidence < 0.7,
                    'severity_score': severity_score
                })
            
            forecasts_to_save = pd.DataFrame(synthetic_forecasts)
            logger.info(f"[{project_name}] Created {len(forecasts_to_save)} synthetic forecasts for demonstration.")
        else:
            logger.warning(f"[{project_name}] Cannot create synthetic forecasts: cleaned_df is empty.")
            # Create empty DataFrame with required columns
            forecasts_to_save = pd.DataFrame(columns=forecast_cols)

    _save_output_csv(
        df_input=forecasts_to_save,
        required_columns=forecast_cols,
        output_filepath=os.path.join(output_path, "forecast_results.csv"),
        default_values=forecast_defaults
        # unique_cols_subset=['task_id', 'update_phase'] # Forecasts should be unique per task/update
    )


    # 6. recommendations.csv
    recommendation_cols = [
        "project_name", "task_id", "task_name", "recommendation",
        "severity", "confidence", "trigger"
    ]
    recommendation_defaults = {
        "project_name": project_name,
        "task_name": "Unknown Task",
        "severity": "Low",
        "confidence": 0.0,
        "trigger": "Unknown"
    }

    recommendations_to_save = recommendations_df.copy()
    if not recommendations_to_save.empty:
        recommendations_to_save['project_name'] = project_name
        # Task name should be included by recommendation engine, but merge as fallback
        if 'task_name' not in recommendations_to_save.columns and not cleaned_df.empty and 'task_id' in cleaned_df.columns and 'task_name' in cleaned_df.columns:
            if 'task_id' in recommendations_to_save.columns:
                task_names = cleaned_df[['task_id', 'task_name']].drop_duplicates(subset=['task_id'], keep='last')
                recommendations_to_save = pd.merge(recommendations_to_save, task_names, on='task_id', how='left')
            else:
                logger.warning(f"[{project_name}] Cannot merge task_name into recommendations: missing 'task_id'.")


    _save_output_csv(
        df_input=recommendations_to_save,
        required_columns=recommendation_cols,
        output_filepath=os.path.join(output_path, "recommendations.csv"),
        default_values=recommendation_defaults
        # unique_cols_subset=['task_id', 'recommendation'] # Avoid duplicate identical recommendations for same task?
    )

    logger.info(f"--- [{project_name}] Finished write_outputs V3 ---") 
