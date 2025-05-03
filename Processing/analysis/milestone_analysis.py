#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging

def analyse_milestones(df: pd.DataFrame) -> pd.DataFrame:
    """Analyses milestone tasks within the project data.

    Identifies tasks marked as milestones (typically where 'is_milestone' is True 
    or equivalent) in the input DataFrame. It then extracts key information 
    such as finish dates and slippage for these milestones.

    The function performs the following steps:
    1. Validates the presence of the 'is_milestone' column.
    2. Robustly converts the 'is_milestone' column to boolean, handling various 
       truthy/falsy representations (e.g., 'Yes', 'True', 1, 'No', 0, None).
    3. Filters the DataFrame to retain only rows identified as milestones.
    4. Selects a predefined set of relevant columns for the output.
    5. Handles cases where expected columns (like 'slip_days') might be missing.
    6. Returns a DataFrame containing only milestone data with the selected columns,
       or an empty DataFrame if no milestones are found or critical errors occur.

    Args:
        df (pd.DataFrame): The input DataFrame, expected to be standardised and 
                           cleaned, containing project task data. Requires columns 
                           like 'task_id', 'task_name', 'is_milestone', 
                           'actual_finish', 'baseline_end_date'. The presence of 
                           'slip_days' is beneficial but optional.

    Returns:
        pd.DataFrame: A DataFrame containing only the milestone tasks and selected 
                      columns (task_id, task_name, actual_finish, baseline_end_date, 
                      slip_days), or an empty DataFrame if no milestones are found 
                      or required columns are missing.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Starting milestone analysis...")

    # Define the expected output columns for consistency, even in empty returns.
    expected_output_columns = ['task_id', 'task_name', 'actual_finish', 'baseline_end_date', 'slip_days']

    # --- Input Validation ---
    # Check if the essential 'is_milestone' column exists for filtering.
    if 'is_milestone' not in df.columns:
        logger.warning("'is_milestone' column not found in DataFrame. Cannot perform milestone analysis.")
        # Return an empty DataFrame with expected columns for consistency downstream.
        return pd.DataFrame(columns=expected_output_columns)

    # --- Milestone Filtering --- 
    # Filter the DataFrame to include only rows where 'is_milestone' indicates True.
    # This requires robust conversion as the input might contain strings ('True', 'Yes', '1')
    # or numbers (1) instead of Python booleans.
    milestones_df = pd.DataFrame() # Initialise empty
    try:
        # Define sets of common truthy and falsy string representations.
        true_values = {True, 1, '1', 't', 'true', 'y', 'yes'}
        # Convert the column to string, lowercase, handle NAs, and check against true_values.
        # pd.isna(x) is used to map actual None/NaN/NaT values to False.
        is_milestone_bool_series = df['is_milestone'].apply(
            lambda x: False if pd.isna(x) else str(x).strip().lower() in true_values
        )
        # Filter the original DataFrame based on the boolean series.
        milestones_df = df[is_milestone_bool_series].copy()
        logger.debug(f"Filtered {len(milestones_df)} potential milestone tasks based on 'is_milestone' column.")
    except Exception as e:
        logger.error(f"Error during filtering or boolean conversion for milestones based on 'is_milestone' column: {e}", exc_info=True)
        return pd.DataFrame(columns=expected_output_columns) # Return empty on error
        
    # Check if any milestones were actually found after filtering.
    if milestones_df.empty:
        logger.info("No milestone tasks identified in the data after filtering.")
        return pd.DataFrame(columns=expected_output_columns)

    # --- Output Column Selection --- 
    # Define the core columns required for the milestone report.
    output_cols_base = ['task_id', 'task_name', 'actual_finish', 'baseline_end_date']
    # Add additional analysis columns (may not exist yet but will be populated later)
    additional_cols = ['slip_days', 'severity_score', 'update_phase']
    # Include columns that already exist in the data, others will be added later
    output_cols_final = output_cols_base + [col for col in additional_cols if col in milestones_df.columns]
    
    # Log missing columns that will need to be handled in output_writer
    missing_analytics = [col for col in additional_cols if col not in milestones_df.columns]
    if missing_analytics:
        logger.warning(f"Some analytics columns missing in milestone data: {missing_analytics}. They will be populated during output processing.")
        
    # Verify all selected columns actually exist in the filtered DataFrame before selection.
    # This acts as a safeguard against unexpected missing columns.
    missing_cols = [col for col in output_cols_final if col not in milestones_df.columns]
    if missing_cols:
         logger.error(f"Milestone analysis output preparation failed: Missing expected columns in filtered data: {missing_cols}. Returning empty DataFrame.")
         # Return empty frame with *intended* columns if selection fails.
         return pd.DataFrame(columns=expected_output_columns)
         
    # Select the final columns to create the result DataFrame.
    milestones_result_df = milestones_df[output_cols_final].copy()

    logger.info(f"Milestone analysis complete. Found {len(milestones_result_df)} milestones.")
    return milestones_result_df
