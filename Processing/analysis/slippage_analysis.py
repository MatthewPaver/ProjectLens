import pandas as pd
import logging
import numpy as np

# Define thresholds for classifying slippage severity.
# These bins categorise the number of slippage days into severity levels.
SLIPPAGE_BINS = [-float('inf'), 0, 7, 14, 30, float('inf')]  # Bins: Early/On Time, Minor, Moderate, Major, Severe
SLIPPAGE_LABELS = [0, 1, 2, 3, 4]  # Corresponding numerical labels for severity bins.

def calculate_slippage(df: pd.DataFrame, project_name: str) -> pd.DataFrame:
    """Calculates the difference in days between baseline and actual finish dates.

    Args:
        df: DataFrame containing project task data. Requires 'baseline_end_date' 
            and 'actual_finish' columns after standardisation.
        project_name: Name of the project for logging context.
            
    Returns:
        DataFrame with an added 'slip_days' column, or the original DataFrame 
        if required columns are missing or conversion fails.
    """
    logger = logging.getLogger(__name__)
    required_cols = ['baseline_end_date', 'actual_finish']
    
    # Check if required columns exist
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"[{project_name}] Slippage calculation requires columns: {required_cols}. Missing: {missing}. Cannot calculate 'slip_days'.")
        # Return df without slip_days, allowing downstream processes that might not need it
        return df

    try:
        # Add NaN logging before conversion/calculation
        baseline_nan_count = df['baseline_end_date'].isna().sum()
        actual_nan_count = df['actual_finish'].isna().sum()
        if baseline_nan_count > 0 or actual_nan_count > 0:
            logger.debug(f"[{project_name}] NaN counts BEFORE conversion: baseline_end_date={baseline_nan_count}, actual_finish={actual_nan_count}")
        
        # Use the standardised column name 'baseline_end_date'
        baseline_finish_dt = pd.to_datetime(df['baseline_end_date'], errors='coerce') 
        actual_finish_dt = pd.to_datetime(df['actual_finish'], errors='coerce')

        # Calculate the difference in days.
        # Positive values indicate a finish date later than the baseline (slippage).
        # Negative values indicate finishing earlier than the baseline.
        # The .dt accessor provides convenient datetime operations.
        df['slip_days'] = (actual_finish_dt - baseline_finish_dt).dt.days
        
        # Log information about resulting NaNs after calculation.
        slip_nan_count = df['slip_days'].isna().sum()
        if slip_nan_count > 0:
            logger.warning(f"[{project_name}] Calculated 'slip_days' contains {slip_nan_count} NaN values (out of {len(df)} rows), likely due to invalid or missing dates in input.")
            
        logger.info(f"[{project_name}] Successfully calculated 'slip_days'.")

    except Exception as e:
        logger.error(f"[{project_name}] Error calculating slip_days: {e}", exc_info=True)
        # Add a NaN column in case of error so the column exists downstream
        df['slip_days'] = np.nan 
        logger.warning(f"[{project_name}] Added 'slip_days' column with NaN due to calculation error.")

    return df # Always return the DataFrame, potentially with NaNs in slip_days

def classify_change_type(change_in_slip: float) -> str:
    """Classifies the type of change based on the difference in slip days.
       Handles NaN for initial entries."""
    if pd.isna(change_in_slip):
        # Distinguish the first entry (NaN from diff) from an actual zero change.
        return "Initial" # Or "Baseline", "Unknown" - "Initial" seems clear.
    elif change_in_slip > 0:
        return "Slipped Further"
    elif change_in_slip < 0:
        return "Recovered"
    else: # change_in_slip == 0
        return "No Change"

def run_slippage_analysis(df: pd.DataFrame, project_name: str) -> pd.DataFrame | None:
    """
    Performs slippage analysis on the cleaned and standardised project data.
    Requires data sorted by task ID and update phase.
    
    Steps:
    1. Calculate raw slip days (actual_finish - baseline_end_date).
    2. Calculate historical severity score (capped at 10) based on slip_days and is_critical.
    3. Sort data by task_id and update_phase (essential for change calculation).
    4. Calculate the change in slip_days between consecutive updates for each task.
    5. Classify the type of change (Slipped Further, Recovered, No Change, Initial).
    
    Args:
        df: Cleaned and standardised DataFrame. Must contain at least
            'task_id', 'update_phase', 'baseline_end_date', 'actual_finish', 'is_critical'.
        project_name: Name of the project for logging.

    Returns:
        DataFrame with added 'slip_days', 'severity_score', and 'change_type' columns,
        or None if critical errors occur (like missing essential columns).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[{project_name}] Running slippage analysis...")
    
    required_cols = ['task_id', 'update_phase', 'baseline_end_date', 'actual_finish', 'is_critical']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"[{project_name}] Slippage analysis failed: Missing required columns: {missing}")
        return None # Return None if essential columns are missing

    # Operate on a copy to avoid modifying the original DataFrame passed in
    df_analysis = df.copy()

    # Calculate Slip Days
    df_analysis = calculate_slippage(df_analysis, project_name) # calculate_slippage modifies df in place

    # Check if slip_days calculation failed
    if 'slip_days' not in df_analysis.columns:
         logger.error(f"[{project_name}] 'slip_days' column was not added during calculation. Cannot proceed.")
         # Return current df, subsequent steps will fail gracefully or handle missing cols
         return df_analysis

    # Calculate Severity Score (Historical)
    logger.debug(f"[{project_name}] Calculating historical severity score...")
    try:
        # Ensure columns are numeric/boolean, fill NaNs appropriately before calculation
        # slip_days might have NaNs if dates were invalid - treat as 0 slip for severity calc? Or keep NaN? Let's treat as 0.
        slip_days_filled = df_analysis['slip_days'].fillna(0)
        # is_critical should be boolean from cleaning, fill potential NaNs with False
        is_critical_filled = df_analysis['is_critical'].fillna(False).astype(int) # Convert bool/NaN to 1/0

        # Calculate raw score
        raw_score = slip_days_filled * 1.0 + is_critical_filled * 5.0

        # Cap the score between 0 and 10
        df_analysis['severity_score'] = np.clip(raw_score, 0, 10)

        # Handle cases where original slip_days was NaN - severity should also be NaN?
        # Let's make severity NaN if slip_days was originally NaN
        df_analysis.loc[df_analysis['slip_days'].isna(), 'severity_score'] = np.nan

        logger.info(f"[{project_name}] Successfully calculated 'severity_score' (capped at 10).")
    except KeyError as e:
         logger.error(f"[{project_name}] Missing column required for severity score calculation: {e}. Skipping score calculation.")
         df_analysis['severity_score'] = np.nan # Add fallback NaN values when calculation fails
    except Exception as e_sev:
        logger.error(f"[{project_name}] Error calculating severity score: {e_sev}", exc_info=True)
        df_analysis['severity_score'] = np.nan # Add fallback NaN values for error handling

    # Change Analysis
    # Calculate the difference in slip_days between consecutive updates *within each task group*.
    # Requires sorting first.
    logger.debug(f"[{project_name}] Calculating change in slip days (requires sorting)...")
    try:
        # Sort FIRST to ensure correct diff calculation within groups
        df_sorted = df_analysis.sort_values(by=['task_id', 'update_phase'])

        # Group by task_id and calculate the difference between consecutive slip_days
        # .diff() calculates the difference between the current and previous element.
        # This will result in NaN for the first entry of each task_id group.
        df_sorted['change_in_slip'] = df_sorted.groupby('task_id')['slip_days'].diff()

        # Classify the change type based on the calculated difference (handles NaN as 'Initial')
        df_sorted['change_type'] = df_sorted['change_in_slip'].apply(classify_change_type)

        logger.info(f"[{project_name}] Successfully calculated 'change_in_slip' and 'change_type' (with 'Initial' state).")

    except Exception as e:
        logger.error(f"[{project_name}] Error calculating change in slip: {e}", exc_info=True)
        # Add fallback values if change analysis fails to ensure downstream compatibility
        df_sorted = df_analysis # Fallback to df before change calc attempt
        if 'change_type' not in df_sorted.columns:
             df_sorted['change_type'] = "Error"
        if 'change_in_slip' not in df_sorted.columns:
             df_sorted['change_in_slip'] = np.nan
        logger.warning(f"[{project_name}] Added fallback 'change_type'/'change_in_slip' values due to error.")

    # Check final DataFrame before returning
    logger.debug(f"[{project_name}] Final columns in slippage analysis result: {df_sorted.columns.tolist()}")
    if 'baseline_end_date' in df_sorted.columns and 'actual_finish' in df_sorted.columns:
        logger.debug(f"[{project_name}] Data types before return: baseline_end_date={df_sorted['baseline_end_date'].dtype}, actual_finish={df_sorted['actual_finish'].dtype}")
        logger.debug(f"[{project_name}] Null counts before return: baseline_end_date={df_sorted['baseline_end_date'].isnull().sum()}, actual_finish={df_sorted['actual_finish'].isnull().sum()}")

    logger.info(f"[{project_name}] Slippage analysis complete. Final shape: {df_sorted.shape}")
    return df_sorted
