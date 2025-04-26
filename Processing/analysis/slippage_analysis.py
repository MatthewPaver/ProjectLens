import pandas as pd
import logging
import numpy as np

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configuration likely done elsewhere

def run_slippage_analysis(df: pd.DataFrame, project_name: str) -> pd.DataFrame:
    """
    Calculates task slippage based on baseline and actual end dates.

    Args:
        df: DataFrame containing task data (expects 'task_id', 'baseline_end_date', 'actual_finish').
        project_name: Name of the project for context in logs/output.

    Returns:
        DataFrame with slippage information per task.
    """
    logging.info(f"[{project_name}] Running slippage analysis...")
    
    # Input validation and logging
    if df.empty:
        logging.warning(f"[{project_name}] Slippage analysis: Input DataFrame is empty.")
        return pd.DataFrame()
        
    logging.info(f"[{project_name}] Slippage analysis: Input shape={df.shape}, Columns={df.columns.tolist()}")

    # Ensure required columns exist using standardised names
    required_cols = ['task_id', 'baseline_end_date', 'actual_finish', 'update_phase', 'task_name'] # Added update_phase and task_name for output context
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"[{project_name}] Slippage analysis: Missing required columns: {missing_cols}")
        return pd.DataFrame()

    # Use standardised column names directly
    baseline_col = 'baseline_end_date'
    actual_col = 'actual_finish'
    
    # Convert date columns to datetime, coercing errors to NaT
    df[baseline_col] = pd.to_datetime(df[baseline_col], errors='coerce')
    df[actual_col] = pd.to_datetime(df[actual_col], errors='coerce')
    logging.info(f"[{project_name}] Slippage analysis: Date columns converted (errors coerced to NaT).")

    # Log count of rows with NaT dates before dropping
    nat_baseline_count = df[baseline_col].isna().sum()
    nat_actual_count = df[actual_col].isna().sum()
    nat_either_count = df[[baseline_col, actual_col]].isna().any(axis=1).sum()
    if nat_either_count > 0:
        logging.warning(f"[{project_name}] Slippage analysis: Found NaT dates before dropping - Baseline: {nat_baseline_count}, Actual: {nat_actual_count}, Rows with either: {nat_either_count}")

    # Drop rows where either essential date is missing *after* conversion attempt
    original_rows = len(df)
    df_filtered = df.dropna(subset=[baseline_col, actual_col])
    rows_after_dropna = len(df_filtered)
    logging.info(f"[{project_name}] Slippage analysis: Dropped {original_rows - rows_after_dropna} rows with missing baseline or actual dates.")

    if df_filtered.empty:
        logging.warning(f"[{project_name}] Slippage analysis: No rows remaining after dropping missing dates.")
        return pd.DataFrame()

    # Calculate slippage in days
    # Ensure calculations are done only on valid date pairs
    df_filtered['slip_days'] = (df_filtered[actual_col] - df_filtered[baseline_col]).dt.days
    logging.info(f"[{project_name}] Slippage analysis: Calculated slip_days.")

    # --- Severity Score Calculation (Example) ---
    # More sophisticated scoring could be added here based on days slipped, task importance etc.
    conditions = [
        (df_filtered['slip_days'] <= 0),
        (df_filtered['slip_days'] > 0) & (df_filtered['slip_days'] <= 7),
        (df_filtered['slip_days'] > 7) & (df_filtered['slip_days'] <= 30),
        (df_filtered['slip_days'] > 30)
    ]
    scores = [0, 1, 2, 3] # Example: 0=On time/Early, 1=Minor, 2=Moderate, 3=Major
    df_filtered['severity_score'] = np.select(conditions, scores, default=0)
    logging.info(f"[{project_name}] Slippage analysis: Calculated severity_score.")
    # --- End Severity Score ---

    # Determine change type (simplified)
    df_filtered['change_type'] = np.where(df_filtered['slip_days'] > 0, 'Delay',
                                         np.where(df_filtered['slip_days'] < 0, 'Early Finish', 'On Time'))
    logging.info(f"[{project_name}] Slippage analysis: Determined change_type.")


    # Prepare final output dataframe
    # Select and rename columns for clarity - keeping original standardised names where possible
    slippage_data = df_filtered[[
        'task_id', 'update_phase', 'task_name', # Added for context
        baseline_col, actual_col, 'slip_days', 'severity_score', 'change_type'
    ]].rename(columns={
        baseline_col: 'baseline_end', # Keep consistent naming for output
        actual_col: 'end_date'        # Keep consistent naming for output
    })

    # Add project name column
    slippage_data.insert(0, 'project_name', project_name) 
    
    logging.info(f"[{project_name}] Slippage analysis completed. Result shape={slippage_data.shape}")
    
    return slippage_data
