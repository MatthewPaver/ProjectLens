import pandas as pd
import ruptures as rpt
import numpy as np
import logging

def detect_change_points(df: pd.DataFrame, project_name: str) -> pd.DataFrame:
    """
    Detects change points in task completion dates using Pelt algorithm.

    Args:
        df: DataFrame with task data (expects 'task_id', 'update_phase', 'actual_finish').
        project_name: Name of the project for context.

    Returns:
        DataFrame with identified change points.
    """
    logging.info(f"[{project_name}] Running change point detection...")
    results = []

    # Input validation and logging
    if df.empty:
        logging.warning(f"[{project_name}] Change point detection: Input DataFrame is empty.")
        return pd.DataFrame()
        
    logging.info(f"[{project_name}] Change point detection: Input shape={df.shape}, Columns={df.columns.tolist()}")

    # Ensure required columns exist using standardised names
    required_cols = ['task_id', 'update_phase', 'actual_finish', 'task_name'] # Added task_name
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"[{project_name}] Change point detection: Missing required columns: {missing_cols}")
        return pd.DataFrame()

    # Use standardised column name
    date_col = 'actual_finish'

    # Convert date column to datetime, coercing errors
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    logging.info(f"[{project_name}] Change point detection: Date column converted (errors coerced).")

    # Group by task_id
    grouped = df.groupby('task_id')
    logging.info(f"[{project_name}] Change point detection: Processing {len(grouped)} task groups.")

    for task_id, group in grouped:
        # Sort by update phase
        group = group.sort_values(by='update_phase')
        # Use standardised name and drop NaT values
        dates = group[date_col].dropna()
        task_name = group['task_name'].iloc[0] if not group.empty else 'Unknown Task'

        if len(dates) < 3:
            logging.debug(f"[{project_name}] Skipping task {task_id} ({task_name}): Insufficient data points ({len(dates)} < 3) for change point detection.")
            continue
        
        logging.debug(f"[{project_name}] Processing task {task_id} ({task_name}) with {len(dates)} data points for change point detection.")

        # Convert to UNIX timestamps for numerical analysis
        signal = dates.astype(np.int64) // 10**9
        signal = signal.values.reshape(-1, 1)

        try:
            # Apply Pelt algorithm
            algo = rpt.Pelt(model="rbf").fit(signal)
            # Result indices are *end* points of segments, so shift by -1 for change point
            change_points_indices = algo.predict(pen=2) # Lowered penalty further for more sensitivity (was 3)
            logging.debug(f"[{project_name}] Task {task_id} ({task_name}): Found {len(change_points_indices)-1} actual change points (indices: {change_points_indices}).")

            # Extract change point dates and details
            for idx in change_points_indices[:-1]:
                if idx > 0 and idx <= len(dates):
                    change_date = dates.iloc[idx-1]
                    update_phase = group['update_phase'].iloc[idx-1]
                    change_score = 1.0 # Placeholder score
                    
                    results.append({
                        "project_name": project_name,
                        "task_id": task_id,
                        "task_name": task_name,
                        "update_phase": update_phase,
                        "change_date": change_date,
                        "change_score": change_score,
                        "method": "Pelt"
                    })

        except Exception as e:
            logging.error(f"[{project_name}] Error detecting change points for task {task_id} ({task_name}): {e}")
            continue

    if not results:
        logging.warning(f"[{project_name}] Change point detection: No change points identified.")
        return pd.DataFrame(columns=["project_name", "task_id", "task_name", "update_phase", "change_date", "change_score", "method"])

    result_df = pd.DataFrame(results)
    logging.info(f"[{project_name}] Change point detection completed. Result shape={result_df.shape}")
    return result_df
