import pandas as pd

def analyse_milestones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and analyses milestone tasks for drift and volatility.
    Calculates slippage, forecast alignment, and status classification.
    """

    # Filter for milestones
    if "is_milestone" not in df.columns:
        return pd.DataFrame()  # No milestone tagging available

    milestones = df[df["is_milestone"] == True].copy()
    if milestones.empty:
        return pd.DataFrame()

    # Calculate slippage if possible
    # Check if the necessary standardised columns exist
    if "baseline_end_date" in milestones.columns and "actual_finish" in milestones.columns:
        # Convert columns to datetime, coercing errors
        milestones["actual_finish"] = pd.to_datetime(milestones["actual_finish"], errors='coerce')
        milestones["baseline_end_date"] = pd.to_datetime(milestones["baseline_end_date"], errors='coerce')

        # Calculate slip_days only where both dates are valid
        valid_dates = milestones["actual_finish"].notna() & milestones["baseline_end_date"].notna()
        milestones.loc[valid_dates, "slip_days"] = (milestones.loc[valid_dates, "actual_finish"] - milestones.loc[valid_dates, "baseline_end_date"]).dt.days
        milestones["slip_days"].fillna(0, inplace=True) # Fill remaining NaNs (due to NaT dates) with 0
    else:
        milestones["slip_days"] = 0 # Assign 0 if columns are missing

    # Calculate deviation_percent
    milestones['deviation_percent'] = 0.0 # Initialize column
    if "baseline_start_date" in milestones.columns and "baseline_end_date" in milestones.columns:
        # Convert baseline start to datetime
        milestones["baseline_start_date"] = pd.to_datetime(milestones["baseline_start_date"], errors='coerce')
        # Calculate baseline duration in days where possible
        valid_baseline_dates = milestones["baseline_start_date"].notna() & milestones["baseline_end_date"].notna()
        milestones.loc[valid_baseline_dates, 'baseline_duration_days'] = \
            (milestones.loc[valid_baseline_dates, "baseline_end_date"] - milestones.loc[valid_baseline_dates, "baseline_start_date"]).dt.days
        
        # Calculate deviation percentage where baseline duration is not zero
        valid_deviation_calc = valid_baseline_dates & milestones['baseline_duration_days'].notna() & (milestones['baseline_duration_days'] != 0)
        milestones.loc[valid_deviation_calc, 'deviation_percent'] = \
            (milestones.loc[valid_deviation_calc, 'slip_days'] / milestones.loc[valid_deviation_calc, 'baseline_duration_days']) * 100
        
        # Clean up temporary column
        milestones.drop(columns=['baseline_duration_days'], inplace=True, errors='ignore')
    else:
        # If baseline dates are missing, deviation is undefined, keep as 0
        pass 

    # Stability classification
    milestones["volatility_flag"] = milestones["slip_days"].apply(
        lambda x: "stable" if abs(x) < 3 else ("volatile" if abs(x) > 10 else "moderate")
    )

    milestones["status"] = milestones.apply(
        lambda row: "delayed" if row["slip_days"] > 0 else "early" if row["slip_days"] < 0 else "on_time",
        axis=1
    )

    # Select and return only available standardised columns
    output_columns = [
        "task_id", "task_name",
        "baseline_end_date", "actual_finish", # Use standardised names
        "slip_days", "deviation_percent", "status", "volatility_flag" # Keep calculated/status cols
    ]
    # Ensure only columns that actually exist in the DataFrame are selected
    existing_output_columns = [col for col in output_columns if col in milestones.columns]

    return milestones[existing_output_columns]