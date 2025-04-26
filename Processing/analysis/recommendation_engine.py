import pandas as pd
import numpy as np
from collections import defaultdict
import logging
import json

# Define recommendation structure keys for clarity
REC_DESC = "description"
REC_CONF = "confidence_score"
REC_ACTION = "recommended_action"
REC_TASK_ID = "task_id" # Added to associate task_id directly

def generate_recommendations(df, slippages, forecasts, changepoints, milestones, failed_forecast_tasks=None):
    """
    Generate a list of data quality and performance improvement suggestions.
    Each recommendation is now a dictionary.

    Args:
        df: The cleaned main DataFrame.
        slippages: DataFrame from slippage analysis.
        forecasts: DataFrame from forecast engine.
        changepoints: DataFrame from change point detection.
        milestones: DataFrame from milestone analysis.
        failed_forecast_tasks: Optional list of task IDs that failed forecasting due to insufficient data.

    Returns:
        A dictionary with category -> list of recommendation dictionaries,
        each dictionary containing description, confidence_score, recommended_action, and task_id (optional).
    """
    # --->>> Get Logger and Log Inputs <<<---
    logger = logging.getLogger(__name__)
    logger.info("Starting recommendation generation...")
    # Log shapes of input DataFrames (if they are DataFrames)
    logger.debug(f"Input df shape: {df.shape if isinstance(df, pd.DataFrame) else type(df)}")
    logger.debug(f"Input slippages shape: {slippages.shape if isinstance(slippages, pd.DataFrame) else type(slippages)}")
    logger.debug(f"Input forecasts shape: {forecasts.shape if isinstance(forecasts, pd.DataFrame) else type(forecasts)}")
    logger.debug(f"Input changepoints shape: {changepoints.shape if isinstance(changepoints, pd.DataFrame) else type(changepoints)}")
    logger.debug(f"Input milestones shape: {milestones.shape if isinstance(milestones, pd.DataFrame) else type(milestones)}")
    logger.debug(f"Input failed_forecast_tasks count: {len(failed_forecast_tasks) if failed_forecast_tasks is not None else 'None'}")
    # --->>> END Log Inputs <<<---

    recs = defaultdict(list)
    # Initialize the list if None is passed
    if failed_forecast_tasks is None:
        failed_forecast_tasks = []

    # Helper to create recommendation dictionary
    def create_rec(desc, conf, action, task_id="N/A"):
        return {REC_DESC: desc, REC_CONF: conf, REC_ACTION: action, REC_TASK_ID: task_id}

    # 1. 🧱 Missing Field Check (Using Standardised Names)
    required = ["actual_start", "actual_finish", "baseline_end_date", "percent_complete", "duration"]
    if not isinstance(df, pd.DataFrame):
        recs["data_input_error"].append(create_rec(
            "Main DataFrame is not valid.",
            1.0,
            "ACTION: Investigate data loading process. Ensure input files are correctly formatted and accessible."
        ))
        # Return early if the main dataframe is unusable
        return dict(recs)

    for col in required:
        # Check if column exists and is not entirely null
        # --->>> Log Missing Field Check <<<---
        col_exists = col in df.columns
        col_is_null = df[col].isnull().all() if col_exists else True
        if not col_exists or col_is_null:
            logger.debug(f"Check: Required field '{col}' missing or all null.")
            recs["missing_fields"].append(create_rec(
                f"Missing or unusable standard field: {col}",
                1.0, # High confidence: the field is definitively missing/empty
                f"ACTION: Populate the '{col}' column in the input data source."
            ))
        else:
            logger.debug(f"Check: Required field '{col}' present and has data.")
        # --->>> END Log Missing Field Check <<<---

    # 2. 📈 Few Updates Warning - REMOVED (as per previous change)

    # 3. 📉 Low Confidence Forecasting
    if isinstance(forecasts, pd.DataFrame) and not forecasts.empty and "forecast_confidence" in forecasts.columns:
        low_conf_threshold = 0.75
        # Added low_confidence_flag check as well
        low_conf_tasks_df = forecasts[forecasts["low_confidence_flag"] == True]
        num_low_conf_tasks = len(low_conf_tasks_df)
        total_forecast_tasks = len(forecasts)
        logger.debug(f"Check: Found {num_low_conf_tasks} tasks marked with low_confidence_flag=True.")

        if num_low_conf_tasks > 0:
            # Confidence based on proportion, capped at 0.9 max (never fully certain without manual review)
            confidence = min(0.9, 0.5 + (num_low_conf_tasks / total_forecast_tasks) * 0.4)
            task_ids_low_conf = low_conf_tasks_df['task_id'].tolist()
            tasks_to_mention = 3
            example_tasks = ", ".join(task_ids_low_conf[:tasks_to_mention])
            if len(task_ids_low_conf) > tasks_to_mention:
                example_tasks += ", ..."

            recs["forecasting"].append(create_rec(
                f"{num_low_conf_tasks} tasks have low forecast confidence (e.g., {example_tasks}). This may indicate volatile history or data gaps.",
                confidence,
                f"ACTION: Review historical data for tasks like {example_tasks}. Check for volatility, outliers, or missing updates.",
                task_id="Multiple" # Indicate it applies to multiple tasks
            ))
    elif not isinstance(forecasts, pd.DataFrame) or forecasts.empty:
        logger.debug("Check: Forecasts DataFrame missing, empty, or lacks 'forecast_confidence' column.")
        recs["forecasting"].append(create_rec(
            "Forecasting data is missing or empty, cannot assess confidence.",
            1.0, # High confidence: the data is definitively missing
            "ACTION: Ensure the forecasting engine runs successfully and produces output."
        ))

    # 3b. Tasks that failed forecasting due to insufficient data
    if failed_forecast_tasks:
        max_tasks_to_list = 5
        num_failed = len(failed_forecast_tasks)
        if num_failed <= max_tasks_to_list:
            task_list_str = ", ".join(failed_forecast_tasks)
            desc = f"Forecasting skipped for tasks ({task_list_str}) due to insufficient historical data (< 3 points)."
        else:
            first_tasks_str = ", ".join(failed_forecast_tasks[:max_tasks_to_list])
            desc = f"{num_failed} tasks (including {first_tasks_str}, ...) skipped forecasting due to insufficient historical data (< 3 points)."

        recs["forecasting"].append(create_rec(
            desc,
            1.0, # High confidence: forecasting was definitively skipped
            f"ACTION: Provide more historical data points (at least 3) for affected tasks to enable forecasting.",
            task_id="Multiple" # Indicate it applies to multiple tasks
        ))
        logger.debug(f"Check: Incorporating messages for {num_failed} tasks that failed forecasting.")

    # 4. 🔍 Field Improvement Suggestions (Using Standardised/Expected Names)
    if isinstance(df, pd.DataFrame):
        # Check for 'predicted_end_date' in forecasts
        if not isinstance(forecasts, pd.DataFrame) or forecasts.empty or "predicted_end_date" not in forecasts.columns:
             logger.debug("Check: 'predicted_end_date' missing from valid forecasts DataFrame.")
             recs["field_suggestions"].append(create_rec(
                "Forecasting analysis did not generate 'predicted_end_date'.",
                1.0, # High confidence: the column is missing
                "ACTION: Ensure forecasting engine runs successfully and generates 'predicted_end_date'."
             ))
        else:
             logger.debug("Check: 'predicted_end_date' present in forecasts DataFrame.")

        # Check for optional standardised fields
        optional_fields = {"is_critical": "critical path analysis", "risk_flag": "risk-based filtering"}
        for field, purpose in optional_fields.items():
            if field not in df.columns:
                logger.debug(f"Check: Optional field '{field}' missing from main df.")
                recs["field_suggestions"].append(create_rec(
                    f"Optional field '{field}' is missing.",
                    0.8, # High confidence, but it's optional
                    f"ACTION: Consider adding '{field}' column to input data to enable {purpose}."
                ))

    # 5. 🚨 Severity Risk Cluster Logging (Using Standardised Name)
    # Merging severity score from slippages/milestones if available
    severity_tasks = pd.DataFrame()
    if isinstance(slippages, pd.DataFrame) and 'severity_score' in slippages.columns and not slippages.empty:
        severity_tasks = pd.concat([severity_tasks, slippages[['task_id', 'task_name', 'severity_score']].copy()])
    if isinstance(milestones, pd.DataFrame) and 'severity_score' in milestones.columns and not milestones.empty:
         severity_tasks = pd.concat([severity_tasks, milestones[['task_id', 'task_name', 'severity_score']].copy()])

    # Deduplicate based on task_id, keeping the highest severity score if duplicates exist
    if not severity_tasks.empty:
        severity_tasks['severity_score'] = pd.to_numeric(severity_tasks['severity_score'], errors='coerce')
        severity_tasks.dropna(subset=['severity_score'], inplace=True)
        severity_tasks = severity_tasks.sort_values('severity_score', ascending=False).drop_duplicates(subset=['task_id'], keep='first')

        logger.debug(f"Check: Processing {len(severity_tasks)} unique tasks with severity scores.")

        if not severity_tasks.empty:
            # Using quantiles might be better than fixed top N, but let's stick to top 5 for now
            top_risk = severity_tasks.head(5)
            logger.debug(f"Check: Identified top {len(top_risk)} tasks with highest severity scores.")
            for _, row in top_risk.iterrows():
                task_id = row.get('task_id', 'N/A')
                task_name = row.get('task_name', 'Unknown Task')
                severity = row.get('severity_score', 0.0)
                # Use severity score directly as confidence (capped at 1.0)
                confidence = min(1.0, severity if pd.notna(severity) else 0.0)
                recs["severity_clusters"].append(create_rec(
                     f"High severity task: {task_name} (Score: {severity:.2f})",
                     confidence,
                     f"ACTION: Prioritise investigation of high-severity task '{task_name}' (ID: {task_id}). Check root causes (e.g., slippage, milestone delays).",
                     task_id=task_id
                ))
        else:
             # Check if severity score column was expected but missing/empty in sources
             severity_expected = ('severity_score' in slippages.columns) or ('severity_score' in milestones.columns)
             if severity_expected:
                 logger.warning("Severity score column exists in sources but no valid scores found after processing.")
                 recs["severity_clusters"].append(create_rec(
                     "Severity scores are present but could not be processed (check data types or values).",
                     0.9, # Confident there's an issue if column exists but no data
                     "ACTION: Verify 'severity_score' calculation in slippage/milestone analysis and data types."
                 ))
             else:
                 logger.debug("Severity score column not found in slippage or milestone data.")
                 recs["severity_clusters"].append(create_rec(
                    "'severity_score' column missing from inputs (slippage/milestone analysis), cannot identify high severity tasks.",
                     1.0, # Confident column is missing
                     "ACTION: Ensure slippage and/or milestone analysis calculates and returns 'severity_score'."
                 ))

    # 6. ❗ Critical Path Slippage
    critical_slip_threshold_days = 7 # Define threshold for significant slip
    logger.debug("Check: Starting critical path slippage check...")
    if isinstance(df, pd.DataFrame) and 'is_critical' in df.columns and isinstance(slippages, pd.DataFrame) and 'slip_days' in slippages.columns:
        # Use latest record per task for critical flag
        latest_df = df.sort_values('update_phase', ascending=False).drop_duplicates(subset=['task_id'])
        
        logger.debug("Check: Merging critical flags with slippage data.")
        # Merge slippage data with critical flag from latest df records
        # Ensure only necessary columns are selected before merge
        critical_check_df = pd.merge(
            latest_df[['task_id', 'task_name', 'is_critical']].copy(),
            slippages[['task_id', 'slip_days']].copy().dropna(subset=['slip_days']), # Ensure slip_days is not NaN
            on='task_id',
            how='inner' # Only consider tasks present in both
        )

        # Convert is_critical robustly (handle potential strings 'True', 'False', '1', '0')
        critical_check_df['is_critical'] = critical_check_df['is_critical'].apply(
            lambda x: str(x).strip().lower() in ['true', '1', 'yes'] if pd.notna(x) else False
        )
        # Convert slip_days to numeric
        critical_check_df['slip_days'] = pd.to_numeric(critical_check_df['slip_days'], errors='coerce')
        critical_check_df.dropna(subset=['slip_days'], inplace=True) # Drop if conversion failed

        # Filter for critical tasks with significant slippage
        critical_slippers = critical_check_df[
            (critical_check_df['is_critical'] == True) &
            (critical_check_df['slip_days'] > critical_slip_threshold_days)
        ]
        logger.debug(f"Check: Found {len(critical_slippers)} critical tasks slipping > {critical_slip_threshold_days} days (after merge and filtering, before deduplication).")

        # Deduplicate based on task_id after filtering for critical slippage.
        # Keep the first occurrence (which will have the latest slip_days due to how merge/sort might implicitly work, or we can be more explicit if needed)
        unique_critical_slippers = critical_slippers.drop_duplicates(subset=['task_id'], keep='first')
        logger.debug(f"Check: Found {len(unique_critical_slippers)} unique critical tasks slipping > {critical_slip_threshold_days} days.")

        if not unique_critical_slippers.empty:
            for _, row in unique_critical_slippers.head(5).iterrows(): # Limit output
                task_id = row['task_id']
                task_name = row['task_name']
                slip_days = row['slip_days']
                recs["performance_alerts"].append(create_rec(
                    f"Critical task '{task_name}' (ID: {task_id}) has slipped by {slip_days:.0f} days.",
                    1.0, # High confidence: based on data flags and calculation
                    f"ACTION: Immediately investigate critical task '{task_name}' (ID: {task_id}). Assess impact and mitigation plan.",
                    task_id=task_id
                ))
            if len(unique_critical_slippers) > 5:
                recs["performance_alerts"].append(create_rec(
                    f"...and {len(unique_critical_slippers) - 5} more critical tasks with significant slippage.",
                    0.9, # Slightly lower confidence for the summary message
                    "ACTION: Review the full slippage report for all critical tasks.",
                    task_id="Multiple"
                ))
        else:
            logger.debug("Check: No critical tasks found with significant slippage.")
    else:
        missing_crit_info = []
        if not isinstance(df, pd.DataFrame): missing_crit_info.append("main df")
        elif 'is_critical' not in df.columns: missing_crit_info.append("'is_critical' column")
        if not isinstance(slippages, pd.DataFrame): missing_crit_info.append("slippages df")
        elif 'slip_days' not in slippages.columns: missing_crit_info.append("'slip_days' column")
        logger.warning(f"Skipping Critical Path Slippage check. Missing data: {', '.join(missing_crit_info)}")
        recs["performance_alerts"].append(create_rec(
             f"Could not perform critical path slippage check (Missing: {', '.join(missing_crit_info)}).",
             0.95, # Confident check couldn't run
             "ACTION: Ensure input data includes 'is_critical' and slippage analysis provides 'slip_days'."
        ))

    # 7. ⏳ Forecast vs Baseline Deviation
    forecast_deviation_threshold_days = 14 # Define threshold
    logger.debug("Check: Starting Forecast vs Baseline Deviation check...")
    # Required columns: task_id, baseline_end_date (from df), predicted_end_date (from forecasts)
    forecast_vs_baseline_possible = (
        isinstance(df, pd.DataFrame) and 'baseline_end_date' in df.columns and 'task_id' in df.columns and
        isinstance(forecasts, pd.DataFrame) and 'predicted_end_date' in forecasts.columns and 'task_id' in forecasts.columns and not forecasts.empty
    )

    if forecast_vs_baseline_possible:
        try:
            # Get the latest baseline date for each task from the main cleaned df
            latest_cleaned_records = df.sort_values('update_phase', ascending=False).drop_duplicates(subset=['task_id'])
            baseline_info = latest_cleaned_records[['task_id', 'task_name', 'baseline_end_date']].copy()
            # Convert baseline_end_date to datetime
            baseline_info['baseline_end_date'] = pd.to_datetime(baseline_info['baseline_end_date'], errors='coerce')

            # Prepare forecasts data
            forecast_info = forecasts[['task_id', 'predicted_end_date']].copy()
            forecast_info['predicted_end_date'] = pd.to_datetime(forecast_info['predicted_end_date'], errors='coerce')

            # Merge forecasts with baseline info
            forecast_check_df = pd.merge(
                forecast_info,
                baseline_info,
                on='task_id',
                how='inner' # Only compare tasks present in both with valid dates
            )
            # Drop rows where date conversion failed
            forecast_check_df.dropna(subset=['predicted_end_date', 'baseline_end_date'], inplace=True)
            logger.debug(f"Check: Merged {len(forecast_check_df)} tasks for forecast vs baseline comparison.")

            # Calculate deviation in days
            forecast_check_df['deviation_days'] = (forecast_check_df['predicted_end_date'] - forecast_check_df['baseline_end_date']).dt.days

            # Filter for significant deviations (absolute value)
            significant_deviation = forecast_check_df[forecast_check_df['deviation_days'].abs() > forecast_deviation_threshold_days]
            logger.debug(f"Check: Found {len(significant_deviation)} tasks with forecast deviation > {forecast_deviation_threshold_days} days.")

            if not significant_deviation.empty:
                for _, row in significant_deviation.head(5).iterrows(): # Limit output
                    task_id = row['task_id']
                    task_name = row['task_name']
                    deviation = row['deviation_days']
                    direction = "later" if deviation > 0 else "earlier"
                    recs["performance_alerts"].append(create_rec(
                        f"Forecast for '{task_name}' (ID: {task_id}) deviates significantly ({abs(deviation):.0f} days {direction}) from baseline.",
                        1.0, # High confidence: based on calculation
                        f"ACTION: Review task '{task_name}' (ID: {task_id}). Understand the deviation. Update baseline if necessary.",
                        task_id=task_id
                    ))
                if len(significant_deviation) > 5:
                     recs["performance_alerts"].append(create_rec(
                        f"...and {len(significant_deviation) - 5} more tasks with significant forecast vs baseline deviation.",
                        0.9, # Slightly lower confidence for summary
                        "ACTION: Review the full forecast report for deviations from baseline.",
                        task_id="Multiple"
                     ))
            else:
                logger.debug("Check: No tasks found with significant forecast vs baseline deviation.")

        except Exception as e:
            logger.error(f"Error during Forecast vs Baseline Deviation check: {e}", exc_info=True)
            recs["calculation_error"].append(create_rec(
                 "Error occurred during Forecast vs Baseline Deviation calculation.",
                 1.0, # Confident an error happened
                 f"ACTION: Check logs for details on the error: {e}"
            ))
    else:
        missing_fvb_info = []
        if not isinstance(df, pd.DataFrame): missing_fvb_info.append("main df")
        elif 'baseline_end_date' not in df.columns: missing_fvb_info.append("'baseline_end_date'")
        if not isinstance(forecasts, pd.DataFrame) or forecasts.empty: missing_fvb_info.append("forecasts df")
        elif 'predicted_end_date' not in forecasts.columns: missing_fvb_info.append("'predicted_end_date'")
        logger.warning(f"Skipping Forecast vs Baseline Deviation check. Missing data: {', '.join(missing_fvb_info)}")
        recs["performance_alerts"].append(create_rec(
            f"Could not perform Forecast vs Baseline Deviation check (Missing: {', '.join(missing_fvb_info)}).",
            0.95, # Confident check couldn't run
            "ACTION: Ensure input data and forecasts include required date columns ('baseline_end_date', 'predicted_end_date')."
        ))

    # --->>> Flatten Recommendations <<<---
    # Convert the dictionary of lists of dicts into a flat list of dicts
    # Add 'recommendation_type' based on the dictionary key
    flat_recs = []
    for category, rec_list in recs.items():
        for rec_dict in rec_list:
            rec_dict['recommendation_type'] = category
            flat_recs.append(rec_dict)

    logger.info(f"Generated {len(flat_recs)} recommendations across {len(recs)} categories.")
    # --->>> Log Output Shape <<<---
    logger.debug(f"Output recommendations list length: {len(flat_recs)}")
    # --->>> END Log Output <<<---

    return flat_recs # Return the flat list of dictionaries

# --- Example Usage Placeholder (for testing) ---
if __name__ == '__main__':
    # Create dummy data for testing
    # This part would need more elaborate dummy dataframes to test all paths
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info("Running recommendation engine example...")
    # Example: Create minimal dummy dataframes
    dummy_df = pd.DataFrame({
        'task_id': ['T1', 'T2', 'T3'],
        'task_name': ['Task 1', 'Task 2', 'Task 3'],
        'actual_start': pd.to_datetime(['2023-01-01', '2023-01-05', None]),
        'actual_finish': pd.to_datetime(['2023-02-01', None, None]),
        'baseline_end_date': pd.to_datetime(['2023-01-30', '2023-02-15', '2023-03-01']),
        'percent_complete': [100, 50, 0],
        'duration': [31, 41, 60],
        'is_critical': [True, False, True],
        'update_phase': [1, 1, 1] # Minimal update phase for checks
        # 'severity_score': [0.8, 0.3, 0.9] # Add if testing severity directly here
    })
    dummy_slippages = pd.DataFrame({
        'task_id': ['T1', 'T2'],
        'slip_days': [2, 0],
        'severity_score': [0.8, 0.3]
    })
    dummy_forecasts = pd.DataFrame({
        'task_id': ['T1', 'T2'],
        'predicted_end_date': pd.to_datetime(['2023-02-02', '2023-02-20']),
        'forecast_confidence': [0.9, 0.6], # T2 has low confidence
        'low_confidence_flag': [False, True]
    })
    dummy_failed = ['T3']

    recommendations = generate_recommendations(dummy_df, dummy_slippages, dummy_forecasts, None, None, dummy_failed)

    # Print results
    print("\n--- Generated Recommendations ---")
    # Convert to DataFrame for pretty printing
    if recommendations:
        recs_df = pd.DataFrame(recommendations)
        print(recs_df.to_string())
    else:
        print("No recommendations generated.")
