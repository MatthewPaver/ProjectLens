import pandas as pd
import numpy as np
from collections import defaultdict
import logging
import json
from typing import List, Dict, Any, Optional

# Define recommendation structure keys for clarity and consistency
REC_CATEGORY = "category"
REC_DESC = "description"
REC_CONF = "confidence_score"
REC_ACTION = "recommended_action"
REC_TASK_ID = "task_id" # Added to associate task_id directly

def generate_recommendations(
    df: pd.DataFrame,
    slippages: Optional[pd.DataFrame] = None,
    forecasts: Optional[pd.DataFrame] = None,
    changepoints: Optional[pd.DataFrame] = None,
    milestones: Optional[pd.DataFrame] = None,
    failed_forecast_tasks: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Generates actionable recommendations based on combined analysis results.

    This engine synthesises findings from various analysis modules (slippage, 
    forecasting, change points, milestones) to provide targeted recommendations 
    for project tasks. Recommendations are generated based on predefined rules and 
    thresholds applied to the analysis outputs.

    Args:
        df (pd.DataFrame): The main standardised project data DataFrame.
        slippages (Optional[pd.DataFrame]): DataFrame containing slippage analysis results.
        forecasts (Optional[pd.DataFrame]): DataFrame containing forecasting results.
        changepoints (Optional[pd.DataFrame]): DataFrame listing detected change points.
        milestones (Optional[pd.DataFrame]): DataFrame containing milestone analysis results.
        failed_forecast_tasks (Optional[List[str]]): List of task IDs where forecasting failed.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents 
                              a single recommendation. Each recommendation includes 
                              details like task ID, task name, recommendation text, 
                              severity, confidence, and the triggering criteria.
    """
    logger = logging.getLogger(__name__) 
    logger.info("Starting recommendation generation...")
    logger.debug(f"Input df shape: {df.shape if isinstance(df, pd.DataFrame) else type(df)}")
    logger.debug(f"Input slippages shape: {slippages.shape if isinstance(slippages, pd.DataFrame) else type(slippages)}")
    logger.debug(f"Input forecasts shape: {forecasts.shape if isinstance(forecasts, pd.DataFrame) else type(forecasts)}")
    logger.debug(f"Input changepoints shape: {changepoints.shape if isinstance(changepoints, pd.DataFrame) else type(changepoints)}")
    logger.debug(f"Input milestones shape: {milestones.shape if isinstance(milestones, pd.DataFrame) else type(milestones)}")
    logger.debug(f"Input failed_forecast_tasks count: {len(failed_forecast_tasks) if failed_forecast_tasks is not None else 'None'}")
    
    recommendations = []
    processed_tasks = set() # Keep track of tasks for which a recommendation was already generated
    
    # Ensure optional inputs are treated as empty DataFrames if None
    slippages = slippages if slippages is not None and not slippages.empty else pd.DataFrame()
    forecasts = forecasts if forecasts is not None and not forecasts.empty else pd.DataFrame()
    changepoints = changepoints if changepoints is not None and not changepoints.empty else pd.DataFrame()
    milestones = milestones if milestones is not None and not milestones.empty else pd.DataFrame()
    failed_forecast_tasks = failed_forecast_tasks if failed_forecast_tasks else []

    # --- Recommendation Rule 1: Significant Slippage --- 
    if not slippages.empty and 'slip_days' in slippages.columns:
        logger.debug("Checking for significant slippage...")
        significant_slip_threshold = 7  # Explicitly set to 7 days
        recent_slippage = slippages.sort_values(by=['task_id', 'update_phase']).groupby('task_id').last()
        high_slippage_tasks = recent_slippage[recent_slippage['slip_days'] > significant_slip_threshold]
        logger.debug(f"Found {len(high_slippage_tasks)} tasks with slip_days > {significant_slip_threshold}")
        for task_id, row in high_slippage_tasks.iterrows():
            if task_id not in processed_tasks:
                rec = {
                    "task_id": task_id,
                    "task_name": row.get('task_name', 'Unknown Task'),
                    "recommendation": f"Task shows significant slippage ({row['slip_days']:.0f} days). Investigate root causes (resource constraints, scope creep, dependencies).",
                    "severity": "High",
                    "confidence": 0.85,
                    "trigger": f"Slippage > {significant_slip_threshold} days"
                }
                recommendations.append(rec)
                processed_tasks.add(task_id)
                logger.info(f"Recommendation added for significant slippage: Task {task_id}")
        if len(high_slippage_tasks) == 0:
            logger.info(f"No tasks found with slip_days > {significant_slip_threshold} for Rule 1.")
    else:
        logger.info("Slippages DataFrame empty or missing 'slip_days' column for Rule 1.")

    # --- Recommendation Rule 2: Forecasted Delay with Low Confidence --- 
    if not forecasts.empty and 'predicted_end_date' in forecasts.columns and 'forecast_confidence' in forecasts.columns:
        logger.debug("Checking for forecasted delays with low confidence...")
        low_confidence_threshold = 0.7
        low_conf_forecasts = forecasts[forecasts['forecast_confidence'] < low_confidence_threshold]
        logger.debug(f"Found {len(low_conf_forecasts)} tasks with forecast_confidence < {low_confidence_threshold}")
        
        for task_id, row in low_conf_forecasts.iterrows():
             if task_id not in processed_tasks:
                rec = {
                    "task_id": task_id,
                    "task_name": row.get('task_name', 'Unknown Task'),
                    "recommendation": f"Forecast for task completion ({row['predicted_end_date'].strftime('%Y-%m-%d')}) has low confidence ({row['forecast_confidence']:.2f}). Review underlying data stability and consider alternative forecasting or closer monitoring.",
                    "severity": "Medium",
                    "confidence": row['forecast_confidence'],
                    "trigger": f"Forecast Confidence < {low_confidence_threshold}"
                }
                recommendations.append(rec)
                processed_tasks.add(task_id)
                logger.info(f"Recommendation added for low confidence forecast: Task {task_id}")
        if len(low_conf_forecasts) == 0:
            logger.info(f"No tasks found with forecast_confidence < {low_confidence_threshold} for Rule 2.")
    else:
        logger.info("Forecasts DataFrame empty or missing required columns for Rule 2.")

    # --- Recommendation Rule 3: Detected Negative Changepoint (Increased Slippage) --- 
    if not changepoints.empty and 'slip_days' in changepoints.columns:
        logger.debug("Checking for negative change points in slippage...")
        for task_id, row in changepoints.iterrows():
            if task_id not in processed_tasks:
                 rec = {
                    "task_id": task_id,
                    "task_name": row.get('task_name', 'Unknown Task'),
                    "recommendation": f"A significant change point in slippage trend was detected around update phase '{row['update_phase']}' (Slippage: {row['slip_days']:.0f} days). Investigate events or factors causing this shift.",
                    "severity": "Medium",
                    "confidence": 0.75,
                    "trigger": "Change Point Detected (PELT)"
                }
                 recommendations.append(rec)
                 processed_tasks.add(task_id)
                 logger.info(f"Recommendation added for detected change point: Task {task_id}")
        if len(changepoints) == 0:
            logger.info("No change points detected for Rule 3.")
    else:
        logger.info("Changepoints DataFrame empty or missing 'slip_days' column for Rule 3.")

    # --- Recommendation Rule 4: Milestone Slippage --- 
    if not milestones.empty and 'slip_days' in milestones.columns:
        logger.debug("Checking for milestone slippage...")
        slipping_milestones = milestones[milestones['slip_days'] > 0]
        logger.debug(f"Found {len(slipping_milestones)} milestones with slip_days > 0")
        
        for task_id, row in slipping_milestones.iterrows():
            if task_id not in processed_tasks:
                rec = {
                    "task_id": task_id,
                    "task_name": row.get('task_name', 'Unknown Milestone'),
                    "recommendation": f"Milestone task has slipped by {row['slip_days']:.0f} days. Prioritise actions to recover schedule or replan dependent tasks.",
                    "severity": "High",
                    "confidence": 0.90,
                    "trigger": "Milestone Slippage > 0 days"
                }
                recommendations.append(rec)
                processed_tasks.add(task_id)
                logger.info(f"Recommendation added for milestone slippage: Task {task_id}")
        if len(slipping_milestones) == 0:
            logger.info("No milestones found with slip_days > 0 for Rule 4.")
    else:
        logger.info("Milestones DataFrame empty or missing 'slip_days' column for Rule 4.")

    # --- Recommendation Rule 5: Forecasting Failure --- 
    # Enhanced logging for Rule 5 to debug issues
    if failed_forecast_tasks:
        logger.debug(f"Rule 5: Processing {len(failed_forecast_tasks)} failed forecast tasks")
        logger.debug(f"Rule 5: First 10 failed tasks: {failed_forecast_tasks[:10]}")
        
        # Check if we can get task names from the main DataFrame
        if not df.empty and 'task_id' in df.columns:
            logger.debug(f"Rule 5: Main DataFrame has {len(df)} rows with task_id column")
            if 'task_name' in df.columns:
                # Create a task name lookup dictionary
                task_name_map = df.set_index('task_id')['task_name'].to_dict()
                logger.debug(f"Rule 5: Created task name map with {len(task_name_map)} entries")
                
                # Count how many failed tasks are in the main DataFrame
                found_tasks = [task_id for task_id in failed_forecast_tasks if task_id in task_name_map]
                logger.debug(f"Rule 5: Found {len(found_tasks)}/{len(failed_forecast_tasks)} failed tasks in main DataFrame")
                
                # Process recommendations for failed forecast tasks
                recommendations_added = 0
                for task_id in failed_forecast_tasks:
                    if task_id not in processed_tasks:
                        task_name = task_name_map.get(task_id, 'Unknown Task')
                        rec = {
                            "task_id": task_id,
                            "task_name": task_name,
                            "recommendation": "Forecasting models failed to produce a reliable prediction for this task. Data might be too sparse, erratic, or contain non-numeric values. Manual review of task history is recommended.",
                            "severity": "Low",
                            "confidence": 0.95,
                            "trigger": "Forecast Engine Failed"
                        }
                        recommendations.append(rec)
                        processed_tasks.add(task_id)
                        recommendations_added += 1
                        logger.info(f"Recommendation added for forecast failure: Task {task_id}")
                
                logger.debug(f"Rule 5: Added {recommendations_added} recommendations for failed forecast tasks")
            else:
                logger.warning("Rule 5: 'task_name' column missing from main DataFrame")
        else:
            logger.warning("Rule 5: Main DataFrame is empty or missing 'task_id' column")
    else:
        logger.info("Rule 5: No failed forecast tasks provided")

    logger.info(f"Generated {len(recommendations)} recommendations.")
    # Add detailed summary of recommendation types
    if recommendations:
        trigger_counts = {}
        for rec in recommendations:
            trigger = rec.get('trigger', 'Unknown')
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        logger.debug(f"Recommendation breakdown by trigger: {trigger_counts}")
    
    return recommendations
