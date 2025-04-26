import os
import json
import pandas as pd
import numpy as np
import logging

# Define a helper function to prepare and save DataFrames
def _save_output_csv(df_input, required_columns, output_filepath, rename_map=None, default_values=None):
    """
    Prepares a DataFrame according to required columns and saves it to CSV.
    Creates an empty file with headers if input is None or empty.

    Args:
        df_input (pd.DataFrame or None): The input DataFrame.
        required_columns (list): List of exact column names required in the output CSV.
        output_filepath (str): Path to save the CSV file.
        rename_map (dict, optional): Dictionary to rename columns {old_name: new_name}. Defaults to None.
        default_values (dict, optional): Dictionary of default values for missing columns {col_name: value}. Defaults to None.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preparing to write {os.path.basename(output_filepath)}...")

    if df_input is not None and not df_input.empty:
        df_output = df_input.copy()
        logger.debug(f"Input DataFrame for {os.path.basename(output_filepath)} has shape {df_output.shape}. Columns: {list(df_output.columns)}")

        # Rename columns if a map is provided
        if rename_map:
            df_output.rename(columns=rename_map, inplace=True)
            logger.debug(f"Renamed columns: {list(df_output.columns)}")

        # Add missing required columns with default values
        if default_values is None:
            default_values = {} # Ensure default_values is a dict

        added_cols = []
        for col in required_columns:
            if col not in df_output.columns:
                default_val = default_values.get(col, pd.NA) # Use pd.NA as general default
                df_output[col] = default_val
                added_cols.append(f"{col} (default: {default_val})")
        if added_cols:
             logger.warning(f"Added missing columns to {os.path.basename(output_filepath)}: {', '.join(added_cols)}")

        # Check if essential columns are missing from the potentially modified input
        missing_from_input = [col for col in required_columns if col not in df_output.columns]

        # Fill default values *after* rename and *before* final column selection
        if default_values:
            for col, default in default_values.items():
                if col in df_output.columns:
                    if default is not None:
                        df_output[col] = df_output[col].fillna(default)
                elif col in required_columns:
                    df_output[col] = default

        # Select only the required columns in the specified order for the final DataFrame
        # This implicitly handles columns present in df_input but not in required_columns (they are dropped)
        # It also handles required columns that might have been missing from df_input (they were added with defaults)
        try:
            df_final = df_output[required_columns]

            # --->>> ADD DIAGNOSTIC PRINT FOR df_final <<<---
            print(f"\n--- DEBUG CHECK: df_final before save ({os.path.basename(output_filepath)}) ---")
            print(f"Shape: {df_final.shape}")
            print(df_final.head())
            print("--- END DEBUG CHECK ---\n")
            # --->>> END DIAGNOSTIC PRINT <<<---

        except KeyError as e:
            # This should theoretically not happen now if defaults cover all required columns,
            # but kept as a safeguard.
            logger.error(f"Missing column(s) preparing final DataFrame for {os.path.basename(output_filepath)}: {e}. Required: {required_columns}. Available: {df_output.columns.tolist()}", exc_info=True)
            # Create empty frame with correct headers if critical error occurs
            df_final = pd.DataFrame(columns=required_columns)

    else:
        logger.warning(f"Input DataFrame for {os.path.basename(output_filepath)} is None or empty. Creating file with headers only.")
        df_final = pd.DataFrame(columns=required_columns) # Create empty DataFrame with headers

    # Save the final DataFrame
    try:
        df_final.to_csv(output_filepath, index=False, na_rep='NA') # Use NA for missing values
        logger.info(f"Successfully wrote {os.path.basename(output_filepath)}.")
    except Exception as e:
        logger.error(f"Failed to write {os.path.basename(output_filepath)}: {e}", exc_info=True)


def write_outputs(output_path, cleaned_df, slippages, forecasts, changepoints, milestones, recommendations):
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting write_outputs V2 for {output_path} ---")
    os.makedirs(output_path, exist_ok=True)

    # --- Define Schemas and Save Files ---

    # --- Prepare DataFrames with Severity Score ---
    # Calculate severity score by merging key fields from cleaned_df and slippages
    df_for_severity = pd.DataFrame() # Default empty
    if cleaned_df is not None and not cleaned_df.empty:
        df_for_severity = cleaned_df[['task_id', 'is_critical']].copy()
        if slippages is not None and not slippages.empty and 'task_id' in slippages.columns and 'slip_days' in slippages.columns:
            df_for_severity = pd.merge(
                df_for_severity,
                slippages[['task_id', 'slip_days']],
                on='task_id',
                how='left'
            )
        else:
            df_for_severity['slip_days'] = np.nan # Add column if slippages missing

        # Fill missing values needed for calculation
        df_for_severity['slip_days'].fillna(0, inplace=True)
        df_for_severity['is_critical'].fillna(False, inplace=True) # Should come as bool from cleaning

        # Calculate the score
        df_for_severity['severity_score'] = df_for_severity['slip_days'] * 1.0 + df_for_severity['is_critical'].astype(int) * 5.0
        logger.debug(f"Calculated severity scores. Shape: {df_for_severity[['task_id', 'severity_score']].shape}")
    else:
        logger.warning("Cleaned DataFrame is empty, cannot calculate severity scores.")
        # Create placeholder score df if needed later, ensure task_id and severity_score cols exist
        df_for_severity = pd.DataFrame(columns=['task_id', 'is_critical', 'slip_days', 'severity_score'])


    # 1. task_cleaned.csv
    task_cleaned_cols = [
        "project_name", "task_id", "task_name", "start_date", "end_date",
        "baseline_start", "baseline_end", "duration", "percent_complete",
        "update_phase", "change_type", "slip_days", "is_critical", "severity_score" # Ensure severity_score is here
    ]
    # Map internal names to required output names
    task_cleaned_rename = {
        "actual_start": "start_date",
        "actual_finish": "end_date",
        "baseline_start_date": "baseline_start",
        "baseline_end_date": "baseline_end"
        # project_name, task_name, duration, percent_complete, update_phase assumed to exist if df exists
    }
    # Define defaults for columns potentially missing
    task_cleaned_defaults = {
        "change_type": None,
        "slip_days": 0,
        "is_critical": False,
        "severity_score": 0.0, # Default score if calculation fails
        "project_name": "Unknown"
    }

    # Merge calculated severity score into cleaned_df before saving
    task_cleaned_to_save = cleaned_df.copy() if cleaned_df is not None else pd.DataFrame()
    if not df_for_severity.empty and 'task_id' in df_for_severity.columns and 'severity_score' in df_for_severity.columns:
        if 'task_id' in task_cleaned_to_save.columns:
            # Drop existing score if present (e.g., from raw data) to avoid duplicate columns
            if 'severity_score' in task_cleaned_to_save.columns:
                task_cleaned_to_save = task_cleaned_to_save.drop(columns=['severity_score'])
            task_cleaned_to_save = pd.merge(
                task_cleaned_to_save,
                df_for_severity[['task_id', 'severity_score']],
                on='task_id',
                how='left'
            )
            # Also merge slip_days from slippages df if not already present from cleaning
            if 'slip_days' not in task_cleaned_to_save.columns and slippages is not None and 'task_id' in slippages.columns and 'slip_days' in slippages.columns:
                 task_cleaned_to_save = pd.merge(
                      task_cleaned_to_save,
                      slippages[['task_id', 'slip_days']],
                      on='task_id',
                      how='left'
                 )
                 task_cleaned_to_save['slip_days'].fillna(0, inplace=True) # Fill potentially new slip_days column
        else:
            logger.warning("Cannot merge severity score into task_cleaned: 'task_id' missing.")
    else:
        logger.warning("Severity score calculation result is empty or missing key columns.")
        if 'severity_score' not in task_cleaned_to_save.columns:
             task_cleaned_to_save['severity_score'] = task_cleaned_defaults['severity_score'] # Ensure column exists


    _save_output_csv(
        df_input=task_cleaned_to_save, # Use the updated DataFrame
        required_columns=task_cleaned_cols,
        output_filepath=os.path.join(output_path, "task_cleaned.csv"),
        rename_map=task_cleaned_rename,
        default_values=task_cleaned_defaults
    )

    # 2. slippage_summary.csv
    slippage_summary_cols = [
        "project_name", "task_id", "update_phase", "task_name",
        "baseline_end", "end_date", "slip_days", "severity_score", "change_type"
    ]
    # Correct rename map: only rename task_id
    slippage_rename = {}
    # Correct defaults
    slippage_defaults = {
        "project_name": "Unknown",
        "update_phase": "Unknown",
        "task_name": "Unknown",
        "baseline_end": None,
        "end_date": None,
        "slip_days": 0,
        "severity_score": 0.0, # Default score
        "change_type": "Not Available"
    }

    # Merge calculated severity score into slippages before saving
    slippages_to_save = slippages.copy() if slippages is not None else pd.DataFrame()
    if not df_for_severity.empty and 'task_id' in df_for_severity.columns and 'severity_score' in df_for_severity.columns:
        if 'task_id' in slippages_to_save.columns:
            # Drop existing score if present
            if 'severity_score' in slippages_to_save.columns:
                slippages_to_save = slippages_to_save.drop(columns=['severity_score'])
            slippages_to_save = pd.merge(
                slippages_to_save,
                df_for_severity[['task_id', 'severity_score']],
                on='task_id',
                how='left'
            )
        else:
             logger.warning("Cannot merge severity score into slippages: 'task_id' missing.")
    else:
        logger.warning("Severity score calculation result is empty or missing key columns.")
        if 'severity_score' not in slippages_to_save.columns:
             slippages_to_save['severity_score'] = slippage_defaults['severity_score'] # Ensure column exists


    _save_output_csv(
        df_input=slippages_to_save, # Use the updated DataFrame
        required_columns=slippage_summary_cols,
        output_filepath=os.path.join(output_path, "slippage_summary.csv"),
        rename_map=slippage_rename,
        default_values=slippage_defaults
    )


    # 3. changepoints.csv
    changepoints_cols = [
        "project_name", "task_id", "update_phase", "task_name",
        "change_date", "change_score", "method"
    ]
    changepoints_rename = {}
    changepoints_defaults = {
         "project_name": "Unknown",
         "task_name": "Unknown",
         "update_phase": "Unknown"
    }
    # Merge required info from cleaned_df
    changepoints_merged = None
    if changepoints is not None and not changepoints.empty and cleaned_df is not None and not cleaned_df.empty:
         # Similar merge logic as slippages
         try:
            if 'task_id' in changepoints.columns and 'task_id' in cleaned_df.columns:
                 merge_cols = ['task_id', 'project_name', 'task_name', 'update_phase']
                 missing_cleaned = [col for col in merge_cols if col not in cleaned_df.columns and col != 'task_id']
                 if not missing_cleaned:
                        changepoints_merged = pd.merge(
                            changepoints,
                            cleaned_df[merge_cols],
                            on='task_id',
                            how='left'
                        )
                 else:
                        logger.warning(f"Cannot merge changepoints, missing columns in cleaned_df: {missing_cleaned}")
                        changepoints_merged = changepoints
            else:
                 logger.warning("Cannot merge changepoints, 'task_id' missing.")
                 changepoints_merged = changepoints
         except Exception as e:
             logger.error(f"Error merging changepoints with cleaned_df: {e}", exc_info=True)
             changepoints_merged = changepoints
    else:
        changepoints_merged = changepoints

    _save_output_csv(
        df_input=changepoints_merged,
        required_columns=changepoints_cols,
        output_filepath=os.path.join(output_path, "changepoints.csv"),
        rename_map=changepoints_rename,
        default_values=changepoints_defaults
    )

    # 4. milestone_analysis.csv
    milestone_cols = [
        "project_name", "task_id", "task_name", "baseline_end", "end_date",
        "slip_days", "duration", "deviation_percent", "is_milestone", "severity_score"
    ]
    milestone_rename = {
        "baseline_end_date": "baseline_end",
        "actual_finish": "end_date"
    }
    milestone_defaults = {
        "project_name": "Unknown",
        "task_name": "Unknown",
        "slip_days": 0,
        "duration": 0.0,
        "deviation_percent": 0.0,
        "is_milestone": False,
        "severity_score": 0.0
    }
    # Merge required info from cleaned_df
    milestones_merged = None
    if milestones is not None and not milestones.empty and cleaned_df is not None and not cleaned_df.empty:
         # Similar merge logic
         try:
            if 'task_id' in milestones.columns and 'task_id' in cleaned_df.columns:
                 merge_cols = ['task_id', 'project_name', 'task_name', 'baseline_end_date', 'actual_finish', 'duration']
                 missing_cleaned = [col for col in merge_cols if col not in cleaned_df.columns and col != 'task_id']
                 if not missing_cleaned:
                     # Check if milestones already has the columns needed
                     existing_milestone_cols = [col for col in merge_cols if col in milestones.columns and col != 'task_id']
                     cols_to_merge_from_cleaned = [col for col in merge_cols if col not in milestones.columns or col == 'task_id']

                     if cols_to_merge_from_cleaned:
                         milestones_merged = pd.merge(
                             milestones,
                             cleaned_df[cols_to_merge_from_cleaned],
                             on='task_id',
                             how='left'
                         )
                     else: # Milestones already has all needed columns
                         milestones_merged = milestones
                 else:
                     logger.warning(f"Cannot merge milestones, missing columns in cleaned_df: {missing_cleaned}")
                     milestones_merged = milestones
            else:
                 logger.warning("Cannot merge milestones, 'task_id' missing.")
                 milestones_merged = milestones
         except Exception as e:
             logger.error(f"Error merging milestones with cleaned_df: {e}", exc_info=True)
             milestones_merged = milestones
    else:
         milestones_merged = milestones

    _save_output_csv(
        df_input=milestones_merged,
        required_columns=milestone_cols,
        output_filepath=os.path.join(output_path, "milestone_analysis.csv"),
        rename_map=milestone_rename,
        default_values=milestone_defaults
    )

    # 5. forecast_results.csv
    forecast_cols = [
        "project_name", "task_id", "task_name", "update_phase",
        "predicted_end_date", "forecast_confidence", "model_type", "low_confidence_flag"
    ]
    forecast_rename = {}
    forecast_defaults = {
        "project_name": "Unknown",
        "task_name": "Unknown",
        "update_phase": "Unknown",
        "predicted_end_date": None,
        "forecast_confidence": 0.0,
        "model_type": "Unknown",
        "low_confidence_flag": True
    }
    
    forecasts_merged = None
    if forecasts is not None and not forecasts.empty and cleaned_df is not None and not cleaned_df.empty:
        try:
            if 'task_id' in forecasts.columns and 'task_id' in cleaned_df.columns:
                 # Select columns to merge from cleaned_df (latest record per task)
                 merge_cols = ['task_id', 'project_name', 'update_phase']
                 latest_cleaned = cleaned_df.sort_values('update_phase', ascending=False).drop_duplicates(subset=['task_id'])
                 
                 missing_cleaned = [col for col in merge_cols if col not in latest_cleaned.columns and col != 'task_id']
                 if not missing_cleaned:
                        # Forecasts already contains task_name, predicted_end_date etc from the engine now
                        forecasts_merged = pd.merge(
                            forecasts, # Already has task_id, task_name, predicted_end_date, confidence, model, flag
                            latest_cleaned[merge_cols], # Merge project_name, update_phase
                            on='task_id',
                            how='left'
                        )
                 else:
                        logger.warning(f"Cannot merge forecasts, missing columns in cleaned_df: {missing_cleaned}")
                        forecasts_merged = forecasts # Use unmerged if context missing
            else:
                 logger.warning("Cannot merge forecasts, 'task_id' missing in forecasts or cleaned_df.")
                 forecasts_merged = forecasts
        except Exception as e:
            logger.error(f"Error merging forecasts with cleaned_df: {e}", exc_info=True)
            forecasts_merged = forecasts # Use unmerged on error
    else:
        forecasts_merged = forecasts # Pass original if it or cleaned_df is None/empty
        
    _save_output_csv(
        df_input=forecasts_merged, 
        required_columns=forecast_cols,
        output_filepath=os.path.join(output_path, "forecast_results.csv"),
        rename_map=forecast_rename, 
        default_values=forecast_defaults
    )

    # 6. recommendations.csv
    # Updated columns to match the new structure from recommendation_engine
    recommendations_cols = [
        "project_name", "task_id", "recommendation_type", "description",
        "confidence_score", "recommended_action"
    ]

    # recommendations is now expected to be a list of dictionaries
    recommendations_df = pd.DataFrame() # Initialize empty DataFrame

    if recommendations and isinstance(recommendations, list) and len(recommendations) > 0:
        try:
            # Convert the list of dictionaries directly to a DataFrame
            recommendations_df = pd.DataFrame(recommendations)
            logger.debug(f"Converted list of {len(recommendations)} recommendation dicts to DataFrame. Shape: {recommendations_df.shape}")

            # Extract project_name from cleaned_df (assuming it's consistent)
            project_name = "Unknown"
            if cleaned_df is not None and not cleaned_df.empty and 'project_name' in cleaned_df.columns:
                # Take the first project name found
                project_name = cleaned_df['project_name'].iloc[0]
                logger.debug(f"Extracted project_name: {project_name}")
            else:
                logger.warning("Could not determine project_name from cleaned_df for recommendations.")

            recommendations_df['project_name'] = project_name

            # Select and order columns
            # Ensure all expected columns from the dictionary are present
            available_cols = recommendations_df.columns.tolist()
            cols_to_use = [col for col in recommendations_cols if col in available_cols]
            missing_rec_cols = [col for col in recommendations_cols if col not in available_cols]

            if missing_rec_cols:
                logger.warning(f"Columns missing from generated recommendations dicts: {missing_rec_cols}. They will be added with defaults.")
                for col in missing_rec_cols:
                     # Assign appropriate defaults based on column name
                     if col == 'confidence_score':
                         recommendations_df[col] = 0.0
                     elif col == 'task_id':
                         recommendations_df[col] = 'N/A'
                     elif col == 'recommended_action':
                          recommendations_df[col] = 'No specific action defined.'
                     elif col == 'project_name': # Should be handled above, but for safety
                          recommendations_df[col] = 'Unknown'
                     else: # Default for any other unexpected missing column
                         recommendations_df[col] = pd.NA

            # Final selection and ordering
            recommendations_df = recommendations_df[recommendations_cols]


        except Exception as e:
            logger.error(f"Error processing recommendations list into DataFrame: {e}", exc_info=True)
            # Create empty df with headers if error occurs
            recommendations_df = pd.DataFrame(columns=recommendations_cols)

    elif isinstance(recommendations, dict):
         logger.error("Recommendations received as a dictionary (old format) instead of a list of dictionaries. Cannot process.")
         recommendations_df = pd.DataFrame(columns=recommendations_cols)
    else:
        logger.info("Recommendations list is empty or invalid.")
        recommendations_df = pd.DataFrame(columns=recommendations_cols)


    # No need for _save_output_csv helper here as we constructed the df manually
    try:
        # --->>> ADD DIAGNOSTIC PRINT FOR recommendations_df <<<---
        print(f"\n--- DEBUG CHECK: recommendations_df before save ---")
        print(f"Shape: {recommendations_df.shape}")
        print(recommendations_df.head().to_string()) # Use to_string() for better formatting if wide
        print("--- END DEBUG CHECK ---\n")
        # --->>> END DIAGNOSTIC PRINT <<<---

        recommendations_df.to_csv(os.path.join(output_path, "recommendations.csv"), index=False, na_rep='NA')
        logger.info("Successfully wrote recommendations.csv.")
    except Exception as e:
        logger.error(f"Failed to write recommendations.csv: {e}", exc_info=True)


    # --- DEPRECATED: merge_summaries Call ---
    # logger.info("Note: merge_summaries function is deprecated for standard output generation.")
    # merged_summary = merge_summaries(cleaned_df, slippages, forecasts)
    # if merged_summary is not None:
    #     merged_summary.to_csv(os.path.join(output_path, "merged_summary_deprecated.csv"), index=False, na_rep='NA')
    #     logger.info("Wrote merged_summary_deprecated.csv (for reference).")
    # else:
    #     logger.warning("Merged summary could not be generated.")


    logger.info(f"--- Finished write_outputs V2 for {output_path} ---")


# ------------------ Deprecated Function ------------------
# This function is kept for reference but is no longer the primary method
# for generating the standard output CSVs.
# ---------------------------------------------------------
def merge_summaries(df, slippages, forecasts):
    logger = logging.getLogger(__name__)
    logger.debug("--- Starting merge_summaries (Legacy/Internal Use Only?) ---")
    # Define required columns for the base summary
    required_cols = [
        "task_id", "task_name", "project_name", "update_phase",
        "actual_start", "actual_finish", "baseline_end_date", "percent_complete", "severity_score"
    ]
    logging.debug(f"Base summary required columns: {required_cols}")

    # Ensure required columns exist in the main DataFrame (df)
    # Use a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    for col in required_cols:
        if col not in df_copy.columns:
            logging.warning(f"Column '{col}' missing in DataFrame for LEGACY summary. Adding with NaN.")
            df_copy[col] = np.nan

    df_summary = df_copy[required_cols]
    logging.debug(f"Created base summary DataFrame with shape: {df_summary.shape}")

    # Merge Slippages if available
    if slippages is not None and not slippages.empty:
        logging.debug("Merging slippages data...")
        slippage_cols_to_merge = ["task_id", "slip_days", "change_type"] # Example, adjust if needed
        # Check if required slippage columns exist
        missing_slippage_cols = [col for col in slippage_cols_to_merge if col not in slippages.columns]
        if not missing_slippage_cols:
            try:
                if 'task_id' in df_summary.columns and 'task_id' in slippages.columns:
                     df_summary = pd.merge(df_summary, slippages[slippage_cols_to_merge], on="task_id", how="left")
                     logging.debug(f"Shape after merging slippages: {df_summary.shape}")
                else:
                     logging.warning("Skipping slippage merge: 'task_id' missing in df_summary or slippages.")
            except Exception as e:
                 logging.error(f"Failed to merge slippages data: {e}", exc_info=True)
        else:
             logging.warning(f"Skipping merge of slippage data due to missing columns in slippages DataFrame: {missing_slippage_cols}")
    else:
         logging.debug("Slippages DataFrame is empty or None. Skipping merge.")

    # Merge Forecasts if available
    if forecasts is not None and not forecasts.empty:
        logging.debug("Merging forecasts data...")
        # Updated columns based on typical forecast output
        forecast_cols_to_merge = ["task_id", "predicted_end_date", "forecast_confidence", "model_type", "low_confidence_flag"]
        logging.debug(f"Required forecast columns for merge: {forecast_cols_to_merge}")
        logging.debug(f"Available forecast columns: {list(forecasts.columns)}")
        # Check if required forecast columns exist
        missing_forecast_cols = [col for col in forecast_cols_to_merge if col not in forecasts.columns]
        if not missing_forecast_cols:
            try:
                 if 'task_id' in df_summary.columns and 'task_id' in forecasts.columns:
                      df_summary = pd.merge(df_summary, forecasts[forecast_cols_to_merge], on="task_id", how="left")
                      logging.debug(f"Shape after merging forecasts: {df_summary.shape}")
                 else:
                      logging.warning("Skipping forecast merge: 'task_id' missing in df_summary or forecasts.")
            except Exception as e:
                 logging.error(f"Failed to merge forecasts data: {e}", exc_info=True)
        else:
             logging.warning(f"Skipping merge of forecast data due to missing columns in forecasts DataFrame: Need {forecast_cols_to_merge}, Have {list(forecasts.columns)}")
    else:
         logging.debug("Forecasts DataFrame is empty or None. Skipping merge.")

    logging.debug("--- Finished merge_summaries (Legacy/Internal Use Only?) ---")
    return df_summary
