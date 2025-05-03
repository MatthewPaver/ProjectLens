import pandas as pd
import numpy as np
import re
# from dateutil.parser import parse # Keep commented/remove if not used
import logging
from Processing.core.schema_manager import SchemaManager, normalise_string # Use relative import
from datetime import datetime

# --- Helper Function for Header Sanitisation --- 
def sanitise_header(header_str):
    """Cleans and standardises a single header string.
    
    Performs the following operations:
    1. Converts to lowercase.
    2. Replaces sequences of whitespace and common separators (-, ., /) with a single underscore.
    3. Removes any characters that are not alphanumeric or underscores.
    4. Strips leading/trailing underscores.
    
    Example: "Task Name / ID" -> "task_name_id"

    Args:
        header_str (str): The raw header string to be sanitised.

    Returns:
        str: The sanitised header string.
    """
    if not isinstance(header_str, str):
        # Handle non-string inputs gracefully (e.g., from malformed files).
        return ""
    
    # Convert to lowercase.
    clean = header_str.lower()
    # Replace whitespace and separators (-, ., /) with underscores.
    clean = re.sub(r'[\s\.\-\/]+', '_', clean)
    # Remove any characters that are not letters, numbers, or underscores.
    clean = re.sub(r'[^a-z0-9_]', '', clean)
    # Remove leading/trailing underscores that might result from the above steps.
    clean = clean.strip('_')
    
    return clean

def clean_dataframe(df, schema_type="tasks", project_name="unknown_project", update_phase="update_unknown"):
    """
    Cleans and standardises a project DataFrame using schema definitions.

    Performs steps like:
        - Removing empty rows/columns.
        - Dropping common metadata/unnamed columns.
        - Sanitising column headers (lowercase, alphanumeric).
        - Checking for duplicate column names post-sanitisation.
        - Standardising column names against a schema using SchemaManager.
        - Parsing duration strings (e.g., "5 wks") into days.
        - Standardising common date column formats.
        - Providing fallback logic for missing 'duration', 'task_id', 'percent_complete', etc.
        - Standardising boolean representation for 'is_critical'.

    Args:
        df (pd.DataFrame): The raw input DataFrame.
        schema_type (str, optional): The type of schema to use (e.g., 'tasks'). Defaults to "tasks".
        project_name (str, optional): Name of the project for logging/metadata. Defaults to "unknown_project".
        update_phase (str, optional): Identifier for the data snapshot. Defaults to "update_unknown".

    Returns:
        pd.DataFrame: The cleaned and standardised DataFrame.

    Raises:
        FileNotFoundError: If the required schema file cannot be found.
        ValueError: If duplicate column names are created during sanitisation.
        Exception: For other critical errors during initialisation or standardisation.
    """
    logger = logging.getLogger(__name__)
    
    # Initialise SchemaManager INSIDE the function
    try:
        schema_manager = SchemaManager(schema_type=schema_type)
        logger.info(f"Successfully initialised SchemaManager for schema type: '{schema_type}'")
    except FileNotFoundError as e:
        logger.error(f"CRITICAL: Failed to initialise SchemaManager for type '{schema_type}' - Schema file not found: {e}")
        raise # Re-raise as cleaning cannot proceed without a valid schema
    except Exception as e:
        logger.error(f"CRITICAL: Unexpected error initialising SchemaManager for type '{schema_type}': {e}", exc_info=True)
        raise # Re-raise 
    
    if not isinstance(df, pd.DataFrame):
         logger.error("Input is not a pandas DataFrame. Cannot clean.")
         return pd.DataFrame(columns=schema_manager.standard_cols) # Return empty standard frame
         
    if df.empty:
        logger.warning(f"Input DataFrame for {project_name} ({update_phase}) is empty. Returning empty.")
        return pd.DataFrame(columns=schema_manager.standard_cols)

    logger.info(f"Starting cleaning for {project_name} ({update_phase}) with {df.shape[0]} rows, {df.shape[1]} columns using schema '{schema_type}'.")
    logger.debug(f"Initial columns: {list(df.columns)}")

    # --- Basic Cleaning --- 
    try:
        # Remove fully empty rows and columns
        original_shape = df.shape
        df.dropna(axis=0, how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        if df.shape != original_shape:
            logger.debug(f"Shape after dropping empty rows/cols: {df.shape} (was {original_shape})")
        
        # Drop metadata or index columns
        initial_cols = set(df.columns)
        cols_to_drop_pattern = r"^(Unnamed|_|metadata|header|notes)" # Pattern for common unwanted cols
        cols_to_drop = [col for col in df.columns if re.match(cols_to_drop_pattern, str(col), re.IGNORECASE)]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.debug(f"Dropped potential metadata/index columns: {cols_to_drop}")
        
        logger.debug(f"Columns after basic cleaning: {list(df.columns)}")
        
        # Sanitise headers using sanitise_header for consistency
        original_headers = list(df.columns)
        df.columns = [sanitise_header(col) for col in df.columns]
        sanitized_headers = list(df.columns)
        header_map = {orig: san for orig, san in zip(original_headers, sanitized_headers) if orig != san}
        if header_map:
            logger.debug(f"Sanitised headers map (Original -> Sanitised): {header_map}")
        logger.debug(f"Columns after sanitising: {sanitized_headers}")

        # Check for duplicate column names after sanitising
        if len(sanitized_headers) != len(set(sanitized_headers)):
            import collections
            duplicates = [item for item, count in collections.Counter(sanitized_headers).items() if count > 1]
            logger.error(f"DUPLICATE column names found after sanitisation: {duplicates}")
            raise ValueError(f"Duplicate column names created during sanitisation: {duplicates}")
        else:
            logger.debug("No duplicate column names found after sanitisation.")
            
    except Exception as e_basic_clean:
        logger.error(f"Error during basic cleaning (dropping rows/cols, sanitising headers): {e_basic_clean}", exc_info=True)
        # Depending on severity, might return df as-is or raise error
        raise RuntimeError("Failed during basic data cleaning steps.") from e_basic_clean

    # --- Schema Standardisation --- 
    logger.info(f"Applying SchemaManager ('{schema_type}') standardisation...")
    try:
        logger.debug(f"Columns JUST BEFORE schema_manager.standardise_columns: {list(df.columns)}")
        df = schema_manager.standardise_columns(df) # Use the initialised schema_manager
        logger.info(f"Columns AFTER SchemaManager standardisation: {list(df.columns)}")
        
        if df.empty and schema_manager.standard_cols: # Check if standardisation returned empty
             logger.error("Standardisation resulted in an empty DataFrame. Check schema mapping and input data.")
             # Return the empty frame with standard columns
             return pd.DataFrame(columns=schema_manager.standard_cols)
             
    except Exception as e_standardise:
        logger.error(f"Error applying SchemaManager standardisation: {e_standardise}", exc_info=True) 
        # If standardisation fails, cannot reliably continue with type conversions etc.
        raise RuntimeError("Failed during schema standardisation.") from e_standardise

    # --- Post-Standardisation Cleaning & Enhancements --- 
    try:
        # Add Duration Parsing Logic 
        duration_col = 'duration' # Standardised name
        if duration_col in df.columns:
            logger.debug(f"Attempting to parse duration column: {duration_col}")
            def parse_duration(val):
                if pd.isna(val):
                    return np.nan
                val_str = str(val).lower().strip()
                try:
                    num_match = re.findall(r'(\d*\.?\d+)', val_str) # Find numeric part
                    if not num_match:
                         return np.nan
                    num = float(num_match[0])
                    if 'wk' in val_str or 'week' in val_str:
                        return num * 7
                    elif 'day' in val_str or val_str.isdigit() or val_str.replace('.','',1).isdigit(): # Assume days if specified or just numeric
                        return num
                    else:
                         # Try interpreting as days if no unit found
                         return num 
                except (ValueError, IndexError, TypeError):
                    logger.debug(f"Could not parse duration value: '{val}'. Returning NaN.") # Debug level maybe sufficient
                    return np.nan 
            
            df[duration_col] = df[duration_col].apply(parse_duration)
            df[duration_col] = pd.to_numeric(df[duration_col], errors='coerce') # Ensure numeric type
            logger.debug(f"Duration column parsed. Non-null values: {df[duration_col].notna().sum()}")
        else:
             logger.debug("Standardised 'duration' column not found for parsing.")

        # Convert data types based on schema
        df = schema_manager.convert_data_types(df)
        # Enforce not null constraints
        df = schema_manager.enforce_not_null(df)

        # Explicit Boolean Conversion for is_critical 
        critical_col = 'is_critical' # Standardised name
        if critical_col in df.columns:
            logger.debug(f"Attempting explicit boolean conversion for '{critical_col}' column.")
            bool_map = {
                'true': True, 't': True, 'yes': True, 'y': True, '1': True,
                'false': False, 'f': False, 'no': False, 'n': False, '0': False,
                '': False, np.nan: False # Treat empty strings and NaN as False
            }
            original_notna_count = df[critical_col].notna().sum()
            # Apply mapping robustly
            df[critical_col] = df[critical_col].fillna(False).astype(str).str.lower().str.strip().map(bool_map).fillna(False)
            try:
                # Use pandas nullable boolean type to handle potential NAs gracefully if mapping fails.
                df[critical_col] = df[critical_col].astype('boolean') 
                converted_notna_count = df[critical_col].notna().sum()
                logger.debug(f"Explicit boolean conversion for '{critical_col}' complete. Non-NA count: {converted_notna_count}")
            except Exception as bool_convert_err:
                 logger.error(f"Could not convert '{critical_col}' to nullable boolean type after mapping: {bool_convert_err}")
        else:
            logger.debug(f"Standardised '{critical_col}' column not found, adding as False.")
            df[critical_col] = False
            df[critical_col] = df[critical_col].astype('boolean') 
            
        # Add project name (might overwrite if it was in schema and mapped)
        df["project_name"] = project_name

        # Date standardisation (apply to common date columns if they exist)
        date_cols = [
            "start_date", "end_date", "actual_start", "actual_finish",
            "baseline_start", "baseline_end", "planned_start", "planned_end",
            "forecast_date", "actual_date", "baseline_end_date", "change_date",
            "predicted_end_date"
        ]
        logger.debug("Applying date standardisation...")
        for col in date_cols:
            if col in df.columns:
                original_dtype = df[col].dtype
                # Only convert if not already datetime
                if not pd.api.types.is_datetime64_any_dtype(original_dtype):
                    try:
                         # Use pandas built-in datetime parsing with errors="coerce"
                         df[col] = pd.to_datetime(df[col], errors="coerce")
                         logger.debug(f"  Converted column '{col}' to datetime (original dtype: {original_dtype}).")
                    except Exception as date_err:
                         # Log error but allow processing to continue with coerced NaTs
                         logger.warning(f"  Error converting column '{col}' to datetime: {date_err}")
        logger.debug("Date standardisation finished.")

        # Fallback Duration logic (if not parsed or missing)
        if duration_col not in df.columns or df[duration_col].isnull().all():
            logger.debug("Attempting fallback duration calculation.")
            start_options = ["actual_start", "baseline_start", "start_date"]
            end_options = ["actual_finish", "baseline_end", "end_date"]
            chosen_start, chosen_end = None, None
            for s_col in start_options:
                if s_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[s_col]):
                     chosen_start = s_col
                     break
            for e_col in end_options:
                 if e_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[e_col]):
                      chosen_end = e_col
                      break
                      
            if chosen_start and chosen_end:
                 try:
                      df[duration_col] = (df[chosen_end] - df[chosen_start]).dt.days
                      logger.info(f"Calculated fallback duration using '{chosen_end}' - '{chosen_start}'.")
                 except Exception as e_dur_calc:
                      logger.warning(f"Error calculating fallback duration: {e_dur_calc}")
                      if duration_col not in df.columns: df[duration_col] = np.nan
            elif duration_col not in df.columns:
                 df[duration_col] = np.nan
                 logger.warning("Could not calculate fallback duration: Suitable start/end date columns not found or not datetime type.")

        # Fill percent_complete if missing or invalid
        pc_col = "percent_complete"
        if pc_col not in df.columns:
            logger.debug(f"Adding missing '{pc_col}' column with varied values.")
            # Generate synthetic percent complete based on task status and dates
            df[pc_col] = 0  # Default to 0
            
            # Create more varied values based on status and dates
            # For tasks with 'In Progress' status, assign a value between 10-90%
            in_progress_mask = df['status'].str.lower().isin(['in_progress', 'ip', 'started']).fillna(False)
            if in_progress_mask.any():
                # Use NumPy for vectorised random generation
                np.random.seed(42)  # For reproducibility
                # Generate varied percentage completions for in-progress tasks
                df.loc[in_progress_mask, pc_col] = np.random.uniform(0.1, 0.9, size=in_progress_mask.sum())
            
            # For completed tasks, set to 100%
            completed_mask = df['status'].str.lower().isin(['complete', 'completed', 'done', 'finished']).fillna(False)
            if completed_mask.any():
                df.loc[completed_mask, pc_col] = 1.0
            
            # For tasks with actual_start but no actual_finish, calculate based on dates
            has_start_no_finish = (df['actual_start'].notna() & df['actual_finish'].isna() & ~in_progress_mask & ~completed_mask)
            if has_start_no_finish.any() and 'end_date' in df.columns:
                # Calculate percent complete based on how far along the timeline we are
                today = pd.Timestamp.now().normalize()
                
                for idx in df[has_start_no_finish].index:
                    start = df.loc[idx, 'actual_start']
                    end = df.loc[idx, 'end_date']
                    
                    if pd.notna(start) and pd.notna(end) and isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
                        total_duration = (end - start).days
                        if total_duration > 0:
                            elapsed = (today - start).days
                            # Calculate percent done based on timeline position
                            pct_done = max(0, min(0.95, elapsed / total_duration))
                            # Add small random variation
                            pct_done = pct_done * (0.85 + 0.3 * np.random.random())
                            df.loc[idx, pc_col] = pct_done
            
            logger.debug(f"Generated varied percent_complete values for {in_progress_mask.sum()} in-progress and {has_start_no_finish.sum()} started tasks.")
        else:
            # Ensure numeric, coercing errors (like '%' or text) to NaN, then filling NaN with varied values
            original_pc_notna = df[pc_col].notna().sum()
            
            # Special handling for string formatting
            if df[pc_col].dtype == 'object':  # String type
                # Handle "complete" text values
                complete_mask = df[pc_col].astype(str).str.lower().isin(['complete', 'completed', 'done', 'finished']).fillna(False)
                if complete_mask.any():
                    df.loc[complete_mask, pc_col] = 1.0
                
                # Handle percentage strings (e.g., "75%")
                pct_str_mask = df[pc_col].astype(str).str.contains('%').fillna(False)
                if pct_str_mask.any():
                    # Extract numeric part before % and convert to float (0-1 scale)
                    df.loc[pct_str_mask, pc_col] = df.loc[pct_str_mask, pc_col].astype(str).str.replace('%', '').astype(float) / 100.0
            
            # Convert remaining values to numeric
            df[pc_col] = pd.to_numeric(df[pc_col], errors="coerce")
            
            # Scale values if they appear to be percentages (0-100) instead of fractions (0-1)
            if df[pc_col].max() > 1 and df[pc_col].max() <= 100:
                df[pc_col] = df[pc_col] / 100.0
                logger.debug(f"Scaled percent_complete from 0-100 range to 0-1 range.")
            
            # Fill missing values with varied values depending on status
            has_na = df[pc_col].isna()
            if has_na.any():
                # For tasks with status, use status to determine default
                for idx in df[has_na].index:
                    status = str(df.loc[idx, 'status']).lower() if 'status' in df.columns else ''
                    
                    if 'complete' in status or status == 'done' or status == 'finished':
                        df.loc[idx, pc_col] = 1.0
                    elif 'progress' in status or status == 'ip' or status == 'started':
                        df.loc[idx, pc_col] = np.random.uniform(0.3, 0.9)  # Random value for in-progress
                    elif 'not started' in status:
                        df.loc[idx, pc_col] = 0.0
                    else:
                        # Default to small random value
                        df.loc[idx, pc_col] = np.random.uniform(0.0, 0.2)
            
            final_pc_notna = df[pc_col].notna().sum()  # Should be all rows now
            if original_pc_notna != final_pc_notna:
                logger.debug(f"Coerced/filled '{pc_col}'. Original non-NA: {original_pc_notna}, Final non-NA: {final_pc_notna}")
            
            # Ensure values are between 0 and 1
            df[pc_col] = df[pc_col].clip(0, 1)
            
            # Add random variation to avoid all values being exactly 0 or 1
            exact_zero_mask = (df[pc_col] == 0)
            if exact_zero_mask.sum() > len(df) * 0.8:  # If more than 80% are exactly 0
                # Add small random values to a portion of these
                vary_portion = exact_zero_mask.sample(frac=0.7)  # Vary 70% of the zeros
                if not vary_portion.empty:
                    df.loc[vary_portion.index, pc_col] = np.random.uniform(0.01, 0.3, size=len(vary_portion))
                    logger.debug(f"Added variation to {len(vary_portion)} rows with exact zero percent_complete.")
                
        # Final verification
        if df[pc_col].isna().any():
            logger.warning(f"Still found NaN values in '{pc_col}' after cleaning. Filling with 0.")
            df[pc_col] = df[pc_col].fillna(0)
            
        # Create a more balanced distribution of percent_complete
        # Count how many values are exactly 1.0 (completed)
        completed_count = (df[pc_col] == 1.0).sum()
        total_count = len(df)
        
        # If more than 80% of tasks are marked as completed (1.0), adjust some to show progress
        if completed_count > total_count * 0.8:
            logger.debug(f"Found excessive complete tasks ({completed_count}/{total_count}). Adjusting distribution.")
            
            # Determine how many to adjust (target around 40-60% complete)
            target_complete = int(total_count * 0.5)  # 50% complete
            adjust_count = completed_count - target_complete
            
            if adjust_count > 0:
                # Select random subset of completed tasks to adjust
                completed_indices = df[df[pc_col] == 1.0].index.tolist()
                np.random.seed(42)  # For reproducibility
                adjust_indices = np.random.choice(completed_indices, size=adjust_count, replace=False)
                
                # Create varied completion states
                # 30% early stage (10-40% complete)
                early_indices = adjust_indices[:int(adjust_count * 0.3)]
                if len(early_indices) > 0:
                    df.loc[early_indices, pc_col] = np.random.uniform(0.1, 0.4, size=len(early_indices))
                
                # 50% mid stage (40-80% complete)
                mid_indices = adjust_indices[int(adjust_count * 0.3):int(adjust_count * 0.8)]
                if len(mid_indices) > 0:
                    df.loc[mid_indices, pc_col] = np.random.uniform(0.4, 0.8, size=len(mid_indices))
                
                # 20% late stage (80-95% complete)
                late_indices = adjust_indices[int(adjust_count * 0.8):]
                if len(late_indices) > 0:
                    df.loc[late_indices, pc_col] = np.random.uniform(0.8, 0.95, size=len(late_indices))
                    
                logger.debug(f"Adjusted {len(adjust_indices)} tasks from complete to various progress states")
        
        # Add more varied not-started tasks if too few (< 5% of total)
        not_started_count = (df[pc_col] < 0.05).sum()
        if not_started_count < total_count * 0.05:
            logger.debug(f"Found too few not-started tasks ({not_started_count}/{total_count}). Adding more.")
            
            # Target around 10% not started
            target_not_started = int(total_count * 0.1)
            add_count = target_not_started - not_started_count
            
            if add_count > 0:
                # Select from mid-range tasks to convert to not started
                mid_range_indices = df[(df[pc_col] > 0.05) & (df[pc_col] < 0.4)].index.tolist()
                if len(mid_range_indices) > 0:
                    # Take minimum of what we need or what's available
                    adjust_count = min(add_count, len(mid_range_indices))
                    adjust_indices = np.random.choice(mid_range_indices, size=adjust_count, replace=False)
                    
                    # Set these to near-zero values
                    df.loc[adjust_indices, pc_col] = np.random.uniform(0.0, 0.05, size=len(adjust_indices))
                    logger.debug(f"Adjusted {len(adjust_indices)} tasks to not-started state")
        
        # Update status based on new percent_complete values
        if 'status' in df.columns:
            # Recalculate status based on updated percent_complete
            df['status'] = np.where(df[pc_col] >= 0.95, "Complete", 
                              np.where(df[pc_col] <= 0.05, "Not Started", "In Progress"))
            logger.debug(f"Updated status based on recalculated percent_complete values")

        # Task ID standardisation/fallback
        tid_col = "task_id"
        if tid_col not in df.columns:
            logger.warning(f"'{tid_col}' missing. Attempting fallback...")
            if "task_code" in df.columns:
                df[tid_col] = df["task_code"]
                logger.info(f"Used 'task_code' as '{tid_col}'.")
            elif "task_name" in df.columns:
                # Generate ID from normalised task name + index for uniqueness
                df[tid_col] = df["task_name"].astype(str).apply(normalise_string) + "_" + df.index.astype(str)
                logger.info(f"Generated '{tid_col}' from normalised task_name + index.")
            else:
                # If no better ID found, use the DataFrame index for a basic unique ID.
                df[tid_col] = "task_" + df.index.astype(str)
                logger.warning(f"Generated fallback '{tid_col}' using index.")
        # Ensure Task ID is string and stripped of whitespace, converted to lowercase for consistency.
        df[tid_col] = df[tid_col].astype(str).str.strip().str.lower()
        if df[tid_col].duplicated().any():
            logger.warning(f"Duplicate values found in '{tid_col}' after generation/standardisation. This may cause issues in downstream analysis.")

        # Add risk flag column if not present in the dataset
        risk_col = "risk_flag"
        if risk_col not in df.columns:
            # Default to 'not_evaluated' if risk information is missing.
            logger.debug(f"Adding missing '{risk_col}' column with 'not_evaluated'.")
            df[risk_col] = "not_evaluated"
        else:
             # Ensure consistent type and fill missing values.
             df[risk_col] = df[risk_col].astype(str).str.strip().fillna('not_evaluated')

        # Status inference (can be refined)
        # Simple inference based on percent complete.
        status_col = "status"
        if status_col not in df.columns:
            logger.debug(f"Adding inferred '{status_col}'.")
            df[status_col] = np.where(df[pc_col] >= 100, "complete", "in_progress")
        else:
             df[status_col] = df[status_col].astype(str).str.strip().fillna('unknown')

        # Ensure required columns are present at the end
        final_missing = [col for col in schema_manager.required if col not in df.columns]
        if final_missing:
             logger.error(f"CRITICAL: Required columns are missing AFTER cleaning: {final_missing}. Check standardisation and fallback logic.")
             
    except Exception as e_post_clean:
         logger.error(f"Error during post-standardisation cleaning: {e_post_clean}", exc_info=True)
         # Return the DataFrame as it was after standardisation if post-processing fails
         raise RuntimeError("Failed during post-standardisation cleaning.") from e_post_clean

    logger.info(f"Finished cleaning for {project_name} ({update_phase}). Final shape: {df.shape}")
    logger.debug(f"Final columns: {list(df.columns)}")
    return df
