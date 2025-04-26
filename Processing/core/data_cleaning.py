import pandas as pd
import numpy as np
import re
from dateutil.parser import parse
import logging
# from Processing.core.schema_manager import SchemaManager, normalize_string
from .schema_manager import SchemaManager, normalize_string # Use relative import

# # Initialise SchemaManager # <<< REMOVE this global initialisation
# schema_manager = SchemaManager()

def clean_dataframe(df, schema_type="tasks", project_name="unknown_project", update_phase="update_unknown"):
    """
    Cleans and standardises a project DataFrame using schema definitions.
    """
    # --- Initialise SchemaManager INSIDE the function --- 
    try:
        schema_manager = SchemaManager(schema_type=schema_type)
    except FileNotFoundError as e:
        logging.error(f"Failed to initialise SchemaManager for type '{schema_type}': {e}")
        raise # Re-raise as cleaning cannot proceed without a valid schema
    except Exception as e:
        logging.error(f"Unexpected error initialising SchemaManager for type '{schema_type}': {e}", exc_info=True)
        raise # Re-raise 
    # --- End Initialisation ---

    logging.info(f"Starting cleaning for {project_name} ({update_phase}) with {df.shape[0]} rows, {df.shape[1]} columns using schema '{schema_type}'.")
    logging.debug(f"Initial columns: {list(df.columns)}")

    # Remove fully empty rows and columns
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    logging.debug(f"Columns after dropping empty: {list(df.columns)}")

    # Drop metadata or index columns
    initial_cols = set(df.columns)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    df = df.loc[:, ~df.columns.str.contains("metadata|header|notes", case=False)]
    dropped_meta_cols = initial_cols - set(df.columns)
    if dropped_meta_cols:
        logging.debug(f"Dropped metadata/unnamed columns: {list(dropped_meta_cols)}")
    logging.debug(f"Columns after dropping metadata: {list(df.columns)}")

    # Sanitize headers using normalize_string for consistency
    original_headers = list(df.columns)
    df.columns = [normalize_string(col) for col in df.columns]
    sanitized_headers = list(df.columns)
    header_map = {orig: san for orig, san in zip(original_headers, sanitized_headers) if orig != san}
    if header_map:
        logging.debug(f"Sanitized headers: {header_map}")
    logging.debug(f"Columns after sanitizing: {sanitized_headers}")

    # --- Check for duplicate column names after sanitizing --- 
    if len(sanitized_headers) != len(set(sanitized_headers)):
        import collections
        duplicates = [item for item, count in collections.Counter(sanitized_headers).items() if count > 1]
        logging.error(f"DUPLICATE column names found after sanitization: {duplicates}")
        print(f"ERROR: DUPLICATE column names found after sanitization: {duplicates}")
        # Decide how to handle - Option 1: Raise error, Option 2: Attempt rename, Option 3: Drop duplicates?
        # For now, let's raise an error to make it obvious
        raise ValueError(f"Duplicate column names created during sanitization: {duplicates}")
    else:
        logging.debug("No duplicate column names found after sanitization.")
    # --- End duplicate check ---

    # --- DEBUG: Print columns just before SchemaManager --- 
    print("\n--- DEBUG: Columns BEFORE SchemaManager.standardise_columns ---")
    print(list(df.columns))
    try:
        print(df.head().to_string())
    except Exception as e:
        print(f"Could not print df.head(): {e}") # Handle potential errors printing complex data
    print("--- END DEBUG ---\n")
    # --- END DEBUG --- 

    # Standardise column names using SchemaManager
    logging.info(f"Applying SchemaManager ('{schema_type}') standardisation...")
    try:
        # --- Remove prints around SchemaManager usage (schema_manager now initialised above) ---
        # print("DEBUG: data_cleaning - About to initialize SchemaManager") # <<< REMOVED
        # schema = SchemaManager(schema_type) # <<< REMOVED (already initialized as schema_manager)
        # print("DEBUG: data_cleaning - SchemaManager initialized successfully. Schema object:", schema) # <<< REMOVED
        
        logging.debug(f"Columns JUST BEFORE schema_manager.standardise_columns: {list(df.columns)}")

        # print("DEBUG: data_cleaning - About to call schema.standardise_columns") # <<< REMOVED
        # Assign the result back to df
        df = schema_manager.standardise_columns(df) # Use the initialized schema_manager
        # print("DEBUG: data_cleaning - Returned from schema.standardise_columns") # <<< REMOVED

        # Now df holds the standardised dataframe
        logging.info(f"Columns AFTER SchemaManager standardisation: {list(df.columns)}")

        # --- Add Duration Parsing Logic ---
        duration_col = 'duration' # Assuming 'duration' is the standardised name
        if duration_col in df.columns:
            logging.debug(f"Attempting to parse duration column: {duration_col}")
            # Function to parse duration string (e.g., "5 wks", "10 days", "15")
            def parse_duration(val):
                if pd.isna(val):
                    return np.nan
                val_str = str(val).lower().strip()
                try:
                    # Check for weeks
                    if 'wk' in val_str:
                        num = re.findall(r'\d+\.?\d*|\d+', val_str)[0]
                        return float(num) * 7
                    # Check for days
                    elif 'day' in val_str:
                        num = re.findall(r'\d+\.?\d*|\d+', val_str)[0]
                        return float(num)
                    # Assume numeric otherwise
                    else:
                        return float(val_str)
                except (ValueError, IndexError):
                    logging.warning(f"Could not parse duration value: {val}. Returning NaN.")
                    return np.nan # Return NaN if parsing fails
        
            # Apply the parsing function
            df[duration_col] = df[duration_col].apply(parse_duration)
            logging.debug(f"Duration column parsed. Non-null values: {df[duration_col].notna().sum()}")
        # --- End Duration Parsing Logic ---

        # Convert data types based on schema (using the now standardised df)
        df = schema_manager.convert_data_types(df) # Use the initialized schema_manager
        # Enforce not null constraints (using the now standardised df)
        df = schema_manager.enforce_not_null(df) # Use the initialized schema_manager

    except FileNotFoundError as e:
        logging.error(f"Schema file not found during standardisation: {e}")
        # Decide how to handle missing schema - skip processing?
        raise # Re-raise for now to make it clear processing cannot continue
    except Exception as e:
        # Log the specific error during standardisation
        logging.error(f"Error in SchemaManager standardisation: {e}", exc_info=True) # Include stack trace
        # Depending on requirements, maybe raise the exception or return the unstandardised df?
        # For now, return the potentially partially processed df to see what state it's in.
        # return df # Return df as is if standardisation failed

    # --- DEBUG: Log columns JUST BEFORE returning from clean_data ---
    logging.debug(f"Columns BEFORE returning from clean_data: {list(df.columns)}")

    # Add project-level metadata
    # df["update_phase"] = update_phase # <<< REMOVED: This overwrites the column kept by SchemaManager
    df["project_name"] = project_name

    # Date standardisation
    date_cols = [
        "start_date", "end_date", "actual_start", "actual_end",
        "baseline_start", "baseline_end", "planned_start", "planned_end",
        "forecast_date", "actual_date"
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Duration logic
    if "duration" not in df.columns:
        if "start_date" in df.columns and "end_date" in df.columns:
            df["duration"] = (df["end_date"] - df["start_date"]).dt.days
        elif "baseline_start" in df.columns and "baseline_end" in df.columns:
            df["duration"] = (df["baseline_end"] - df["baseline_start"]).dt.days
        else:
            df["duration"] = np.nan

    # Fill percent_complete if missing
    if "percent_complete" not in df.columns:
        df["percent_complete"] = 0
    else:
        df["percent_complete"] = pd.to_numeric(df["percent_complete"], errors="coerce").fillna(0)

    # Task ID standardisation
    if "task_id" not in df.columns:
        if "task_code" in df.columns:
            df["task_id"] = df["task_code"]
        elif "task_name" in df.columns:
            df["task_id"] = df["task_name"].str.lower().str.replace(" ", "_")
        else:
            df["task_id"] = "unknown_" + df.index.astype(str)

    df["task_id"] = df["task_id"].astype(str).str.strip().str.lower()

    # Risk flag placeholder if not present
    if "risk_flag" not in df.columns:
        df["risk_flag"] = "not_evaluated"

    # Forecast and deviation logic placeholder
    if "forecast_date" in df.columns and "end_date" in df.columns:
        df["deviation_days"] = (df["forecast_date"] - df["end_date"]).dt.days
    else:
        df["deviation_days"] = np.nan

    # Critical path handling
    if "is_critical" not in df.columns:
        df["is_critical"] = False

    # Status inference
    if "status" not in df.columns:
        df["status"] = np.where(df["percent_complete"] >= 100, "complete", "in_progress")

    return df