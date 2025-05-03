import json
import os
from difflib import get_close_matches
from .config_loader import resolve_path
import logging
import pandas as pd
import numpy as np
import re

# --- Helper function for simple string normalisation ---
def normalise_string(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())

# Backwards compatibility alias for normalise_string
normalize_string = normalise_string

class SchemaManager:
    """Manages loading, standardising, and applying data schemas.
    
    This class loads a schema definition from a JSON file (e.g., `tasks_schema.json`)
    and provides methods to standardise a DataFrame according to that schema.
    
    Expected Schema JSON Structure:
    {
        "properties": {
            "standard_col_name_1": {"type": "string", "description": "Unique identifier for the task"},
            "standard_col_name_2": {"type": "number", "format": "float"},
            "standard_col_name_3": {"type": "date"},
            "standard_col_name_4": {"type": "boolean"},
            "standard_col_name_5": {
                "type": "string",
                "synonyms": ["alternate_name_a", "alternate_name_b"]
            }

        },
        "required_columns": ["standard_col_name_1", "standard_col_name_3"],
        "synonyms": { 
            // Top-level synonyms (can be overridden by property-level synonyms)
            "standard_col_name_1": ["alternative_1", "legacy_name_1"],
            "standard_col_name_2": ["alternative_2"]
        }
    }
    
    Key functionalities:
    - Loads schema from `Data/schemas/<schema_type>_schema.json` (with fallback).
    - Merges top-level and property-level synonyms.
    - `standardise_columns`: Renames DataFrame columns to match schema using exact,
      synonym, and normalised matching strategies.
    - `convert_data_types`: Converts columns to types specified in schema properties.
    - `enforce_not_null`: Checks for nulls in columns marked as required.
    """
    def __init__(self, schema_type):
        """Initialises the SchemaManager for a specific schema type.
        
        Loads the schema file, extracts standard column names, required columns,
        and merges top-level and property-level synonyms.
        
        Args:
            schema_type (str): The name of the schema to load (e.g., 'tasks'),
                               corresponding to a file in `Data/schemas/`.
                               
        Raises:
            FileNotFoundError: If the schema file cannot be found.
            json.JSONDecodeError: If the schema file contains invalid JSON.
            IOError: For other file reading errors.
        """
        # Try with _schema.json suffix first (new convention)
        schema_path = resolve_path(f"Data/schemas/{schema_type}_schema.json")
        
        # Fallback to .json if _schema.json doesn't exist (backwards compatibility)
        if not os.path.exists(schema_path):
            fallback_path = resolve_path(f"Data/schemas/{schema_type}.json")
            if os.path.exists(fallback_path):
                schema_path = fallback_path
                logging.warning(f"Using legacy schema path: {schema_type}.json")
            else:
                logging.error(f"Could not find schema file for {schema_type}")
                
        self.schema_path = schema_path
        self.suggestions_path = resolve_path("Data/schemas/schema_suggestions.json")
        self.schema = self.load_schema()
        
        # --- Load synonyms from top level AND from properties --- 
        # 1. Initialise with top-level synonyms
        self.synonyms = self.schema.get("synonyms", {}).copy()

        # 2. Get standard column names from properties
        properties = self.schema.get("properties", {})
        self.standard_cols = list(properties.keys())
        if not self.standard_cols:
             logging.warning(f"Schema {schema_type} loaded, but no columns found under 'properties'. Standardisation may fail.")

        # 3. Iterate through properties to find nested synonyms
        logging.debug("Checking properties for nested synonyms...")
        for prop_name, prop_details in properties.items():
            if isinstance(prop_details, dict) and "synonyms" in prop_details:
                prop_synonyms = prop_details["synonyms"]
                if isinstance(prop_synonyms, list):
                    # Ensure the property name itself has an entry in self.synonyms
                    if prop_name not in self.synonyms:
                        self.synonyms[prop_name] = []
                    # Add nested synonyms, avoiding duplicates
                    for syn in prop_synonyms:
                        if syn not in self.synonyms[prop_name]:
                            self.synonyms[prop_name].append(syn)
                    logging.debug(f"  Added/merged synonyms for '{prop_name}': {self.synonyms[prop_name]}")
                else:
                    logging.warning(f"Synonyms defined for property '{prop_name}' are not a list: {prop_synonyms}")
        
        # Load required columns (can be done after loading properties)
        # These are columns defined in the schema's "required_columns" list.
        self.required = self.schema.get("required_columns", [])
        
        # Log final loaded synonyms for debugging
        logging.debug(f"Schema Manager initialised for '{schema_type}'. Standard columns: {self.standard_cols}")
        logging.debug(f"FINAL Loaded synonyms after merging: {self.synonyms}")

    def load_schema(self):
        try:
            with open(self.schema_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Schema file not found: {self.schema_path}")
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in schema file: {self.schema_path} - {e}")
            raise json.JSONDecodeError(f"Invalid JSON in schema file: {self.schema_path} - {e.msg}", e.doc, e.pos)
        except Exception as e:
            logging.error(f"Error loading schema file {self.schema_path}: {e}", exc_info=True)
            raise IOError(f"Error loading schema file {self.schema_path}: {e}")

    def standardise_columns(self, df):
        # Ensure logger is capturing DEBUG messages for this method
        logger = logging.getLogger()
        effective_level = logger.getEffectiveLevel()
        # Temporarily set level to DEBUG if it's higher
        original_level = logger.level
        if effective_level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)
            logging.debug(f"Temporarily set logger level to DEBUG (was {logging.getLevelName(effective_level)})")

        logging.debug("--- Starting SchemaManager.standardise_columns ---")
        rename_map = {}
        processed_input_cols = set() # Keep track of input cols already mapped

        # Assume df.columns are already cleaned by data_cleaning.py
        input_cols_present = set(df.columns)
        # Create a mapping of normalised input column names to original names for faster lookup
        normalised_input_cols_map = {normalise_string(col): col for col in input_cols_present}
        logging.debug(f"SchemaManager - Input columns received: {list(input_cols_present)}")
        logging.debug(f"SchemaManager - Normalised input map created.")

        # Iterate through the standard columns defined in the schema
        logging.debug(f"SchemaManager - Iterating through standard columns: {self.standard_cols}")
        for std_col in self.standard_cols:
            logging.debug(f"--- SchemaManager - Processing standard column: '{std_col}' ---")
            normalised_std_col = normalise_string(std_col) # Normalise standard column name once
            logging.debug(f"    Normalised standard column: '{normalised_std_col}'")
            found_match = False
            input_col_to_map = None # Track which input column matched

            # 1. Check for exact match between standard name and a cleaned input name
            logging.debug(f"  SchemaManager - Checking Exact Match: Is '{std_col}' present in input columns?")
            if std_col in input_cols_present:
                logging.debug(f"    Exact Match Found: Standard column '{std_col}' is present in input.")
                input_col_to_map = std_col
                if input_col_to_map not in processed_input_cols:
                    processed_input_cols.add(input_col_to_map)
                    rename_map[input_col_to_map] = std_col
                    logging.info(f"  Schema Mapping (Exact): Input '{input_col_to_map}' -> '{std_col}'")
                    found_match = True
                else:
                    logging.debug(f"    Input column '{input_col_to_map}' (matched by exact '{std_col}') already processed.")
            else:
                 logging.debug(f"    Exact Match Not Found.")

            # 2. Check for synonym match
            if not found_match and std_col in self.synonyms:
                logging.debug(f"  SchemaManager - Checking Synonyms for '{std_col}': {self.synonyms[std_col]}")
                for synonym in self.synonyms[std_col]:
                    # Clean the synonym itself using the same logic as data_cleaning.py (or similar)
                    cleaned_synonym = normalise_string(synonym)
                    logging.debug(f"    Checking normalised synonym: '{cleaned_synonym}' (original: '{synonym}') for std_col: '{std_col}'")
                    
                    match_in_map = cleaned_synonym in normalised_input_cols_map
                    logging.debug(f"      Is '{cleaned_synonym}' in normalised map keys? {match_in_map}")

                    if match_in_map: # Check against *normalised* input cols
                        logging.debug(f"    Synonym Match Found: Normalised synonym '{cleaned_synonym}' is present in normalised input columns.")
                        # Find the original input column name corresponding to the normalised match
                        input_col_to_map = normalised_input_cols_map[cleaned_synonym]
                        logging.debug(f"    Mapped to original input column: '{input_col_to_map}'")
                        if input_col_to_map not in processed_input_cols:
                             if std_col not in rename_map.values():
                                 rename_map[input_col_to_map] = std_col
                                 processed_input_cols.add(input_col_to_map)
                                 logging.info(f"  Schema Mapping (Synonym): Input '{input_col_to_map}' (from synonym '{synonym}') -> '{std_col}'")
                                 found_match = True
                                 break
                             else:
                                 logging.debug(f"    Target standard column '{std_col}' is already mapped. Skipping synonym match for '{input_col_to_map}'.")
                        else:
                             logging.debug(f"    Input column '{input_col_to_map}' (matched by synonym '{synonym}') already processed. Skipping.")
                    else:
                         logging.debug(f"    Synonym '{cleaned_synonym}' not found in input column names.")
                if not found_match:
                     logging.debug(f"    No synonym match found for '{std_col}' after checking all synonyms.")
            elif not found_match:
                 logging.debug(f"  SchemaManager - No synonyms defined or checked for '{std_col}'.")

            # 3. --- Simplified Fuzzy Matching Fallback (using normalisation) ---
            if not found_match:
                logging.debug(f"  SchemaManager - Checking Normalised Fallback for '{std_col}' (normalised: '{normalised_std_col}')")
                # Check if normalised standard column matches any *unprocessed* normalised input column
                match_found_via_norm = False
                for norm_input, orig_input in normalised_input_cols_map.items():
                    if norm_input == normalised_std_col:
                        # Found a potential match via normalisation
                        input_col_to_map = orig_input
                        logging.debug(f"    Normalised Match Found: Standard '{std_col}' (norm: '{normalised_std_col}') matches Input '{input_col_to_map}' (norm: '{norm_input}')")
                        if input_col_to_map not in processed_input_cols:
                            if std_col not in rename_map.values():
                                rename_map[input_col_to_map] = std_col
                                processed_input_cols.add(input_col_to_map)
                                logging.info(f"  Schema Mapping (Normalised): Input '{input_col_to_map}' -> '{std_col}'")
                                found_match = True
                                match_found_via_norm = True
                                break # Found match for this std_col
                            else:
                                logging.debug(f"    Target standard column '{std_col}' is already mapped. Skipping normalised match for '{input_col_to_map}'.")
                        else:
                            logging.debug(f"    Input column '{input_col_to_map}' (matched by normalisation) already processed. Skipping.")

                if not match_found_via_norm:
                    # Check if normalised *synonyms* match any normalised input column
                    if std_col in self.synonyms:
                        logging.debug(f"    Checking Normalised Synonyms for '{std_col}'")
                        for synonym in self.synonyms[std_col]:
                             normalised_synonym = normalise_string(synonym)
                             if normalised_synonym in normalised_input_cols_map:
                                input_col_to_map = normalised_input_cols_map[normalised_synonym]
                                logging.debug(f"    Normalised Synonym Match: Std '{std_col}', Synonym '{synonym}' (norm: '{normalised_synonym}') matches Input '{input_col_to_map}' (norm: '{normalised_synonym}')")
                                if input_col_to_map not in processed_input_cols:
                                    if std_col not in rename_map.values():
                                        rename_map[input_col_to_map] = std_col
                                        processed_input_cols.add(input_col_to_map)
                                        logging.info(f"  Schema Mapping (Normalised Synonym): Input '{input_col_to_map}' -> '{std_col}'")
                                        found_match = True
                                        break # Found match for this std_col
                                    else:
                                         logging.debug(f"    Target standard column '{std_col}' is already mapped. Skipping normalised synonym match for '{input_col_to_map}'.")
                                else:
                                     logging.debug(f"    Input column '{input_col_to_map}' (matched by normalised synonym) already processed. Skipping.")
                        if found_match: break # Exit outer loop if match found via normalised synonym


            if not found_match:
                logging.debug(f"  No match found for standard column '{std_col}' after all checks.")


        # Log columns that couldn't be mapped
        logging.debug("--- SchemaManager - Checking for unmapped columns ---")
        unmapped_count = 0
        for input_col in input_cols_present: # Iterate over actual input columns
            if input_col not in processed_input_cols:
                 # Use input_col directly in log message
                 logging.warning(f"Column '{input_col}' could not be mapped to schema and will be dropped.")
                 unmapped_count += 1
        if unmapped_count == 0:
             logging.debug("SchemaManager - All input columns were successfully mapped.")
        else:
             logging.warning(f"SchemaManager - {unmapped_count} columns could not be mapped.")

        logging.debug(f"SchemaManager - Final rename_map before applying: {rename_map}")

        # Select only the columns that were successfully mapped and rename them
        mapped_input_cols = list(rename_map.keys())
        if not mapped_input_cols:
            logging.error("No columns were successfully mapped to the schema. Returning empty DataFrame.")
            return pd.DataFrame(columns=self.standard_cols) # Return empty DF with standard cols

        df_renamed = df[mapped_input_cols].rename(columns=rename_map)

        # Add missing standard columns (those in schema but not in df_renamed after mapping)
        present_std_cols = set(df_renamed.columns)
        missing_std_cols = set(self.standard_cols) - present_std_cols
        if missing_std_cols:
             logging.debug(f"Adding missing standard columns with NaN: {missing_std_cols}")
             for col in missing_std_cols:
                 df_renamed[col] = np.nan
        else:
             logging.debug("No standard columns missing from mapped DataFrame.")

        # Reorder columns to match schema definition
        final_col_order = [col for col in self.standard_cols if col in df_renamed.columns]
        # Check if any columns were renamed *to* something not in standard_cols (shouldn't happen)
        extra_cols = [col for col in df_renamed.columns if col not in self.standard_cols]
        if extra_cols:
            logging.warning(f"Mapped columns exist that are not in the final standard column list: {extra_cols}. These will be dropped.")

        df_final = df_renamed[final_col_order]
        logging.debug(f"Final columns after standardisation and reordering: {list(df_final.columns)}")
        logging.debug("--- Finished SchemaManager.standardise_columns ---")

        # --- Restore original logger level if changed --- 
        if effective_level > logging.DEBUG:
            logger.setLevel(original_level)
            logging.debug(f"Restored logger level to {logging.getLevelName(original_level)}")
        return df_final

    def convert_data_types(self, df):
        """
        Converts DataFrame columns to types specified in the schema properties.
        Handles string, number, integer, boolean, and date/datetime types.
        Coerces errors to NaN/NaT and logs warnings for any conversion issues.
        """
        import pandas as pd
        import numpy as np
        import logging
        properties = self.schema.get("properties", {})
        for col, props in properties.items():
            if col not in df.columns:
                continue
            target_type = props.get("type", None)
            if not target_type:
                continue
            try:
                if target_type == "number":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif target_type == "integer":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype('Int64')
                elif target_type == "boolean":
                    # Map common string representations to boolean
                    bool_map = {
                        'true': True, 't': True, 'yes': True, 'y': True, '1': True,
                        'false': False, 'f': False, 'no': False, 'n': False, '0': False,
                        '': False, np.nan: False
                    }
                    df[col] = df[col].fillna(False).astype(str).str.lower().str.strip().map(bool_map).fillna(False)
                    df[col] = df[col].astype('boolean')
                elif target_type == "string":
                    df[col] = df[col].astype(str)
                elif target_type == "date" or (target_type == "string" and props.get("format") in ["date", "date-time"]):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception as e:
                logging.warning(f"Could not convert column '{col}' to type '{target_type}': {e}")
        return df

    def enforce_not_null(self, df):
        """
        Checks required columns for null values and logs a warning if any are found.
        Does not drop rows or fill values automatically.
        """
        import logging
        required_cols_in_df = [col for col in self.required if col in df.columns]
        if required_cols_in_df:
            null_counts = df[required_cols_in_df].isnull().sum()
            null_cols = null_counts[null_counts > 0]
            if not null_cols.empty:
                logging.warning(f"Found null values in required columns: {null_cols.to_dict()}")
        return df

    def log_suggestion(self, column_name):
        if not os.path.exists(self.suggestions_path):
            with open(self.suggestions_path, "w") as f:
                json.dump({"suggestions": []}, f)

        with open(self.suggestions_path, "r") as f:
            current = json.load(f)

        if column_name not in current.get("suggestions", []):
            current["suggestions"].append(column_name)
            with open(self.suggestions_path, "w") as f:
                json.dump(current, f, indent=4)
