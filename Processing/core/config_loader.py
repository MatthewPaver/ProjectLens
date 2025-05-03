#!/usr/bin/env python3
"""Configuration Loader Module.

This module handles loading configuration settings for the ProjectLens application.
It primarily loads settings from a `config.json` file expected to be located 
at the project root directory.

The expected structure of `config.json` might include:
```json
{
  "logging_level": "INFO",
  "some_api_key": "your_key_here",
  "analysis_parameters": {
    "slippage_threshold_major": 14,
    "forecast_confidence_threshold": 0.7
  }
}
```

It defines a `ROOT_DIR` based on the script's location and provides a 
`resolve_path` utility function to create absolute paths relative to this root.
A `get_config` function allows safe access to configuration values.
Error handling is included for file not found and JSON decoding errors.
"""
import os
import json
import logging

# Add other imports as needed

# --- Configuration Settings --- 

# Define the root directory of the project.
# This assumes the script exists within a subdirectory (like 'core') 
# one level down from the actual project root.
# `os.path.dirname(__file__)` gets the directory of the current script (e.g., /path/to/ProjectLens/Processing/core)
# `os.path.abspath()` ensures the path is absolute.
# `os.path.join(..., '..', '..')` navigates up two levels to reach the intended project root (e.g., /path/to/ProjectLens).
# NB: This approach is somewhat brittle. If the file structure changes (e.g., this script moves),
# this path calculation might break. Consider alternative methods like environment variables
# or a configuration file placed at the known root for more robust root detection.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- Path Resolution Function --- 
def resolve_path(relative_path):
    """Constructs an absolute path by joining a relative path to the project's ROOT_DIR.
    
    This utility ensures consistent path resolution throughout the application,
    making file access independent of the script's current working directory.

    Args:
        relative_path (str): A path relative to the project root directory (e.g., "Data/input").

    Returns:
        str: The absolute path corresponding to the input relative path.
    """
    # Joins the calculated ROOT_DIR with the provided relative path.
    return os.path.join(ROOT_DIR, relative_path)

# --- Configuration Loading --- 
# Load the main configuration file (config.json)
CONFIG_PATH = resolve_path("config.json") # Use resolve_path for consistency.
CONFIG = {}

# Attempt to load the configuration file during module import.
logger = logging.getLogger(__name__) # Use module-specific logger.
try:
    # Ensure the configuration file exists before attempting to open it.
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            CONFIG = json.load(f)
            logger.info(f"Configuration loaded successfully from: {CONFIG_PATH}")
    else:
        # Log a warning if the main config file is not found.
        # The application might proceed with default settings or fail later depending on usage.
        logger.warning(f"Configuration file not found at: {CONFIG_PATH}. Using default empty config.")
        CONFIG = {} # Ensure CONFIG is an empty dict if file not found.

except json.JSONDecodeError as e:
    # Handle errors if the config file contains invalid JSON.
    logger.error(f"Error decoding JSON from config file {CONFIG_PATH}: {e}", exc_info=True)
    # Depending on severity, could raise an error or proceed with empty config.
    CONFIG = {}
except Exception as e:
    # Catch any other unexpected errors during file reading or processing.
    logger.error(f"Failed to load configuration from {CONFIG_PATH}: {e}", exc_info=True)
    CONFIG = {}

# --- Configuration Access Function --- 
def get_config(key, default=None):
    """Retrieves a configuration value for a given key.

    Args:
        key (str): The configuration key to retrieve.
        default (any, optional): The default value to return if the key is not found. 
                                 Defaults to None.

    Returns:
        any: The value associated with the key in the configuration, or the default value.
    """
    # Uses dict.get() for safe access, returning the default if the key is absent.
    return CONFIG.get(key, default)

# (End of module)
