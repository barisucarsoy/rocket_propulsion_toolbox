import yaml
import os
import sys
from typing import Dict, Any

import os
import sys
from typing import Dict, Any

# --- Configuration ---
CONFIG_FILENAME = 'input.yaml'  # Name of the YAML config file expected in src/

# --- Determine Config Path ---
# Get the directory containing *this* __init__.py file (which is the src directory)
_src_dir = os.path.dirname(os.path.abspath(__file__))
_config_filepath = os.path.join(_src_dir, CONFIG_FILENAME)

def load_config(config_filepath: str) -> Dict[str, Any]:

    print(f"[ConfigLoader] Attempting to load configuration from: {config_filepath}\n")

    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"Configuration file not found at: {config_filepath}")

    try:
        with open(config_filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise TypeError(f"Expected top-level dictionary in '{config_filepath}', found {type(data)}")

        return data

    except yaml.YAMLError as e:
        # Add context to the exception message
        raise yaml.YAMLError(f"Error parsing YAML file '{config_filepath}': {e}") from e
    except Exception as e:
        # Catch other potential errors (like permission issues, TypeErrors from validation)
        # and wrap them or re-raise with context if necessary.
        print(f"[ConfigLoader] An unexpected error occurred processing '{config_filepath}': {e}", file=sys.stderr)
        # Re-raising allows the caller (__init__.py) to decide how to handle it (e.g., exit)
        raise # Or wrap in a custom exception: raise RuntimeError(f"Failed to load configuration: {e}") from e

# --- Load Configuration ---
# This block runs once when the 'src' package is first imported.
try:
    # Call the loading function from config_loader
    config_data: Dict[str, Any] = load_config(_config_filepath)
    # print(f"[Config Init] 'src' package initialized. Configuration assigned to 'src.config_data'.")

except (FileNotFoundError, TypeError, ValueError, Exception) as e: # Catch specific errors from load_config
    # Handle critical configuration loading errors by exiting
    print(f"[Config Init] CRITICAL ERROR: Failed to load or validate configuration '{CONFIG_FILENAME}'.", file=sys.stderr)
    print(f"Error Details: {e}", file=sys.stderr)
    sys.exit(f"Exiting due to configuration error.") # Exit the application
