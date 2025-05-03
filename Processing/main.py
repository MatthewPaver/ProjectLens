#!/usr/bin/env python3
"""ProjectLens Main Entry Point & Environment Setup.

This script is the primary entry point for the user to run the ProjectLens pipeline.
Its main responsibilities are:

1.  **Environment Check & Setup:**
    - Determines the project root directory.
    - Checks for platform compatibility issues (e.g., x86 Python on ARM Mac).
    - Creates a Python virtual environment (`.venv`) in the project root if it 
      doesn't exist, using a specific Python version (e.g., python3.11) for 
      compatibility with dependencies like TensorFlow.
    - Installs or updates dependencies from `requirements.txt` into the 
      virtual environment using pip.
    - Handles platform-specific dependency installation (e.g., `tensorflow-metal` 
      for macOS ARM GPU acceleration).

2.  **Pipeline Execution:**
    - Identifies the correct Python executable within the created/verified 
      virtual environment.
    - Executes the main application logic script (`core/main_runner.py`) as a 
      subprocess using the virtual environment's Python interpreter.
    - Captures and prints the stdout/stderr from the subprocess for visibility.
    - Exits with an appropriate status code based on the success or failure of 
      the subprocess.

Usage:
    python Processing/main.py

Note: This script uses `print` statements for status updates during the setup phase,
as the main file logging is initialised later within `core/main_runner.py`.
"""
import os
import sys
import subprocess
import platform
import time
import logging
import logging.config
import re

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging for setup phase
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def setup_virtualenv():
    # Point venv_path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    venv_path = os.path.join(project_root, ".venv") 
    is_macos_arm = (platform.system() == "Darwin" and platform.machine() == "arm64")

    # Detect if running with x86 Python on ARM Mac
    if is_macos_arm:
        try:
            # Refined check: Check if platform.platform() contains 'x86_64' when platform.machine() is 'arm64'
            # Also check sys.executable path as a fallback
            platform_str = platform.platform().lower()
            processor_type = platform.processor() # Can sometimes report 'i386' under Rosetta
            is_x86_on_arm = ('x86_64' in platform_str and 'arm64' not in platform_str) or \
                            ('x86_64' in sys.executable) or \
                            (processor_type == 'i386')

            if is_x86_on_arm:
                logging.error("You appear to be using an x86_64 (Intel) Python interpreter on an ARM64 Mac.")
                logging.error(f"   Interpreter: {sys.executable}")
                logging.error(f"   Platform detected: {platform_str}")
                logging.error(f"   Processor reported: {processor_type}")
                logging.error("   This configuration is incompatible with required ML libraries (like TensorFlow/JAX).")
                logging.error("   Please install and run this script using a native ARM64 Python interpreter.")
                logging.error("   Common sources for ARM64 Python:")
                logging.error("     - Miniforge (conda-forge, arm64 version)")
                logging.error("     - Homebrew (`brew install python`)")
                logging.error("     - Official Python installer from python.org (universal2)")
                sys.exit("Exiting due to incompatible Python architecture.")
        except Exception as e:
             logging.warning(f"Could not definitively confirm Python architecture compatibility: {e}")


    # Create venv if not exists
    if not os.path.exists(venv_path):
        logging.info("Creating virtual environment...")
        try:
            # Use python3.11 specifically for venv creation
            # TensorFlow currently requires Python <= 3.11. The global python might be newer.
            # Ensure python3.11 is installed (e.g., via `brew install python@3.11`)
            python_for_venv = "python3.11"
            logging.info(f"   Using '{python_for_venv}' to create the virtual environment.")
            # Use the specific python version to create the venv
            # Check=True will raise CalledProcessError on failure
            subprocess.run(
                [python_for_venv, "-m", "venv", venv_path],
                check=True, capture_output=True, text=True, encoding='utf-8'
            )
            logging.info("Successfully created virtual environment.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create virtual environment using '{python_for_venv}'.")
            logging.error(f"   Command: {' '.join(e.cmd)}")
            logging.error(f"   Stderr: {e.stderr}")
            logging.error(f"   Stdout: {e.stdout}")
            logging.error(f"   Ensure '{python_for_venv}' is installed and in your PATH.")
            sys.exit("Exiting due to venv creation failure.")
        except FileNotFoundError:
            # This error means the 'python3.11' command itself was not found
            logging.error(f"Could not find command '{python_for_venv}' to create the virtual environment.")
            logging.error(f"   Please install Python 3.11 (e.g., 'brew install python@3.11') and ensure it's in your PATH.")
            sys.exit("Exiting due to missing Python executable for venv.")

    # Choose correct python executable path inside venv
    if os.name == "nt": # Windows
        python_exec = os.path.join(venv_path, "Scripts", "python.exe")
    else: # macOS/Linux
        python_exec = os.path.join(venv_path, "bin", "python")
        # Fallback to python3 if python doesn't exist (less common now but safe)
        if not os.path.exists(python_exec):
             python_exec_alt = os.path.join(venv_path, "bin", "python3")
             if os.path.exists(python_exec_alt):
                  python_exec = python_exec_alt
             else:
                  logging.error(f"Could not find 'python' or 'python3' in venv bin directory: {os.path.join(venv_path, 'bin')}")
                  sys.exit("Exiting due to missing Python executable in venv.")

    logging.info(f"Using Python executable from venv: {python_exec}")

    # Define the requirements file path
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(requirements_file):
        logging.warning(f"{requirements_file} not found. Skipping dependency installation.")
    else:
        logging.info("Installing/Upgrading pip...")
        try:
            subprocess.run([python_exec, "-m", "pip", "install", "--upgrade", "pip"], check=True, capture_output=True, text=True, encoding='utf-8')
        except subprocess.CalledProcessError as e:
            logging.warning("Failed to upgrade pip, proceeding anyway...")
            logging.warning(f"   Command: {' '.join(e.cmd)}")
            logging.warning(f"   Stderr: {e.stderr}")
        except FileNotFoundError:
             logging.error(f"Could not find '{python_exec}' to upgrade pip.")
             sys.exit("Exiting due to missing venv Python for pip upgrade.")


        logging.info(f"Installing dependencies from {requirements_file}...")
        try:
            subprocess.run([python_exec, "-m", "pip", "install", "-r", requirements_file], check=True, capture_output=True, text=True, encoding='utf-8')
            logging.info(f"Base dependencies installed from {requirements_file}.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install dependencies from {requirements_file}.")
            logging.error(f"   Command: {' '.join(e.cmd)}")
            if e.stderr:
                sys.stderr.write("   --- pip stderr START ---\n")
                sys.stderr.write(e.stderr.strip() + "\n")
                sys.stderr.write("   --- pip stderr END ---\n")
            else:
                 sys.stderr.write("   (pip stderr not captured or empty)\n")
            sys.exit("Exiting due to requirements installation failure.")
        except FileNotFoundError:
             logging.error(f"Could not find '{python_exec}' to install requirements.")
             sys.exit("Exiting due to missing venv Python for requirements install.")


        # Platform-specific TensorFlow installation for macOS ARM64
        if is_macos_arm:
            logging.info("Detected Apple Silicon (ARM64). Ensuring ARM-optimised TensorFlow...")

            try:
                # Install standard 'tensorflow' and 'tensorflow-metal' for GPU acceleration.
                # Use --upgrade to ensure latest compatible versions.
                tf_install_cmd = [python_exec, "-m", "pip", "install", "--upgrade", "tensorflow", "tensorflow-metal"]
                logging.info(f"   Running: {' '.join(tf_install_cmd)}")
                subprocess.run(tf_install_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                logging.info("Successfully installed/updated tensorflow and tensorflow-metal.")
            except subprocess.CalledProcessError as e:
                logging.error("Failed to install tensorflow/tensorflow-metal.")
                logging.error(f"   Command: {' '.join(e.cmd)}")
                logging.error(f"   Stderr: {e.stderr}")
                logging.error(f"   Stdout: {e.stdout}")
                sys.exit("Exiting due to TensorFlow ARM installation failure.")
            except FileNotFoundError:
                logging.error(f"Could not find '{python_exec}' to install TensorFlow for ARM.")
                sys.exit("Exiting due to missing venv Python for TensorFlow ARM install.")

    logging.info("Virtual environment setup and dependency installation complete.")
    return python_exec

def run_pipeline(python_exec):
    logging.info("Running main project pipeline via subprocess...\n")
    # This script (main.py) now only sets up the environment and calls the main runner script
    # using the python executable from the virtual environment.

    main_runner_script = os.path.join(os.path.dirname(__file__), "core", "main_runner.py")

    logging.info(f"Attempting to run: {python_exec} {main_runner_script}")

    try:
        # Run the subprocess, capture output
        result = subprocess.run(
            [python_exec, main_runner_script],
            capture_output=True,  # Capture stdout and stderr
            text=True,            # Decode output as text
            encoding='utf-8'      # Specify encoding
        )

        # Print captured output regardless of success
        print("--- Subprocess stdout: ---")
        print(result.stdout)
        print("--- Subprocess stderr: ---")
        print(result.stderr)
        print("-------------------------")

        # Now check the return code after printing output
        if result.returncode != 0:
            logging.error(f"Pipeline execution failed with exit code {result.returncode}.")
            sys.exit(result.returncode)
        else:
            logging.info("Pipeline subprocess completed successfully.")

    except FileNotFoundError:
        logging.error(f"Could not find '{python_exec}' or '{main_runner_script}' to run the pipeline.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while trying to run the pipeline subprocess: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure the script is running from the intended project root directory
    expected_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.getcwd() != expected_root:
        logging.info(f"Changing working directory to project root: {expected_root}")
        os.chdir(expected_root)

    python_path = setup_virtualenv()
    run_pipeline(python_path)
