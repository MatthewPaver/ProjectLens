import os
import sys
import subprocess
import platform
import time  # Import time
import logging
import logging.config
import re
# Project-specific imports are moved into run_pipeline
# # from analysis import slippage_analysis, changepoint_detector, milestone_analysis, forecast_engine, recommendation_engine
# from analysis import slippage_analysis, changepoint_detector, milestone_analysis, forecast_engine, recommendation_engine
# # from core import file_loader, data_cleaning, project_config # Incorrect: file_loader is in ingestion
# from core import data_cleaning, project_config # Import only core modules
# from ingestion import file_loader # Correctly import file_loader from ingestion
# from output import output_writer
# # from ingestion import data_loader # Redundant if file_loader is used?

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- New setup_virtualenv function provided by user ---
def setup_virtualenv():
    venv_path = os.path.join(os.getcwd(), ".venv")
    is_macos_arm = (platform.system() == "Darwin" and platform.machine() == "arm64")

    # Detect if running with x86 Python on ARM Mac
    # Check includes platform.platform() heuristic
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
                print("❌ ERROR: You appear to be using an x86_64 (Intel) Python interpreter on an ARM64 Mac.")
                print(f"   Interpreter: {sys.executable}")
                print(f"   Platform detected: {platform_str}")
                print(f"   Processor reported: {processor_type}")
                print("   This configuration is incompatible with required ML libraries (like TensorFlow/JAX).")
                print("   Please install and run this script using a native ARM64 Python interpreter.")
                print("   Common sources for ARM64 Python:")
                print("     - Miniforge (conda-forge, arm64 version)")
                print("     - Homebrew (`brew install python`)")
                print("     - Official Python installer from python.org (universal2)")
                sys.exit("Exiting due to incompatible Python architecture.")
        except Exception as e:
             print(f"⚠️ Warning: Could not definitively confirm Python architecture compatibility: {e}")


    # Create venv if not exists
    if not os.path.exists(venv_path):
        print("🧱 Creating virtual environment...")
        try:
            # --- Modification: Use python3.11 specifically for venv creation ---
            # TensorFlow currently requires Python <= 3.11. The global python might be newer.
            # Ensure python3.11 is installed (e.g., via `brew install python@3.11`)
            python_for_venv = "python3.11"
            print(f"   Using '{python_for_venv}' to create the virtual environment.")
            # Use the specific python version to create the venv
            # Check=True will raise CalledProcessError on failure
            subprocess.run(
                [python_for_venv, "-m", "venv", venv_path],
                check=True, capture_output=True, text=True, encoding='utf-8'
            )
            # --- End Modification ---
            print("✅ Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment using '{python_for_venv}'.")
            print(f"   Command: {' '.join(e.cmd)}")
            print(f"   Stderr: {e.stderr}")
            print(f"   Stdout: {e.stdout}")
            print(f"   Ensure '{python_for_venv}' is installed and in your PATH.")
            sys.exit("Exiting due to venv creation failure.")
        except FileNotFoundError:
            # This error means the 'python3.11' command itself was not found
            print(f"❌ Error: Could not find command '{python_for_venv}' to create the virtual environment.")
            print(f"   Please install Python 3.11 (e.g., 'brew install python@3.11') and ensure it's in your PATH.")
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
                  print(f"❌ Error: Could not find 'python' or 'python3' in venv bin directory: {os.path.join(venv_path, 'bin')}")
                  sys.exit("Exiting due to missing Python executable in venv.")

    print(f"🐍 Using Python executable from venv: {python_exec}")

    # Define the requirements file path
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(requirements_file):
        print(f"⚠️ Warning: {requirements_file} not found. Skipping dependency installation.")
    else:
        print("📦 Installing/Upgrading pip...")
        try:
            subprocess.run([python_exec, "-m", "pip", "install", "--upgrade", "pip"], check=True, capture_output=True, text=True, encoding='utf-8')
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to upgrade pip, proceeding anyway...")
            print(f"   Command: {' '.join(e.cmd)}")
            print(f"   Stderr: {e.stderr}")
        except FileNotFoundError:
             print(f"❌ Error: Could not find '{python_exec}' to upgrade pip.")
             sys.exit("Exiting due to missing venv Python for pip upgrade.")


        print(f"📦 Installing dependencies from {requirements_file}...")
        try:
            # Restore original simple install command
            # install_cmd = [
            #     python_exec, "-m", "pip", "install", "--force-reinstall", "numpy",
            #     "--no-binary", ":all:", "pmdarima",
            #     "-r", requirements_file
            # ]
            # print(f"   Running install command: {' '.join(install_cmd)}")
            subprocess.run([python_exec, "-m", "pip", "install", "-r", requirements_file], check=True, capture_output=True, text=True, encoding='utf-8')
            # subprocess.run(install_cmd, check=True, capture_output=True, text=True, encoding='utf-8')

            print(f"✅ Base dependencies installed from {requirements_file}.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies from {requirements_file}.")
            print(f"   Command: {' '.join(e.cmd)}")
            print(f"   Stderr: {e.stderr}")
            print(f"   Stdout: {e.stdout}")
            sys.exit("Exiting due to requirements installation failure.")
        except FileNotFoundError:
             print(f"❌ Error: Could not find '{python_exec}' to install requirements.")
             sys.exit("Exiting due to missing venv Python for requirements install.")


        # Platform-specific TensorFlow installation for macOS ARM64
        if is_macos_arm:
            print("🍏 Detected Apple Silicon (ARM64). Ensuring ARM-optimized TensorFlow...")

            # --- Debugging Steps ---
            print("   🔍 Checking Python version in venv...")
            try:
                subprocess.run([python_exec, "--version"], check=True)
            except Exception as e:
                print(f"   ⚠️ Failed to check Python version: {e}")

            print("   🧹 Clearing pip cache...")
            try:
                subprocess.run([python_exec, "-m", "pip", "cache", "purge"], check=True, capture_output=True, text=True, encoding='utf-8')
                print("   ✅ pip cache cleared.")
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️ Failed to clear pip cache, proceeding anyway...")
                # Optionally print stderr for debugging cache clear failure
                # print(f"      Cache clear stderr: {e.stderr.strip()}")
            except FileNotFoundError:
                 print(f"   ❌ Error: Could not find '{python_exec}' to clear pip cache.")
                 # Decide if this is critical - likely okay to continue if TF install works
            # --- End Debugging Steps ---

            try:
                # Update: Install standard 'tensorflow' and 'tensorflow-metal' for GPU acceleration.
                # 'tensorflow-macos' seems deprecated or incorrect.
                # Use --upgrade to ensure latest compatible versions.
                tf_install_cmd = [python_exec, "-m", "pip", "install", "--upgrade", "tensorflow", "tensorflow-metal"]
                print(f"   Running: {' '.join(tf_install_cmd)}")
                # Remove --force-reinstall unless necessary, let pip handle dependencies.
                subprocess.run(tf_install_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                print("✅ Successfully installed/updated tensorflow and tensorflow-metal.")
            except subprocess.CalledProcessError as e:
                # Update error message
                print(f"❌ Failed to install tensorflow/tensorflow-metal.")
                print(f"   Command: {' '.join(e.cmd)}")
                print(f"   Stderr: {e.stderr}")
                print(f"   Stdout: {e.stdout}")
                # Decide if we should exit or try to continue. Exit is safer.
                sys.exit("Exiting due to TensorFlow ARM installation failure.")
            except FileNotFoundError:
                print(f"❌ Error: Could not find '{python_exec}' to install TensorFlow for ARM.")
                sys.exit("Exiting due to missing venv Python for TensorFlow ARM install.")

    print("✅ Virtual environment setup and dependency installation complete.")
    return python_exec
# --- End new setup_virtualenv ---

def run_pipeline(python_exec):
    # --- Project-specific imports removed from here ---
    # These imports were causing issues because they were run by the global python
    # interpreter that launched main.py, not the venv interpreter.
    # The actual project logic and imports should reside within the script
    # executed by the subprocess below (e.g., core/main_runner.py).
    # print("🐍 Importing project modules...")
    # try:
    #     # from analysis import slippage_analysis, changepoint_detector, milestone_analysis, forecast_engine, recommendation_engine
    #     from analysis import slippage_analysis, changepoint_detector, milestone_analysis, forecast_engine, recommendation_engine
    #     # from core import file_loader, data_cleaning, project_config # Incorrect: file_loader is in ingestion
    #     from core import data_cleaning, project_config # Import only core modules
    #     from ingestion import file_loader # Correctly import file_loader from ingestion
    #     from output import output_writer
    #     # from ingestion import data_loader # Redundant if file_loader is used?
    #     print("✅ Project modules imported successfully.")
    # except ImportError as e:
    #     print(f"❌ Failed to import project modules: {e}")
    #     print("   This might indicate an issue with the virtual environment or dependency installation.")
    #     # Check if the venv python was used correctly by setup_virtualenv
    #     print(f"   The script attempted to use Python executable: {python_exec}")
    #     print(f"   Current sys.path: {sys.path}")
    #     sys.exit("Exiting due to module import failure within run_pipeline.")
    # --- End project-specific imports ---

    print("🚀 Running main project pipeline via subprocess...\n")
    # This script (main.py) now only sets up the environment and calls the main runner script
    # using the python executable from the virtual environment.

    main_runner_script = os.path.join(os.path.dirname(__file__), "core", "main_runner.py")

    print(f"Attempting to run: {python_exec} {main_runner_script}") # Log the command

    try:
        # Run the subprocess, capture output, don't check=True immediately
        result = subprocess.run(
            [python_exec, main_runner_script],
            capture_output=True,  # Capture stdout and stderr
            text=True,            # Decode output as text
            encoding='utf-8'     # Specify encoding
            # check=False # Temporarily remove check=True to see output even on error
        )

        # Print captured output regardless of success
        print("--- Subprocess stdout: ---")
        print(result.stdout)
        print("--- Subprocess stderr: ---")
        print(result.stderr)
        print("-------------------------")

        # Now check the return code after printing output
        if result.returncode != 0:
            print(f"❌ Pipeline execution failed with exit code {result.returncode}.")
            sys.exit(result.returncode)
        else:
            print("✅ Pipeline subprocess completed successfully.")

    # Keep FileNotFoundError handling
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{python_exec}' or '{main_runner_script}' to run the pipeline.")
        sys.exit(1)
    # Add generic exception handling for other potential issues during subprocess setup/run
    except Exception as e:
        print(f"❌ An unexpected error occurred while trying to run the pipeline subprocess: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure the script is running from the intended project root directory
    expected_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.getcwd() != expected_root:
        print(f"Changing working directory to project root: {expected_root}")
        os.chdir(expected_root)

    python_path = setup_virtualenv()
    run_pipeline(python_path)
