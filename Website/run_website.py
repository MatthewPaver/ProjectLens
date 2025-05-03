import os
import sys
import subprocess
import platform
import shutil
import logging  # Add logging import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
PYTHON_FOR_VENV = "python3.11"  # Explicitly require Python 3.11
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VENV_PATH = os.path.join(PROJECT_ROOT, ".venv")
PROCESSING_REQ_FILE = os.path.join(PROJECT_ROOT, 'Processing', 'requirements.txt')


def check_python_version(python_executable):
    """Checks if the python executable is version 3.11."""
    if not os.path.exists(python_executable):
        return False, None
    try:
        result = subprocess.run([python_executable, "--version"], capture_output=True, text=True, check=True, encoding='utf-8')
        version_str = result.stdout.strip().split(' ')[1]
        logging.info(f"   Found Python version: {version_str} in {python_executable}")
        return version_str.startswith("3.11."), version_str
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, Exception) as e:
        logging.warning(f"   Warning: Could not determine version for {python_executable}: {e}")
        return False, None


def setup_shared_venv():
    """Ensures the shared root virtual environment exists, uses Python 3.11, 
       and has dependencies for both Website and Processing installed."""

    logging.info(f"--- Shared Virtual Environment Setup ({PYTHON_FOR_VENV} required) ---")
    python_in_venv = None

    if not os.path.exists(VENV_PATH):
        logging.info(f"   Root virtual environment not found at: {VENV_PATH}")
        logging.info(f"   Attempting to create using '{PYTHON_FOR_VENV}'...")

        # Verify python3.11 command exists before trying to create
        try:
            subprocess.run([PYTHON_FOR_VENV, "--version"], check=True, capture_output=True)
            logging.info(f"   Found '{PYTHON_FOR_VENV}'. Proceeding with venv creation.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.error(f"   Error: Command '{PYTHON_FOR_VENV}' not found or failed.")
            logging.info(f"   Please install Python 3.11 and ensure it is in your PATH.")
            sys.exit(f"Exiting: {PYTHON_FOR_VENV} is required to create the virtual environment.")

        # Create venv using python3.11
        try:
            subprocess.run(
                [PYTHON_FOR_VENV, "-m", "venv", VENV_PATH],
                check=True, capture_output=True, text=True, encoding='utf-8'
            )
            logging.info(f"   Root virtual environment created successfully with {PYTHON_FOR_VENV}.")
        except subprocess.CalledProcessError as e:
            logging.error(f"   Failed to create root virtual environment using '{PYTHON_FOR_VENV}'.")
            logging.error(f"   Command: {' '.join(e.cmd)}")
            logging.error(f"   Stderr: {e.stderr}")
            sys.exit("Exiting due to venv creation failure.")

    else:
        logging.info(f"   Root virtual environment found at: {VENV_PATH}")

    # --- Determine Python Executable Path Inside Venv ---
    if platform.system() == "Windows":
        python_in_venv = os.path.join(VENV_PATH, "Scripts", "python.exe")
    else:  # macOS/Linux
        python_in_venv = os.path.join(VENV_PATH, "bin", "python")
        # Fallback for safety, though 'python' should exist if created properly
        if not os.path.exists(python_in_venv):
            python_exec_alt = os.path.join(VENV_PATH, "bin", "python3")
            if os.path.exists(python_exec_alt):
                python_in_venv = python_exec_alt
            else:
                logging.error(f"   Error: Could not find 'python' or 'python3' in venv bin: {os.path.join(VENV_PATH, 'bin')}")
                sys.exit("Exiting due to missing Python executable in venv.")

    logging.info(f"   Using Python from venv: {python_in_venv}")

    # --- Verify Python Version within Venv ---
    is_correct_version, found_version = check_python_version(python_in_venv)
    if not is_correct_version:
        logging.error(f"   ERROR: Python version inside the existing .venv ('{found_version or 'Unknown'}') is NOT {PYTHON_FOR_VENV}.")
        logging.error(f"   The processing pipeline requires {PYTHON_FOR_VENV}.")
        logging.info(f"   To fix this, please manually delete the '.venv' directory:")
        logging.info(f"     cd {PROJECT_ROOT}")
        logging.info(f"     rm -rf .venv")
        logging.info(f"   Then, ensure '{PYTHON_FOR_VENV}' is installed and run this script again to recreate it.")
        sys.exit(f"Exiting due to incorrect Python version in existing .venv.")

    # --- Install/Upgrade Pip ---
    logging.info(f"   Checking/Installing pip upgrade in venv...")
    try:
        subprocess.run([python_in_venv, "-m", "pip", "install", "--upgrade", "pip"], check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info("   pip is up to date.")
    except subprocess.CalledProcessError as e:
        logging.warning(f"   Warning: Failed to upgrade pip, proceeding anyway...")
        logging.warning(f"      Command: {' '.join(e.cmd)}")
        logging.warning(f"      Stderr: {e.stderr[:200]}...")
    except FileNotFoundError:
        logging.error(f"   Error: Could not find '{python_in_venv}' to upgrade pip.")
        sys.exit("Exiting due to missing venv Python executable.")

    # --- Install Flask ---
    logging.info(f"   Checking/Installing Flask in venv...")
    try:
        subprocess.run([python_in_venv, "-m", "pip", "install", "Flask"], check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info("   Flask installed/verified.")
    except subprocess.CalledProcessError as e:
        logging.error(f"   Failed to install Flask.")
        logging.error(f"   Command: {' '.join(e.cmd)}")
        logging.error(f"   Stderr: {e.stderr}")
        sys.exit("Exiting due to Flask installation failure.")
    except FileNotFoundError:
        logging.error(f"   Error: Could not find '{python_in_venv}' to install Flask.")
        sys.exit("Exiting due to missing venv Python executable.")

    # --- Install Processing Dependencies ---
    logging.info(f"   Installing Processing dependencies from {PROCESSING_REQ_FILE}...")
    if not os.path.exists(PROCESSING_REQ_FILE):
        logging.warning(f"   Warning: Requirements file not found at {PROCESSING_REQ_FILE}. Skipping Processing dependencies.")
    else:
        try:
            install_cmd = [python_in_venv, "-m", "pip", "install", "-r", PROCESSING_REQ_FILE]
            logging.info(f"   Running: {' '.join(install_cmd)}")
            result = subprocess.run(install_cmd, check=False, capture_output=True, text=True, encoding='utf-8', timeout=600)

            if result.stdout:
                logging.info("   --- pip stdout ---")
                logging.info(result.stdout.strip())
                logging.info("   --- pip stdout END ---")
            if result.stderr:
                logging.warning("   --- pip stderr ---")
                logging.warning(result.stderr.strip())
                logging.warning("   --- pip stderr END ---")

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, install_cmd, output=result.stdout, stderr=result.stderr)

            logging.info(f"   Processing dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"   Failed to install dependencies from {PROCESSING_REQ_FILE}.")
            logging.error(f"   Command: {' '.join(e.cmd)}")
            sys.exit("Exiting due to Processing requirements installation failure.")
        except subprocess.TimeoutExpired:
            logging.error(f"   Timeout expired while installing Processing dependencies.")
            sys.exit("Exiting due to installation timeout.")
        except FileNotFoundError:
            logging.error(f"   Error: Could not find '{python_in_venv}' to install Processing requirements.")
            sys.exit("Exiting due to missing venv Python executable.")

    logging.info("--- Shared Virtual Environment Setup Complete ---")
    return python_in_venv


def run_server(python_exec):
    """Runs the server.py script using the specified Python interpreter."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(script_dir, "server.py")

    if not os.path.exists(server_script):
        logging.error(f"   Error: Cannot find server script at {server_script}")
        sys.exit("Server script missing.")

    logging.info(f"\n   Launching Flask server ({server_script}) using {python_exec}...")
    try:
        process = subprocess.Popen([python_exec, server_script])
        process.wait()
    except KeyboardInterrupt:
        logging.info("\n   Server stopped by user.")
    except Exception as e:
        logging.error(f"   Failed to run server: {e}")
        sys.exit("Server execution failed.")


if __name__ == "__main__":
    shared_venv_python = setup_shared_venv()
    run_server(shared_venv_python)