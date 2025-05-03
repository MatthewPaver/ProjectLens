# ProjectLens - Web Application & Project Data Analysis Pipeline

ProjectLens provides a web interface and backend pipeline for analysing project schedule data, identifying risks, and generating actionable insights. Upload your project data, trigger the analysis, and view results—all from your browser.

---

<p align="center">
  <img src="https://github.com/MatthewPaver/ProjectLens/blob/main/Portfolio%20Overview%20Thumbnail.png?raw=true" alt="Portfolio Overview" width="1000"/>
</p>

## 🚀 Quick Start: How to Run the Website

1. **Open your terminal and navigate to the ProjectLens root directory:**
   ```bash
   cd /path/to/ProjectLens
   ```
   Replace `/path/to/ProjectLens` with the actual path where you have cloned or extracted the ProjectLens repository.

2. **Start the website using Python 3.11:**
   ```bash
   python3.11 Website/run_website.py
   ```
   - **Do NOT** use `bash` or try to run the script as a shell script.
   - If you see an error like `cannot execute binary file`, you likely used the wrong command. Always use `python3.11 Website/run_website.py`.
   - The script will automatically set up a virtual environment (`.venv/`) and install all dependencies if needed.

3. **Open your browser and go to:**
   - [http://127.0.0.1:5000](http://127.0.0.1:5000) (or the address shown in your terminal)

4. **Upload your project data** using the web interface and follow on-screen instructions to process and view results.

---

## 📁 Project Structure (Key Folders)

- `Website/` — Flask web app (frontend & API)
- `Processing/` — Backend data pipeline (analysis, cleaning, output)
- `Data/input/` — Place project folders here for processing (or upload via web)
- `Data/output/` — Results and analysis outputs
- `Data/archive/` — Processed input folders (success/failed)
- `.venv/` — Shared Python virtual environment (auto-managed)

---

## 🛠️ Prerequisites

- **Python 3.11** must be installed and available as `python3.11` in your terminal.
  - On macOS, you can install it with:
    ```bash
    brew install python@3.11
    ```
- **C/C++ build tools** may be required for some dependencies:
  - On macOS: `xcode-select --install`

---

## 📝 Typical Workflow

1. **Start the website** (see Quick Start above).
2. **Upload project data** via the web interface (CSV/Excel files, one folder per project).
3. **Trigger analysis** for your uploaded project.
4. **View/download results** from the web interface or in `Data/output/<project_name>/`.
5. **Check logs** in `Processing/logs/pipeline.log` if you need troubleshooting info.

---

## ⚙️ Advanced: Manual Data Processing

- You can place project folders directly in `Data/input/` and run the backend pipeline manually:
  ```bash
  python3.11 Processing/main.py
  ```
- Results will appear in `Data/output/` and processed folders will be moved to `Data/archive/`.

---

## 📦 Dependencies

- All dependencies are managed in `.venv/` and installed automatically by `Website/run_website.py`.
- Main libraries: `Flask`, `pandas`, `numpy`, `ruptures`, `pmdarima`, etc. (see `Processing/requirements.txt`)

---

## ❓ FAQ

- **Q: I get `/opt/homebrew/bin/python3.11: cannot execute binary file`?**
  - **A:** You tried to run the Python binary as a shell script. Use `python3.11 Website/run_website.py` (not `bash ...`).
- **Q: Where do I put my data?**
  - **A:** Upload via the web interface, or place folders in `Data/input/`.
- **Q: Where are results?**
  - **A:** In `Data/output/<project_name>/` after processing.

---

For more details, see comments in the code and the rest of this README.

---

## Features

*   **Web Interface:** Allows users to upload project files, trigger processing, and potentially view analysis results (built with Flask).
*   **Backend Pipeline:** Organised into distinct stages: ingestion, cleaning, analysis, and output.
*   **Schema Enforcement:** Uses JSON schemas (`Data/schemas`) to validate and standardise input data during processing.
*   **Multiple Analyses:** Includes modules for:
    *   Slippage Analysis
    *   Change Point Detection (using `ruptures` library)
    *   Milestone Analysis
    *   Forecasting (potentially using time series models like ARIMA via `pmdarima`)
    *   Recommendation Generation (basic placeholder)
*   **Shared Virtual Environment:** A single virtual environment (`.venv/`) at the project root manages dependencies for both the Website and Processing components, set up via the main run script.
*   **Structured Output:** Generates cleaned data and analysis results as CSV files in the `Data/output/<project_name>/` directory.

## Project Structure

```
ProjectLens/
├── .venv/                 # Shared Python virtual environment
├── Data/
│   ├── input/             # Place project folders (e.g., Alpha, Delta) here for processing
│   │   ├── Alpha/
│   │   │   └── task_updates.csv
│   │   └── Delta/
│   │       └── task_updates.csv
│   ├── output/            # Generated analysis results
│   ├── archive/           # Input folders moved here after processing (success/failed)
│   └── schemas/           # JSON schemas for data validation (e.g., tasks.json)
├── Processing/            # Backend data processing modules
│   ├── core/              # Core pipeline logic (main_runner, data_pipeline, etc.)
│   ├── ingestion/         # Data loading module
│   ├── analysis/          # Analysis modules (slippage, changepoint, etc.)
│   ├── output/            # Output writing module
│   ├── logs/              # Pipeline execution logs (pipeline.log)
│   ├── tests/             # Unit and integration tests
│   ├── main.py            # Entry point for the processing pipeline (usually run via Website)
│   └── requirements.txt   # Python dependencies for the Processing backend
├── Website/
│   ├── server.py          # Flask application logic and API endpoints
│   ├── run_website.py     # Main script to set up venv and run the web server
│   ├── static/            # CSS, JavaScript, images for the website
│   └── templates/         # HTML templates for the website
└── README.md              # This file
```

## Setup and Execution

**The primary way to use ProjectLens is through the web interface.**

1.  **Run the Web Application:**
    *   Navigate to the project root directory (`ProjectLens/`) in your terminal.
    *   Execute the main website run script **using your system's `python3.11` interpreter**:
        ```bash
        python3.11 Website/run_website.py
        ```
    *   **Important:** Do *not* try to run this script using the Python executable from within the `.venv` directory (e.g., `.venv/bin/python Website/run_website.py`). The script itself is responsible for creating and managing the virtual environment.
    *   This script will:
        *   Check Python environment compatibility (requires Python 3.11).
        *   Create a shared virtual environment (`.venv/` at the project root) using `python3.11` if it doesn't exist, or verify an existing one.
        *   Install/update dependencies for both the Website (Flask) and the Processing backend (from `Processing/requirements.txt`) into the shared virtual environment.
        *   Launch the Flask web server.

2.  **Access the Web Interface:**
    *   Open your web browser and navigate to the address provided by Flask (usually `http://127.0.0.1:5000` or similar).
    *   Use the interface to upload new project data.

3.  **Trigger Processing:**
    *   The web interface should provide a mechanism (e.g., a button after upload) to trigger the backend processing pipeline for a specific project.
    *   This typically involves the frontend calling the `/api/process/<project_name>` endpoint on the Flask server, which in turn executes `Processing/main.py`.

4.  **Check Outputs & Logs:**
    *   Processed data and analysis results will be saved in `Data/output/<project_name>/`.
    *   Pipeline execution logs can be found in `Processing/logs/pipeline.log`.
    *   The original project folder from `Data/input/` (if processing was triggered for it) will be moved to `Data/archive/success/` or `Data/archive/failed/`.

**Additional Information:**

*   **Prerequisites:**
    *   Python 3.11 (specifically required for the setup script and TensorFlow/ARM compatibility checks, ensure `python3.11` is available in your PATH). You might install it via `brew install python@3.11` on macOS.
    *   Ensure you have the necessary C/C++ build tools if packages require compilation (e.g., run `xcode-select --install` on macOS). This might be needed for libraries like `numpy` or `ruptures`.

*   **Prepare Input Data (Optional - for initial processing):**
    *   If you want to process data *before* starting the web server (less common), you can place your project data folders (each containing relevant CSV/Excel files, e.g., `task_updates.csv`) inside the `Data/input/` directory.
    *   Example: `Data/input/MyProject/tasks.csv`
    *   However, the standard workflow is to upload files via the web interface once it's running.

## Dependencies

*   **Website:** Flask
*   **Processing:** Key dependencies are listed in `Processing/requirements.txt` and include libraries like `pandas`, `numpy`, `ruptures`, `pmdarima`, etc.

All dependencies are installed into the shared `.venv/` environment by the `Website/run_website.py` script.
