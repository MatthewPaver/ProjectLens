# ProjectLens - Project Data Analysis Pipeline

This project provides a data processing pipeline to analyse project schedule data, identify potential risks, and generate insights. It loads data from input files, cleans it, performs various analyses (slippage, changepoint detection, milestone analysis, forecasting), and outputs structured CSV files suitable for further reporting (e.g., in Power BI).

## Features

*   **Modular Pipeline:** Organised into distinct stages: ingestion, cleaning, analysis, and output.
*   **Schema Enforcement:** Uses JSON schemas (`Data/schemas`) to validate and standardise input data.
*   **Multiple Analyses:** Includes modules for:
    *   Slippage Analysis
    *   Change Point Detection (using `ruptures` library)
    *   Milestone Analysis
    *   Forecasting (potentially using time series models like ARIMA via `pmdarima`)
    *   Recommendation Generation (basic placeholder)
*   **Structured Output:** Generates cleaned data and analysis results as CSV files in the `Data/output/<project_name>/` directory.
*   **Virtual Environment Setup:** Includes a script to automatically set up a `.venv` and install dependencies.

## Project Structure

```
ProjectLens/
├── Data/
│   ├── input/             # Place project folders (e.g., Alpha, Delta) here
│   │   ├── Alpha/
│   │   │   └── task_updates.csv
│   │   └── Delta/
│   │       └── task_updates.csv
│   ├── output/            # Generated analysis results
│   ├── archive/           # Input folders moved here after processing (success/failed)
│   └── schemas/           # JSON schemas for data validation (e.g., tasks.json)
├── Processing/
│   ├── core/              # Core pipeline logic (main_runner, data_pipeline, schema_manager, etc.)
│   ├── ingestion/         # Data loading module (file_loader)
│   ├── analysis/          # Analysis modules (slippage, changepoint, etc.)
│   ├── output/            # Output writing module (output_writer)
│   ├── logs/              # Pipeline execution logs (pipeline.log)
│   ├── tests/             # Unit and integration tests
│   ├── main.py            # Main entry point to set up venv and run pipeline
│   └── requirements.txt   # Python dependencies
└── README.md              # This file
```

## Setup and Execution

1.  **Prerequisites:**
    *   Python 3.11 (specifically required for TensorFlow/ARM compatibility checks in `main.py`, ensure `python3.11` is in your PATH). You might install it via `brew install python@3.11` on macOS.
    *   Ensure you have the necessary C build tools if packages require compilation (common for scientific Python packages).

2.  **Prepare Input Data:**
    *   Place your project data folders (each containing relevant CSV/Excel files, e.g., `task_updates.csv`) inside the `Data/input/` directory.
    *   Example: `Data/input/MyProject/tasks.csv`

3.  **Run the Pipeline:**
    *   Navigate to the project root directory (`ProjectLens/`) in your terminal.
    *   Execute the main script:
        ```bash
        python Processing/main.py
        ```
    *   This script will:
        *   Check Python environment compatibility (especially for macOS ARM).
        *   Create a virtual environment (`.venv/`) using `python3.11` if it doesn't exist.
        *   Install/update dependencies from `Processing/requirements.txt` into the virtual environment.
        *   Execute the main pipeline logic (`Processing/core/main_runner.py`) using the virtual environment's Python interpreter.

4.  **Check Outputs:**
    *   Processed data and analysis results will be saved in `Data/output/<project_name>/`.
    *   Logs can be found in `Processing/logs/pipeline.log`.
    *   The original project folder from `Data/input/` will be moved to `Data/archive/success/` or `Data/archive/failed/`.

## Dependencies

Key dependencies are listed in `Processing/requirements.txt` and include libraries like `pandas`, `numpy`, `ruptures`, `pmdarima`, and potentially `tensorflow`.

## Licence

[Specify Your Licence Here, e.g., MIT License]
