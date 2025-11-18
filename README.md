# ProjectLens

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3670A0?style=flat&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=flat&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

**Web Application & Project Data Analysis Pipeline**

*Upload your project data, trigger analysis, and view results—all from your browser.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Usage](#-installation--usage)
- [Workflow](#-workflow)
- [Analysis Modules](#-analysis-modules)
- [Dependencies](#-dependencies)
- [FAQ](#-faq)

---

## 🎯 Overview

ProjectLens is a comprehensive web-based platform for analysing project schedule data, identifying risks, and generating actionable insights. Built with Flask, it provides an intuitive interface for uploading project data and automatically processes it through a robust backend pipeline that performs multiple types of analysis including slippage detection, change point analysis, milestone tracking, and forecasting.

### Key Capabilities

- **Web Interface**: Upload project files, trigger processing, and view analysis results through a clean, user-friendly interface
- **Automated Pipeline**: End-to-end data processing from ingestion to output generation
- **Multiple Analysis Types**: Slippage analysis, change point detection, milestone tracking, forecasting, and recommendations
- **Schema Validation**: JSON schema enforcement ensures data quality and standardisation
- **Power BI Integration**: Generated outputs are compatible with Power BI dashboards for advanced visualisation

---

## ✨ Features

### 🔍 Analysis Modules

- **Slippage Analysis** — Identifies tasks and projects that are behind schedule
- **Change Point Detection** — Uses the `ruptures` library to detect significant changes in project timelines
- **Milestone Analysis** — Tracks and evaluates milestone completion
- **Forecasting** — Time series forecasting using ARIMA models (via `pmdarima`)
- **Recommendation Generation** — Provides actionable insights based on analysis results

### 🏗️ Architecture

- **Frontend**: Flask-based web application with responsive UI
- **Backend Pipeline**: Modular processing stages (ingestion → cleaning → analysis → output)
- **Data Management**: Structured input/output directories with automatic archiving
- **Environment Management**: Shared virtual environment for both web and processing components

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MatthewPaver/ProjectLens.git
cd ProjectLens
```

### 2. Start the Web Application

```bash
python3.11 Website/run_website.py
```

**Important Notes:**
- Use `python3.11` directly (not `bash` or `.venv/bin/python`)
- The script automatically creates and manages the virtual environment
- All dependencies will be installed automatically

### 3. Access the Web Interface

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 4. Upload and Process Data

1. Upload your project data files (CSV/Excel) via the web interface
2. Trigger analysis for your uploaded project
3. View results in the web interface or download from `Data/output/<project_name>/`

---

## 📁 Project Structure

```
ProjectLens/
├── .venv/                 # Shared Python virtual environment (auto-managed)
├── Data/
│   ├── input/             # Upload project folders here (or via web interface)
│   ├── output/            # Generated analysis results (Power BI compatible)
│   ├── archive/           # Processed folders (success/failed)
│   └── schemas/           # JSON schemas for data validation
├── Processing/            # Backend data processing pipeline
│   ├── core/              # Core pipeline logic
│   ├── ingestion/         # Data loading module
│   ├── analysis/          # Analysis modules (slippage, changepoint, etc.)
│   ├── output/            # Output writing module
│   ├── logs/              # Pipeline execution logs
│   ├── tests/             # Unit and integration tests
│   ├── main.py            # Pipeline entry point
│   └── requirements.txt   # Processing dependencies
├── Website/               # Flask web application
│   ├── server.py          # Flask app and API endpoints
│   ├── run_website.py     # Main startup script
│   ├── static/            # CSS, JavaScript, images
│   └── templates/         # HTML templates
├── Output/
│   └── ProjectLens.pbix   # Power BI dashboard
└── README.md
```

---

## 🛠️ Prerequisites

### Required

- **Python 3.11** — Must be installed and available as `python3.11` in your terminal
  ```bash
  # macOS
  brew install python@3.11
  ```

### Optional (for some dependencies)

- **C/C++ Build Tools** — Required for compiling certain packages
  ```bash
  # macOS
  xcode-select --install
  ```

---

## 📦 Installation & Usage

### Primary Method: Web Interface

The recommended way to use ProjectLens is through the web interface:

1. **Start the server:**
   ```bash
   python3.11 Website/run_website.py
   ```

2. **Access the web interface** at `http://127.0.0.1:5000`

3. **Upload project data** via the web interface

4. **Trigger processing** through the UI

### Alternative Method: Manual Processing

For command-line processing:

1. **Place project folders** in `Data/input/`:
   ```
   Data/input/MyProject/task_updates.csv
   ```

2. **Run the pipeline manually:**
   ```bash
   python3.11 Processing/main.py
   ```

3. **Results** will appear in `Data/output/<project_name>/`

---

## 🔄 Workflow

<div align="center">

```
Upload Data → Validate Schema → Process Pipeline → Generate Output → Archive Input
     ↓              ↓                  ↓                ↓              ↓
  Web UI      JSON Schemas      Analysis Modules    CSV Files    Archive Folder
```

</div>

### Typical Steps

1. **Start the website** using `python3.11 Website/run_website.py`
2. **Upload project data** via the web interface (CSV/Excel files, one folder per project)
3. **Trigger analysis** for your uploaded project
4. **View/download results** from the web interface or in `Data/output/<project_name>/`
5. **Check logs** in `Processing/logs/pipeline.log` for troubleshooting

---

## 🔬 Analysis Modules

### Slippage Analysis
Identifies tasks and projects that are behind schedule by comparing planned vs. actual completion dates.

### Change Point Detection
Uses the `ruptures` library to detect significant changes in project timelines, helping identify when projects deviate from expected patterns.

### Milestone Analysis
Tracks milestone completion and evaluates progress against key project deliverables.

### Forecasting
Time series forecasting using ARIMA models (via `pmdarima`) to predict future project performance.

### Recommendation Generation
Provides actionable insights and recommendations based on analysis results.

---

## 📚 Dependencies

### Web Application
- **Flask** — Web framework

### Processing Pipeline
Key dependencies (see `Processing/requirements.txt`):
- `pandas` — Data manipulation
- `numpy` — Numerical computing
- `ruptures` — Change point detection
- `pmdarima` — Time series forecasting

All dependencies are automatically installed into the shared `.venv/` environment by `Website/run_website.py`.

---

## ❓ FAQ

### Q: I get `/opt/homebrew/bin/python3.11: cannot execute binary file`?

**A:** You tried to run the Python binary as a shell script. Always use:
```bash
python3.11 Website/run_website.py
```
Not `bash Website/run_website.py` or `.venv/bin/python Website/run_website.py`.

### Q: Where do I put my data?

**A:** Upload via the web interface (recommended), or place folders directly in `Data/input/`.

### Q: Where are the results?

**A:** Results are saved in `Data/output/<project_name>/` after processing. You can also view them through the web interface.

### Q: Can I use a different Python version?

**A:** ProjectLens requires Python 3.11 specifically for compatibility with TensorFlow and ARM architecture support.

### Q: How do I view the Power BI dashboard?

**A:** Open `Output/ProjectLens.pbix` in Power BI Desktop. The dashboard connects to data in `Data/output/`.

---

## 🎨 Screenshot

<div align="center">

![Portfolio Overview](https://github.com/MatthewPaver/ProjectLens/blob/main/Portfolio%20Overview%20Thumbnail.png?raw=true)

</div>

---

## 💻 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat&logo=powerbi&logoColor=000000)

</div>

---

## 📝 License

This project is provided as-is for demonstration and educational purposes.

---

## 🤝 Contributing

This is a personal project, but suggestions and feedback are welcome! Feel free to open an issue or submit a pull request.

---

<div align="center">

**Made with ❤️ for project management and data analysis**

[⬆ Back to Top](#projectlens)

</div>
