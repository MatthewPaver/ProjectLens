# ProjectLens

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

**Project schedule analysis application**

Upload project data, run the pipeline, and review risk-oriented output through a lightweight Flask interface.

</div>

![ProjectLens overview](Portfolio%20Overview%20Thumbnail.png)

---

## Portfolio Quick Read

| Section | Where to look |
|:---|:---|
| What it solves | Turns project schedule files into delivery-risk reporting and dashboard outputs |
| Quick start | [`make serve`](#quick-start) or [`make pipeline`](#running-the-pipeline-directly) |
| Screenshot | [Portfolio Store](https://matthewpaver.github.io/MatthewPaver/store/) |
| Architecture | [What The Pipeline Does](#what-the-pipeline-does) |
| Tests | `make test` |
| Tech stack | `Python` `Flask` `pandas` `statsforecast` `Power BI` |

## Status

`Runnable application`

ProjectLens combines:

- a Flask web interface in [`Website/`](Website)
- a data processing pipeline in [`Processing/`](Processing)
- structured input/output folders under [`Data/`](Data)

## Reviewer Pack

| Area | Details |
|:---|:---|
| What it solves | Project schedule files become slippage, milestone, changepoint, forecast, and reporting outputs. |
| Screenshot | [Portfolio Store preview](https://matthewpaver.github.io/MatthewPaver/store/preview.html?app=projectlens) |
| Run locally | `make install && make serve`, then open `http://127.0.0.1:5000` |
| Pipeline only | `make pipeline` |
| Tests | `make test` |
| Demo data | Sample project folders and output examples are included under `Data/`. |
| Architecture | Flask upload UI -> file loader -> cleaning -> risk analysis modules -> CSV outputs -> Power BI artefacts |
| Limitations | Designed as a local portfolio application; production use would need auth, storage hardening, and deployment packaging. |

## Reviewer Notes

- **Reproducible path:** root `requirements.txt` and `make` targets are the supported setup route.
- **Product signal:** the Flask UI wraps the analysis pipeline so a reviewer can use the workflow, not just read code.
- **Analytics signal:** slippage, milestone pressure, changepoint detection, and forecasting are separated into pipeline outputs.
- **Verification path:** run `make test`, then `make serve` or `make pipeline` depending on whether you want the UI or batch flow.

## Canonical Setup

The canonical dependency entry point for this repo is the root [`requirements.txt`](requirements.txt).

### Option 1: Standard local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python Website/server.py
```

Open `http://127.0.0.1:5000`.

### Option 2: Convenience targets

```bash
make install
make serve
```

### Option 3: Bootstrap script

If you want the repo to create and manage `.venv` for you:

```bash
python3.11 Website/run_website.py
```

## Running The Pipeline Directly

```bash
python3.11 Processing/main.py
```

Or with the Makefile:

```bash
make pipeline
```

Results are written to `Data/output/<project_name>/`.

## What The Pipeline Does

- validates project input structure
- loads CSV and Excel source files
- runs slippage, milestone, changepoint, and forecasting analysis
- writes output files suitable for downstream review and Power BI use
- archives processed inputs

## Repository Layout

```text
Data/         input, output, archive, and schema assets
Processing/   pipeline logic and tests
Website/      Flask server, templates, and static assets
Output/       Power BI dashboard artefacts
```

## Notes

- Python `3.11` is the supported runtime for the packaged flows in this repo.
- Root `requirements.txt` is canonical.
- `Processing/requirements.txt` is kept only as a compatibility shim for older workflows.

## License

MIT. See [`LICENSE`](LICENSE).
