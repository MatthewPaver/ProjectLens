# ProjectLens

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3670A0?style=flat&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

**Project schedule analysis application**

Upload project data, run the pipeline, and review risk-oriented output through a lightweight Flask interface.

</div>

![ProjectLens overview](Portfolio%20Overview%20Thumbnail.png)

---

## Status

`Runnable application`

ProjectLens combines:

- a Flask web interface in [`Website/`](Website)
- a data processing pipeline in [`Processing/`](Processing)
- structured input/output folders under [`Data/`](Data)

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
