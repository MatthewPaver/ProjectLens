PYTHON ?= python3.11
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python
PIP_BIN := $(PYTHON_BIN) -m pip

.PHONY: venv install install-rag pipeline public-data test browser-test precedent-rag precedent-eval

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP_BIN) install --upgrade pip
	$(PIP_BIN) install -r requirements.txt

pipeline: install
	$(PYTHON_BIN) Processing/main.py

public-data: install
	$(PYTHON_BIN) Processing/gmpp_pipeline.py

test: install
	$(PYTHON_BIN) -m pytest Processing/tests -q

browser-test: install
	$(PYTHON_BIN) -m playwright install chromium
	$(PYTHON_BIN) scripts/run_browser_tests.py


install-rag: install
	$(PIP_BIN) install -r requirements-rag.txt

# Local sidecar — XER stays in the browser; only narrative/filters are posted.
precedent-rag: install-rag
	PYTHONPATH=. $(PYTHON_BIN) -m Processing.precedent_rag.server

precedent-eval: install-rag
	PYTHONPATH=. $(PYTHON_BIN) -m Processing.precedent_rag.cli eval
