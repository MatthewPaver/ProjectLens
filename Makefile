PYTHON ?= python3.11
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python
PIP_BIN := $(PYTHON_BIN) -m pip

.PHONY: venv install install-browser install-rag pipeline public-data public-precedents test browser-test precedent-rag precedent-eval precedent-eval-offline

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP_BIN) install --upgrade pip
	$(PIP_BIN) install -r requirements.txt

pipeline: install
	$(PYTHON_BIN) Processing/main.py

public-data: install
	$(PYTHON_BIN) Processing/gmpp_pipeline.py

public-precedents:
	python3 Processing/precedent_rag/build_public_cases.py

test: install-rag
	$(PYTHON_BIN) -m pytest Processing/tests -q

install-browser: venv
	$(PIP_BIN) install -r requirements-browser.txt

browser-test: install-browser
	$(PYTHON_BIN) -m playwright install chromium
	$(PYTHON_BIN) scripts/run_browser_tests.py


install-rag: install
	$(PIP_BIN) install -r requirements-rag.txt

# Local sidecar — XER stays in the browser; only narrative/filters are posted.
precedent-rag: install-rag
	PYTHONPATH=. $(PYTHON_BIN) -m Processing.precedent_rag.server

precedent-eval: install-rag
	PYTHONPATH=. $(PYTHON_BIN) -m Processing.precedent_rag.cli eval

# Deterministic keyless baseline: hashing embedder, no Gemini or LangSmith calls.
precedent-eval-offline: install-rag
	PYTHONPATH=. $(PYTHON_BIN) -m Processing.precedent_rag.cli eval --offline --limit 5
