PYTHON ?= python3.11
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python
PIP_BIN := $(PYTHON_BIN) -m pip

.PHONY: venv install serve web pipeline test

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP_BIN) install --upgrade pip
	$(PIP_BIN) install -r requirements.txt

serve: install
	$(PYTHON_BIN) Website/server.py

web:
	$(PYTHON) Website/run_website.py

pipeline:
	$(PYTHON) Processing/main.py

test: install
	$(PYTHON_BIN) -m pytest Processing/tests -q
