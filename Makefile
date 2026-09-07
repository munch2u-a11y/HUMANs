PYTHON ?= python3
VENV ?= .venv
MIND_DATABASE ?= state/habitus.sqlite
WORKSPACE ?= workspace

.PHONY: setup test run smoke clean-local

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e '.[test,cortex]'

test:
	$(VENV)/bin/python -m pytest

run:
	mkdir -p state $(WORKSPACE)
	$(VENV)/bin/habitus-mind --database $(MIND_DATABASE) --workspace $(WORKSPACE)

smoke:
	mkdir -p state $(WORKSPACE)
	$(VENV)/bin/habitus-mind --database $(MIND_DATABASE) --workspace $(WORKSPACE) --once "Hello" --json

clean-local:
	@echo "Local minds and workspaces are intentionally not deleted automatically."
