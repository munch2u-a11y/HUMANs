PYTHON ?= python3
VENV ?= .venv
MIND_DATABASE ?= state/habitus.sqlite
WORKSPACE ?= workspace
MODEL ?= qwen3.5:2b
OLLAMA_URL ?= http://127.0.0.1:11434
HUMAN_NAME ?= Human
AGENT_NAME ?= Habitus
LOCAL_PYTHON = PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(CURDIR)/src" "$(VENV)/bin/python"

.DEFAULT_GOAL := help

.PHONY: help setup doctor portability docs playground test verify run smoke clean-local

help:
	@echo "Habitus Mind developer commands"
	@echo "  make setup        Create .venv and install test + cortex dependencies"
	@echo "  make playground   Exercise conversation, memory, read, and run offline"
	@echo "  make test         Run the complete deterministic CPU test suite"
	@echo "  make verify       Run portability, docs, playground, and tests"
	@echo "  make doctor       Check the local Ollama setup without changing state"
	@echo "  make smoke        Run one live Ollama-backed turn and print JSON"
	@echo "  make run          Open the persistent interactive runtime"
	@echo ""
	@echo "Override paths and identity with MIND_DATABASE, WORKSPACE, MODEL,"
	@echo "OLLAMA_URL, HUMAN_NAME, and AGENT_NAME. All defaults are repo-relative."

setup:
	$(PYTHON) -m venv $(VENV)
	"$(VENV)/bin/python" -m pip install --upgrade pip
	"$(VENV)/bin/python" -m pip install -e '.[test,cortex]'

doctor:
	$(LOCAL_PYTHON) -m habitus_ai.doctor --model "$(MODEL)" --ollama-url "$(OLLAMA_URL)"

portability:
	$(PYTHON) scripts/check_portability.py

docs:
	$(PYTHON) scripts/check_docs.py

playground:
	$(LOCAL_PYTHON) examples/api_playground.py

test:
	$(LOCAL_PYTHON) -m pytest -ra

verify: portability docs playground test

run:
	mkdir -p "$(WORKSPACE)"
	$(LOCAL_PYTHON) -m habitus_ai.integrated_agent --database "$(MIND_DATABASE)" --workspace "$(WORKSPACE)" --model "$(MODEL)" --ollama-url "$(OLLAMA_URL)" --human-name "$(HUMAN_NAME)" --agent-name "$(AGENT_NAME)"

smoke:
	mkdir -p "$(WORKSPACE)"
	$(LOCAL_PYTHON) -m habitus_ai.integrated_agent --database "$(MIND_DATABASE)" --workspace "$(WORKSPACE)" --model "$(MODEL)" --ollama-url "$(OLLAMA_URL)" --human-name "$(HUMAN_NAME)" --agent-name "$(AGENT_NAME)" --once "Hello" --json

clean-local:
	@echo "Local minds and workspaces are intentionally not deleted automatically."
