# Dr-Titan BioNeMo Web Console — common tasks.
.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= python3
PORT ?= 8501
BACKEND ?= auto

.PHONY: help setup run test lint demo docker compose nims fetch clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv, install deps, vendor the 3D viewer
	bash scripts/setup.sh

run: ## Start the console (PORT=8501 BACKEND=auto by default)
	PORT=$(PORT) BIONEMO_BACKEND=$(BACKEND) bash scripts/run_console.sh

test: ## Run the test suite
	PYTHONPATH=. $(PY) -m pytest -q

lint: ## Syntax-check every module (no external linter needed)
	PYTHONPATH=. $(PY) -m compileall -q app.py bionemo_web tests && echo "compileall OK"

demo: ## Run the console in pure simulation mode (no GPU, no keys needed)
	PORT=$(PORT) BIONEMO_BACKEND=simulate bash scripts/run_console.sh

nims: ## Start local NIM containers (needs NGC_API_KEY + GPU)
	bash scripts/start_nims.sh $(NIM)

fetch: ## Clone the BioNeMo Framework source for reference into third_party/
	bash scripts/fetch_bionemo.sh

docker: ## Build the console image
	docker build -t dr-titan-console:1.0.0 .

compose: ## Bring the stack up with docker compose
	docker compose up --build

clean: ## Remove caches and runtime state (keeps the virtualenv)
	rm -rf .runtime .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
