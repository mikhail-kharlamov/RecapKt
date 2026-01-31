.PHONY: help run-main run-tool-plan run-tool-metrics run-tool-metrics-sh test

# Prefer local venv if present, fall back to system python.
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || (command -v python3 >/dev/null 2>&1 && echo python3 || echo python))

# Optional args for some targets:
#   make run-tool-metrics ARG=base_recsum
ARG ?= base_recsum

help:
	@echo "Available targets:"
	@echo "  make run-main"
	@echo "  make run-tool-plan"
	@echo "  make run-tool-metrics ARG=<arg>"
	@echo "  make run-tool-metrics-sh ARG=<arg>"
	@echo "  make test"

run-main:
	$(PYTHON) -m src.main

run-tool-plan:
	cd src/benchmark/tool_plan_benchmarking && $(PYTHON) -m run.py

run-tool-metrics:
	$(MAKE) run-tool-plan

run-tool-metrics-sh:
	./run.sh $(ARG)

test:
	$(PYTHON) -m pytest -q
