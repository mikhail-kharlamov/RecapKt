# Contributor guide (AGENTS)

## Project structure

- `src/` – main package code.
  - `src/summarize_algorithms/` – dialogue summarization implementations (e.g. `memory_bank/`, `recsum/`, shared `core/`).
  - `src/benchmarking/` – evaluation scripts, metrics, log parsing, and plotting.
  - `src/utils/` – small shared helpers (logging/config parsing).
  - Entry point: `src/main.py` (also exposed as a script in `pyproject.toml`: `recapkt = "src.main:main"`).
- `tests/` – pytest suite (files follow `test_*.py`).
- `requirements.txt`, `requirements.dev.txt` – runtime/dev dependencies.

## Build, test, and development commands

This repo targets **Python >= 3.12** (see `pyproject.toml`). CI uses **uv**.

- Create env + install deps (recommended):
  ```bash
  uv venv
  uv pip install -r requirements.txt -r requirements.dev.txt
  ```
- Run the example entry point:
  ```bash
  python -m src.main
  # or
  uv run recapkt
  ```
- Lint / format (Ruff):
  ```bash
  ruff check .
  ruff format .
  ```
- Type-check (Mypy):
  ```bash
  uv run mypy
  ```
- Run tests:
  ```bash
  uv run python -m pytest
  ```
- Tool-metrics benchmarking helper:
  ```bash
  ./run.sh <arg>   # runs src/benchmark/tool_plan_benchmarking/run.py
  ```

## Code style and naming

- Formatting/linting: Ruff is the source of truth (line length **120**, double quotes).
- Typing: keep functions typed; the project configuration disallows untyped defs in `src/`.
- Naming:
  - modules/files: `snake_case.py`
  - classes: `CamelCase`
  - tests: `tests/test_<unit>.py`, test functions `test_<behavior>()`

## VCS: commits and pull requests

- Commit messages follow a lightweight Conventional Commits style seen in history: `feat: ...`, `fix: ...`.
- PRs should:
  - describe the change + rationale,
  - include how to reproduce/verify (commands or a minimal snippet),
  - keep CI green (GitHub Actions runs `ruff check`, `mypy`, `pytest` on PRs).

## Secrets and local config

- Don’t commit `.env`. If your change needs new settings, document them and keep defaults safe.
