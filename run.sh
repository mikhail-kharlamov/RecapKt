#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR"

cd "$ROOT_DIR/src/benchmark/tool_plan_benchmarking"
python run.py "${1:-}"
