#!/bin/bash

source .venv/bin/activate
cd src/benchmark/tool_plan_benchmarking
python run.py $1
