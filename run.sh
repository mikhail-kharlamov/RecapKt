#!/bin/bash

source .venv/bin/activate
cd src/benchmarking/tool_metrics
python run.py $1
