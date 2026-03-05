#!/bin/bash
# EdgeDeploy-Inference Benchmark Executor

echo "=========================================================="
echo "Starting EdgeDeploy Cross-Platform Inference Benchmarking"
echo "=========================================================="

# Auto-detect path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_BIN=$(which python3 || which python)

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python 3 is not installed or not in PATH."
    exit 1
fi

echo "Running full latency, accuracy, and power benchmark suite..."
$PYTHON_BIN "$SCRIPT_DIR/benchmark_suite.py" --platform auto --runs 100

echo "Benchmarking complete."
