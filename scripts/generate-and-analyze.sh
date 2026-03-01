#!/bin/bash
# Generate mock logs and trigger koi to analyze them
set -e

cd "$(dirname "$0")/.."
mkdir -p mock/logs

# Generate logs with ~30% error rate for demo
python3 mock/generate_logs.py --count 50 --error-rate 0.30 > mock/logs/latest.log

echo "Generated $(wc -l < mock/logs/latest.log) log lines → mock/logs/latest.log"
