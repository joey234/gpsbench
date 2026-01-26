#!/bin/bash
# Run GPSBench on additional Qwen models via OpenRouter

set -e

echo "========================================"
echo "Running GPSBench on Qwen3-14B"
echo "========================================"
python run_benchmark.py \
    --provider openrouter \
    --model qwen/qwen3-14b \
    --concurrent \
    --max-workers 20 \
    --track both

echo ""
echo "========================================"
echo "Running GPSBench on Qwen3-8B"
echo "========================================"
python run_benchmark.py \
    --provider openrouter \
    --model qwen/qwen3-8b \
    --concurrent \
    --max-workers 20 \
    --track both

echo ""
echo "========================================"
echo "All Qwen models completed!"
echo "========================================"
