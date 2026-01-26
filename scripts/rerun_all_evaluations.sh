#!/bin/bash
# Re-run GPSBench evaluations for all models on new test splits
# Generated after data fixes and split regeneration

set -e

echo "=========================================="
echo "GPSBench Re-evaluation Script"
echo "=========================================="
echo ""
echo "This script will re-run evaluations for all 15 models"
echo "on the newly regenerated test splits."
echo ""
echo "Estimated time: Several hours (depending on API rate limits)"
echo "Estimated cost: Check with your API provider"
echo ""

# Create a log directory
LOG_DIR="logs/rerun_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Function to run benchmark for a model
run_model() {
    local provider=$1
    local model=$2
    local log_file="$LOG_DIR/${model//\//_}.log"

    echo "----------------------------------------"
    echo "Running: $model (provider: $provider)"
    echo "Log: $log_file"
    echo "----------------------------------------"

    python run_benchmark.py \
        --provider "$provider" \
        --model "$model" \
        --track both \
        --concurrent \
        --max-workers 5 \
        2>&1 | tee "$log_file"

    echo ""
    echo "Completed: $model"
    echo ""
}

# OpenAI models
echo "=== OpenAI Models ==="
run_model "openai" "gpt-4.1"
run_model "openai" "gpt-4.1-mini"
run_model "openai" "gpt-5.1"
run_model "openai" "gpt-5-mini-2025-08-07"
run_model "openai" "gpt-5-nano-2025-08-07"

# Google Gemini models (auto-detected)
echo "=== Gemini Models ==="
run_model "gemini" "gemini-2.5-flash"
run_model "gemini" "gemini-2.5-pro"
run_model "gemini" "gemini-3-pro-preview"

# OpenRouter models (Anthropic, Mistral, Qwen)
echo "=== OpenRouter Models ==="
run_model "openrouter" "anthropic/claude-haiku-4.5"
run_model "openrouter" "mistralai/mistral-large-2512"
run_model "openrouter" "mistralai/mistral-small-24b-instruct-2501"
run_model "openrouter" "qwen/qwen3-235b-a22b-2507"
run_model "openrouter" "qwen/qwen3-30b-a3b-instruct-2507"
# Note: qwen/qwen3-8b was not previously run successfully

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to: results/"
echo "Logs saved to: $LOG_DIR"
echo "=========================================="
