#!/bin/bash
# Run GPSBench evaluation for a single model (modified tasks only)
# Results saved to existing directory
#
# Usage: ./run_single_model.sh <model_number>
#
# Available models:
#   1  - gpt-4.1
#   2  - gpt-4.1-mini
#   3  - gpt-5.1
#   4  - gpt-5-mini-2025-08-07
#   5  - gpt-5-nano-2025-08-07
#   6  - gemini-2.5-flash
#   7  - gemini-2.5-pro
#   8  - gemini-3-pro-preview
#   9  - anthropic/claude-haiku-4.5
#   10 - mistralai/mistral-large-2512
#   11 - mistralai/mistral-small-24b-instruct-2501
#   12 - qwen/qwen3-235b-a22b-2507
#   13 - qwen/qwen3-30b-a3b-instruct-2507

set -e

# Tasks to re-run (partial name matching)
TASKS="format_conversion coordinate_system_transformation area_perimeter boundary_analysis missing_data terrain_classification"

if [ -z "$1" ]; then
    echo "Usage: ./run_single_model.sh <model_number>"
    echo ""
    echo "Available models:"
    echo "  1  - gpt-4.1"
    echo "  2  - gpt-4.1-mini"
    echo "  3  - gpt-5.1"
    echo "  4  - gpt-5-mini-2025-08-07"
    echo "  5  - gpt-5-nano-2025-08-07"
    echo "  6  - gemini-2.5-flash"
    echo "  7  - gemini-2.5-pro"
    echo "  8  - gemini-3-pro-preview"
    echo "  9  - anthropic/claude-haiku-4.5"
    echo "  10 - mistralai/mistral-large-2512"
    echo "  11 - mistralai/mistral-small-24b-instruct-2501"
    echo "  12 - qwen/qwen3-235b-a22b-2507"
    echo "  13 - qwen/qwen3-30b-a3b-instruct-2507"
    exit 1
fi

case $1 in
    1)  PROVIDER="openai"; MODEL="gpt-4.1"; FOLDER="gpsbench_gpt-4.1_20251201_235326"; USE_BATCH=true ;;
    2)  PROVIDER="openai"; MODEL="gpt-4.1-mini"; FOLDER="gpsbench_gpt-4.1-mini_20251201_221915"; USE_BATCH=true ;;
    3)  PROVIDER="openai"; MODEL="gpt-5.1"; FOLDER="gpsbench_gpt-5.1_20251202_114655"; USE_BATCH=true ;;
    4)  PROVIDER="openai"; MODEL="gpt-5-mini-2025-08-07"; FOLDER="gpsbench_gpt-5-mini-2025-08-07_20251205_142538"; USE_BATCH=true ;;
    5)  PROVIDER="openai"; MODEL="gpt-5-nano-2025-08-07"; FOLDER="gpsbench_gpt-5-nano-2025-08-07_20251205_162633"; USE_BATCH=true ;;
    6)  PROVIDER="gemini"; MODEL="gemini-2.5-flash"; FOLDER="gpsbench_gemini-2.5-flash_20251228_065342"; USE_BATCH=true ;;
    7)  PROVIDER="gemini"; MODEL="gemini-2.5-pro"; FOLDER="gpsbench_gemini-2.5-pro_20251210_182109"; USE_BATCH=true ;;
    8)  PROVIDER="gemini"; MODEL="gemini-3-pro-preview"; FOLDER="gpsbench_gemini-3-pro-preview_20251210_103351"; USE_BATCH=true ;;
    9)  PROVIDER="openrouter"; MODEL="anthropic/claude-haiku-4.5"; FOLDER="gpsbench_anthropic_claude-haiku-4.5_20260101_181018"; USE_BATCH=false ;;
    10) PROVIDER="openrouter"; MODEL="mistralai/mistral-large-2512"; FOLDER="gpsbench_mistralai_mistral-large-2512_20251228_094842"; USE_BATCH=false ;;
    11) PROVIDER="openrouter"; MODEL="mistralai/mistral-small-24b-instruct-2501"; FOLDER="gpsbench_mistralai_mistral-small-24b-instruct-2501_20251228_174803"; USE_BATCH=false ;;
    12) PROVIDER="openrouter"; MODEL="qwen/qwen3-235b-a22b-2507"; FOLDER="gpsbench_qwen_qwen3-235b-a22b-2507_20251205_015659"; USE_BATCH=false ;;
    13) PROVIDER="openrouter"; MODEL="qwen/qwen3-30b-a3b-instruct-2507"; FOLDER="gpsbench_qwen_qwen3-30b-a3b-instruct-2507_20251203_194935"; USE_BATCH=false ;;
    *)
        echo "Invalid model number: $1"
        exit 1
        ;;
esac

echo "=========================================="
echo "Running GPSBench evaluation"
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Update folder: $FOLDER"
echo "Tasks: $TASKS"
echo "Batch mode: $USE_BATCH"
echo "=========================================="

if [ "$USE_BATCH" = true ]; then
    # Use batch mode for OpenAI and Gemini (50% cost savings)
    python run_benchmark.py \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --track both \
        --tasks $TASKS \
        --update-existing "$FOLDER" \
        --use-batch
else
    # Use concurrent mode for OpenRouter
    python run_benchmark.py \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --track both \
        --tasks $TASKS \
        --update-existing "$FOLDER" \
        --concurrent \
        --max-workers 5
fi

echo ""
echo "Evaluation completed for: $MODEL"
echo "Results updated in: results/$FOLDER"
