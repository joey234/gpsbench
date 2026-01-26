#!/bin/bash
# Re-run fixed tasks on all existing models
#
# Fixed tasks:
# - Place Association: Added Gaussian GPS noise levels (clean, 10m, 50m, 100m, 500m, 1000m)
#
# Usage:
#   ./rerun_fixed_tasks.sh           # Re-run on all models
#   ./rerun_fixed_tasks.sh --dry-run # Show what would be run without executing

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - Commands will be shown but not executed ==="
    echo ""
fi

# Tasks to re-run
TASKS="place_association"

echo "=============================================="
echo "Re-running Fixed Tasks on All Models"
echo "=============================================="
echo ""
echo "Tasks to re-run: $TASKS"
echo ""

# Define models - each line: PROVIDER|MODEL|FOLDER|REASONING_EFFORT
# Reasoning effort: minimal for gpt-5*, none for gpt-5.1, empty for others
CONFIGS=(
    "openai|gpt-4.1|gpsbench_gpt-4.1_20251201_235326|"
    "openai|gpt-4.1-mini|gpsbench_gpt-4.1-mini_20251201_221915|"
    "openai|gpt-5-mini-2025-08-07|gpsbench_gpt-5-mini-2025-08-07_20251205_142538|minimal"
    "openai|gpt-5-nano-2025-08-07|gpsbench_gpt-5-nano-2025-08-07_20251205_162633|minimal"
    "openai|gpt-5.1|gpsbench_gpt-5.1_20251202_114655|none"
    "openrouter|qwen/qwen3-235b-a22b-2507|gpsbench_qwen_qwen3-235b-a22b-2507_20251205_015659|"
    "openrouter|qwen/qwen3-30b-a3b-instruct-2507|gpsbench_qwen_qwen3-30b-a3b-instruct-2507_20251203_194935|"
)

for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r PROVIDER MODEL FOLDER REASONING <<< "$CONFIG"

    echo "----------------------------------------------"
    echo "Model: $MODEL"
    echo "Provider: $PROVIDER"
    echo "Folder: $FOLDER"
    echo "----------------------------------------------"

    # Build command based on provider
    if [[ "$PROVIDER" == "openai" ]]; then
        # OpenAI: use batch API
        CMD="python run_benchmark.py --provider $PROVIDER --model $MODEL --tasks $TASKS --update-existing $FOLDER --use-batch"
        # Add reasoning effort if specified
        if [[ -n "$REASONING" ]]; then
            CMD="$CMD --reasoning-effort $REASONING"
        fi
    else
        # OpenRouter: use concurrent mode
        CMD="python run_benchmark.py --provider $PROVIDER --model $MODEL --tasks $TASKS --update-existing $FOLDER --concurrent --max-workers 20"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "Would run: $CMD"
    else
        echo "Running: $CMD"
        $CMD
    fi
    echo ""
done

echo "=============================================="
echo "Done!"
echo "=============================================="
