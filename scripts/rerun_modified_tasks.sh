#!/bin/bash
# Re-run GPSBench evaluations ONLY for modified tasks
# Results are saved to existing result directories
#
# Usage:
#   ./rerun_modified_tasks.sh                  # Run all models
#   ./rerun_modified_tasks.sh 1                # Run by number
#   ./rerun_modified_tasks.sh gpt-4.1          # Run by name (partial match)
#   ./rerun_modified_tasks.sh gemini           # Run all gemini models
#   ./rerun_modified_tasks.sh qwen             # Run all qwen models
#
# Tasks requiring re-evaluation:
#   - task1_format_conversion (added 67 samples)
#   - task3_coordinate_system_transformation (added 67 samples)
#   - task7_area_perimeter (added 13 samples)
#   - task8_boundary_analysis (100% regenerated)
#   - task11_missing_data (added 87 samples)
#   - task13_terrain_classification (format change)

set -e

# Tasks to re-run (partial name matching)
TASKS="format_conversion coordinate_system_transformation area_perimeter boundary_analysis missing_data terrain_classification"

# Define all models: "provider|model|folder|use_batch"
MODELS=(
    "openai|gpt-4.1|gpsbench_gpt-4.1_20251201_235326|true"
    "openai|gpt-4.1-mini|gpsbench_gpt-4.1-mini_20251201_221915|true"
    "openai|gpt-5.1|gpsbench_gpt-5.1_20251202_114655|true"
    "openai|gpt-5-mini-2025-08-07|gpsbench_gpt-5-mini-2025-08-07_20251205_142538|true"
    "openai|gpt-5-nano-2025-08-07|gpsbench_gpt-5-nano-2025-08-07_20251205_162633|true"
    "gemini|gemini-2.5-flash|gpsbench_gemini-2.5-flash_20251228_065342|true"
    "gemini|gemini-2.5-pro|gpsbench_gemini-2.5-pro_20251210_182109|true"
    "gemini|gemini-3-pro-preview|gpsbench_gemini-3-pro-preview_20251210_103351|true"
    "openrouter|anthropic/claude-haiku-4.5|gpsbench_anthropic_claude-haiku-4.5_20260101_181018|false"
    "openrouter|mistralai/mistral-large-2512|gpsbench_mistralai_mistral-large-2512_20251228_094842|false"
    "openrouter|mistralai/mistral-small-24b-instruct-2501|gpsbench_mistralai_mistral-small-24b-instruct-2501_20251228_174803|false"
    "openrouter|qwen/qwen3-235b-a22b-2507|gpsbench_qwen_qwen3-235b-a22b-2507_20251205_015659|false"
    "openrouter|qwen/qwen3-30b-a3b-instruct-2507|gpsbench_qwen_qwen3-30b-a3b-instruct-2507_20251203_194935|false"
)

show_models() {
    echo "Available models:"
    echo ""
    for i in "${!MODELS[@]}"; do
        IFS='|' read -r provider model folder <<< "${MODELS[$i]}"
        printf "  %2d) %-45s [%s]\n" "$((i+1))" "$model" "$provider"
    done
    echo ""
    echo "Usage:"
    echo "  ./rerun_modified_tasks.sh              # Run all models"
    echo "  ./rerun_modified_tasks.sh 1            # Run model #1"
    echo "  ./rerun_modified_tasks.sh gpt-4.1      # Run by name (exact or partial)"
    echo "  ./rerun_modified_tasks.sh gemini       # Run all matching models"
    echo "  ./rerun_modified_tasks.sh openai       # Run all OpenAI models"
}

run_model() {
    local provider=$1
    local model=$2
    local folder=$3
    local use_batch=$4

    echo "----------------------------------------"
    echo "Running: $model"
    echo "Provider: $provider"
    echo "Update folder: $folder"
    echo "Batch mode: $use_batch"
    echo "Tasks: $TASKS"
    echo "----------------------------------------"

    if [ "$use_batch" = "true" ]; then
        # Use batch mode for OpenAI and Gemini (50% cost savings)
        python run_benchmark.py \
            --provider "$provider" \
            --model "$model" \
            --track both \
            --tasks $TASKS \
            --update-existing "$folder" \
            --use-batch
    else
        # Use concurrent mode for OpenRouter
        python run_benchmark.py \
            --provider "$provider" \
            --model "$model" \
            --track both \
            --tasks $TASKS \
            --update-existing "$folder" \
            --concurrent \
            --max-workers 5
    fi

    echo ""
    echo "Completed: $model -> $folder"
    echo ""
}

# Main logic
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_models
    exit 0
fi

echo "=========================================="
echo "GPSBench Re-evaluation - Modified Tasks Only"
echo "=========================================="
echo ""

LOG_DIR="logs/rerun_modified_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Determine which models to run
models_to_run=()

if [ -z "$1" ]; then
    # No argument: run all models
    echo "Running ALL models..."
    models_to_run=("${MODELS[@]}")
elif [[ "$1" =~ ^[0-9]+$ ]]; then
    # Numeric argument: run specific model by index
    idx=$(($1 - 1))
    if [ $idx -ge 0 ] && [ $idx -lt ${#MODELS[@]} ]; then
        models_to_run=("${MODELS[$idx]}")
    else
        echo "Error: Invalid model number $1 (valid: 1-${#MODELS[@]})"
        show_models
        exit 1
    fi
else
    # String argument: match by name or provider
    pattern="$1"
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r provider model folder <<< "$entry"
        if [[ "$model" == *"$pattern"* ]] || [[ "$provider" == *"$pattern"* ]] || [[ "$folder" == *"$pattern"* ]]; then
            models_to_run+=("$entry")
        fi
    done

    if [ ${#models_to_run[@]} -eq 0 ]; then
        echo "Error: No models matching '$pattern'"
        echo ""
        show_models
        exit 1
    fi
fi

echo "Models to run: ${#models_to_run[@]}"
for entry in "${models_to_run[@]}"; do
    IFS='|' read -r provider model folder use_batch <<< "$entry"
    batch_str=""
    if [ "$use_batch" = "true" ]; then
        batch_str=" [batch]"
    fi
    echo "  - $model ($provider)$batch_str"
done
echo ""

# Run selected models
for entry in "${models_to_run[@]}"; do
    IFS='|' read -r provider model folder use_batch <<< "$entry"
    log_file="$LOG_DIR/${model//\//_}.log"
    run_model "$provider" "$model" "$folder" "$use_batch" 2>&1 | tee "$log_file"
done

echo "=========================================="
echo "Evaluations completed!"
echo "Results updated in existing directories"
echo "Logs saved to: $LOG_DIR"
echo "=========================================="
