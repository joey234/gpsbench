#!/bin/bash
# Single-task training: Train on one specific task from split files

set -e

# Default task
TASK="${1:-task1_format_conversion}"
TRACK="${2:-pure_gps}"

echo "=========================================="
echo "GPSBench Single-Task Training with Tinker"
echo "=========================================="
echo ""
echo "Task: $TASK"
echo "Track: $TRACK"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Please run: ./setup_api_keys.sh"
    exit 1
fi

# Check if API keys are set
source .env 2>/dev/null || true

if [ -z "$TINKER_API_KEY" ] || [ "$TINKER_API_KEY" = "your_tinker_api_key_here" ]; then
    echo "Error: TINKER_API_KEY not set in .env file"
    exit 1
fi

echo "✓ API keys verified"
echo ""

# Run single-task training
echo "Starting single-task training..."
echo "This will train only on $TASK from $TRACK track"
echo ""

python train.py \
    dataset_mode=split_files \
    track=$TRACK \
    tasks="['$TASK']" \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    learning_rate=5e-4 \
    batch_size=32 \
    lora_rank=64 \
    num_epochs=5 \
    eval_every=100 \
    save_every=200 \
    log_path="./outputs/single_task_${TASK}"

echo ""
echo "=========================================="
echo "Single-task training complete!"
echo "=========================================="
echo ""
echo "Model saved to: ./outputs/single_task_${TASK}"
echo ""
echo "Usage:"
echo "  ./run_single_task.sh <task_name> <track>"
echo "  Example: ./run_single_task.sh task1_format_conversion pure_gps"
echo "  Example: ./run_single_task.sh task1_place_association applied"
