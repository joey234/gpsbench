#!/bin/bash
# Medium training for place_association task
# Good balance between speed and seeing real learning progress
# ~15-20 minutes, 500 samples, 3 epochs

set -e

echo "=========================================="
echo "GPSBench Place Association - MEDIUM RUN"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Please run: ./setup_api_keys.sh"
    exit 1
fi

source .env 2>/dev/null || true

if [ -z "$TINKER_API_KEY" ] || [ "$TINKER_API_KEY" = "your_tinker_api_key_here" ]; then
    echo "Error: TINKER_API_KEY not set in .env file"
    exit 1
fi

echo "API keys verified"
echo ""

# Configuration
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-50}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LORA_RANK="${LORA_RANK:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

# Calculate steps
steps_per_epoch=$((MAX_SAMPLES / BATCH_SIZE))
total_steps=$((steps_per_epoch * NUM_EPOCHS))

echo "Medium run configuration:"
echo "  Model: $MODEL"
echo "  Train samples: $MAX_SAMPLES"
echo "  Eval samples: $MAX_EVAL_SAMPLES"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size: $BATCH_SIZE"
echo "  LoRA rank: $LORA_RANK"
echo "  Epochs: $NUM_EPOCHS"
echo "  Steps per epoch: ~$steps_per_epoch"
echo "  Total steps: ~$total_steps"
echo "  Estimated time: 15-20 minutes"
echo ""
echo "Expected results:"
echo "  Loss: should drop from ~1.3 to ~0.6-0.8"
echo "  Accuracy: should reach 10-30%"
echo ""

read -p "Start medium run? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "Starting medium training..."
echo ""

python train.py \
    filter_tasks="['place_association']" \
    model_name="$MODEL" \
    max_samples=$MAX_SAMPLES \
    max_eval_samples=$MAX_EVAL_SAMPLES \
    learning_rate=$LEARNING_RATE \
    batch_size=$BATCH_SIZE \
    lora_rank=$LORA_RANK \
    num_epochs=$NUM_EPOCHS \
    eval_every=25 \
    save_every=50 \
    log_path="./outputs/place_association_medium" \
    wandb_project=gpsbench-place-association

echo ""
echo "=========================================="
echo "Medium run complete!"
echo "=========================================="
echo ""
echo "Model saved to: ./outputs/place_association_medium"
echo ""
echo "If results look good, run full training:"
echo "  ./run_place_association_full.sh"
echo ""
echo "To customize, set environment variables:"
echo "  MAX_SAMPLES=1000 NUM_EPOCHS=4 ./run_place_association_medium.sh"
