#!/bin/bash
# Full training for place_association task
# Uses all 2,040 training samples

set -e

echo "=========================================="
echo "GPSBench Place Association - FULL TRAINING"
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

# Count samples
pa_train=$(python3 -c "
import json
with open('../data/train_instructions.json') as f:
    data = json.load(f)
count = sum(1 for ex in data if ex.get('metadata', {}).get('task') == 'place_association')
print(count)
" 2>/dev/null || echo "0")

pa_dev=$(python3 -c "
import json
with open('../data/dev_instructions.json') as f:
    data = json.load(f)
count = sum(1 for ex in data if ex.get('metadata', {}).get('task') == 'place_association')
print(count)
" 2>/dev/null || echo "0")

echo "Dataset:"
echo "  Training samples: $pa_train"
echo "  Dev samples: $pa_dev"
echo ""

# Configuration
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LORA_RANK="${LORA_RANK:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"  # Single epoch to prevent overfitting

# Calculate steps
steps_per_epoch=$((pa_train / BATCH_SIZE))
total_steps=$((steps_per_epoch * NUM_EPOCHS))

echo "Training configuration:"
echo "  Model: $MODEL"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size: $BATCH_SIZE"
echo "  LoRA rank: $LORA_RANK"
echo "  Epochs: $NUM_EPOCHS"
echo "  Steps per epoch: ~$steps_per_epoch"
echo "  Total steps: ~$total_steps"
echo "  Estimated time: 20-40 minutes"
echo ""

read -p "Start full training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "Starting full training..."
echo ""

python train.py \
    filter_tasks="['place_association']" \
    model_name="$MODEL" \
    learning_rate=$LEARNING_RATE \
    batch_size=$BATCH_SIZE \
    lora_rank=$LORA_RANK \
    num_epochs=$NUM_EPOCHS \
    eval_every=100 \
    save_every=200 \
    log_path="./outputs/place_association_full" \
    wandb_project=gpsbench-place-association

echo ""
echo "=========================================="
echo "Full training complete!"
echo "=========================================="
echo ""
echo "Model saved to: ./outputs/place_association_full"
echo ""
echo "To customize, set environment variables:"
echo "  MODEL=Qwen/Qwen3-8B-Base ./run_place_association_full.sh"
echo "  BATCH_SIZE=64 NUM_EPOCHS=3 ./run_place_association_full.sh"
