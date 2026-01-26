#!/bin/bash
# Train on place_association task only
# This task teaches the model to identify locations from GPS coordinates

set -e

echo "=========================================="
echo "GPSBench Place Association Training"
echo "=========================================="
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

# Configuration
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LORA_RANK="${LORA_RANK:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"  # Single epoch to prevent overfitting
EVAL_EVERY="${EVAL_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-200}"

echo "Training Configuration:"
echo "  Model: $MODEL"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"
echo "  LoRA Rank: $LORA_RANK"
echo "  Epochs: $NUM_EPOCHS"
echo ""

# Check for training data
if [ -f "../data/train_instructions.json" ]; then
    # Count place_association samples
    pa_count=$(python3 -c "
import json
with open('../data/train_instructions.json') as f:
    data = json.load(f)
count = sum(1 for ex in data if ex.get('metadata', {}).get('task') == 'place_association')
print(count)
" 2>/dev/null || echo "0")
    echo "Found $pa_count place_association samples in train_instructions.json"
    USE_INSTRUCTION_FILES=true
else
    echo "Using split files from data/track_applied/splits/"
    USE_INSTRUCTION_FILES=false
fi

echo ""
echo "Starting place_association training..."
echo ""

if [ "$USE_INSTRUCTION_FILES" = true ]; then
    # Use instruction_files mode with filter
    python train.py \
        dataset_mode=instruction_files \
        filter_tasks="['place_association']" \
        model_name="$MODEL" \
        learning_rate=$LEARNING_RATE \
        batch_size=$BATCH_SIZE \
        lora_rank=$LORA_RANK \
        num_epochs=$NUM_EPOCHS \
        eval_every=$EVAL_EVERY \
        save_every=$SAVE_EVERY \
        log_path="./outputs/place_association" \
        wandb_project=gpsbench-place-association
else
    # Use split_files mode
    python train.py \
        dataset_mode=split_files \
        track=applied \
        tasks="['task1_place_association']" \
        model_name="$MODEL" \
        learning_rate=$LEARNING_RATE \
        batch_size=$BATCH_SIZE \
        lora_rank=$LORA_RANK \
        num_epochs=$NUM_EPOCHS \
        eval_every=$EVAL_EVERY \
        save_every=$SAVE_EVERY \
        log_path="./outputs/place_association" \
        wandb_project=gpsbench-place-association
fi

echo ""
echo "=========================================="
echo "Place Association Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: ./outputs/place_association"
echo ""
echo "To customize training, set environment variables:"
echo "  MODEL=Qwen/Qwen3-8B-Base ./run_place_association.sh"
echo "  BATCH_SIZE=32 NUM_EPOCHS=10 ./run_place_association.sh"
