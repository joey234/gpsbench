#!/bin/bash
# Train on augmented place_association data (10,000 samples with Gaussian noise)

set -e

echo "=========================================="
echo "GPSBench Place Association - AUGMENTED"
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

# Check for augmented data
TRAIN_FILE="../data/train_place_association_augmented.json"
DEV_FILE="../data/dev_place_association_augmented.json"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Augmented data not found. Generating..."
    cd ..
    python3 augment_place_association.py --target-samples 10000
    cd tinker
    echo ""
fi

# Count samples
train_count=$(python3 -c "import json; print(len(json.load(open('$TRAIN_FILE'))))")
dev_count=$(python3 -c "import json; print(len(json.load(open('$DEV_FILE'))))")

echo "Augmented Dataset:"
echo "  Train: $train_count samples"
echo "  Dev: $dev_count samples"
echo ""

# Configuration - Updated hyperparameters to prevent catastrophic forgetting
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"  # Much lower LR to preserve model capabilities
BATCH_SIZE="${BATCH_SIZE:-32}"
LORA_RANK="${LORA_RANK:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"  # Single epoch to prevent overfitting

# Calculate steps
steps_per_epoch=$((train_count / BATCH_SIZE))
total_steps=$((steps_per_epoch * NUM_EPOCHS))

echo "Training configuration:"
echo "  Model: $MODEL"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size: $BATCH_SIZE"
echo "  LoRA rank: $LORA_RANK"
echo "  Epochs: $NUM_EPOCHS"
echo "  Steps per epoch: ~$steps_per_epoch"
echo "  Total steps: ~$total_steps"
echo "  Estimated time: 30-60 minutes"
echo ""

read -p "Start training on augmented data? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "Starting training on augmented data..."
echo ""

# Use custom data files
python train.py \
    dataset_mode=instruction_files \
    data_path="../data" \
    train_file="$TRAIN_FILE" \
    dev_file="$DEV_FILE" \
    model_name="$MODEL" \
    learning_rate=$LEARNING_RATE \
    batch_size=$BATCH_SIZE \
    lora_rank=$LORA_RANK \
    num_epochs=$NUM_EPOCHS \
    eval_every=200 \
    save_every=400 \
    log_path="./outputs/place_association_augmented" \
    wandb_project=gpsbench-place-association

echo ""
echo "=========================================="
echo "Augmented Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: ./outputs/place_association_augmented"
