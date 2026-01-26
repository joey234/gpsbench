#!/bin/bash
# Track-specific training: Train on all tasks from a specific track

set -e

# Default track
TRACK="${1:-both}"

echo "=========================================="
echo "GPSBench Track Training with Tinker"
echo "=========================================="
echo ""
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

# Count available tasks
if [ "$TRACK" = "pure_gps" ]; then
    TASK_COUNT=$(ls ../../data/track_pure_gps/splits/*_train.json 2>/dev/null | wc -l | tr -d ' ')
elif [ "$TRACK" = "applied" ]; then
    TASK_COUNT=$(ls ../../data/track_applied/splits/*_train.json 2>/dev/null | wc -l | tr -d ' ')
else
    TASK_COUNT_PURE=$(ls ../../data/track_pure_gps/splits/*_train.json 2>/dev/null | wc -l | tr -d ' ')
    TASK_COUNT_APPLIED=$(ls ../../data/track_applied/splits/*_train.json 2>/dev/null | wc -l | tr -d ' ')
    TASK_COUNT=$((TASK_COUNT_PURE + TASK_COUNT_APPLIED))
fi

echo "Found $TASK_COUNT tasks in $TRACK track"
echo ""

# Calculate steps per epoch for checkpoint saving
# Dataset sizes: ~2040 examples per task, batch_size=64
BATCH_SIZE=64
SAMPLES_PER_TASK=2040

if [ "$TRACK" = "pure_gps" ]; then
    TOTAL_SAMPLES=$((TASK_COUNT * SAMPLES_PER_TASK))
elif [ "$TRACK" = "applied" ]; then
    TOTAL_SAMPLES=$((TASK_COUNT * SAMPLES_PER_TASK))
else
    # Both tracks
    TOTAL_SAMPLES=$((TASK_COUNT * SAMPLES_PER_TASK))
fi

STEPS_PER_EPOCH=$((TOTAL_SAMPLES / BATCH_SIZE))

echo "Training configuration:"
echo "  Total samples: ~$TOTAL_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Steps per epoch: ~$STEPS_PER_EPOCH"
echo "  Checkpoints will be saved after each epoch"
echo ""

# Confirm with user
echo "This will train on ALL tasks from $TRACK track for 3 epochs"
echo "Estimated time: 1-3 hours (depending on dataset size and Tinker queue)"
echo ""
read -p "Continue with training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

echo ""
echo "Starting track training..."
echo ""

# Run track training with epoch-based checkpoint saving
python train.py \
    dataset_mode=split_files \
    track=$TRACK \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    num_epochs=3 \
    eval_every=$STEPS_PER_EPOCH \
    save_every=$STEPS_PER_EPOCH \
    wandb_project=gpsbench-finetuning \
    log_path="./outputs/track_${TRACK}"

echo ""
echo "=========================================="
echo "Track training complete!"
echo "=========================================="
echo ""
echo "Model saved to: ./outputs/track_${TRACK}"
echo ""
echo "Usage:"
echo "  ./run_track_training.sh <track>"
echo "  Options: pure_gps, applied, both"
echo "  Example: ./run_track_training.sh pure_gps"
