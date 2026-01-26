#!/bin/bash
# Train on all tasks with chain-of-thought reasoning
#
# This trains on all Pure GPS and Applied tasks with step-by-step reasoning
# in the training outputs to help the model learn mathematical reasoning.
#
# Usage:
#   ./run_all_tasks_with_reasoning.sh

set -e

# Configuration
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/all_tasks_with_reasoning}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LORA_RANK="${LORA_RANK:-64}"

echo "Training GPSBench with Chain-of-Thought Reasoning"
echo "=================================================="
echo "Model: $MODEL"
echo "Output dir: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "LoRA rank: $LORA_RANK"
echo ""

# Training command with reasoning enabled
# Parameters are passed directly to TrainingConfig via chz
python train.py \
  model_name="$MODEL" \
  log_path="$OUTPUT_DIR" \
  learning_rate="$LEARNING_RATE" \
  num_epochs="$NUM_EPOCHS" \
  batch_size="$BATCH_SIZE" \
  lora_rank="$LORA_RANK" \
  max_length=16384 \
  dataset_mode="split_files" \
  gpsbench_root="../../" \
  track="both" \
  include_reasoning=true \
  max_eval_samples=500 \
  eval_max_samples=500

echo ""
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
