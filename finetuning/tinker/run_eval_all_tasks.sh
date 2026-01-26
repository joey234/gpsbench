#!/bin/bash
# Evaluate finetuned model on ALL GPSBench test sets
#
# Usage:
#   ./run_eval_all_tasks.sh                    # Full evaluation
#   ./run_eval_all_tasks.sh --max-samples 50   # Quick test

set -e

cd "$(dirname "$0")"

CHECKPOINT="${CHECKPOINT:-final}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/all_tasks_with_reasoning}"
DATA_PATH="${DATA_PATH:-../../data}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

echo "=============================================="
echo "GPSBench All Tasks Evaluation"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "Data path:  $DATA_PATH"
echo "Model:      $MODEL_NAME"
echo "=============================================="
echo ""

python evaluate_all_tasks.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --data-path "$DATA_PATH" \
  --model-name "$MODEL_NAME" \
  "$@"
