#!/bin/bash
# Evaluate finetuned model on the place_association test set
# Uses the latest checkpoint from place_association_augmented training

set -e

echo "==========================================="
echo "Evaluate Finetuned Model on Test Set"
echo "==========================================="
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
CHECKPOINT="${CHECKPOINT:-final}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/place_association_augmented}"
DATA_PATH="${DATA_PATH:-../../data}"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Empty = all samples

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output dir: $OUTPUT_DIR"
echo "  Data path: $DATA_PATH"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  Max samples: $MAX_SAMPLES"
else
    echo "  Max samples: all (1020)"
fi
echo ""

# Check if checkpoint exists
if [ ! -f "$OUTPUT_DIR/checkpoints.jsonl" ]; then
    echo "Error: No checkpoints found at $OUTPUT_DIR/checkpoints.jsonl"
    exit 1
fi

# Show available checkpoints
echo "Available checkpoints:"
cat "$OUTPUT_DIR/checkpoints.jsonl" | python3 -c "
import json, sys
for line in sys.stdin:
    if line.strip():
        ckpt = json.loads(line)
        print(f\"  - {ckpt['name']} (epoch {ckpt['epoch']}, batch {ckpt['batch']})\")"
echo ""

# Build command
CMD="python evaluate_on_test.py --checkpoint $CHECKPOINT --model-name $MODEL --output-dir $OUTPUT_DIR --data-path $DATA_PATH"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

echo "Running evaluation..."
echo ""

$CMD

echo ""
echo "==========================================="
echo "Evaluation complete!"
echo "==========================================="
