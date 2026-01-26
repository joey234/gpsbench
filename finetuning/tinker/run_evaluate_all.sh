#!/bin/bash
# Evaluate finetuned model on ALL GPSBench tasks (both tracks)
# Runs the complete benchmark: pure_gps + applied

set -e

echo "==========================================="
echo "GPSBench Full Evaluation - Finetuned Model"
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
TRACK="${TRACK:-both}"  # "pure_gps", "applied", or "both"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Empty = all samples
MAX_TOKENS="${MAX_TOKENS:-512}"  # Keep consistent with training
BATCH_SIZE="${BATCH_SIZE:-16}"   # Parallel batch size for speed

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output dir: $OUTPUT_DIR"
echo "  Track: $TRACK"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  Max samples per task: $MAX_SAMPLES"
else
    echo "  Max samples per task: all"
fi
echo "  Max tokens: $MAX_TOKENS"
echo "  Batch size: $BATCH_SIZE"
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
CMD="python run_gpsbench_finetuned.py --checkpoint $CHECKPOINT --model-name $MODEL --output-dir $OUTPUT_DIR --track $TRACK --max-tokens $MAX_TOKENS --batch-size $BATCH_SIZE"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

echo "Running full GPSBench evaluation..."
echo "This may take a while depending on the number of tasks..."
echo ""

$CMD

echo ""
echo "==========================================="
echo "Full evaluation complete!"
echo "==========================================="
