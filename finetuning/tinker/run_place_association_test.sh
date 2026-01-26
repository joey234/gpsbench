#!/bin/bash
# Quick test for place_association training (5-10 minutes)
# Use this to verify setup before full training

set -e

echo "=========================================="
echo "GPSBench Place Association - QUICK TEST"
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
pa_count=$(python3 -c "
import json
with open('../data/train_instructions.json') as f:
    data = json.load(f)
count = sum(1 for ex in data if ex.get('metadata', {}).get('task') == 'place_association')
print(count)
" 2>/dev/null || echo "0")

echo "Place association samples available: $pa_count"
echo ""
echo "Quick test configuration:"
echo "  Samples: 100 train, 20 eval"
echo "  Epochs: 1"
echo "  Batch size: 8"
echo "  Estimated time: 5-10 minutes"
echo ""

read -p "Start quick test? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "Starting quick test..."
echo ""

python train.py \
    filter_tasks="['place_association']" \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    max_samples=100 \
    max_eval_samples=20 \
    learning_rate=5e-4 \
    batch_size=8 \
    lora_rank=32 \
    num_epochs=1 \
    eval_every=10 \
    save_every=25 \
    log_path="./outputs/place_association_test" \
    wandb_project=gpsbench-place-association-test

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "=========================================="
echo ""
echo "If successful, run full training with:"
echo "  ./run_place_association_full.sh"
