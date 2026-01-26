#!/bin/bash
# Example training run for GPSBench Tinker fine-tuning

set -e

echo "======================================"
echo "GPSBench Tinker Example Training Run"
echo "======================================"

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
    echo "Please run: ./setup_api_keys.sh"
    exit 1
fi

echo "✓ API keys found"
echo ""

# Check data
if [ ! -f "../data/train_instructions.json" ]; then
    echo "Error: Training data not found at ../data/train_instructions.json"
    echo "Please prepare data first"
    exit 1
fi

echo "✓ Training data found"
echo ""

# Count samples
train_samples=$(python3 -c "import json; data=json.load(open('../data/train_instructions.json')); print(len(data))")
echo "Training samples: $train_samples"
echo ""

# Run example training
echo "Starting example training run..."
echo "This will train on 100 samples for 1 epoch (~5-10 minutes)"
echo ""

python train.py \
    max_samples=100 \
    max_eval_samples=20 \
    batch_size=4 \
    num_epochs=1 \
    eval_every=10 \
    save_every=25 \
    log_path=./outputs/example_run
    model_name=Qwen/Qwen3-30B-A3B-Instruct-2507

echo ""
echo "======================================"
echo "Example run complete!"
echo "======================================"
echo ""
echo "Model saved to: ./outputs/example_run"
