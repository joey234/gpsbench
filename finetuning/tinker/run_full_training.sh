#!/bin/bash
# Full training run for GPSBench Tinker fine-tuning

set -e

echo "======================================"
echo "GPSBench Tinker Full Training Run"
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
    exit 1
fi

if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
    echo "Error: HF_TOKEN not set in .env file"
    exit 1
fi

echo "✓ API keys verified"
echo ""

# Check data
if [ ! -f "../data/train_instructions.json" ]; then
    echo "Error: Training data not found at ../data/train_instructions.json"
    exit 1
fi

echo "✓ Training data found"
echo ""

# Count samples
train_samples=$(python3 -c "import json; data=json.load(open('../data/train_instructions.json')); print(len(data))")
dev_samples=$(python3 -c "import json; data=json.load(open('../data/dev_instructions.json')); print(len(data))")

echo "Dataset statistics:"
echo "  Training samples: $train_samples"
echo "  Dev samples: $dev_samples"
echo ""

# Confirm with user
echo "This will train on ALL samples for 3 epochs"
echo "Estimated time: 2-4 hours (depending on dataset size and Tinker queue)"
echo ""
read -p "Continue with full training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

echo ""
echo "Starting full training..."
echo ""

# Full training with recommended hyperparameters
python train.py \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    num_epochs=3 \
    eval_every=500 \
    save_every=500 \
    wandb_project=gpsbench-finetuning

echo ""
echo "======================================"
echo "Full training complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Check the output directory for trained model"
echo "2. Evaluate model performance"
echo "3. Run inference on test data"
