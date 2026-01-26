#!/bin/bash
# Complete GPSBench Fine-Tuning Pipeline
# This script runs the full pipeline from data preparation to model training

set -e  # Exit on any error

echo "============================================================"
echo "GPSBench Fine-Tuning Pipeline"
echo "============================================================"
echo ""
echo "This script will:"
echo "  1. Create train/dev/test splits"
echo "  2. Convert to instruction format"
echo "  3. Prepare files for training"
echo "  4. Fine-tune the model"
echo ""
echo "Estimated time: ~50 minutes (40 min training + 10 min prep)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Create train/dev/test splits
echo ""
echo "============================================================"
echo "Step 1: Creating train/dev/test splits..."
echo "============================================================"
echo ""
python scripts/create_splits.py --track both --seed 42

# Step 2: Convert to instruction format
echo ""
echo "============================================================"
echo "Step 2: Converting to instruction format..."
echo "============================================================"
echo ""
bash scripts/prepare_finetuning_data.sh both

# Step 3: Prepare files for training
echo ""
echo "============================================================"
echo "Step 3: Preparing files for training..."
echo "============================================================"
echo ""
echo "Copying train_instructions.json -> train_dataset.json"
cp finetuning/data/train_instructions.json finetuning/data/train_dataset.json
echo "Copying dev_instructions.json -> val_dataset.json"
cp finetuning/data/dev_instructions.json finetuning/data/val_dataset.json

# Show statistics
echo ""
echo "Dataset statistics:"
for file in train_dataset val_dataset test_instructions; do
  if [ -f "finetuning/data/${file}.json" ]; then
    count=$(python3 -c "import json; print(len(json.load(open('finetuning/data/${file}.json'))))")
    echo "  ${file}: ${count} examples"
  fi
done

# Step 4: Fine-tune the model
echo ""
echo "============================================================"
echo "Step 4: Fine-tuning model..."
echo "============================================================"
echo ""
echo "This will take ~40 minutes on a GPU..."
echo ""

cd finetuning/scripts
python train_lora.py \
    --model-name Qwen/Qwen2.5-0.5B \
    --data-dir ../data \
    --output-dir ../models/qwen-gps-lora \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4

# Done!
echo ""
echo "============================================================"
echo "Pipeline complete! 🎉"
echo "============================================================"
echo ""
echo "Model saved to: finetuning/models/qwen-gps-lora"
echo ""
echo "Next steps:"
echo "  1. Evaluate on test set:"
echo "     cd finetuning/scripts"
echo "     python inference.py --lora-path ../models/qwen-gps-lora --batch-file ../data/test_instructions.json"
echo ""
echo "  2. Interactive testing:"
echo "     python inference.py --lora-path ../models/qwen-gps-lora --mode interactive"
echo ""
