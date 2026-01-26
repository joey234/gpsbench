#!/bin/bash
# Complete workflow: GPSBench splits → Instruction format → Fine-tuning

set -e

echo "========================================"
echo "Prepare Fine-Tuning Data from GPSBench"
echo "========================================"
echo ""

# Parse arguments
TRACK="${1:-both}"  # Default to both tracks

# Check if splits exist (look for track structure)
if [ ! -d "data/track_pure_gps/splits" ] && [ ! -d "data/track_applied/splits" ] && [ ! -d "data/legacy_layers/layer_a/splits" ] && [ ! -d "data/layer_a/splits" ]; then
    echo "Error: Splits not found!"
    echo "Please run: python scripts/create_splits.py --track both --seed 42"
    exit 1
fi

# Create output directory
mkdir -p finetuning/data

# Determine which track(s) to convert
if [ "$TRACK" == "pure_gps" ] || [ "$TRACK" == "applied" ] || [ "$TRACK" == "both" ]; then
    TRACK_ARG="--track $TRACK"
    echo "Converting track(s): $TRACK"
else
    TRACK_ARG=""
    echo "Converting all available data"
fi

# Convert train split
echo ""
echo "Step 1: Converting training split..."
python scripts/convert_to_instruction_format.py \
    --all-splits train \
    $TRACK_ARG \
    --output-dir finetuning/data

echo ""
echo "Step 2: Converting dev split..."
python scripts/convert_to_instruction_format.py \
    --all-splits dev \
    $TRACK_ARG \
    --output-dir finetuning/data

echo ""
echo "Step 3: Converting test split..."
python scripts/convert_to_instruction_format.py \
    --all-splits test \
    $TRACK_ARG \
    --output-dir finetuning/data

echo ""
echo "========================================"
echo "Conversion Complete!"
echo "========================================"
echo ""
echo "Created files:"
if [ "$TRACK" == "pure_gps" ]; then
    echo "  finetuning/data/train_pure_gps_instructions.json"
    echo "  finetuning/data/dev_pure_gps_instructions.json"
    echo "  finetuning/data/test_pure_gps_instructions.json"
elif [ "$TRACK" == "applied" ]; then
    echo "  finetuning/data/train_applied_instructions.json"
    echo "  finetuning/data/dev_applied_instructions.json"
    echo "  finetuning/data/test_applied_instructions.json"
else
    echo "  finetuning/data/train_instructions.json"
    echo "  finetuning/data/dev_instructions.json"
    echo "  finetuning/data/test_instructions.json"
fi
echo ""
echo "Next step:"
echo "  cd finetuning/scripts"
if [ "$TRACK" == "pure_gps" ] || [ "$TRACK" == "applied" ]; then
    echo "  python train_lora.py --data-dir ../data --track $TRACK"
else
    echo "  python train_lora.py --data-dir ../data"
fi
echo ""
echo "Usage:"
echo "  bash scripts/prepare_finetuning_data.sh [pure_gps|applied|both]"
echo "  Default: both"
