#!/bin/bash
# Quickstart script for LoRA fine-tuning pipeline

set -e  # Exit on error

echo "==================================="
echo "GPSBench LoRA Fine-tuning Pipeline"
echo "==================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if cities data exists
if [ ! -f "../data/raw/cities15000.txt" ]; then
    echo "Error: cities15000.txt not found!"
    echo "Please download GeoNames data first."
    exit 1
fi

# Step 1: Install dependencies
echo -e "${BLUE}Step 1: Installing dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 2: Generate dataset
echo -e "${BLUE}Step 2: Generating dataset...${NC}"
cd scripts
python generate_dataset.py \
    --cities-file ../../data/raw/cities15000.txt \
    --output-dir ../data \
    --max-cities 5000 \
    --train-ratio 0.8 \
    --noise-levels 0.001 0.005 0.01 \
    --samples-per-city 3 \
    --seed 42

echo -e "${GREEN}✓ Dataset generated${NC}"
echo ""

# Step 3: Train model
echo -e "${BLUE}Step 3: Fine-tuning model (this may take a while)...${NC}"
python train_lora.py \
    --model-name Qwen/Qwen2.5-0.5B \
    --data-dir ../data \
    --output-dir ../models/qwen-lora-location \
    --batch-size 4 \
    --epochs 3 \
    --learning-rate 2e-4 \
    --lora-r 16 \
    --lora-alpha 32

echo -e "${GREEN}✓ Model fine-tuned${NC}"
echo ""

# Step 4: Test model
echo -e "${BLUE}Step 4: Testing model...${NC}"
echo "Testing with New York coordinates (40.7128, -74.0060):"
python inference.py \
    --base-model Qwen/Qwen2.5-0.5B \
    --lora-path ../models/qwen-lora-location \
    --mode single \
    --lat 40.7128 \
    --lon -74.0060

echo ""
echo -e "${GREEN}✓ Pipeline complete!${NC}"
echo ""
echo "Next steps:"
echo "  - Run interactive mode: cd scripts && python inference.py --mode interactive"
echo "  - Evaluate on custom coordinates: python inference.py --mode single --lat <lat> --lon <lon>"
echo "  - Run batch predictions: python inference.py --mode batch --batch-file <file>"
