#!/bin/bash
# Quickstart script for multi-task LoRA fine-tuning pipeline

set -e  # Exit on error

echo "=========================================="
echo "GPSBench Multi-Task Fine-tuning Pipeline"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
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

# Step 2: Generate multi-task dataset
echo -e "${BLUE}Step 2: Generating multi-task dataset...${NC}"
echo -e "${YELLOW}This will create training data for 10 geospatial tasks:${NC}"
echo "  1. Format Conversion"
echo "  2. Distance Calculation"
echo "  3. Bearing Computation"
echo "  4. Interpolation"
echo "  5. Area Calculation"
echo "  6. Reverse Geocoding"
echo "  7. Forward Geocoding"
echo "  8. Regional Knowledge"
echo "  9. Relative Position"
echo "  10. Proximity Ranking"
echo ""

cd scripts
python generate_multitask_dataset.py \
    --cities-file ../../data/raw/cities15000.txt \
    --output-dir ../data \
    --max-cities 5000 \
    --samples-per-task 500 \
    --train-ratio 0.8 \
    --noise-levels 0 0.001 0.005 0.01 \
    --seed 42

echo -e "${GREEN}✓ Multi-task dataset generated${NC}"
echo ""

# Step 3: Train model
echo -e "${BLUE}Step 3: Fine-tuning model on all tasks...${NC}"
echo -e "${YELLOW}This may take 1-2 hours depending on your hardware${NC}"
python train_lora.py \
    --model-name Qwen/Qwen2.5-0.5B \
    --data-dir ../data \
    --output-dir ../models/qwen-multitask \
    --batch-size 4 \
    --epochs 3 \
    --learning-rate 2e-4 \
    --lora-r 16 \
    --lora-alpha 32

echo -e "${GREEN}✓ Model fine-tuned${NC}"
echo ""

# Step 4: Test model on different tasks
echo -e "${BLUE}Step 4: Testing model on sample tasks...${NC}"

echo ""
echo "=== Test 1: Distance Calculation ==="
python -c "
from inference import load_model, format_prompt
import torch

model, tokenizer = load_model('Qwen/Qwen2.5-0.5B', '../models/qwen-multitask')

prompt = format_prompt(
    'Calculate the distance between these two points:\\n' +
    'Point A: (40.7128, -74.0060)\\n' +
    'Point B: (51.5074, -0.1278)\\n' +
    'Provide the distance in kilometers.'
)

inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True).split('assistant\\n')[-1])
"

echo ""
echo "=== Test 2: Reverse Geocoding ==="
python inference.py \
    --base-model Qwen/Qwen2.5-0.5B \
    --lora-path ../models/qwen-multitask \
    --mode single \
    --lat 48.8566 \
    --lon 2.3522

echo ""
echo "=== Test 3: Format Conversion ==="
python -c "
from inference import load_model
model, tokenizer = load_model('Qwen/Qwen2.5-0.5B', '../models/qwen-multitask')
# Add format conversion test
print('(Format conversion test - implement if needed)')
"

echo ""
echo -e "${GREEN}✓ Pipeline complete!${NC}"
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "✓ Installed dependencies"
echo "✓ Generated multi-task dataset (10 tasks, ~5000 examples)"
echo "✓ Fine-tuned Qwen-0.5B with LoRA"
echo "✓ Tested on sample tasks"
echo ""
echo "Next steps:"
echo "  - Interactive testing: cd scripts && python inference.py --lora-path ../models/qwen-multitask --mode interactive"
echo "  - Evaluate on GPSBench: python gpsbench_integration.py --lora-path ../models/qwen-multitask"
echo "  - Compare with baselines: See MULTITASK_README.md for details"
echo ""
echo "Documentation:"
echo "  - finetuning/MULTITASK_README.md - Multi-task approach overview"
echo "  - finetuning/MULTI_TASK_DESIGN.md - Design rationale"
echo "  - finetuning/README.md - General fine-tuning guide"
