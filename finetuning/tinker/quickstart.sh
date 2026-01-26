#!/bin/bash
# Quick Start Script for GPSBench Tinker Fine-tuning

set -e  # Exit on error

echo "======================================"
echo "GPSBench Tinker Fine-tuning Quickstart"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "Error: Please run this script from the finetuning/tinker directory"
    exit 1
fi

# Step 1: Check Python version
echo -e "\n${BLUE}Step 1: Checking Python version...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Step 2: Install dependencies
echo -e "\n${BLUE}Step 2: Installing dependencies...${NC}"
echo "This may take a few minutes..."

if pip show tinker-cookbook > /dev/null 2>&1; then
    echo -e "${GREEN}✓ tinker-cookbook already installed${NC}"
else
    echo "Installing tinker-cookbook..."
    pip install tinker-cookbook
fi

# Install other requirements
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Check data
echo -e "\n${BLUE}Step 3: Checking for training data...${NC}"
data_dir="../data"

if [ ! -f "$data_dir/train_instructions.json" ]; then
    echo -e "${YELLOW}Warning: Training data not found at $data_dir/train_instructions.json${NC}"
    echo "Please run data preparation first:"
    echo "  cd ../scripts"
    echo "  python generate_dataset.py"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    # Count samples
    train_samples=$(cat "$data_dir/train_instructions.json" | grep -o '"instruction"' | wc -l)
    echo -e "${GREEN}✓ Found training data with ~$train_samples samples${NC}"
fi

# Step 4: Choose training mode
echo -e "\n${BLUE}Step 4: Choose training mode:${NC}"
echo "1) Quick test (100 samples, 50 steps) - 5 minutes"
echo "2) Small model (Qwen-0.5B, full data) - 1-2 hours"
echo "3) Full training (Qwen-3B, recommended) - 2-4 hours"
echo "4) Custom configuration"

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo -e "\n${GREEN}Running quick test...${NC}"
        python train.py \
            model_name=Qwen/Qwen2.5-3B \
            dataset=gpsbench \
            max_samples=100 \
            max_eval_samples=20 \
            batch_size=4 \
            max_steps=50 \
            output_dir=./outputs/quick_test
        ;;
    2)
        echo -e "\n${GREEN}Training small model...${NC}"
        python train.py \
            model_name=Qwen/Qwen2.5-0.5B \
            dataset=gpsbench \
            batch_size=32 \
            lora_rank=32 \
            lora_alpha=64 \
            num_epochs=3 \
            output_dir=./outputs/qwen0.5b_lora
        ;;
    3)
        echo -e "\n${GREEN}Running full training (recommended)...${NC}"
        python train.py \
            model_name=Qwen/Qwen2.5-3B \
            dataset=gpsbench \
            learning_rate=5e-4 \
            batch_size=64 \
            lora_rank=64 \
            num_epochs=3 \
            output_dir=./outputs/qwen3b_lora
        ;;
    4)
        echo -e "\n${BLUE}Custom configuration${NC}"
        read -p "Model name (default: Qwen/Qwen2.5-3B): " model_name
        model_name=${model_name:-Qwen/Qwen2.5-3B}

        read -p "Batch size (default: 64): " batch_size
        batch_size=${batch_size:-64}

        read -p "LoRA rank (default: 64): " lora_rank
        lora_rank=${lora_rank:-64}

        read -p "Learning rate (default: 5e-4): " lr
        lr=${lr:-5e-4}

        read -p "Number of epochs (default: 3): " epochs
        epochs=${epochs:-3}

        echo -e "\n${GREEN}Starting custom training...${NC}"
        python train.py \
            model_name=$model_name \
            dataset=gpsbench \
            batch_size=$batch_size \
            lora_rank=$lora_rank \
            learning_rate=$lr \
            num_epochs=$epochs \
            output_dir=./outputs/custom_lora
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Step 5: Training complete
echo -e "\n${GREEN}======================================"
echo "Training Complete!"
echo "======================================${NC}"

echo -e "\nYour fine-tuned model is saved in the outputs directory."
echo -e "\nNext steps:"
echo "1. Check training logs for performance metrics"
echo "2. Run inference on test data"
echo "3. Integrate with GPSBench evaluation"

echo -e "\n${BLUE}Example inference:${NC}"
cat << 'EOF'
from tinker import Tinker

model = Tinker.from_pretrained(
    base_model="Qwen/Qwen2.5-3B",
    lora_path="./outputs/qwen3b_lora/final_model"
)

response = model.generate(
    "Convert 40.7580° N, 73.9855° W to DMS format.",
    max_new_tokens=256
)
print(response)
EOF

echo -e "\n${GREEN}Happy fine-tuning!${NC}"
