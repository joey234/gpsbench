#!/bin/bash
# Interactive script to setup API keys for GPSBench Tinker fine-tuning

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================"
echo "GPSBench API Key Setup"
echo -e "======================================${NC}\n"

# Check if .env already exists
if [ -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file already exists${NC}"
    read -p "Do you want to overwrite it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file"
        exit 0
    fi
    mv .env .env.backup
    echo -e "${GREEN}Backed up existing .env to .env.backup${NC}"
fi

# Create .env file
echo "# Generated API Keys Configuration" > .env
echo "# Generated on $(date)" >> .env
echo "" >> .env

# ============================================================
# Tinker API Key
# ============================================================
echo -e "${BLUE}1. Tinker API Key (Required)${NC}"
echo "Required for training with the Tinker platform"
echo "Get your API key from: https://tinker-docs.thinkingmachines.ai/"
echo ""
read -p "Do you have a Tinker API key? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -sp "Enter your Tinker API key: " tinker_key
    echo
    if [ -n "$tinker_key" ]; then
        echo "TINKER_API_KEY=$tinker_key" >> .env
        echo -e "${GREEN}✓ Tinker API key added${NC}"
    fi
else
    echo "# TINKER_API_KEY=your_key_here" >> .env
    echo -e "${YELLOW}⚠ Tinker API key required for training!${NC}"
    echo "Please sign up at: https://tinker-docs.thinkingmachines.ai/"
fi
echo "" >> .env

# ============================================================
# Hugging Face Token
# ============================================================
echo -e "\n${BLUE}2. Hugging Face Token${NC}"
echo "Required for downloading Qwen models"
echo "Get your token from: https://huggingface.co/settings/tokens"
echo ""
read -p "Do you want to add a Hugging Face token? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -sp "Enter your Hugging Face token: " hf_token
    echo
    if [ -n "$hf_token" ]; then
        echo "HF_TOKEN=$hf_token" >> .env
        echo "HUGGINGFACE_TOKEN=$hf_token" >> .env
        echo "HUGGING_FACE_HUB_TOKEN=$hf_token" >> .env
        echo -e "${GREEN}✓ Hugging Face token added${NC}"
    fi
else
    echo "# HF_TOKEN=your_token_here" >> .env
    echo -e "${YELLOW}Skipped - you can add it later by editing .env${NC}"
fi
echo "" >> .env

# ============================================================
# Weights & Biases
# ============================================================
echo -e "\n${BLUE}3. Weights & Biases (W&B) API Key${NC}"
echo "Optional - for experiment tracking and logging"
echo "Get your key from: https://wandb.ai/authorize"
echo ""
read -p "Do you want to add a W&B API key? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -sp "Enter your W&B API key: " wandb_key
    echo
    if [ -n "$wandb_key" ]; then
        echo "WANDB_API_KEY=$wandb_key" >> .env

        # Optional: ask for project and entity
        read -p "Enter W&B project name (default: gpsbench-finetuning): " wandb_project
        wandb_project=${wandb_project:-gpsbench-finetuning}
        echo "WANDB_PROJECT=$wandb_project" >> .env

        read -p "Enter W&B username/entity (press enter to skip): " wandb_entity
        if [ -n "$wandb_entity" ]; then
            echo "WANDB_ENTITY=$wandb_entity" >> .env
        fi

        echo -e "${GREEN}✓ W&B configuration added${NC}"
    fi
else
    echo "# WANDB_API_KEY=your_key_here" >> .env
    echo "# WANDB_PROJECT=gpsbench-finetuning" >> .env
    echo -e "${YELLOW}Skipped - experiment tracking disabled${NC}"
fi
echo "" >> .env

# ============================================================
# OpenAI (optional)
# ============================================================
echo -e "\n${BLUE}4. OpenAI API Key${NC}"
echo "Optional - only needed for GPT model evaluation/comparison"
echo "Get your key from: https://platform.openai.com/api-keys"
echo ""
read -p "Do you want to add an OpenAI API key? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -sp "Enter your OpenAI API key: " openai_key
    echo
    if [ -n "$openai_key" ]; then
        echo "OPENAI_API_KEY=$openai_key" >> .env
        echo -e "${GREEN}✓ OpenAI API key added${NC}"
    fi
else
    echo "# OPENAI_API_KEY=your_key_here" >> .env
    echo -e "${YELLOW}Skipped${NC}"
fi
echo "" >> .env

# ============================================================
# Additional Settings
# ============================================================
echo -e "\n${BLUE}5. Additional Settings${NC}"

# GPU selection
read -p "Specify GPU IDs (e.g., 0,1) or press enter for all GPUs: " gpu_ids
if [ -n "$gpu_ids" ]; then
    echo "CUDA_VISIBLE_DEVICES=$gpu_ids" >> .env
    echo -e "${GREEN}✓ GPU selection configured${NC}"
else
    echo "# CUDA_VISIBLE_DEVICES=0,1" >> .env
fi

# Tokenizer parallelism
echo "TOKENIZERS_PARALLELISM=false" >> .env

# ============================================================
# Summary
# ============================================================
echo -e "\n${GREEN}======================================"
echo "Setup Complete!"
echo -e "======================================${NC}\n"

echo "Created .env file with your configuration"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Verify your settings: cat .env"
echo "2. Test your setup: python train.py max_samples=10 max_steps=5"
echo "3. Start training: ./quickstart.sh"
echo ""
echo -e "${YELLOW}Security Notes:${NC}"
echo "- The .env file contains sensitive API keys"
echo "- It's excluded from git by .gitignore"
echo "- Never commit .env to version control"
echo "- Share .env.example instead"
echo ""

# Test Tinker API key if provided
if grep -q "^TINKER_API_KEY=" .env && ! grep -q "^TINKER_API_KEY=your_key_here" .env; then
    echo -e "${BLUE}Testing Tinker API key...${NC}"
    if command -v python &> /dev/null; then
        python -c "
import os
from dotenv import load_dotenv

load_dotenv()
tinker_key = os.getenv('TINKER_API_KEY')

if tinker_key and tinker_key != 'your_key_here':
    print('✓ Tinker API key is set')
    print(f'  Key length: {len(tinker_key)} characters')
    # Actual validation would require Tinker SDK
    try:
        # Try importing tinker to validate
        import tinker
        print('✓ Tinker SDK is installed')
    except ImportError:
        print('⚠ Install Tinker SDK: pip install tinker-cookbook')
else:
    print('✗ Tinker API key not found or invalid')
" 2>/dev/null || echo -e "${YELLOW}Install python-dotenv to test: pip install python-dotenv${NC}"
    fi
fi

# Test HF token if provided
if grep -q "^HF_TOKEN=" .env && ! grep -q "^HF_TOKEN=your_" .env; then
    echo -e "${BLUE}Testing Hugging Face token...${NC}"
    if command -v python &> /dev/null; then
        python -c "
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv('HF_TOKEN')

try:
    api = HfApi(token=token)
    user = api.whoami()
    print('✓ Hugging Face token is valid')
    print(f'  Logged in as: {user[\"name\"]}')
except Exception as e:
    print('✗ Hugging Face token test failed:', str(e))
" 2>/dev/null || echo -e "${YELLOW}Install python-dotenv to test: pip install python-dotenv${NC}"
    fi
fi
