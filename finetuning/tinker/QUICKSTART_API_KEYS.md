# Quick Start: API Keys Setup

## TL;DR

```bash
# Run the interactive setup
./setup_api_keys.sh

# Or manually create .env with both required keys
cat > .env << EOF
TINKER_API_KEY=your_tinker_api_key_here
HF_TOKEN=your_huggingface_token_here
EOF

# Start training
python train.py max_samples=100 max_steps=50
```

## What API Keys Do You Need?

### Required:
- **Tinker API Key** - To authenticate with Tinker training platform
  - Get it: https://tinker-docs.thinkingmachines.ai/
  - Add to `.env`: `TINKER_API_KEY=tinker_xxxxx`

- **Hugging Face Token** - To download Qwen models
  - Get it: https://huggingface.co/settings/tokens
  - Add to `.env`: `HF_TOKEN=hf_xxxxx`

### Optional:
- **Weights & Biases** - For experiment tracking
  - Get it: https://wandb.ai/authorize
  - Add to `.env`: `WANDB_API_KEY=xxxxx`

## Three Ways to Set Up

### 1. Interactive (Easiest)

```bash
./setup_api_keys.sh
```

Walks you through each key with prompts.

### 2. Copy Template

```bash
cp .env.example .env
nano .env  # Edit and add your keys
```

### 3. Export Variables

```bash
export TINKER_API_KEY=tinker_xxxxxxxxxxxxx
export HF_TOKEN=hf_xxxxxxxxxxxxx
python train.py
```

## Where Does it Go?

All API keys go in a `.env` file in this directory:

```
finetuning/tinker/
├── .env          ← Your API keys (git-ignored)
├── .env.example  ← Template to copy
└── train.py
```

## Test Your Setup

```bash
# Test Tinker API key
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Tinker API Key:', 'Set ✓' if os.getenv('TINKER_API_KEY') else 'Missing ✗')"

# Test Hugging Face token
python -c "from huggingface_hub import HfApi; from dotenv import load_dotenv; import os; load_dotenv(); api = HfApi(token=os.getenv('HF_TOKEN')); print('✓ HF token works:', api.whoami()['name'])"

# Run quick training test
python train.py max_samples=10 max_steps=5
```

## Security

✅ `.env` is in `.gitignore` - won't be committed
✅ Only share `.env.example` - not your actual keys
✅ Use read-only tokens when possible

## Need Help?

See [API_KEYS_GUIDE.md](API_KEYS_GUIDE.md) for the complete guide.
