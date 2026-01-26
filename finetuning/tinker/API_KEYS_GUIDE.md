# API Keys Configuration Guide

This guide explains how to configure API keys for GPSBench Tinker fine-tuning.

## Quick Setup

### Option 1: Interactive Setup (Recommended)

Run the interactive setup script:

```bash
./setup_api_keys.sh
```

This will guide you through setting up:
- Hugging Face token (required)
- Weights & Biases key (optional)
- OpenAI API key (optional)
- GPU configuration

### Option 2: Manual Setup

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your keys:
   ```bash
   nano .env  # or use your preferred editor
   ```

3. Replace placeholder values with your actual API keys

### Option 3: Environment Variables

Set environment variables directly (no .env file):

```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
python train.py
```

## Required API Keys

### 1. Tinker API Key (Required)

**Why needed?** Tinker is a hosted training platform - the API key authenticates your training jobs

**Get your key:**
1. Go to https://tinker-docs.thinkingmachines.ai/
2. Sign up for a Tinker account
3. Navigate to your dashboard
4. Copy your API key from the "API Keys" section

**Add to .env:**
```bash
TINKER_API_KEY=tinker_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Verify your key:**
```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Tinker API Key:', 'Set ✓' if os.getenv('TINKER_API_KEY') else 'Missing ✗')
"
```

**Pricing & Limits:**
- Check current pricing at: https://tinker-docs.thinkingmachines.ai/pricing
- Free tier available for testing
- GPU hours billed separately

### 2. Hugging Face Token (Required)

**Why needed?** To download Qwen models from Hugging Face Hub

**Get your token:**
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Read" permission
4. Copy the token

**Add to .env:**
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Alternative: Login via CLI:**
```bash
huggingface-cli login
```

## Optional API Keys

### 3. Weights & Biases (Optional)

**Why needed?** For experiment tracking, metric logging, and visualization

**Get your key:**
1. Go to https://wandb.ai/authorize
2. Copy your API key

**Add to .env:**
```bash
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=gpsbench-finetuning
WANDB_ENTITY=your_username
```

**Enable in training:**
```bash
python train.py wandb_project=gpsbench-finetuning
```

**Alternative: Login via CLI:**
```bash
wandb login
```

### 4. OpenAI API Key (Optional)

**Why needed?** Only if comparing with GPT models during evaluation

**Get your key:**
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Copy the key

**Add to .env:**
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Configuration Examples

### Minimal Configuration (Required)

```bash
# .env
TINKER_API_KEY=tinker_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TOKENIZERS_PARALLELISM=false
```

### Full Configuration (With Experiment Tracking)

```bash
# .env
TINKER_API_KEY=tinker_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=gpsbench-finetuning
WANDB_ENTITY=myusername
CUDA_VISIBLE_DEVICES=0,1
TOKENIZERS_PARALLELISM=false
```

### Production Configuration

```bash
# .env
TINKER_API_KEY=tinker_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=gpsbench-production
WANDB_ENTITY=myteam
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false
```

## Verification

### Test Tinker API Key

```bash
python -c "
import os
from dotenv import load_dotenv

load_dotenv()
tinker_key = os.getenv('TINKER_API_KEY')

if tinker_key and tinker_key != 'your_tinker_api_key_here':
    print(f'✓ Tinker API key is set ({len(tinker_key)} characters)')
    # Full validation requires Tinker SDK
    try:
        from tinker import Tinker
        print('✓ Tinker SDK imported successfully')
    except ImportError:
        print('⚠ Tinker SDK not installed: pip install tinker-cookbook')
else:
    print('✗ Tinker API key not found')
    print('  Get your key from: https://tinker-docs.thinkingmachines.ai/')
"
```

### Test Hugging Face Token

```bash
python -c "
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()
api = HfApi(token=os.getenv('HF_TOKEN'))
user = api.whoami()
print(f'✓ Logged in as: {user[\"name\"]}')
"
```

### Test Weights & Biases

```bash
python -c "
import wandb
from dotenv import load_dotenv
import os

load_dotenv()
wandb.login(key=os.getenv('WANDB_API_KEY'))
print('✓ W&B authentication successful')
"
```

### Test OpenAI API

```bash
python -c "
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
models = openai.Model.list()
print(f'✓ OpenAI API working, found {len(models.data)} models')
"
```

## GPU Configuration

### Select Specific GPUs

```bash
# In .env
CUDA_VISIBLE_DEVICES=0,1

# Or via command line
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

### Check Available GPUs

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

## Security Best Practices

### 1. Never Commit API Keys

The `.gitignore` file is configured to exclude `.env`:

```gitignore
.env
*.env
!.env.example
```

**Always verify before committing:**
```bash
git status  # Check that .env is not listed
```

### 2. Use Read-Only Tokens

For Hugging Face, use tokens with minimal permissions:
- Training: "Read" permission only
- Model upload: Create separate token with "Write" permission

### 3. Rotate Keys Regularly

- Change API keys periodically
- Revoke old keys after rotation
- Use different keys for development vs production

### 4. Use Team Secrets for Shared Projects

For team projects, use secret management:
- GitHub Secrets (for CI/CD)
- AWS Secrets Manager
- HashiCorp Vault

### 5. Environment-Specific Configuration

Use different .env files for different environments:

```bash
.env.development
.env.staging
.env.production
```

Load the appropriate one:
```bash
# Development
cp .env.development .env
python train.py

# Production
cp .env.production .env
python train.py
```

## Troubleshooting

### Error: "Could not authenticate"

**Problem:** Invalid or missing API key

**Solutions:**
1. Check that .env file exists: `ls -la .env`
2. Verify key format is correct (no extra spaces)
3. Check key is not expired (regenerate if needed)
4. Ensure python-dotenv is installed: `pip install python-dotenv`

### Error: "No module named 'dotenv'"

**Problem:** python-dotenv not installed

**Solution:**
```bash
pip install python-dotenv
```

### Keys Not Loading

**Problem:** .env file not being read

**Solutions:**
1. Check file is named exactly `.env` (not `env` or `.env.txt`)
2. Ensure .env is in the same directory as train.py
3. Try setting environment variables directly:
   ```bash
   export HF_TOKEN=your_token
   ```

### Rate Limiting

**Problem:** Too many API requests

**Solutions:**
1. Use Hugging Face cache: Models are cached after first download
2. For W&B: Reduce logging frequency in config
3. For OpenAI: Implement request throttling

## Reference

### Environment Variable Priority

When multiple sources exist, priority is:

1. Command line environment variables
   ```bash
   HF_TOKEN=xxx python train.py
   ```

2. .env file in current directory
   ```bash
   # .env
   HF_TOKEN=xxx
   ```

3. System environment variables
   ```bash
   # ~/.bashrc or ~/.zshrc
   export HF_TOKEN=xxx
   ```

4. Default values in code

### Complete .env Template

```bash
# Tinker API (Required)
TINKER_API_KEY=tinker_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Hugging Face (Required)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Weights & Biases (Optional)
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=gpsbench-finetuning
WANDB_ENTITY=myusername

# OpenAI (Optional)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Misc
TOKENIZERS_PARALLELISM=false
LOG_LEVEL=INFO
```

## Getting Help

If you encounter issues:

1. Check this guide
2. Verify .env file format
3. Test each API key individually
4. Check API service status pages
5. Consult the main README.md

## Related Documentation

- [Hugging Face Tokens](https://huggingface.co/docs/hub/security-tokens)
- [Weights & Biases Authentication](https://docs.wandb.ai/guides/track/public-api-guide)
- [OpenAI API Keys](https://platform.openai.com/docs/api-reference/authentication)
