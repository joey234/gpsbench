# GPSBench Tinker Training - Quick Start

Complete guide to start training GPSBench models using Tinker cloud platform.

## Setup (One-time)

### 1. Navigate to tinker directory

```bash
cd /Users/hungthinh/workspace/GPSBench/finetuning/tinker
```

### 2. API keys are already configured

You've already set up:
- ✅ TINKER_API_KEY
- ✅ HF_TOKEN

Your `.env` file is ready!

### 3. Dependencies are installed

The following are already installed:
- ✅ tinker SDK (v0.6.3)
- ✅ tinker-cookbook (v0.1.0)
- ✅ All required dependencies

## Running Training

### Option 1: Example Run (Recommended First)

Quick test to verify everything works:

```bash
./run_example.sh
```

**What it does:**
- Trains on 100 samples
- Runs for 50 steps
- Takes ~5-10 minutes
- Saves to `./outputs/example_run`

### Option 2: Full Training

Complete training on all data:

```bash
./run_full_training.sh
```

**What it does:**
- Trains on all samples (~5000+)
- Runs for 3 epochs
- Takes 2-4 hours
- Saves to auto-generated output directory

### Option 3: Custom Training

Direct command with custom parameters:

```bash
python train.py \
    model_name="Qwen/Qwen2.5-3B" \
    renderer_name=qwen_chat \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    num_epochs=3 \
    max_samples=1000 \
    wandb_project=my-experiment
```

## Key Parameters

### Model Configuration
- `model_name` - HuggingFace model ID (default: "Qwen/Qwen2.5-3B")
- `renderer_name` - Chat template (use `qwen_chat` for Qwen2.5)

### Training Hyperparameters
- `learning_rate` - Learning rate (default: 5e-4)
- `batch_size` - Batch size (default: 64)
- `lora_rank` - LoRA rank/capacity (default: 64)
- `num_epochs` - Number of epochs (default: 3)

### Dataset Control
- `max_samples` - Limit training samples (default: None = all)
- `max_eval_samples` - Limit eval samples (default: None = all)
- `max_steps` - Maximum steps (overrides epochs)

### Checkpointing
- `eval_every` - Evaluation frequency in steps (default: 500)
- `save_every` - Checkpoint frequency in steps (default: 500)
- `log_path` - Output directory (default: auto-generated)

### Logging
- `wandb_project` - W&B project name (default: "gpsbench-finetuning")
- `wandb_name` - W&B run name (default: auto-generated)

## Example Commands

### Quick smoke test (2 minutes)

```bash
python train.py \
    max_samples=10 \
    max_eval_samples=5 \
    max_steps=10 \
    batch_size=1 \
    renderer_name=qwen_chat \
    log_path=./outputs/smoke_test
```

### Medium-scale experiment (30-60 minutes)

```bash
python train.py \
    max_samples=1000 \
    max_eval_samples=200 \
    batch_size=32 \
    lora_rank=32 \
    renderer_name=qwen_chat \
    wandb_project=gpsbench-medium
```

### Large model training

```bash
python train.py \
    model_name="Qwen3-8B" \
    renderer_name=qwen_chat \
    batch_size=128 \
    lora_rank=128 \
    learning_rate=3e-4 \
    num_epochs=5
```

## After Training

### Check outputs

```bash
ls -lh outputs/
```

You'll see directories like:
```
outputs/
├── gpsbench-Qwen-Qwen2.5-3B-64rank-0.0005lr-64batch-2025-12-03-12-00/
│   ├── checkpoints/
│   ├── final_model/
│   └── logs/
```

### Use the trained model

The cookbook handles model loading and inference. See the Tinker documentation for how to:
1. Load checkpoints
2. Run inference
3. Merge LoRA weights with base model

## Monitoring

### Terminal Output

Training progress shown in terminal:
- Step number
- Loss values
- Evaluation metrics
- Time estimates

### Weights & Biases

If you set `wandb_project`, view real-time metrics at:
```
https://wandb.ai/your-username/gpsbench-finetuning
```

### Check Checkpoints

```bash
# List checkpoints
find outputs/ -name "checkpoint-*" -type d

# Check latest loss
tail -20 outputs/*/logs/training.log
```

## Troubleshooting

### Issue: TINKER_API_KEY not found

```bash
./setup_api_keys.sh
```

### Issue: Model not found

Qwen2.5 models are not in tinker-cookbook's model registry. Always specify `renderer_name=qwen_chat`.

For Qwen3 or other models, check supported models:
```python
python -c "from tinker_cookbook import model_info; print(list(model_info.get_qwen_info().keys()))"
```

### Issue: Out of memory

Reduce batch size:
```bash
python train.py batch_size=16 lora_rank=32 renderer_name=qwen_chat
```

### Issue: Training too slow

Increase batch size (if memory allows):
```bash
python train.py batch_size=128 renderer_name=qwen_chat
```

## Need Help?

- **Tinker docs:** https://tinker-docs.thinkingmachines.ai/
- **Cookbook repo:** https://github.com/thinking-machines-lab/tinker-cookbook
- **GPSBench issues:** Main repository

## Next Steps

After training completes:
1. Evaluate model on GPSBench test set
2. Run inference on sample coordinates
3. Compare with baseline models
4. Fine-tune hyperparameters for better performance
