# Running GPSBench Tinker Training

Complete guide for running training after setting up API keys.

## Prerequisites

✅ API keys configured (Tinker + HuggingFace)
✅ Dependencies installed (`pip install -r requirements.txt`)
✅ Training data available in `../data/`

## Quick Start

### Option 1: Example Run (Recommended First)

Test the setup with a quick training run:

```bash
./run_example.sh
```

**What it does:**
- Trains on 100 samples
- Runs for 50 steps
- Takes ~5-10 minutes
- Saves to `./outputs/example_run`

### Option 2: Full Training

Run complete training on all data:

```bash
./run_full_training.sh
```

**What it does:**
- Trains on all samples
- Runs for 3 epochs
- Takes 2-4 hours
- Saves to `./outputs/qwen3b_gpsbench_lora`

### Option 3: Custom Training

Use `train.py` directly with your own parameters:

```bash
python train.py \
  model_name=Qwen/Qwen2.5-3B \
  dataset=gpsbench \
  learning_rate=5e-4 \
  batch_size=64 \
  lora_rank=64 \
  num_epochs=3
```

## Training Commands Explained

### Example Run

```bash
python train.py \
  model_name=Qwen/Qwen2.5-3B \      # Base model to fine-tune
  dataset=gpsbench \                 # Dataset to use
  max_samples=100 \                  # Only use 100 training samples
  max_eval_samples=20 \              # Only use 20 eval samples
  batch_size=4 \                     # Small batch for quick test
  max_steps=50 \                     # Stop after 50 steps
  eval_every=10 \                    # Evaluate every 10 steps
  save_every=25 \                    # Save checkpoint every 25 steps
  output_dir=./outputs/example_run   # Where to save results
```

**Time:** ~5-10 minutes
**Purpose:** Test the setup, verify everything works

### Full Training (Recommended)

```bash
python train.py \
  model_name=Qwen/Qwen2.5-3B \      # Base model (3B parameters)
  dataset=gpsbench \                 # GPSBench dataset
  learning_rate=5e-4 \              # Learning rate (from Tinker cookbook)
  batch_size=64 \                    # Batch size (adjust for GPU)
  lora_rank=64 \                     # LoRA rank (capacity)
  lora_alpha=128 \                   # LoRA alpha (2x rank)
  num_epochs=3 \                     # Train for 3 epochs
  eval_every=500 \                   # Evaluate every 500 steps
  save_every=500 \                   # Save every 500 steps
  output_dir=./outputs/qwen3b_lora \ # Output directory
  wandb_project=gpsbench-finetuning  # W&B project name (optional)
```

**Time:** 2-4 hours (depends on dataset size)
**Purpose:** Production training

### Small Model (Low Memory)

```bash
python train.py \
  model_name=Qwen/Qwen2.5-0.5B \    # Smaller model
  dataset=gpsbench \
  batch_size=32 \                    # Smaller batch
  lora_rank=32 \                     # Lower rank
  lora_alpha=64 \
  num_epochs=3 \
  output_dir=./outputs/qwen0.5b_lora
```

**Time:** ~1 hour
**Purpose:** Training with limited GPU memory

### Large Model (Best Performance)

```bash
python train.py \
  model_name=Qwen/Qwen2.5-8B \      # Larger model
  dataset=gpsbench \
  learning_rate=3e-4 \              # Slightly lower LR
  batch_size=128 \                   # Larger batch
  lora_rank=128 \                    # Higher rank
  lora_alpha=256 \
  num_epochs=5 \                     # More epochs
  output_dir=./outputs/qwen8b_lora
```

**Time:** 4-8 hours
**Purpose:** Best possible performance

## Understanding Output

### During Training

You'll see:

```
====================================
GPSBench Fine-tuning with Tinker
====================================

Model: Qwen/Qwen2.5-3B
Dataset: gpsbench
Learning rate: 0.0005
Batch size: 64
LoRA rank: 64

Loading datasets from ../data...
Train samples: 5000
Eval samples: 1000

✓ Loaded environment variables from .env file
✓ Tinker API key found (42 chars)

Initializing Tinker trainer...
[Training progress bars and metrics]
```

### Metrics to Watch

- **Training Loss**: Should decrease over time
  - Good: Drops from ~2.5 to ~0.5
  - Issue: Stays flat or increases

- **Eval Loss**: Validation performance
  - Good: Decreases and stays low
  - Issue: Increases (overfitting)

- **Learning Rate**: Gradually decreases
  - Good: Smooth decay
  - Issue: Sudden jumps

### Output Directory

After training completes:

```
outputs/qwen3b_gpsbench_lora/
├── final_model/              # Final trained model
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   └── tokenizer files...
├── checkpoint-500/           # Intermediate checkpoints
├── checkpoint-1000/
└── training_log.txt         # Training logs
```

## Monitoring Training

### Option 1: Watch Terminal

Training progress shown in terminal with:
- Step number
- Loss values
- Evaluation metrics
- Time estimates

### Option 2: Weights & Biases

If you set `wandb_project`:

1. Training logs automatically uploaded
2. Visit: https://wandb.ai/your-username/gpsbench-finetuning
3. View real-time metrics, graphs, system stats

### Option 3: Check Checkpoints

```bash
# List checkpoints
ls -lh outputs/qwen3b_gpsbench_lora/checkpoint-*

# Check latest loss
tail -20 outputs/qwen3b_gpsbench_lora/training_log.txt
```

## After Training

### 1. Verify Model Saved

```bash
ls -lh outputs/qwen3b_gpsbench_lora/final_model/
```

Should see:
- `adapter_model.bin` or `adapter_model.safetensors`
- `adapter_config.json`
- Tokenizer files

### 2. Quick Test

```python
from tinker import Tinker

# Load model
model = Tinker.from_pretrained(
    base_model="Qwen/Qwen2.5-3B",
    lora_path="./outputs/qwen3b_gpsbench_lora/final_model"
)

# Test inference
response = model.generate(
    "Convert 40.7580, -73.9855 from decimal_degrees to DMS format.",
    max_new_tokens=256,
    temperature=0.1
)
print(response)
```

### 3. Full Evaluation

```bash
# Integrate with GPSBench evaluation
cd ../../evaluation
python evaluate.py --model ../finetuning/tinker/outputs/qwen3b_gpsbench_lora/final_model
```

## Common Issues

### Issue: Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce batch size: `batch_size=16`
2. Lower LoRA rank: `lora_rank=32`
3. Use smaller model: `model_name=Qwen/Qwen2.5-0.5B`
4. Enable gradient checkpointing (edit config)

### Issue: Training Too Slow

**Solutions:**
1. Increase batch size (if memory allows)
2. Reduce evaluation frequency: `eval_every=1000`
3. Use multiple GPUs
4. Use smaller dataset for testing

### Issue: Poor Performance

**Symptoms:**
- Loss not decreasing
- Eval metrics not improving

**Solutions:**
1. Train longer: `num_epochs=5`
2. Increase model capacity: `lora_rank=128`
3. Adjust learning rate: `learning_rate=3e-4`
4. Check data quality
5. Verify data format is correct

### Issue: Tinker API Error

**Error:** `Tinker authentication failed`

**Solutions:**
1. Check API key in .env: `cat .env | grep TINKER_API_KEY`
2. Verify key is valid (not expired)
3. Check Tinker service status
4. Ensure you have credits/quota

### Issue: Model Not Saving

**Solutions:**
1. Check disk space: `df -h`
2. Verify output directory permissions
3. Check for errors in logs
4. Ensure training completed (not interrupted)

## Advanced Usage

### Resume from Checkpoint

```bash
python train.py \
  model_name=Qwen/Qwen2.5-3B \
  dataset=gpsbench \
  output_dir=./outputs/qwen3b_lora \
  resume_from_checkpoint=./outputs/qwen3b_lora/checkpoint-1000
```

### Multi-GPU Training

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
  model_name=Qwen/Qwen2.5-3B \
  dataset=gpsbench \
  batch_size=256  # Increase for multi-GPU
```

### Custom Data Path

```bash
python train.py \
  model_name=Qwen/Qwen2.5-3B \
  dataset=gpsbench \
  data_path=/path/to/your/data
```

### Experiment Tracking

```bash
python train.py \
  model_name=Qwen/Qwen2.5-3B \
  dataset=gpsbench \
  wandb_project=gpsbench-experiments \
  wandb_entity=myteam \
  output_dir=./outputs/experiment_1
```

## Best Practices

### 1. Always Start with Example Run

```bash
./run_example.sh
```

Verify everything works before committing to long training.

### 2. Monitor Training

Don't just start and forget. Check:
- Loss is decreasing
- No errors in logs
- GPU utilization is high

### 3. Save Intermediate Checkpoints

Use `save_every=500` so you can resume if interrupted.

### 4. Use Version Control

Track your experiments:
```bash
git add configs/
git commit -m "Training run with batch_size=64, lr=5e-4"
```

### 5. Document Hyperparameters

Keep notes on what works:
```
Run 1: batch_size=32, lr=5e-4 → Loss: 0.8
Run 2: batch_size=64, lr=3e-4 → Loss: 0.6 ✓
Run 3: batch_size=128, lr=1e-4 → Loss: 0.9
```

## Quick Reference

| Command | Purpose | Time |
|---------|---------|------|
| `./run_example.sh` | Quick test | 5-10 min |
| `./run_full_training.sh` | Production training | 2-4 hours |
| `python train.py --help` | See all options | - |
| `ls outputs/*/final_model/` | List trained models | - |

## Need Help?

- **Setup issues:** See [API_KEYS_GUIDE.md](API_KEYS_GUIDE.md)
- **Training issues:** See [README.md](README.md)
- **Tinker issues:** https://tinker-docs.thinkingmachines.ai/
- **GPSBench issues:** Main repository README
