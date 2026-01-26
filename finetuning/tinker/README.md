# GPSBench Fine-tuning with Tinker

This directory contains a Tinker-based fine-tuning implementation for GPSBench coordinate tasks. The implementation follows the [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) supervised learning recipes for chat model fine-tuning.

## Overview

Tinker is a streamlined framework for fine-tuning language models with LoRA (Low-Rank Adaptation). This implementation adapts Tinker's supervised learning approach for GPSBench's coordinate-based geographic tasks.

### Why Tinker?

- **Simplified Training**: Clean API for supervised fine-tuning with sensible defaults
- **Efficient LoRA**: Built-in support for parameter-efficient fine-tuning
- **Proven Hyperparameters**: Based on successful fine-tuning recipes
- **Easy Configuration**: Hydra-based configuration management
- **Production Ready**: Supports multi-GPU, mixed precision, and experiment tracking

## Quick Start

### 1. Installation

```bash
cd finetuning/tinker

# Install Tinker and dependencies
pip install -r requirements.txt

# Or install Tinker directly
pip install tinker-cookbook
```

### 2. Configure API Keys

Set up your API keys (required for downloading models):

**Option A: Interactive Setup (Recommended)**
```bash
./setup_api_keys.sh
```

**Option B: Manual Setup**
```bash
cp .env.example .env
# Edit .env and add your Hugging Face token
nano .env
```

**Option C: Environment Variables**
```bash
export HF_TOKEN=your_huggingface_token
```

See [API_KEYS_GUIDE.md](API_KEYS_GUIDE.md) for detailed instructions.

### 3. Prepare Data

Ensure you have GPSBench training data in the correct format:

```bash
# From the finetuning directory, generate training data
cd ../
python prepare_single_task_data.py  # Or use existing data preparation scripts
```

Expected data structure:
```
finetuning/data/
├── train_instructions.json
├── dev_instructions.json
└── test_instructions.json
```

### 4. Quick Training Test

Test the training pipeline with a small dataset:

```bash
python train.py \
  model_name=Qwen/Qwen2.5-3B \
  dataset=gpsbench \
  max_samples=100 \
  max_eval_samples=20 \
  batch_size=4 \
  max_steps=50
```

### 5. Full Training

Train on the complete dataset:

```bash
python train.py \
  model_name=Qwen/Qwen2.5-3B \
  dataset=gpsbench \
  learning_rate=5e-4 \
  batch_size=64 \
  lora_rank=64
```

## Training Configurations

### Basic Training (Recommended)

Following Tinker cookbook recommendations for chat fine-tuning:

```bash
python train.py \
  model_name=Qwen/Qwen2.5-3B \
  dataset=gpsbench \
  learning_rate=5e-4 \
  batch_size=64 \
  lora_rank=64 \
  num_epochs=3
```

**Expected Results:**
- Training time: ~2-4 hours on single GPU (depending on dataset size)
- Eval loss should decrease to ~1.5-2.0
- Good performance on coordinate conversion and location tasks

### Small Model (Low Memory)

For systems with limited GPU memory:

```bash
python train.py \
  model_name=Qwen/Qwen2.5-0.5B \
  dataset=gpsbench \
  batch_size=32 \
  lora_rank=32
```

### Large Scale (Production)

For best performance with larger models:

```bash
python train.py \
  model_name=Qwen/Qwen2.5-8B \
  dataset=gpsbench \
  learning_rate=3e-4 \
  batch_size=128 \
  lora_rank=128 \
  num_epochs=5
```

### Multi-task Training

Train on all GPSBench tasks simultaneously:

```bash
python train.py \
  model_name=Qwen/Qwen2.5-3B \
  dataset=gpsbench_multitask \
  learning_rate=3e-4 \
  batch_size=64 \
  num_epochs=4
```

## Configuration

The training script uses Hydra for configuration management. You can:

1. **Edit the config file**: Modify `configs/train_config.yaml`
2. **Override from command line**: Use `key=value` syntax
3. **Use predefined configs**: Reference configs in the YAML file

### Key Hyperparameters

Based on Tinker cookbook recommendations:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 5e-4 | Learning rate (5e-4 works well for most tasks) |
| `batch_size` | 64 | Batch size (increase if GPU allows) |
| `lora_rank` | 64 | LoRA rank (higher = more capacity) |
| `lora_alpha` | 128 | LoRA alpha (typically 2x rank) |
| `num_epochs` | 3 | Number of training epochs |
| `eval_every` | 500 | Evaluate every N steps |

### Example Configurations

The `train_config.yaml` includes several predefined configurations:

```bash
# Quick test
python train.py --config-name=quick_test

# Small model
python train.py --config-name=small_model

# Large scale
python train.py --config-name=large_scale

# Multi-task
python train.py --config-name=multitask
```

## Dataset Format

The GPSBench dataset loader expects JSON files with the following structure:

```json
[
  {
    "instruction": "Convert 40°45'N, 73°58'W from DMS to decimal_degrees...",
    "input": "",
    "output": "Step 1: Parse DMS string...\nFINAL ANSWER: 40.750000, -73.966667",
    "metadata": {
      "task": "format_conversion",
      "source_task": "track_pure_gps/task1_format_conversion"
    }
  }
]
```

The loader automatically:
- Formats data into chat conversations with system/user/assistant roles
- Adds task-specific system prompts based on metadata
- Handles different task types (format_conversion, place_association, etc.)

## Training Results

After training, you'll find:

```
outputs/gpsbench_lora/
├── final_model/              # Final fine-tuned model
│   ├── adapter_model.bin     # LoRA adapter weights
│   ├── adapter_config.json   # LoRA configuration
│   └── ...
├── checkpoint-500/           # Intermediate checkpoints
├── checkpoint-1000/
└── ...
```

## Inference

Load and use the fine-tuned model:

```python
from tinker import Tinker

# Load model
tinker = Tinker.from_pretrained(
    base_model="Qwen/Qwen2.5-3B",
    lora_path="./outputs/gpsbench_lora/final_model"
)

# Run inference
prompt = "Convert 40.7580° N, 73.9855° W from decimal_degrees to DMS."
response = tinker.generate(prompt, max_new_tokens=256)
print(response)
```

Or use with transformers directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "./outputs/gpsbench_lora/final_model")

# Generate
messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Convert 40.7580° N, 73.9855° W to DMS."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## Integration with GPSBench Evaluation

To evaluate the fine-tuned model on GPSBench:

```python
# In your evaluation script
from tinker import Tinker

# Load model
model = Tinker.from_pretrained(
    base_model="Qwen/Qwen2.5-3B",
    lora_path="./finetuning/tinker/outputs/gpsbench_lora/final_model"
)

# Use in GPSBench evaluation
def predict(prompt):
    return model.generate(prompt, max_new_tokens=256, temperature=0.1)

# Run GPSBench evaluation with this model
# ...
```

## Experiment Tracking

### Weights & Biases

Enable W&B tracking by setting the `wandb_project` parameter:

```bash
python train.py \
  model_name=Qwen/Qwen2.5-3B \
  dataset=gpsbench \
  wandb_project=gpsbench-finetuning
```

This will log:
- Training/validation loss
- Learning rate schedule
- Model checkpoints
- Hyperparameters

## Troubleshooting

### Out of Memory

1. **Reduce batch size:**
   ```bash
   python train.py batch_size=16
   ```

2. **Use smaller LoRA rank:**
   ```bash
   python train.py lora_rank=32 lora_alpha=64
   ```

3. **Use smaller model:**
   ```bash
   python train.py model_name=Qwen/Qwen2.5-0.5B
   ```

### Slow Training

1. **Increase batch size** (if GPU memory allows):
   ```bash
   python train.py batch_size=128
   ```

2. **Enable mixed precision** (enabled by default in Tinker)

3. **Reduce evaluation frequency:**
   ```bash
   python train.py eval_every=1000
   ```

### Poor Performance

1. **Train longer:**
   ```bash
   python train.py num_epochs=5
   ```

2. **Use higher LoRA rank:**
   ```bash
   python train.py lora_rank=128 lora_alpha=256
   ```

3. **Adjust learning rate:**
   ```bash
   python train.py learning_rate=3e-4
   ```

## Comparison with Original Implementation

| Feature | Original (train_lora.py) | Tinker-based |
|---------|-------------------------|--------------|
| Framework | HuggingFace Trainer | Tinker |
| Configuration | Argparse | Hydra (YAML + CLI) |
| Dataset Loading | Custom | Tinker-compatible wrapper |
| LoRA Setup | Manual PEFT | Built-in Tinker |
| Hyperparameters | Custom | Cookbook-recommended |
| Experiment Tracking | Optional | Built-in W&B support |
| Multi-GPU | Manual | Built-in Accelerate |

## References

- **Tinker Cookbook**: https://github.com/thinking-machines-lab/tinker-cookbook
- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Qwen Models**: https://github.com/QwenLM/Qwen
- **Hydra**: https://hydra.cc/

## Contributing

To add new task-specific datasets:

1. Update `gpsbench_datasets.py` with task-specific system prompts
2. Add dataset configuration to `train_config.yaml`
3. Test with `python train.py dataset=your_new_dataset max_samples=10`

## License

This code is part of the GPSBench project and follows the same license.
