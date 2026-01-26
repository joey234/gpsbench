# Tinker Implementation Guide for GPSBench

This document provides a detailed guide to the Tinker-based fine-tuning implementation for GPSBench.

## Architecture Overview

```
finetuning/tinker/
├── gpsbench_datasets.py       # Dataset loader compatible with Tinker
├── train.py                   # Main training script using Tinker
├── configs/
│   └── train_config.yaml      # Hydra configuration
├── requirements.txt           # Python dependencies
├── quickstart.sh             # Quick start script
└── README.md                 # User documentation
```

## Key Components

### 1. Dataset Loader (`gpsbench_datasets.py`)

The dataset loader provides three classes:

#### GPSBenchDataset (Base)
- Loads JSON data from disk
- Formats examples into conversation format
- Adds task-specific system prompts
- Can be used standalone or with Tinker

```python
dataset = GPSBenchDataset(
    data_path="../data",
    split="train",
    max_samples=1000
)

example = dataset[0]
# Returns: {"messages": [...], "metadata": {...}}
```

#### GPSBenchHFDataset (Tinker-compatible)
- Wraps GPSBenchDataset in HuggingFace Dataset format
- Inherits from Tinker's SupervisedDatasetFromHFDataset
- Compatible with Tinker's training pipeline

```python
dataset = GPSBenchHFDataset(
    data_path="../data",
    split="train"
)
# Can be passed directly to Tinker trainer
```

#### Key Features:
- **Task-specific prompts**: Different system prompts for format_conversion, place_association, etc.
- **Conversation format**: Properly formatted for chat model fine-tuning
- **Metadata preservation**: Keeps task metadata for analysis
- **Flexible loading**: Supports different file naming conventions

### 2. Training Script (`train.py`)

Main training script using Hydra configuration and Tinker framework.

#### Key Functions:

**load_gpsbench_for_tinker()**
- Loads GPSBench data in Tinker-compatible format
- Handles data conversion and wrapping
- Supports sample limiting for testing

**main()**
- Hydra-decorated entry point
- Initializes Tinker trainer with config
- Manages training loop
- Saves checkpoints and final model

#### Hydra Integration:
```python
@hydra.main(config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    # Configuration is loaded automatically
    # Can be overridden from command line
```

### 3. Configuration (`configs/train_config.yaml`)

Hierarchical configuration following Tinker recommendations:

```yaml
# Core settings
model_name: "Qwen/Qwen2.5-3B"
dataset: "gpsbench"
learning_rate: 5e-4
batch_size: 64
lora_rank: 64

# Alternative configs
quick_test:
  max_samples: 100
  batch_size: 4
```

## Design Decisions

### Why Tinker?

1. **Proven Recipes**: Based on successful chat fine-tuning approaches
2. **Clean API**: Simplified interface compared to raw HuggingFace Trainer
3. **Built-in LoRA**: No manual PEFT setup required
4. **Configuration Management**: Hydra provides powerful config system
5. **Production Features**: Multi-GPU, mixed precision, experiment tracking

### Dataset Format Choice

The implementation uses conversation format:
```python
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

**Benefits:**
- Compatible with chat model training
- Supports task-specific system prompts
- Natural for instruction-following tasks
- Easy to extend for multi-turn conversations

### Hyperparameter Defaults

Based on Tinker cookbook experiments:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| learning_rate | 5e-4 | Works well for chat fine-tuning per cookbook |
| batch_size | 64 | Balances speed and stability |
| lora_rank | 64 | Good capacity without over-parameterization |
| lora_alpha | 128 | 2x rank (standard practice) |
| num_epochs | 3 | Sufficient for convergence on most tasks |

## Integration with Existing System

### Compatibility with Original Implementation

The Tinker implementation coexists with the original `train_lora.py`:

```
finetuning/
├── scripts/
│   └── train_lora.py          # Original implementation
├── tinker/                     # New Tinker-based implementation
│   ├── train.py
│   └── ...
└── data/                       # Shared data directory
```

Both implementations:
- Use the same data format
- Support the same tasks
- Produce compatible LoRA adapters
- Can share evaluation scripts

### Data Sharing

The Tinker implementation uses the same data files:
- `data/train_instructions.json`
- `data/dev_instructions.json`
- `data/test_instructions.json`

No data conversion needed!

## Usage Patterns

### Quick Testing

```bash
# Test with small dataset
python train.py \
  max_samples=10 \
  max_steps=5 \
  batch_size=2
```

### Development

```bash
# Medium-scale training for development
python train.py \
  model_name=Qwen/Qwen2.5-0.5B \
  max_samples=1000 \
  num_epochs=2
```

### Production

```bash
# Full-scale training
python train.py \
  model_name=Qwen/Qwen2.5-3B \
  dataset=gpsbench \
  learning_rate=5e-4 \
  batch_size=64 \
  lora_rank=64 \
  num_epochs=3 \
  wandb_project=gpsbench
```

## Extending the Implementation

### Adding New Tasks

1. **Update dataset loader** with task-specific prompt:
```python
# In gpsbench_datasets.py
task_prompts = {
    'your_new_task': "You are a helpful assistant that...",
    # ...
}
```

2. **Add dataset config**:
```yaml
# In train_config.yaml
your_task:
  dataset: "gpsbench_your_task"
  data_path: "../data/your_task"
```

3. **Train**:
```bash
python train.py dataset=gpsbench_your_task
```

### Custom Data Loaders

Implement custom dataset by inheriting from `GPSBenchDataset`:

```python
class CustomGPSDataset(GPSBenchDataset):
    def _format_conversation(self, example):
        # Custom formatting logic
        return conversation
```

### Advanced Training Features

#### Multi-GPU Training

Tinker uses Accelerate automatically:
```bash
accelerate config  # Configure once
accelerate launch train.py model_name=Qwen/Qwen2.5-3B
```

#### Custom Callbacks

Add custom callbacks to Tinker trainer:
```python
tinker = Tinker(
    # ... other args
    callbacks=[
        MyCustomCallback(),
        EarlyStoppingCallback(patience=3)
    ]
)
```

#### Learning Rate Scheduling

Configure in train_config.yaml:
```yaml
training:
  learning_rate: 5e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
```

## Performance Optimization

### Memory Optimization

1. **Reduce batch size**:
   ```bash
   python train.py batch_size=16
   ```

2. **Lower LoRA rank**:
   ```bash
   python train.py lora_rank=32
   ```

3. **Use gradient checkpointing** (enabled by default in Tinker)

4. **Use smaller model**:
   ```bash
   python train.py model_name=Qwen/Qwen2.5-0.5B
   ```

### Speed Optimization

1. **Increase batch size**:
   ```bash
   python train.py batch_size=128
   ```

2. **Use multiple GPUs**:
   ```bash
   accelerate launch --multi_gpu train.py
   ```

3. **Reduce evaluation frequency**:
   ```bash
   python train.py eval_every=1000
   ```

## Testing and Validation

### Unit Testing Dataset Loader

```bash
cd tinker
python gpsbench_datasets.py
```

Expected output:
```
Testing GPSBench dataset loader...
Loaded 5 training examples

First example:
Messages: [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

### Integration Testing

```bash
# Quick training test
python train.py max_samples=10 max_steps=5
```

Should complete in <5 minutes without errors.

### Model Validation

After training:
```python
from tinker import Tinker

model = Tinker.from_pretrained(
    base_model="Qwen/Qwen2.5-3B",
    lora_path="./outputs/gpsbench_lora/final_model"
)

# Test on sample
response = model.generate("Convert 40.7580, -73.9855 to DMS.")
print(response)
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'tinker_cookbook'`

**Solution**:
```bash
pip install tinker-cookbook
```

### Data Not Found

**Problem**: `FileNotFoundError: Could not find train data file`

**Solution**:
```bash
# Check data path
ls ../data/train_instructions.json

# Or specify custom path
python train.py data_path=/path/to/your/data
```

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `python train.py batch_size=8`
2. Reduce LoRA rank: `python train.py lora_rank=32`
3. Use smaller model: `python train.py model_name=Qwen/Qwen2.5-0.5B`

### Poor Training Results

**Problem**: Loss not decreasing or model not learning

**Solutions**:
1. Check data quality and format
2. Increase training time: `python train.py num_epochs=5`
3. Adjust learning rate: `python train.py learning_rate=3e-4`
4. Increase model capacity: `python train.py lora_rank=128`

## Comparison with Original Implementation

| Aspect | Original | Tinker |
|--------|----------|--------|
| **Configuration** | Argparse | Hydra YAML |
| **Dataset** | Custom loader | Tinker-compatible wrapper |
| **Training** | HF Trainer directly | Tinker abstraction |
| **LoRA Setup** | Manual PEFT | Built-in |
| **Hyperparameters** | Custom | Cookbook-based |
| **Multi-GPU** | Manual setup | Built-in Accelerate |
| **Experiment Tracking** | Optional W&B | Built-in W&B |
| **Lines of Code** | ~450 | ~200 |

## Future Enhancements

1. **Streaming datasets** for very large datasets
2. **Multi-task learning** with task balancing
3. **Curriculum learning** from easy to hard examples
4. **Active learning** with uncertainty sampling
5. **Model merging** for task-specific adapters

## References

- [Tinker Cookbook GitHub](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Tinker Chat SL Recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/chat_sl)
- [Hydra Documentation](https://hydra.cc/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
