# LoRA Fine-tuning for Coordinate-to-Location Prediction

This directory contains scripts for fine-tuning language models using LoRA (Low-Rank Adaptation) to predict locations from GPS coordinates. The training data includes Gaussian noise to improve model robustness.

## Overview

### Task
Predict location names (city, country) from GPS coordinates with Gaussian noise.

**Example:**
- **Input:** `What location is at coordinates (40.758896, -73.985130)?`
- **Output:** `New York, US`

### Model
- Base model: **Qwen/Qwen2.5-0.5B** (or Qwen/Qwen3-0.6B if available)
- Fine-tuning method: **LoRA (Low-Rank Adaptation)**
- Quantization: **4-bit** (for memory efficiency)

### Dataset
The dataset is generated from GPSBench's city data with:
- **Clean coordinates** (no noise)
- **Noisy coordinates** with multiple Gaussian noise levels
  - 0.001° (~111 meters)
  - 0.005° (~555 meters)
  - 0.01° (~1.1 kilometers)

## Directory Structure

```
finetuning/
├── data/                          # Generated datasets
│   ├── train_dataset.json        # Training data
│   └── val_dataset.json          # Validation data
├── models/                        # Fine-tuned models
│   ├── qwen-lora-location/       # Final LoRA adapter
│   └── checkpoints/              # Training checkpoints
├── scripts/
│   ├── generate_dataset.py       # Dataset generation
│   ├── train_lora.py            # LoRA fine-tuning
│   └── inference.py             # Model inference
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup

### 1. Install Dependencies

```bash
cd finetuning
pip install -r requirements.txt
```

**GPU Requirements:**
- Recommended: NVIDIA GPU with 16GB+ VRAM
- With 4-bit quantization: 8GB+ VRAM should work
- CPU training is possible but very slow

### 2. Generate Dataset

Generate training and validation datasets from GPSBench city data:

```bash
cd scripts
python generate_dataset.py \
    --cities-file ../../data/raw/cities15000.txt \
    --output-dir ../data \
    --max-cities 10000 \
    --train-ratio 0.8 \
    --noise-levels 0.001 0.005 0.01 \
    --samples-per-city 3 \
    --seed 42
```

**Parameters:**
- `--cities-file`: Path to GeoNames cities data
- `--output-dir`: Output directory for datasets
- `--max-cities`: Maximum number of cities to use
- `--train-ratio`: Training split ratio (0-1)
- `--noise-levels`: Gaussian noise standard deviations in degrees
- `--samples-per-city`: Number of noisy samples per city per noise level
- `--seed`: Random seed for reproducibility

**Output:**
- `train_dataset.json`: Training examples
- `val_dataset.json`: Validation examples

**Dataset Statistics (default settings):**
- Training: ~32,000 examples (8,000 cities × 4 samples)
- Validation: ~8,000 examples (2,000 cities × 4 samples)

### 3. Fine-tune Model

Train the model with LoRA:

```bash
python train_lora.py \
    --model-name Qwen/Qwen2.5-0.5B \
    --data-dir ../data \
    --output-dir ../models/qwen-lora-location \
    --batch-size 4 \
    --epochs 3 \
    --learning-rate 2e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --max-seq-length 256
```

**Key Parameters:**

**Model Settings:**
- `--model-name`: Base model (Qwen/Qwen2.5-0.5B or Qwen/Qwen3-0.6B)
- `--no-4bit`: Disable 4-bit quantization (requires more VRAM)

**Training Settings:**
- `--batch-size`: Batch size per device
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate
- `--max-seq-length`: Maximum sequence length

**LoRA Settings:**
- `--lora-r`: LoRA rank (higher = more capacity, default: 16)
- `--lora-alpha`: LoRA scaling factor (default: 32)

**Training Time:**
- With 10K cities and 3 epochs: ~1-2 hours on RTX 4090
- Memory usage: ~8-10GB VRAM with 4-bit quantization

### 4. Run Inference

#### Interactive Mode

```bash
python inference.py \
    --base-model Qwen/Qwen2.5-0.5B \
    --lora-path ../models/qwen-lora-location \
    --mode interactive
```

Example session:
```
Enter coordinates: 40.7128,-74.0060
Predicted location: New York, US

Enter coordinates: 48.8566,2.3522
Predicted location: Paris, FR
```

#### Single Prediction

```bash
python inference.py \
    --base-model Qwen/Qwen2.5-0.5B \
    --lora-path ../models/qwen-lora-location \
    --mode single \
    --lat 40.7128 \
    --lon -74.0060
```

#### Batch Inference

Create a CSV file with coordinates:
```csv
latitude,longitude,ground_truth
40.7128,-74.0060,New York
48.8566,2.3522,Paris
35.6762,139.6503,Tokyo
```

Run batch inference:
```bash
python inference.py \
    --base-model Qwen/Qwen2.5-0.5B \
    --lora-path ../models/qwen-lora-location \
    --mode batch \
    --batch-file coordinates.csv \
    --output-file predictions.csv
```

## Advanced Usage

### Custom Noise Levels

To test robustness with different noise levels:

```bash
python generate_dataset.py \
    --noise-levels 0.0001 0.001 0.01 0.1 \
    --samples-per-city 5
```

### Hyperparameter Tuning

Experiment with different LoRA configurations:

```bash
# Higher rank for more capacity
python train_lora.py --lora-r 32 --lora-alpha 64

# More training
python train_lora.py --epochs 5 --learning-rate 1e-4

# Larger batches (if VRAM allows)
python train_lora.py --batch-size 8 --gradient-accumulation-steps 2
```

### Using Different Base Models

```bash
# Use a larger model (requires more VRAM)
python train_lora.py --model-name Qwen/Qwen2.5-1.5B

# Or a smaller one
python train_lora.py --model-name Qwen/Qwen2.5-0.5B
```

## Evaluation

### Quick Validation

Check model performance on validation set:

```python
from scripts.inference import load_model, predict_location
import json

# Load model
model, tokenizer = load_model(
    'Qwen/Qwen2.5-0.5B',
    '../models/qwen-lora-location'
)

# Load validation data
with open('../data/val_dataset.json') as f:
    val_data = json.load(f)

# Test on samples
correct = 0
for example in val_data[:100]:
    # Extract coordinates from instruction
    # (parse from "What location is at coordinates (lat, lon)?")
    instruction = example['instruction']
    coords = instruction.split('(')[1].split(')')[0]
    lat, lon = map(float, coords.split(','))

    prediction = predict_location(model, tokenizer, lat, lon)
    ground_truth = example['output']

    if prediction.lower() == ground_truth.lower():
        correct += 1

accuracy = correct / 100
print(f"Accuracy: {accuracy:.2%}")
```

### Noise Sensitivity Analysis

Test how noise affects predictions:

```python
import numpy as np

# Test a single location with varying noise
base_lat, base_lon = 40.7128, -74.0060

for noise_std in [0, 0.001, 0.01, 0.1]:
    noisy_lat = base_lat + np.random.normal(0, noise_std)
    noisy_lon = base_lon + np.random.normal(0, noise_std)

    pred = predict_location(model, tokenizer, noisy_lat, noisy_lon)
    print(f"Noise {noise_std:.4f}°: {pred}")
```

## Integration with GPSBench

To use the fine-tuned model in GPSBench evaluation:

```python
from finetuning.scripts.inference import load_model, predict_location

# Load model
model, tokenizer = load_model(
    'Qwen/Qwen2.5-0.5B',
    'finetuning/models/qwen-lora-location'
)

# Use in GPSBench tasks
def predict_location_from_coords(lat, lon):
    return predict_location(model, tokenizer, lat, lon)

# Run GPSBench evaluation with fine-tuned model
# ...
```

## Troubleshooting

### Out of Memory Errors

1. **Reduce batch size:**
   ```bash
   python train_lora.py --batch-size 2
   ```

2. **Enable gradient checkpointing** (edit `train_lora.py`):
   ```python
   training_args = TrainingArguments(
       ...
       gradient_checkpointing=True,
   )
   ```

3. **Use smaller model:**
   ```bash
   python train_lora.py --model-name Qwen/Qwen2.5-0.5B
   ```

### Slow Training

- Ensure you're using a GPU: Check with `nvidia-smi`
- Use larger batch size if VRAM allows
- Reduce dataset size: `--max-cities 5000`

### Poor Performance

- **Increase training epochs:** `--epochs 5`
- **Use larger LoRA rank:** `--lora-r 32 --lora-alpha 64`
- **More training data:** `--max-cities 20000`
- **Add more noise samples:** `--samples-per-city 5`

## References

- **LoRA Paper:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Qwen2.5:** [Qwen2.5: A Party of Foundation Models](https://qwenlm.github.io/blog/qwen2.5/)
- **PEFT Library:** [Hugging Face PEFT](https://github.com/huggingface/peft)
- **GPSBench:** Main benchmark repository

## Citation

If you use this fine-tuning setup in your research, please cite GPSBench:

```bibtex
@article{gpsbench2024,
  title={GPSBench: A Benchmark for GPS-based Location Tasks},
  author={...},
  year={2024}
}
```

## License

This code is part of the GPSBench project and follows the same license.
