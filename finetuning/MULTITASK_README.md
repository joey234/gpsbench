# Multi-Task Fine-Tuning for Geospatial LLMs

## Overview

This directory implements **multi-task fine-tuning** to improve general geospatial capabilities of language models. Unlike single-task fine-tuning (which only teaches coordinate-to-location mapping), this approach trains models on **10 diverse geospatial tasks** covering coordinate operations, geographic knowledge, and spatial reasoning.

## Why Multi-Task Fine-Tuning?

### The Problem with Single-Task Approaches

Training only on reverse geocoding (coordinate → city name) is too narrow:
- ❌ Doesn't improve coordinate system understanding
- ❌ Doesn't teach spatial mathematics (distance, bearing)
- ❌ Doesn't develop geometric reasoning
- ❌ Limited transfer to other geospatial tasks

### Multi-Task Advantages

Training on diverse tasks improves **general geospatial capabilities**:
- ✅ Understands coordinate formats and transformations
- ✅ Can calculate distances and bearings
- ✅ Learns spherical geometry
- ✅ Develops spatial reasoning skills
- ✅ Transfers knowledge across tasks
- ✅ Robust to GPS noise

## 10 Geospatial Tasks

### 1. **Format Conversion** (DMS ↔ Decimal)
Teaches coordinate system understanding.

```
Input: "Convert 40°42'46"N, 74°0'22"W to decimal degrees"
Output: "40.712778, -74.006111"
```

### 2. **Distance Calculation** (Haversine)
Teaches spherical geometry.

```
Input: "Calculate distance from (40.7128, -74.0060) to (51.5074, -0.1278)"
Output: "5,570.3 km"
```

### 3. **Bearing Computation**
Teaches directional reasoning.

```
Input: "Calculate bearing from NYC to LA"
Output: "273.4 degrees (West)"
```

### 4. **Interpolation**
Teaches great circle paths.

```
Input: "Find midpoint between (40.7128, -74.0060) and (51.5074, -0.1278)"
Output: "(51.3021, -31.4215)"
```

### 5. **Area Calculation**
Teaches polygon geometry on spheres.

```
Input: "Calculate area of polygon with vertices [(40.71, -74.00), ...]"
Output: "15.23 km²"
```

### 6. **Reverse Geocoding** (Coordinate → Location)
Teaches coordinate-to-place mapping.

```
Input: "What location is at (40.7128, -74.0060)?"
Output: "New York, US"
```

### 7. **Forward Geocoding** (Location → Coordinate)
Teaches place-to-coordinate mapping.

```
Input: "What are the coordinates of Paris, France?"
Output: "(48.8566, 2.3522)"
```

### 8. **Regional Knowledge**
Teaches political boundaries.

```
Input: "Which country is (35.6762, 139.6503) in?"
Output: "Japan"
```

### 9. **Relative Position**
Teaches spatial comparison.

```
Input: "Is NYC (40.7128, -74.0060) north of LA (34.0522, -118.2437)?"
Output: "Yes"
```

### 10. **Proximity Ranking**
Teaches distance comparison.

```
Input: "Which is closest to Paris: London, Berlin, or Rome?"
Output: "London (343 km)"
```

## Usage

### Quick Start

```bash
# Generate multi-task dataset
cd finetuning/scripts
python generate_multitask_dataset.py \
    --cities-file ../../data/raw/cities15000.txt \
    --samples-per-task 1000 \
    --noise-levels 0 0.001 0.005 0.01

# Train model
python train_lora.py \
    --data-dir ../data \
    --model-name Qwen/Qwen2.5-0.5B \
    --epochs 3
```

### Dataset Generation Options

```bash
python generate_multitask_dataset.py \
    --max-cities 10000 \
    --samples-per-task 1000 \
    --train-ratio 0.8 \
    --noise-levels 0 0.001 0.005 0.01 0.05 \
    --seed 42
```

**Parameters:**
- `--samples-per-task`: Samples per task type (default: 1000)
  - 10 tasks × 1000 samples = 10,000 total examples
- `--noise-levels`: Gaussian noise std in degrees
  - 0 = clean coordinates
  - 0.001° ≈ 111 meters
  - 0.01° ≈ 1.1 kilometers
- `--train-ratio`: Train/val split (default: 0.8)

### Dataset Statistics

With default settings (1000 samples/task, 4 noise levels):
- **Total examples:** 10,000
- **Training:** 8,000
- **Validation:** 2,000
- **Task distribution:** Balanced (1,000 each)
- **Noise distribution:**
  - Clean (0m): ~25%
  - Low (111m): ~25%
  - Medium (555m): ~25%
  - High (1.1km): ~25%

## Training

The multi-task dataset works with the same training script:

```bash
python train_lora.py \
    --data-dir ../data \
    --batch-size 4 \
    --epochs 3 \
    --learning-rate 2e-4
```

The model automatically learns to handle multiple task types through the instruction-following format.

## Evaluation

### Task-Specific Evaluation

Evaluate model performance on each task separately:

```python
from scripts.evaluate_multitask import evaluate_by_task

results = evaluate_by_task(
    model,
    tokenizer,
    val_file='../data/val_multitask_dataset.json'
)

# Results by task type
for task, metrics in results.items():
    print(f"{task}: {metrics['accuracy']:.2%}")
```

### Transfer Learning Analysis

Test if training on one task improves others:

```python
# Train on subset of tasks
train_tasks = ['distance_calculation', 'bearing_computation']

# Evaluate on all tasks
# Do tasks with geometric reasoning transfer to each other?
```

### Compare with Single-Task

```bash
# Single-task (old approach)
python generate_dataset.py --max-cities 10000

# Multi-task (new approach)
python generate_multitask_dataset.py --samples-per-task 1000

# Compare performance on GPSBench
```

## Expected Improvements

### Coordinate Understanding
- Can convert between DMS and decimal
- Understands precision implications
- Handles different coordinate formats

### Geometric Reasoning
- Calculates distances accurately
- Computes bearings
- Interpolates along great circles
- Calculates polygon areas

### Geographic Knowledge
- Maps coordinates to locations (and vice versa)
- Knows country/continent boundaries
- Understands regional geography

### Spatial Reasoning
- Compares relative positions
- Ranks by proximity
- Understands spatial relationships

### Robustness
- Handles GPS noise (10m - 1km)
- Stable under coordinate perturbations
- Generalizes across noise levels

## Research Questions

This setup enables answering key questions:

### 1. Fine-tuning vs. Prompting
**Q:** Does fine-tuning beat prompting on geospatial tasks?

**Experiment:**
- Fine-tune Qwen-0.6B on multi-task dataset
- Compare with GPT-4/Claude with zero-shot prompts
- Measure accuracy on GPSBench tasks

**Hypothesis:** Fine-tuned models excel at:
- Geometric calculations (distance, bearing)
- Format conversions
- Noisy coordinate handling

### 2. Task Transfer
**Q:** Does training on task A improve performance on task B?

**Experiment:**
- Train on geometric tasks only (distance, bearing, area)
- Evaluate on spatial reasoning tasks (proximity, position)
- Measure transfer effects

**Hypothesis:** Related tasks show positive transfer:
- Distance ↔ Proximity
- Bearing ↔ Relative Position
- Interpolation ↔ Route Analysis

### 3. Model Size vs. Fine-tuning
**Q:** Can smaller fine-tuned models beat larger prompted ones?

**Experiment:**
- Fine-tuned Qwen-0.6B (0.6B params)
- vs. Prompted GPT-4 (~1.7T params)

**Hypothesis:** For specialized geospatial tasks, small fine-tuned models can match or exceed large prompted models.

### 4. Noise Robustness
**Q:** Does noise-augmented training improve robustness?

**Experiment:**
- Compare models trained with/without noise
- Evaluate on GPSBench noise tolerance task
- Test generalization to unseen noise levels

**Hypothesis:** Noise-augmented training significantly improves robustness.

### 5. Task Difficulty
**Q:** Which tasks benefit most from fine-tuning?

**Experiment:**
- Measure improvement over baseline for each task
- Identify tasks where fine-tuning has largest impact

**Hypothesis:**
- **High benefit:** Geometric calculations, format conversion
- **Medium benefit:** Reverse/forward geocoding
- **Low benefit:** Simple spatial comparisons

## Integration with GPSBench

### Evaluate on GPSBench Tasks

```python
from scripts.gpsbench_integration import evaluate_on_gpsbench

results = evaluate_on_gpsbench(
    model,
    tokenizer,
    task_files=[
        '../../data/layer_c/task1_distance_calculation.json',
        '../../data/layer_c/task2_bearing_computation.json',
        '../../data/layer_b/task1_place_association.json',
    ]
)
```

### Compare with Baselines

```bash
# Fine-tuned model
python gpsbench_integration.py \
    --lora-path ../models/qwen-lora-location \
    --task-dir ../../data/layer_c \
    --output-file ../results/finetuned_results.json

# OpenAI baseline (from GPSBench)
# Compare results
```

## Data Sources

### From GeoNames
- **cities15000.txt**: 23,000+ cities with coordinates
- Used for all tasks requiring city data
- Provides country codes and population

### Synthetic Generation
- **Distance/bearing**: Random city pairs
- **Interpolation**: Waypoints between cities
- **Area**: Generated polygons around cities
- **Relative position**: Coordinate comparisons

### From GPSBench
Can optionally incorporate:
- Layer A: Format conversion examples
- Layer C: Distance/bearing ground truth
- Layer D: Spatial relationship examples

## File Structure

```
finetuning/
├── MULTI_TASK_DESIGN.md         # Design rationale
├── MULTITASK_README.md           # This file
├── scripts/
│   ├── generate_multitask_dataset.py  # Multi-task generator
│   ├── generate_dataset.py            # Single-task (old)
│   ├── train_lora.py                  # Training (works with both)
│   ├── evaluate_multitask.py          # Task-specific evaluation
│   └── gpsbench_integration.py        # GPSBench evaluation
└── data/
    ├── train_multitask_dataset.json   # Multi-task training data
    ├── val_multitask_dataset.json     # Multi-task validation
    ├── train_dataset.json             # Single-task (old)
    └── val_dataset.json               # Single-task (old)
```

## Comparison: Single-Task vs. Multi-Task

| Aspect | Single-Task | Multi-Task |
|--------|-------------|------------|
| **Tasks** | 1 (reverse geocoding) | 10 (diverse geospatial) |
| **Skills** | Location lookup | Geometry + knowledge + reasoning |
| **Dataset size** | ~32K examples | ~10K examples (more efficient!) |
| **Transfer** | Limited | High |
| **Robustness** | Task-specific | Generalizes |
| **Use case** | Geocoding only | General geospatial reasoning |

## Next Steps

1. ✅ **Implement multi-task generator**
2. ⏳ **Train and evaluate**
3. ⏳ **Compare with single-task baseline**
4. ⏳ **Compare with OpenAI/OpenRouter prompting**
5. ⏳ **Analyze task transfer effects**
6. ⏳ **Publish results**

## Citation

If you use this multi-task fine-tuning approach, please cite GPSBench:

```bibtex
@article{gpsbench2024,
  title={GPSBench: A Comprehensive Benchmark for Geospatial Reasoning},
  author={...},
  year={2024}
}
```

## License

Part of the GPSBench project. Same license applies.
