# GPSBench Evaluation Metrics in Tinker Training

## Overview

The Tinker training now includes **task-specific evaluation metrics** integrated from GPSBench's evaluation system.

## Training vs. Evaluation Metrics

### Training Loss: Cross-Entropy ✅
**What it is:**
- Standard language modeling loss
- Measures how well the model predicts the next token
- Applied to all tokens in the assistant's response

**Why it's used:**
- Correct for fine-tuning language models
- Teaches the model to generate text matching the target output
- Universal across all task types

### Evaluation Metrics: Task-Specific Accuracy ✅ (NEW!)
**What they are:**
- Task-appropriate accuracy metrics
- Computed during training (every `eval_every` steps)
- Logged to Weights & Biases

**Metrics by task type:**

#### 1. Format Conversion
- **Metric:** Coordinate number matching
- **Tolerance:** 0.01° (~1km)
- **How it works:** Extracts numbers from prediction and ground truth, checks if they match within tolerance

#### 2. Numerical Tasks (Distance, Bearing, Area)
- **Metric:** Relative error
- **Tolerance:** 5% relative error
- **Tasks:** distance_calculation, bearing_computation, area_perimeter, interpolation

#### 3. Place Association
- **Metric:** Fuzzy string matching
- **Scoring:**
  - 1.0 = Exact match
  - 0.5 = Partial match (contains or word overlap)
  - 0.0 = No match

#### 4. Other Tasks
- **Metric:** Exact string match (case-insensitive)
- **Scoring:** Binary (0.0 or 1.0)

## Metrics Logged to W&B

During training, you'll see these metrics:

```
gpsbench_eval/accuracy                    # Overall accuracy across all tasks
gpsbench_eval/format_conversion_accuracy  # Format conversion task
gpsbench_eval/distance_calculation_accuracy
gpsbench_eval/place_association_accuracy
gpsbench_eval/...                         # Other task-specific metrics
```

## Configuration

### Enable/Disable Evaluation
```bash
# Enabled by default
python train.py

# Disable for faster training
python train.py enable_task_eval=False
```

### Control Evaluation Sample Size
```bash
# Default: 50 samples
python train.py eval_max_samples=50

# More thorough (slower)
python train.py eval_max_samples=200

# Faster
python train.py eval_max_samples=20
```

### Evaluation Frequency
```bash
# Evaluate every 500 steps (default)
python train.py eval_every=500

# More frequent evaluation
python train.py eval_every=100

# Less frequent (faster training)
python train.py eval_every=1000
```

## Example Training with Evaluation

```bash
python train.py \
    model_name="meta-llama/Llama-3.2-3B" \
    enable_task_eval=True \
    eval_max_samples=100 \
    eval_every=500 \
    wandb_project=gpsbench-finetuning
```

## Interpreting Metrics

### What Good Metrics Look Like

**Format Conversion:**
- Initial: ~0.1-0.3 (10-30% accuracy)
- After training: >0.8 (>80% accuracy)

**Numerical Tasks:**
- Initial: ~0.2-0.4 (20-40% accuracy)
- After training: >0.7 (>70% accuracy)

**Place Association:**
- Initial: ~0.1-0.2 (10-20% accuracy)
- After training: >0.5 (>50% accuracy)

### Debugging Poor Performance

**If accuracy isn't improving:**
1. Check training loss is decreasing
2. Increase `num_epochs` or lower `learning_rate`
3. Check task examples with `eval_max_samples=200` for more data

**If one task lags:**
- Use single-task training for that task
- Check data quality for that task
- Increase training epochs

## Comparison with Post-Training Evaluation

### During Training (Current)
```python
# Evaluated automatically every eval_every steps
# Metrics logged to W&B
# Uses eval_max_samples samples (default 50)
```

### Post-Training (run_benchmark.py)
```bash
# Full evaluation on test set
python run_benchmark.py --model finetuned --track both
```

**Recommendation:** Use both!
- **During training:** Monitor progress, debug issues
- **Post-training:** Final comprehensive evaluation

## Technical Details

### Loss Function
- **Training:** Cross-entropy loss on all tokens
- **Applied to:** Assistant response tokens only (not user prompt)
- **Computed:** Every forward pass

### Evaluation
- **Frequency:** Every `eval_every` training steps
- **Samples:** `eval_max_samples` from dev set
- **Mode:** Greedy sampling (temperature=0)
- **Extraction:** Uses `FINAL ANSWER:` parsing (same as GPSBench)

### Integration
The evaluator:
1. Loads examples from dataset builder
2. Generates predictions using Tinker's sampling client
3. Extracts final answers using GPSBench's `extract_final_answer()`
4. Evaluates using task-specific logic from `evaluation/task_evaluators.py`
5. Reports metrics to W&B

## Example W&B Dashboard

```
Metrics you'll see:

├── train/loss              # Cross-entropy training loss
├── eval/loss              # Cross-entropy eval loss
└── gpsbench_eval/
    ├── accuracy           # Overall task accuracy
    ├── count              # Number of evaluated samples
    ├── format_conversion_accuracy
    ├── format_conversion_count
    ├── distance_calculation_accuracy
    ├── distance_calculation_count
    ├── place_association_accuracy
    └── ...
```

## Performance Impact

**With task evaluation enabled (default):**
- Adds ~10-20% to training time
- Evaluates 50 samples every 500 steps
- ~5-10 seconds per evaluation

**Optimization options:**
```bash
# Faster training (disable evaluation)
python train.py enable_task_eval=False

# Balance (fewer samples)
python train.py eval_max_samples=20

# Comprehensive (more samples)
python train.py eval_max_samples=200
```

## Summary

| Component | What It Does | When It Runs |
|-----------|-------------|--------------|
| **Cross-Entropy Loss** | Train model to generate correct tokens | Every training step |
| **Task Accuracy** | Measure task performance | Every eval_every steps |
| **GPSBench Eval** | Full test set evaluation | After training |

**Best practice:**
1. Enable task evaluation during training (default)
2. Monitor W&B for accuracy trends
3. Run full GPSBench evaluation after training

## See Also

- **`DATASET_LOADING_GUIDE.md`** - Data loading options
- **`TRAINING_OPTIONS.md`** - Training configurations
- **`run_benchmark.py`** - Full GPSBench evaluation
- **`evaluation/task_evaluators.py`** - Evaluation logic source
