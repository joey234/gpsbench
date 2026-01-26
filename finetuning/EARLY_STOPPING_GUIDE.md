# Early Stopping Implementation Guide

**Date:** November 12, 2025
**Feature:** Automatic early stopping to prevent overfitting
**Status:** ✅ Implemented in `train_single_task.py`

---

## What is Early Stopping?

Early stopping automatically stops training when the model stops improving on the validation set. This:
- **Prevents overfitting** - Stops before the model memorizes training data
- **Saves time** - No need to wait for all epochs if converged early
- **Saves cost** - Reduces GPU hours on cloud platforms
- **Auto-selects best model** - Loads the checkpoint with best validation loss

---

## How It Works

### During Training

1. **After each epoch**, the model is evaluated on the dev set
2. **Validation loss is tracked** and compared to previous best
3. **If loss improves** by at least the threshold → Reset patience counter
4. **If loss doesn't improve** → Increment patience counter
5. **If patience exceeded** → Stop training and load best checkpoint

### Example Training Flow

```
Epoch 1: val_loss = 0.850 → Best so far! (patience: 0/2)
Epoch 2: val_loss = 0.620 → Improved by 0.230! (patience: 0/2)
Epoch 3: val_loss = 0.615 → Improved by 0.005! (patience: 0/2)
Epoch 4: val_loss = 0.618 → No improvement (patience: 1/2)
Epoch 5: val_loss = 0.621 → No improvement (patience: 2/2)

⚠️  Early stopping triggered!
✓ Loading best model from Epoch 3 (val_loss = 0.615)
✓ Training stopped at Epoch 5 (instead of 10)
✓ Saved 50% training time!
```

---

## Usage

### Default Settings (Recommended)

```bash
python scripts/train_single_task.py \
    --train-file data/train_single_task_large.json \
    --dev-file data/dev_single_task_large.json \
    --num-epochs 10 \
    --auto-adjust
```

**Default early stopping:**
- Patience: 2 evaluations (2 epochs with no improvement)
- Threshold: 0.001 (must improve by at least 0.001)
- Auto-enabled: Yes

### Custom Patience

```bash
# More patient (waits longer before stopping)
python scripts/train_single_task.py \
    --train-file data/train_single_task_large.json \
    --dev-file data/dev_single_task_large.json \
    --num-epochs 10 \
    --early-stopping-patience 5 \
    --auto-adjust
```

**Use higher patience when:**
- Training is noisy (loss fluctuates a lot)
- You want to be more conservative
- Dataset is very large

### Custom Threshold

```bash
# Require larger improvement to count as "better"
python scripts/train_single_task.py \
    --train-file data/train_single_task_large.json \
    --dev-file data/dev_single_task_large.json \
    --num-epochs 10 \
    --early-stopping-threshold 0.01 \
    --auto-adjust
```

**Use higher threshold when:**
- You want to avoid stopping on small fluctuations
- Validation loss is noisy
- You want significant improvement only

### Disable Early Stopping

```bash
# Train for exact number of epochs (not recommended)
python scripts/train_single_task.py \
    --train-file data/train_single_task_large.json \
    --dev-file data/dev_single_task_large.json \
    --num-epochs 10 \
    --no-early-stopping
```

**Use when:**
- Debugging or experimenting
- You know exact epochs needed
- Running ablation studies

---

## Parameters

### `--early-stopping-patience` (default: 2)

**Definition:** Number of evaluations without improvement before stopping

**Recommendations:**
| Dataset Size | Recommended Patience |
|--------------|---------------------|
| Small (<10K) | 3-5 |
| Medium (10K-100K) | 2-3 |
| Large (100K+) | 1-2 |

**Why:** Large datasets converge faster and are less noisy, so lower patience is safe.

### `--early-stopping-threshold` (default: 0.001)

**Definition:** Minimum improvement required to count as "better"

**Recommendations:**
| Loss Range | Recommended Threshold |
|------------|----------------------|
| > 1.0 | 0.01 |
| 0.1 - 1.0 | 0.001 |
| < 0.1 | 0.0001 |

**Why:** Smaller losses need smaller thresholds to detect meaningful improvements.

### `--no-early-stopping` (flag)

**Definition:** Completely disable early stopping

**When to use:**
- Never for production training
- Only for experiments where you need exact epoch counts
- When you want to see full training curve

---

## Benefits

### 1. Prevents Overfitting

**Without early stopping:**
```
Epoch 1: val_loss = 0.850, accuracy = 45%
Epoch 2: val_loss = 0.620, accuracy = 72%
Epoch 3: val_loss = 0.615, accuracy = 78% ← Best model
Epoch 4: val_loss = 0.618, accuracy = 77%
Epoch 5: val_loss = 0.621, accuracy = 76%
Epoch 6: val_loss = 0.630, accuracy = 74% ← Worse!
Epoch 7: val_loss = 0.645, accuracy = 72%
...
```

**With early stopping:**
```
Epoch 1: val_loss = 0.850, accuracy = 45%
Epoch 2: val_loss = 0.620, accuracy = 72%
Epoch 3: val_loss = 0.615, accuracy = 78% ← Best model
Epoch 4: val_loss = 0.618, accuracy = 77% (patience: 1/2)
Epoch 5: val_loss = 0.621, accuracy = 76% (patience: 2/2)
✓ Stopped! Using Epoch 3 model (78% accuracy)
```

### 2. Saves Training Time

**Example with 250K dataset:**

| Scenario | Epochs Trained | Time | GPU Cost |
|----------|----------------|------|----------|
| No early stopping | 10 (all) | 20h | $40-60 |
| Early stopping @ Epoch 5 | 5 | 10h | $20-30 |
| **Savings** | **50%** | **10h** | **$20-30** |

### 3. Auto-Selects Best Model

**You don't need to:**
- Manually check which epoch was best
- Re-run training with different epoch counts
- Worry about picking wrong checkpoint

**The system automatically:**
- Tracks validation loss every epoch
- Saves best checkpoint
- Loads best model at the end

---

## Expected Behavior

### Normal Training (Converges Well)

```
Starting Training...
================================================================================

Epoch 1/10:
  train_loss: 1.850
  eval_loss: 0.850
  ✓ New best model! (patience: 0/2)

Epoch 2/10:
  train_loss: 0.720
  eval_loss: 0.620
  ✓ Improved by 0.230! (patience: 0/2)

Epoch 3/10:
  train_loss: 0.680
  eval_loss: 0.615
  ✓ Improved by 0.005! (patience: 0/2)

Epoch 4/10:
  train_loss: 0.650
  eval_loss: 0.618
  ⚠️  No improvement (patience: 1/2)

Epoch 5/10:
  train_loss: 0.630
  eval_loss: 0.621
  ⚠️  No improvement (patience: 2/2)

⚠️  Early stopping triggered after 2 evaluations without improvement
✓ Loading best model from checkpoint-3
✓ Best eval_loss: 0.615

Training Complete!
Total epochs: 5/10 (stopped early)
Best checkpoint: checkpoint-3
```

### Training Without Improvement (Data/Model Issue)

```
Epoch 1/10:
  eval_loss: 2.850
  ✓ New best model!

Epoch 2/10:
  eval_loss: 2.845
  ✓ Improved by 0.005!

Epoch 3/10:
  eval_loss: 2.849
  ⚠️  No improvement (patience: 1/2)

Epoch 4/10:
  eval_loss: 2.851
  ⚠️  No improvement (patience: 2/2)

⚠️  Early stopping triggered
⚠️  WARNING: Model stopped very early with high loss
⚠️  Possible issues:
   - Learning rate too high/low
   - Data quality problems
   - Model capacity issues
```

**Action:** Check training logs and adjust hyperparameters

---

## Troubleshooting

### Issue 1: Stops Too Early (After 1-2 Epochs)

**Symptoms:**
```
Epoch 1: eval_loss = 1.2
Epoch 2: eval_loss = 1.18
Epoch 3: eval_loss = 1.19 (patience: 1/2)
Epoch 4: eval_loss = 1.20 (patience: 2/2)
⚠️  Stopped at epoch 4
```

**Solutions:**

1. **Increase patience:**
   ```bash
   --early-stopping-patience 5
   ```

2. **Lower threshold:**
   ```bash
   --early-stopping-threshold 0.0001
   ```

3. **Check learning rate:** May be too high, causing oscillation

### Issue 2: Never Stops (Goes to Max Epochs)

**Symptoms:**
```
Epoch 1: eval_loss = 1.2
Epoch 2: eval_loss = 1.18
Epoch 3: eval_loss = 1.16
...
Epoch 10: eval_loss = 0.95
⚠️  No early stopping triggered
```

**Solutions:**

1. **This is actually OK!** Model is continuously improving
2. **Increase max epochs:** Model might need more training
   ```bash
   --num-epochs 15
   ```
3. **Check if overfitting:** Compare train_loss vs eval_loss

### Issue 3: Loss Fluctuates, Stops Randomly

**Symptoms:**
```
Epoch 1: eval_loss = 1.2
Epoch 2: eval_loss = 0.9
Epoch 3: eval_loss = 1.1  ← Goes up
Epoch 4: eval_loss = 0.85
Epoch 5: eval_loss = 0.95 ← Goes up again
⚠️  Stopped due to fluctuation
```

**Solutions:**

1. **Increase patience:**
   ```bash
   --early-stopping-patience 5
   ```

2. **Increase threshold:**
   ```bash
   --early-stopping-threshold 0.01
   ```

3. **Check batch size:** Too small may cause noise
   ```bash
   --batch-size 8  # Increase from 4
   ```

4. **Lower learning rate:**
   ```bash
   --learning-rate 5e-5  # From 1e-4
   ```

---

## Comparison: With vs Without

### Scenario 1: Model Converges Well

| Metric | Without Early Stopping | With Early Stopping |
|--------|----------------------|-------------------|
| Epochs trained | 10 | 5 |
| Training time | 20 hours | 10 hours |
| GPU cost | $40 | $20 |
| Best val_loss | 0.615 @ Epoch 3 | 0.615 @ Epoch 3 |
| Final model | Epoch 10 (overfitted, 0.650 loss) | Epoch 3 (best, 0.615 loss) |
| **Result** | ❌ Worse model, wasted time | ✅ Best model, saved time |

### Scenario 2: Model Needs More Training

| Metric | Without Early Stopping | With Early Stopping |
|--------|----------------------|-------------------|
| Epochs trained | 10 | 10 |
| Training time | 20 hours | 20 hours |
| GPU cost | $40 | $40 |
| Best val_loss | 0.550 @ Epoch 10 | 0.550 @ Epoch 10 |
| **Result** | ✅ Good model | ✅ Good model (same) |

**Early stopping doesn't hurt when model keeps improving!**

---

## Best Practices

### 1. Use Default Settings for First Run

```bash
# Let early stopping do its job
python scripts/train_single_task.py \
    --train-file data/train_single_task_large.json \
    --dev-file data/dev_single_task_large.json \
    --num-epochs 10 \
    --auto-adjust
```

### 2. Set Max Epochs High

```bash
# Set high max epochs, let early stopping find optimal
--num-epochs 20
```

Don't worry about wasting time - early stopping will stop when appropriate!

### 3. Monitor First Few Epochs

```bash
# Watch the first 3-5 epochs
tail -f models/qwen3-4b-place-association-lora*/training_log.txt
```

If loss is improving steadily → Trust early stopping
If loss is stuck or fluctuating → Adjust hyperparameters

### 4. Combine with Auto-Adjust

```bash
# Best combo for large datasets
--auto-adjust \
--early-stopping-patience 2
```

Auto-adjust sets reasonable epoch count, early stopping finds exact stopping point.

### 5. Check Training Logs

After training completes:
```python
import json

with open('models/qwen3-4b-place-association-lora/trainer_state.json') as f:
    state = json.load(f)

# Check why it stopped
if state['epoch'] < args.num_epochs:
    print(f"Early stopping triggered at epoch {state['epoch']}")
    print(f"Best checkpoint: {state['best_model_checkpoint']}")
else:
    print("Trained for all epochs (no early stopping)")
```

---

## Summary

### What You Get

✅ **Automatic overfitting prevention**
✅ **50-75% time savings** (when model converges early)
✅ **Best model auto-selected**
✅ **No manual checkpoint selection needed**
✅ **Reduced cloud costs**

### Recommended Settings

| Dataset Size | Patience | Threshold | Max Epochs |
|--------------|----------|-----------|------------|
| <10K | 3-5 | 0.001 | 20 |
| 10K-100K | 2-3 | 0.001 | 10-15 |
| >100K | 1-2 | 0.001 | 10 |

### When to Disable

❌ **Never for production**
✅ For experiments requiring exact epoch counts
✅ For debugging specific epoch behavior
✅ For creating training curves / ablations

---

**Implementation:** `finetuning/scripts/train_single_task.py:220-232`
**Created:** November 12, 2025
**Status:** ✅ Production Ready
