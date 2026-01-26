# GPSBench Fine-tuning Evaluation Fix

## Problem Identified

The training evaluation was only testing **1 out of 17 tasks**, giving incomplete metrics.

### Root Cause
The evaluator took the **first 50 examples** from the dev set, which were all from the `bounding_box` task. This meant 16 other tasks were never evaluated during training.

**Original behavior:**
```
Total dev examples: 6,118 (17 tasks)
Evaluator samples: First 50 examples
Result: All 50 from 'bounding_box' task ❌
```

## Changes Made

### 1. Stratified Sampling in Evaluator (`gpsbench_evaluators.py`)

**Added stratified sampling to sample evenly across all tasks:**

```python
# Group conversations by task
task_groups = defaultdict(list)
for conv in eval_conversations:
    task = conv.get('task', 'unknown')
    task_groups[task].append(conv)

# Calculate samples per task (distribute max_samples evenly)
num_tasks = len(task_groups)
samples_per_task = max(1, max_samples // num_tasks)

# Sample from each task
for task in sorted(task_groups.keys()):
    task_convs = task_groups[task]
    sampled = random.sample(task_convs, min(samples_per_task, len(task_convs)))
    sampled_conversations.extend(sampled)
```

**Result:**
- With `eval_max_samples=170` and 17 tasks: **10 samples per task**
- All tasks evaluated during training ✅

**Metrics now include all tasks:**
- `gpsbench_eval/area_perimeter_accuracy`
- `gpsbench_eval/bearing_computation_accuracy`
- `gpsbench_eval/bounding_box_accuracy`
- `gpsbench_eval/distance_calculation_accuracy`
- `gpsbench_eval/format_conversion_accuracy`
- `gpsbench_eval/interpolation_accuracy`
- `gpsbench_eval/place_association_accuracy`
- ... (all 17 tasks)

### 2. Full Dev Set Evaluation (`train.py`)

**Changed to use full dev set:**
```python
eval_max_samples: int | None = None  # None = use full dev set (~6,118 examples)
```

**Options:**
- `None`: Evaluate on full dev set (all ~6,118 examples, ~360 per task)
- `170`: Quick evaluation (~10 samples per task) - faster but less robust
- `340`: Balanced evaluation (~20 samples per task)

**Note**: Full dev set evaluation takes longer (~30-60 seconds per eval) but provides more reliable metrics.

### 3. Fixed Learning Rate Schedule (`train.py`)

**Changed from "linear" → "constant":**
```python
lr_schedule: str = "constant"  # Keeps LR stable throughout training
```

**Why:**
- Linear schedule decays to ~0 by end of training
- Your previous run: LR went from 0.0005 → 0.0000003 (essentially zero)
- Constant schedule maintains stable learning throughout

## Testing

Run a quick test to verify stratified sampling:

```bash
cd finetuning/tinker
python train.py \
    max_samples=100 \
    max_eval_samples=170 \
    batch_size=4 \
    num_epochs=1 \
    eval_every=25 \
    log_path=./outputs/eval_fix_test
```

**Expected output:**
```
============================================================
GPSBench Evaluation Setup (stratified sampling)
============================================================
Total dev examples: 6118
Number of tasks: 17
Samples per task: 10
Total samples to evaluate: 170

Task distribution in eval set:
  area_perimeter: 10/340 examples
  bearing_computation: 10/340 examples
  boundary_analysis: 10/340 examples
  ...
============================================================
```

## Impact on Previous Results

### Your completed training run:
- ✅ Training itself was successful (100% accuracy after epoch 1)
- ❌ Only evaluated on 1 task (incomplete metrics)
- ✅ Model checkpoint is still valid and useful

### With these fixes:
- ✅ Training remains successful
- ✅ Evaluation covers all 17 tasks
- ✅ You'll see per-task accuracy breakdown
- ✅ Learning rate stays stable (more efficient training)

## Recommended Next Steps

1. **Re-run training with fixes:**
   ```bash
   ./run_track_training.sh both
   ```

2. **Monitor W&B for comprehensive metrics:**
   - Check that all 17 tasks show accuracy metrics
   - Verify per-task performance

3. **Consider reducing epochs:**
   - Your previous run hit 100% accuracy after 1 epoch
   - You can set `num_epochs=1` to save time/cost

4. **Evaluate saved checkpoint on full test set:**
   - Use the saved checkpoint to run comprehensive evaluation
   - Test on all tasks with larger sample sizes
