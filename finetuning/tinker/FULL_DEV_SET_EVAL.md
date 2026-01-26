# Full Dev Set Evaluation - Configuration

## Changes Made

Updated the GPSBench fine-tuning pipeline to evaluate on the **full dev set** instead of a subset.

### Before
```python
eval_max_samples: int = 170  # Only 170 samples (~10 per task)
```

### After
```python
eval_max_samples: int | None = None  # None = use full dev set
```

## Dev Set Statistics

**Total dev examples**: ~6,118 across 17 tasks
**Examples per task**: ~340-360 each

### Task Breakdown
| Task | Dev Examples |
|------|-------------|
| area_perimeter | 340 |
| bearing_computation | 340 |
| boundary_analysis | 340 |
| bounding_box | 340 |
| coordinate_system_transformation | 339 |
| distance_calculation | 340 |
| format_conversion | 340 |
| interpolation | 340 |
| missing_data_handling | 340 |
| name_disambiguation | 340 |
| place_association | 339 |
| proximity | 340 |
| relative_position | 680 |
| route_analysis | 340 |
| route_geometry | 340 |
| spatial_patterns | 340 |
| terrain_classification | 340 |
| **TOTAL** | **6,118** |

## Configuration Options

You can control evaluation size via `eval_max_samples`:

### Option 1: Full Dev Set (Default)
```python
eval_max_samples=None
```
- **Samples**: All ~6,118 examples
- **Per task**: ~340-360 examples
- **Pros**: Most accurate metrics, comprehensive coverage
- **Cons**: Slower evaluation (~1-2 minutes per eval)
- **Best for**: Final training runs, model comparison

### Option 2: Balanced Sampling
```python
eval_max_samples=340
```
- **Samples**: 340 total (~20 per task)
- **Pros**: Faster evaluation, still good coverage
- **Cons**: Less comprehensive than full set
- **Best for**: Development iterations

### Option 3: Quick Sampling
```python
eval_max_samples=170
```
- **Samples**: 170 total (~10 per task)
- **Pros**: Very fast evaluation
- **Cons**: Less stable metrics
- **Best for**: Quick experiments, debugging

## Evaluation Frequency

Current setting in `run_track_training.sh`:
```bash
eval_every=$STEPS_PER_EPOCH  # ~573 steps = once per epoch
```

**With full dev set**:
- Eval time: ~1-2 minutes per evaluation
- Total evals per training: 3 (once per epoch)
- Total eval time: ~3-6 minutes over full training

This is **reasonable** - full dev set evaluation adds minimal overhead (~1-2% of total training time).

## Expected Output

When training starts, you'll see:

```
============================================================
GPSBench Evaluation Setup (stratified sampling)
============================================================
Total dev examples: 6118
Number of tasks: 17
Using FULL dev set (all 6118 examples)

Task distribution in eval set:
  area_perimeter: 340 examples
  bearing_computation: 340 examples
  boundary_analysis: 340 examples
  bounding_box: 340 examples
  coordinate_system_transformation: 339 examples
  distance_calculation: 340 examples
  format_conversion: 340 examples
  interpolation: 340 examples
  missing_data_handling: 340 examples
  name_disambiguation: 340 examples
  place_association: 339 examples
  proximity: 340 examples
  relative_position: 680 examples
  route_analysis: 340 examples
  route_geometry: 340 examples
  spatial_patterns: 340 examples
  terrain_classification: 340 examples
============================================================
```

## W&B Metrics

With full dev set evaluation, you'll get comprehensive per-task metrics:

```
gpsbench_eval/accuracy: 0.95              # Overall accuracy
gpsbench_eval/count: 6118                 # Total examples

# Per-task metrics (all 17 tasks)
gpsbench_eval/area_perimeter_accuracy: 0.98
gpsbench_eval/area_perimeter_count: 340

gpsbench_eval/bearing_computation_accuracy: 0.96
gpsbench_eval/bearing_computation_count: 340

gpsbench_eval/distance_calculation_accuracy: 1.0
gpsbench_eval/distance_calculation_count: 340

# ... all 17 tasks
```

## Performance Considerations

### Tinker API Limits
- Evaluation makes ~6,118 API calls per eval
- Tinker handles this efficiently with async processing
- Batched inference keeps latency reasonable

### Cost Estimate
Assuming Tinker pricing (varies by model):
- Full dev set eval: ~6,118 tokens generated
- Per eval cost: ~$X (check Tinker pricing)
- Per training run (3 epochs): ~$3X

### Time Estimate
- Full dev set eval: ~1-2 minutes
- Quick eval (170 samples): ~10-20 seconds
- Difference: ~90-110 seconds per eval
- Over 3 epochs: ~4-6 minutes additional time

## Recommendation

**Use full dev set evaluation** for:
- ✅ Final training runs
- ✅ Model comparison/selection
- ✅ Publication/benchmark results
- ✅ Understanding per-task performance

**Use sampled evaluation** (170-340) for:
- ✅ Quick experiments
- ✅ Hyperparameter tuning
- ✅ Debugging data/code issues
- ✅ Rapid iteration

## Usage

### Full Dev Set (Default)
```bash
cd finetuning/tinker
./run_track_training.sh both
```

### Custom Sample Size
```bash
cd finetuning/tinker
python train.py \
    dataset_mode=split_files \
    track=both \
    eval_max_samples=340  # 20 per task
```

### Disable Task Evaluation
```bash
python train.py \
    dataset_mode=split_files \
    track=both \
    enable_task_eval=False  # Faster training, no task metrics
```

## Files Modified

1. `train.py:110` - Changed `eval_max_samples` default to `None`
2. `gpsbench_evaluators.py:137` - Updated to handle `None` (use all samples)
3. `gpsbench_evaluators.py:318-325` - Added logic for full dev set
