# GPSBench Dataset Loading Guide for Tinker

This guide explains how to load and train on GPSBench data using Tinker's cloud platform.

## Two Loading Modes

### 1. Instruction Files Mode (Default)

Uses pre-converted instruction files:
- `../data/train_instructions.json` (15,775 samples)
- `../data/dev_instructions.json`

**Pros:**
- Fast loading (data already converted)
- Multi-task training (38 tasks combined)
- Tested and verified format

**Usage:**
```bash
python train.py
# OR explicitly
python train.py dataset_mode=instruction_files data_path=../data
```

**Filter specific tasks:**
```bash
python train.py filter_tasks="['format_conversion', 'distance_calculation']"
```

### 2. Split Files Mode

Loads directly from GPSBench task split files in `data/track_pure_gps/splits/` and `data/track_applied/splits/`.

**Pros:**
- Fine-grained task control
- Train on specific tasks or tracks
- Single-task or custom multi-task training

**Usage:**

#### Load all tasks from both tracks:
```bash
python train.py dataset_mode=split_files track=both
```

#### Load only Pure GPS track:
```bash
python train.py dataset_mode=split_files track=pure_gps
```

#### Load only Applied track:
```bash
python train.py dataset_mode=split_files track=applied
```

#### Load specific tasks:
```bash
# Single task
python train.py dataset_mode=split_files track=pure_gps tasks="['task1_format_conversion']"

# Multiple tasks
python train.py dataset_mode=split_files track=both tasks="['task1_format_conversion', 'task4_distance_calculation']"
```

## Quick Start Scripts

### Example Run (Quick Test)
```bash
./run_example.sh
```
- 100 samples, 1 epoch
- ~5-10 minutes
- Tests that everything works

### Full Training (All Tasks, All Data)
```bash
./run_full_training.sh
```
- Uses instruction_files mode
- 15,775 samples, 3 epochs
- ~2-4 hours

### Single Task Training
```bash
./run_single_task.sh task1_format_conversion pure_gps
```
- Train on one specific task
- Higher learning with focused data
- ~20-40 minutes per task

### Track Training
```bash
./run_track_training.sh pure_gps
```
- Train on all tasks from one track
- Pure GPS: ~9 tasks
- Applied: ~9 tasks
- ~1-2 hours per track

## Available Tasks

### Pure GPS Track (data/track_pure_gps/splits/)
- task1_format_conversion
- task3_coordinate_system_transformation
- task4_distance_calculation
- task5_bearing_computation
- task6_interpolation
- task7_area_perimeter
- task8_bounding_box
- task9_route_geometry
- task10_relative_position

### Applied Track (data/track_applied/splits/)
- task1_place_association
- task3_name_disambiguation
- task5_relative_position
- task6_proximity
- task7_route_analysis
- task8_boundary_analysis
- task9_spatial_patterns
- task11_missing_data
- task13_terrain_classification

## Configuration Options

### Dataset Options
```bash
# Mode selection
dataset_mode=instruction_files  # Use pre-converted files (default)
dataset_mode=split_files         # Load from split files

# For instruction_files mode
data_path=../data                           # Path to instruction files
filter_tasks="['format_conversion']"        # Filter by task name

# For split_files mode
gpsbench_root=../../                        # Path to GPSBench root
track=both                                  # pure_gps, applied, or both
tasks="['task1_format_conversion']"         # Specific tasks to load

# Data limits
max_samples=1000                 # Limit training samples
max_eval_samples=200             # Limit eval samples
```

### Training Options
```bash
# Model
model_name="meta-llama/Llama-3.2-3B"  # Tinker supported model

# Hyperparameters
learning_rate=5e-4
batch_size=64
lora_rank=64
num_epochs=3

# Checkpointing
eval_every=500
save_every=500

# Logging
wandb_project=gpsbench-finetuning
log_path=./outputs/my_run
```

## Examples

### 1. Quick smoke test
```bash
python train.py \
    max_samples=10 \
    max_eval_samples=5 \
    batch_size=1 \
    num_epochs=1
```

### 2. Single task (format conversion)
```bash
python train.py \
    dataset_mode=split_files \
    track=pure_gps \
    tasks="['task1_format_conversion']" \
    batch_size=32 \
    num_epochs=5 \
    log_path=./outputs/format_conversion
```

### 3. Multi-task (3 geometric tasks)
```bash
python train.py \
    dataset_mode=split_files \
    track=pure_gps \
    tasks="['task4_distance_calculation', 'task5_bearing_computation', 'task7_area_perimeter']" \
    batch_size=64 \
    num_epochs=3 \
    log_path=./outputs/geometry_tasks
```

### 4. Full Pure GPS track
```bash
python train.py \
    dataset_mode=split_files \
    track=pure_gps \
    batch_size=64 \
    lora_rank=64 \
    num_epochs=3 \
    wandb_project=gpsbench-pure-gps
```

### 5. Full Applied track
```bash
python train.py \
    dataset_mode=split_files \
    track=applied \
    batch_size=64 \
    lora_rank=64 \
    num_epochs=3 \
    wandb_project=gpsbench-applied
```

### 6. Both tracks (equivalent to instruction_files mode)
```bash
python train.py \
    dataset_mode=split_files \
    track=both \
    batch_size=64 \
    num_epochs=3
```

## Monitoring

### Terminal Output
Real-time progress in terminal:
- Step/epoch progress
- Training loss
- Evaluation loss
- Time estimates

### Weights & Biases
View training metrics at:
```
https://wandb.ai/joey234/gpsbench-finetuning
```

## Outputs

Training saves to `./outputs/` with auto-generated names:

```
outputs/
└── gpsbench-meta-llama-Llama-3.2-3B-64rank-0.0005lr-64batch-2025-12-03-15-42/
    ├── checkpoints/
    │   ├── checkpoint-500/
    │   ├── checkpoint-1000/
    │   └── ...
    ├── final_model/
    │   └── lora_weights.safetensors
    ├── logs/
    │   └── training.log
    └── wandb/
```

## Recommendations

### For Experimentation
- Use `dataset_mode=split_files` with specific tasks
- Start with single tasks to understand performance
- Use `max_samples` to iterate quickly

### For Production
- Use `dataset_mode=instruction_files` for full multi-task training
- OR use `dataset_mode=split_files track=both` for same result
- Train for 3-5 epochs
- Monitor W&B for convergence

### For Single-Task Specialists
- Use `dataset_mode=split_files` with one task
- Train for more epochs (5-10)
- Use higher learning rate (1e-3)
- Smaller batch size (16-32) for better generalization

## Troubleshooting

### Issue: "FileNotFoundError: train_instructions.json"
**Solution:** Either:
1. Switch to split_files mode: `dataset_mode=split_files`
2. Or generate instruction files using GPSBench scripts

### Issue: "No split files found"
**Solution:** Ensure you're in the correct directory:
```bash
cd /Users/hungthinh/workspace/GPSBench/finetuning/tinker
ls ../../data/track_pure_gps/splits/  # Should see task files
```

### Issue: "TINKER_API_KEY not found"
**Solution:** Run setup script:
```bash
./setup_api_keys.sh
```

## Next Steps

1. **Choose your training mode** - Single-task or multi-task
2. **Run example training** - Test with `./run_example.sh`
3. **Monitor the run** - Check W&B dashboard
4. **Run full training** - Use appropriate script
5. **Evaluate model** - Test on GPSBench test set

## Support

- **Tinker docs:** https://tinker-docs.thinkingmachines.ai/
- **GPSBench repo:** https://github.com/rohinmanvi/GPSBench
- **W&B dashboard:** https://wandb.ai/joey234/gpsbench-finetuning
