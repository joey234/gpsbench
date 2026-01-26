# GPSBench Tinker Training Options

Complete guide to all training options available with the updated Tinker setup.

## Quick Reference

```bash
# 1. Quick test (5-10 min)
./run_example.sh

# 2. Full multi-task (2-4 hours)
./run_full_training.sh

# 3. Single task (20-40 min)
./run_single_task.sh task1_format_conversion pure_gps

# 4. Track-specific (1-2 hours)
./run_track_training.sh pure_gps

# 5. Custom training
python train.py <config_overrides>
```

## Training Scripts

### 1. `run_example.sh` - Quick Test
**Purpose:** Verify setup works correctly
**Time:** ~5-10 minutes
**Data:** 100 training samples, 20 eval samples
**Config:** 1 epoch, batch_size=4

```bash
./run_example.sh
```

**When to use:**
- First time setup verification
- Testing configuration changes
- Quick iteration on prompts

---

### 2. `run_full_training.sh` - Multi-Task Training
**Purpose:** Train on all 38 GPSBench tasks
**Time:** ~2-4 hours
**Data:** 15,775 training samples across 38 tasks
**Config:** 3 epochs, batch_size=64, lora_rank=64

```bash
./run_full_training.sh
```

**When to use:**
- Building a general GPS assistant
- Multi-task learning across all capabilities
- Production deployment

---

### 3. `run_single_task.sh` - Single Task Expert
**Purpose:** Train on one specific task
**Time:** ~20-40 minutes per task
**Config:** 5 epochs, batch_size=32, lora_rank=64

```bash
# Syntax
./run_single_task.sh <task_name> <track>

# Examples
./run_single_task.sh task1_format_conversion pure_gps
./run_single_task.sh task1_place_association applied
./run_single_task.sh task4_distance_calculation pure_gps
```

**When to use:**
- Specialized task performance needed
- Creating task-specific experts
- Debugging specific task performance
- Faster iteration on one task

**Available tasks:**
- Pure GPS: task1, task3, task4, task5, task6, task7, task8, task9, task10
- Applied: task1, task3, task5, task6, task7, task8, task9, task11, task13

---

### 4. `run_track_training.sh` - Track-Specific Training
**Purpose:** Train on all tasks from one track
**Time:** ~1-2 hours per track
**Config:** 3 epochs, batch_size=64, lora_rank=64

```bash
# Syntax
./run_track_training.sh <track>

# Examples
./run_track_training.sh pure_gps   # 9 Pure GPS tasks
./run_track_training.sh applied    # 9 Applied tasks
./run_track_training.sh both       # All 38 tasks (same as run_full_training.sh)
```

**When to use:**
- Focused on GPS operations vs. location grounding
- Comparing track performance
- Building track-specific models

---

### 5. Custom Training - Direct Python
**Purpose:** Full control over all parameters
**Time:** Variable
**Config:** Fully customizable

```bash
python train.py <config_overrides>
```

**When to use:**
- Advanced experimentation
- Custom hyperparameter tuning
- Non-standard configurations

## Configuration Options

### Dataset Loading

#### Instruction Files Mode (Default)
```bash
python train.py
# OR
python train.py dataset_mode=instruction_files
```
- Uses `../data/train_instructions.json`
- 15,775 samples, 38 tasks
- Fast loading, pre-converted

#### Split Files Mode
```bash
python train.py dataset_mode=split_files track=<track> tasks=<tasks>
```
- Loads from `data/track_*/splits/`
- Supports task/track filtering
- Single or custom multi-task

### Dataset Filters

#### Track Selection (split_files mode only)
```bash
track=pure_gps     # Pure GPS tasks only (9 tasks)
track=applied      # Applied tasks only (9 tasks)
track=both         # All tasks (18 total)
```

#### Task Selection (split_files mode only)
```bash
# Single task
tasks="['task1_format_conversion']"

# Multiple tasks
tasks="['task1_format_conversion', 'task4_distance_calculation', 'task5_bearing_computation']"
```

#### Task Filtering (instruction_files mode only)
```bash
# Filter by task type
filter_tasks="['format_conversion']"
filter_tasks="['format_conversion', 'distance_calculation']"
```

### Model Selection

```bash
# Default (recommended)
model_name="meta-llama/Llama-3.2-3B"

# Smaller (faster, lower memory)
model_name="meta-llama/Llama-3.2-1B"

# Larger (better performance)
model_name="meta-llama/Llama-3.1-8B"
model_name="Qwen/Qwen3-8B-Base"
```

See `SUPPORTED_MODELS.md` for full list.

### Hyperparameters

```bash
# Learning rate
learning_rate=5e-4    # Default
learning_rate=3e-4    # For larger models
learning_rate=1e-3    # For single-task training

# Batch size
batch_size=64         # Default (multi-task)
batch_size=32         # Single-task
batch_size=128        # Large models
batch_size=4          # Quick testing

# LoRA configuration
lora_rank=64          # Default
lora_rank=32          # Lower memory
lora_rank=128         # More capacity

# Training epochs
num_epochs=3          # Default (multi-task)
num_epochs=5          # Single-task
num_epochs=1          # Quick testing
```

### Data Limits

```bash
# Limit training samples
max_samples=1000           # Train on 1000 samples
max_eval_samples=200       # Eval on 200 samples

# Useful for:
# - Quick iteration
# - Testing new tasks
# - Debugging
```

### Checkpointing

```bash
# Checkpoint frequency
save_every=500         # Save every 500 steps
eval_every=500         # Evaluate every 500 steps

# For faster iteration:
save_every=100
eval_every=100
```

### Logging

```bash
# Weights & Biases
wandb_project=gpsbench-finetuning
wandb_name=my_experiment

# Output directory
log_path=./outputs/my_run
```

## Example Configurations

### 1. Quick Smoke Test
```bash
python train.py \
    max_samples=10 \
    max_eval_samples=5 \
    batch_size=1 \
    num_epochs=1 \
    eval_every=5 \
    save_every=10
```

### 2. Format Conversion Expert
```bash
python train.py \
    dataset_mode=split_files \
    track=pure_gps \
    tasks="['task1_format_conversion']" \
    learning_rate=1e-3 \
    batch_size=32 \
    lora_rank=64 \
    num_epochs=10 \
    log_path=./outputs/format_expert
```

### 3. Place Name Specialist
```bash
python train.py \
    dataset_mode=split_files \
    track=applied \
    tasks="['task1_place_association']" \
    learning_rate=5e-4 \
    batch_size=32 \
    num_epochs=8 \
    log_path=./outputs/place_specialist
```

### 4. Geometry Tasks Multi-Task
```bash
python train.py \
    dataset_mode=split_files \
    track=pure_gps \
    tasks="['task4_distance_calculation', 'task5_bearing_computation', 'task7_area_perimeter']" \
    batch_size=48 \
    lora_rank=64 \
    num_epochs=5 \
    log_path=./outputs/geometry_multi
```

### 5. Pure GPS Track
```bash
python train.py \
    dataset_mode=split_files \
    track=pure_gps \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    num_epochs=3 \
    wandb_project=gpsbench-pure-gps
```

### 6. Applied Track
```bash
python train.py \
    dataset_mode=split_files \
    track=applied \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    num_epochs=3 \
    wandb_project=gpsbench-applied
```

### 7. Full Multi-Task (Both Modes)
```bash
# Mode 1: Using instruction files (default)
python train.py \
    batch_size=64 \
    num_epochs=3

# Mode 2: Using split files
python train.py \
    dataset_mode=split_files \
    track=both \
    batch_size=64 \
    num_epochs=3
```

### 8. Small Model (Low Memory)
```bash
python train.py \
    model_name="meta-llama/Llama-3.2-1B" \
    batch_size=16 \
    lora_rank=32 \
    max_samples=5000
```

### 9. Large Model (High Performance)
```bash
python train.py \
    model_name="meta-llama/Llama-3.1-8B" \
    learning_rate=3e-4 \
    batch_size=128 \
    lora_rank=128 \
    num_epochs=5
```

## Recommended Configurations

### For Experimentation
```bash
# Fast iteration
max_samples=100
batch_size=4
num_epochs=1
eval_every=10
```

### For Single-Task Specialists
```bash
dataset_mode=split_files
tasks="['task_name']"
learning_rate=1e-3
batch_size=32
num_epochs=8-10
```

### For Multi-Task General Models
```bash
dataset_mode=instruction_files  # OR split_files track=both
learning_rate=5e-4
batch_size=64
lora_rank=64
num_epochs=3-5
```

### For Production Deployment
```bash
model_name="meta-llama/Llama-3.1-8B"
dataset_mode=instruction_files
learning_rate=3e-4
batch_size=128
lora_rank=128
num_epochs=5
save_every=500
eval_every=500
wandb_project=gpsbench-production
```

## All Available Tasks

### Pure GPS Track (9 tasks)
1. **task1_format_conversion** - DMS ↔ Decimal
2. **task3_coordinate_system_transformation** - CRS transforms
3. **task4_distance_calculation** - Haversine distance
4. **task5_bearing_computation** - Bearing angles
5. **task6_interpolation** - Point interpolation
6. **task7_area_perimeter** - Polygon calculations
7. **task8_bounding_box** - Bounding boxes
8. **task9_route_geometry** - Route analysis
9. **task10_relative_position** - Relative positions

### Applied Track (9 tasks)
1. **task1_place_association** - Coord → Location
2. **task3_name_disambiguation** - Place name resolution
3. **task5_relative_position** - Spatial relations
4. **task6_proximity** - Nearby locations
5. **task7_route_analysis** - Route analytics
6. **task8_boundary_analysis** - Boundary operations
7. **task9_spatial_patterns** - Pattern detection
8. **task11_missing_data** - Missing coord handling
9. **task13_terrain_classification** - Terrain types

## Summary Table

| Script | Time | Data Size | Use Case |
|--------|------|-----------|----------|
| `run_example.sh` | 5-10 min | 100 samples | Quick test |
| `run_single_task.sh` | 20-40 min | 1 task | Specialist |
| `run_track_training.sh` | 1-2 hrs | 9 tasks | Track-specific |
| `run_full_training.sh` | 2-4 hrs | 38 tasks | General model |

## See Also

- **`README.md`** - Overview and setup
- **`DATASET_LOADING_GUIDE.md`** - Dataset loading details
- **`SETUP_COMPLETE.md`** - Setup summary
- **`SUPPORTED_MODELS.md`** - Available models
