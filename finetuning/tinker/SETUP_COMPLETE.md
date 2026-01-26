# ✅ GPSBench Tinker Setup Complete!

Your Tinker fine-tuning environment is ready to use.

## What's Configured

### ✅ Training Infrastructure
- **tinker SDK** (v0.6.3) - Connected to Tinker cloud platform
- **tinker-cookbook** (v0.1.0) - High-level supervised learning interface
- **API Keys** - TINKER_API_KEY and HF_TOKEN configured in `.env`
- **Training Data** - GPSBench data ready at `../data/` (train/dev/test)

### ✅ Training Script
- `train.py` - Main training script using tinker-cookbook
- Automatic dataset loading from GPSBench JSON files
- Configurable via command-line arguments (chz framework)
- Distributed cloud GPU training (no local GPU needed)

### ✅ Quick Start Scripts
- `run_example.sh` - Quick test run (100 samples, 1 epoch, ~5-10 min)
- `run_full_training.sh` - Full training (all data, 3 epochs, ~2-4 hours)

## Supported Models

Tinker supports these base models (NOT Qwen2.5):

**Recommended:**
- `meta-llama/Llama-3.2-3B` ← **Default** (3B, same size as Qwen2.5-3B)
- `meta-llama/Llama-3.2-1B` (1B, smaller/faster)
- `meta-llama/Llama-3.1-8B` (8B, better performance)

**Also available:**
- `Qwen/Qwen3-8B-Base` (8B, multilingual)
- `Qwen/Qwen3-30B-A3B-Base` (30B MoE)
- `meta-llama/Llama-3.1-70B` (70B, best performance)
- `deepseek-ai/DeepSeek-V3.1-Base`

See `SUPPORTED_MODELS.md` for details.

## How to Start Training

### Option 1: Example Run (Recommended First)

Test with a quick training run:

```bash
cd /Users/hungthinh/workspace/GPSBench/finetuning/tinker
./run_example.sh
```

### Option 2: Full Training (All Tasks)

Run complete multi-task training on all 38 tasks:

```bash
./run_full_training.sh
```

### Option 3: Single Task Training

Train on one specific task from GPSBench split files:

```bash
./run_single_task.sh task1_format_conversion pure_gps
```

### Option 4: Track Training

Train on all tasks from a specific track:

```bash
./run_track_training.sh pure_gps  # Pure GPS track (9 tasks)
./run_track_training.sh applied   # Applied track (9 tasks)
./run_track_training.sh both      # Both tracks (all 38 tasks)
```

### Option 5: Custom Configuration

Direct command with your own parameters:

```bash
# Using instruction files (default, all tasks)
python train.py \
    model_name="meta-llama/Llama-3.2-3B" \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    num_epochs=3

# Using split files (specific tasks/tracks)
python train.py \
    dataset_mode=split_files \
    track=pure_gps \
    tasks="['task1_format_conversion', 'task4_distance_calculation']" \
    batch_size=32 \
    num_epochs=5
```

## Key Parameters

```bash
python train.py \
    model_name="meta-llama/Llama-3.2-3B" \  # Model to fine-tune
    learning_rate=5e-4 \                     # Learning rate
    batch_size=64 \                          # Batch size
    lora_rank=64 \                           # LoRA rank (capacity)
    num_epochs=3 \                           # Number of epochs
    max_samples=1000 \                       # Limit training samples (optional)
    max_eval_samples=200 \                   # Limit eval samples (optional)
    eval_every=500 \                         # Evaluation frequency (steps)
    save_every=500 \                         # Checkpoint frequency (steps)
    wandb_project=my-experiment \            # Weights & Biases project
    log_path=./outputs/my_run                # Output directory
```

## What Happens During Training

1. **Data Loading**: Loads GPSBench JSON data and converts to conversation format
2. **Model Setup**: Initializes model with LoRA adapters on Tinker cloud
3. **Training**: Distributed training on cloud GPUs with cross-entropy loss
4. **Task Evaluation**: Evaluates task-specific accuracy every 500 steps (**NEW!**)
   - Format conversion: Coordinate matching (0.01° tolerance)
   - Distance/bearing/area: Numerical accuracy (5% relative error)
   - Place association: Location name matching
5. **Checkpointing**: Saves checkpoints every N steps
6. **Logging**: Logs metrics to terminal and Weights & Biases

## Monitoring

### Terminal Output
Real-time progress shown in terminal:
- Step/epoch progress
- Training loss
- Evaluation loss
- Time estimates

### Weights & Biases
View training metrics at:
```
https://wandb.ai/joey234/gpsbench-finetuning
```

**Metrics you'll see:**
- `train/loss` - Training cross-entropy loss
- `eval/loss` - Evaluation cross-entropy loss
- `gpsbench_eval/accuracy` - Overall task accuracy (**NEW!**)
- `gpsbench_eval/format_conversion_accuracy` - Format conversion task
- `gpsbench_eval/distance_calculation_accuracy` - Distance task
- `gpsbench_eval/place_association_accuracy` - Place name task
- And more per-task metrics!

See `EVALUATION_METRICS.md` for details.

## Outputs

Training saves to `./outputs/` with auto-generated names:

```
outputs/
└── gpsbench-Qwen3-4B-64rank-0.0005lr-64batch-2025-12-03-15-42/
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

## Verification Test

Run a minimal smoke test:

```bash
python train.py \
    max_samples=10 \
    max_eval_samples=5 \
    batch_size=1 \
    num_epochs=1 \
    log_path=./outputs/smoke_test
```

Expected output:
- ✓ API connection successful
- ✓ Data loaded
- ✓ Training client created
- ✓ Training starts on cloud GPUs

## Common Issues

### Issue: "base_model X is not supported"

**Solution:** Use a supported model (see `SUPPORTED_MODELS.md`):
```bash
python train.py model_name="meta-llama/Llama-3.2-3B"
```

### Issue: "TINKER_API_KEY not found"

**Solution:** Run setup script:
```bash
./setup_api_keys.sh
```

### Issue: Training data not found

**Solution:** Ensure you're in the tinker directory:
```bash
cd /Users/hungthinh/workspace/GPSBench/finetuning/tinker
ls ../data/train_instructions.json  # Should exist
```

## Next Steps

1. **Run example training** - Verify everything works
2. **Monitor the run** - Check W&B dashboard
3. **Run full training** - Train on all data
4. **Evaluate model** - Test on GPSBench test set
5. **Iterate** - Adjust hyperparameters based on results

## Dataset Loading Modes

Tinker setup now supports **two data loading modes**:

### 1. Instruction Files Mode (Default)
- Uses pre-converted `train_instructions.json` and `dev_instructions.json`
- 15,775 samples across 38 tasks
- Fast loading, verified format
- Best for: Multi-task training on all tasks

### 2. Split Files Mode (New!)
- Loads directly from GPSBench task split files
- `data/track_pure_gps/splits/` - Pure GPS tasks
- `data/track_applied/splits/` - Applied tasks
- Supports:
  - Single-task training
  - Multi-task with custom task selection
  - Track-specific training (pure_gps, applied, or both)

**See `DATASET_LOADING_GUIDE.md` for detailed examples and usage.**

## Available Tasks

### Pure GPS Track (9 tasks)
- task1_format_conversion
- task3_coordinate_system_transformation
- task4_distance_calculation
- task5_bearing_computation
- task6_interpolation
- task7_area_perimeter
- task8_bounding_box
- task9_route_geometry
- task10_relative_position

### Applied Track (9 tasks)
- task1_place_association
- task3_name_disambiguation
- task5_relative_position
- task6_proximity
- task7_route_analysis
- task8_boundary_analysis
- task9_spatial_patterns
- task11_missing_data
- task13_terrain_classification

## Documentation

- `SETUP_COMPLETE.md` - **This file** - Setup summary and quick start
- `EVALUATION_METRICS.md` - **NEW!** Task-specific evaluation metrics
- `DATASET_LOADING_GUIDE.md` - Dataset loading modes and examples
- `TRAINING_OPTIONS.md` - All training configurations
- `SUPPORTED_MODELS.md` - List of Tinker-supported models
- `QUICKSTART.md` - Quick start guide
- `RUNNING_TRAINING.md` - Detailed training guide
- `API_KEYS_GUIDE.md` - API key setup
- `TINKER_SIGNUP.md` - Tinker platform signup

## Support

- **Tinker docs:** https://tinker-docs.thinkingmachines.ai/
- **Cookbook repo:** https://github.com/thinking-machines-lab/tinker-cookbook
- **W&B dashboard:** https://wandb.ai/joey234/gpsbench-finetuning

---

**You're all set!** Run `./run_example.sh` to start your first training job on Tinker cloud. 🚀
