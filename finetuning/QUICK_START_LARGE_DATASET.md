# Quick Start: Large Dataset Training

## TL;DR - 3 Commands

```bash
# 1. Generate 250K training examples (2 minutes)
cd /Users/hungthinh/workspace/GPSBench/finetuning
python generate_large_single_task_dataset.py

# 2. Train model (6-8 hours on GPU)
python scripts/train_single_task.py \
    --train-file data/train_single_task_large.json \
    --dev-file data/dev_single_task_large.json \
    --num-epochs 3 \
    --auto-adjust

# 3. Evaluate (5 minutes)
cd ../evaluation
python evaluate_applied.py \
    --lora-path ../finetuning/models/qwen3-4b-place-association-lora \
    --task place_association
```

**Expected result:** 80-85% accuracy (currently 3%)

**✨ New: Automatic early stopping** - Training will automatically stop when validation loss stops improving, saving time and preventing overfitting!

---

## Dataset Generation Options

### Full Dataset (Recommended for Production)

```bash
# 250K examples, all 32K cities
python generate_large_single_task_dataset.py
```

- Training examples: 250,218
- Training time: 6-8 hours
- Expected accuracy: **80-85%**

### Medium Dataset (Good Balance)

```bash
# 80K examples, major cities only (50K+ population)
python generate_large_single_task_dataset.py --min-population 50000
```

- Training examples: ~80,000
- Training time: 3-4 hours
- Expected accuracy: **70-80%**

### Small Dataset (Quick Testing)

```bash
# 5K examples, large cities only (500K+ population)
python generate_large_single_task_dataset.py --min-population 500000
```

- Training examples: ~5,000
- Training time: 30 minutes
- Expected accuracy: **50-60%**

---

## Training Commands

### Basic Training (Auto-adjust epochs)

```bash
python scripts/train_single_task.py \
    --train-file data/train_single_task_large.json \
    --dev-file data/dev_single_task_large.json \
    --auto-adjust
```

Auto-adjusts epochs based on dataset size:
- 250K examples → 3 epochs
- 80K examples → 4 epochs
- 10K examples → 6 epochs
- 1K examples → 10 epochs

**✨ Early stopping enabled by default:**
- Stops training when validation loss stops improving
- Patience: 2 evaluations (can customize with `--early-stopping-patience`)
- Automatically loads best checkpoint
- See `EARLY_STOPPING_GUIDE.md` for details

### Custom Settings

```bash
python scripts/train_single_task.py \
    --train-file data/train_single_task_large.json \
    --dev-file data/dev_single_task_large.json \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --learning-rate 1e-4 \
    --num-epochs 3 \
    --batch-size 4 \
    --lora-r 16 \
    --lora-alpha 32
```

### For 8B Model

```bash
python scripts/train_single_task.py \
    --train-file data/train_single_task_large.json \
    --dev-file data/dev_single_task_large.json \
    --model-name Qwen/Qwen3-8B-Instruct \
    --batch-size 2 \
    --lora-r 16 \
    --num-epochs 3
```

Expected accuracy: **85-90%** (better than 4B)

### Low Memory Settings

```bash
python scripts/train_single_task.py \
    --train-file data/train_single_task_large.json \
    --dev-file data/dev_single_task_large.json \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --lora-r 8 \
    --max-length 256
```

Works on 16GB GPU

---

## Monitoring Training

### Watch Progress

```bash
# In separate terminal
watch -n 10 tail -20 models/qwen3-4b-place-association-lora_*/log.txt
```

### Expected Progress

```
Epoch 1:
  Step 1000: loss 1.8 → Learning basic patterns
  Step 5000: loss 1.2 → Getting better
  Step 10000: loss 0.9 → Decent accuracy

Epoch 2:
  Step 20000: loss 0.6 → Good accuracy
  Dev eval: ~70%

Epoch 3:
  Step 30000: loss 0.4 → Near optimal
  Dev eval: ~80-85%
```

---

## Evaluation

### Single Task

```bash
cd evaluation
python evaluate_applied.py \
    --lora-path ../finetuning/models/qwen3-4b-place-association-lora \
    --task place_association \
    --max-samples 500
```

### Compare with Base Model

```bash
# Base model
python evaluate_applied.py \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --task place_association

# Fine-tuned
python evaluate_applied.py \
    --lora-path ../finetuning/models/qwen3-4b-place-association-lora \
    --task place_association
```

### Check Noise Robustness

```bash
# Evaluate at different noise levels
python analyze_noise_robustness.py \
    --lora-path ../finetuning/models/qwen3-4b-place-association-lora \
    --test-file ../finetuning/data/test_single_task_large.json
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch-size 2 --gradient-accumulation-steps 4

# OR reduce LoRA rank
--lora-r 8 --lora-alpha 16
```

### Training Too Slow

```bash
# Use medium dataset instead
python generate_large_single_task_dataset.py --min-population 50000

# OR reduce epochs
--num-epochs 2
```

### Not Converging

```bash
# Check if learning rate too high
--learning-rate 5e-5  # Lower

# OR increase epochs
--num-epochs 5
```

---

## Files Created

After running the commands, you'll have:

```
finetuning/
├── data/
│   ├── train_single_task_large.json  (250K examples, ~500MB)
│   ├── dev_single_task_large.json    (16K examples, ~30MB)
│   └── test_single_task_large.json   (8K examples, ~15MB)
│
└── models/
    └── qwen3-4b-place-association-lora/
        ├── adapter_config.json
        ├── adapter_model.bin
        └── ...
```

---

## Expected Timeline

| Step | Time | Action |
|------|------|--------|
| 0:00 | Start | Generate dataset |
| 0:02 | 2 min | Dataset ready |
| 0:02 | Start training | 250K examples, 3 epochs |
| 2:00 | 2 hours | Epoch 1 complete (~40% dev) |
| 4:00 | 4 hours | Epoch 2 complete (~70% dev) |
| 6:00 | 6 hours | Epoch 3 complete (~80% dev) |
| 6:05 | Done | Model saved, ready to eval |

---

## Cost Breakdown

| Resource | Cost |
|----------|------|
| Dataset generation | Free (local) |
| Training (A100, 6-8h) | $12-24 |
| Storage (~600MB) | <$1/month |
| **Total** | **~$13-25** |

Compare to:
- Multi-task training: $10-24 (but gets 3% accuracy!)
- GPT-4 API: $30 per 1000 queries (vs $0.10 for your model)

**ROI:** Train once for $15-25, use forever at 1/300th the cost

---

## Success Metrics

After training, expect:

| Metric | Target |
|--------|--------|
| place_association accuracy | 80-85% |
| Clean data accuracy | 90-95% |
| Noisy data (111m) accuracy | 85-90% |
| Heavy noise (1km) accuracy | 60-75% |
| Inference latency | 50-100ms |
| Cost per query | $0.0001 |

Compare to current:
- place_association: **3%** ❌
- All noise levels: **3-7%** ❌

---

## What's Next?

### After Successful Training

1. **Deploy to production**
   - Use LoRA adapter with base model
   - ~100ms latency per query
   - Can handle 10-20 QPS per GPU

2. **Extend to more tasks** (optional)
   - Add regional_knowledge (country detection)
   - Add format_conversion (coordinate formats)
   - Keep single-task benefits

3. **Continuous improvement**
   - Add user feedback loop
   - Retrain monthly with new data
   - Monitor accuracy over time

---

**Created:** November 11, 2025
**Status:** Ready to run
**Expected Result:** 80-85% accuracy (from 3%)
