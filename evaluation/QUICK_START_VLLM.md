# Quick Start: vLLM-Powered Evaluation ⚡

## TL;DR

```bash
# 1. Install vLLM (one-time)
pip install vllm

# 2. Run evaluation (10-20 minutes instead of hours!)
cd evaluation
python compare_base_vs_finetuned.py \
    --base-model Qwen/Qwen2.5-0.5B \
    --lora-path ../finetuning/models/qwen2.5-0.5b-lora \
    --track both
```

## Speed Comparison

| Method | Time |
|--------|------|
| 🐌 Without vLLM | 2-4 hours |
| 🚀 With vLLM | **10-20 minutes** |

## Commands

### Fast Evaluation (Recommended)
```bash
python compare_base_vs_finetuned.py \
    --base-model Qwen/Qwen2.5-0.5B \
    --lora-path ../finetuning/models/qwen2.5-0.5b-lora \
    --track both
```

### Quick Test (30 seconds)
```bash
python compare_base_vs_finetuned.py \
    --base-model Qwen/Qwen2.5-0.5B \
    --lora-path ../finetuning/models/qwen2.5-0.5b-lora \
    --track both \
    --max-samples 10
```

### Multi-GPU (Large Models)
```bash
python compare_base_vs_finetuned.py \
    --base-model Qwen/Qwen2.5-7B \
    --lora-path ../finetuning/models/qwen2.5-7b-lora \
    --track both \
    --tensor-parallel-size 2
```

### Without vLLM (Slower)
```bash
python compare_base_vs_finetuned.py \
    --base-model Qwen/Qwen2.5-0.5B \
    --lora-path ../finetuning/models/qwen2.5-0.5b-lora \
    --track both \
    --no-vllm
```

## What You Get

```
results/
├── comparison_Qwen_Qwen2.5_0.5B_20241110_123456.json
└── comparison_summary_20241110_123456.txt
```

**JSON**: Detailed metrics for analysis
**TXT**: Human-readable summary report

## Troubleshooting

### vLLM Not Installed?
```bash
pip install vllm
```

### CUDA Out of Memory?
```bash
# Option 1: Use smaller model
--base-model Qwen/Qwen2.5-0.5B

# Option 2: Split across GPUs
--tensor-parallel-size 2

# Option 3: Disable vLLM
--no-vllm
```

## More Info

- Full guide: `../VLLM_INTEGRATION.md`
- Comparison guide: `../EVALUATION_COMPARISON_GUIDE.md`
- Summary: `../VLLM_INTEGRATION_SUMMARY.md`

---

**Ready? Just run:**
```bash
pip install vllm
python compare_base_vs_finetuned.py \
    --base-model Qwen/Qwen2.5-0.5B \
    --lora-path ../finetuning/models/qwen2.5-0.5b-lora \
    --track both
```

🎉 Done in ~10-20 minutes!
