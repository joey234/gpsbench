# Tinker Supported Models

Complete list of models available on Tinker's cloud platform for fine-tuning.

## Available Base Models

Tinker currently supports these base models (as of December 2025):

### Llama Models (Meta)

1. **meta-llama/Llama-3.2-1B** ← Smallest, fastest
   - 1B parameters
   - Good for: Quick experiments, limited resources
   - Chat template: `llama3`

2. **meta-llama/Llama-3.2-3B** ← **Recommended default**
   - 3B parameters (same size as Qwen2.5-3B)
   - Good for: General purpose, balanced performance/speed
   - Chat template: `llama3`

3. **meta-llama/Llama-3.1-8B**
   - 8B parameters
   - Good for: Better performance, more capacity
   - Chat template: `llama3`

4. **meta-llama/Llama-3.1-70B**
   - 70B parameters
   - Good for: Best performance, large scale
   - Chat template: `llama3`

### Qwen Models (Alibaba)

5. **Qwen/Qwen3-8B-Base**
   - 8B parameters
   - Good for: Multilingual tasks, strong reasoning
   - Chat template: `qwen_chat`

6. **Qwen/Qwen3-30B-A3B-Base**
   - 30B parameters (with 3B active)
   - Good for: MoE architecture, efficiency
   - Chat template: `qwen_chat`

### DeepSeek Models

7. **deepseek-ai/DeepSeek-V3.1-Base**
   - Large MoE model
   - Good for: Code and reasoning tasks
   - Chat template: `deepseek_chat`

## Model Selection Guide

### For GPSBench Training

**Recommended choices:**

1. **meta-llama/Llama-3.2-3B** (Default)
   - Same size as originally requested Qwen2.5-3B
   - Good balance of performance and speed
   - Well-supported, widely used

2. **meta-llama/Llama-3.1-8B**
   - Better performance than 3B
   - Still reasonable training time
   - Good for production use

3. **Qwen/Qwen3-8B-Base**
   - Strong multilingual capabilities
   - Good for geographic/location tasks
   - Similar performance to Llama-3.1-8B

### Size Comparison

| Model | Parameters | Relative Speed | Relative Performance |
|-------|-----------|----------------|---------------------|
| Llama-3.2-1B | 1B | ⚡⚡⚡⚡⚡ Fastest | ⭐⭐ Basic |
| Llama-3.2-3B | 3B | ⚡⚡⚡⚡ Fast | ⭐⭐⭐ Good |
| Llama-3.1-8B | 8B | ⚡⚡⚡ Medium | ⭐⭐⭐⭐ Great |
| Qwen3-8B | 8B | ⚡⚡⚡ Medium | ⭐⭐⭐⭐ Great |
| Llama-3.1-70B | 70B | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent |

## Usage Examples

### Llama-3.2-3B (Default)

```bash
python train.py \
    model_name="meta-llama/Llama-3.2-3B" \
    batch_size=64 \
    lora_rank=64 \
    num_epochs=3
```

### Llama-3.1-8B (Better Performance)

```bash
python train.py \
    model_name="meta-llama/Llama-3.1-8B" \
    batch_size=128 \
    lora_rank=128 \
    num_epochs=5
```

### Qwen3-8B (Multilingual)

```bash
python train.py \
    model_name="Qwen/Qwen3-8B-Base" \
    batch_size=128 \
    lora_rank=128 \
    num_epochs=3
```

### Small/Fast Model (Testing)

```bash
python train.py \
    model_name="meta-llama/Llama-3.2-1B" \
    batch_size=32 \
    lora_rank=32 \
    num_epochs=1
```

## Chat Templates

The chat template (renderer) is **auto-detected** based on model name:
- Llama models → `llama3`
- Qwen models → `qwen_chat`
- DeepSeek models → `deepseek_chat`

You can override with:
```bash
python train.py model_name="meta-llama/Llama-3.2-3B" renderer_name=llama3
```

## Training Tips

### Memory Requirements

Approximate GPU memory needed (with LoRA):
- 1B model: ~8GB
- 3B model: ~16GB
- 8B model: ~32GB
- 70B model: ~128GB+

Tinker handles this automatically on cloud GPUs.

### Batch Size Recommendations

| Model Size | Recommended Batch Size | LoRA Rank |
|-----------|----------------------|-----------|
| 1B | 32-64 | 32 |
| 3B | 64-128 | 64 |
| 8B | 128-256 | 128 |
| 70B | 256-512 | 256 |

### Learning Rate

Recommended learning rates:
- Small models (1-3B): `5e-4`
- Medium models (8B): `3e-4`
- Large models (70B): `1e-4`

## Model Updates

Tinker periodically adds new models. Check the latest:
```
https://tinker-docs.thinkingmachines.ai/model-lineup
```

## Common Questions

### Q: Can I use Qwen2.5 models?

No, Tinker only supports Qwen3 models (`Qwen/Qwen3-8B-Base`, etc.). Use Llama-3.2-3B as an alternative.

### Q: What happened to smaller Qwen3 models?

Tinker only supports `Qwen3-8B-Base` and `Qwen3-30B-A3B-Base` from the Qwen3 family. For smaller models, use Llama-3.2-1B or Llama-3.2-3B.

### Q: Which model is best for GPSBench?

For most cases: **meta-llama/Llama-3.2-3B**
- Same size as originally requested Qwen2.5-3B
- Good balance of performance and speed
- Well-tested and reliable

For better performance: **meta-llama/Llama-3.1-8B**

### Q: Can I use instruct/chat models?

The listed models are **base models** for fine-tuning. Instruct models are not available for custom fine-tuning on Tinker.

### Q: How do I check what models are currently available?

Check Tinker's documentation:
```
https://tinker-docs.thinkingmachines.ai/model-lineup
```

## Need Help?

- **Model selection:** See this guide
- **Training setup:** See `QUICKSTART.md`
- **Tinker docs:** https://tinker-docs.thinkingmachines.ai/
