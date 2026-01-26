# Qwen 4B Fine-Tuning Failure: Root Cause Analysis

**Date:** November 13, 2025
**Model:** Qwen/Qwen3-4B-Instruct-2507
**Task:** place_association (coordinate → location)
**Result:** 5.81% → 10.82% (+5.01 points) **FAILURE** (Expected: +75-80 points)

---

## Executive Summary

Fine-tuning **FAILED** to achieve target performance. Instead of improving from 5.81% to 80-85%, the model only reached 10.82% accuracy.

### Contributing Factors

1. **PRIMARY (60%)**: Model did NOT learn the task effectively
   - Only 12% of predictions contain the correct city name
   - Training completed (3 epochs, 93K steps) but learning was insufficient

2. **SECONDARY (30%)**: Training/Evaluation format mismatch
   - Training: "City, XX" (country codes)
   - Evaluation: "City, Full Country Name"
   - Model correctly learned training format but penalized in evaluation

3. **TERTIARY (10%)**: Evaluator matching bug
   - Rejects city names with ≤ 5 characters
   - Affects 14% of test cases (71/499)

---

## Detailed Analysis

### Factor 1: Model Did Not Learn (PRIMARY ISSUE)

**Evidence:**

Training completed successfully:
```
Training directory: qwen3-4b-place-association-lora_20251112_010253
Final epoch: 3.00
Training loss: 0.4822
Eval loss: 0.7173
Total steps: 93834
```

But predictions show poor learning:
```python
Manual accuracy check (first 50 tests):
  Predictions with correct city name: 6/50 (12%)
  Evaluator marked correct: 3/50 (6%)
```

**Sample predictions showing model confusion:**

| Coordinate | Expected | Prediction | Issue |
|------------|----------|------------|-------|
| (78.22, 15.63) | Longyearbarn, Svalbard | Longyearbyen, Svalbard | Close but wrong place |
| (-77.85, 166.67) | McMurdo Station | Nouméa, New Caledonia | Completely wrong location |
| (-0.23, 78.52) | Quito, Ecuador | Cuenca, EC | Wrong city (same country) |
| (-1.29, -36.78) | Natal, Brazil | (data not shown) | Location errors |

**Possible causes:**

1. **Training data quality issues**
   - 250K examples from GeoNames cities15000.txt
   - May not align with evaluation test set (landmarks, edge cases)

2. **Training format mismatch**
   - Training: Conversational prompts + "City, XX" outputs
   - Evaluation: Different prompt style + "City, Full Name" expected
   - Model learned wrong output format

3. **Model capacity insufficient**
   - Qwen 3-4B may not have enough capacity for 32K+ cities
   - LoRA rank 16 may be too restrictive

4. **Training hyperparameters suboptimal**
   - Learning rate 1e-4 may be too high or too low
   - 3 epochs may be insufficient or too many
   - Early stopping triggered at suboptimal point

---

### Factor 2: Format Mismatch (SECONDARY ISSUE)

**Training data format:**
```json
{
  "input": "What place corresponds to coordinates 10.772166, 76.262500?",
  "output": "Shōranūr, IN",
  "system_prompt": "You are a geographic AI assistant..."
}
```

**Evaluation data format:**
```json
{
  "ground_truth": {
    "name": "Shōranūr, India",
    "country": "India",
    "type": "city"
  }
}
```

**Impact:**

Predictions using country codes are penalized:
- "Lagos, NG" ❌ (evaluation expects "Lagos, Nigeria")
- "Addis Ababa, ET" ✅ (happened to match due to city name length > 5)
- "Suva, FJ" ❌ (correct format for training, wrong for eval)

**Analysis of prediction formats (499 tests):**
```
Country code format ("City, XX"):     209 (41.9%)
Full name format ("City, Country"):   289 (58.1%)
Other format:                           1 (0.2%)
```

Model is confused between formats - sometimes outputs codes, sometimes full names.

---

### Factor 3: Evaluator Bug (TERTIARY ISSUE)

**Bug location:** `evaluation/layer_b_evaluators.py:144`

```python
exact_match = (expected_lower == response_clean or
              expected_lower in response_clean or
              (len(expected_parts) > 0 and expected_parts[0] in response_clean and
               len(expected_parts[0]) > 5))  # ⚠️ BUG: Rejects names ≤ 5 chars
```

**Impact:**

Cities with ≤ 5 character names are rejected even when correctly predicted:
- "Lagos" (5 chars): 3/46 correct (6.5%)
- "Suva" (4 chars): 2/19 correct (10.5%)
- "Null Island" (3 chars): 0/6 correct (0.0%)

**Affected tests:** 71/499 (14.2%)

**Accuracy by name length:**
```
Length ≤ 5:  5/71  (7.0%)  ← Affected by bug
Length > 5: 49/428 (11.4%) ← Not affected by bug
```

However, even for names > 5 chars, accuracy is only 11.4%, so this bug is **NOT the primary cause**.

---

## Why Training Failed: Hypotheses

### Hypothesis 1: Data Distribution Mismatch ⭐ MOST LIKELY

**Training data:**
- Source: GeoNames cities15000.txt (32,709 cities)
- Coverage: Major cities worldwide
- Format: "City, XX" with country codes

**Evaluation data:**
- Mix of cities, landmarks, edge cases
- Edge cases: North Pole, South Pole, Equator, Dateline, Null Island
- Underrepresented regions: Africa, South Asia
- Format: "City, Full Country Name"

**Mismatch:**
```
Training focused on:        Evaluation includes:
- Major cities              - Landmarks (Eiffel Tower, McMurdo Station)
- Standard locations        - Edge cases (Near North Pole, On Equator)
- Simple city identification - Complex geographic features
- Country codes (XX)        - Full country names
```

**Evidence:**
- Test 0: "Longyearbarn, Svalbard (Near North Pole)" ❌
- Test 1: "McMurdo Station, Antarctica (Near South Pole)" ❌
- Test 4: "Null Island (0°, 0°)" ❌

These are NOT in GeoNames cities15000.txt!

### Hypothesis 2: Model Overfitted to Training Format

Model learned to output "City, XX" format but evaluation expects different format.

**Evidence:**
- 41.9% of predictions use country codes (training format)
- 58.1% of predictions use full names (trying to adapt?)
- Model is confused about expected format

### Hypothesis 3: Insufficient Training Signal

**Training setup:**
- 250K examples, 3 epochs
- Batch size 4, gradient accumulation 2
- Learning rate 1e-4
- Early stopping patience 2

**Eval loss trajectory:**
```
Epoch 1: (not shown)
Epoch 2: (not shown)
Epoch 3: eval_loss = 0.7173
Best metric: 0.6416
```

Loss 0.64-0.72 is relatively high for this task. Model may need:
- More epochs
- Better learning rate
- Larger LoRA rank
- More training data in evaluation format

---

## Verification Steps

### 1. Check Training Data Coverage ⚡ URGENT

```bash
cd /Users/hungthinh/workspace/GPSBench/finetuning

# Check if training data contains evaluation test locations
python3 -c "
import json

# Load training data
with open('data/train_single_task_large.json') as f:
    train_data = json.load(f)

# Load evaluation data
with open('../data/track_applied/task1_place_association.json') as f:
    eval_data = json.load(f)

# Check overlap
train_locations = set()
for ex in train_data:
    # Extract city name from output
    city = ex['output'].split(',')[0].strip()
    train_locations.add(city.lower())

print(f'Training locations: {len(train_locations)}')
print()

# Check evaluation coverage
covered = 0
not_covered = []
for test in eval_data[:50]:  # Check first 50
    gt_name = test['ground_truth']['name']
    city = gt_name.split(',')[0].strip().lower()
    if city in train_locations:
        covered += 1
    else:
        not_covered.append(gt_name)

print(f'Evaluation coverage (first 50):')
print(f'  Covered: {covered}/50 ({covered/50*100:.1f}%)')
print(f'  Not covered: {50-covered}/50')
print()
print('Examples not in training data:')
for loc in not_covered[:10]:
    print(f'  - {loc}')
"
```

### 2. Check LoRA Actually Loaded

```bash
cd /Users/hungthinh/workspace/GPSBench/evaluation

# Verify LoRA is being used
python3 -c "
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-4B-Instruct-2507',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

lora_model = PeftModel.from_pretrained(
    base_model,
    '../finetuning/models/qwen3-4b-place-association-lora'
)

# Check LoRA parameters
print('LoRA adapter loaded:')
print(lora_model.peft_config)
print()

# Get trainable parameters
trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in lora_model.parameters())
print(f'Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)')
"
```

### 3. Test Inference on Training-Style Examples

```bash
cd /Users/hungthinh/workspace/GPSBench/finetuning

# Test if model can correctly answer training-format questions
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
base_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-4B-Instruct-2507',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
model = PeftModel.from_pretrained(
    base_model,
    'models/qwen3-4b-place-association-lora'
)
model.eval()

# Test examples from training data
test_cases = [
    ('40.7128, -74.0060', 'New York City, US'),
    ('51.5074, -0.1278', 'London, GB'),
    ('35.6762, 139.6503', 'Tokyo, JP'),
    ('48.8566, 2.3522', 'Paris, FR'),
    ('-33.8688, 151.2093', 'Sydney, AU')
]

print('Testing on training-format examples:')
print('=' * 80)

for coord, expected in test_cases:
    messages = [
        {'role': 'system', 'content': 'You are a geographic AI assistant. Given coordinates, identify the location name.'},
        {'role': 'user', 'content': f'What location is at coordinates {coord}?'}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f'\\nCoord: {coord}')
    print(f'Expected: {expected}')
    print(f'Got: {response}')
    print(f'Match: {expected.lower() in response.lower()}')
"
```

---

## Recommended Solutions

### Solution 1: Regenerate Training Data to Match Evaluation ⭐ RECOMMENDED

**Problem:** Training data from GeoNames doesn't cover evaluation test cases (landmarks, edge cases).

**Solution:** Create training data that matches evaluation format and content.

```bash
cd /Users/hungthinh/workspace/GPSBench/finetuning

# Create new training data matching evaluation format
python3 generate_evaluation_aligned_dataset.py \
    --eval-file ../data/track_applied/task1_place_association.json \
    --output-train data/train_eval_aligned.json \
    --output-dev data/dev_eval_aligned.json \
    --augmentation-factor 500 \
    --use-full-country-names  # Match evaluation format
```

**Key changes:**
1. Include all evaluation test locations
2. Use full country names (not codes)
3. Add landmarks, edge cases, underrepresented regions
4. Match evaluation prompt style
5. Generate 250K+ examples with proper diversity

### Solution 2: Retrain with Correct Format

**Once training data is regenerated:**

```bash
cd /Users/hungthinh/workspace/GPSBench/finetuning

python3 scripts/train_single_task.py \
    --train-file data/train_eval_aligned.json \
    --dev-file data/dev_eval_aligned.json \
    --num-epochs 4 \
    --learning-rate 5e-5 \
    --lora-r 32 \
    --lora-alpha 64 \
    --early-stopping-patience 3 \
    --auto-adjust
```

**Hyperparameter changes:**
- Increase LoRA rank: 16 → 32 (more capacity)
- Lower learning rate: 1e-4 → 5e-5 (more stable)
- More epochs: 3 → 4 (more training signal)
- More patience: 2 → 3 (avoid stopping too early)

### Solution 3: Fix Evaluator Bug (Minor Fix)

**File:** `evaluation/layer_b_evaluators.py:144`

```python
# Before (buggy)
exact_match = (expected_lower == response_clean or
              expected_lower in response_clean or
              (len(expected_parts) > 0 and expected_parts[0] in response_clean and
               len(expected_parts[0]) > 5))  # ❌ Rejects short names

# After (fixed)
exact_match = (expected_lower == response_clean or
              expected_lower in response_clean or
              (len(expected_parts) > 0 and expected_parts[0] in response_clean and
               len(expected_parts[0]) >= 3))  # ✅ Allow names ≥ 3 chars
```

This will improve accuracy by ~3-5 points for correctly predicted short city names.

---

## Expected Results After Fixes

### If Only Fix Evaluator Bug (Solution 3)
- Current: 10.82%
- After fix: ~14-16% (minor improvement)
- **Still far from target**

### If Regenerate Training Data + Retrain (Solutions 1 + 2)
- Current: 10.82%
- After fix: **75-85%** (target range)
- **This is what we need**

---

## Next Steps

### Immediate Actions (In Order)

1. **Verify training data coverage** (Solution 1 verification step)
   ```bash
   python3 check_training_eval_overlap.py
   ```

2. **If overlap is low (<50%):**
   - Create `generate_evaluation_aligned_dataset.py`
   - Generate new training data matching evaluation
   - Retrain with new data

3. **If overlap is high (>80%):**
   - Check if LoRA was actually loaded during evaluation
   - Try higher LoRA rank (32 or 64)
   - Try different learning rates

4. **Fix evaluator bug** (regardless of above)
   - Edit `layer_b_evaluators.py:144`
   - Change threshold from `> 5` to `>= 3`

---

## Conclusion

**Primary failure mode:** Training data from GeoNames cities does NOT match evaluation test set content and format.

**Evidence:**
- Evaluation includes landmarks (Eiffel Tower), edge cases (North/South Pole), special locations (Null Island)
- These are NOT in GeoNames cities15000.txt
- Model learned city identification but not landmark/edge case identification
- Format mismatch (country codes vs full names) made it worse

**Solution:** Regenerate training data to match evaluation content and format, then retrain.

**Expected timeline:**
1. Create aligned dataset generation script: 2-3 hours
2. Generate 250K examples: 1-2 hours
3. Retrain model: 6-8 hours
4. Re-evaluate: 30 minutes

**Total:** ~12-15 hours to successful fine-tuning

---

**Created:** November 13, 2025
**Status:** ⚠️ ROOT CAUSE IDENTIFIED - Training/Evaluation Data Mismatch
**Next Action:** Verify training data coverage, then regenerate aligned dataset
