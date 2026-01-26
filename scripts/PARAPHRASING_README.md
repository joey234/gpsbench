# Reasoning Steps Paraphrasing Guide

This guide explains how to use LLM-based paraphrasing to increase diversity in reasoning steps across all GPSBench tasks.

## Overview

The current reasoning steps use formulaic patterns like:
- "Step 1: Identify the coordinates..."
- "Step 2: Apply the Haversine formula..."
- "Answer: ..."

The paraphrasing script generates diverse variations while maintaining:
- ✅ **Semantic correctness** (same meaning)
- ✅ **Numerical accuracy** (same numbers and formulas)
- ✅ **Logical structure** (same step sequence)
- ✅ **Answer preservation** (same final result)

## Setup

### 1. Install Dependencies

```bash
pip install openai python-dotenv
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
# Option 1: OpenAI (recommended for quality)
OPENAI_API_KEY=sk-...

# Option 2: OpenRouter (more model options)
OPENROUTER_API_KEY=sk-or-...
```

### 3. Test Connection

```bash
python scripts/paraphrase_reasoning_steps.py --task distance --limit 1 --dry-run
```

## Usage

### Basic Usage

Paraphrase 100 examples from distance calculation task:
```bash
python scripts/paraphrase_reasoning_steps.py --task distance --limit 100
```

### Process All Tasks

Paraphrase all tasks (default: 100 examples per task):
```bash
python scripts/paraphrase_reasoning_steps.py --task all
```

### Process All Examples

Use `-1` to paraphrase ALL examples in a task:
```bash
python scripts/paraphrase_reasoning_steps.py --task all --limit -1
```

### Use Specific Model

```bash
# OpenAI models (gpt-4.1-mini is default)
python scripts/paraphrase_reasoning_steps.py --task all --model gpt-4.1-mini

# Alternative models
python scripts/paraphrase_reasoning_steps.py --task all --model gpt-4o-mini

# OpenRouter models
python scripts/paraphrase_reasoning_steps.py --task all --model anthropic/claude-3-haiku --provider openrouter
```

### Dry Run (Preview Changes)

```bash
python scripts/paraphrase_reasoning_steps.py --task distance --limit 10 --dry-run
```

## Paraphrasing Styles

The script cycles through 5 different styles to maximize diversity:

### 1. **Formal Mathematical**
```
Step 1: Let P₁ = (-1.2864°, 36.8172°) and P₂ = (-0.9373°, 37.12275°)
Step 2: Apply the geodesic distance formula: d = R·c where c = 2·arcsin(√(...))
```

### 2. **Tutorial/Educational**
```
Step 1: First, we need to identify our two points. Point A is Nairobi at (-1.29°, 36.82°)...
Step 2: Now, let's use the Haversine formula which accounts for Earth's curvature...
```

### 3. **Concise/Direct**
```
Step 1: Points: A(-1.29°, 36.82°), B(-0.94°, 37.12°)
Step 2: Haversine: d = 51.44 km
```

### 4. **Detailed/Verbose**
```
Step 1: We begin by extracting the coordinate pairs. The starting point A is located at latitude -1.2864° and longitude 36.8172°, which corresponds to Nairobi, Kenya. The destination point B is...
```

### 5. **Conversational**
```
Step 1: Okay, so we're starting with two locations here - Nairobi and Kenol.
Step 2: To find the distance, we'll use the Haversine formula because it handles Earth's spherical shape...
```

## Configuration Options

```bash
python scripts/paraphrase_reasoning_steps.py \
  --task TASK \          # Task name or 'all'
  --model MODEL \        # LLM model (default: gpt-4.1-mini)
  --provider PROVIDER \  # openai, openrouter, or auto
  --limit N \            # Examples per task (default: 100, -1 for all)
  --styles STYLES \      # Space-separated style names
  --temperature T \      # Sampling temperature (default: 0.7)
  --dry-run              # Preview without saving
```

### Available Tasks

- **Pure GPS Track:**
  - `distance` - Distance Calculation
  - `bearing` - Bearing Computation
  - `interpolation` - Interpolation
  - `area` - Area & Perimeter
  - `bbox` - Bounding Box
  - `route_geometry` - Route Geometry
  - `format` - Format Conversion
  - `precision` - Precision Understanding
  - `transforms` - Coordinate Transforms
  - `noise` - Noise Tolerance
  - `missing` - Missing Data

- **Applied Track:**
  - `proximity` - Proximity/Nearest Neighbor
  - `route_analysis` - Route Analysis
  - `relative` - Relative Position
  - `spatial` - Spatial Patterns
  - `place` - Place Association
  - `regional` - Regional Knowledge
  - `terrain` - Terrain Classification
  - `disambiguation` - Name Disambiguation

## Cost Estimation

### Per Example Costs (approximate)

| Model | Input Tokens | Output Tokens | Cost/Example |
|-------|-------------|---------------|--------------|
| gpt-4.1-mini (default) | ~300 | ~200 | $0.0001 |
| gpt-4o-mini | ~300 | ~200 | $0.0001 |
| gpt-3.5-turbo | ~300 | ~200 | $0.0002 |
| claude-3-haiku | ~300 | ~200 | $0.0001 |

### Total Costs

- **100 examples/task × 19 tasks = 1,900 examples**
  - gpt-4.1-mini: ~$0.20
  - gpt-4o-mini: ~$0.20
  - gpt-3.5-turbo: ~$0.40

- **All examples (16,968 total)**
  - gpt-4.1-mini: ~$1.70
  - gpt-4o-mini: ~$1.70
  - gpt-3.5-turbo: ~$3.40

## Validation

The script automatically validates paraphrased reasoning:

### ✅ Valid Paraphrasing
```
Original: Step 1: Identify the coordinates
          Point A: (-1.2864°, 36.8172°)

Valid:    Step 1: We start with two coordinate pairs:
          Point A is at latitude -1.2864° and longitude 36.8172°
```

### ❌ Invalid Paraphrasing (Rejected)
```
Original: Distance = 51.44 km

Invalid:  Distance ≈ 51 km  ← Number changed, REJECTED
```

## Monitoring Progress

The script provides real-time feedback:

```
Processing task5_distance_calculation.json...
  Examples with reasoning: 1000
  Paraphrasing: 100 examples
  [1/100] Style: Tutorial/Educational ✓
  [2/100] Style: Formal Mathematical ✓
  [3/100] Style: Concise/Direct ✗ (kept original)
  ...
  Saved 97 paraphrased examples

SUMMARY
==================================================
Total processed: 1900
Successfully paraphrased: 1847
Failed/kept original: 53
```

## Best Practices

### 1. Start Small
```bash
# Test with 10 examples first
python scripts/paraphrase_reasoning_steps.py --task distance --limit 10 --dry-run
```

### 2. Use Cheaper Models
```bash
# gpt-4.1-mini is the default - fast and cheap
python scripts/paraphrase_reasoning_steps.py --task all --model gpt-4.1-mini

# gpt-4o-mini is an alternative
python scripts/paraphrase_reasoning_steps.py --task all --model gpt-4o-mini
```

### 3. Monitor Quality
Check a few paraphrased examples manually:
```bash
# After running paraphrasing
python << 'EOF'
import json
with open('data/track_pure_gps/task5_distance_calculation.json') as f:
    data = json.load(f)
    for ex in data[:3]:
        if 'reasoning_style' in ex:
            print(f"Style: {ex['reasoning_style']}")
            for step in ex['reasoning_steps']:
                print(f"  {step}")
            print()
EOF
```

### 4. Process in Batches
```bash
# Process one track at a time
python scripts/paraphrase_reasoning_steps.py --task distance --limit 200
python scripts/paraphrase_reasoning_steps.py --task bearing --limit 200
# etc.
```

## Troubleshooting

### Issue: API Key Not Found
```
Error initializing LLM client: OPENAI_API_KEY not found in environment
```

**Solution:** Create `.env` file with your API key

### Issue: Validation Failures
```
[1/100] Style: Tutorial ✗ (kept original)
Validation failed: Step count mismatch (4 != 5)
```

**Solution:** This is normal. The script keeps the original on validation failure. If >20% fail, try:
- Lower temperature: `--temperature 0.3`
- Different model
- Adjust validation rules in code

### Issue: Rate Limiting
```
Error: Rate limit exceeded
```

**Solution:** Add delay between requests (modify code):
```python
time.sleep(0.5)  # Add in paraphrase_task_file loop
```

## Examples

### Example 1: Quick Test
```bash
python scripts/paraphrase_reasoning_steps.py \
  --task distance \
  --limit 5 \
  --dry-run
```

### Example 2: Production Run
```bash
python scripts/paraphrase_reasoning_steps.py \
  --task all \
  --model gpt-4.1-mini \
  --limit 100 \
  --temperature 0.7
```

### Example 3: Specific Styles
```bash
python scripts/paraphrase_reasoning_steps.py \
  --task distance \
  --limit 200 \
  --styles tutorial conversational
```

## Output

### Before Paraphrasing
```json
{
  "task": "distance_calculation",
  "reasoning_steps": [
    "Step 1: Identify the coordinates",
    "  Point A: (-1.2864°, 36.8172°) - Nairobi, Kenya",
    "..."
  ]
}
```

### After Paraphrasing
```json
{
  "task": "distance_calculation",
  "reasoning_steps": [
    "Step 1: Let's start by noting our two locations and their coordinates",
    "  The first point is Nairobi, Kenya at latitude -1.2864° and longitude 36.8172°",
    "..."
  ],
  "reasoning_style": "tutorial"
}
```

## Quality Assurance

After paraphrasing, run validation:

```bash
# Check coverage
python << 'EOF'
import json
from pathlib import Path

for task_file in Path('data').rglob('task*.json'):
    with open(task_file) as f:
        data = json.load(f)
        with_style = sum(1 for ex in data if 'reasoning_style' in ex)
        if with_style > 0:
            print(f"{task_file.name}: {with_style}/{len(data)} paraphrased")
EOF
```

## Notes

- Paraphrasing preserves numerical accuracy through validation
- Failed paraphrasing keeps the original reasoning
- Multiple styles ensure diverse training data
- Temperature 0.7 balances creativity and accuracy
- All changes are saved to the original task files
