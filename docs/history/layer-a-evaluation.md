# Historical documentation — superseded

Use [the current evaluator](../../evaluation/README.md).

# GPSBench Layer A - LLM Evaluation

Evaluation framework for testing Large Language Models on GPS coordinate perception tasks.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Run Evaluation

**Evaluate with OpenAI (GPT-3.5-Turbo):**

```bash
python evaluation/evaluate_layer_a.py \
  --provider openai \
  --model gpt-3.5-turbo \
  --max-samples 10
```

**Evaluate with OpenRouter (Claude 3 Sonnet):**

```bash
python evaluation/evaluate_layer_a.py \
  --provider openrouter \
  --model anthropic/claude-3-sonnet \
  --max-samples 10
```

**Evaluate all tasks with full dataset:**

```bash
python evaluation/evaluate_layer_a.py \
  --provider openai \
  --model gpt-4 \
  --task all
```

### 4. Analyze Results

**View latest results:**

```bash
python evaluation/analyze_results.py
```

**Compare all results:**

```bash
python evaluation/analyze_results.py --compare
```

**Export to CSV:**

```bash
python evaluation/analyze_results.py --compare --export-csv comparison.csv
```

## Command Line Options

### Evaluation Script

```
python evaluation/evaluate_layer_a.py [OPTIONS]

Model Configuration:
  --provider {openai,openrouter}
                        API provider (default: openai)
  --model MODEL         Model name (default: gpt-3.5-turbo)
  --temperature TEMP    Temperature for generation (default: 0.0)

Evaluation Configuration:
  --task {format_recognition,format_conversion,validation,
          coordinate_system_recognition,precision_understanding,all}
                        Task to evaluate (default: all)
  --max-samples N       Maximum samples per task (default: all)
  --delay SECONDS       Delay between API calls (default: 0.5)

Output Configuration:
  --data-dir DIR        Data directory (default: data/layer_a)
  --results-dir DIR     Results directory (default: results)
  --output FILENAME     Output filename (default: auto-generated)
```

### Analysis Script

```
python evaluation/analyze_results.py [OPTIONS]

Options:
  --results-dir DIR     Results directory (default: results)
  --file FILENAME       Specific result file to analyze
  --compare             Compare all results in directory
  --export-csv FILE     Export comparison to CSV file
```

## Supported Models

### OpenAI Models

```bash
# GPT-4 models
--model gpt-4
--model gpt-4-turbo
--model gpt-4-turbo-preview

# GPT-3.5 models
--model gpt-3.5-turbo
--model gpt-3.5-turbo-16k
```

### OpenRouter Models

OpenRouter provides access to many models from different providers:

```bash
# OpenAI via OpenRouter
--model openai/gpt-4
--model openai/gpt-3.5-turbo

# Anthropic Claude
--model anthropic/claude-3-opus
--model anthropic/claude-3-sonnet
--model anthropic/claude-3-haiku

# Google Gemini
--model google/gemini-pro
--model google/gemini-pro-1.5

# Meta Llama
--model meta-llama/llama-3-70b-instruct
--model meta-llama/llama-3-8b-instruct

# Mistral
--model mistralai/mistral-large
--model mistralai/mistral-medium
```

See [OpenRouter Models](https://openrouter.ai/models) for the full list.

## Example Workflows

### 1. Quick Test (10 samples per task)

```bash
# Test GPT-3.5-Turbo on a small subset
python evaluation/evaluate_layer_a.py \
  --provider openai \
  --model gpt-3.5-turbo \
  --max-samples 10 \
  --delay 0.5
```

### 2. Full Evaluation (All 2,329 test cases)

```bash
# Evaluate GPT-4 on full dataset
python evaluation/evaluate_layer_a.py \
  --provider openai \
  --model gpt-4 \
  --task all \
  --delay 1.0
```

**Note**: This will make ~2,329 API calls. Estimated time: ~45 minutes with 1s delay.

### 3. Single Task Evaluation

```bash
# Test only format recognition
python evaluation/evaluate_layer_a.py \
  --provider openai \
  --model gpt-4 \
  --task format_recognition \
  --max-samples 50
```

### 4. Compare Multiple Models

```bash
# Evaluate GPT-3.5
python evaluation/evaluate_layer_a.py \
  --provider openai \
  --model gpt-3.5-turbo \
  --max-samples 100

# Evaluate GPT-4
python evaluation/evaluate_layer_a.py \
  --provider openai \
  --model gpt-4 \
  --max-samples 100

# Evaluate Claude 3 Sonnet
python evaluation/evaluate_layer_a.py \
  --provider openrouter \
  --model anthropic/claude-3-sonnet \
  --max-samples 100

# Compare results
python evaluation/analyze_results.py --compare
```

### 5. Export Results for Publication

```bash
# Generate comparison table
python evaluation/analyze_results.py \
  --compare \
  --export-csv results_for_paper.csv
```

## Output Format

### Results JSON Structure

```json
{
  "model": "gpt-4",
  "provider": "openai",
  "timestamp": "2025-10-20T15:30:00",
  "tasks": {
    "format_recognition": {
      "task": "format_recognition",
      "metrics": {
        "accuracy": 95.5,
        "correct_count": 764,
        "total_count": 800,
        "error_count": 0,
        "total_tokens": 156000,
        "avg_latency_ms": 450
      },
      "test_results": [
        {
          "test_index": 0,
          "input": "40°51'30\"N, 73°59'45\"W",
          "location": "New York",
          "response": "DMS",
          "correct": true,
          "evaluation": {...},
          "tokens": 195,
          "latency_ms": 423
        },
        ...
      ]
    },
    ...
  },
  "overall_metrics": {
    "overall_accuracy": 87.3,
    "total_correct": 2033,
    "total_cases": 2329,
    "task_accuracies": {...}
  }
}
```

## Task Descriptions

### Task 1: Format Recognition (800 tests)
Identify the format of coordinate strings (DMS, decimal degrees, ISO 6709, Plus Codes, MGRS, UTM).

**Example:**
- Input: `40°51'30"N, 73°59'45"W`
- Expected: `DMS`

### Task 2: Format Conversion (500 tests)
Convert coordinates between different formats.

**Example:**
- Input: `40°51'30"N, 73°59'45"W` (DMS)
- Target: `decimal_degrees`
- Expected: `40.858333, -73.995833`

### Task 3: Validation & Error Detection (89 tests)
Detect invalid coordinates and common errors.

**Example:**
- Input: `91.5, 45.0`
- Expected: `INVALID - Latitude exceeds maximum value of 90°`

### Task 4: Coordinate System Recognition (240 tests)
Identify which coordinate reference system is being used.

**Example:**
- Input: `Zone 18N 585628E 4515585N`
- Expected: `UTM`

### Task 5: Precision Understanding (700 tests)
Understand coordinate precision and its implications for accuracy.

**Example:**
- Input: `40.858, -73.996` (3 decimals)
- Expected: `~111m accuracy, street-level`

## Cost Estimation

Approximate costs for full evaluation (2,329 test cases):

| Model | Input Tokens | Output Tokens | Est. Cost |
|-------|--------------|---------------|-----------|
| GPT-3.5-Turbo | ~350K | ~50K | ~$0.50 |
| GPT-4 | ~350K | ~50K | ~$15.00 |
| GPT-4-Turbo | ~350K | ~50K | ~$5.00 |
| Claude 3 Sonnet | ~350K | ~50K | ~$4.00 |
| Claude 3 Haiku | ~350K | ~50K | ~$0.50 |

**Note**: These are estimates. Actual costs depend on exact token usage and current API pricing.

## Rate Limiting

To avoid rate limits:

1. **Use delay parameter**: `--delay 1.0` (1 second between requests)
2. **Start with small samples**: `--max-samples 10`
3. **Monitor your API usage**: Check your provider dashboard

**Recommended delays:**
- OpenAI (free tier): 1-2 seconds
- OpenAI (paid tier): 0.5 seconds
- OpenRouter: 0.5-1 second

## Troubleshooting

### "OPENAI_API_KEY not found"

Make sure you:
1. Created a `.env` file in the project root
2. Added your API key: `OPENAI_API_KEY=sk-...`
3. The `.env` file is in the same directory as the scripts

### "Rate limit exceeded"

Increase the delay:
```bash
--delay 2.0
```

Or reduce samples:
```bash
--max-samples 10
```

### "Model not found"

For OpenRouter models, use the full model path:
```bash
--model anthropic/claude-3-sonnet
```

Not:
```bash
--model claude-3-sonnet  # Wrong!
```

## Benchmark Analysis

### Geographic Distribution Analysis

Analyze the geographic coverage and distribution of test locations:

```bash
# View distribution report
python evaluation/analyze_geographic_distribution.py

# Save statistics to JSON
python evaluation/analyze_geographic_distribution.py --output stats.json

# Verbose output
python evaluation/analyze_geographic_distribution.py --verbose
```

**Output includes:**
- Continent distribution
- Country coverage (211 countries)
- Hemisphere balance
- Latitude zone coverage (Polar, Temperate, Tropical)
- Potential geographic biases
- Coordinate range statistics

**Example output:**
```
📊 OVERALL STATISTICS
Total coordinate points:    14,773
Unique locations:           4,610
Unique countries:           211
Unique continents:          7

🌍 CONTINENT DISTRIBUTION
Asia                 1,950 ( 46.5%) ███████████████████████
Europe                 696 ( 16.6%) ████████
Africa                 553 ( 13.2%) ██████
North America          518 ( 12.4%) ██████
South America          421 ( 10.0%) █████
```

## File Structure

```
evaluation/
├── README.md                           # This file
├── track_config.py                     # Two-track system configuration
├── llm_client.py                       # LLM API client
├── task_evaluators.py                  # Task-specific evaluators
├── evaluate_pure_gps.py                # Pure GPS track evaluation
├── evaluate_applied.py                 # Applied track evaluation
├── compare_tracks.py                   # Track comparison
├── analyze_results.py                  # Results analysis
└── analyze_geographic_distribution.py  # Geographic coverage analysis
```

## Contributing

To add support for a new LLM provider:

1. Add provider configuration to `llm_client.py`
2. Update `get_available_models()` with supported models
3. Test with a small sample first

To add a new evaluation metric:

1. Modify the appropriate evaluator in `task_evaluators.py`
2. Update the `evaluate_response()` method
3. Add metric to results JSON

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@misc{gpsbench_eval_2025,
  title={GPSBench: Evaluation Framework for GPS Coordinate Perception in LLMs},
  year={2025},
  author={[Your Name]},
  note={Layer A Evaluation Framework}
}
```

## License

[Specify License]

---

**Version**: 1.0
**Last Updated**: October 2025
