# GPSBench: A Benchmark for GPS Reasoning in Large Language Models

GPSBench is a comprehensive benchmark for evaluating GPS and geographic reasoning capabilities in Large Language Models. It comprises **57,800 samples across 17 tasks** organized into two tracks.

## Benchmark Overview

### Applied Track (8 tasks)
Tests real-world geographic reasoning requiring world knowledge:
- Place Association
- Name Disambiguation
- Relative Position
- Proximity
- Route Analysis
- Boundary Analysis
- Spatial Patterns
- Terrain Classification

### Pure GPS Track (9 tasks)
Tests coordinate manipulation without geographic knowledge:
- Format Conversion
- Coordinate Transformation
- Distance Calculation
- Bearing Computation
- Coordinate Interpolation
- Area & Perimeter
- Bounding Box
- Route Geometry
- Relative Position (coordinate-based)

## Installation

```bash
pip install -r requirements.txt
```

Configure API keys:
```bash
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, etc.)
```

## Quick Start

Run the benchmark on a model:

```bash
# Evaluate all tasks
python run_benchmark.py --model gpt-4o --provider openai

# Evaluate specific track
python run_benchmark.py --model gpt-4o --track applied
python run_benchmark.py --model gpt-4o --track pure_gps

# Evaluate specific task
python run_benchmark.py --model gpt-4o --task distance_calculation

# Limit samples for quick testing
python run_benchmark.py --model gpt-4o --max-samples 100
```

### Supported Providers

- `openai`: OpenAI models (gpt-4o, gpt-4-turbo, etc.)
- `anthropic`: Anthropic models (claude-3-opus, claude-3-sonnet, etc.)
- `google`: Google models (gemini-pro, gemini-1.5-pro, etc.)
- `openrouter`: Access multiple providers via OpenRouter

## Repository Structure

```
GPSBench/
├── run_benchmark.py          # Main entry point
├── requirements.txt          # Dependencies
├── .env.example              # API key template
│
├── data/                     # Benchmark data
│   ├── applied/              # Applied track tasks
│   └── pure_gps/             # Pure GPS track tasks
│
├── evaluation/               # Evaluation framework
│   ├── llm_client.py         # LLM API clients
│   ├── task_evaluators.py    # Task-specific evaluators
│   └── ...
│
├── scripts/                  # Utility scripts
│   ├── analyze_results.py    # Results analysis
│   ├── visualize_results.py  # Generate plots
│   ├── generate_*.py         # Data generation
│   └── ...
│
├── results/                  # Evaluation results
│
└── paper/                    # Paper LaTeX source
```

## Results

Results are saved to `results/gpsbench_{model}_{timestamp}/`:
- `summary.json`: Overall accuracy by track and task
- `task_results/`: Per-sample predictions and scores

## Data Format

Each task file contains samples with:
```json
{
  "id": "sample_001",
  "question": "What is the distance between...",
  "ground_truth": {"distance_km": 1234.5, "tolerance_km": 61.7},
  "metadata": {...}
}
```

## Citation

```bibtex
@article{gpsbench2025,
  title={GPSBench: A Benchmark for GPS Reasoning in Large Language Models},
  author={...},
  year={2025}
}
```

## License

MIT License
