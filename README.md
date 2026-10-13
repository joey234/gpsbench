# GPSBench: Do Large Language Models Understand GPS Coordinates?

GPSBench evaluates reasoning with GPS coordinates and real-world geography. It contains **58,614 examples across 17 tasks**: nine Pure GPS tasks and eight Applied tasks. The released splits contain **35,522 training, 5,750 development, and 17,342 test examples**. Missing Data is an additional 3,400-example probe, excluded from benchmark averages.

- [Dataset and splits](data/README.md)
- [Evaluation guide](evaluation/README.md) and [current scoring rules](evaluation/SCORING_SPEC.md)
- [Verified model scores](results/current/scores.json) and [project website](leaderboard/)
- [Repository guide](docs/REPOSITORY.md)

## Setup

Use Python 3.10 or newer and a project-local environment. The repository uses requirements files.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env
```

Set the key for your provider in `.env`. The runner supports `openai`, `openrouter`, `gemini`, and automatic provider detection. Gemini additionally requires `google-genai`:

```bash
uv pip install --python .venv/bin/python -r requirements-gemini.txt
```

## Run the benchmark

Run from the repository root. These commands send requests to the selected model provider.

```bash
# Small run across the 17 main tasks
uv run --no-project --python .venv/bin/python python run_benchmark.py \
  --provider openai --model gpt-4.1-mini --max-samples 10

# A complete track
uv run --no-project --python .venv/bin/python python run_benchmark.py \
  --provider openai --model gpt-4.1-mini --track pure_gps

# One task
uv run --no-project --python .venv/bin/python python run_benchmark.py \
  --provider openai --model gpt-4.1-mini --tasks distance_calculation

# Optional probe; excluded from main averages
uv run --no-project --python .venv/bin/python python run_benchmark.py \
  --provider openai --model gpt-4.1-mini --include-diagnostics --tasks missing_data
```

The runner uses the released test splits. Do not regenerate splits to reproduce the reported results. Saved summaries identify the scoring version and whether the run covers all 17 complete test sets. Partial runs report averages only over the main tasks actually evaluated.

## Scoring

The current version is `gpsbench-neurips-metrics-restored-v1`.

- Distance and Area use `max(0, 100 × (1 − mean relative error))`.
- Bearing uses the same formula with wrapped angular error divided by 180°.
- These numerical scores include finite, parsed errors below 1000; parsing and inclusion counts are also reported.
- Interpolation requires both coordinates within ±0.5°. The remaining tasks use the corrected matching rules described in the [scoring specification](evaluation/SCORING_SPEC.md).
- Track and overall **scores** equally weight tasks. Saved `accuracy` and `overall_accuracy` fields are binary diagnostics, not the headline score.

Rescore an existing run without API calls or changing its saved responses:

```bash
uv run --no-project --python .venv/bin/python python -m evaluation.current \
  results/YOUR_RUN --out /tmp/gpsbench-current-scores.json
```

## Verify the release

```bash
uv run --no-project --python .venv/bin/python python scripts/verify_dataset.py
uv run --no-project --python .venv/bin/python python -m unittest \
  evaluation.test_strict_evaluator evaluation.test_neurips_metrics evaluation.test_current
```

[Current results](results/current/README.md) were rescored from the saved responses for the fixed 14-model cohort. They include input checksums. Raw response archives are not included in this Git checkout; reproducing those model results requires the corresponding archives or new model runs.

## Tasks

| Pure GPS (9) | Applied (8) |
|---|---|
| Format Conversion | Place Association |
| Coordinate Transformation | Name Disambiguation |
| Distance | Relative Position |
| Bearing | Proximity |
| Interpolation | Route Analysis |
| Area | Boundary Analysis |
| Bounding Box | Spatial Patterns |
| Route Geometry | Terrain Classification |
| Relative Position | |

## License

Code and evaluator: [MIT](LICENSE). Derived task files: [CC BY 4.0](DATA_LICENSE). GeoNames source data also requires attribution under CC BY 4.0.
