# Current results

`scores.json` contains the 14-model comparison under `gpsbench-neurips-metrics-restored-v1`, with 17 task scores per model and SHA-256 checksums of the inputs and scoring code. Scores are on a 0–100 scale; Distance, Bearing, and Area are error-based scores. Missing Data is excluded.

These scores were recomputed directly from saved responses with the current evaluator. Every task score was compared with the corresponding current manuscript analysis. Historical `track_*.json` accuracy summaries use older rules and should not be substituted for these results.

To reproduce with the original response archives available at the paths in `evaluation/cohort.json`:

```bash
uv run --no-project --python .venv/bin/python python scripts/export_current_results.py
uv run --no-project --python .venv/bin/python python scripts/export_leaderboard.py
```

The raw response archives are not tracked in this repository. Without them, the dataset and scoring code can still be verified, and new responses can be generated with `run_benchmark.py`.
