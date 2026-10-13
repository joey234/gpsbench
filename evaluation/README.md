# Evaluation

Run new evaluations with [`run_benchmark.py`](../run_benchmark.py). Score saved responses with:

```bash
uv run --no-project --python .venv/bin/python python -m evaluation.current \
  results/YOUR_RUN --data data --out /tmp/current-scores.json
```

The output file must not already exist. This command reads one run's `task_results/` files, validates example IDs and saved references against the released test splits, and writes a separate summary. It makes no API calls. Missing or incomplete task files are marked as a partial evaluation, not a complete benchmark result.

## Current components

| File | Purpose |
|---|---|
| `registry.py` | Single task list: 17 main tasks and one optional probe |
| `current.py` | Current answer scoring, task aggregation, and offline CLI |
| `strict_evaluator.py` | Corrected final-answer parsing and task matching |
| `neurips_metrics.py` | Numerical headline metrics and Interpolation precision |
| `SCORING_SPEC.md` | Current metric definitions and denominator rules |
| `cohort.json` | Fixed saved-run directories for the 14-model paper comparison |
| `llm_client.py` | Model-provider clients used by the runner |
| `track_config.py` | Current task names/counts for older track utilities |

Use `score`, not `overall_accuracy`, for benchmark comparisons. Distance, Bearing, and Area also retain tolerance-based correctness as a diagnostic. `parsed` and `included` disclose the denominator for numerical scores. Missing Data has `role: diagnostic` and never enters the 17-task mean.

## Archived code

`evaluate_pure_gps.py`, `evaluate_applied.py`, `task_evaluators.py`, and the `layer_*` modules are earlier evaluation workflows. They remain for historical experiments and do not implement the current paper protocol. Fine-tuning comparison utilities also retain their original behavior. Start from `current.py` for current scoring.

Historical scoring descriptions are in `SCORING_SPEC_V1.md` and `SCORING_SPEC_V2.md`; the older Layer A guide is in [`docs/history/`](../docs/history/). Historical result summaries must be rescored before comparison with current scores.
