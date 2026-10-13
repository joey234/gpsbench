# Evaluation results

Use [`current/scores.json`](current/scores.json) for the verified 14-model comparison under the current scoring rules. See [`current/README.md`](current/README.md) for provenance and reproduction.

New runs are saved under `gpsbench_MODEL_TIMESTAMP/`:

- `summary.json`: scoring version, completeness, track scores, and overall score.
- `track_pure_gps.json` / `track_applied.json`: task summaries.
- `task_results/TRACK_TASK.json`: prompts, model responses, example IDs, references, and per-example evaluations.
- `README.md`: readable run summary.

Raw predictions and generated run summaries are ignored by Git. Some historical metadata and analysis files remain tracked for reference. Earlier `accuracy` and `overall_accuracy` values are not current headline scores.

To rescore a saved run without model requests or changing the archive:

```bash
uv run --no-project --python .venv/bin/python python -m evaluation.current \
  results/YOUR_RUN --out /tmp/current-scores.json
```
