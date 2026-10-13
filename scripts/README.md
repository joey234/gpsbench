# Script guide

Run from the repository root through the local uv environment.

| Current release task | Command after `uv run --no-project --python .venv/bin/python python` |
|---|---|
| Verify dataset counts, checksums, and unsplit consistency | `scripts/verify_dataset.py` |
| Score one saved run without API calls | `-m evaluation.current results/YOUR_RUN --out /tmp/scores.json` |
| Rebuild public model scores from fixed response archives | `scripts/export_current_results.py` |
| Refresh website data from public scores | `scripts/export_leaderboard.py` |
| Rebuild manifest after an intentional dataset release change | `scripts/build_data_manifest.py` |

The remaining scripts support data generation, older analyses, plotting, fine-tuning, or manuscript preparation. Their filenames do not imply that they use current scoring. In particular, legacy accuracy summaries must not replace `results/current/scores.json`.

Data-generation and split-generation scripts modify the dataset and are not part of the reproduction workflow. Manuscript-specific scripts can require local files absent from the public repository.
