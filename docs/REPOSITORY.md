# Repository guide

## Current benchmark

- `data/`: released examples, split metadata, checksums, and dataset guide.
- `evaluation/`: current scoring, task registry, provider client, and tests. Its README identifies earlier evaluators.
- `run_benchmark.py`: entry point for new model runs, including batch runs.
- `prompts/`: templates used to construct model inputs.
- `results/current/`: verified current scores and provenance; earlier result folders preserve historical experiments.
- `leaderboard/`: static project website, generated score data, and task descriptions.
- `scripts/`: dataset checks and result export, plus historical generation/analysis tools. See its README before choosing a script.

## Research and history

- `analysis/`: experiment outputs and research analyses; these may use different scoring versions.
- `paper/`: historical manuscript source and figures.
- `finetuning/`: earlier fine-tuning experiments, with their own dependencies.
- `downstream-gps-aug-eval/`: coordinate-augmentation experiments.
- `docs/history/`: superseded documentation.

Manuscript working copies and review notes may exist locally without being part of the public benchmark release. They are not required to run the current evaluator. Established paths are preserved so older experiments can still find their inputs.

## Updating a release

1. Run `scripts/verify_dataset.py`, `scripts/export_scoring_references.py --check`, and the current evaluator tests. See [paper alignment checks](PAPER_ALIGNMENT.md).
2. With the saved response archives present, run `scripts/export_current_results.py`.
3. Run `scripts/export_leaderboard.py` to refresh the website from `results/current/scores.json`.
4. Review score differences and the staged file list before publishing.

Do not regenerate dataset splits or overwrite response archives as part of a scoring update. New model runs and offline scoring are separate operations.
