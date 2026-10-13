# Dataset and paper alignment

Checked against the current ACL manuscript on 2026-10-13.

| Item | Verified result |
|---|---|
| Main tasks | 17: nine Pure GPS and eight Applied |
| Main examples | 58,614: 35,522 train / 5,750 dev / 17,342 test |
| Additional Missing Data probe | 3,400 examples; excluded from main averages |
| Split metadata | All 18 tasks match the actual split sizes |
| Dataset integrity | 72 split/metadata checksums match; unsplit examples match split examples, ignoring added historical worked solutions |
| Name Disambiguation | 1,020 test rows, 470 unique question strings |
| Question overlap | No identical Place Association coordinate strings or Name Disambiguation question strings across splits |
| Scoring references | All 17,342 main and 1,020 probe test reference answers are accepted by the current scorer |
| Published table values | 238 task scores and 42 track/overall scores match the current manuscript to its displayed precision |

## Historical annotations versus scoring references

The scorer's Relative Position rule uses wrapped coordinate-axis differences. Some historical `ground_truth.answer` fields use different directions. The corrected references are now explicit in `data/scoring_references.json`:

| Track | Train corrections | Dev corrections | Test corrections |
|---|---:|---:|---:|
| Pure GPS | 524 | 77 | 243 |
| Applied | 499 | 91 | 263 |

These directions were already used in the published model scores; the reference export does not change those scores. `evaluation.references.scoring_example` returns the current labels and an accepted answer, omitting historical `reasoning_steps`. Archived source files remain unchanged so saved-response joins and checksums remain reproducible.

Interpolation's historical annotation is 0.01°, while the paper and current scorer explicitly use 0.5° as the primary threshold. This distinction is documented in both the paper and scoring specification.

Distance, Bearing, and Area use complementary mean-error scores, with the stated inclusion rules. All other main tasks use accuracy. Overall and track means equally weight tasks. Tests cover projection, format conversion, bounding-box and missing-coordinate precision boundaries, parsing, invalid answers, and aggregation.

These checks establish agreement among the paper, released examples, current references, and evaluator. They do not independently validate every geographic annotation or historical worked solution.

## Repeat the checks

```bash
uv run --no-project --python .venv/bin/python python scripts/verify_dataset.py
uv run --no-project --python .venv/bin/python python scripts/export_scoring_references.py --check
uv run --no-project --python .venv/bin/python python -m unittest \
  evaluation.test_strict_evaluator evaluation.test_neurips_metrics \
  evaluation.test_current evaluation.test_references
uv run --no-project --python .venv/bin/python python scripts/verify_paper_scores.py PATH_TO_CURRENT_PAPER.tex
```
