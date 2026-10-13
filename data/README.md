# Released dataset

Use `track_pure_gps/splits/` and `track_applied/splits/`. Files ending in `_train.json`, `_dev.json`, and `_test.json` are the released splits. The runner evaluates test files only.

| Scope | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| 17 main tasks | 35,522 | 5,750 | 17,342 | 58,614 |
| Missing Data probe | 2,040 | 340 | 1,020 | 3,400 |

Most main tasks contain 3,400 examples (2,040/340/1,020). Coordinate Transformation uses 2,039/339/1,022. Name Disambiguation contains 4,214 examples (2,883/311/1,020), including repeated questions: its 1,020 test rows represent 470 unique questions. Preserve those rows when reproducing the paper.

Unsplit JSON files live beside each `splits/` directory. Split files may additionally contain historical `reasoning_steps`; some use older labels or answer formats, so they are not authoritative scoring references or model inputs. Original annotations and split membership are preserved. The Name Disambiguation unsplit file and split metadata have been corrected to reflect the released splits.

## Current scoring references

Use [`scoring_references.json`](scoring_references.json) or `evaluation.references.scoring_example` for the current reference answers. The export contains answers for all 17,342 main test examples and 1,020 Missing Data test examples, plus explicit direction corrections for every affected train/dev/test row.

The paper's Relative Position rule differs from 243 Pure GPS and 263 Applied stored test labels. Across all splits, 1,697 direction labels differ. The current scorer already recomputes these directions; the published model scores use that rule. Interpolation's stored 0.01° annotation is also superseded by the paper's primary 0.5° threshold.

```python
from evaluation.references import scoring_example

example = scoring_example(source_row)
# Current ground_truth and scoring_reference_answer; historical reasoning_steps omitted.
```

`scoring_example` returns a copy and does not change the archived source row. Consumers training on the current benchmark should use this view, not the historical direction labels or worked solutions.

Each row has a task identifier, task-specific input fields, and `ground_truth`. See the actual JSON for each task's schema; there is no universal `question` field.

[MANIFEST.json](MANIFEST.json) records counts and SHA-256 checksums for all split files and split metadata. [MANIFEST.md](MANIFEST.md) presents the same information for reading.

```bash
uv run --no-project --python .venv/bin/python python scripts/verify_dataset.py
```

The historical GeoNames source dump is local and ignored by Git. It is not needed to score released examples. Newer GeoNames downloads may differ from the source used to build this release. See [DATA_LICENSE](../DATA_LICENSE) for attribution.
