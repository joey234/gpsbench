# Released dataset

Use `track_pure_gps/splits/` and `track_applied/splits/`. Files ending in `_train.json`, `_dev.json`, and `_test.json` are the released splits. The runner evaluates test files only.

| Scope | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| 17 main tasks | 35,522 | 5,750 | 17,342 | 58,614 |
| Missing Data probe | 2,040 | 340 | 1,020 | 3,400 |

Most main tasks contain 3,400 examples (2,040/340/1,020). Coordinate Transformation uses 2,039/339/1,022. Name Disambiguation contains 4,214 examples (2,883/311/1,020), including repeated questions: its 1,020 test rows represent 470 unique questions. Preserve those rows when reproducing the paper.

Unsplit JSON files live beside each `splits/` directory. Split files may additionally contain `reasoning_steps`; these are reference solutions, not model inputs. Test labels and split membership have not been regenerated. The Name Disambiguation unsplit file and split metadata have been corrected to reflect the released splits.

Each row has a task identifier, task-specific input fields, and `ground_truth`. See the actual JSON for each task's schema; there is no universal `question` field.

[MANIFEST.json](MANIFEST.json) records counts and SHA-256 checksums for all split files and split metadata. [MANIFEST.md](MANIFEST.md) presents the same information for reading.

```bash
uv run --no-project --python .venv/bin/python python scripts/verify_dataset.py
```

The historical GeoNames source dump is local and ignored by Git. It is not needed to score released examples. Newer GeoNames downloads may differ from the source used to build this release. See [DATA_LICENSE](../DATA_LICENSE) for attribution.
