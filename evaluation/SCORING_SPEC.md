# Current GPSBench scoring

Version: `gpsbench-neurips-metrics-restored-v1`. Implemented by `current.py`, using `strict_evaluator.py` for extraction and matching and `neurips_metrics.py` for aggregation.

## Answer extraction

Use the explicit `FINAL ANSWER:` payload. Contradictory repeated final answers are invalid. Without that marker, accept only a direct, single-line response of at most 300 characters. Do not search reasoning for a convenient number or city name. Quantities must be finite and unambiguous. Invalid answers count as incorrect on accuracy tasks.

## Headline metrics

| Task | Scoring rule |
|---|---|
| Distance | `max(0, 100 × (1 − mean(abs(predicted − gold) / abs(gold))))` |
| Area | Same relative-error score; score area, not perimeter |
| Bearing | Wrapped shortest angular error / 180°, then the same complementary mean-error score |
| Interpolation | Both latitude and longitude within ±0.5° |
| Bounding Box | Each of four bounds within ±0.5° |
| Coordinate Transformation | Both UTM coordinates within ±10 m, or both Web Mercator coordinates within ±100 m; zone is not scored |
| Format Conversion | Requested coordinate format, both axes within 0.00005° |
| Relative Position (both tracks) | One correct cardinal direction, computed from latitude/longitude differences |
| Place Association | Canonical GeoNames city name; any country/administrative qualifiers must match the reference |
| Name Disambiguation | Correct candidate identity, allowing identical displayed candidates |
| Proximity | One correct candidate name |
| Route Analysis | One unambiguous Yes/No answer |
| Boundary Analysis | All cities assigned to their correct continental groups |
| Spatial Patterns | One correct outlier name |
| Terrain Classification | Correct option |
| Route Geometry | Correct option or the complete reference set of retained point indices |
| Missing Data (additional probe) | Missing coordinate within ±0.1°; excluded from track and overall scores |

For the three numerical tasks, include only parsed, finite normalized errors with `0 <= error < 1000`. Unparsed answers, undefined relative errors, and extreme errors are excluded from the mean. No included answers yields score zero. Clip only after averaging; do not average individually clipped scores. Report parsed and included counts alongside these scores.

The `correct`/`accuracy` fields for Distance, Bearing, and Area retain binary tolerance diagnostics. They are not the headline metrics. Interpolation's stored `tolerance_deg: 0.01` is a data-generation annotation; the current primary evaluator explicitly overrides it with 0.5°. The underlying dataset is preserved.

Track means give equal weight to the nine Pure GPS or eight Applied task scores. Overall gives equal weight to all 17 tasks. Missing Data is always excluded. A subset run averages only its evaluated main tasks and is marked incomplete; it is not a full benchmark score.

## Historical versions

Current reference answers are exported in `data/scoring_references.json`. Relative Position recomputation changes 243 Pure GPS and 263 Applied test labels compared with the stored historical `ground_truth.answer`; these corrections were already used for the published scores. The export also lists corrections for training and development data. Use `evaluation.references.scoring_example` to obtain current labels and an accepted answer without historical worked solutions.

`SCORING_SPEC_V1.md` describes the original answer extraction. `SCORING_SPEC_V2.md` describes the earlier binary analysis. Current scoring combines corrected extraction/matching with the restored numerical metrics above. Neither historical document overrides this specification.
