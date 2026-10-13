# GPSBench v2.2 all-task scoring protocol

This document describes the earlier binary analysis retained in `analysis/paper_revision_v2`. The current ACL paper restores the NeurIPS numerical metrics and ±0.5° Interpolation threshold; see [the current scoring and reproduction notes](SCORING_SPEC.md). The parsing and unaffected task-matching rules below are still used.

This version rescores a fixed cohort of historical responses. It does not replace `frozen_evaluator.py`, change source labels, or claim that the entire release is semantically validated. `cohort.json` lists the exact 14 directories. Result rows join source rows by `example_id`, or the saved `index` field for GPT-5.1 Area/Missing Data; uniqueness, complete coverage, and saved/source ground-truth compatibility are asserted.

## Scope and primary metric

All 17 main tasks are evaluated: nine Pure GPS and eight Applied. Each has 1,020 test items except Transformation (1,022), giving 17,342 items per model and 242,788 responses across 14 models. Overall equally weights all 17 task accuracies; track means equally weight their nine/eight tasks. Missing Data is separate. Invalid, unparsed, and empty answers are failures. Spatial Patterns contributes to the aggregate and is interpreted alongside its 100% last-candidate baseline.

## Answer extraction and parsing

Use the explicit `FINAL ANSWER:` payload. Ignore empty answer headings immediately followed by another marker; multiple differing nonempty payloads fail. Without that marker, accept only a single-line direct response of at most 300 characters. Do not scan reasoning for an answer number. Normalize Unicode, simple LaTeX math wrappers, grouped digits, signs, `e`/`E` exponents, and multiplication-by-ten exponent notation. Require one finite scalar, a complete pair, four bounds, or one task-specific selection. Multiple known candidate names or contradictory Boolean tokens fail.

This is a deliberately conservative formatting protocol. It can reject semantically correct answers with extra numerical commentary, unsupported unit variants, aliases, unknown city qualifiers, or option explanations outside the declared syntax. Report its accuracy as protocol compliance plus answer correctness, not pure knowledge. The reference self-check tests agreement with stored targets, not independent gold correctness.

## Task conventions

| Task | Target and acceptance |
|---|---|
| Format Conversion | Both latitude and longitude, each within `5e-5` degrees; require the requested decimal or DMS format. Hemisphere signs and physical ranges are checked. The release has 26 test reference strings with seconds rounded to exactly 60.0 without carrying: normalize that boundary arithmetically. Minutes >=60 and seconds >60 fail. |
| Transformation | Two projected coordinates. Remove labeled UTM zone/EPSG metadata before parsing. UTM ±10 m; Web Mercator ±100 m. This scores the pair, not separate zone/CRS identity. |
| Distance | One quantity in km, meters, or miles; convert to km. Use stored `tolerance_km` (typically 5%). Bare numbers mean km. |
| Bearing | One degree value (or explicit radians converted to degrees), wrapped angular difference <=5 degrees. Bare numbers mean degrees. |
| Interpolation | Latitude/longitude within the stored `tolerance_deg` on each axis (0.01 degrees for every retained test item). The primary version is `gpsbench-v2.2-all-tasks`. Larger 0.1/0.5-degree thresholds are sensitivities only. |
| Area | Area only, ±5% using stored percentage. km² or m²; bare numbers mean km². Explicit `Area:`/`Perimeter:` labels allow isolating area. Perimeter is not scored. |
| Bounding Box | Exactly four numbers in min-lat/max-lat/min-lon/max-lon order, physical coordinate ranges, ±0.5 degrees each. |
| Route Geometry | One A–D option letter, or an explicit unique index list equal to the stored kept-index set. |
| Place Association | Canonical city name after case/whitespace/outer punctuation normalization. Optional comma-separated or parenthesized qualifiers must match stored admin/country names or country code (US formatting variants are enumerated). Preserve canonical names containing commas and `Or Yehuda`. No alternate-city-name expansion or independent administrative validation. |
| Proximity | Exactly one known candidate name, with limited explanation/unit suffix syntax. Other known names and alternatives fail. No substring acceptance of all-candidate lists. |
| Route Analysis | One Boolean polarity using whole-word Yes/No/True/False/Correct/Incorrect tokens, beginning the payload. Contradictory polarities fail; `incorrect` does not match `correct`. |
| Terrain | One A–F option letter. Six displayed options give a uniform-choice expected accuracy of 1/6. |
| Relative Position (both tracks) | Compute the dominant signed latitude/longitude difference from B to A, in degrees. Wrap longitude to [-180,180]; latitude wins ties. Use `compute_actual_direction`'s rule from `transform_cardinal_to_mcq.py` for every item, without its Boolean-query shortcut. Accept only one North/South/East/West word; reject diagonals and alternatives. Reference directions are in `analysis/paper_revision_v2/relative_position_targets.json`; original source labels remain unchanged. There are 243 Pure GPS and 263 Applied disagreements with the source labels under this uniform definition. This is an operational coordinate-axis convention, not a four-sector initial-bearing definition. |
| Name Disambiguation | Map a selected letter to the displayed candidate; check name, country, and population against the source target. On three items, two displayed candidates have identical metadata; either is credited. Direct metadata answers must resolve a single displayed identity, modulo equivalent entries. A shared name alone fails. Uniform chance is 25.1% when equivalents are included. |
| Boundary Analysis | Parse the explicit final grouping, including multiline/bullet, semicolon, and city-first forms. Require the complete multiset of (city name, continent) assignments to match the reference, including every occurrence of repeated city names. Australia and Oceania are equivalent. Missing cities, excess occurrences, conflicting assignments, or a continent list without city assignments fail. A parsed complete mapping can still be incorrect. |
| Spatial Patterns | Require exactly one known candidate name and compare it to the target outlier. Do not credit lists or alternatives. Target position is always last; retain the task and report its 100% positional baseline. |
| Missing Data | Separate diagnostic: one finite missing-axis value, physical range and stored ±0.1-degree tolerance. |

## Secondary metrics and controls

Report Route Analysis positive/negative recall and balanced accuracy: half the sum of the two recalls. Ambiguous outputs remain incorrect in their true class. Always-No has 89.9% raw and 50% balanced accuracy. Proximity first-position baseline is 55.2%; Spatial Patterns last-position baseline is 100%. Source-copy Format Conversion accuracy is 0% under v2. MCQ chance controls are expected accuracies, not actual random-model experiments.

Conditional numerical scores use `max(0, 100*(1-mean(error)))` across finite parsed quantities. Distance/Area error is absolute relative error; Bearing error is wrapped degrees divided by 180. No extreme-error deletion. These secondary scores never enter the primary aggregate. Always report valid parsing counts out of 1,020.

## Uncertainty

5,000 percentile within-task item bootstrap samples, seed 42, preserve equal task weights. For binary outcomes, binomial resampling of success counts is equivalent to the marginal empirical item bootstrap. For the top pair, jointly resample paired correctness categories within each task (multinomial implemented by conditional binomials), preserving covariance. The selected top-pair comparison has no selection/multiplicity correction. Intervals are conditional on the saved responses and fixed task distribution; they do not model repeated inference, shared entities, geography clusters, endpoint drift, or unseen-place generalization. Entity-aware uncertainty requires a separate, specified grouping design.

## Reproduction

Requires Python >=3.12 for standard-library `Random.binomialvariate`.

```
uv run --no-project python -m unittest evaluation.test_strict_evaluator -v
uv run --no-project python scripts/rebuild_arr_v2.py --sync-paper
uv run --no-project python scripts/prepare_shuffled_candidates.py
```

The builder asserts all 17,342 canonical reference outputs pass. It writes input hashes, unrounded scores, denominators, confusion counts, intervals, and deterministic compressed per-item outcomes to `analysis/paper_revision_v2/`. Generated table and result-text markers sync directly into the open ARR master without replacing its style or other text. The frozen scorer and old analyses remain intact.

The all-task builder in `scripts/restore_all_task_results.py` also reanalyses existing parsed predictions at different Interpolation thresholds, without changing answer extraction. Interpolation uses ±0.01/0.1/0.5 degrees on each axis, with all other task cells fixed when reporting overall sensitivity. Its change is divided by 17 when updating the overall mean. Projection uses common ±1/10/100/1,000-m thresholds on each output axis, separately reporting UTM and Web Mercator and weighting their combined 507/515 test rows. Primary scores use stored Interpolation precision and the declared mixed projection tolerances. Hashes, counts, and unrounded sensitivity values are recorded in `tolerance_sensitivity.json`. No new model outputs or labels are generated by this reanalysis.

The current manuscript scope uses only existing artifacts. Earlier prepared candidate-order pairs are unevaluated and outside the paper evidence. No further experiments or budget requests are part of this scope; Spatial Patterns is included with its fixed-position baseline; no causal candidate-order effect is estimated.
