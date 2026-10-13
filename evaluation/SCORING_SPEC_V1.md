# Historical v1 scoring — superseded

Current scoring is documented in [SCORING_SPEC.md](SCORING_SPEC.md).

# GPSBench v1.0 scoring specification

Canonical protocol from the author rebuttal. Section 4.1, Appendix C, and
`evaluation/frozen_evaluator.py` implement this table.

| Metric | Tasks |
|---|---|
| **1−MAPE** (headline) | Distance, Bearing, Area & Perimeter |
| per-axis ±0.5° | Interpolation, Bounding Box |
| absolute (Web Mercator ±100 m, UTM ±10 m) | Coordinate Transformation |
| ±0.1° | Missing Data (diagnostic) |
| exact / set / MCQ match | Format Conversion, Route Geometry, Relative Position, Proximity, Route Analysis, Boundary Analysis, Spatial Patterns, Name Disambiguation, Terrain |
| hierarchical name match | Place Association (headline = city-level name; country/region in RQ2) |

```
python3 evaluation/frozen_evaluator.py results --data data \
  --out evaluation/frozen_rescore/frozen_scores.json
python3 scripts/export_v1_scores.py
```

## 1−MAPE (non-MCQ calculation tasks)

Matches `scripts/compute_error_metrics.py` (`compute_metrics_from_errors`):

- Parse the first number from `FINAL ANSWER:`.
- Relative error \(e_i = | \hat y_i - y_i | / | y_i |\) on items with a parsed number and \(y_i \ne 0\).
- Drop \(|e_i| \ge 1000\). Bearing uses angular error / 180° (wraparound) so the ratio is well-defined near 0°.
- \(\mathrm{MAPE} = 100 \times \mathrm{mean}(e_i)\). Headline score \(= \max(0,\; 100 - \mathrm{MAPE})\).

Binary tolerances for these three tasks are stored only as a sensitivity column.

## Other official rules

- Interpolation / Bounding Box: each reported axis within **±0.5°** (rebuttal; not the 5 km / 0.1° runner defaults).
- Coordinate Transformation: Web Mercator ±100 m, UTM ±10 m.
- Place Association headline: gold city name in the answer. RQ2 reports the same items at country / region / city.
- Missing Data is not part of track or overall means.
