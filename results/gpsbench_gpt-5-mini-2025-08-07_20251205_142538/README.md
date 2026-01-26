# GPSBench Evaluation Results

**Model:** gpt-5-mini-2025-08-07
**Provider:** openai
**Timestamp:** 2025-12-05T14:25:38.267304
**Batch Mode:** True

## Overall Results

- **Total Cases:** 18364
- **Total Correct:** 14489
- **Overall Accuracy:** 78.90%

## Track Results

### Pure GPS Track

- **Accuracy:** 82.48%
- **Correct:** 7573/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 86.4% (881/1020)
  ✅ **Coordinate System Transformation:** 84.6% (865/1022)
  ✅ **Distance Calculation:** 100.0% (1020/1020)
  ✅ **Bearing Computation:** 99.7% (1017/1020)
  ✅ **Coordinate Interpolation:** 71.4% (728/1020)
  ❌ **Polygon Area:** 4.7% (48/1020)
  ✅ **Bounding Box:** 99.7% (1017/1020)
  ✅ **Route Geometry:** 96.3% (982/1020)
  ✅ **Relative Position:** 99.5% (1015/1020)

### Applied Track

- **Accuracy:** 75.32%
- **Correct:** 6916/9182
- **Tasks:** 9

  ❌ **Place Association:** 39.7% (406/1022)
  ✅ **Name Disambiguation:** 99.6% (1016/1020)
  ✅ **Relative Position:** 99.6% (1016/1020)
  ✅ **Proximity & Nearest Neighbor:** 73.3% (748/1020)
  ✅ **Route Analysis:** 81.7% (833/1020)
  ✅ **Boundary Analysis:** 99.6% (1016/1020)
  ✅ **Spatial Patterns:** 93.7% (956/1020)
  ✅ **Missing Data Inference:** 72.0% (734/1020)
  ❌ **Terrain Classification:** 18.7% (191/1020)


## Error Metrics
Detailed error analysis for tasks with numeric outputs:

### Pure Gps Task4 Distance Calculation
**Absolute Error (km):**
- Mean: 5.29 km
- Median: 1.83 km
- 95th percentile: 21.35 km

**Relative Error:**
- Mean: 0.064%
- Median: 0.027%

**Accuracy at Different Tolerances:**

| Tolerance | Accuracy | Examples |
|-----------|----------|----------|
| ±0.1% of distance | 81.0% | 826/1020 |
| ±0.5% of distance | 98.8% | 1008/1020 |
| ±1.0% of distance | 100.0% | 1020/1020 |
| ±2.0% of distance | 100.0% | 1020/1020 |
| ±3.0% of distance | 100.0% | 1020/1020 |
| ±5.0% of distance | 100.0% | 1020/1020 |

### Pure Gps Task5 Bearing Computation
**Absolute Error (degrees):**
- Mean: 0.159°
- Median: 0.020°
- 95th percentile: 0.331°

**Accuracy at Different Tolerances:**

| Tolerance | Accuracy | Examples |
|-----------|----------|----------|
| ±0.1° | 83.9% | 856/1020 |
| ±0.5° | 96.4% | 983/1020 |
| ±1.0° | 99.2% | 1012/1020 |
| ±2.0° | 99.5% | 1015/1020 |
| ±3.0° | 99.5% | 1015/1020 |
| ±5.0° | 99.7% | 1017/1020 |

## File Structure

```
.
├── README.md              # This file
├── summary.json           # Overall metrics and configuration
├── track_pure_gps.json    # Pure GPS track summary
├── track_applied.json     # Applied track summary
└── task_results/          # Individual task results
    ├── pure_gps_task1_format_conversion.json
    ├── pure_gps_task3_coordinate_system_transformation.json
    └── ...
```

## Files Description

- **summary.json**: High-level overview with overall metrics
- **track_*.json**: Track-level summaries with task references
- **task_results/*.json**: Complete results for each task including:
  - All prompts and responses
  - Ground truth comparisons
  - Individual sample evaluations
  - Token usage and latency metrics
