# GPSBench Evaluation Results

**Model:** mistralai/mistral-small-24b-instruct-2501
**Provider:** openrouter
**Timestamp:** 2025-12-28T10:29:35.767236
**Batch Mode:** False

## Overall Results

- **Total Cases:** 18362
- **Total Correct:** 8360
- **Overall Accuracy:** 45.53%

## Track Results

### Pure GPS Track

- **Accuracy:** 39.71%
- **Correct:** 3646/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 89.0% (908/1020)
  ❌ **Coordinate System Transformation:** 8.7% (89/1022)
  ❌ **Distance Calculation:** 28.6% (292/1020)
  ❌ **Bearing Computation:** 16.2% (165/1020)
  ❌ **Coordinate Interpolation:** 0.1% (1/1020)
  ❌ **Polygon Area:** 1.6% (16/1020)
  ✅ **Bounding Box:** 81.2% (828/1020)
  ❌ **Route Geometry:** 37.1% (378/1020)
  ✅ **Relative Position:** 95.0% (969/1020)

### Applied Track

- **Accuracy:** 51.35%
- **Correct:** 4714/9180
- **Tasks:** 9

  ❌ **Place Association:** 2.6% (27/1020)
  ✅ **Name Disambiguation:** 99.5% (1015/1020)
  ❌ **Relative Position:** 0.2% (2/1020)
  ❌ **Proximity & Nearest Neighbor:** 0.6% (6/1020)
  ✅ **Route Analysis:** 82.0% (836/1020)
  ✅ **Boundary Analysis:** 99.3% (1013/1020)
  ✅ **Spatial Patterns:** 85.9% (876/1020)
  ✅ **Missing Data Inference:** 75.7% (772/1020)
  ❌ **Terrain Classification:** 16.4% (167/1020)

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
