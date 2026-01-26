# GPSBench Evaluation Results

**Model:** gpt-4.1-mini
**Provider:** openai
**Timestamp:** 2025-12-01T22:19:15.952023
**Batch Mode:** True

## Overall Results

- **Total Cases:** 18362
- **Total Correct:** 12582
- **Overall Accuracy:** 68.52%

## Track Results

### Pure GPS Track

- **Accuracy:** 62.09%
- **Correct:** 5701/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 86.4% (881/1020)
  ❌ **Coordinate System Transformation:** 15.7% (160/1022)
  ✅ **Distance Calculation:** 94.0% (959/1020)
  ✅ **Bearing Computation:** 92.2% (940/1020)
  ❌ **Coordinate Interpolation:** 9.1% (93/1020)
  ❌ **Polygon Area:** 3.5% (36/1020)
  ✅ **Bounding Box:** 99.9% (1019/1020)
  ✅ **Route Geometry:** 59.0% (602/1020)
  ✅ **Relative Position:** 99.1% (1011/1020)

### Applied Track

- **Accuracy:** 74.96%
- **Correct:** 6881/9180
- **Tasks:** 9

  ❌ **Place Association:** 4.0% (41/1020)
  ✅ **Name Disambiguation:** 100.0% (1020/1020)
  ✅ **Relative Position:** 99.4% (1014/1020)
  ✅ **Proximity & Nearest Neighbor:** 82.4% (840/1020)
  ✅ **Route Analysis:** 89.3% (911/1020)
  ✅ **Boundary Analysis:** 99.6% (1016/1020)
  ✅ **Spatial Patterns:** 92.5% (944/1020)
  ✅ **Missing Data Inference:** 88.4% (902/1020)
  ❌ **Terrain Classification:** 18.9% (193/1020)

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
