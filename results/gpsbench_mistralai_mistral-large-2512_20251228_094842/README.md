# GPSBench Evaluation Results

**Model:** mistralai/mistral-large-2512
**Provider:** openrouter
**Timestamp:** 2025-12-28T07:43:08.325589
**Batch Mode:** False

## Overall Results

- **Total Cases:** 18362
- **Total Correct:** 11054
- **Overall Accuracy:** 60.20%

## Track Results

### Pure GPS Track

- **Accuracy:** 50.10%
- **Correct:** 4600/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 86.4% (881/1020)
  ✅ **Coordinate System Transformation:** 62.1% (635/1022)
  ❌ **Distance Calculation:** 47.4% (483/1020)
  ❌ **Bearing Computation:** 21.9% (223/1020)
  ❌ **Coordinate Interpolation:** 0.7% (7/1020)
  ❌ **Polygon Area:** 1.9% (19/1020)
  ✅ **Bounding Box:** 99.7% (1017/1020)
  ❌ **Route Geometry:** 39.0% (398/1020)
  ✅ **Relative Position:** 91.9% (937/1020)

### Applied Track

- **Accuracy:** 70.31%
- **Correct:** 6454/9180
- **Tasks:** 9

  ❌ **Place Association:** 10.6% (108/1020)
  ✅ **Name Disambiguation:** 99.3% (1013/1020)
  ✅ **Relative Position:** 74.3% (758/1020)
  ✅ **Proximity & Nearest Neighbor:** 81.8% (834/1020)
  ✅ **Route Analysis:** 87.9% (897/1020)
  ✅ **Boundary Analysis:** 83.6% (853/1020)
  ✅ **Spatial Patterns:** 90.1% (919/1020)
  ✅ **Missing Data Inference:** 87.4% (891/1020)
  ❌ **Terrain Classification:** 17.7% (181/1020)

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
