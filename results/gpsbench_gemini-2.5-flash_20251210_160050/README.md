# GPSBench Evaluation Results

**Model:** gemini-2.5-flash
**Provider:** gemini
**Timestamp:** 2025-12-10T16:00:50.213227
**Batch Mode:** True

## Overall Results

- **Total Cases:** 18359
- **Total Correct:** 11795
- **Overall Accuracy:** 64.25%

## Track Results

### Pure GPS Track

- **Accuracy:** 58.79%
- **Correct:** 5396/9179
- **Tasks:** 9

  ✅ **Format Conversion:** 86.4% (881/1020)
  ❌ **Coordinate System Transformation:** 41.0% (419/1022)
  ✅ **Distance Calculation:** 75.7% (772/1020)
  ✅ **Bearing Computation:** 79.9% (815/1020)
  ❌ **Coordinate Interpolation:** 2.9% (30/1020)
  ❌ **Polygon Area:** 7.2% (73/1017)
  ✅ **Bounding Box:** 100.0% (1020/1020)
  ❌ **Route Geometry:** 38.7% (395/1020)
  ✅ **Relative Position:** 97.2% (991/1020)

### Applied Track

- **Accuracy:** 69.71%
- **Correct:** 6399/9180
- **Tasks:** 9

  ❌ **Place Association:** 17.2% (175/1020)
  ✅ **Name Disambiguation:** 99.3% (1013/1020)
  ✅ **Relative Position:** 96.6% (985/1020)
  ✅ **Proximity & Nearest Neighbor:** 80.1% (817/1020)
  ✅ **Route Analysis:** 87.1% (888/1020)
  ❌ **Boundary Analysis:** 47.4% (483/1020)
  ✅ **Spatial Patterns:** 92.4% (942/1020)
  ✅ **Missing Data Inference:** 87.4% (891/1020)
  ❌ **Terrain Classification:** 20.1% (205/1020)

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
