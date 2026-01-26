# GPSBench Evaluation Results

**Model:** gemini-2.5-pro
**Provider:** gemini
**Timestamp:** 2025-12-10T18:21:09.248983
**Batch Mode:** True

## Overall Results

- **Total Cases:** 18359
- **Total Correct:** 13278
- **Overall Accuracy:** 72.32%

## Track Results

### Pure GPS Track

- **Accuracy:** 74.64%
- **Correct:** 6851/9179
- **Tasks:** 9

  ✅ **Format Conversion:** 86.4% (881/1020)
  ✅ **Coordinate System Transformation:** 82.5% (843/1022)
  ✅ **Distance Calculation:** 99.9% (1019/1020)
  ✅ **Bearing Computation:** 92.5% (943/1020)
  ✅ **Coordinate Interpolation:** 52.1% (531/1020)
  ❌ **Polygon Area:** 15.2% (155/1017)
  ✅ **Bounding Box:** 100.0% (1020/1020)
  ❌ **Route Geometry:** 43.6% (445/1020)
  ✅ **Relative Position:** 99.4% (1014/1020)

### Applied Track

- **Accuracy:** 70.01%
- **Correct:** 6427/9180
- **Tasks:** 9

  ❌ **Place Association:** 23.0% (235/1020)
  ✅ **Name Disambiguation:** 99.9% (1019/1020)
  ✅ **Relative Position:** 99.2% (1012/1020)
  ✅ **Proximity & Nearest Neighbor:** 86.4% (881/1020)
  ✅ **Route Analysis:** 91.7% (935/1020)
  ❌ **Boundary Analysis:** 15.8% (161/1020)
  ✅ **Spatial Patterns:** 95.4% (973/1020)
  ✅ **Missing Data Inference:** 97.1% (990/1020)
  ❌ **Terrain Classification:** 21.7% (221/1020)

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
