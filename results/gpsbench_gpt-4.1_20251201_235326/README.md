# GPSBench Evaluation Results

**Model:** gpt-4.1
**Provider:** openai
**Timestamp:** 2025-12-01T23:53:26.917019
**Batch Mode:** True

## Overall Results

- **Total Cases:** 18362
- **Total Correct:** 12966
- **Overall Accuracy:** 70.61%

## Track Results

### Pure GPS Track

- **Accuracy:** 63.95%
- **Correct:** 5872/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 86.4% (881/1020)
  ❌ **Coordinate System Transformation:** 27.8% (284/1022)
  ✅ **Distance Calculation:** 96.6% (985/1020)
  ✅ **Bearing Computation:** 97.7% (997/1020)
  ❌ **Coordinate Interpolation:** 11.4% (116/1020)
  ❌ **Polygon Area:** 10.2% (104/1020)
  ✅ **Bounding Box:** 100.0% (1020/1020)
  ❌ **Route Geometry:** 46.0% (469/1020)
  ✅ **Relative Position:** 99.6% (1016/1020)

### Applied Track

- **Accuracy:** 77.28%
- **Correct:** 7094/9180
- **Tasks:** 9

  ❌ **Place Association:** 14.9% (152/1020)
  ✅ **Name Disambiguation:** 100.0% (1020/1020)
  ✅ **Relative Position:** 99.8% (1018/1020)
  ✅ **Proximity & Nearest Neighbor:** 84.4% (861/1020)
  ✅ **Route Analysis:** 88.6% (904/1020)
  ✅ **Boundary Analysis:** 99.6% (1016/1020)
  ✅ **Spatial Patterns:** 94.2% (961/1020)
  ✅ **Missing Data Inference:** 94.7% (966/1020)
  ❌ **Terrain Classification:** 19.2% (196/1020)

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
