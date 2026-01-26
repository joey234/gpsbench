# GPSBench Evaluation Results

**Model:** qwen/qwen3-235b-a22b-2507
**Provider:** openrouter
**Timestamp:** 2025-12-04T21:24:40.745961
**Batch Mode:** False

## Overall Results

- **Total Cases:** 18364
- **Total Correct:** 11858
- **Overall Accuracy:** 64.57%

## Track Results

### Pure GPS Track

- **Accuracy:** 56.07%
- **Correct:** 5148/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 86.3% (880/1020)
  ❌ **Coordinate System Transformation:** 14.7% (150/1022)
  ✅ **Distance Calculation:** 96.6% (985/1020)
  ✅ **Bearing Computation:** 62.5% (637/1020)
  ❌ **Coordinate Interpolation:** 11.9% (121/1020)
  ❌ **Polygon Area:** 4.4% (45/1020)
  ✅ **Bounding Box:** 99.7% (1017/1020)
  ❌ **Route Geometry:** 30.5% (311/1020)
  ✅ **Relative Position:** 98.2% (1002/1020)

### Applied Track

- **Accuracy:** 73.08%
- **Correct:** 6710/9182
- **Tasks:** 9

  ❌ **Place Association:** 3.5% (36/1022)
  ✅ **Name Disambiguation:** 100.0% (1020/1020)
  ✅ **Relative Position:** 98.4% (1004/1020)
  ✅ **Proximity & Nearest Neighbor:** 70.9% (723/1020)
  ✅ **Route Analysis:** 88.3% (901/1020)
  ✅ **Boundary Analysis:** 97.4% (993/1020)
  ✅ **Spatial Patterns:** 92.6% (945/1020)
  ✅ **Missing Data Inference:** 87.8% (896/1020)
  ❌ **Terrain Classification:** 18.8% (192/1020)

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
