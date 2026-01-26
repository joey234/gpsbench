# GPSBench Evaluation Results

**Model:** qwen/qwen3-30b-a3b-instruct-2507
**Provider:** openrouter
**Timestamp:** 2025-12-03T11:15:57.094920
**Batch Mode:** False

## Overall Results

- **Total Cases:** 18364
- **Total Correct:** 10167
- **Overall Accuracy:** 55.36%

## Track Results

### Pure GPS Track

- **Accuracy:** 55.35%
- **Correct:** 5082/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 86.4% (881/1020)
  ❌ **Coordinate System Transformation:** 12.0% (123/1022)
  ✅ **Distance Calculation:** 94.3% (962/1020)
  ✅ **Bearing Computation:** 81.7% (833/1020)
  ❌ **Coordinate Interpolation:** 6.4% (65/1020)
  ❌ **Polygon Area:** 1.3% (13/1020)
  ✅ **Bounding Box:** 100.0% (1020/1020)
  ❌ **Route Geometry:** 18.7% (191/1020)
  ✅ **Relative Position:** 97.5% (994/1020)

### Applied Track

- **Accuracy:** 55.38%
- **Correct:** 5085/9182
- **Tasks:** 9

  ❌ **Place Association:** 1.3% (13/1022)
  ✅ **Name Disambiguation:** 100.0% (1020/1020)
  ✅ **Relative Position:** 97.9% (999/1020)
  ❌ **Proximity & Nearest Neighbor:** 24.2% (247/1020)
  ✅ **Route Analysis:** 85.7% (874/1020)
  ✅ **Boundary Analysis:** 99.4% (1014/1020)
  ❌ **Spatial Patterns:** 8.1% (83/1020)
  ✅ **Missing Data Inference:** 65.4% (667/1020)
  ❌ **Terrain Classification:** 16.5% (168/1020)

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
