# GPSBench Evaluation Results

**Model:** qwen/qwen3-8b
**Provider:** openrouter
**Timestamp:** 2026-01-14T15:44:54.489391
**Batch Mode:** False

## Overall Results

- **Total Cases:** 18362
- **Total Correct:** 9744
- **Overall Accuracy:** 53.07%

## Track Results

### Pure GPS Track

- **Accuracy:** 43.90%
- **Correct:** 4031/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 87.5% (893/1020)
  ❌ **Coordinate System Transformation:** 10.9% (111/1022)
  ✅ **Distance Calculation:** 72.5% (740/1020)
  ✅ **Bearing Computation:** 63.0% (643/1020)
  ❌ **Coordinate Interpolation:** 0.9% (9/1020)
  ❌ **Polygon Area:** 1.5% (15/1020)
  ✅ **Bounding Box:** 94.9% (968/1020)
  ❌ **Route Geometry:** 28.5% (291/1020)
  ❌ **Relative Position:** 35.4% (361/1020)

### Applied Track

- **Accuracy:** 62.23%
- **Correct:** 5713/9180
- **Tasks:** 9

  ❌ **Place Association:** 1.0% (10/1020)
  ✅ **Name Disambiguation:** 96.4% (983/1020)
  ❌ **Relative Position:** 44.1% (450/1020)
  ✅ **Proximity & Nearest Neighbor:** 55.5% (566/1020)
  ✅ **Route Analysis:** 81.4% (830/1020)
  ✅ **Boundary Analysis:** 97.2% (991/1020)
  ✅ **Spatial Patterns:** 83.3% (850/1020)
  ✅ **Missing Data Inference:** 57.4% (585/1020)
  ❌ **Terrain Classification:** 43.9% (448/1020)

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
