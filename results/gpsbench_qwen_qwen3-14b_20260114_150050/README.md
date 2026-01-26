# GPSBench Evaluation Results

**Model:** qwen/qwen3-14b
**Provider:** openrouter
**Timestamp:** 2026-01-14T15:00:50.116252
**Batch Mode:** False

## Overall Results

- **Total Cases:** 18362
- **Total Correct:** 9224
- **Overall Accuracy:** 50.23%

## Track Results

### Pure GPS Track

- **Accuracy:** 41.02%
- **Correct:** 3766/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 83.1% (848/1020)
  ❌ **Coordinate System Transformation:** 6.0% (61/1022)
  ✅ **Distance Calculation:** 54.2% (553/1020)
  ✅ **Bearing Computation:** 57.9% (591/1020)
  ❌ **Coordinate Interpolation:** 1.6% (16/1020)
  ❌ **Polygon Area:** 4.2% (43/1020)
  ✅ **Bounding Box:** 79.2% (808/1020)
  ❌ **Route Geometry:** 32.5% (332/1020)
  ✅ **Relative Position:** 50.4% (514/1020)

### Applied Track

- **Accuracy:** 59.46%
- **Correct:** 5458/9180
- **Tasks:** 9

  ❌ **Place Association:** 2.2% (22/1020)
  ✅ **Name Disambiguation:** 96.2% (981/1020)
  ❌ **Relative Position:** 43.4% (443/1020)
  ✅ **Proximity & Nearest Neighbor:** 56.5% (576/1020)
  ✅ **Route Analysis:** 76.7% (782/1020)
  ✅ **Boundary Analysis:** 88.0% (898/1020)
  ✅ **Spatial Patterns:** 70.7% (721/1020)
  ✅ **Missing Data Inference:** 52.9% (540/1020)
  ❌ **Terrain Classification:** 48.5% (495/1020)

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
