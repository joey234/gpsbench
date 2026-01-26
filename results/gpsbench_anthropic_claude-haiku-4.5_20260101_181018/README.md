# GPSBench Evaluation Results

**Model:** anthropic/claude-haiku-4.5
**Provider:** openrouter
**Timestamp:** 2026-01-01T16:10:25.047192
**Batch Mode:** False

## Overall Results

- **Total Cases:** 18362
- **Total Correct:** 11782
- **Overall Accuracy:** 64.17%

## Track Results

### Pure GPS Track

- **Accuracy:** 58.52%
- **Correct:** 5373/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 87.2% (889/1020)
  ❌ **Coordinate System Transformation:** 20.3% (207/1022)
  ✅ **Distance Calculation:** 90.7% (925/1020)
  ✅ **Bearing Computation:** 86.4% (881/1020)
  ❌ **Coordinate Interpolation:** 2.6% (27/1020)
  ❌ **Polygon Area:** 3.6% (37/1020)
  ✅ **Bounding Box:** 99.3% (1013/1020)
  ❌ **Route Geometry:** 40.4% (412/1020)
  ✅ **Relative Position:** 96.3% (982/1020)

### Applied Track

- **Accuracy:** 69.81%
- **Correct:** 6409/9180
- **Tasks:** 9

  ❌ **Place Association:** 3.6% (37/1020)
  ✅ **Name Disambiguation:** 99.7% (1017/1020)
  ✅ **Relative Position:** 80.6% (822/1020)
  ✅ **Proximity & Nearest Neighbor:** 71.3% (727/1020)
  ✅ **Route Analysis:** 89.7% (915/1020)
  ✅ **Boundary Analysis:** 98.4% (1004/1020)
  ✅ **Spatial Patterns:** 87.4% (891/1020)
  ✅ **Missing Data Inference:** 77.1% (786/1020)
  ❌ **Terrain Classification:** 20.6% (210/1020)

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
