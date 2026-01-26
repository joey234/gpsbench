# GPSBench Evaluation Results

**Model:** gpt-5.1
**Provider:** openai
**Timestamp:** 2025-12-02T11:46:55.894350
**Batch Mode:** True

## Overall Results

- **Total Cases:** 18362
- **Total Correct:** 14365
- **Overall Accuracy:** 78.23%

## Track Results

### Pure GPS Track

- **Accuracy:** 75.90%
- **Correct:** 6969/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 85.7% (874/1020)
  ✅ **Coordinate System Transformation:** 56.4% (576/1022)
  ✅ **Distance Calculation:** 97.0% (989/1020)
  ✅ **Bearing Computation:** 99.1% (1011/1020)
  ✅ **Coordinate Interpolation:** 54.1% (552/1020)
  ❌ **Polygon Area:** 2.3% (23/1020)
  ✅ **Bounding Box:** 97.5% (994/1020)
  ✅ **Route Geometry:** 91.8% (936/1020)
  ✅ **Relative Position:** 99.4% (1014/1020)

### Applied Track

- **Accuracy:** 80.57%
- **Correct:** 7396/9180
- **Tasks:** 9

  ✅ **Place Association:** 56.0% (571/1020)
  ✅ **Name Disambiguation:** 100.0% (1020/1020)
  ✅ **Relative Position:** 99.5% (1015/1020)
  ✅ **Proximity & Nearest Neighbor:** 89.6% (914/1020)
  ✅ **Route Analysis:** 90.1% (919/1020)
  ✅ **Boundary Analysis:** 87.5% (893/1020)
  ✅ **Spatial Patterns:** 95.2% (971/1020)
  ✅ **Missing Data Inference:** 88.4% (902/1020)
  ❌ **Terrain Classification:** 18.7% (191/1020)

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
