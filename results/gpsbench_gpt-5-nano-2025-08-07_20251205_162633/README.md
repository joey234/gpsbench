# GPSBench Evaluation Results

**Model:** gpt-5-nano-2025-08-07
**Provider:** openai
**Timestamp:** 2025-12-05T16:26:33.871543
**Batch Mode:** True

## Overall Results

- **Total Cases:** 18364
- **Total Correct:** 9053
- **Overall Accuracy:** 49.30%

## Track Results

### Pure GPS Track

- **Accuracy:** 42.18%
- **Correct:** 3873/9182
- **Tasks:** 9

  ✅ **Format Conversion:** 90.0% (918/1020)
  ❌ **Coordinate System Transformation:** 7.5% (77/1022)
  ❌ **Distance Calculation:** 48.0% (490/1020)
  ✅ **Bearing Computation:** 54.5% (556/1020)
  ❌ **Coordinate Interpolation:** 0.3% (3/1020)
  ❌ **Polygon Area:** 0.6% (6/1020)
  ✅ **Bounding Box:** 97.5% (994/1020)
  ❌ **Route Geometry:** 26.4% (269/1020)
  ✅ **Relative Position:** 54.9% (560/1020)

### Applied Track

- **Accuracy:** 56.41%
- **Correct:** 5180/9182
- **Tasks:** 9

  ❌ **Place Association:** 2.0% (20/1022)
  ✅ **Name Disambiguation:** 100.0% (1020/1020)
  ✅ **Relative Position:** 51.9% (529/1020)
  ❌ **Proximity & Nearest Neighbor:** 33.1% (338/1020)
  ✅ **Route Analysis:** 84.8% (865/1020)
  ✅ **Boundary Analysis:** 97.0% (989/1020)
  ✅ **Spatial Patterns:** 59.4% (606/1020)
  ✅ **Missing Data Inference:** 66.2% (675/1020)
  ❌ **Terrain Classification:** 13.5% (138/1020)

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
