# GPSBench Instruction Templates - Complete Implementation

## Summary

Created proper natural language instruction templates for **all 17 tasks** in GPSBench, replacing the generic JSON dump fallback.

## Changes Made

### Before
Only 3 tasks had proper templates:
- `format_conversion`
- `distance_calculation`
- `place_association`

The remaining 14 tasks used a generic fallback:
```python
instruction = f"Task: {task}\n\nData: {json.dumps(example, indent=2)}\n\nFINAL ANSWER: [your answer]"
output = f"FINAL ANSWER: {json.dumps(example.get('ground_truth', {}))}"
```

### After
**All 17 tasks** now have natural, task-specific instruction templates:

#### Pure GPS Tasks (10)
1. ✅ **format_conversion** - Convert coordinates between formats (DMS, decimal degrees, etc.)
2. ✅ **distance_calculation** - Calculate distance between two points
3. ✅ **bearing_computation** - Calculate initial bearing between points
4. ✅ **interpolation** - Find point N% between two locations
5. ✅ **area_perimeter** - Calculate polygon area and perimeter
6. ✅ **bounding_box** - Find min/max coordinates and center
7. ✅ **coordinate_system_transformation** - WGS84 to UTM conversion
8. ✅ **route_geometry** - Polyline simplification (multiple choice)
9. ✅ **relative_position** - Cardinal direction questions (Yes/No)
10. ✅ **route_analysis** - On-route detection (Yes/No)

#### Applied Tasks (7)
11. ✅ **place_association** - Reverse geocoding (coordinate → location name)
12. ✅ **proximity** - Find nearest point (multiple choice)
13. ✅ **spatial_patterns** - Outlier detection (multiple choice)
14. ✅ **name_disambiguation** - Identify which city with duplicate name
15. ✅ **boundary_analysis** - Regional grouping by continent
16. ✅ **terrain_classification** - Terrain type from coordinates (multiple choice)
17. ✅ **missing_data_handling** - Infer missing longitude/latitude

## Example Templates

### Format Conversion (Pure GPS)
```
Convert the coordinate from DMS to decimal_degrees.

Source coordinate (DMS): 34°43'1.8"S, 55°57'30.2"W

Provide the converted coordinate in decimal_degrees format.

FINAL ANSWER: [converted coordinate]
```

Expected output:
```
FINAL ANSWER: -34.717167, -55.958389
```

### Interpolation (Pure GPS)
```
Find the point that is 25% of the way from Point A to Point B.

Point A: Viacha
Coordinates: -16.65, -68.3

Point B: Jaguaré
Coordinates: -18.90556, -40.07611

Provide the interpolated point's coordinates.

FINAL ANSWER: [latitude], [longitude]
```

Expected output:
```
FINAL ANSWER: -17.595827, -61.321320
```

### Proximity (Applied)
```
Which point is nearest to the reference point?

Reference: Alfenas (-21.42917, -45.94722)

Candidates:
  A) Montgomery Village: 39.17677, -77.19526
  B) Taozhuang: 34.85, 117.33333
  C) Patrātu: 23.66505, 85.30353
  D) Parakou: 7.53333, 1.13333

FINAL ANSWER: [letter]
```

Expected output:
```
FINAL ANSWER: A
```

### Spatial Patterns - Outlier Detection (Applied)
```
Which point is the geographic outlier (furthest from the group)?

  A) Finote Selam: 10.7, 37.26667
  B) Mtwapa: -3.93938, 39.74877
  C) Mendī: 9.8, 35.1
  D) Bronkhorstspruit: -25.81015, 28.74248
  E) Cicero: 43.17562, -76.11937

FINAL ANSWER: [letter]
```

Expected output:
```
FINAL ANSWER: E
```

## Benefits

### 1. **Better Training Quality**
- Natural language instructions are easier for models to learn from
- Clear, consistent format across all tasks
- No confusing JSON dumps in training data

### 2. **Improved Evaluation**
- Evaluators can now properly assess all task types
- Consistent answer format makes parsing easier
- Better alignment between training and evaluation

### 3. **Robustness**
- All templates use `.get()` methods with defaults
- Handles missing or incomplete ground truth gracefully
- No KeyError exceptions during training

## Implementation Details

**File**: `finetuning/tinker/gpsbench_datasets.py`
**Function**: `convert_to_instruction` (lines 208-426)

**Key features**:
- Task-specific templates for all 17 tasks
- Safe dictionary access with `.get()` methods
- Natural language formatting
- Consistent "FINAL ANSWER:" format
- Proper handling of multiple choice tasks (A, B, C, D format)
- Yes/No format for boolean questions

## Testing

Run this to verify all templates work:

```bash
cd finetuning/tinker
python3 << 'EOF'
from gpsbench_datasets import GPSBenchSplitDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

common_config = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer="Qwen/Qwen3-30B-A3B-Instruct-2507",
    renderer_name="qwen_chat",
    max_length=16384,
    batch_size=64,
)

dataset_builder = GPSBenchSplitDataset(
    common_config=common_config,
    gpsbench_root="../../",
    track="both",
)

train_conversations = dataset_builder.load_conversations("train")
print(f"✓ Successfully loaded {len(train_conversations)} training examples")
EOF
```

## Impact on Training

With these templates, your training data now looks like:

**Before (generic fallback)**:
```
Task: bounding_box

Data: {
  "task": "bounding_box",
  "points": [{"lat": ..., "lon": ...}, ...],
  ...
}

FINAL ANSWER: [your answer]
```

**After (natural template)**:
```
Find the bounding box (min/max latitude and longitude) for these points:

  - Ouled Slama Fouaga: 36.54828, 3.11006
  - Farrokh Shahr: 32.27174, 50.98008
  - Skelmersdale: 53.55024, -2.77348
  ...

Provide the min/max coordinates and center point.

FINAL ANSWER: Min lat: [min_lat], Max lat: [max_lat], ...
```

Much more natural and learnable!

## Next Steps

Your fine-tuning pipeline now has:
1. ✅ Stratified evaluation across all 17 tasks
2. ✅ Natural language instruction templates for all tasks
3. ✅ Constant learning rate schedule
4. ✅ Proper eval sample size (170 samples)

Ready to train with:
```bash
cd finetuning/tinker
./run_track_training.sh both
```
