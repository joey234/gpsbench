# Multi-Task Fine-tuning Design

## Problem with Current Approach

The current implementation only teaches **reverse geocoding** (coordinate → city name). This is too narrow and won't significantly improve **general geospatial capabilities**.

## Proposed Multi-Task Approach

To actually improve LLM geospatial reasoning, we need to fine-tune on diverse tasks that cover:

### 1. **Coordinate Understanding** (Pure GPS Knowledge)
Tasks that teach coordinate systems and geometric operations:

- ✅ **Format Conversion** (DMS ↔ Decimal)
  - Input: "Convert 40°42'46"N, 74°0'22"W to decimal degrees"
  - Output: "40.712778, -74.006111"
  - **Teaches:** Coordinate format understanding

- ✅ **Distance Calculation** (Haversine)
  - Input: "Calculate distance from (40.7128, -74.0060) to (51.5074, -0.1278)"
  - Output: "5,570 km"
  - **Teaches:** Spherical geometry

- ✅ **Bearing Computation**
  - Input: "Calculate bearing from NYC (40.7128, -74.0060) to LA (34.0522, -118.2437)"
  - Output: "273.4 degrees (West)"
  - **Teaches:** Directional reasoning

- ✅ **Interpolation**
  - Input: "Find midpoint between (40.7128, -74.0060) and (51.5074, -0.1278)"
  - Output: "(51.3, -31.4)"
  - **Teaches:** Great circle paths

- ✅ **Area Calculation**
  - Input: "Calculate area of polygon with vertices [(40.7128, -74.0060), ...]"
  - Output: "15.2 km²"
  - **Teaches:** Polygon geometry on sphere

### 2. **Location Grounding** (Geographic Knowledge)
Tasks that connect coordinates to real-world locations:

- ✅ **Reverse Geocoding** (Coordinate → Location) [Current task]
  - Input: "What location is at (40.7128, -74.0060)?"
  - Output: "New York, US"
  - **Teaches:** Coordinate-to-place mapping

- ✅ **Forward Geocoding** (Location → Coordinate) [NEW]
  - Input: "What are the coordinates of Paris, France?"
  - Output: "48.8566, 2.3522"
  - **Teaches:** Place-to-coordinate mapping

- ✅ **Regional Knowledge**
  - Input: "Which country is (40.7128, -74.0060) in?"
  - Output: "United States"
  - **Teaches:** Political boundaries

- ✅ **Continent Identification**
  - Input: "What continent is (35.6762, 139.6503) on?"
  - Output: "Asia"
  - **Teaches:** Continental boundaries

### 3. **Spatial Reasoning**
Tasks that teach spatial relationships:

- ✅ **Relative Position**
  - Input: "Is NYC (40.7128, -74.0060) north of LA (34.0522, -118.2437)?"
  - Output: "Yes"
  - **Teaches:** Spatial comparison

- ✅ **Proximity Ranking**
  - Input: "Which is closest to Paris (48.8566, 2.3522): London, Berlin, or Rome?"
  - Output: "London (343 km)"
  - **Teaches:** Distance comparison

- ✅ **Route Analysis**
  - Input: "Is Chicago on the path from NYC to LA?"
  - Output: "Yes (detour is only 6%)"
  - **Teaches:** Path understanding

### 4. **Robustness Training**
Add Gaussian noise to all tasks to improve robustness:

- Noise levels: σ = 0m, 10m, 50m, 100m, 500m
- Applied to all coordinate-based tasks
- Tests model's ability to handle GPS uncertainty

## Dataset Composition

### Task Distribution (Balanced Multi-Task)

| Task Type | Subtasks | Samples | Purpose |
|-----------|----------|---------|---------|
| **Coordinate Operations** | Format, Distance, Bearing, Interpolation, Area | ~8,000 | Pure GPS math |
| **Location Grounding** | Reverse/Forward geocoding, Regional | ~8,000 | Geographic knowledge |
| **Spatial Reasoning** | Position, Proximity, Routes | ~4,000 | Spatial relationships |
| **TOTAL** | 10 distinct task types | **~20,000** | Comprehensive geospatial |

### Noise Distribution
- **Clean** (no noise): 40% of samples
- **Low noise** (σ=10-50m): 30% of samples
- **Medium noise** (σ=100-200m): 20% of samples
- **High noise** (σ=500m): 10% of samples

## Implementation Plan

### Phase 1: Extend Dataset Generation ✅
- [x] Current: Reverse geocoding with noise
- [ ] Add: 9 additional task types
- [ ] Use GPSBench data + GeoNames cities

### Phase 2: Multi-Task Training
- [ ] Update training script for task diversity
- [ ] Add task type embeddings (optional)
- [ ] Balance sampling across task types

### Phase 3: Evaluation
- [ ] Evaluate on each task type separately
- [ ] Compare with GPSBench baseline
- [ ] Measure transfer learning effects

## Expected Benefits

### 1. **Improved Coordinate Understanding**
- Can convert between formats
- Can calculate distances and bearings
- Understands spherical geometry

### 2. **Better Geographic Knowledge**
- Knows major cities and landmarks
- Understands country/continent boundaries
- Can map locations ↔ coordinates

### 3. **Enhanced Spatial Reasoning**
- Can compare positions (north/south/east/west)
- Can rank by proximity
- Understands routes and paths

### 4. **Robustness to Noise**
- Handles GPS uncertainty
- Stable predictions under coordinate perturbations

## Why This Works Better

| Aspect | Single-Task (Current) | Multi-Task (Proposed) |
|--------|----------------------|----------------------|
| **Tasks covered** | 1 (reverse geocoding) | 10 (diverse geospatial) |
| **Skills learned** | Coordinate→location lookup | Geometric reasoning + knowledge |
| **Transfer ability** | Low | High |
| **Robustness** | Limited to one task | Generalizes across tasks |
| **Real-world utility** | Narrow | Comprehensive |

## Comparison with Prompting

This fine-tuning baseline will let us answer:

1. **Does fine-tuning beat prompting?**
   - Compare fine-tuned model vs GPT-4/Claude on GPSBench

2. **What tasks benefit from fine-tuning?**
   - Which tasks improve more: math (distance) or knowledge (geocoding)?

3. **Is smaller fine-tuned > larger prompted?**
   - Can Qwen-0.6B fine-tuned beat GPT-4 prompted?

4. **Transfer learning effects?**
   - Does training on task A improve task B?

## Data Sources

### From GPSBench
- Layer A: Format conversion examples
- Layer B: Location names, regional knowledge
- Layer C: Distance, bearing, interpolation, area calculations
- Layer D: Spatial relationships, proximity
- Layer E: Robustness testing

### From GeoNames (cities15000.txt)
- 23,000+ cities with coordinates
- Country codes
- Population data
- Use for geocoding tasks

### Generated Synthetically
- Distance pairs (random city pairs)
- Bearing calculations
- Interpolation (midpoints, waypoints)
- Relative position (north/south comparisons)

## Next Steps

1. ✅ Design multi-task framework
2. ⏳ Implement multi-task dataset generator
3. ⏳ Update training script
4. ⏳ Create task-specific evaluation
5. ⏳ Run experiments and compare with prompting baselines
