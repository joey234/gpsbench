#!/usr/bin/env python3
"""
Add reasoning steps to GPSBench split files (train/dev/test).

This script adds step-by-step reasoning to training data to help models
learn to solve GPS tasks through chain-of-thought reasoning.

Usage:
    python scripts/add_reasoning_to_splits.py
    python scripts/add_reasoning_to_splits.py --split train
    python scripts/add_reasoning_to_splits.py --split all --track pure_gps
"""

import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_json(filepath: Path) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(filepath: Path, data: List[Dict]):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# MATHEMATICAL HELPERS
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance in km."""
    R = 6371  # Earth radius in km
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


# =============================================================================
# PURE GPS TRACK - REASONING GENERATORS
# =============================================================================

def add_format_conversion_reasoning(example: Dict) -> Dict:
    """Add reasoning for format conversion (DMS <-> decimal degrees)."""
    source_format = example.get('source_format', '')
    target_format = example.get('target_format', '')
    source = example.get('source_string', '')
    gt = example.get('ground_truth', {})

    if source_format == 'decimal_degrees' and target_format == 'DMS':
        lat = gt.get('latitude', 0)
        lon = gt.get('longitude', 0)

        # Calculate DMS components
        lat_deg = int(abs(lat))
        lat_min = int((abs(lat) - lat_deg) * 60)
        lat_sec = ((abs(lat) - lat_deg) * 60 - lat_min) * 60
        lat_dir = 'N' if lat >= 0 else 'S'

        lon_deg = int(abs(lon))
        lon_min = int((abs(lon) - lon_deg) * 60)
        lon_sec = ((abs(lon) - lon_deg) * 60 - lon_min) * 60
        lon_dir = 'E' if lon >= 0 else 'W'

        steps = [
            f"Step 1: Parse decimal degrees",
            f"  Input: {source}",
            f"  Latitude: {lat}°",
            f"  Longitude: {lon}°",
            f"Step 2: Convert latitude to DMS",
            f"  Degrees = {lat_deg}",
            f"  Minutes = int(({abs(lat):.6f} - {lat_deg}) × 60) = {lat_min}",
            f"  Seconds = (({abs(lat):.6f} - {lat_deg}) × 60 - {lat_min}) × 60 = {lat_sec:.1f}",
            f"  Direction = {lat_dir} (positive = N, negative = S)",
            f"Step 3: Convert longitude to DMS",
            f"  Degrees = {lon_deg}",
            f"  Minutes = {lon_min}",
            f"  Seconds = {lon_sec:.1f}",
            f"  Direction = {lon_dir} (positive = E, negative = W)",
            f"Step 4: Format result",
            f"  {gt.get('target_string', '')}",
            f"FINAL ANSWER: {gt.get('target_string', '')}"
        ]
    elif source_format == 'DMS' and target_format == 'decimal_degrees':
        lat = gt.get('latitude', 0)
        lon = gt.get('longitude', 0)

        steps = [
            f"Step 1: Parse DMS string",
            f"  Input: {source}",
            f"Step 2: Convert to decimal degrees using formula",
            f"  decimal = degrees + minutes/60 + seconds/3600",
            f"  Apply negative sign for S (latitude) or W (longitude)",
            f"Step 3: Calculate latitude",
            f"  Latitude: {lat}°",
            f"Step 4: Calculate longitude",
            f"  Longitude: {lon}°",
            f"FINAL ANSWER: {gt.get('target_string', f'{lat}, {lon}')}"
        ]
    else:
        steps = [
            f"Step 1: Parse source format ({source_format})",
            f"  Input: {source}",
            f"Step 2: Apply conversion formula for {target_format}",
            f"Step 3: Result: {gt.get('target_string', '')}",
            f"FINAL ANSWER: {gt.get('target_string', '')}"
        ]

    example['reasoning_steps'] = steps
    return example


def add_coordinate_transform_reasoning(example: Dict) -> Dict:
    """Add reasoning for coordinate system transformation (WGS84 -> UTM/Web Mercator)."""
    source = example.get('source_coords', example.get('coordinate', {}))
    gt = example.get('ground_truth', {})
    subtask = example.get('subtask', '')

    lat = source.get('lat', source.get('latitude', 0))
    lon = source.get('lon', source.get('longitude', 0))

    if 'web_mercator' in subtask.lower() or 'mercator' in subtask.lower() or gt.get('x') is not None:
        # Web Mercator transformation
        x = gt.get('x', 0)
        y = gt.get('y', 0)

        steps = [
            f"Step 1: Identify source coordinates (WGS84 / EPSG:4326)",
            f"  Latitude: {lat}°",
            f"  Longitude: {lon}°",
            f"Step 2: Apply Web Mercator projection (EPSG:3857)",
            f"  X = longitude × 20037508.34 / 180",
            f"  X = {lon} × 20037508.34 / 180 = {x:.2f} m",
            f"  Y = ln(tan((90 + latitude) × π / 360)) × 20037508.34 / π",
            f"  Y = {y:.2f} m",
            f"Step 3: Result in Web Mercator coordinates",
            f"  X: {x:.2f} m",
            f"  Y: {y:.2f} m",
            f"FINAL ANSWER: X: {x:.2f} m, Y: {y:.2f} m"
        ]
    else:
        # UTM transformation
        zone = gt.get('zone', int((lon + 180) / 6) + 1)
        hemisphere = gt.get('hemisphere', 'N' if lat >= 0 else 'S')
        easting = gt.get('easting', 0)
        northing = gt.get('northing', 0)

        steps = [
            f"Step 1: Identify source coordinates (WGS84)",
            f"  Latitude: {lat}°",
            f"  Longitude: {lon}°",
            f"Step 2: Determine UTM zone",
            f"  Zone = floor((longitude + 180) / 6) + 1",
            f"  Zone = floor(({lon} + 180) / 6) + 1 = {zone}",
            f"  Hemisphere: {hemisphere} ({'Northern' if hemisphere == 'N' else 'Southern'})",
            f"Step 3: Apply Transverse Mercator projection",
            f"  Central meridian = (zone - 1) × 6 - 180 + 3 = {(zone - 1) * 6 - 180 + 3}°",
            f"Step 4: Calculate UTM coordinates",
            f"  Easting: {easting:.2f} m",
            f"  Northing: {northing:.2f} m",
            f"FINAL ANSWER: Easting: {easting:.2f} m, Northing: {northing:.2f} m, Zone: {zone}{hemisphere}"
        ]

    example['reasoning_steps'] = steps
    return example


def add_distance_reasoning(example: Dict) -> Dict:
    """Add reasoning for distance calculation using Haversine formula."""
    p_a = example.get('point_a', {})
    p_b = example.get('point_b', {})
    gt = example.get('ground_truth', {})

    lat1, lon1 = p_a.get('lat', 0), p_a.get('lon', 0)
    lat2, lon2 = p_b.get('lat', 0), p_b.get('lon', 0)
    distance_km = gt.get('distance_km', 0)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    steps = [
        f"Step 1: Identify the coordinates",
        f"  Point A: {p_a.get('name', 'A')} at ({lat1}°, {lon1}°)",
        f"  Point B: {p_b.get('name', 'B')} at ({lat2}°, {lon2}°)",
        f"Step 2: Calculate differences",
        f"  Δlat = {lat2} - {lat1} = {dlat:.4f}°",
        f"  Δlon = {lon2} - {lon1} = {dlon:.4f}°",
        f"Step 3: Apply Haversine formula",
        f"  a = sin²(Δlat/2) + cos(lat₁) × cos(lat₂) × sin²(Δlon/2)",
        f"  c = 2 × arcsin(√a)",
        f"  d = R × c, where R = 6371 km (Earth radius)",
        f"Step 4: Calculate distance",
        f"  Distance = {distance_km:.2f} km",
    ]

    if 'distance_miles' in gt:
        steps.append(f"  In miles: {gt['distance_miles']:.2f} miles")

    steps.append(f"FINAL ANSWER: {distance_km:.2f} km")

    example['reasoning_steps'] = steps
    return example


def add_bearing_reasoning(example: Dict) -> Dict:
    """Add reasoning for bearing/azimuth calculation."""
    p_a = example.get('point_a', {})
    p_b = example.get('point_b', {})
    gt = example.get('ground_truth', {})

    lat1, lon1 = p_a.get('lat', 0), p_a.get('lon', 0)
    lat2, lon2 = p_b.get('lat', 0), p_b.get('lon', 0)
    bearing = gt.get('initial_bearing_deg', gt.get('bearing_deg', 0))
    cardinal = gt.get('cardinal_direction', '')

    dlon = lon2 - lon1

    steps = [
        f"Step 1: Identify the coordinates",
        f"  From: {p_a.get('name', 'A')} at ({lat1}°, {lon1}°)",
        f"  To: {p_b.get('name', 'B')} at ({lat2}°, {lon2}°)",
        f"Step 2: Calculate longitude difference",
        f"  Δlon = {lon2} - {lon1} = {dlon:.4f}°",
        f"Step 3: Apply bearing formula",
        f"  θ = atan2(sin(Δlon) × cos(lat₂), cos(lat₁) × sin(lat₂) - sin(lat₁) × cos(lat₂) × cos(Δlon))",
        f"Step 4: Convert to degrees and normalize to 0-360°",
        f"  Initial bearing = {bearing:.2f}°",
    ]

    if cardinal:
        steps.append(f"Step 5: Determine cardinal direction")
        steps.append(f"  {bearing:.1f}° → {cardinal}")
        steps.append(f"FINAL ANSWER: {bearing:.2f}° ({cardinal})")
    else:
        steps.append(f"FINAL ANSWER: {bearing:.2f}°")

    example['reasoning_steps'] = steps
    return example


def add_interpolation_reasoning(example: Dict) -> Dict:
    """Add reasoning for great circle interpolation."""
    p_a = example.get('point_a', {})
    p_b = example.get('point_b', {})
    fraction = example.get('fraction', 0.5)
    gt = example.get('ground_truth', {})
    result = gt.get('interpolated_point', {})

    lat1, lon1 = p_a.get('lat', 0), p_a.get('lon', 0)
    lat2, lon2 = p_b.get('lat', 0), p_b.get('lon', 0)
    result_lat = result.get('lat', 0)
    result_lon = result.get('lon', 0)

    steps = [
        f"Step 1: Identify the endpoints",
        f"  Point A: ({lat1}°, {lon1}°)",
        f"  Point B: ({lat2}°, {lon2}°)",
        f"  Fraction f = {fraction}",
        f"Step 2: Convert to radians and 3D Cartesian",
        f"  Convert lat/lon to unit sphere coordinates (x, y, z)",
        f"Step 3: Calculate angular distance d between points",
        f"  d = arccos(sin(lat₁)sin(lat₂) + cos(lat₁)cos(lat₂)cos(Δlon))",
        f"Step 4: Apply spherical linear interpolation (slerp)",
        f"  a = sin((1-f)×d) / sin(d)",
        f"  b = sin(f×d) / sin(d)",
        f"  P = a×A + b×B",
        f"Step 5: Convert back to lat/lon",
        f"  Latitude: {result_lat:.6f}°",
        f"  Longitude: {result_lon:.6f}°",
        f"FINAL ANSWER: {result_lat:.6f}, {result_lon:.6f}"
    ]

    example['reasoning_steps'] = steps
    return example


def add_area_perimeter_reasoning(example: Dict) -> Dict:
    """Add reasoning for polygon area and perimeter calculation."""
    vertices = example.get('vertices', example.get('polygon', {}).get('vertices', []))
    gt = example.get('ground_truth', {})

    n = len(vertices)
    area = gt.get('area_km2', 0)
    perimeter = gt.get('perimeter_km', 0)

    steps = [
        f"Step 1: Identify polygon vertices ({n} vertices)",
    ]

    for i, v in enumerate(vertices):
        name = v.get('name', f"V{i+1}")
        steps.append(f"  {name}: ({v.get('lat', 0):.4f}°, {v.get('lon', 0):.4f}°)")

    steps.extend([
        f"Step 2: Calculate perimeter",
        f"  Sum distances between consecutive vertices using Haversine",
        f"  Perimeter = {perimeter:.2f} km",
        f"Step 3: Calculate area using spherical excess formula",
        f"  Area on sphere = R² × E, where E is spherical excess",
        f"  Area = {area:.2f} km²",
        f"FINAL ANSWER: Area: {area:.2f} km², Perimeter: {perimeter:.2f} km"
    ])

    example['reasoning_steps'] = steps
    return example


def add_bounding_box_reasoning(example: Dict) -> Dict:
    """Add reasoning for bounding box calculation."""
    points = example.get('points', [])
    gt = example.get('ground_truth', {})

    min_lat = gt.get('min_lat', 0)
    max_lat = gt.get('max_lat', 0)
    min_lon = gt.get('min_lon', 0)
    max_lon = gt.get('max_lon', 0)

    steps = [
        f"Step 1: Extract coordinates from {len(points)} points",
    ]

    # Show first few points
    for i, p in enumerate(points[:3]):
        steps.append(f"  {p.get('name', f'P{i+1}')}: ({p.get('lat', 0):.4f}°, {p.get('lon', 0):.4f}°)")
    if len(points) > 3:
        steps.append(f"  ... ({len(points) - 3} more points)")

    steps.extend([
        f"Step 2: Find min/max latitude",
        f"  Min latitude: {min_lat}°",
        f"  Max latitude: {max_lat}°",
        f"Step 3: Find min/max longitude",
        f"  Min longitude: {min_lon}°",
        f"  Max longitude: {max_lon}°",
        f"Step 4: Construct bounding box",
        f"  SW corner: ({min_lat}, {min_lon})",
        f"  NE corner: ({max_lat}, {max_lon})",
        f"FINAL ANSWER: {min_lat:.6f}, {max_lat:.6f}, {min_lon:.6f}, {max_lon:.6f}"
    ])

    example['reasoning_steps'] = steps
    return example


def add_route_geometry_reasoning(example: Dict) -> Dict:
    """Add reasoning for route geometry (Douglas-Peucker simplification)."""
    points = example.get('points', [])
    tolerance = example.get('tolerance_km', example.get('epsilon_km', 1.0))
    gt = example.get('ground_truth', {})
    answer = gt.get('answer', '')

    steps = [
        f"Step 1: Input route with {len(points)} points",
        f"  Tolerance: {tolerance} km",
        f"Step 2: Apply Douglas-Peucker algorithm",
        f"  1. Keep first and last points",
        f"  2. Find point with max perpendicular distance to line",
        f"  3. If max_distance > tolerance, keep that point and recurse",
        f"  4. Otherwise, discard intermediate points",
        f"Step 3: Identify which points to keep",
    ]

    if 'points_kept' in gt:
        kept = gt['points_kept']
        steps.append(f"  Points kept: {kept}")
        steps.append(f"  Reduction: {len(points)} → {len(kept)} points")

    steps.append(f"FINAL ANSWER: {answer}")

    example['reasoning_steps'] = steps
    return example


def add_relative_position_pure_gps_reasoning(example: Dict) -> Dict:
    """Add reasoning for relative position (Pure GPS - coordinate comparison)."""
    p_a = example.get('point_a', {})
    p_b = example.get('point_b', {})
    gt = example.get('ground_truth', {})
    question = example.get('question', '')
    answer = gt.get('answer', False)

    lat_a, lon_a = p_a.get('lat', 0), p_a.get('lon', 0)
    lat_b, lon_b = p_b.get('lat', 0), p_b.get('lon', 0)

    # Detect direction from question
    direction = None
    for d in ['north', 'south', 'east', 'west']:
        if d in question.lower():
            direction = d
            break

    steps = [
        f"Step 1: Identify the coordinates",
        f"  Point A: ({lat_a}°, {lon_a}°)",
        f"  Point B: ({lat_b}°, {lon_b}°)",
    ]

    if direction in ['north', 'south']:
        steps.append(f"Step 2: Compare latitudes (north/south)")
        steps.append(f"  Point A latitude: {lat_a}°")
        steps.append(f"  Point B latitude: {lat_b}°")
        if direction == 'north':
            is_true = lat_a > lat_b
            steps.append(f"  Is A north of B? {lat_a}° > {lat_b}° → {is_true}")
        else:
            is_true = lat_a < lat_b
            steps.append(f"  Is A south of B? {lat_a}° < {lat_b}° → {is_true}")
    else:
        steps.append(f"Step 2: Compare longitudes (east/west)")
        steps.append(f"  Point A longitude: {lon_a}°")
        steps.append(f"  Point B longitude: {lon_b}°")
        if direction == 'east':
            is_true = lon_a > lon_b
            steps.append(f"  Is A east of B? {lon_a}° > {lon_b}° → {is_true}")
        else:
            is_true = lon_a < lon_b
            steps.append(f"  Is A west of B? {lon_a}° < {lon_b}° → {is_true}")

    answer_str = 'Yes' if answer else 'No'
    steps.append(f"FINAL ANSWER: {answer_str}")

    example['reasoning_steps'] = steps
    return example


# =============================================================================
# APPLIED TRACK - REASONING GENERATORS
# =============================================================================

def add_place_association_reasoning(example: Dict) -> Dict:
    """Add reasoning for place association (reverse geocoding)."""
    coord = example.get('coordinate_string', '')
    gt = example.get('ground_truth', {})
    location = gt.get('location_name', gt.get('name', 'Unknown'))

    steps = [
        f"Step 1: Parse coordinates",
        f"  Input: {coord}",
        f"Step 2: Search geographic database",
        f"  Query: Find location nearest to ({coord})",
        f"Step 3: Match to known location",
        f"  Nearest city/place: {location}",
    ]

    if 'country' in gt:
        steps.append(f"  Country: {gt['country']}")
    if 'region' in gt:
        steps.append(f"  Region: {gt['region']}")

    steps.append(f"FINAL ANSWER: {location}")

    example['reasoning_steps'] = steps
    return example


def add_name_disambiguation_reasoning(example: Dict) -> Dict:
    """Add reasoning for name disambiguation (multiple cities with same name)."""
    question = example.get('question', '')
    candidates = example.get('candidates', [])
    gt = example.get('ground_truth', {})
    coord_str = example.get('coordinate_string', '')

    # Get the correct answer
    correct_name = gt.get('name', 'Unknown')
    correct_country = gt.get('country', '')

    steps = [
        f"Step 1: Parse the question",
        f"  {question[:100]}...",
        f"  Target coordinates: {coord_str}",
        f"Step 2: List candidate locations with same name",
    ]

    # List candidates with their coordinates
    for i, cand in enumerate(candidates):
        country = cand.get('country', cand.get('country_code', 'Unknown'))
        lat = cand.get('lat', cand.get('latitude', 0))
        lon = cand.get('lon', cand.get('longitude', 0))
        steps.append(f"  {i+1}. {correct_name}, {country} at ({lat:.4f}, {lon:.4f})")

    steps.extend([
        f"Step 3: Compare target coordinates to each candidate",
        f"  Target: {coord_str}",
        f"  Find closest match",
        f"Step 4: Identify correct location",
        f"  Match: {correct_name}, {correct_country}",
        f"FINAL ANSWER: {correct_name}, {correct_country}"
    ])

    example['reasoning_steps'] = steps
    return example


def add_relative_position_applied_reasoning(example: Dict) -> Dict:
    """Add reasoning for relative position (Applied - named cities)."""
    question = example.get('question', '')
    gt = example.get('ground_truth', {})
    answer_raw = gt.get('answer', '')

    # Convert True/False to Yes/No for consistency
    if answer_raw is True or answer_raw == 'True':
        answer = 'Yes'
    elif answer_raw is False or answer_raw == 'False':
        answer = 'No'
    else:
        answer = str(answer_raw)

    steps = [
        f"Step 1: Parse the question",
        f"  {question}",
        f"Step 2: Look up coordinates of mentioned cities",
        f"Step 3: Compare coordinates",
    ]

    # Detect direction
    for d in ['north', 'south', 'east', 'west']:
        if d in question.lower():
            if d in ['north', 'south']:
                steps.append(f"  Compare latitudes for {d}")
            else:
                steps.append(f"  Compare longitudes for {d}")
            break

    steps.extend([
        f"Step 4: Determine answer",
        f"FINAL ANSWER: {answer}"
    ])

    example['reasoning_steps'] = steps
    return example


def add_proximity_reasoning(example: Dict) -> Dict:
    """Add reasoning for proximity/nearest neighbor."""
    question = example.get('question', '')
    gt = example.get('ground_truth', {})
    answer = gt.get('answer', '')
    distance = gt.get('distance_km', 0)

    steps = [
        f"Step 1: Identify reference point and candidates",
        f"  {question[:80]}...",
        f"Step 2: Calculate distance to each candidate",
        f"  Using Haversine formula for each",
    ]

    if 'all_distances_km' in gt:
        for name, dist in gt['all_distances_km'].items():
            steps.append(f"  {name}: {dist:.1f} km")

    steps.extend([
        f"Step 3: Find minimum distance",
        f"  Closest: {answer} at {distance:.1f} km",
        f"FINAL ANSWER: {answer}"
    ])

    example['reasoning_steps'] = steps
    return example


def add_route_analysis_reasoning(example: Dict) -> Dict:
    """Add reasoning for route analysis (is waypoint on path)."""
    start = example.get('start', {})
    end = example.get('end', {})
    waypoint = example.get('waypoint', {})
    gt = example.get('ground_truth', {})

    direct = gt.get('direct_distance_km', 0)
    via = gt.get('via_waypoint_km', 0)
    detour = gt.get('detour_km', 0)
    detour_pct = gt.get('detour_percent', 0)
    answer = gt.get('answer', False)

    steps = [
        f"Step 1: Identify the three points",
        f"  Start: {start.get('name', 'Start')}",
        f"  End: {end.get('name', 'End')}",
        f"  Waypoint: {waypoint.get('name', 'Waypoint')}",
        f"Step 2: Calculate direct distance",
        f"  Start → End: {direct:.1f} km",
        f"Step 3: Calculate distance via waypoint",
        f"  Start → Waypoint → End: {via:.1f} km",
        f"Step 4: Calculate detour",
        f"  Detour = {via:.1f} - {direct:.1f} = {detour:.1f} km",
        f"  Detour % = ({detour:.1f} / {direct:.1f}) × 100 = {detour_pct:.1f}%",
        f"Step 5: Determine if on route (typically < 10% detour)",
        f"  {detour_pct:.1f}% {'<' if answer else '>'} 10%",
        f"FINAL ANSWER: {'Yes' if answer else 'No'}"
    ]

    example['reasoning_steps'] = steps
    return example


def add_boundary_analysis_reasoning(example: Dict) -> Dict:
    """Add reasoning for boundary/continent analysis."""
    question = example.get('question', '')
    gt = example.get('ground_truth', {})
    answer = gt.get('answer', gt.get('num_continents', ''))

    steps = [
        f"Step 1: Parse the question",
        f"  {question[:100]}...",
        f"Step 2: Identify locations and their continents",
    ]

    if 'locations' in example:
        for loc in example['locations'][:5]:
            name = loc.get('name', 'Unknown')
            continent = loc.get('continent', 'Unknown')
            steps.append(f"  {name}: {continent}")

    steps.extend([
        f"Step 3: Count unique continents or analyze boundaries",
        f"Step 4: Determine answer",
        f"FINAL ANSWER: {answer}"
    ])

    example['reasoning_steps'] = steps
    return example


def add_spatial_patterns_reasoning(example: Dict) -> Dict:
    """Add reasoning for spatial patterns (outlier detection)."""
    points = example.get('points', [])
    gt = example.get('ground_truth', {})
    answer = gt.get('answer', '')

    steps = [
        f"Step 1: Identify all points ({len(points)} locations)",
    ]

    for p in points[:5]:
        steps.append(f"  {p.get('name', 'Unknown')}: ({p.get('lat', 0):.2f}°, {p.get('lon', 0):.2f}°)")

    if len(points) > 5:
        steps.append(f"  ... and {len(points) - 5} more")

    steps.extend([
        f"Step 2: Calculate centroid of all points",
        f"Step 3: Calculate distance from each point to centroid",
    ])

    if 'distances_to_centroid' in gt:
        for name, dist in sorted(gt['distances_to_centroid'].items(), key=lambda x: -x[1])[:3]:
            steps.append(f"  {name}: {dist:.1f} km from centroid")

    steps.extend([
        f"Step 4: Identify geographic outlier (furthest from centroid)",
        f"  Outlier: {answer}",
        f"FINAL ANSWER: {answer}"
    ])

    example['reasoning_steps'] = steps
    return example


def add_missing_data_reasoning(example: Dict) -> Dict:
    """Add reasoning for missing data inference."""
    question = example.get('question', '')
    gt = example.get('ground_truth', {})
    location_name = example.get('location_name', 'Unknown')
    partial = example.get('partial_coordinate', {})

    # Determine what's known and what's missing
    known_lat = partial.get('latitude')
    known_lon = partial.get('longitude')

    steps = [
        f"Step 1: Parse the question",
        f"  {question[:100]}...",
        f"Step 2: Identify known information",
        f"  Location name: {location_name}",
    ]

    if known_lat is not None:
        steps.append(f"  Known latitude: {known_lat}°")
        steps.append(f"  Missing: longitude")
        answer = gt.get('longitude', gt.get('complete_coordinate', {}).get('longitude', 'Unknown'))
    elif known_lon is not None:
        steps.append(f"  Known longitude: {known_lon}°")
        steps.append(f"  Missing: latitude")
        answer = gt.get('latitude', gt.get('complete_coordinate', {}).get('latitude', 'Unknown'))
    else:
        answer = gt.get('answer', 'Unknown')

    complete = gt.get('complete_coordinate', {})
    if complete:
        steps.append(f"Step 3: Search geographic database for '{location_name}'")
        steps.append(f"  Found: ({complete.get('latitude', 'N/A')}, {complete.get('longitude', 'N/A')})")
    else:
        steps.append(f"Step 3: Search geographic database")

    steps.append(f"Step 4: Retrieve missing coordinate")
    steps.append(f"FINAL ANSWER: {answer}")

    example['reasoning_steps'] = steps
    return example


def add_terrain_classification_reasoning(example: Dict) -> Dict:
    """Add reasoning for terrain classification."""
    question = example.get('question', '')
    candidates = example.get('candidates', [])
    coord_str = example.get('coordinate_string', '')
    gt = example.get('ground_truth', {})
    answer = gt.get('answer', '')
    terrain_type = gt.get('terrain_type', '')
    location = gt.get('specific_location', '')

    steps = [
        f"Step 1: Parse the coordinates",
        f"  Coordinates: {coord_str}",
        f"  {question[:80]}...",
        f"Step 2: List terrain options",
    ]

    # Use candidates if available
    if candidates:
        for cand in candidates:
            # Candidates already have letter prefixes like "A) Forest"
            steps.append(f"  {cand}")
    else:
        steps.append(f"  (Options listed in question)")

    steps.extend([
        f"Step 3: Analyze geographic characteristics",
        f"  Location: {location}" if location else f"  Analyze elevation, water proximity, climate",
        f"Step 4: Classify terrain type",
        f"  Terrain: {terrain_type}",
        f"FINAL ANSWER: {answer}"
    ])

    example['reasoning_steps'] = steps
    return example


# =============================================================================
# TASK MAPPING
# =============================================================================

# Mapping: (track, task_name) -> reasoning_function
TASK_PROCESSORS = {
    # Pure GPS Track
    ('pure_gps', 'task1_format_conversion'): add_format_conversion_reasoning,
    ('pure_gps', 'task3_coordinate_system_transformation'): add_coordinate_transform_reasoning,
    ('pure_gps', 'task4_distance_calculation'): add_distance_reasoning,
    ('pure_gps', 'task5_bearing_computation'): add_bearing_reasoning,
    ('pure_gps', 'task6_interpolation'): add_interpolation_reasoning,
    ('pure_gps', 'task7_area_perimeter'): add_area_perimeter_reasoning,
    ('pure_gps', 'task8_bounding_box'): add_bounding_box_reasoning,
    ('pure_gps', 'task9_route_geometry'): add_route_geometry_reasoning,
    ('pure_gps', 'task10_relative_position'): add_relative_position_pure_gps_reasoning,

    # Applied Track
    ('applied', 'task1_place_association'): add_place_association_reasoning,
    ('applied', 'task3_name_disambiguation'): add_name_disambiguation_reasoning,
    ('applied', 'task5_relative_position'): add_relative_position_applied_reasoning,
    ('applied', 'task6_proximity'): add_proximity_reasoning,
    ('applied', 'task7_route_analysis'): add_route_analysis_reasoning,
    ('applied', 'task8_boundary_analysis'): add_boundary_analysis_reasoning,
    ('applied', 'task9_spatial_patterns'): add_spatial_patterns_reasoning,
    ('applied', 'task11_missing_data'): add_missing_data_reasoning,
    ('applied', 'task13_terrain_classification'): add_terrain_classification_reasoning,
}


def process_split_file(filepath: Path, processor_fn) -> int:
    """Add reasoning to all examples in a split file."""
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return 0

    data = load_json(filepath)
    updated = 0
    errors = 0

    for example in data:
        try:
            processor_fn(example)
            updated += 1
        except Exception as e:
            errors += 1
            if errors <= 3:  # Only show first 3 errors
                print(f"  Warning: {e}")

    save_json(filepath, data)

    if errors > 3:
        print(f"  ... and {errors - 3} more errors")

    return updated


def main():
    parser = argparse.ArgumentParser(description="Add reasoning steps to GPSBench splits")
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'dev', 'test', 'all'],
                        help='Which split to process')
    parser.add_argument('--track', type=str, default='both',
                        choices=['pure_gps', 'applied', 'both'],
                        help='Which track to process')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without making changes')
    args = parser.parse_args()

    splits = ['train', 'dev', 'test'] if args.split == 'all' else [args.split]
    tracks = ['pure_gps', 'applied'] if args.track == 'both' else [args.track]

    print("=" * 70)
    print("ADDING REASONING STEPS TO GPSBENCH SPLITS")
    print("=" * 70)
    print(f"Splits: {splits}")
    print(f"Tracks: {tracks}")
    print()

    total_updated = 0

    for track in tracks:
        print(f"\n{'='*70}")
        print(f"Track: {track}")
        print(f"{'='*70}")

        track_dir = DATA_DIR / f"track_{track}" / "splits"

        for (proc_track, task_name), processor_fn in TASK_PROCESSORS.items():
            if proc_track != track:
                continue

            print(f"\n  Processing: {task_name}")

            for split in splits:
                filepath = track_dir / f"{task_name}_{split}.json"

                if args.dry_run:
                    if filepath.exists():
                        data = load_json(filepath)
                        print(f"    {split}: {len(data)} examples (dry run)")
                    else:
                        print(f"    {split}: file not found")
                else:
                    updated = process_split_file(filepath, processor_fn)
                    if updated > 0:
                        print(f"    {split}: {updated} examples updated")
                        total_updated += updated

    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN COMPLETE - No files were modified")
    else:
        print(f"COMPLETE: Added reasoning to {total_updated} examples")
    print("=" * 70)


if __name__ == "__main__":
    main()
