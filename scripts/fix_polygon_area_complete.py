#!/usr/bin/env python3
"""
Complete fix for Polygon Area task:
1. Update ground truth using standard L'Huilier spherical excess formula
2. Sort vertices properly for simple polygon
3. Re-evaluate all model outputs against new ground truth
"""

import json
import math
import re
from pathlib import Path
import shutil
from datetime import datetime
from typing import Optional, List, Tuple


def spherical_polygon_area(vertices: List[Tuple[float, float]]) -> float:
    """
    Calculate spherical polygon area in km² using L'Huilier's formula.

    This is the standard method used in geodesy and what LLMs would learn.
    vertices: List of (lat, lon) tuples in degrees
    """
    R = 6371.0
    n = len(vertices)
    if n < 3:
        return 0.0

    def to_cartesian(lat, lon):
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x = math.cos(lat_rad) * math.cos(lon_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad)
        z = math.sin(lat_rad)
        return (x, y, z)

    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def angle_between(a, b):
        d = dot(a, b)
        d = max(-1.0, min(1.0, d))
        return math.acos(d)

    def spherical_triangle_area(p1, p2, p3):
        a = angle_between(p2, p3)
        b = angle_between(p1, p3)
        c = angle_between(p1, p2)
        s = (a + b + c) / 2.0

        try:
            t1 = math.tan(s / 2.0)
            t2 = math.tan((s - a) / 2.0)
            t3 = math.tan((s - b) / 2.0)
            t4 = math.tan((s - c) / 2.0)
            product = t1 * t2 * t3 * t4
            if product < 0:
                product = 0
            tan_E_4 = math.sqrt(product)
            E = 4.0 * math.atan(tan_E_4)
            return E
        except (ValueError, ZeroDivisionError):
            return 0.0

    points = [to_cartesian(v[0], v[1]) for v in vertices]

    total_excess = 0.0
    for i in range(1, n - 1):
        excess = spherical_triangle_area(points[0], points[i], points[i + 1])
        total_excess += excess

    return R * R * total_excess


def polygon_perimeter(vertices: List[Tuple[float, float]]) -> float:
    """Calculate polygon perimeter in km using Haversine."""
    R = 6371.0

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    perimeter = 0.0
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        perimeter += haversine(vertices[i][0], vertices[i][1],
                               vertices[j][0], vertices[j][1])
    return perimeter


def sort_vertices(vertices: List[dict]) -> List[dict]:
    """Sort vertices by angle from centroid to form a simple polygon."""
    centroid_lat = sum(v['lat'] for v in vertices) / len(vertices)
    centroid_lon = sum(v['lon'] for v in vertices) / len(vertices)

    def angle(v):
        return math.atan2(v['lon'] - centroid_lon, v['lat'] - centroid_lat)

    return sorted(vertices, key=angle)


def update_task_data(task_file: Path) -> dict:
    """Update task data with sorted vertices and new ground truth."""
    print(f"\nUpdating task data: {task_file}")

    # Backup original
    backup_file = task_file.with_suffix('.json.pre_lhuilier_backup')
    if not backup_file.exists():
        shutil.copy(task_file, backup_file)
        print(f"  Backup saved: {backup_file.name}")

    with open(task_file, 'r') as f:
        data = json.load(f)

    stats = {'total': len(data), 'area_changed': 0, 'order_changed': 0}

    for sample in data:
        # Sort vertices
        original_vertices = sample['vertices']
        sorted_vertices = sort_vertices(original_vertices)

        if original_vertices != sorted_vertices:
            stats['order_changed'] += 1
            sample['vertices'] = sorted_vertices

        # Recalculate ground truth with new formula
        coords = [(v['lat'], v['lon']) for v in sorted_vertices]
        new_area = spherical_polygon_area(coords)
        new_perimeter = polygon_perimeter(coords)

        old_area = sample['ground_truth']['area_km2']
        if abs(new_area - old_area) / max(old_area, 1) > 0.01:
            stats['area_changed'] += 1

        sample['ground_truth']['area_km2'] = round(new_area, 2)
        sample['ground_truth']['perimeter_km'] = round(new_perimeter, 2)

    with open(task_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Total examples: {stats['total']}")
    print(f"  Vertex order fixed: {stats['order_changed']}")
    print(f"  Area values changed (>1%): {stats['area_changed']}")

    return stats


def extract_area(response: str) -> Optional[float]:
    """Extract area from model response."""
    match = re.search(r'FINAL ANSWER:?\s*([^\n]+)', response, re.IGNORECASE)
    if not match:
        return None

    ans = match.group(1).strip()
    ans = re.sub(r'\s*km[²2]?\s*$', '', ans, flags=re.IGNORECASE)
    ans = ans.replace(',', '').strip()

    # Scientific notation: 1.79 × 10^8
    sci = re.search(r'([\d.]+)\s*[×x]\s*10\^?(\d+)', ans)
    if sci:
        return float(sci.group(1)) * (10 ** int(sci.group(2)))

    # E notation: 1.79e8
    e_match = re.search(r'([\d.]+)[eE]\+?(\d+)', ans)
    if e_match:
        return float(e_match.group(1)) * (10 ** int(e_match.group(2)))

    # Plain number
    try:
        num = re.search(r'[\d.]+', ans)
        if num:
            return float(num.group())
    except:
        pass

    return None


def reevaluate_model(result_file: Path, task_data: List[dict]) -> dict:
    """Re-evaluate a model's results against new ground truth."""
    with open(result_file, 'r') as f:
        results = json.load(f)

    # Build ground truth lookup
    gt_lookup = {i: sample['ground_truth'] for i, sample in enumerate(task_data)}

    old_correct = results['correct']
    old_accuracy = results['accuracy']

    new_correct = 0
    total = 0

    for result in results['results']:
        idx = result['example_id']
        if idx not in gt_lookup:
            continue

        total += 1
        gt = gt_lookup[idx]
        result['ground_truth'] = gt  # Update stored GT

        pred = extract_area(result['response'])
        tolerance = gt.get('tolerance_percent', 5.0) / 100

        if pred is not None:
            error = abs(pred - gt['area_km2']) / gt['area_km2']
            is_correct = error <= tolerance
        else:
            is_correct = False

        result['correct'] = is_correct
        if is_correct:
            new_correct += 1

    new_accuracy = (new_correct / total * 100) if total > 0 else 0

    results['correct'] = new_correct
    results['accuracy'] = new_accuracy
    results['_updated'] = f"Re-evaluated with L'Huilier formula on {datetime.now().isoformat()}"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    return {
        'old_correct': old_correct,
        'old_accuracy': old_accuracy,
        'new_correct': new_correct,
        'new_accuracy': new_accuracy,
        'total': total,
        'change': new_correct - old_correct
    }


def update_summary(summary_file: Path, old_correct: int, new_correct: int, new_accuracy: float):
    """Update summary.json with new results."""
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    diff = new_correct - old_correct

    if 'tracks' in summary and 'pure_gps' in summary['tracks']:
        track = summary['tracks']['pure_gps']

        tasks_key = 'tasks_summary' if 'tasks_summary' in track else 'tasks'
        if tasks_key in track:
            for task in track[tasks_key]:
                if task['task_name'] == 'Polygon Area':
                    task['correct'] = new_correct
                    task['accuracy'] = new_accuracy
                    break

        track['total_correct'] += diff
        track['overall_accuracy'] = track['total_correct'] / track['total_cases'] * 100

    if 'overall' in summary:
        summary['overall']['total_correct'] += diff
        summary['overall']['overall_accuracy'] = (
            summary['overall']['total_correct'] / summary['overall']['total_cases'] * 100
        )

    summary['_polygon_area_fix'] = f"Updated with L'Huilier formula on {datetime.now().isoformat()}"

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    print("=" * 70)
    print("POLYGON AREA TASK - COMPLETE FIX")
    print("Using standard L'Huilier spherical excess formula")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent
    task_file = base_dir / 'data' / 'track_pure_gps' / 'task7_area_perimeter.json'
    results_dir = base_dir / 'results'

    # Step 1: Update task data
    print("\n" + "=" * 70)
    print("STEP 1: Updating task data with L'Huilier formula")
    print("=" * 70)
    update_task_data(task_file)

    # Load updated task data
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    # Step 2: Re-evaluate all models
    print("\n" + "=" * 70)
    print("STEP 2: Re-evaluating model results")
    print("=" * 70)

    model_stats = {}

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir() or result_dir.name.startswith('.'):
            continue

        result_file = result_dir / 'task_results' / 'pure_gps_task7_area_perimeter.json'
        if not result_file.exists():
            continue

        model_name = result_dir.name.replace('gpsbench_', '').split('_20')[0]
        print(f"\n  {model_name}:")

        stats = reevaluate_model(result_file, task_data)
        model_stats[model_name] = stats

        print(f"    Before: {stats['old_correct']}/{stats['total']} ({stats['old_accuracy']:.2f}%)")
        print(f"    After:  {stats['new_correct']}/{stats['total']} ({stats['new_accuracy']:.2f}%)")
        print(f"    Change: {stats['change']:+d} ({stats['new_accuracy'] - stats['old_accuracy']:+.2f}%)")

        # Update summary
        summary_file = result_dir / 'summary.json'
        if summary_file.exists():
            update_summary(summary_file, stats['old_correct'], stats['new_correct'], stats['new_accuracy'])

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<40} {'Before':<12} {'After':<12} {'Change':<10}")
    print("-" * 75)

    for model, stats in sorted(model_stats.items(), key=lambda x: x[1]['new_accuracy'], reverse=True):
        print(f"{model:<40} {stats['old_accuracy']:>10.2f}% {stats['new_accuracy']:>10.2f}% {stats['new_accuracy'] - stats['old_accuracy']:>+9.2f}%")

    print("\n" + "=" * 70)
    print("Done! Task data and all model results have been updated.")
    print("=" * 70)


if __name__ == '__main__':
    main()
