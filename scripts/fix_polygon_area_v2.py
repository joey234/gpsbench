#!/usr/bin/env python3
"""
Fix Polygon Area task - v2

Key insight: Models were given UNSORTED vertices and computed L'Huilier on those.
The fix: Calculate ground truth with L'Huilier on the SAME unsorted vertices.

This keeps vertex order unchanged but uses the standard formula.
"""

import json
import math
import re
from pathlib import Path
import shutil
from datetime import datetime
from typing import Optional, List, Tuple


def lhuilier_spherical_area(vertices: List[Tuple[float, float]]) -> float:
    """
    Calculate spherical polygon area using L'Huilier's formula.
    vertices: List of (lat, lon) tuples - order matters!
    """
    R = 6371.0
    n = len(vertices)
    if n < 3:
        return 0.0

    def to_xyz(lat, lon):
        lat_r, lon_r = math.radians(lat), math.radians(lon)
        return (math.cos(lat_r) * math.cos(lon_r),
                math.cos(lat_r) * math.sin(lon_r),
                math.sin(lat_r))

    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def angle(a, b):
        d = max(-1.0, min(1.0, dot(a, b)))
        return math.acos(d)

    def triangle_excess(p1, p2, p3):
        a = angle(p2, p3)
        b = angle(p1, p3)
        c = angle(p1, p2)
        s = (a + b + c) / 2.0
        try:
            prod = math.tan(s/2) * math.tan((s-a)/2) * math.tan((s-b)/2) * math.tan((s-c)/2)
            if prod < 0:
                prod = 0
            return 4.0 * math.atan(math.sqrt(prod))
        except:
            return 0.0

    pts = [to_xyz(v[0], v[1]) for v in vertices]
    excess = sum(triangle_excess(pts[0], pts[i], pts[i+1]) for i in range(1, n-1))
    return R * R * excess


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def perimeter(vertices):
    n = len(vertices)
    return sum(haversine(vertices[i][0], vertices[i][1],
                         vertices[(i+1)%n][0], vertices[(i+1)%n][1])
               for i in range(n))


def extract_area(response: str) -> Optional[float]:
    match = re.search(r'FINAL ANSWER:?\s*([^\n]+)', response, re.IGNORECASE)
    if not match:
        return None
    ans = match.group(1).strip()
    ans = re.sub(r'\s*km[²2]?\s*$', '', ans, flags=re.IGNORECASE)
    ans = ans.replace(',', '').strip()

    sci = re.search(r'([\d.]+)\s*[×x]\s*10\^?(\d+)', ans)
    if sci:
        return float(sci.group(1)) * (10 ** int(sci.group(2)))

    e_match = re.search(r'([\d.]+)[eE]\+?(\d+)', ans)
    if e_match:
        return float(e_match.group(1)) * (10 ** int(e_match.group(2)))

    try:
        return float(re.search(r'[\d.]+', ans).group())
    except:
        return None


def main():
    print("=" * 70)
    print("POLYGON AREA FIX v2")
    print("Using L'Huilier on ORIGINAL vertex order (what models computed)")
    print("=" * 70)

    base = Path(__file__).parent.parent

    # Restore original task data first
    task_file = base / 'data' / 'track_pure_gps' / 'task7_area_perimeter.json'
    backup = task_file.with_suffix('.json.pre_lhuilier_backup')

    if backup.exists():
        shutil.copy(backup, task_file)
        print(f"\nRestored original task data from backup")

    # Load original data
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    # Update ground truth with L'Huilier (keeping original vertex order!)
    print(f"\nUpdating {len(task_data)} examples with L'Huilier formula...")

    for sample in task_data:
        vertices = [(v['lat'], v['lon']) for v in sample['vertices']]
        new_area = lhuilier_spherical_area(vertices)
        new_perim = perimeter(vertices)
        sample['ground_truth']['area_km2'] = round(new_area, 2)
        sample['ground_truth']['perimeter_km'] = round(new_perim, 2)

    with open(task_file, 'w') as f:
        json.dump(task_data, f, indent=2)

    print("Task data updated!")

    # Re-evaluate all models
    print("\n" + "=" * 70)
    print("Re-evaluating models...")
    print("=" * 70)

    results_dir = base / 'results'
    model_stats = {}

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir() or result_dir.name.startswith('.'):
            continue

        # Restore original results first
        result_file = result_dir / 'task_results' / 'pure_gps_task7_area_perimeter.json'
        result_backup = result_file.with_suffix('.json.backup_before_fix')

        if not result_file.exists() and not result_backup.exists():
            continue

        if result_backup.exists():
            shutil.copy(result_backup, result_file)

        model = result_dir.name.replace('gpsbench_', '').split('_20')[0]

        with open(result_file, 'r') as f:
            results = json.load(f)

        old_correct = results['correct']
        new_correct = 0
        total = len(results['results'])

        for i, r in enumerate(results['results']):
            idx = r['example_id']
            gt = task_data[idx]['ground_truth']
            r['ground_truth'] = gt

            pred = extract_area(r['response'])
            tol = gt.get('tolerance_percent', 5.0) / 100

            if pred and gt['area_km2'] > 0:
                err = abs(pred - gt['area_km2']) / gt['area_km2']
                correct = err <= tol
            else:
                correct = False

            r['correct'] = correct
            if correct:
                new_correct += 1

        new_acc = new_correct / total * 100 if total > 0 else 0
        old_acc = results['accuracy']

        results['correct'] = new_correct
        results['accuracy'] = new_acc

        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Update summary
        summary_file = result_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            diff = new_correct - old_correct
            if 'tracks' in summary and 'pure_gps' in summary['tracks']:
                track = summary['tracks']['pure_gps']
                tasks_key = 'tasks_summary' if 'tasks_summary' in track else 'tasks'
                for task in track.get(tasks_key, []):
                    if task['task_name'] == 'Polygon Area':
                        task['correct'] = new_correct
                        task['accuracy'] = new_acc
                        break
                track['total_correct'] += diff
                track['overall_accuracy'] = track['total_correct'] / track['total_cases'] * 100

            if 'overall' in summary:
                summary['overall']['total_correct'] += diff
                summary['overall']['overall_accuracy'] = (
                    summary['overall']['total_correct'] / summary['overall']['total_cases'] * 100
                )

            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

        print(f"  {model}: {old_acc:.2f}% -> {new_acc:.2f}% ({new_correct - old_correct:+d})")
        model_stats[model] = {'old': old_acc, 'new': new_acc}

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<40} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 70)
    for m, s in sorted(model_stats.items(), key=lambda x: x[1]['new'], reverse=True):
        print(f"{m:<40} {s['old']:>9.2f}% {s['new']:>9.2f}% {s['new']-s['old']:>+9.2f}%")


if __name__ == '__main__':
    main()
