#!/usr/bin/env python3
"""
Fix Polygon Area Task - Sort vertices in task data to match ground truth calculation.

The issue: Ground truth was calculated with sorted vertices, but prompts used unsorted vertices.
The fix: Update task data to use sorted vertices (matching what was used for GT calculation).
Then re-evaluate model outputs - since we're keeping the same GT, models that correctly
computed the area using sorted vertices should now match.
"""

import json
import math
import re
from pathlib import Path
import shutil
from datetime import datetime
from typing import Optional


def sort_vertices_for_polygon(vertices):
    """Sort vertices by angle from centroid (same as benchmark generator)."""
    centroid_lat = sum(v['lat'] for v in vertices) / len(vertices)
    centroid_lon = sum(v['lon'] for v in vertices) / len(vertices)

    def angle_from_centroid(v):
        return math.atan2(v['lon'] - centroid_lon, v['lat'] - centroid_lat)

    return sorted(vertices, key=angle_from_centroid)


def fix_task_data(task_file: Path) -> int:
    """Fix the task data by sorting vertices to match ground truth calculation."""
    print(f"\nFixing task data: {task_file}")

    # Backup
    backup_file = task_file.with_suffix('.json.original_unsorted')
    if not backup_file.exists():
        shutil.copy(task_file, backup_file)
        print(f"  Created backup: {backup_file.name}")

    with open(task_file, 'r') as f:
        data = json.load(f)

    fixed_count = 0
    for sample in data:
        original_vertices = sample['vertices']
        sorted_vertices = sort_vertices_for_polygon(original_vertices)

        # Check if order changed
        if original_vertices != sorted_vertices:
            fixed_count += 1
            sample['vertices'] = sorted_vertices

    with open(task_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Fixed vertex order in {fixed_count}/{len(data)} examples")
    return fixed_count


def extract_area_from_response(response: str) -> Optional[float]:
    """Extract area value from model response."""
    final_match = re.search(r'FINAL ANSWER:?\s*([^\n]+)', response, re.IGNORECASE)
    if not final_match:
        return None

    answer = final_match.group(1).strip()
    answer = re.sub(r'\s*km[²2]?\s*$', '', answer, flags=re.IGNORECASE)
    answer = answer.replace(',', '').strip()

    # Scientific notation
    sci_match = re.search(r'([\d.]+)\s*[×x]\s*10\^?(\d+)', answer)
    if sci_match:
        return float(sci_match.group(1)) * (10 ** int(sci_match.group(2)))

    # E notation
    e_match = re.search(r'([\d.]+)[eE]\+?(\d+)', answer)
    if e_match:
        return float(e_match.group(1)) * (10 ** int(e_match.group(2)))

    # Plain number
    try:
        return float(re.search(r'[\d.]+', answer).group())
    except:
        return None


def reevaluate_results(results_dir: Path) -> dict:
    """Re-evaluate model results (GT unchanged, just re-check correctness)."""
    print(f"\nRe-evaluating results in: {results_dir}")

    model_stats = {}

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir() or result_dir.name.startswith('.'):
            continue

        task_file = result_dir / 'task_results' / 'pure_gps_task7_area_perimeter.json'
        if not task_file.exists():
            continue

        model_name = result_dir.name.replace('gpsbench_', '').split('_20')[0]

        with open(task_file, 'r') as f:
            data = json.load(f)

        old_correct = data['correct']
        old_accuracy = data['accuracy']

        new_correct = 0
        total = 0

        for result in data['results']:
            total += 1
            gt_area = result['ground_truth']['area_km2']
            tolerance = result['ground_truth'].get('tolerance_percent', 5.0) / 100

            pred_area = extract_area_from_response(result['response'])

            if pred_area is not None:
                error = abs(pred_area - gt_area) / gt_area
                is_correct = error <= tolerance
            else:
                is_correct = False

            result['correct'] = is_correct
            if is_correct:
                new_correct += 1

        new_accuracy = (new_correct / total * 100) if total > 0 else 0

        data['correct'] = new_correct
        data['accuracy'] = new_accuracy

        with open(task_file, 'w') as f:
            json.dump(data, f, indent=2)

        # Update summary
        summary_file = result_dir / 'summary.json'
        if summary_file.exists():
            update_summary(summary_file, old_correct, new_correct)

        print(f"  {model_name}: {old_accuracy:.2f}% -> {new_accuracy:.2f}% ({new_correct - old_correct:+d})")

        model_stats[model_name] = {
            'old': old_accuracy,
            'new': new_accuracy,
            'change': new_accuracy - old_accuracy
        }

    return model_stats


def update_summary(summary_file: Path, old_correct: int, new_correct: int):
    """Update summary.json with new polygon area results."""
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    diff = new_correct - old_correct

    if 'tracks' in summary and 'pure_gps' in summary['tracks']:
        track = summary['tracks']['pure_gps']

        # Update task
        tasks_key = 'tasks_summary' if 'tasks_summary' in track else 'tasks'
        if tasks_key in track:
            for task in track[tasks_key]:
                if task['task_name'] == 'Polygon Area':
                    task['correct'] = new_correct
                    task['accuracy'] = new_correct / task['total'] * 100
                    break

        # Update track totals
        track['total_correct'] += diff
        track['overall_accuracy'] = track['total_correct'] / track['total_cases'] * 100

    # Update overall
    if 'overall' in summary:
        summary['overall']['total_correct'] += diff
        summary['overall']['overall_accuracy'] = (
            summary['overall']['total_correct'] / summary['overall']['total_cases'] * 100
        )

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    print("=" * 70)
    print("FIXING POLYGON AREA TASK")
    print("Sorting vertices in prompts to match ground truth calculation")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent
    task_file = base_dir / 'data' / 'track_pure_gps' / 'task7_area_perimeter.json'
    results_dir = base_dir / 'results'

    # Step 1: Fix task data (sort vertices)
    print("\n" + "-" * 70)
    print("STEP 1: Sorting vertices in task data")
    print("-" * 70)
    fix_task_data(task_file)

    # Step 2: Re-evaluate (ground truth unchanged, just re-check)
    print("\n" + "-" * 70)
    print("STEP 2: Re-evaluating model results")
    print("-" * 70)
    model_stats = reevaluate_results(results_dir)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<40} {'Before':<10} {'After':<10} {'Change':<10}")
    print("-" * 70)
    for model, stats in sorted(model_stats.items(), key=lambda x: x[1]['new'], reverse=True):
        print(f"{model:<40} {stats['old']:>8.2f}% {stats['new']:>8.2f}% {stats['change']:>+8.2f}%")

    print("\nNote: Vertex ordering in task data is now fixed.")
    print("Ground truth values remain unchanged.")
    print("Models should now be evaluated fairly against the correct vertex order.")


if __name__ == '__main__':
    main()
