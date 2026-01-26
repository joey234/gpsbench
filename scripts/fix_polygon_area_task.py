#!/usr/bin/env python3
"""
Fix Polygon Area Task Data and Re-evaluate Model Outputs

This script fixes the vertex ordering bug where ground truth was calculated
with sorted vertices but prompts used original vertex order.

The fix: Recalculate ground truth using the actual vertex order in the data file.
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil
from datetime import datetime


def spherical_polygon_area(vertices: List[tuple]) -> float:
    """Calculate spherical polygon area in km² using the vertex order as given."""
    R = 6371
    n = len(vertices)
    if n < 3:
        return 0

    total = 0
    for i in range(n):
        j = (i + 1) % n
        lat1 = math.radians(vertices[i][0])
        lon1 = math.radians(vertices[i][1])
        lat2 = math.radians(vertices[j][0])
        lon2 = math.radians(vertices[j][1])
        total += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))

    area = abs(total) * R * R / 2
    return area


def polygon_perimeter(vertices: List[tuple]) -> float:
    """Calculate polygon perimeter in km."""
    R = 6371

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    perimeter = 0
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        perimeter += haversine(vertices[i][0], vertices[i][1], vertices[j][0], vertices[j][1])
    return perimeter


def fix_task_data(task_file: Path) -> int:
    """Fix the ground truth in the task data file. Returns number of fixed examples."""
    print(f"\nFixing task data: {task_file}")

    # Backup original
    backup_file = task_file.with_suffix('.json.backup')
    if not backup_file.exists():
        shutil.copy(task_file, backup_file)
        print(f"  Created backup: {backup_file}")

    with open(task_file, 'r') as f:
        data = json.load(f)

    fixed_count = 0
    for sample in data:
        vertices = [(v['lat'], v['lon']) for v in sample['vertices']]

        # Recalculate using the actual vertex order
        new_area = spherical_polygon_area(vertices)
        new_perimeter = polygon_perimeter(vertices)

        old_area = sample['ground_truth']['area_km2']
        old_perimeter = sample['ground_truth']['perimeter_km']

        # Check if there's a significant difference
        if abs(new_area - old_area) / max(old_area, 1) > 0.01:  # >1% difference
            fixed_count += 1

        sample['ground_truth']['area_km2'] = round(new_area, 2)
        sample['ground_truth']['perimeter_km'] = round(new_perimeter, 2)

    with open(task_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Fixed {fixed_count}/{len(data)} examples with >1% area difference")
    return fixed_count


def extract_area_from_response(response: str) -> Optional[float]:
    """Extract area value from model response."""
    # Look for FINAL ANSWER pattern
    final_match = re.search(r'FINAL ANSWER:?\s*([^\n]+)', response, re.IGNORECASE)
    if not final_match:
        return None

    answer_text = final_match.group(1).strip()

    # Remove units and clean up
    answer_text = re.sub(r'\s*km[²2]?\s*$', '', answer_text, flags=re.IGNORECASE)
    answer_text = answer_text.replace(',', '').strip()

    # Handle scientific notation (1.79 × 10^8, 1.79e8, etc.)
    sci_match = re.search(r'([\d.]+)\s*[×x]\s*10\^?(\d+)', answer_text)
    if sci_match:
        mantissa = float(sci_match.group(1))
        exponent = int(sci_match.group(2))
        return mantissa * (10 ** exponent)

    # Handle e notation
    e_match = re.search(r'([\d.]+)[eE]\+?(\d+)', answer_text)
    if e_match:
        mantissa = float(e_match.group(1))
        exponent = int(e_match.group(2))
        return mantissa * (10 ** exponent)

    # Plain number
    num_match = re.search(r'[\d.]+', answer_text)
    if num_match:
        try:
            return float(num_match.group())
        except:
            pass

    return None


def reevaluate_model_results(results_dir: Path, task_file: Path) -> Dict[str, Any]:
    """Re-evaluate all model results for the polygon area task."""
    print(f"\nRe-evaluating model results in: {results_dir}")

    # Load corrected task data
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    # Create ground truth lookup
    gt_lookup = {}
    for i, sample in enumerate(task_data):
        gt_lookup[i] = sample['ground_truth']

    model_results = {}

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir() or result_dir.name.startswith('.'):
            continue

        task_result_file = result_dir / 'task_results' / 'pure_gps_task7_area_perimeter.json'
        if not task_result_file.exists():
            continue

        model_name = result_dir.name.replace('gpsbench_', '').split('_')[0]
        print(f"\n  Processing: {model_name}")

        with open(task_result_file, 'r') as f:
            result_data = json.load(f)

        # Backup original
        backup_file = task_result_file.with_suffix('.json.backup_before_fix')
        if not backup_file.exists():
            shutil.copy(task_result_file, backup_file)

        old_correct = result_data['correct']
        old_accuracy = result_data['accuracy']

        new_correct = 0
        total = 0

        for result in result_data['results']:
            example_id = result['example_id']
            if example_id not in gt_lookup:
                continue

            total += 1
            gt = gt_lookup[example_id]
            gt_area = gt['area_km2']
            tolerance = gt.get('tolerance_percent', 5.0) / 100

            # Update ground truth in result
            result['ground_truth'] = gt

            # Extract predicted area from response
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

        result_data['correct'] = new_correct
        result_data['accuracy'] = new_accuracy
        result_data['_fix_note'] = f"Re-evaluated on {datetime.now().isoformat()} after fixing vertex ordering bug"

        # Save updated results
        with open(task_result_file, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"    Old: {old_correct}/{total} correct ({old_accuracy:.2f}%)")
        print(f"    New: {new_correct}/{total} correct ({new_accuracy:.2f}%)")
        print(f"    Change: {new_correct - old_correct:+d} ({new_accuracy - old_accuracy:+.2f}%)")

        model_results[model_name] = {
            'old_correct': old_correct,
            'old_accuracy': old_accuracy,
            'new_correct': new_correct,
            'new_accuracy': new_accuracy,
            'total': total
        }

        # Also update summary.json if it exists
        summary_file = result_dir / 'summary.json'
        if summary_file.exists():
            update_summary(summary_file, new_correct, total, new_accuracy)

    return model_results


def update_summary(summary_file: Path, new_correct: int, total: int, new_accuracy: float):
    """Update the summary.json file with corrected polygon area results."""
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    # Update pure_gps track
    if 'tracks' in summary and 'pure_gps' in summary['tracks']:
        track = summary['tracks']['pure_gps']

        # Find and update the polygon area task
        tasks_key = 'tasks_summary' if 'tasks_summary' in track else 'tasks'
        if tasks_key in track:
            for task in track[tasks_key]:
                if task['task_name'] == 'Polygon Area':
                    old_correct = task['correct']
                    task['correct'] = new_correct
                    task['accuracy'] = new_accuracy

                    # Update track totals
                    track['total_correct'] = track['total_correct'] - old_correct + new_correct
                    track['overall_accuracy'] = track['total_correct'] / track['total_cases'] * 100
                    break

        # Update overall totals
        if 'overall' in summary:
            # Recalculate from tracks
            total_correct = sum(
                t.get('total_correct', 0)
                for t in summary['tracks'].values()
            )
            total_cases = sum(
                t.get('total_cases', 0)
                for t in summary['tracks'].values()
            )
            summary['overall']['total_correct'] = total_correct
            summary['overall']['overall_accuracy'] = total_correct / total_cases * 100 if total_cases > 0 else 0

    summary['_fix_note'] = f"Updated on {datetime.now().isoformat()} after fixing polygon area vertex ordering bug"

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    print("=" * 70)
    print("FIXING POLYGON AREA TASK - Vertex Ordering Bug")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent
    task_file = base_dir / 'data' / 'track_pure_gps' / 'task7_area_perimeter.json'
    results_dir = base_dir / 'results'

    # Step 1: Fix task data
    print("\n" + "=" * 70)
    print("STEP 1: Fixing task data ground truth")
    print("=" * 70)
    fixed_count = fix_task_data(task_file)

    # Step 2: Re-evaluate model results
    print("\n" + "=" * 70)
    print("STEP 2: Re-evaluating model results")
    print("=" * 70)
    model_results = reevaluate_model_results(results_dir, task_file)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Old Acc':<12} {'New Acc':<12} {'Change':<12}")
    print("-" * 70)

    for model, stats in sorted(model_results.items(), key=lambda x: x[1]['new_accuracy'], reverse=True):
        change = stats['new_accuracy'] - stats['old_accuracy']
        print(f"{model:<35} {stats['old_accuracy']:>10.2f}% {stats['new_accuracy']:>10.2f}% {change:>+10.2f}%")

    print("\n" + "=" * 70)
    print("Done! All results have been updated.")
    print("Backups saved with .backup extension")
    print("=" * 70)


if __name__ == '__main__':
    main()
