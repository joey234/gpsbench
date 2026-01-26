#!/usr/bin/env python3
"""
Regenerate track summary files (track_applied.json, track_pure_gps.json)
from individual task result files.

This fixes the issue where track summaries are outdated after task files
were re-evaluated.

For calculation tasks, uses 1-MAPE (Mean Absolute Percentage Error) as score
instead of binary accuracy.
"""

import json
import re
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import warnings


# Tasks that should use 1-MAPE scoring (calculation tasks with positive numeric outputs)
# Key is the task filename without prefix
# Note: Coordinate tasks (interpolation, transformation) use binary accuracy
# because coordinates can cross zero and MAPE doesn't apply well
MAPE_TASKS = {
    'task4_distance_calculation.json': {
        'gt_field': 'distance_km',
        'task_name': 'Distance Calculation',
    },
    'task5_bearing_computation.json': {
        'gt_field': 'initial_bearing_deg',
        'task_name': 'Bearing Computation',
    },
    'task7_area_perimeter.json': {
        'gt_field': 'area_km2',
        'task_name': 'Polygon Area',
    },
}

# Normalize task names to ensure consistency across all models
TASK_NAME_NORMALIZATION = {
    # Applied track
    'Missing Data': 'Missing Data Inference',
    'Missing Data Inference': 'Missing Data Inference',
    # Pure GPS track
    'Area Perimeter': 'Polygon Area',
    'Polygon Area': 'Polygon Area',
    'Area Perimeter Calc': 'Polygon Area',
}


def extract_number_from_response(response: str) -> Optional[float]:
    """Extract numerical answer from FINAL ANSWER line."""
    match = re.search(r'FINAL ANSWER:?\s*([^\n]+)', response, re.IGNORECASE)
    if not match:
        return None

    answer = match.group(1).strip()

    # Remove common units
    answer = re.sub(r'\s*(km[²2]?|degrees?|deg|°|kilometers?|square\s+kilometers?|m)\s*$', '', answer, flags=re.IGNORECASE)
    answer = answer.replace(',', '').strip()

    # Scientific notation: 1.79 × 10^8
    sci_match = re.search(r'([\d.]+)\s*[×x]\s*10\^?(\d+)', answer)
    if sci_match:
        return float(sci_match.group(1)) * (10 ** int(sci_match.group(2)))

    # E notation: 1.79e8
    e_match = re.search(r'([\d.]+)[eE]\+?(-?\d+)', answer)
    if e_match:
        return float(e_match.group(1)) * (10 ** int(e_match.group(2)))

    # Plain number (including negative)
    num_match = re.search(r'-?[\d.]+', answer)
    if num_match:
        try:
            return float(num_match.group())
        except:
            pass

    return None


def compute_mape_score(results: list, gt_field: str, task_name: str) -> Tuple[float, int]:
    """
    Compute 1-MAPE score for calculation tasks.
    Returns (score as percentage 0-100, valid_count).

    MAPE = Mean Absolute Percentage Error = mean(|pred - actual| / |actual|)
    Score = (1 - MAPE) * 100, capped at [0, 100]
    """
    percentage_errors = []

    for r in results:
        gt = r.get('ground_truth', {})
        response = r.get('response', '') or r.get('model_response', '')

        # Extract numeric prediction
        pred = extract_number_from_response(response)
        if pred is None:
            continue

        gt_value = gt.get(gt_field)
        if gt_value is None or gt_value == 0:
            continue

        pct_error = abs(pred - gt_value) / abs(gt_value)
        percentage_errors.append(pct_error)

    if not percentage_errors:
        return 0.0, 0

    # Filter out extreme outliers (>1000% error) to avoid skewing
    valid_errors = [e for e in percentage_errors if e < 10.0]

    if not valid_errors:
        return 0.0, len(percentage_errors)

    mape = sum(valid_errors) / len(valid_errors)

    # Score is 1 - MAPE, capped at 0-100%
    score = max(0, min(100, (1 - mape) * 100))

    return score, len(percentage_errors)


def get_correct_count(results):
    """Get correct count handling both 'correct' and 'is_correct' fields."""
    if not results:
        return 0

    first = results[0]
    if 'correct' in first:
        return sum(1 for r in results if r.get('correct', False))
    elif 'is_correct' in first:
        return sum(1 for r in results if r.get('is_correct', False))
    return 0


def regenerate_track_summary(results_dir: Path, track: str):
    """Regenerate track summary from task files."""
    task_results_dir = results_dir / 'task_results'
    prefix = 'applied_' if track == 'applied' else 'pure_gps_'

    tasks = []
    total_score = 0.0
    total_samples = 0

    for task_file in sorted(task_results_dir.glob(f'{prefix}*.json')):
        # Skip backup files
        if task_file.suffix != '.json' or '.backup' in task_file.name:
            continue

        with open(task_file) as f:
            data = json.load(f)

        results = data.get('results', [])
        total = len(results)

        # Get task name from file or data
        task_name = data.get('task_name')
        if not task_name:
            # Extract from filename: applied_task1_place_association.json -> Place Association
            name_part = task_file.stem.replace(prefix, '').split('_', 1)[-1]
            task_name = name_part.replace('_', ' ').title()

        # Normalize task name for consistency
        task_name = TASK_NAME_NORMALIZATION.get(task_name, task_name)

        # Check if this is a MAPE task (only for pure_gps track)
        task_filename = task_file.name.replace(prefix, '')

        if track == 'pure_gps' and task_filename in MAPE_TASKS:
            # Use 1-MAPE scoring for calculation tasks
            config = MAPE_TASKS[task_filename]
            score, valid_count = compute_mape_score(results, config['gt_field'], task_name)
            accuracy = score
            correct = valid_count  # Number of valid predictions
            scoring_method = 'mape'
        else:
            # Use binary accuracy for other tasks
            correct = get_correct_count(results)
            accuracy = (correct / total * 100) if total > 0 else 0
            scoring_method = 'binary'

        tasks.append({
            'task_name': task_name,
            'task_file': task_filename,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'scoring_method': scoring_method,
        })

        # For overall accuracy, weight by number of samples
        total_score += accuracy * total
        total_samples += total

    overall_accuracy = (total_score / total_samples) if total_samples > 0 else 0

    summary = {
        'track': track,
        'overall_accuracy': overall_accuracy,
        'total_samples': total_samples,
        'tasks': tasks,
        'regenerated_at': datetime.now().isoformat(),
    }

    return summary


def main():
    results_base = Path('results')

    # Process all model directories
    for d in sorted(results_base.iterdir()):
        if not d.is_dir() or d.name.startswith('.'):
            continue

        task_results_dir = d / 'task_results'
        if not task_results_dir.exists():
            continue

        print(f"\nProcessing: {d.name}")

        # Check if there are task files
        applied_files = list(task_results_dir.glob('applied_*.json'))
        pure_gps_files = list(task_results_dir.glob('pure_gps_*.json'))

        if applied_files:
            # Regenerate applied track summary
            applied_summary = regenerate_track_summary(d, 'applied')

            # Backup old file
            old_file = d / 'track_applied.json'
            if old_file.exists():
                backup_file = d / 'track_applied.json.bak'
                old_file.rename(backup_file)

            # Write new summary
            with open(d / 'track_applied.json', 'w') as f:
                json.dump(applied_summary, f, indent=2)

            print(f"  Applied: {applied_summary['overall_accuracy']:.1f}% ({applied_summary['total_samples']} samples)")

        if pure_gps_files:
            # Regenerate pure GPS track summary
            pure_gps_summary = regenerate_track_summary(d, 'pure_gps')

            # Backup old file
            old_file = d / 'track_pure_gps.json'
            if old_file.exists():
                backup_file = d / 'track_pure_gps.json.bak'
                old_file.rename(backup_file)

            # Write new summary
            with open(d / 'track_pure_gps.json', 'w') as f:
                json.dump(pure_gps_summary, f, indent=2)

            # Print MAPE task scores
            mape_tasks = [t for t in pure_gps_summary['tasks'] if t.get('scoring_method') == 'mape']
            if mape_tasks:
                mape_info = ', '.join([f"{t['task_name'][:8]}:{t['accuracy']:.0f}%" for t in mape_tasks[:3]])
                print(f"  Pure GPS: {pure_gps_summary['overall_accuracy']:.1f}% (MAPE tasks: {mape_info})")

    print("\n" + "=" * 60)
    print("Done! Track summaries have been regenerated from task files.")
    print("Old summaries backed up as *.json.bak")


if __name__ == "__main__":
    main()
