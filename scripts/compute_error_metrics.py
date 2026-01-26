#!/usr/bin/env python3
"""
Compute Error Metrics for Numerical Tasks

Instead of binary tolerance-based accuracy, compute actual error distributions
for tasks like distance calculation, bearing computation, and area calculation.

Metrics computed:
- Mean Absolute Error (MAE)
- Median Absolute Error
- Standard Deviation
- Percentiles (P50, P75, P90, P95, P99)
- Mean Absolute Percentage Error (MAPE)
- Accuracy at different tolerance levels

Usage:
    python scripts/compute_error_metrics.py                    # Process all models
    python scripts/compute_error_metrics.py results/model_dir  # Process single model
    python scripts/compute_error_metrics.py --save             # Save to summary.json
"""

import json
import sys
import re
import math
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse


# Task configurations
NUMERICAL_TASKS = {
    'pure_gps_task4_distance_calculation.json': {
        'gt_field': 'distance_km',
        'unit': 'km',
        'task_name': 'Distance Calculation',
        'type': 'distance'
    },
    'pure_gps_task5_bearing_computation.json': {
        'gt_field': 'initial_bearing_deg',
        'unit': '°',
        'task_name': 'Bearing Computation',
        'type': 'bearing'
    },
    'pure_gps_task7_area_perimeter.json': {
        'gt_field': 'area_km2',
        'unit': 'km²',
        'task_name': 'Polygon Area',
        'type': 'area'
    },
}


def extract_number_from_response(response: str) -> Optional[float]:
    """Extract numerical answer from FINAL ANSWER line."""
    match = re.search(r'FINAL ANSWER:?\s*([^\n]+)', response, re.IGNORECASE)
    if not match:
        return None

    answer = match.group(1).strip()

    # Remove common units
    answer = re.sub(r'\s*(km[²2]?|degrees?|deg|°|kilometers?|square\s+kilometers?)\s*$', '', answer, flags=re.IGNORECASE)
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


def compute_metrics_from_errors(abs_errors: List[float], pct_errors: List[float]) -> Dict[str, Any]:
    """Compute comprehensive error metrics from error lists."""
    if not abs_errors:
        return {}

    # Filter out invalid percentage errors
    valid_pct = [e for e in pct_errors if not math.isnan(e) and not math.isinf(e) and abs(e) < 1000]

    metrics = {
        'count': len(abs_errors),
        'mae': statistics.mean(abs_errors),
        'median': statistics.median(abs_errors),
        'std': statistics.stdev(abs_errors) if len(abs_errors) > 1 else 0,
        'min': min(abs_errors),
        'max': max(abs_errors),
    }

    # Percentiles
    sorted_errors = sorted(abs_errors)
    n = len(sorted_errors)
    metrics['p50'] = sorted_errors[int(n * 0.50)] if n > 0 else 0
    metrics['p75'] = sorted_errors[int(n * 0.75)] if n > 0 else 0
    metrics['p90'] = sorted_errors[int(n * 0.90)] if n > 0 else 0
    metrics['p95'] = sorted_errors[int(n * 0.95)] if n > 0 else 0
    metrics['p99'] = sorted_errors[min(int(n * 0.99), n-1)] if n > 0 else 0

    # Percentage errors (MAPE)
    if valid_pct:
        metrics['mape'] = statistics.mean(valid_pct) * 100  # As percentage
        metrics['median_pct'] = statistics.median(valid_pct) * 100

    return metrics


def compute_accuracy_at_tolerances(pct_errors: List[float], tolerances: List[float]) -> Dict[str, float]:
    """Compute accuracy at different tolerance levels."""
    valid_pct = [abs(e) * 100 for e in pct_errors if not math.isnan(e) and not math.isinf(e)]
    if not valid_pct:
        return {}

    result = {}
    for tol in tolerances:
        within = sum(1 for e in valid_pct if e <= tol)
        result[f'{tol}%'] = round(within / len(valid_pct) * 100, 2)

    return result


def process_task_file(result_file: Path, config: Dict) -> Dict[str, Any]:
    """Process a single task result file and compute error metrics."""
    gt_field = config['gt_field']

    with open(result_file, 'r') as f:
        data = json.load(f)

    abs_errors = []
    pct_errors = []
    extracted = 0
    total = len(data.get('results', []))

    for result in data.get('results', []):
        gt = result.get('ground_truth', {})
        gt_value = gt.get(gt_field)

        if gt_value is None:
            continue

        pred = extract_number_from_response(result.get('response', ''))

        if pred is not None:
            extracted += 1
            error = abs(pred - gt_value)
            abs_errors.append(error)

            if gt_value != 0:
                pct_errors.append(error / gt_value)

    metrics = compute_metrics_from_errors(abs_errors, pct_errors)

    # Add extraction stats
    metrics['extracted'] = extracted
    metrics['total'] = total
    metrics['extraction_rate'] = round(extracted / total * 100, 1) if total > 0 else 0

    # Add accuracy at different tolerances
    tolerances = [1, 2, 3, 5, 10, 15, 20] if config['type'] != 'bearing' else [0.5, 1, 2, 3, 5]
    metrics['accuracy_at'] = compute_accuracy_at_tolerances(pct_errors, tolerances)

    return metrics


def process_model(model_dir: Path) -> Dict[str, Dict]:
    """Process all numerical tasks for a single model."""
    task_results_dir = model_dir / 'task_results'
    if not task_results_dir.exists():
        return {}

    results = {}

    for task_file, config in NUMERICAL_TASKS.items():
        result_file = task_results_dir / task_file
        if not result_file.exists():
            continue

        metrics = process_task_file(result_file, config)
        if metrics.get('count', 0) > 0:
            results[config['task_name']] = {
                'metrics': metrics,
                'unit': config['unit'],
                'type': config['type']
            }

    return results


def save_to_summary(model_dir: Path, error_metrics: Dict):
    """Save error metrics to the model's summary.json."""
    summary_file = model_dir / 'summary.json'

    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}

    summary['error_metrics'] = error_metrics

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def print_task_comparison(all_results: Dict[str, Dict], task_name: str):
    """Print comparison table for a single task across all models."""
    print(f"\n{'='*90}")
    print(f"TASK: {task_name}")
    print('='*90)

    # Get task info
    models_with_task = [(m, r[task_name]) for m, r in all_results.items() if task_name in r]
    if not models_with_task:
        print("  No data available")
        return

    unit = models_with_task[0][1]['unit']
    task_type = models_with_task[0][1]['type']

    # Header
    print(f"\n{'Model':<35} {'MAE':>10} {'Median':>10} {'MAPE':>8} {'P90':>10} {'Acc@5%':>8}")
    print('-' * 90)

    # Sort by MAE
    models_with_task.sort(key=lambda x: x[1]['metrics'].get('mae', float('inf')))

    for model, task_data in models_with_task:
        m = task_data['metrics']
        mae = m.get('mae', 0)
        median = m.get('median', 0)
        mape = m.get('mape', 0)
        p90 = m.get('p90', 0)
        acc5 = m.get('accuracy_at', {}).get('5%', 0)

        print(f"{model:<35} {mae:>8.2f}{unit:<2} {median:>8.2f}{unit:<2} {mape:>6.1f}% {p90:>8.2f}{unit:<2} {acc5:>6.1f}%")


def print_summary_table(all_results: Dict[str, Dict]):
    """Print overall summary of best models per task."""
    print("\n" + "=" * 90)
    print("SUMMARY: Best Models by Task (Lowest MAE)")
    print("=" * 90)

    tasks = set()
    for model_results in all_results.values():
        tasks.update(model_results.keys())

    for task in sorted(tasks):
        best_model = None
        best_mae = float('inf')
        best_unit = ''

        for model, results in all_results.items():
            if task in results:
                mae = results[task]['metrics'].get('mae', float('inf'))
                if mae < best_mae:
                    best_mae = mae
                    best_model = model
                    best_unit = results[task]['unit']

        if best_model:
            mape = all_results[best_model][task]['metrics'].get('mape', 0)
            print(f"  {task:<25} {best_model:<35} MAE: {best_mae:>8.2f}{best_unit} ({mape:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Compute error metrics for numerical tasks')
    parser.add_argument('results_path', nargs='?', type=Path,
                        default=Path(__file__).parent.parent / 'results',
                        help='Results directory or specific model folder')
    parser.add_argument('--save', action='store_true',
                        help='Save metrics to summary.json files')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output JSON file for all metrics')
    args = parser.parse_args()

    print("=" * 90)
    print("ERROR METRICS FOR NUMERICAL TASKS")
    print("Computing MAE, MAPE, percentiles instead of binary accuracy")
    print("=" * 90)

    results_path = args.results_path

    # Check if single model or all models
    if (results_path / 'task_results').exists():
        # Single model
        model_name = results_path.name.replace('gpsbench_', '').split('_20')[0]
        print(f"\nProcessing single model: {model_name}")

        model_results = process_model(results_path)
        if model_results:
            all_results = {model_name: model_results}
            if args.save:
                save_to_summary(results_path, model_results)
    else:
        # All models in directory
        all_results = {}

        for model_dir in sorted(results_path.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue

            model_name = model_dir.name.replace('gpsbench_', '').split('_20')[0]

            model_results = process_model(model_dir)
            if model_results:
                all_results[model_name] = model_results
                print(f"  Processed: {model_name} ({len(model_results)} tasks)")

                if args.save:
                    save_to_summary(model_dir, model_results)

    if not all_results:
        print("\nNo numerical task results found.")
        return

    # Print comparison tables for each task
    tasks = set()
    for model_results in all_results.values():
        tasks.update(model_results.keys())

    for task in sorted(tasks):
        print_task_comparison(all_results, task)

    # Print summary
    print_summary_table(all_results)

    # Save output if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull metrics saved to: {args.output}")

    if args.save:
        print("\nMetrics saved to each model's summary.json under 'error_metrics' key")


if __name__ == '__main__':
    main()
