#!/usr/bin/env python3
"""
Analyze GPSBench result files to identify samples with missing responses.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_task_results(task_file_path):
    """Analyze a single task result file for missing responses."""
    try:
        with open(task_file_path, 'r') as f:
            data = json.load(f)

        missing_responses = []
        total_samples = len(data.get('results', []))

        for result in data.get('results', []):
            response = result.get('response', '').strip()
            example_id = result.get('example_id', 'unknown')

            # Check if response is missing, empty, or null
            if not response or response.lower() in ['null', 'none', 'nan']:
                missing_responses.append({
                    'example_id': example_id,
                    'prompt': result.get('prompt', '')[:100] + '...',  # First 100 chars
                    'response': response,
                    'ground_truth': result.get('ground_truth', {})
                })

        return {
            'task_name': data.get('task_name', 'Unknown'),
            'task_file': data.get('task_file', 'Unknown'),
            'total_samples': total_samples,
            'missing_count': len(missing_responses),
            'missing_responses': missing_responses
        }
    except Exception as e:
        return {
            'error': str(e),
            'file': str(task_file_path)
        }

def main():
    results_dir = Path('results')

    if not results_dir.exists():
        print(f"Error: {results_dir} directory not found")
        return

    # Find all result directories
    result_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    print("=" * 100)
    print("GPSBench RESULTS ANALYSIS - MISSING RESPONSES")
    print("=" * 100)
    print()

    overall_stats = {
        'total_models': 0,
        'total_samples': 0,
        'total_missing': 0,
        'models_with_missing': []
    }

    for result_dir in sorted(result_dirs):
        model_name = result_dir.name
        task_results_dir = result_dir / 'task_results'

        if not task_results_dir.exists():
            continue

        print(f"\n{'─' * 100}")
        print(f"Model: {model_name}")
        print(f"{'─' * 100}")

        model_total_samples = 0
        model_total_missing = 0
        tasks_with_missing = []

        # Analyze each task result file
        task_files = sorted(task_results_dir.glob('*.json'))

        for task_file in task_files:
            results = analyze_task_results(task_file)

            if 'error' in results:
                print(f"  ✗ Error reading {task_file.name}: {results['error']}")
                continue

            total = results['total_samples']
            missing = results['missing_count']
            model_total_samples += total
            model_total_missing += missing

            status = "✓" if missing == 0 else "✗"
            task_type = "Pure GPS" if "pure_gps" in task_file.name else "Applied"

            print(f"  {status} [{task_type:10s}] {results['task_name']:40s} | Total: {total:4d} | Missing: {missing:4d}")

            if missing > 0:
                tasks_with_missing.append({
                    'task_name': results['task_name'],
                    'task_file': task_file.name,
                    'missing_count': missing,
                    'missing_responses': results['missing_responses']
                })

        # Model summary
        print()
        if model_total_missing == 0:
            print(f"  ✓ Model Summary: {model_total_samples} samples, 0 missing (100% complete)")
        else:
            missing_rate = (model_total_missing / model_total_samples * 100) if model_total_samples > 0 else 0
            print(f"  ✗ Model Summary: {model_total_samples} samples, {model_total_missing} missing ({missing_rate:.2f}% missing)")
            overall_stats['models_with_missing'].append({
                'model': model_name,
                'missing_count': model_total_missing,
                'tasks': tasks_with_missing
            })

        overall_stats['total_models'] += 1
        overall_stats['total_samples'] += model_total_samples
        overall_stats['total_missing'] += model_total_missing

    # Overall summary
    print()
    print("=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    print(f"Total models analyzed: {overall_stats['total_models']}")
    print(f"Total samples across all models: {overall_stats['total_samples']}")
    print(f"Total missing responses: {overall_stats['total_missing']}")

    if overall_stats['total_samples'] > 0:
        missing_rate = overall_stats['total_missing'] / overall_stats['total_samples'] * 100
        print(f"Overall missing rate: {missing_rate:.2f}%")

    print()

    if overall_stats['total_missing'] == 0:
        print("✓ All samples across all models have responses!")
    else:
        print(f"✗ Found {overall_stats['total_missing']} samples with missing responses")
        print()
        print("Models with missing responses:")
        for model_info in overall_stats['models_with_missing']:
            print(f"\n  • {model_info['model']} ({model_info['missing_count']} missing)")
            for task in model_info['tasks'][:3]:  # Show first 3 tasks
                print(f"    - {task['task_name']}: {task['missing_count']} missing")
                # Show first missing sample
                if task['missing_responses']:
                    sample = task['missing_responses'][0]
                    print(f"      Example ID {sample['example_id']}: response='{sample['response']}'")

if __name__ == '__main__':
    main()
