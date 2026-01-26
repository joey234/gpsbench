#!/usr/bin/env python3
"""
Analyze prediction files to identify samples with missing responses.
"""

import csv
import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_predictions_csv(csv_path):
    """Analyze a predictions CSV file for missing responses."""
    missing = []
    total = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            total += 1
            prediction = row.get('prediction', '').strip()

            # Check if prediction is missing, empty, or null
            if not prediction or prediction.lower() in ['null', 'none', 'nan', '']:
                missing.append({
                    'row': idx + 2,  # +2 because CSV has header and is 1-indexed
                    'lat': row.get('lat', 'N/A'),
                    'lon': row.get('lon', 'N/A'),
                    'prediction': prediction
                })

    return {
        'total': total,
        'missing': missing,
        'missing_count': len(missing)
    }

def main():
    # Find all predictions.csv files
    results_dir = Path('geollm_results')

    if not results_dir.exists():
        print(f"Error: {results_dir} directory not found")
        return

    all_results = defaultdict(dict)

    # Scan for all prediction files
    for pred_file in results_dir.rglob('predictions.csv'):
        relative_path = pred_file.relative_to(results_dir)
        parts = list(relative_path.parts)

        # Determine task and noise level
        if len(parts) == 3:  # e.g., Average_Income/clean/predictions.csv
            task = parts[0]
            noise_level = parts[1]
        elif len(parts) == 2:  # e.g., clean/predictions.csv
            task = 'general'
            noise_level = parts[0]
        else:
            task = 'unknown'
            noise_level = 'unknown'

        # Analyze the file
        results = analyze_predictions_csv(pred_file)
        all_results[task][noise_level] = results

    # Print summary
    print("=" * 80)
    print("PREDICTION FILE ANALYSIS - MISSING RESPONSES")
    print("=" * 80)
    print()

    total_missing = 0
    total_samples = 0

    for task in sorted(all_results.keys()):
        print(f"\n{'─' * 80}")
        print(f"Task: {task}")
        print(f"{'─' * 80}")

        for noise_level in sorted(all_results[task].keys()):
            results = all_results[task][noise_level]
            total = results['total']
            missing_count = results['missing_count']
            total_missing += missing_count
            total_samples += total

            status = "✓" if missing_count == 0 else "✗"
            print(f"  {status} {noise_level:15s} | Total: {total:4d} | Missing: {missing_count:4d} | Rate: {missing_count/total*100 if total > 0 else 0:5.1f}%")

            # Show details of missing samples (limit to first 5)
            if missing_count > 0 and results['missing']:
                print(f"      Missing samples:")
                for sample in results['missing'][:5]:
                    print(f"        Row {sample['row']:4d}: lat={sample['lat']}, lon={sample['lon']}, prediction='{sample['prediction']}'")
                if missing_count > 5:
                    print(f"        ... and {missing_count - 5} more")

    # Overall summary
    print()
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total samples analyzed: {total_samples}")
    print(f"Total missing responses: {total_missing}")
    print(f"Overall missing rate: {total_missing/total_samples*100 if total_samples > 0 else 0:.2f}%")
    print()

    if total_missing == 0:
        print("✓ All samples have predictions!")
    else:
        print(f"✗ Found {total_missing} samples with missing predictions")

if __name__ == '__main__':
    main()
