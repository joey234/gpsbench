#!/usr/bin/env python3
"""
List all saved evaluation results
Shows a summary table of all evaluations that have been run
"""

import json
from pathlib import Path
from datetime import datetime
from tabulate import tabulate


def format_timestamp(iso_timestamp):
    """Format ISO timestamp to readable format"""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_timestamp


def get_result_summary(filepath):
    """Extract summary info from a result file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        summary = {
            'File': filepath.name,
            'Model': data.get('model', 'Unknown'),
            'Provider': data.get('provider', 'Unknown'),
            'Timestamp': format_timestamp(data.get('timestamp', 'Unknown')),
        }

        # Add overall accuracy if available
        if 'overall_metrics' in data:
            summary['Overall Accuracy'] = f"{data['overall_metrics'].get('overall_accuracy', 0):.2f}%"
            summary['Total Cases'] = data['overall_metrics'].get('total_cases', 0)

        # Count tasks evaluated
        if 'tasks' in data:
            summary['Tasks'] = len(data['tasks'])

        return summary
    except Exception as e:
        return {
            'File': filepath.name,
            'Error': str(e)
        }


def main():
    results_dir = Path('results')

    if not results_dir.exists():
        print(f"No results directory found at: {results_dir}")
        print("Run an evaluation first:")
        print("  python evaluation/evaluate_layer_a.py --max-samples 10")
        return

    result_files = sorted(results_dir.glob('eval_*.json'))

    if not result_files:
        print(f"No result files found in: {results_dir}")
        print("\nRun an evaluation to create results:")
        print("  python evaluation/evaluate_layer_a.py --max-samples 10")
        return

    print(f"\n{'='*100}")
    print(f"GPSBench Evaluation Results - {len(result_files)} file(s) found")
    print(f"{'='*100}\n")

    # Get summaries
    summaries = []
    for filepath in result_files:
        summary = get_result_summary(filepath)
        summaries.append(summary)

    # Display as table
    if summaries:
        print(tabulate(summaries, headers='keys', tablefmt='grid'))

    print(f"\n{'='*100}")
    print(f"Results directory: {results_dir.absolute()}")
    print(f"{'='*100}\n")

    print("Commands:")
    print("  View latest result:      python evaluation/analyze_results.py")
    print("  Compare all results:     python evaluation/analyze_results.py --compare")
    print("  Export to CSV:           python evaluation/analyze_results.py --compare --export-csv results.csv")
    print()


if __name__ == "__main__":
    main()
