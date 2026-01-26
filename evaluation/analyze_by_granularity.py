#!/usr/bin/env python3
"""
Analyze place_association evaluation results by granularity level

This script breaks down accuracy metrics by granularity level (city, landmark, region, etc.)
to provide detailed insights into model performance at different geographic scales.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List


def analyze_results_by_granularity(results_file: str) -> Dict[str, Any]:
    """
    Analyze evaluation results and break down by granularity level

    Args:
        results_file: Path to evaluation results JSON file

    Returns:
        Dictionary with granularity-level breakdowns
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Group results by granularity level
    by_granularity = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'exact_match': 0,
        'city_match': 0,
        'country_match': 0,
        'match_types': defaultdict(int)
    })

    # Process each result
    for result in results.get('results', []):
        granularity = result.get('granularity_level', 'unknown')

        by_granularity[granularity]['total'] += 1

        if result.get('correct'):
            by_granularity[granularity]['correct'] += 1

        # Track match type
        match_type = result.get('match_type', 'unknown')
        by_granularity[granularity]['match_types'][match_type] += 1

        # Track hierarchical matches
        if result.get('exact_location_match'):
            by_granularity[granularity]['exact_match'] += 1
        if result.get('city_match'):
            by_granularity[granularity]['city_match'] += 1
        if result.get('country_match'):
            by_granularity[granularity]['country_match'] += 1

    # Compute accuracy for each granularity level
    granularity_analysis = {}

    for granularity, stats in by_granularity.items():
        total = stats['total']
        if total == 0:
            continue

        granularity_analysis[granularity] = {
            'total_samples': total,
            'correct': stats['correct'],
            'accuracy': stats['correct'] / total * 100,
            'exact_match_rate': stats['exact_match'] / total * 100,
            'city_match_rate': stats['city_match'] / total * 100,
            'country_match_rate': stats['country_match'] / total * 100,
            'match_type_breakdown': dict(stats['match_types'])
        }

    # Overall statistics
    overall = results.get('overall', {})

    return {
        'overall': overall,
        'by_granularity': granularity_analysis,
        'model': results.get('model', 'unknown'),
        'timestamp': results.get('timestamp', 'unknown')
    }


def print_analysis(analysis: Dict[str, Any]):
    """Print formatted analysis results"""

    print("="*80)
    print("PLACE ASSOCIATION - GRANULARITY LEVEL ANALYSIS")
    print("="*80)
    print(f"\nModel: {analysis['model']}")
    print(f"Timestamp: {analysis['timestamp']}")

    # Overall stats
    if analysis['overall']:
        print(f"\nOverall Accuracy: {analysis['overall'].get('accuracy', 0):.2f}%")
        print(f"Total Samples: {analysis['overall'].get('total', 0)}")

    # By granularity level
    print("\n" + "="*80)
    print("BREAKDOWN BY GRANULARITY LEVEL")
    print("="*80)

    # Sort by total samples (descending)
    sorted_levels = sorted(
        analysis['by_granularity'].items(),
        key=lambda x: x[1]['total_samples'],
        reverse=True
    )

    for granularity, stats in sorted_levels:
        print(f"\n{granularity.upper()} ({stats['total_samples']} samples)")
        print("-" * 60)
        print(f"  Accuracy:           {stats['accuracy']:.2f}%")
        print(f"  Exact Match Rate:   {stats['exact_match_rate']:.2f}%")
        print(f"  City Match Rate:    {stats['city_match_rate']:.2f}%")
        print(f"  Country Match Rate: {stats['country_match_rate']:.2f}%")

        # Match type breakdown
        print(f"\n  Match Type Breakdown:")
        for match_type, count in sorted(stats['match_type_breakdown'].items(), key=lambda x: x[1], reverse=True):
            pct = count / stats['total_samples'] * 100
            print(f"    {match_type:15s}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze place_association results by granularity level'
    )
    parser.add_argument(
        'results_file',
        type=str,
        help='Path to evaluation results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional output file for analysis (JSON)'
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"Error: File not found: {args.results_file}")
        return 1

    # Analyze results
    analysis = analyze_results_by_granularity(args.results_file)

    # Print to console
    print_analysis(analysis)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✓ Analysis saved to: {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
