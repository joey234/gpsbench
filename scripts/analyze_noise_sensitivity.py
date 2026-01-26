#!/usr/bin/env python3
"""
Analyze noise sensitivity evaluation results.

This script analyzes how model performance degrades with increasing GPS noise
and generates comprehensive reports and visualizations.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import statistics


def load_evaluation_results(results_file: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    with open(results_file) as f:
        return json.load(f)


def analyze_by_noise_level(results: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Analyze results grouped by noise level

    Returns:
        Dictionary mapping noise_level -> statistics
    """
    noise_groups = defaultdict(list)

    # Group results by noise level
    for result in results.get('results', []):
        test_case = result.get('test_case', {})
        noise_level = test_case.get('noise_level_meters')

        if noise_level is not None:
            noise_groups[noise_level].append(result)

    # Calculate statistics for each noise level
    stats_by_level = {}
    for noise_level, group_results in sorted(noise_groups.items()):
        correct = sum(1 for r in group_results if r.get('correct', False))
        total = len(group_results)
        accuracy = (correct / total * 100) if total > 0 else 0

        # Calculate average actual noise
        actual_noises = [
            r.get('test_case', {}).get('actual_noise_meters', 0)
            for r in group_results
        ]
        avg_actual_noise = statistics.mean(actual_noises) if actual_noises else 0

        stats_by_level[noise_level] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'avg_actual_noise_meters': round(avg_actual_noise, 2),
            'errors': total - correct
        }

    return stats_by_level


def calculate_degradation_metrics(stats_by_level: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate performance degradation metrics"""
    sorted_levels = sorted(stats_by_level.keys())

    if not sorted_levels:
        return {}

    baseline_accuracy = stats_by_level[sorted_levels[0]]['accuracy']

    degradation = {
        'baseline_noise_meters': sorted_levels[0],
        'baseline_accuracy': baseline_accuracy,
        'degradation_by_level': {}
    }

    for level in sorted_levels:
        current_accuracy = stats_by_level[level]['accuracy']
        degradation['degradation_by_level'][level] = {
            'accuracy': current_accuracy,
            'absolute_drop': baseline_accuracy - current_accuracy,
            'relative_drop_percent': ((baseline_accuracy - current_accuracy) / baseline_accuracy * 100)
                                    if baseline_accuracy > 0 else 0
        }

    return degradation


def identify_critical_thresholds(stats_by_level: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Identify critical noise thresholds where performance drops significantly

    Returns:
        Dictionary with threshold information
    """
    sorted_levels = sorted(stats_by_level.keys())

    thresholds = {
        '90_percent': None,  # Noise level where accuracy drops below 90%
        '75_percent': None,  # Noise level where accuracy drops below 75%
        '50_percent': None,  # Noise level where accuracy drops below 50%
        '25_percent': None,  # Noise level where accuracy drops below 25%
    }

    for level in sorted_levels:
        accuracy = stats_by_level[level]['accuracy']

        if thresholds['90_percent'] is None and accuracy < 90:
            thresholds['90_percent'] = level
        if thresholds['75_percent'] is None and accuracy < 75:
            thresholds['75_percent'] = level
        if thresholds['50_percent'] is None and accuracy < 50:
            thresholds['50_percent'] = level
        if thresholds['25_percent'] is None and accuracy < 25:
            thresholds['25_percent'] = level

    return thresholds


def assess_robustness(
    stats_by_level: Dict[int, Dict[str, Any]],
    thresholds: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assess overall robustness based on performance degradation

    Returns:
        Robustness assessment with score and interpretation
    """
    sorted_levels = sorted(stats_by_level.keys())

    # Calculate robustness score (0-100)
    # Based on: weighted average of accuracy across noise levels
    weights = {
        5: 0.05,    # Very low noise - less important
        10: 0.10,   # Low noise
        25: 0.15,   # Medium noise
        50: 0.20,   # High noise - most important
        100: 0.20,  # Very high noise - most important
        250: 0.15,  # Extreme noise
        500: 0.10,  # Critical noise
        1000: 0.05  # Severe degradation - less important
    }

    weighted_accuracy = 0
    for level in sorted_levels:
        weight = weights.get(level, 1.0 / len(sorted_levels))
        weighted_accuracy += stats_by_level[level]['accuracy'] * weight

    robustness_score = weighted_accuracy

    # Determine robustness level
    if robustness_score >= 80:
        level = "Excellent"
        interpretation = "Model shows robust understanding of geographic locations with high noise tolerance"
    elif robustness_score >= 60:
        level = "Good"
        interpretation = "Model maintains reasonable performance under noise, shows some geographic understanding"
    elif robustness_score >= 40:
        level = "Fair"
        interpretation = "Model shows moderate noise sensitivity, may rely partially on memorization"
    elif robustness_score >= 20:
        level = "Poor"
        interpretation = "Model is highly sensitive to noise, likely relies heavily on memorization"
    else:
        level = "Very Poor"
        interpretation = "Model shows brittle behavior, almost entirely relies on exact coordinate memorization"

    return {
        'robustness_score': round(robustness_score, 2),
        'robustness_level': level,
        'interpretation': interpretation,
        'critical_threshold_50pct': thresholds.get('50_percent'),
        'maintains_90pct_up_to': thresholds.get('90_percent') or max(sorted_levels)
    }


def generate_text_report(
    model_name: str,
    stats_by_level: Dict[int, Dict[str, Any]],
    degradation: Dict[str, Any],
    thresholds: Dict[str, Any],
    robustness: Dict[str, Any]
) -> str:
    """Generate human-readable text report"""
    lines = []
    lines.append("=" * 80)
    lines.append("NOISE SENSITIVITY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"\nModel: {model_name}")
    lines.append(f"Analysis Type: Progressive GPS Noise Tolerance")
    lines.append("")

    # Overall robustness
    lines.append("─" * 80)
    lines.append("OVERALL ROBUSTNESS ASSESSMENT")
    lines.append("─" * 80)
    lines.append(f"Robustness Score: {robustness['robustness_score']:.1f}/100")
    lines.append(f"Robustness Level: {robustness['robustness_level']}")
    lines.append(f"Interpretation: {robustness['interpretation']}")
    lines.append("")

    # Critical thresholds
    lines.append("─" * 80)
    lines.append("CRITICAL NOISE THRESHOLDS")
    lines.append("─" * 80)
    lines.append(f"Maintains >90% accuracy up to: {thresholds.get('90_percent', 'N/A')} meters")
    lines.append(f"Drops below 75% accuracy at:   {thresholds.get('75_percent', 'N/A')} meters")
    lines.append(f"Drops below 50% accuracy at:   {thresholds.get('50_percent', 'N/A')} meters")
    lines.append(f"Drops below 25% accuracy at:   {thresholds.get('25_percent', 'N/A')} meters")
    lines.append("")

    # Performance by noise level
    lines.append("─" * 80)
    lines.append("PERFORMANCE BY NOISE LEVEL")
    lines.append("─" * 80)
    lines.append(f"{'Noise':<12} {'Accuracy':<12} {'Correct':<12} {'Total':<12} {'Degradation':<15}")
    lines.append(f"{'(meters)':<12} {'(%)':<12} {'Tests':<12} {'Tests':<12} {'vs Baseline':<15}")
    lines.append("─" * 80)

    for level in sorted(stats_by_level.keys()):
        stats = stats_by_level[level]
        deg = degradation['degradation_by_level'].get(level, {})

        noise_str = f"{level}m"
        accuracy_str = f"{stats['accuracy']:.1f}%"
        correct_str = f"{stats['correct']}/{stats['total']}"
        total_str = f"{stats['total']}"
        deg_str = f"{deg.get('absolute_drop', 0):.1f}%"

        lines.append(f"{noise_str:<12} {accuracy_str:<12} {correct_str:<12} {total_str:<12} {deg_str:<15}")

    lines.append("")

    # Interpretation guide
    lines.append("─" * 80)
    lines.append("INTERPRETATION GUIDE")
    lines.append("─" * 80)
    lines.append("Noise Levels:")
    lines.append("  5-10m:   Typical consumer GPS (smartphones)")
    lines.append("  25-50m:  Urban areas with signal degradation")
    lines.append("  100m:    Poor GPS conditions")
    lines.append("  250m+:   Severe interference or systematic errors")
    lines.append("")
    lines.append("Expected Behavior:")
    lines.append("  Robust models:  Gradual degradation, >70% accuracy at 100m")
    lines.append("  Memorization:   Sharp drop, <50% accuracy at 50m")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_csv_report(stats_by_level: Dict[int, Dict[str, Any]], output_file: Path):
    """Generate CSV report for easy plotting"""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['noise_meters', 'accuracy', 'correct', 'total', 'errors'])

        for level in sorted(stats_by_level.keys()):
            stats = stats_by_level[level]
            writer.writerow([
                level,
                f"{stats['accuracy']:.2f}",
                stats['correct'],
                stats['total'],
                stats['errors']
            ])


def main():
    parser = argparse.ArgumentParser(description='Analyze noise sensitivity results')
    parser.add_argument('results_file', type=Path, help='Path to evaluation results JSON')
    parser.add_argument('--output-dir', type=Path, help='Output directory for reports')
    parser.add_argument('--export-csv', action='store_true', help='Export CSV for plotting')

    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        return

    # Set output directory
    output_dir = args.output_dir or args.results_file.parent
    output_dir.mkdir(exist_ok=True)

    # Load results
    print(f"Loading results from {args.results_file}...")
    results = load_evaluation_results(args.results_file)

    model_name = results.get('model', 'Unknown')
    print(f"Model: {model_name}")

    # Analyze by noise level
    print("\nAnalyzing performance by noise level...")
    stats_by_level = analyze_by_noise_level(results)

    if not stats_by_level:
        print("Error: No noise sensitivity tests found in results")
        return

    # Calculate metrics
    print("Calculating degradation metrics...")
    degradation = calculate_degradation_metrics(stats_by_level)

    print("Identifying critical thresholds...")
    thresholds = identify_critical_thresholds(stats_by_level)

    print("Assessing overall robustness...")
    robustness = assess_robustness(stats_by_level, thresholds)

    # Generate reports
    print("\nGenerating reports...")

    # Text report
    text_report = generate_text_report(
        model_name,
        stats_by_level,
        degradation,
        thresholds,
        robustness
    )

    print("\n" + text_report)

    # Save text report
    text_report_file = output_dir / f"noise_sensitivity_report_{model_name.replace('/', '_')}.txt"
    with open(text_report_file, 'w') as f:
        f.write(text_report)
    print(f"\nText report saved to: {text_report_file}")

    # Save JSON report
    json_report = {
        'model': model_name,
        'analysis_type': 'noise_sensitivity',
        'stats_by_noise_level': stats_by_level,
        'degradation_analysis': degradation,
        'critical_thresholds': thresholds,
        'robustness_assessment': robustness
    }

    json_report_file = output_dir / f"noise_sensitivity_report_{model_name.replace('/', '_')}.json"
    with open(json_report_file, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"JSON report saved to: {json_report_file}")

    # Export CSV if requested
    if args.export_csv:
        csv_file = output_dir / f"noise_sensitivity_data_{model_name.replace('/', '_')}.csv"
        generate_csv_report(stats_by_level, csv_file)
        print(f"CSV data saved to: {csv_file}")


if __name__ == "__main__":
    main()
