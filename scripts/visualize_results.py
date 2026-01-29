#!/usr/bin/env python3
"""
GPSBench Results Visualization - Radar Charts

Includes both accuracy-based and error-metrics-based visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import math
import argparse


def load_results(results_dir: str = "results"):
    """Load all results from the results directory"""
    results = []
    results_path = Path(results_dir)

    for item in results_path.iterdir():
        if item.name.startswith('.') or item.name == 'README.md':
            continue

        if item.is_dir():
            summary_file = item / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    data['_source'] = item.name
                    results.append(data)

    return results


def get_model_name(result):
    """Extract clean model name"""
    model = result.get('model', 'unknown')
    if '/' in model:
        model = model.split('/')[-1]
    # Shorten long names
    if len(model) > 25:
        model = model[:22] + '...'
    return model


def get_error_metric_score(result, task_name: str, tolerance: str = '5%') -> float:
    """
    Get error-metric-based accuracy score for a task.
    Returns accuracy at specified tolerance level, or None if not available.
    """
    if 'error_metrics' not in result:
        return None

    error_metrics = result['error_metrics']
    if task_name not in error_metrics:
        return None

    task_metrics = error_metrics[task_name]
    metrics = task_metrics.get('metrics', {})
    accuracy_at = metrics.get('accuracy_at', {})

    return accuracy_at.get(tolerance, None)


def create_radar_chart_by_track(results, track: str, output_file: str, min_samples: int = 5000):
    """Create radar chart for a specific track"""

    # Filter results with enough samples
    filtered_results = [r for r in results if r.get('overall', {}).get('total_cases', 0) >= min_samples]

    # Get all tasks for this track
    tasks = set()
    for result in filtered_results:
        if 'tracks' in result and track in result['tracks']:
            track_data = result['tracks'][track]
            tasks_list = track_data.get('tasks_summary') or track_data.get('tasks', [])
            for task in tasks_list:
                tasks.add(task['task_name'])

    tasks = sorted(tasks)
    num_tasks = len(tasks)

    if num_tasks == 0:
        print(f"No tasks found for track: {track}")
        return

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

    # Colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_results)))

    # Plot each model
    for idx, result in enumerate(filtered_results):
        model_name = get_model_name(result)

        if 'tracks' not in result or track not in result['tracks']:
            continue

        track_data = result['tracks'][track]
        tasks_list = track_data.get('tasks_summary') or track_data.get('tasks', [])

        # Get accuracy for each task
        task_accuracies = {}
        for task in tasks_list:
            task_accuracies[task['task_name']] = task.get('accuracy', 0)

        # Create values list in same order as tasks
        values = [task_accuracies.get(task, 0) for task in tasks]
        values += values[:1]  # Complete the loop

        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, size=12)

    # Set y-axis
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=11)

    # Title and legend
    track_name = track.replace('_', ' ').title()
    ax.set_title(f'GPSBench {track_name} Track - Model Comparison\n(Accuracy %)',
                 size=14, fontweight='bold', pad=20)

    # Place legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_overall_radar_chart(results, output_file: str, min_samples: int = 5000):
    """Create radar chart showing overall performance by track"""

    # Filter results with enough samples
    filtered_results = [r for r in results if r.get('overall', {}).get('total_cases', 0) >= min_samples]

    # Categories: Pure GPS, Applied, and maybe subcategories
    categories = ['Pure GPS\nOverall', 'Applied\nOverall']

    # Add key task categories
    key_tasks = [
        ('pure_gps', 'Distance Calculation'),
        ('pure_gps', 'Bearing Computation'),
        ('pure_gps', 'Format Conversion'),
        ('applied', 'Place Association'),
        ('applied', 'Proximity & Nearest Neighbor'),
        ('applied', 'Route Analysis'),
    ]

    for track, task in key_tasks:
        categories.append(task.replace(' ', '\n'))

    num_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(14, 11), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_results)))

    for idx, result in enumerate(filtered_results):
        model_name = get_model_name(result)

        values = []

        # Overall track accuracies
        for track in ['pure_gps', 'applied']:
            if 'tracks' in result and track in result['tracks']:
                values.append(result['tracks'][track].get('overall_accuracy', 0))
            else:
                values.append(0)

        # Key task accuracies
        for track, task_name in key_tasks:
            acc = 0
            if 'tracks' in result and track in result['tracks']:
                track_data = result['tracks'][track]
                tasks_list = track_data.get('tasks_summary') or track_data.get('tasks', [])
                for task in tasks_list:
                    if task['task_name'] == task_name:
                        acc = task.get('accuracy', 0)
                        break
            values.append(acc)

        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx], markersize=6)
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=11)

    ax.set_title('GPSBench Overall Model Comparison\n(Key Metrics)',
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_combined_radar_chart(results, output_file: str, min_samples: int = 5000):
    """Create side-by-side radar charts for both tracks"""

    filtered_results = [r for r in results if r.get('overall', {}).get('total_cases', 0) >= min_samples]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_results)))

    for ax_idx, track in enumerate(['pure_gps', 'applied']):
        ax = axes[ax_idx]

        # Get tasks for this track
        tasks = set()
        for result in filtered_results:
            if 'tracks' in result and track in result['tracks']:
                track_data = result['tracks'][track]
                tasks_list = track_data.get('tasks_summary') or track_data.get('tasks', [])
                for task in tasks_list:
                    tasks.add(task['task_name'])

        tasks = sorted(tasks)
        num_tasks = len(tasks)

        if num_tasks == 0:
            continue

        angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
        angles += angles[:1]

        for idx, result in enumerate(filtered_results):
            model_name = get_model_name(result)

            if 'tracks' not in result or track not in result['tracks']:
                continue

            track_data = result['tracks'][track]
            tasks_list = track_data.get('tasks_summary') or track_data.get('tasks', [])

            task_accuracies = {}
            for task in tasks_list:
                task_accuracies[task['task_name']] = task.get('accuracy', 0)

            values = [task_accuracies.get(task, 0) for task in tasks]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=1.5, label=model_name, color=colors[idx], markersize=4)
            ax.fill(angles, values, alpha=0.05, color=colors[idx])

        # Shorten task names for display
        short_tasks = []
        for t in tasks:
            if len(t) > 15:
                short_tasks.append(t[:12] + '...')
            else:
                short_tasks.append(t)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_tasks, size=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=10)

        track_name = track.replace('_', ' ').title()
        ax.set_title(f'{track_name} Track', size=15, fontweight='bold', pad=15)

    # Single legend for both
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('GPSBench Model Comparison by Track', size=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_error_metrics_radar(results, output_file: str, min_samples: int = 5000):
    """
    Create radar chart specifically for error metrics tasks.
    Shows accuracy at different tolerance levels for numerical tasks.
    """
    filtered_results = [r for r in results if r.get('overall', {}).get('total_cases', 0) >= min_samples]

    # Define tasks and their error metric names (must match keys in error_metrics)
    error_tasks = [
        ('Distance Calculation', '1%'),
        ('Distance Calculation', '5%'),
        ('Bearing Computation', '1%'),
        ('Bearing Computation', '5%'),
        ('Polygon Area', '5%'),
        ('Polygon Area', '20%'),
    ]

    # Create category labels
    categories = [f"{task}\n@{tol}" for task, tol in error_tasks]
    num_cats = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_results)))

    for idx, result in enumerate(filtered_results):
        model_name = get_model_name(result)

        if 'error_metrics' not in result:
            continue

        values = []
        for task_name, tolerance in error_tasks:
            score = get_error_metric_score(result, task_name, tolerance)
            values.append(score if score is not None else 0)

        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx], markersize=6)
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=11)

    ax.set_title('GPSBench Error Metrics - Accuracy at Tolerance\n(Numerical Tasks: Distance, Bearing, Area)',
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_error_metrics_comparison(results, output_file: str, min_samples: int = 5000):
    """
    Create bar chart comparing MAE and MAPE across models for each numerical task.
    """
    filtered_results = [r for r in results if r.get('overall', {}).get('total_cases', 0) >= min_samples]

    tasks = ['Distance Calculation', 'Bearing Computation', 'Polygon Area']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax_idx, task_name in enumerate(tasks):
        ax = axes[ax_idx]

        # Collect data for this task
        model_data = []
        for result in filtered_results:
            if 'error_metrics' not in result or task_name not in result['error_metrics']:
                continue

            model_name = get_model_name(result)
            metrics = result['error_metrics'][task_name].get('metrics', {})
            mape = metrics.get('mape', 0)

            model_data.append({
                'model': model_name,
                'mape': min(mape, 200)  # Cap at 200% for visualization
            })

        if not model_data:
            continue

        # Sort by MAPE (lower is better)
        model_data.sort(key=lambda x: x['mape'])

        models = [d['model'] for d in model_data]
        mapes = [d['mape'] for d in model_data]

        # Color bars by performance
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))

        bars = ax.barh(models, mapes, color=colors)
        ax.set_xlabel('MAPE (%)')
        ax.set_title(task_name, fontweight='bold')
        ax.set_xlim(0, max(mapes) * 1.1)

        # Add value labels
        for bar, mape in zip(bars, mapes):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{mape:.1f}%', va='center', fontsize=11)

    fig.suptitle('Mean Absolute Percentage Error (MAPE) by Task\n(Lower is Better)',
                 size=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_comprehensive_radar(results, output_file: str, min_samples: int = 5000,
                               use_error_metrics: bool = True):
    """
    Create comprehensive radar chart that uses error metrics for numerical tasks
    and regular accuracy for other tasks.
    """
    filtered_results = [r for r in results if r.get('overall', {}).get('total_cases', 0) >= min_samples]

    # Define all metrics to show
    # Format: (display_name, track, task_name, error_metric_task, tolerance)
    # If error_metric_task is not None, use error metrics instead of binary accuracy
    metrics_config = [
        ('Format\nConversion', 'pure_gps', 'Format Conversion', None, None),
        ('Coord\nTransform', 'pure_gps', 'Coordinate System Transformation', None, None),
        ('Distance\n@5%', 'pure_gps', 'Distance Calculation', 'Distance Calculation', '5%'),
        ('Bearing\n@5%', 'pure_gps', 'Bearing Computation', 'Bearing Computation', '5%'),
        ('Area\n@20%', 'pure_gps', 'Polygon Area', 'Polygon Area', '20%'),
        ('Bounding\nBox', 'pure_gps', 'Bounding Box', None, None),
        ('Place\nAssociation', 'applied', 'Place Association', None, None),
        ('Proximity', 'applied', 'Proximity & Nearest Neighbor', None, None),
        ('Route\nAnalysis', 'applied', 'Route Analysis', None, None),
    ]

    categories = [m[0] for m in metrics_config]
    num_cats = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(14, 11), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_results)))

    for idx, result in enumerate(filtered_results):
        model_name = get_model_name(result)

        values = []
        for display, track, task_name, error_task, tolerance in metrics_config:
            score = 0

            # Try error metrics first if configured
            if use_error_metrics and error_task and tolerance:
                err_score = get_error_metric_score(result, error_task, tolerance)
                if err_score is not None:
                    score = err_score
                    values.append(score)
                    continue

            # Fall back to binary accuracy
            if 'tracks' in result and track in result['tracks']:
                track_data = result['tracks'][track]
                tasks_list = track_data.get('tasks_summary') or track_data.get('tasks', [])
                for task in tasks_list:
                    if task['task_name'] == task_name:
                        score = task.get('accuracy', 0)
                        break

            values.append(score)

        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx], markersize=6)
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=11)

    title = 'GPSBench Comprehensive Model Comparison'
    if use_error_metrics:
        title += '\n(Numerical tasks use accuracy@tolerance from error metrics)'
    ax.set_title(title, size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate GPSBench radar chart visualizations')
    parser.add_argument('--no-error-metrics', action='store_true',
                        help='Do not use error metrics for numerical tasks')
    parser.add_argument('--min-samples', type=int, default=5000,
                        help='Minimum samples for model inclusion (default: 5000)')
    args = parser.parse_args()

    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} result files")

    # Filter to models with full evaluations
    full_results = [r for r in results if r.get('overall', {}).get('total_cases', 0) >= args.min_samples]
    print(f"Models with full evaluations (>={args.min_samples} samples): {len(full_results)}")

    for r in full_results:
        model = get_model_name(r)
        total = r.get('overall', {}).get('total_cases', 0)
        has_error = 'error_metrics' in r
        print(f"  - {model}: {total} samples {'(has error_metrics)' if has_error else ''}")

    print("\nGenerating radar charts...")

    # Create individual track charts
    create_radar_chart_by_track(results, 'pure_gps', 'radar_pure_gps.png', args.min_samples)
    create_radar_chart_by_track(results, 'applied', 'radar_applied.png', args.min_samples)

    # Create overall comparison (original, without error metrics)
    create_overall_radar_chart(results, 'radar_overall.png', args.min_samples)

    # Create combined side-by-side
    create_combined_radar_chart(results, 'radar_combined.png', args.min_samples)

    # Create error metrics specific charts
    create_error_metrics_radar(results, 'radar_error_metrics.png', args.min_samples)
    create_error_metrics_comparison(results, 'chart_mape_comparison.png', args.min_samples)

    # Create comprehensive chart with error metrics integrated
    create_comprehensive_radar(results, 'radar_comprehensive.png', args.min_samples,
                               use_error_metrics=not args.no_error_metrics)

    print("\nDone! Generated:")
    print("  - radar_pure_gps.png       (Pure GPS track by task)")
    print("  - radar_applied.png        (Applied track by task)")
    print("  - radar_overall.png        (Key metrics overview)")
    print("  - radar_combined.png       (Side-by-side comparison)")
    print("  - radar_error_metrics.png  (Error metrics: accuracy@tolerance)")
    print("  - chart_mape_comparison.png (MAPE bar chart comparison)")
    print("  - radar_comprehensive.png  (All metrics with error metrics integrated)")


if __name__ == '__main__':
    main()
