#!/usr/bin/env python3
"""
Generate radar (spider) plots from GPSBench evaluation results.

Usage:
    python scripts/generate_radar_plots.py
    python scripts/generate_radar_plots.py --output paper/figures/
    python scripts/generate_radar_plots.py --top 5
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['figure.dpi'] = 150
from pathlib import Path
from typing import Dict, List
from math import pi


def load_all_results(results_dir: Path) -> Dict[str, Dict]:
    """Load results from all model directories."""
    results = {}

    for model_dir in sorted(results_dir.glob("gpsbench_*")):
        summary_file = model_dir / "summary.json"
        if not summary_file.exists():
            continue

        with open(summary_file) as f:
            data = json.load(f)

        model_name = data.get('model', model_dir.name)
        # Skip duplicates (keep most recent by folder name)
        if model_name not in results:
            results[model_name] = data

    return results


def get_short_model_name(model: str) -> str:
    """Get shortened model name for display."""
    name_map = {
        'gpt-4.1': 'GPT-4.1',
        'gpt-4.1-mini': 'GPT-4.1-mini',
        'gpt-5.1': 'GPT-5.1',
        'gpt-5-mini-2025-08-07': 'GPT-5-mini',
        'gpt-5-nano-2025-08-07': 'GPT-5-nano',
        'gemini-2.5-flash': 'Gemini-2.5-Flash',
        'gemini-2.5-pro': 'Gemini-2.5-Pro',
        'gemini-3-pro-preview': 'Gemini-3-Pro',
        'anthropic/claude-haiku-4.5': 'Claude-Haiku-4.5',
        'mistralai/mistral-large-2512': 'Mistral-Large',
        'mistralai/mistral-small-24b-instruct-2501': 'Mistral-Small',
        'qwen/qwen3-235b-a22b-2507': 'Qwen3-235B',
        'qwen/qwen3-30b-a3b-instruct-2507': 'Qwen3-30B',
    }
    return name_map.get(model, model)


def get_short_task_name(task: str) -> str:
    """Get shortened task name for display."""
    name_map = {
        'Format Conversion': 'Format',
        'Coordinate System Transformation': 'Coord Trans',
        'Distance Calculation': 'Distance',
        'Bearing Computation': 'Bearing',
        'Coordinate Interpolation': 'Interpolation',
        'Polygon Area': 'Area',
        'Bounding Box': 'BBox',
        'Route Geometry': 'Route Geom',
        'Relative Position': 'Rel Position',
        'Place Association': 'Place Assoc',
        'Name Disambiguation': 'Name Disamb',
        'Proximity & Nearest Neighbor': 'Proximity',
        'Route Analysis': 'Route Anal',
        'Boundary Analysis': 'Boundary',
        'Spatial Patterns': 'Spatial Pat',
        'Missing Data Inference': 'Missing Data',
        'Terrain Classification': 'Terrain',
    }
    return name_map.get(task, task)


def create_radar_plot(models_data: List[Dict], task_names: List[str],
                      title: str, output_path: str, figsize=(10, 10)):
    """Create a radar plot comparing multiple models."""

    num_vars = len(task_names)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))

    for idx, model_info in enumerate(models_data):
        values = model_info['values']
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-', linewidth=2,
                label=model_info['name'], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    # Set the labels
    short_names = [get_short_task_name(t) for t in task_names]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_names, size=12)

    # Set y-axis
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=11)

    # Add title and legend
    ax.set_title(title, size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_track_radar(results: Dict[str, Dict], track: str,
                       output_path: str, top_n: int = 6):
    """Create radar plot for a specific track."""

    # Get task names
    task_names = []
    for data in results.values():
        tasks = data.get('tracks', {}).get(track, {}).get('tasks_summary', [])
        if tasks:
            task_names = [t['task_name'] for t in tasks]
            break

    if not task_names:
        print(f"No tasks found for track: {track}")
        return

    # Collect model data
    models_data = []
    for model, data in results.items():
        tasks = data.get('tracks', {}).get(track, {}).get('tasks_summary', [])
        task_acc = {t['task_name']: t['accuracy'] for t in tasks}
        overall = data.get('tracks', {}).get(track, {}).get('overall_accuracy', 0)

        values = [task_acc.get(t, 0) for t in task_names]
        models_data.append({
            'name': get_short_model_name(model),
            'values': values,
            'overall': overall,
        })

    # Sort by overall and take top N
    models_data.sort(key=lambda x: x['overall'], reverse=True)
    models_data = models_data[:top_n]

    title = f"{'Pure GPS' if track == 'pure_gps' else 'Applied'} Track Performance"
    create_radar_plot(models_data, task_names, title, output_path)


def create_combined_radar(results: Dict[str, Dict], output_path: str, top_n: int = 5):
    """Create radar plot with all tasks."""

    # Get all task names
    pure_gps_tasks = []
    applied_tasks = []

    for data in results.values():
        pure = data.get('tracks', {}).get('pure_gps', {}).get('tasks_summary', [])
        applied = data.get('tracks', {}).get('applied', {}).get('tasks_summary', [])
        if pure:
            pure_gps_tasks = [t['task_name'] for t in pure]
        if applied:
            applied_tasks = [t['task_name'] for t in applied]
        if pure_gps_tasks and applied_tasks:
            break

    all_tasks = pure_gps_tasks + applied_tasks

    # Collect model data
    models_data = []
    for model, data in results.items():
        pure_tasks = data.get('tracks', {}).get('pure_gps', {}).get('tasks_summary', [])
        applied_tasks_data = data.get('tracks', {}).get('applied', {}).get('tasks_summary', [])

        task_acc = {}
        for t in pure_tasks:
            task_acc[t['task_name']] = t['accuracy']
        for t in applied_tasks_data:
            task_acc[t['task_name']] = t['accuracy']

        overall = data.get('overall', {}).get('overall_accuracy', 0)
        values = [task_acc.get(t, 0) for t in all_tasks]

        models_data.append({
            'name': get_short_model_name(model),
            'values': values,
            'overall': overall,
        })

    # Sort by overall and take top N
    models_data.sort(key=lambda x: x['overall'], reverse=True)
    models_data = models_data[:top_n]

    create_radar_plot(models_data, all_tasks,
                      "GPSBench Overall Performance (Top Models)",
                      output_path, figsize=(12, 12))


def create_category_radar(results: Dict[str, Dict], output_path: str, top_n: int = 6):
    """Create radar plot by task category."""

    # Define categories
    categories = {
        'Coordinate Ops': ['Format Conversion', 'Coordinate System Transformation',
                          'Distance Calculation', 'Bearing Computation', 'Coordinate Interpolation'],
        'Spatial Analysis': ['Polygon Area', 'Bounding Box', 'Route Geometry', 'Relative Position'],
        'Geo Knowledge': ['Place Association', 'Name Disambiguation', 'Proximity & Nearest Neighbor'],
        'Complex Reasoning': ['Route Analysis', 'Boundary Analysis', 'Spatial Patterns',
                             'Missing Data Inference', 'Terrain Classification'],
    }

    category_names = list(categories.keys())

    # Collect model data
    models_data = []
    for model, data in results.items():
        # Get all task accuracies
        task_acc = {}
        for track in ['pure_gps', 'applied']:
            tasks = data.get('tracks', {}).get(track, {}).get('tasks_summary', [])
            for t in tasks:
                task_acc[t['task_name']] = t['accuracy']

        # Calculate category averages
        cat_values = []
        for cat_name, cat_tasks in categories.items():
            task_scores = [task_acc.get(t, 0) for t in cat_tasks if t in task_acc]
            avg = sum(task_scores) / len(task_scores) if task_scores else 0
            cat_values.append(avg)

        overall = data.get('overall', {}).get('overall_accuracy', 0)
        models_data.append({
            'name': get_short_model_name(model),
            'values': cat_values,
            'overall': overall,
        })

    # Sort by overall and take top N
    models_data.sort(key=lambda x: x['overall'], reverse=True)
    models_data = models_data[:top_n]

    create_radar_plot(models_data, category_names,
                      "Performance by Task Category",
                      output_path, figsize=(9, 9))


def main():
    parser = argparse.ArgumentParser(description="Generate radar plots from GPSBench results")
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing result folders')
    parser.add_argument('--output', type=str, default='paper/figures',
                        help='Output directory for plots')
    parser.add_argument('--top', type=int, default=6,
                        help='Number of top models to show (default: 6)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}...")
    results = load_all_results(results_dir)
    print(f"Found {len(results)} unique models\n")

    # Generate radar plots
    print("Generating radar plots...")

    # Pure GPS track
    create_track_radar(
        results, 'pure_gps',
        str(output_dir / 'radar_pure_gps.png'),
        top_n=args.top
    )

    # Applied track
    create_track_radar(
        results, 'applied',
        str(output_dir / 'radar_applied.png'),
        top_n=args.top
    )

    # Combined (all tasks)
    create_combined_radar(
        results,
        str(output_dir / 'radar_combined.png'),
        top_n=min(args.top, 5)  # Fewer models for readability
    )

    # By category
    create_category_radar(
        results,
        str(output_dir / 'radar_categories.png'),
        top_n=args.top
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
