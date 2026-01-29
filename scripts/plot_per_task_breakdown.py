#!/usr/bin/env python3
"""
Generate per-task breakdown plots for GPSBench results.
Creates individual bar charts showing all models' performance on each task.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = Path('paper/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define consistent model family order
MODEL_FAMILY_ORDER = [
    'GPT-5.1', 'GPT-5-mini', 'GPT-5-nano', 'GPT-4.1', 'GPT-4.1-mini',
    'Gemini-2.5-Pro', 'Gemini-2.5-Flash',
    'Claude-Haiku-4.5',
    'Mistral-Large', 'Mistral-Small-24B',
    'Qwen3-235B', 'Qwen3-30B',
]

def get_model_sort_key(model_name):
    """Return sort key for model family ordering."""
    try:
        return MODEL_FAMILY_ORDER.index(model_name)
    except ValueError:
        return len(MODEL_FAMILY_ORDER)  # Put unknown models at end


def load_all_results():
    """Load results for all models (latest run per unique model)."""
    results_dir = Path('results')
    models_dirs = {}  # model_name -> directory path (keep latest)

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name.startswith('.'):
            continue

        # Skip Gemini-3-Pro and VL models
        if 'gemini-3' in d.name.lower() or '-vl-' in d.name.lower():
            continue

        applied_file = d / 'track_applied.json'
        if not applied_file.exists():
            continue

        model_name = d.name.replace('gpsbench_', '').rsplit('_', 2)[0]
        model_name = model_name.replace('_', '-').replace('anthropic-', '').replace('mistralai-', '')

        # Shorten some names
        model_name = model_name.replace('-2025-08-07', '').replace('-2512', '').replace('-2501', '')
        model_name = model_name.replace('qwen-qwen3', 'Qwen3').replace('gpt-', 'GPT-')
        model_name = model_name.replace('gemini-', 'Gemini-').replace('claude-', 'Claude-')
        model_name = model_name.replace('mistral-', 'Mistral-')

        # Fix swapped names in results files
        if model_name == 'GPT-5.1':
            model_name = 'GPT-5-mini'
        elif model_name == 'GPT-5-mini':
            model_name = 'GPT-5.1'

        # Keep latest run for each model (sorted dirs are chronological)
        models_dirs[model_name] = d

    # Now load data from selected directories
    all_models = []
    for model_name, d in models_dirs.items():
        applied_file = d / 'track_applied.json'
        pure_gps_file = d / 'track_pure_gps.json'

        with open(applied_file) as f:
            applied = json.load(f)

        model_data = {
            'name': model_name,
            'applied_overall': applied['overall_accuracy'],
            'applied_tasks': {t['task_name']: t['accuracy'] for t in applied['tasks']},
        }

        if pure_gps_file.exists():
            with open(pure_gps_file) as f:
                pure = json.load(f)
            model_data['pure_gps_overall'] = pure['overall_accuracy']
            model_data['pure_gps_tasks'] = {t['task_name']: t['accuracy'] for t in pure['tasks']}

        all_models.append(model_data)

    # Sort by model family order (not by accuracy)
    all_models.sort(key=lambda x: get_model_sort_key(x['name']))
    return all_models


def plot_single_task_bar(models, task_name, track, output_name, task_display_name=None):
    """Plot bar chart for a single task showing all models."""
    fig, ax = plt.subplots(figsize=(10, 5))

    task_key = 'applied_tasks' if track == 'applied' else 'pure_gps_tasks'

    # Filter models that have this task
    models_with_task = [m for m in models if task_key in m and task_name in m[task_key]]

    if not models_with_task:
        print(f"No models found with task: {task_name}")
        plt.close()
        return

    # Sort by model family order
    models_with_task.sort(key=lambda x: get_model_sort_key(x['name']))

    model_names = [m['name'] for m in models_with_task]
    accuracies = [m[task_key][task_name] for m in models_with_task]

    # Color by accuracy level
    colors = []
    for acc in accuracies:
        if acc >= 90:
            colors.append('#2ecc71')  # Green
        elif acc >= 70:
            colors.append('#3498db')  # Blue
        elif acc >= 50:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red

    bars = ax.bar(range(len(model_names)), accuracies, color=colors, edgecolor='black', linewidth=0.5)

    display_name = task_display_name or task_name
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{display_name}')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
    ax.set_ylim(0, 105)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11)

    # Add mean line
    mean_acc = np.mean(accuracies)
    ax.axhline(y=mean_acc, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(len(model_names)-0.5, mean_acc+2, f'Mean: {mean_acc:.1f}%', fontsize=11, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{output_name}.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{output_name}.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/{output_name}.pdf')
    plt.close()


def plot_applied_tasks_grid(models):
    """Plot a 3x3 grid of all Applied track tasks."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    applied_tasks = [
        ('Place Association', 'Place Association'),
        ('Name Disambiguation', 'Name Disambiguation'),
        ('Relative Position', 'Relative Position'),
        ('Proximity & Nearest Neighbor', 'Proximity'),
        ('Route Analysis', 'Route Analysis'),
        ('Boundary Analysis', 'Boundary Analysis'),
        ('Spatial Patterns', 'Spatial Patterns'),
        ('Terrain Classification', 'Terrain Classification'),
    ]

    for idx, (task_name, display_name) in enumerate(applied_tasks):
        ax = axes[idx]

        models_with_task = [m for m in models if 'applied_tasks' in m and task_name in m['applied_tasks']]
        models_with_task.sort(key=lambda x: get_model_sort_key(x['name']))

        model_names = [m['name'].split('-')[0] for m in models_with_task]  # Shortened names
        accuracies = [m['applied_tasks'][task_name] for m in models_with_task]

        colors = []
        for acc in accuracies:
            if acc >= 90:
                colors.append('#2ecc71')
            elif acc >= 70:
                colors.append('#3498db')
            elif acc >= 50:
                colors.append('#f39c12')
            else:
                colors.append('#e74c3c')

        ax.bar(range(len(model_names)), accuracies, color=colors, edgecolor='black', linewidth=0.3)
        ax.set_title(display_name, fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_ylabel('Accuracy (%)', fontsize=11)

        mean_acc = np.mean(accuracies)
        ax.axhline(y=mean_acc, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='≥90%'),
        mpatches.Patch(facecolor='#3498db', edgecolor='black', label='70-89%'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='50-69%'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='<50%'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('GPSBench Applied Track: Per-Task Performance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_task_applied_grid.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'per_task_applied_grid.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/per_task_applied_grid.pdf')
    plt.close()


def plot_pure_gps_tasks_grid(models):
    """Plot a 3x3 grid of all Pure GPS track tasks."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    pure_gps_tasks = [
        ('Format Conversion', 'Format Conversion'),
        ('Coordinate System Transformation', 'Coord. Transform'),
        ('Distance Calculation', 'Distance Calculation'),
        ('Bearing Computation', 'Bearing Computation'),
        ('Coordinate Interpolation', 'Interpolation'),
        ('Polygon Area', 'Polygon Area'),
        ('Bounding Box', 'Bounding Box'),
        ('Route Geometry', 'Route Geometry'),
        ('Relative Position', 'Relative Position'),
    ]

    models_with_pure = [m for m in models if 'pure_gps_tasks' in m]

    for idx, (task_name, display_name) in enumerate(pure_gps_tasks):
        ax = axes[idx]

        models_with_task = [m for m in models_with_pure if task_name in m['pure_gps_tasks']]
        models_with_task.sort(key=lambda x: get_model_sort_key(x['name']))

        if not models_with_task:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(display_name, fontsize=13, fontweight='bold')
            continue

        model_names = [m['name'].split('-')[0] for m in models_with_task]
        accuracies = [m['pure_gps_tasks'][task_name] for m in models_with_task]

        colors = []
        for acc in accuracies:
            if acc >= 90:
                colors.append('#2ecc71')
            elif acc >= 70:
                colors.append('#3498db')
            elif acc >= 50:
                colors.append('#f39c12')
            else:
                colors.append('#e74c3c')

        ax.bar(range(len(model_names)), accuracies, color=colors, edgecolor='black', linewidth=0.3)
        ax.set_title(display_name, fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_ylabel('Accuracy (%)', fontsize=11)

        mean_acc = np.mean(accuracies)
        ax.axhline(y=mean_acc, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='≥90%'),
        mpatches.Patch(facecolor='#3498db', edgecolor='black', label='70-89%'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='50-69%'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='<50%'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('GPSBench Pure GPS Track: Per-Task Performance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_task_pure_gps_grid.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'per_task_pure_gps_grid.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/per_task_pure_gps_grid.pdf')
    plt.close()


def generate_task_statistics(models):
    """Generate statistics for each task."""
    print("\n" + "="*80)
    print("APPLIED TRACK - Per-Task Statistics")
    print("="*80)

    applied_tasks = [
        'Place Association',
        'Name Disambiguation',
        'Relative Position',
        'Proximity & Nearest Neighbor',
        'Route Analysis',
        'Boundary Analysis',
        'Spatial Patterns',
        'Terrain Classification',
    ]

    for task_name in applied_tasks:
        accuracies = [m['applied_tasks'][task_name] for m in models if task_name in m.get('applied_tasks', {})]
        if accuracies:
            print(f"\n{task_name}:")
            print(f"  Mean: {np.mean(accuracies):.1f}%  |  Std: {np.std(accuracies):.1f}%")
            print(f"  Min: {np.min(accuracies):.1f}%   |  Max: {np.max(accuracies):.1f}%")
            print(f"  Range: {np.max(accuracies) - np.min(accuracies):.1f}%")

    print("\n" + "="*80)
    print("PURE GPS TRACK - Per-Task Statistics")
    print("="*80)

    pure_gps_tasks = [
        'Format Conversion',
        'Coordinate System Transformation',
        'Distance Calculation',
        'Bearing Computation',
        'Coordinate Interpolation',
        'Polygon Area',
        'Bounding Box',
        'Route Geometry',
        'Relative Position',
    ]

    models_with_pure = [m for m in models if 'pure_gps_tasks' in m]

    for task_name in pure_gps_tasks:
        accuracies = [m['pure_gps_tasks'][task_name] for m in models_with_pure if task_name in m.get('pure_gps_tasks', {})]
        if accuracies:
            print(f"\n{task_name}:")
            print(f"  Mean: {np.mean(accuracies):.1f}%  |  Std: {np.std(accuracies):.1f}%")
            print(f"  Min: {np.min(accuracies):.1f}%   |  Max: {np.max(accuracies):.1f}%")
            print(f"  Range: {np.max(accuracies) - np.min(accuracies):.1f}%")


def generate_latex_table(models):
    """Generate LaTeX table with per-task breakdown."""
    print("\n" + "="*80)
    print("LATEX TABLE - Applied Track Per-Task Breakdown")
    print("="*80)

    applied_tasks = [
        ('Place Association', 'Place Assoc.'),
        ('Name Disambiguation', 'Name Dis.'),
        ('Relative Position', 'Rel. Pos.'),
        ('Proximity & Nearest Neighbor', 'Proximity'),
        ('Route Analysis', 'Route'),
        ('Boundary Analysis', 'Boundary'),
        ('Spatial Patterns', 'Spatial'),
        ('Terrain Classification', 'Terrain'),
    ]

    print("\\begin{table*}[htbp]")
    print("\\centering")
    print("\\caption{Per-task accuracy (\\%) for Applied Track. Bold indicates best performance per task.}")
    print("\\label{tab:per_task_applied}")
    print("\\small")
    print("\\begin{tabular}{l" + "c"*len(applied_tasks) + "}")
    print("\\toprule")
    header = "\\textbf{Model} & " + " & ".join([f"\\textbf{{{t[1]}}}" for t in applied_tasks]) + " \\\\"
    print(header)
    print("\\midrule")

    # Find best for each task
    best_per_task = {}
    for task_name, _ in applied_tasks:
        accs = [(m['name'], m['applied_tasks'].get(task_name, 0)) for m in models]
        best_per_task[task_name] = max(accs, key=lambda x: x[1])[1]

    for m in models:
        row = m['name']
        for task_name, _ in applied_tasks:
            acc = m['applied_tasks'].get(task_name, 0)
            if acc == best_per_task[task_name]:
                row += f" & \\textbf{{{acc:.1f}}}"
            else:
                row += f" & {acc:.1f}"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")

    print("\n" + "="*80)
    print("LATEX TABLE - Pure GPS Track Per-Task Breakdown")
    print("="*80)

    pure_gps_tasks = [
        ('Format Conversion', 'Format'),
        ('Coordinate System Transformation', 'Coord.'),
        ('Distance Calculation', 'Distance'),
        ('Bearing Computation', 'Bearing'),
        ('Coordinate Interpolation', 'Interp.'),
        ('Polygon Area', 'Polygon'),
        ('Bounding Box', 'BBox'),
        ('Route Geometry', 'Route'),
        ('Relative Position', 'Rel. Pos.'),
    ]

    models_with_pure = [m for m in models if 'pure_gps_tasks' in m]

    print("\\begin{table*}[htbp]")
    print("\\centering")
    print("\\caption{Per-task accuracy (\\%) for Pure GPS Track. Bold indicates best performance per task.}")
    print("\\label{tab:per_task_pure_gps}")
    print("\\small")
    print("\\begin{tabular}{l" + "c"*len(pure_gps_tasks) + "}")
    print("\\toprule")
    header = "\\textbf{Model} & " + " & ".join([f"\\textbf{{{t[1]}}}" for t in pure_gps_tasks]) + " \\\\"
    print(header)
    print("\\midrule")

    # Find best for each task
    best_per_task = {}
    for task_name, _ in pure_gps_tasks:
        accs = [(m['name'], m['pure_gps_tasks'].get(task_name, 0)) for m in models_with_pure]
        best_per_task[task_name] = max(accs, key=lambda x: x[1])[1]

    for m in models_with_pure:
        row = m['name']
        for task_name, _ in pure_gps_tasks:
            acc = m['pure_gps_tasks'].get(task_name, 0)
            if acc == best_per_task[task_name]:
                row += f" & \\textbf{{{acc:.1f}}}"
            else:
                row += f" & {acc:.1f}"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")


def main():
    print("Generating per-task breakdown plots...")
    print()

    models = load_all_results()
    print(f"Loaded {len(models)} models")
    print()

    print("1. Plotting Applied track tasks grid...")
    plot_applied_tasks_grid(models)

    print("2. Plotting Pure GPS track tasks grid...")
    plot_pure_gps_tasks_grid(models)

    print("3. Generating task statistics...")
    generate_task_statistics(models)

    print("\n4. Generating LaTeX tables...")
    generate_latex_table(models)

    print()
    print("Done! All plots saved to paper/figures/")


if __name__ == "__main__":
    main()
