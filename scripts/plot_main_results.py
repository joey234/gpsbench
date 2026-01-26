#!/usr/bin/env python3
"""
Generate plots for main GPSBench results:
1. Overall accuracy by model (Applied vs Pure GPS tracks)
2. Per-task accuracy heatmap for Applied track
3. Per-task accuracy heatmap for Pure GPS track
4. Task-level comparison bar chart
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
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = Path('paper/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define consistent model family order
MODEL_FAMILY_ORDER = [
    'GPT-5.1', 'GPT-5-mini', 'GPT-5-nano', 'GPT-4.1', 'GPT-4.1-mini',  # GPT-5.1 is flagship
    'Gemini-2.5-Pro', 'Gemini-2.5-Flash',
    'Claude-Haiku-4.5',
    'Mistral-Large', 'Mistral-Small-24B',
    'Qwen3-235B', 'Qwen3-30B', 'Qwen3-14b', 'Qwen3-8b',
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


def plot_overall_accuracy(models):
    """Plot overall accuracy comparison between tracks."""
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = [m['name'] for m in models]
    applied_acc = [m['applied_overall'] for m in models]
    pure_gps_acc = [m.get('pure_gps_overall', 0) for m in models]

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, applied_acc, width, label='Applied Track', color='#4472C4')
    bars2 = ax.bar(x + width/2, pure_gps_acc, width, label='Pure GPS Track', color='#ED7D31')

    ax.set_xlabel('Model')
    ax.set_ylabel('Overall Accuracy (%)')
    ax.set_title('GPSBench: Overall Model Performance by Track')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'main_results_overall.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'main_results_overall.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/main_results_overall.pdf')
    plt.close()


def plot_applied_heatmap(models):
    """Plot heatmap of per-task accuracy for Applied track."""
    # Get task names (use short versions)
    # Note: Missing Data excluded from benchmark (used for RQ2 memorization analysis)
    task_names_full = [t for t in models[0]['applied_tasks'].keys() if 'missing data' not in t.lower()]
    task_names = [
        'Place Assoc.', 'Name Disamb.', 'Rel. Position', 'Proximity',
        'Route Analysis', 'Boundary', 'Spatial Pat.', 'Terrain'
    ]

    model_names = [m['name'] for m in models]

    # Build matrix
    data = []
    for m in models:
        row = [m['applied_tasks'].get(t, 0) for t in task_names_full]
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=task_names, yticklabels=model_names,
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})

    ax.set_title('GPSBench Applied Track: Per-Task Accuracy (%)')
    ax.set_xlabel('Task')
    ax.set_ylabel('Model')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'main_results_applied_heatmap.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'main_results_applied_heatmap.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/main_results_applied_heatmap.pdf')
    plt.close()


def plot_pure_gps_heatmap(models):
    """Plot heatmap of per-task accuracy for Pure GPS track."""
    # Filter models with pure GPS results
    models_with_pure = [m for m in models if 'pure_gps_tasks' in m]

    task_names_full = list(models_with_pure[0]['pure_gps_tasks'].keys())
    task_names = [
        'Format Conv.', 'Coord. Trans.', 'Distance', 'Bearing',
        'Interpolation', 'Polygon Area', 'Bounding Box', 'Route Geom.', 'Rel. Position'
    ]

    model_names = [m['name'] for m in models_with_pure]

    # Build matrix
    data = []
    for m in models_with_pure:
        row = [m['pure_gps_tasks'].get(t, 0) for t in task_names_full]
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=task_names, yticklabels=model_names,
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})

    ax.set_title('GPSBench Pure GPS Track: Per-Task Accuracy (%)')
    ax.set_xlabel('Task')
    ax.set_ylabel('Model')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'main_results_pure_gps_heatmap.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'main_results_pure_gps_heatmap.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/main_results_pure_gps_heatmap.pdf')
    plt.close()


def plot_task_difficulty_average(models):
    """Plot average accuracy across all models for each task."""
    models_with_both = [m for m in models if 'pure_gps_tasks' in m]

    # Applied track
    applied_task_names_full = [t for t in models_with_both[0]['applied_tasks'].keys() if 'missing data' not in t.lower()]
    applied_task_names = [
        'Place\nAssoc.', 'Name\nDisamb.', 'Rel.\nPosition', 'Proximity',
        'Route\nAnalysis', 'Boundary', 'Spatial\nPat.', 'Terrain'
    ]

    # Pure GPS track
    pure_gps_task_names_full = list(models_with_both[0]['pure_gps_tasks'].keys())
    pure_gps_task_names = [
        'Format\nConv.', 'Coord.\nTrans.', 'Distance', 'Bearing',
        'Interp.', 'Polygon\nArea', 'Bounding\nBox', 'Route\nGeom.', 'Rel.\nPosition'
    ]

    # Compute averages and std
    applied_accs = {t: [] for t in applied_task_names_full}
    pure_gps_accs = {t: [] for t in pure_gps_task_names_full}

    for m in models_with_both:
        for t in applied_task_names_full:
            applied_accs[t].append(m['applied_tasks'].get(t, 0))
        for t in pure_gps_task_names_full:
            pure_gps_accs[t].append(m['pure_gps_tasks'].get(t, 0))

    applied_means = [np.mean(applied_accs[t]) for t in applied_task_names_full]
    applied_stds = [np.std(applied_accs[t]) for t in applied_task_names_full]
    pure_gps_means = [np.mean(pure_gps_accs[t]) for t in pure_gps_task_names_full]
    pure_gps_stds = [np.std(pure_gps_accs[t]) for t in pure_gps_task_names_full]

    # Combine all tasks
    all_task_names = applied_task_names + pure_gps_task_names
    all_means = applied_means + pure_gps_means
    all_stds = applied_stds + pure_gps_stds
    all_tracks = ['Applied'] * len(applied_task_names) + ['Pure GPS'] * len(pure_gps_task_names)

    # Sort by mean accuracy
    sorted_indices = np.argsort(all_means)[::-1]
    all_task_names = [all_task_names[i] for i in sorted_indices]
    all_means = [all_means[i] for i in sorted_indices]
    all_stds = [all_stds[i] for i in sorted_indices]
    all_tracks = [all_tracks[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(all_task_names))
    colors = ['#4472C4' if t == 'Applied' else '#ED7D31' for t in all_tracks]

    bars = ax.bar(x, all_means, yerr=all_stds, capsize=3, color=colors, edgecolor='black', linewidth=0.5)

    # Add difficulty tier lines
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=25, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(len(x)-0.5, 97, 'Solved (>95%)', ha='right', fontsize=9, color='green', fontweight='bold')
    ax.text(len(x)-0.5, 27, 'Unsolved (<25%)', ha='right', fontsize=9, color='red', fontweight='bold')

    ax.set_xlabel('Task', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Task Difficulty: Mean Accuracy Across 14 LLMs (±1 std)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_task_names, fontsize=9)
    ax.set_ylim(0, 110)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4472C4', label='Applied Track'),
                       Patch(facecolor='#ED7D31', label='Pure GPS Track')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'task_difficulty_average.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'task_difficulty_average.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/task_difficulty_average.pdf')
    plt.close()


def plot_combined_heatmap(models):
    """Plot combined heatmap with Applied and Pure GPS tracks as subfigures."""
    # Filter models with both tracks
    models_with_both = [m for m in models if 'pure_gps_tasks' in m]

    # Applied track data
    applied_task_names_full = [t for t in models_with_both[0]['applied_tasks'].keys() if 'missing data' not in t.lower()]
    applied_task_names = [
        'Place\nAssoc.', 'Name\nDisamb.', 'Rel.\nPosition', 'Proximity',
        'Route\nAnalysis', 'Boundary', 'Spatial\nPat.', 'Terrain'
    ]

    # Pure GPS track data
    pure_gps_task_names_full = list(models_with_both[0]['pure_gps_tasks'].keys())
    pure_gps_task_names = [
        'Format\nConv.', 'Coord.\nTrans.', 'Distance', 'Bearing',
        'Interp.', 'Polygon\nArea', 'Bounding\nBox', 'Route\nGeom.', 'Rel.\nPosition'
    ]

    model_names = [m['name'] for m in models_with_both]

    # Build matrices
    applied_data = []
    pure_gps_data = []
    for m in models_with_both:
        applied_row = [m['applied_tasks'].get(t, 0) for t in applied_task_names_full]
        pure_gps_row = [m['pure_gps_tasks'].get(t, 0) for t in pure_gps_task_names_full]
        applied_data.append(applied_row)
        pure_gps_data.append(pure_gps_row)

    applied_data = np.array(applied_data)
    pure_gps_data = np.array(pure_gps_data)

    # Create figure with two subfigures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [8, 9]})

    # Applied track heatmap
    sns.heatmap(applied_data, annot=True, fmt='.0f', cmap='RdYlGn',
                xticklabels=applied_task_names, yticklabels=model_names,
                vmin=0, vmax=100, ax=ax1, cbar=False,
                annot_kws={'size': 9})
    ax1.set_title('(a) Applied Track', fontsize=12, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.tick_params(axis='x', labelsize=9)
    ax1.tick_params(axis='y', labelsize=10)

    # Pure GPS track heatmap
    sns.heatmap(pure_gps_data, annot=True, fmt='.0f', cmap='RdYlGn',
                xticklabels=pure_gps_task_names, yticklabels=[],
                vmin=0, vmax=100, ax=ax2, cbar_kws={'label': 'Accuracy (%)'},
                annot_kws={'size': 9})
    ax2.set_title('(b) Pure GPS Track', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_task_heatmap_combined.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'per_task_heatmap_combined.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/per_task_heatmap_combined.pdf')
    plt.close()


def plot_task_comparison(models):
    """Plot bar chart comparing top models across tasks."""
    # Select top 5 models
    top_models = models[:5]

    # Note: Missing Data excluded from benchmark (used for RQ2 memorization analysis)
    # Use consistent task list to avoid key ordering issues
    task_names_full = [t for t in top_models[0]['applied_tasks'].keys() if 'missing data' not in t.lower()]
    task_names = [
        'Place\nAssoc.', 'Name\nDisamb.', 'Rel.\nPosition', 'Proximity',
        'Route\nAnalysis', 'Boundary', 'Spatial\nPat.', 'Terrain'
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(task_names))
    width = 0.15
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))

    for i, m in enumerate(top_models):
        # Filter out missing_data for each model
        accs = [m['applied_tasks'].get(t, 0) for t in task_names_full if 'missing data' not in t.lower()]
        offset = (i - len(top_models)/2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=m['name'], color=colors[i])

    ax.set_xlabel('Task')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('GPSBench Applied Track: Top 5 Models by Task')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'main_results_task_comparison.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'main_results_task_comparison.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/main_results_task_comparison.pdf')
    plt.close()


def plot_radar_chart(models):
    """Plot radar chart for top models on Applied track."""
    from math import pi

    top_models = models[:5]

    # Note: Missing Data excluded from benchmark (used for RQ2 memorization analysis)
    task_names_full = [t for t in top_models[0]['applied_tasks'].keys() if 'missing data' not in t.lower()]
    task_names = [
        'Place Assoc.', 'Name Disamb.', 'Rel. Position', 'Proximity',
        'Route', 'Boundary', 'Spatial Pat.', 'Terrain'
    ]

    # Number of variables
    N = len(task_names)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))

    for i, m in enumerate(top_models):
        values = [m['applied_tasks'].get(t, 0) for t in task_names_full]
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-', linewidth=2, label=m['name'], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(task_names, size=9)
    ax.set_ylim(0, 100)
    ax.set_title('GPSBench Applied Track: Model Comparison', size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'main_results_radar.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'main_results_radar.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/main_results_radar.pdf')
    plt.close()


def plot_track_comparison_scatter(models):
    """Scatter plot comparing Applied vs Pure GPS performance."""
    fig, ax = plt.subplots(figsize=(12, 9))

    models_with_both = [m for m in models if 'pure_gps_overall' in m]

    applied = [m['applied_overall'] for m in models_with_both]
    pure_gps = [m['pure_gps_overall'] for m in models_with_both]
    names = [m['name'] for m in models_with_both]

    # Color by model family
    family_colors = {
        'GPT': '#1f77b4',
        'Gemini': '#ff7f0e',
        'Claude': '#2ca02c',
        'Mistral': '#d62728',
        'Qwen': '#9467bd',
    }

    def get_family(name):
        for family in family_colors:
            if family in name:
                return family
        return 'Other'

    for i, (a, p, n) in enumerate(zip(applied, pure_gps, names)):
        family = get_family(n)
        color = family_colors.get(family, '#7f7f7f')
        ax.scatter(a, p, s=200, c=color, edgecolors='black', linewidths=0.5, zorder=3)

    # Add diagonal line
    ax.plot([30, 95], [30, 95], 'k--', alpha=0.3, linewidth=1.5)
    ax.text(82, 84, 'y=x', fontsize=9, alpha=0.5)

    ax.set_xlabel('Applied Track Accuracy (%)', fontsize=11)
    ax.set_ylabel('Pure GPS Track Accuracy (%)', fontsize=11)
    ax.set_title('GPSBench: Applied vs Pure GPS Track Performance', fontsize=12)
    ax.set_xlim(42, 80)
    ax.set_ylim(32, 90)
    ax.grid(True, alpha=0.3)

    # Add annotations with full model names and smart positioning
    # Carefully tuned offsets to avoid all overlaps
    label_offsets = {
        'GPT-5.1': (10, 3),
        'GPT-5-mini': (10, 3),
        'GPT-5-nano': (10, -8),
        'GPT-4.1': (10, -10),
        'GPT-4.1-mini': (-90, 8),
        'Gemini-2.5-Pro': (-100, 5),
        'Gemini-2.5-Flash': (10, -8),
        'Claude-Haiku-4.5': (10, 5),
        'Mistral-Large': (-90, -10),
        'Mistral-Small-24B': (-115, 5),
        'Qwen3-235B': (10, 5),
        'Qwen3-30B': (-80, -12),
        'Qwen3-14b': (10, 5),
        'Qwen3-8b': (-70, -10),
    }

    for a, p, n in zip(applied, pure_gps, names):
        offset = label_offsets.get(n, (10, 0))
        ax.annotate(n, (a, p), xytext=offset,
                   textcoords='offset points', fontsize=9,
                   alpha=0.9,
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5) if abs(offset[0]) > 50 else None)

    # Add legend for model families
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10, label=family)
                       for family, color in family_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Model Family')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'main_results_scatter.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'main_results_scatter.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/main_results_scatter.pdf')
    plt.close()


def main():
    print("Generating main results plots...")
    print()

    models = load_all_results()
    print(f"Loaded {len(models)} models")
    print()

    print("1. Plotting overall accuracy comparison...")
    plot_overall_accuracy(models)

    print("2. Plotting Applied track heatmap...")
    plot_applied_heatmap(models)

    print("3. Plotting Pure GPS track heatmap...")
    plot_pure_gps_heatmap(models)

    print("3b. Plotting combined heatmap...")
    plot_combined_heatmap(models)

    print("3c. Plotting task difficulty average...")
    plot_task_difficulty_average(models)

    print("4. Plotting task comparison (top 5 models)...")
    plot_task_comparison(models)

    print("5. Plotting radar chart...")
    plot_radar_chart(models)

    print("6. Plotting track comparison scatter...")
    plot_track_comparison_scatter(models)

    print()
    print("Done! All plots saved to paper/figures/")


if __name__ == "__main__":
    main()
