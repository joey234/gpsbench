#!/usr/bin/env python3
"""
Generate additional plots for GPSBench paper:
1. Combined per-task heatmap (Applied + Pure GPS)
2. Model size vs accuracy scatter plot
3. Noise robustness analysis
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
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

# Model size estimates (in billions of parameters)
# For MoE models, showing total/active
MODEL_SIZES = {
    'GPT-5.1': 500,  # Estimated
    'GPT-4.1': 200,  # Estimated
    'GPT-5-mini': 70,  # Estimated (reasoning optimized)
    'GPT-4.1-mini': 70,  # Estimated
    'GPT-5-nano': 20,  # Estimated
    'Gemini-2.5-Pro': 300,  # Estimated
    'Gemini-2.5-Flash': 100,  # Estimated
    'Claude-Haiku-4.5': 70,  # Estimated
    'Qwen3-235B': 235,  # Known (22B active)
    'Qwen3-30B': 30,  # Known (3B active)
    'Mistral-Large': 123,  # Known
    'Mistral-Small-24B': 24,  # Known
}


def load_all_results():
    """Load results for all models (latest run per unique model)."""
    results_dir = Path('results')
    models_dirs = {}

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
        model_name = model_name.replace('-2025-08-07', '').replace('-2512', '').replace('-2501', '').replace('-2507', '')
        model_name = model_name.replace('-a22b', '').replace('-a3b-instruct', '').replace('-24b-instruct', '-24B')
        model_name = model_name.replace('qwen-qwen3', 'Qwen3').replace('gpt-', 'GPT-')
        model_name = model_name.replace('gemini-', 'Gemini-').replace('claude-', 'Claude-')
        model_name = model_name.replace('mistral-', 'Mistral-').replace('small', 'Small').replace('large', 'Large')

        # Fix swapped names in results files
        if model_name == 'GPT-5.1':
            model_name = 'GPT-5-mini'
        elif model_name == 'GPT-5-mini':
            model_name = 'GPT-5.1'

        models_dirs[model_name] = d

    all_models = []
    for model_name, d in models_dirs.items():
        applied_file = d / 'track_applied.json'
        pure_gps_file = d / 'track_pure_gps.json'

        with open(applied_file) as f:
            applied = json.load(f)

        model_data = {
            'name': model_name,
            'dir': d,
            'applied_overall': applied['overall_accuracy'],
            'applied_tasks': {t['task_name']: t['accuracy'] for t in applied['tasks']},
        }

        if pure_gps_file.exists():
            with open(pure_gps_file) as f:
                pure = json.load(f)
            model_data['pure_gps_overall'] = pure['overall_accuracy']
            model_data['pure_gps_tasks'] = {t['task_name']: t['accuracy'] for t in pure['tasks']}

        all_models.append(model_data)

    all_models.sort(key=lambda x: -x['applied_overall'])
    return all_models


def plot_combined_heatmap(models):
    """Plot combined heatmap for both tracks."""
    # Filter models with both tracks
    models_with_both = [m for m in models if 'pure_gps_tasks' in m]

    # Get task names
    applied_tasks_full = list(models_with_both[0]['applied_tasks'].keys())
    pure_gps_tasks_full = list(models_with_both[0]['pure_gps_tasks'].keys())

    applied_tasks_short = [
        'Place Assoc.', 'Name Disamb.', 'Rel. Pos.', 'Proximity',
        'Route Anal.', 'Boundary', 'Spatial Pat.', 'Missing Data', 'Terrain'
    ]
    pure_gps_tasks_short = [
        'Format Conv.', 'Coord. Trans.', 'Distance', 'Bearing',
        'Interpolation', 'Polygon Area', 'Bbox', 'Route Geom.', 'Rel. Pos.'
    ]

    model_names = [m['name'] for m in models_with_both]

    # Build combined matrix
    applied_data = []
    pure_gps_data = []
    for m in models_with_both:
        applied_row = [m['applied_tasks'].get(t, 0) for t in applied_tasks_full]
        pure_gps_row = [m['pure_gps_tasks'].get(t, 0) for t in pure_gps_tasks_full]
        applied_data.append(applied_row)
        pure_gps_data.append(pure_gps_row)

    applied_data = np.array(applied_data)
    pure_gps_data = np.array(pure_gps_data)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1]})

    # Applied Track heatmap
    sns.heatmap(applied_data, annot=True, fmt='.0f', cmap='RdYlGn',
                xticklabels=applied_tasks_short, yticklabels=model_names,
                vmin=0, vmax=100, ax=ax1, cbar_kws={'label': 'Accuracy (%)'})
    ax1.set_title('Applied Track', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Model')
    ax1.tick_params(axis='x', rotation=45)

    # Pure GPS Track heatmap
    sns.heatmap(pure_gps_data, annot=True, fmt='.0f', cmap='RdYlGn',
                xticklabels=pure_gps_tasks_short, yticklabels=model_names,
                vmin=0, vmax=100, ax=ax2, cbar_kws={'label': 'Accuracy (%)'})
    ax2.set_title('Pure GPS Track', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Task')
    ax2.set_ylabel('')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_task_heatmap_combined.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'per_task_heatmap_combined.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/per_task_heatmap_combined.pdf')
    plt.close()


def plot_size_vs_accuracy(models):
    """Plot model size vs accuracy scatter plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data
    sizes = []
    applied_accs = []
    pure_gps_accs = []
    names = []
    colors = []

    # Color by model family
    family_colors = {
        'GPT': '#1f77b4',
        'Gemini': '#ff7f0e',
        'Claude': '#2ca02c',
        'Qwen': '#d62728',
        'Mistral': '#9467bd',
    }

    for m in models:
        name = m['name']
        size = MODEL_SIZES.get(name)
        if size is None:
            continue

        sizes.append(size)
        applied_accs.append(m['applied_overall'])
        pure_gps_accs.append(m.get('pure_gps_overall', 0))
        names.append(name)

        # Determine family
        family = 'Other'
        for fam in family_colors:
            if fam in name:
                family = fam
                break
        colors.append(family_colors.get(family, '#7f7f7f'))

    sizes = np.array(sizes)
    applied_accs = np.array(applied_accs)
    pure_gps_accs = np.array(pure_gps_accs)

    # Plot Applied Track
    ax1.scatter(sizes, applied_accs, c=colors, s=150, alpha=0.8, edgecolors='black', linewidths=0.5)
    for i, name in enumerate(names):
        ax1.annotate(name.split('-')[0], (sizes[i], applied_accs[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=11, alpha=0.7)

    # Add trend line
    z = np.polyfit(np.log(sizes), applied_accs, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(sizes), max(sizes), 100)
    ax1.plot(x_trend, p(np.log(x_trend)), 'k--', alpha=0.3, label='Log trend')

    ax1.set_xscale('log')
    ax1.set_xlabel('Model Size (Billions of Parameters)')
    ax1.set_ylabel('Applied Track Accuracy (%)')
    ax1.set_title('Model Size vs Applied Track Performance')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(45, 85)

    # Plot Pure GPS Track
    ax2.scatter(sizes, pure_gps_accs, c=colors, s=150, alpha=0.8, edgecolors='black', linewidths=0.5)
    for i, name in enumerate(names):
        ax2.annotate(name.split('-')[0], (sizes[i], pure_gps_accs[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=11, alpha=0.7)

    # Add trend line
    z2 = np.polyfit(np.log(sizes), pure_gps_accs, 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_trend, p2(np.log(x_trend)), 'k--', alpha=0.3, label='Log trend')

    ax2.set_xscale('log')
    ax2.set_xlabel('Model Size (Billions of Parameters)')
    ax2.set_ylabel('Pure GPS Track Accuracy (%)')
    ax2.set_title('Model Size vs Pure GPS Track Performance')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(35, 90)

    # Add legend
    handles = [mpatches.Patch(color=c, label=f) for f, c in family_colors.items()]
    ax1.legend(handles=handles, loc='lower right', fontsize=11)
    ax2.legend(handles=handles, loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'size_vs_accuracy.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'size_vs_accuracy.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/size_vs_accuracy.pdf')
    plt.close()


def load_noise_data(models):
    """Load noise-level performance data from task results."""
    noise_levels = [0, 10, 50, 100, 500, 1000]

    # Load original data to get noise levels per sample
    data_file = Path('data/track_applied/task1_place_association.json')
    with open(data_file) as f:
        original_data = json.load(f)

    # Create mapping from example_id to noise level
    noise_map = {}
    for i, sample in enumerate(original_data):
        noise_sigma = sample.get('ground_truth', {}).get('noise_sigma_m', 0)
        noise_map[i] = noise_sigma

    model_noise_results = {}

    for m in models:
        model_name = m['name']
        results_dir = m['dir']
        task_file = results_dir / 'task_results' / 'applied_task1_place_association.json'

        if not task_file.exists():
            continue

        with open(task_file) as f:
            task_results = json.load(f)

        # Group by noise level
        noise_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

        for result in task_results.get('results', []):
            example_id = result.get('example_id', 0)
            correct = result.get('correct', False)

            # Get noise level from original data
            noise_level = noise_map.get(example_id, 0)

            noise_accuracy[noise_level]['total'] += 1
            if correct:
                noise_accuracy[noise_level]['correct'] += 1

        # Calculate accuracy per noise level
        model_noise_results[model_name] = {}
        for level in noise_levels:
            stats = noise_accuracy[level]
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total'] * 100
            else:
                acc = 0
            model_noise_results[model_name][level] = acc

    return model_noise_results, noise_levels


def plot_noise_robustness(models):
    """Plot noise robustness analysis."""
    noise_results, noise_levels = load_noise_data(models)

    if not noise_results:
        print("Warning: No noise data found, skipping noise_robustness plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color by model family
    family_colors = {
        'GPT': '#1f77b4',
        'Gemini': '#ff7f0e',
        'Claude': '#2ca02c',
        'Qwen': '#d62728',
        'Mistral': '#9467bd',
    }

    def get_color(name):
        for fam, color in family_colors.items():
            if fam in name:
                return color
        return '#7f7f7f'

    # Plot 1: Line plot of accuracy vs noise level
    for model_name, noise_acc in sorted(noise_results.items(), key=lambda x: -sum(x[1].values())):
        accs = [noise_acc.get(level, 0) for level in noise_levels]
        color = get_color(model_name)
        ax1.plot(noise_levels, accs, 'o-', label=model_name, color=color, linewidth=2, markersize=6)

    ax1.set_xscale('symlog', linthresh=10)
    ax1.set_xlabel('Noise Level (meters)')
    ax1.set_ylabel('Place Association Accuracy (%)')
    ax1.set_title('Accuracy vs GPS Noise Level')
    ax1.set_xticks(noise_levels)
    ax1.set_xticklabels(['0', '10', '50', '100', '500', '1000'])
    ax1.legend(loc='lower left', fontsize=11, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 70)

    # Plot 2: Heatmap of accuracy by model and noise level
    model_names = sorted(noise_results.keys(), key=lambda x: -sum(noise_results[x].values()))
    data_matrix = []
    for model_name in model_names:
        row = [noise_results[model_name].get(level, 0) for level in noise_levels]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    sns.heatmap(data_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=['0m', '10m', '50m', '100m', '500m', '1000m'],
                yticklabels=model_names,
                vmin=0, vmax=60, ax=ax2, cbar_kws={'label': 'Accuracy (%)'})
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Model')
    ax2.set_title('Place Association Accuracy by Noise Level')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'noise_robustness.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'noise_robustness.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/noise_robustness.pdf')
    plt.close()


def main():
    print("Generating additional figures...")
    print()

    models = load_all_results()
    print(f"Loaded {len(models)} models")
    print()

    print("1. Plotting combined per-task heatmap...")
    plot_combined_heatmap(models)

    print("2. Plotting size vs accuracy...")
    plot_size_vs_accuracy(models)

    print("3. Plotting noise robustness...")
    plot_noise_robustness(models)

    print()
    print("Done! All plots saved to paper/figures/")


if __name__ == "__main__":
    main()
