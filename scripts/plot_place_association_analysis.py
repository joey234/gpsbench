#!/usr/bin/env python3
"""
Generate plots for place association analysis:
1. Accuracy by noise level
2. Accuracy by geographic granularity
3. Downstream GPS augmentation results
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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


def get_noise_level_data():
    """Collect accuracy by noise level for all models."""
    results_dir = Path('results')
    noise_order = ['clean', 'gps_10m', 'gps_50m', 'gps_100m', 'gps_500m', 'gps_1000m']

    all_results = []

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name.startswith('.'):
            continue

        task_file = d / 'task_results' / 'applied_task1_place_association.json'
        if not task_file.exists():
            continue

        model_name = d.name.replace('gpsbench_', '').rsplit('_', 2)[0]
        # Clean up model names
        model_name = model_name.replace('_', '-').replace('anthropic-', '').replace('mistralai-', '')

        # Fix swapped names in results files
        if 'gpt-5.1' in model_name.lower():
            model_name = model_name.lower().replace('gpt-5.1', 'GPT-5-mini')
        elif 'gpt-5-mini' in model_name.lower():
            model_name = model_name.lower().replace('gpt-5-mini', 'GPT-5.1')

        with open(task_file) as f:
            data = json.load(f)

        results = data.get('results', [])

        by_noise = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            noise = r['ground_truth'].get('noise_type', 'unknown')
            by_noise[noise]['total'] += 1
            if r['correct']:
                by_noise[noise]['correct'] += 1

        accs = {}
        for noise in noise_order:
            if by_noise[noise]['total'] > 0:
                accs[noise] = by_noise[noise]['correct'] / by_noise[noise]['total'] * 100
            else:
                accs[noise] = 0

        total_correct = sum(by_noise[n]['correct'] for n in noise_order)
        total = sum(by_noise[n]['total'] for n in noise_order)
        overall = total_correct / total * 100 if total > 0 else 0

        all_results.append({
            'model': model_name,
            'accs': accs,
            'overall': overall
        })

    # Sort by model family order
    all_results.sort(key=lambda x: get_model_sort_key(x['model']))
    return all_results, noise_order


def get_granularity_data():
    """Collect accuracy by granularity level for all models."""
    # Load augmented test data
    with open('data/track_applied/splits/task1_place_association_test.json') as f:
        task_data = json.load(f)

    coord_to_hierarchy = {}
    for sample in task_data:
        coord_str = sample.get('coordinate_string', '')
        gt = sample['ground_truth']
        coord_to_hierarchy[coord_str] = {
            'city': gt.get('location_name', ''),
            'admin1': gt.get('admin1_name', '') or '',
            'country': gt.get('country_name', '') or '',
            'region': gt.get('region', '') or '',
        }

    results_dir = Path('results')
    all_results = []

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name.startswith('.'):
            continue

        task_file = d / 'task_results' / 'applied_task1_place_association.json'
        if not task_file.exists():
            continue

        model_name = d.name.replace('gpsbench_', '').rsplit('_', 2)[0]
        model_name = model_name.replace('_', '-').replace('anthropic-', '').replace('mistralai-', '')

        # Fix swapped names in results files
        if 'gpt-5.1' in model_name.lower():
            model_name = model_name.lower().replace('gpt-5.1', 'GPT-5-mini')
        elif 'gpt-5-mini' in model_name.lower():
            model_name = model_name.lower().replace('gpt-5-mini', 'GPT-5.1')

        with open(task_file) as f:
            data = json.load(f)

        results = data.get('results', [])

        city_match = 0
        admin1_match = 0
        country_match = 0
        region_match = 0
        total = 0

        for r in results:
            prompt = r.get('prompt', '')
            coord_match = re.search(r'(-?\d+\.\d+),?\s*(-?\d+\.\d+)', prompt)
            if not coord_match:
                continue

            coord_str = f'{coord_match.group(1)}, {coord_match.group(2)}'
            hierarchy = coord_to_hierarchy.get(coord_str)
            if not hierarchy:
                coord_str = f'{coord_match.group(1)},{coord_match.group(2)}'
                hierarchy = coord_to_hierarchy.get(coord_str)
            if not hierarchy:
                continue

            total += 1
            response = r.get('response', '').lower()

            if r['correct']:
                city_match += 1

            admin1 = hierarchy['admin1'].lower()
            if admin1 and len(admin1) > 3 and admin1 in response:
                admin1_match += 1

            country = hierarchy['country'].lower()
            if country and len(country) > 3 and country in response:
                country_match += 1

            region = hierarchy['region'].lower()
            if region and len(region) > 3 and region in response:
                region_match += 1

        if total > 0:
            all_results.append({
                'model': model_name,
                'city': city_match / total * 100,
                'province': admin1_match / total * 100,
                'country': country_match / total * 100,
                'region': region_match / total * 100,
            })

    # Sort by model family order
    all_results.sort(key=lambda x: get_model_sort_key(x['model']))
    return all_results


def plot_noise_level():
    """Plot accuracy by noise level."""
    data, noise_order = get_noise_level_data()

    # Select top models for clarity
    top_models = data[:6]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(noise_order))
    width = 0.12
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))

    noise_labels = ['Clean', '10m', '50m', '100m', '500m', '1km']

    for i, model_data in enumerate(top_models):
        accs = [model_data['accs'][n] for n in noise_order]
        offset = (i - len(top_models)/2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=model_data['model'], color=colors[i])

    ax.set_xlabel('Coordinate Noise Level (σ)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Place Association Accuracy by Coordinate Noise Level')
    ax.set_xticks(x)
    ax.set_xticklabels(noise_labels)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 35)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'place_association_noise.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'place_association_noise.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/place_association_noise.pdf')
    plt.close()


def plot_granularity():
    """Plot accuracy by geographic granularity."""
    data = get_granularity_data()

    # Select top models
    top_models = data[:8]

    fig, ax = plt.subplots(figsize=(10, 5))

    levels = ['city', 'province', 'country', 'region']
    level_labels = ['City\n(exact)', 'Province', 'Country', 'Region']
    x = np.arange(len(levels))
    width = 0.1
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))

    for i, model_data in enumerate(top_models):
        accs = [model_data[l] for l in levels]
        offset = (i - len(top_models)/2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=model_data['model'], color=colors[i])

    ax.set_xlabel('Geographic Granularity Level')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Place Association: Accuracy by Geographic Granularity')
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.legend(loc='upper left', framealpha=0.9, ncol=2)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'place_association_granularity.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'place_association_granularity.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/place_association_granularity.pdf')
    plt.close()


def plot_granularity_line():
    """Plot granularity as line chart showing degradation."""
    data = get_granularity_data()

    # Use all models
    all_models = data

    fig, ax = plt.subplots(figsize=(10, 6))

    levels = ['country', 'province', 'city']
    level_labels = ['Country', 'Province/State', 'City (exact)']
    x = np.arange(len(levels))

    # Color by model family
    family_colors = {
        'GPT': '#1f77b4',
        'Gemini': '#ff7f0e',
        'Claude': '#2ca02c',
        'Mistral': '#d62728',
        'Qwen': '#9467bd',
    }
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P', '<', '>', '8', 'd']

    def get_family(name):
        name_lower = name.lower()
        for family in family_colors:
            if family.lower() in name_lower:
                return family
        return 'Other'

    for i, model_data in enumerate(all_models):
        accs = [model_data[l] for l in levels]
        family = get_family(model_data['model'])
        color = family_colors.get(family, '#7f7f7f')
        ax.plot(x, accs, marker=markers[i % len(markers)], markersize=7, linewidth=1.5,
                label=model_data['model'], color=color, alpha=0.8)

    ax.set_xlabel('Geographic Granularity Level')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Hierarchical Degradation in Geographic Knowledge (All 14 Models)')
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'place_association_granularity_line.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'place_association_granularity_line.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/place_association_granularity_line.pdf')
    plt.close()


def plot_downstream_augmentation():
    """Plot downstream GPS augmentation results."""
    # Data from analysis
    mapeval_data = {
        'categories': ['Trip\nPlanning', 'POI\nQueries', 'Overall'],
        'base': [90.9, 72.5, 75.8],
        'augmented': [100.0, 76.5, 81.8],
    }

    # Only include bias types with non-zero scores
    hierarchical_data = {
        'categories': ['Hierarchical\nBias', 'Alignment\nBias', 'Overall'],
        'base': [70.0, 50.0, 40.9],
        'augmented': [100.0, 100.0, 63.6],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # MapEval plot
    ax1 = axes[0]
    x1 = np.arange(len(mapeval_data['categories']))
    width = 0.35

    bars1 = ax1.bar(x1 - width/2, mapeval_data['base'], width, label='Base', color='#4472C4')
    bars2 = ax1.bar(x1 + width/2, mapeval_data['augmented'], width, label='+GPS', color='#ED7D31')

    ax1.set_xlabel('Category')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('MapEval')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(mapeval_data['categories'])
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 115)

    # Add delta labels
    for i, (b, a) in enumerate(zip(mapeval_data['base'], mapeval_data['augmented'])):
        delta = a - b
        if delta != 0:
            ax1.annotate(f'+{delta:.1f}%', xy=(x1[i] + width/2, a + 2),
                        ha='center', va='bottom', fontsize=12, color='#ED7D31', fontweight='bold')

    # Hierarchical Spatial plot
    ax2 = axes[1]
    x2 = np.arange(len(hierarchical_data['categories']))

    bars3 = ax2.bar(x2 - width/2, hierarchical_data['base'], width, label='Base', color='#4472C4')
    bars4 = ax2.bar(x2 + width/2, hierarchical_data['augmented'], width, label='+GPS', color='#ED7D31')

    ax2.set_xlabel('Bias Type')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Hierarchical Spatial')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(hierarchical_data['categories'])
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 115)

    # Add delta labels
    for i, (b, a) in enumerate(zip(hierarchical_data['base'], hierarchical_data['augmented'])):
        delta = a - b
        if delta > 0:
            ax2.annotate(f'+{delta:.0f}%', xy=(x2[i] + width/2, a + 2),
                        ha='center', va='bottom', fontsize=12, color='#ED7D31', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'downstream_gps_augmentation.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'downstream_gps_augmentation.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/downstream_gps_augmentation.pdf')
    plt.close()


def plot_downstream_combined():
    """Single combined plot showing improvement from GPS augmentation."""
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = ['MapEval\nTrip', 'MapEval\nPOI', 'MapEval\nOverall',
                  'Hier.\nHierarchical', 'Hier.\nAlignment', 'Hier.\nOverall']
    base = [90.9, 72.5, 75.8, 70.0, 50.0, 40.9]
    augmented = [100.0, 76.5, 81.8, 100.0, 100.0, 63.6]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, base, width, label='Base', color='#4472C4', edgecolor='white')
    bars2 = ax.bar(x + width/2, augmented, width, label='+GPS Coordinates', color='#ED7D31', edgecolor='white')

    # Add improvement arrows/labels
    for i, (b, a) in enumerate(zip(base, augmented)):
        delta = a - b
        if delta > 0:
            ax.annotate(f'+{delta:.0f}%', xy=(x[i] + width/2, a + 2),
                       ha='center', va='bottom', fontsize=12, fontweight='bold', color='#C65911')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Effect of GPS Coordinate Augmentation on Downstream Tasks')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 115)

    # Add vertical line to separate datasets
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(1, 108, 'MapEval', ha='center', fontsize=13, style='italic')
    ax.text(4.5, 108, 'Hierarchical Spatial', ha='center', fontsize=13, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'downstream_gps_combined.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'downstream_gps_combined.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/downstream_gps_combined.pdf')
    plt.close()


def get_noise_by_granularity_data():
    """Collect accuracy by noise level AND granularity level for all models."""
    # Load augmented test data
    with open('data/track_applied/splits/task1_place_association_test.json') as f:
        task_data = json.load(f)

    # Build lookup: coord_string -> (hierarchy info, noise_type)
    coord_to_info = {}
    for sample in task_data:
        coord_str = sample.get('coordinate_string', '')
        gt = sample['ground_truth']
        coord_to_info[coord_str] = {
            'city': gt.get('location_name', ''),
            'admin1': gt.get('admin1_name', '') or '',
            'country': gt.get('country_name', '') or '',
            'noise_type': gt.get('noise_type', 'unknown'),
        }

    results_dir = Path('results')
    noise_order = ['clean', 'gps_10m', 'gps_50m', 'gps_100m', 'gps_500m', 'gps_1000m']
    granularity_levels = ['city', 'province', 'country']

    all_results = []

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name.startswith('.'):
            continue

        task_file = d / 'task_results' / 'applied_task1_place_association.json'
        if not task_file.exists():
            continue

        model_name = d.name.replace('gpsbench_', '').rsplit('_', 2)[0]
        model_name = model_name.replace('_', '-').replace('anthropic-', '').replace('mistralai-', '')

        # Fix swapped names in results files
        if 'gpt-5.1' in model_name.lower():
            model_name = model_name.lower().replace('gpt-5.1', 'GPT-5-mini')
        elif 'gpt-5-mini' in model_name.lower():
            model_name = model_name.lower().replace('gpt-5-mini', 'GPT-5.1')

        with open(task_file) as f:
            data = json.load(f)

        results = data.get('results', [])

        # Track: noise_type -> granularity -> {correct, total}
        by_noise_gran = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

        for r in results:
            prompt = r.get('prompt', '')
            coord_match = re.search(r'(-?\d+\.\d+),?\s*(-?\d+\.\d+)', prompt)
            if not coord_match:
                continue

            coord_str = f'{coord_match.group(1)}, {coord_match.group(2)}'
            info = coord_to_info.get(coord_str)
            if not info:
                coord_str = f'{coord_match.group(1)},{coord_match.group(2)}'
                info = coord_to_info.get(coord_str)
            if not info:
                continue

            noise_type = info['noise_type']
            response = r.get('response', '').lower()

            # City level (exact match from evaluation)
            by_noise_gran[noise_type]['city']['total'] += 1
            if r['correct']:
                by_noise_gran[noise_type]['city']['correct'] += 1

            # Province level
            by_noise_gran[noise_type]['province']['total'] += 1
            admin1 = info['admin1'].lower()
            if admin1 and len(admin1) > 3 and admin1 in response:
                by_noise_gran[noise_type]['province']['correct'] += 1

            # Country level
            by_noise_gran[noise_type]['country']['total'] += 1
            country = info['country'].lower()
            if country and len(country) > 3 and country in response:
                by_noise_gran[noise_type]['country']['correct'] += 1

        # Compute accuracies
        model_data = {'model': model_name}
        for noise in noise_order:
            model_data[noise] = {}
            for gran in granularity_levels:
                stats = by_noise_gran[noise][gran]
                if stats['total'] > 0:
                    model_data[noise][gran] = stats['correct'] / stats['total'] * 100
                else:
                    model_data[noise][gran] = 0

        all_results.append(model_data)

    all_results.sort(key=lambda x: get_model_sort_key(x['model']))
    return all_results, noise_order, granularity_levels


def plot_noise_by_granularity():
    """Plot noise effects at each granularity level."""
    data, noise_order, gran_levels = get_noise_by_granularity_data()

    if not data:
        print("No data found for noise by granularity analysis")
        return

    noise_labels = ['Clean', '10m', '50m', '100m', '500m', '1km']
    gran_labels = {'city': 'City', 'province': 'Province', 'country': 'Country'}

    # Compute mean accuracy across models for each noise × granularity
    mean_by_noise_gran = defaultdict(dict)
    for noise in noise_order:
        for gran in gran_levels:
            vals = [m[noise][gran] for m in data if noise in m and gran in m[noise]]
            mean_by_noise_gran[noise][gran] = np.mean(vals) if vals else 0

    # Create line plot: x=noise level, y=accuracy, one line per granularity
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(noise_order))
    colors = {'country': '#2ecc71', 'province': '#f39c12', 'city': '#e74c3c'}
    markers = {'country': 's', 'province': '^', 'city': 'o'}

    for gran in ['country', 'province', 'city']:
        accs = [mean_by_noise_gran[noise][gran] for noise in noise_order]
        ax.plot(x, accs, marker=markers[gran], linewidth=2, markersize=8,
                label=gran_labels[gran], color=colors[gran])

    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Effect of Coordinate Noise by Granularity Level')
    ax.set_xticks(x)
    ax.set_xticklabels(noise_labels)
    ax.legend(loc='right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'place_association_noise_by_granularity.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'place_association_noise_by_granularity.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {OUTPUT_DIR}/place_association_noise_by_granularity.pdf')
    plt.close()

    # Also print the data for the paper
    print("\nNoise × Granularity Mean Accuracy (%):")
    print(f"{'Noise':<10} {'Country':>10} {'Province':>10} {'City':>10}")
    print("-" * 42)
    for i, noise in enumerate(noise_order):
        print(f"{noise_labels[i]:<10} {mean_by_noise_gran[noise]['country']:>10.1f} "
              f"{mean_by_noise_gran[noise]['province']:>10.1f} {mean_by_noise_gran[noise]['city']:>10.1f}")

    # Compute deltas from clean baseline
    print("\nDelta from Clean Baseline:")
    print(f"{'Noise':<10} {'Country':>10} {'Province':>10} {'City':>10}")
    print("-" * 42)
    clean_country = mean_by_noise_gran['clean']['country']
    clean_province = mean_by_noise_gran['clean']['province']
    clean_city = mean_by_noise_gran['clean']['city']
    for i, noise in enumerate(noise_order):
        if noise == 'clean':
            continue
        d_country = mean_by_noise_gran[noise]['country'] - clean_country
        d_province = mean_by_noise_gran[noise]['province'] - clean_province
        d_city = mean_by_noise_gran[noise]['city'] - clean_city
        print(f"{noise_labels[i]:<10} {d_country:>+10.1f} {d_province:>+10.1f} {d_city:>+10.1f}")


def main():
    print("Generating place association analysis plots...")
    print()

    print("1. Plotting accuracy by noise level...")
    plot_noise_level()

    print("2. Plotting accuracy by granularity (bar chart)...")
    plot_granularity()

    print("3. Plotting accuracy by granularity (line chart)...")
    plot_granularity_line()

    print("4. Plotting downstream GPS augmentation results...")
    plot_downstream_augmentation()

    print("5. Plotting combined downstream results...")
    plot_downstream_combined()

    print("6. Plotting noise by granularity analysis...")
    plot_noise_by_granularity()

    print()
    print("Done! All plots saved to paper/figures/")


if __name__ == "__main__":
    main()
