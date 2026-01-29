#!/usr/bin/env python3
"""
Analyze benchmark results by geographic region.

Regions: Europe, Asia, Africa, North America, South America, Oceania
With optional subregions for Europe and Asia.
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse

# Country code to continent mapping
COUNTRY_TO_CONTINENT = {
    # Europe
    'AL': 'Europe', 'AD': 'Europe', 'AT': 'Europe', 'BY': 'Europe', 'BE': 'Europe',
    'BA': 'Europe', 'BG': 'Europe', 'HR': 'Europe', 'CY': 'Europe', 'CZ': 'Europe',
    'DK': 'Europe', 'EE': 'Europe', 'FI': 'Europe', 'FR': 'Europe', 'DE': 'Europe',
    'GR': 'Europe', 'HU': 'Europe', 'IS': 'Europe', 'IE': 'Europe', 'IT': 'Europe',
    'XK': 'Europe', 'LV': 'Europe', 'LI': 'Europe', 'LT': 'Europe', 'LU': 'Europe',
    'MT': 'Europe', 'MD': 'Europe', 'MC': 'Europe', 'ME': 'Europe', 'NL': 'Europe',
    'MK': 'Europe', 'NO': 'Europe', 'PL': 'Europe', 'PT': 'Europe', 'RO': 'Europe',
    'RU': 'Europe', 'SM': 'Europe', 'RS': 'Europe', 'SK': 'Europe', 'SI': 'Europe',
    'ES': 'Europe', 'SE': 'Europe', 'CH': 'Europe', 'UA': 'Europe', 'GB': 'Europe',
    'VA': 'Europe',
    # Asia
    'AF': 'Asia', 'AM': 'Asia', 'AZ': 'Asia', 'BH': 'Asia', 'BD': 'Asia',
    'BT': 'Asia', 'BN': 'Asia', 'KH': 'Asia', 'CN': 'Asia', 'CX': 'Asia',
    'GE': 'Asia', 'HK': 'Asia', 'IN': 'Asia', 'ID': 'Asia', 'IR': 'Asia',
    'IQ': 'Asia', 'IL': 'Asia', 'JP': 'Asia', 'JO': 'Asia', 'KZ': 'Asia',
    'KW': 'Asia', 'KG': 'Asia', 'LA': 'Asia', 'LB': 'Asia', 'MO': 'Asia',
    'MY': 'Asia', 'MV': 'Asia', 'MN': 'Asia', 'MM': 'Asia', 'NP': 'Asia',
    'KP': 'Asia', 'OM': 'Asia', 'PK': 'Asia', 'PS': 'Asia', 'PH': 'Asia',
    'QA': 'Asia', 'SA': 'Asia', 'SG': 'Asia', 'KR': 'Asia', 'LK': 'Asia',
    'SY': 'Asia', 'TW': 'Asia', 'TJ': 'Asia', 'TH': 'Asia', 'TL': 'Asia',
    'TR': 'Asia', 'TM': 'Asia', 'AE': 'Asia', 'UZ': 'Asia', 'VN': 'Asia',
    'YE': 'Asia',
    # Africa
    'DZ': 'Africa', 'AO': 'Africa', 'BJ': 'Africa', 'BW': 'Africa', 'BF': 'Africa',
    'BI': 'Africa', 'CV': 'Africa', 'CM': 'Africa', 'CF': 'Africa', 'TD': 'Africa',
    'KM': 'Africa', 'CG': 'Africa', 'CD': 'Africa', 'CI': 'Africa', 'DJ': 'Africa',
    'EG': 'Africa', 'GQ': 'Africa', 'ER': 'Africa', 'SZ': 'Africa', 'ET': 'Africa',
    'GA': 'Africa', 'GM': 'Africa', 'GH': 'Africa', 'GN': 'Africa', 'GW': 'Africa',
    'KE': 'Africa', 'LS': 'Africa', 'LR': 'Africa', 'LY': 'Africa', 'MG': 'Africa',
    'MW': 'Africa', 'ML': 'Africa', 'MR': 'Africa', 'MU': 'Africa', 'MA': 'Africa',
    'MZ': 'Africa', 'NA': 'Africa', 'NE': 'Africa', 'NG': 'Africa', 'RW': 'Africa',
    'ST': 'Africa', 'SN': 'Africa', 'SC': 'Africa', 'SL': 'Africa', 'SO': 'Africa',
    'ZA': 'Africa', 'SS': 'Africa', 'SD': 'Africa', 'TZ': 'Africa', 'TG': 'Africa',
    'TN': 'Africa', 'UG': 'Africa', 'ZM': 'Africa', 'ZW': 'Africa', 'RE': 'Africa',
    # North America
    'AG': 'North America', 'BS': 'North America', 'BB': 'North America', 'BZ': 'North America',
    'CA': 'North America', 'CR': 'North America', 'CU': 'North America', 'DM': 'North America',
    'DO': 'North America', 'SV': 'North America', 'GD': 'North America', 'GT': 'North America',
    'HT': 'North America', 'HN': 'North America', 'JM': 'North America', 'MX': 'North America',
    'NI': 'North America', 'PA': 'North America', 'KN': 'North America', 'LC': 'North America',
    'VC': 'North America', 'TT': 'North America', 'US': 'North America', 'AI': 'North America',
    'GP': 'North America', 'GU': 'North America', 'PR': 'North America', 'VI': 'North America',
    # South America
    'AR': 'South America', 'BO': 'South America', 'BR': 'South America', 'CL': 'South America',
    'CO': 'South America', 'EC': 'South America', 'GY': 'South America', 'PY': 'South America',
    'PE': 'South America', 'SR': 'South America', 'UY': 'South America', 'VE': 'South America',
    'GF': 'South America', 'FK': 'South America',
    # Oceania
    'AU': 'Oceania', 'FJ': 'Oceania', 'KI': 'Oceania', 'MH': 'Oceania', 'FM': 'Oceania',
    'NR': 'Oceania', 'NZ': 'Oceania', 'PW': 'Oceania', 'PG': 'Oceania', 'WS': 'Oceania',
    'SB': 'Oceania', 'TO': 'Oceania', 'TV': 'Oceania', 'VU': 'Oceania', 'NF': 'Oceania',
    'NC': 'Oceania', 'PF': 'Oceania',
}

# Asia subregions
ASIA_SUBREGIONS = {
    'East Asia': ['CN', 'JP', 'KR', 'KP', 'MN', 'TW', 'HK', 'MO'],
    'Southeast Asia': ['BN', 'KH', 'ID', 'LA', 'MY', 'MM', 'PH', 'SG', 'TH', 'TL', 'VN'],
    'South Asia': ['AF', 'BD', 'BT', 'IN', 'MV', 'NP', 'PK', 'LK'],
    'Central Asia': ['KZ', 'KG', 'TJ', 'TM', 'UZ'],
    'Western Asia': ['AM', 'AZ', 'BH', 'CY', 'GE', 'IR', 'IQ', 'IL', 'JO', 'KW', 'LB',
                     'OM', 'PS', 'QA', 'SA', 'SY', 'TR', 'AE', 'YE'],
}

# Europe subregions
EUROPE_SUBREGIONS = {
    'Western Europe': ['AT', 'BE', 'FR', 'DE', 'LI', 'LU', 'MC', 'NL', 'CH'],
    'Northern Europe': ['DK', 'EE', 'FI', 'IS', 'IE', 'LV', 'LT', 'NO', 'SE', 'GB'],
    'Southern Europe': ['AL', 'AD', 'BA', 'HR', 'GR', 'IT', 'XK', 'MT', 'ME', 'MK',
                        'PT', 'SM', 'RS', 'SI', 'ES', 'VA'],
    'Eastern Europe': ['BY', 'BG', 'CZ', 'HU', 'MD', 'PL', 'RO', 'RU', 'SK', 'UA'],
}


def get_continent_from_country(country_code):
    """Get continent from country code."""
    return COUNTRY_TO_CONTINENT.get(country_code, 'Unknown')


def get_subregion(country_code):
    """Get subregion for Asia/Europe countries."""
    for subregion, countries in ASIA_SUBREGIONS.items():
        if country_code in countries:
            return subregion
    for subregion, countries in EUROPE_SUBREGIONS.items():
        if country_code in countries:
            return subregion
    return None


def get_continent_from_coords(lat, lon):
    """Approximate continent from coordinates."""
    # Simple bounding box approach
    if lat is None or lon is None:
        return 'Unknown'

    # Europe (roughly)
    if 35 <= lat <= 72 and -25 <= lon <= 65:
        if lon < 40:  # Western part
            return 'Europe'

    # Asia
    if -10 <= lat <= 80 and 25 <= lon <= 180:
        return 'Asia'
    if -10 <= lat <= 55 and 60 <= lon <= 180:
        return 'Asia'

    # Africa
    if -35 <= lat <= 37 and -20 <= lon <= 55:
        return 'Africa'

    # North America
    if 5 <= lat <= 85 and -170 <= lon <= -50:
        return 'North America'

    # South America
    if -60 <= lat <= 15 and -85 <= lon <= -30:
        return 'South America'

    # Oceania
    if -50 <= lat <= 0 and 110 <= lon <= 180:
        return 'Oceania'
    if -50 <= lat <= 0 and -180 <= lon <= -130:
        return 'Oceania'

    return 'Unknown'


def extract_coords_from_task(item):
    """Extract representative coordinates from a task item."""
    # Try various coordinate sources
    if 'point_a' in item:
        p = item['point_a']
        return p.get('lat'), p.get('lon')

    if 'source_coords' in item:
        c = item['source_coords']
        return c.get('lat'), c.get('lon')

    if 'vertices' in item and item['vertices']:
        v = item['vertices'][0]
        return v.get('lat'), v.get('lon')

    if 'points' in item and item['points']:
        p = item['points'][0]
        return p.get('lat'), p.get('lon')

    # For ground_truth with coordinates
    gt = item.get('ground_truth', {})
    if 'latitude' in gt:
        return gt.get('latitude'), gt.get('longitude')
    if 'true_coordinates' in gt:
        tc = gt['true_coordinates']
        return tc.get('lat'), tc.get('lon')

    return None, None


def get_region_for_task_item(item, result_gt=None):
    """Determine region for a task item."""
    # First try country code from ground truth
    gt = result_gt or item.get('ground_truth', {})

    country_code = gt.get('country_code') or gt.get('country') or gt.get('start_country')
    if country_code:
        return get_continent_from_country(country_code)

    # Try coordinates
    lat, lon = extract_coords_from_task(item)
    if lat is not None and lon is not None:
        return get_continent_from_coords(lat, lon)

    return 'Unknown'


def load_task_data(task_file):
    """Load task data from file."""
    with open(task_file) as f:
        return json.load(f)


def analyze_results_by_region(result_file, task_data):
    """Analyze results broken down by region."""
    with open(result_file) as f:
        results = json.load(f)

    task_lookup = {i: item for i, item in enumerate(task_data)}

    by_region = defaultdict(lambda: {'correct': 0, 'total': 0})

    for result in results.get('results', []):
        example_id = result['example_id']
        is_correct = result.get('correct', False)

        # Get task item
        task_item = task_lookup.get(example_id, {})
        result_gt = result.get('ground_truth', {})

        # Determine region
        region = get_region_for_task_item(task_item, result_gt)

        by_region[region]['total'] += 1
        if is_correct:
            by_region[region]['correct'] += 1

    # Calculate accuracies
    region_stats = {}
    for region, stats in by_region.items():
        if stats['total'] > 0:
            region_stats[region] = {
                'accuracy': stats['correct'] / stats['total'] * 100,
                'correct': stats['correct'],
                'total': stats['total']
            }

    return region_stats


def get_model_name(dirname):
    """Extract clean model name from directory name."""
    name = dirname.replace('gpsbench_', '').split('_20')[0]
    if len(name) > 25:
        name = name[:22] + '...'
    return name


def get_task_display_name(task_filename):
    """Get display name for task."""
    name_map = {
        'pure_gps_task1_format_conversion': 'Format Conversion',
        'pure_gps_task3_coordinate_system': 'Coordinate Transform',
        'pure_gps_task4_distance_calculation': 'Distance Calculation',
        'pure_gps_task5_bearing_computation': 'Bearing Computation',
        'pure_gps_task6_interpolation': 'Interpolation',
        'pure_gps_task7_area_perimeter': 'Polygon Area',
        'pure_gps_task8_bounding_box': 'Bounding Box',
        'pure_gps_task9_route_geometry': 'Route Geometry',
        'pure_gps_task10_relative_position': 'Relative Position',
        'applied_task1_place_association': 'Place Association',
        'applied_task3_name_disambiguation': 'Name Disambiguation',
        'applied_task5_relative_position': 'Relative Position (App)',
        'applied_task6_proximity': 'Proximity',
        'applied_task7_route_analysis': 'Route Analysis',
        'applied_task8_boundary_analysis': 'Boundary Analysis',
        'applied_task9_spatial_patterns': 'Spatial Patterns',
        'applied_task11_missing_data': 'Missing Data',
        'applied_task13_terrain': 'Terrain Classification',
    }

    for key, display in name_map.items():
        if key in task_filename:
            return display

    return task_filename.replace('.json', '').replace('_', ' ').title()


def process_all_models(results_dir, min_samples=1000):
    """Process all models and tasks."""
    all_results = {}

    # Task file mapping
    task_files = {
        'pure_gps': list(Path('data/track_pure_gps').glob('task*.json')),
        'applied': list(Path('data/track_applied').glob('task*.json')),
    }

    # Load all task data
    task_data_cache = {}
    for track, files in task_files.items():
        for tf in files:
            task_data_cache[tf.name] = load_task_data(tf)

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue

        # Check sample count
        summary_file = model_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            total = summary.get('overall', {}).get('total_cases', 0)
            if total < min_samples:
                continue

        model_name = get_model_name(model_dir.name)
        all_results[model_name] = {}

        task_results_dir = model_dir / 'task_results'
        if not task_results_dir.exists():
            continue

        for result_file in task_results_dir.glob('*.json'):
            # Find matching task file
            task_name = result_file.stem

            # Map result filename to task filename
            if task_name.startswith('pure_gps_'):
                task_filename = task_name.replace('pure_gps_', '') + '.json'
            elif task_name.startswith('applied_'):
                task_filename = task_name.replace('applied_', '') + '.json'
            else:
                continue

            if task_filename not in task_data_cache:
                continue

            task_data = task_data_cache[task_filename]
            region_stats = analyze_results_by_region(result_file, task_data)

            if region_stats:
                display_name = get_task_display_name(task_name)
                all_results[model_name][display_name] = region_stats

    return all_results


def print_summary_by_task(all_results):
    """Print summary organized by task."""
    # Get all tasks
    all_tasks = set()
    for model_data in all_results.values():
        all_tasks.update(model_data.keys())

    regions = ['Europe', 'Asia', 'Africa', 'North America', 'South America', 'Oceania']

    for task in sorted(all_tasks):
        print(f"\n{'='*100}")
        print(f"TASK: {task}")
        print('='*100)

        print(f"\n{'Model':<28}", end='')
        for r in regions:
            print(f"{r:>14}", end='')
        print(f"{'Overall':>12}")
        print('-'*100)

        for model in sorted(all_results.keys()):
            if task not in all_results[model]:
                continue

            task_data = all_results[model][task]
            print(f"{model:<28}", end='')

            total_correct = 0
            total_count = 0

            for region in regions:
                if region in task_data:
                    acc = task_data[region]['accuracy']
                    count = task_data[region]['total']
                    print(f"{acc:>10.1f}% ({count:>3})", end='')
                    total_correct += task_data[region]['correct']
                    total_count += task_data[region]['total']
                else:
                    print(f"{'--':>14}", end='')

            # Overall
            if total_count > 0:
                overall = total_correct / total_count * 100
                print(f"{overall:>10.1f}%")
            else:
                print(f"{'--':>12}")


def create_region_heatmap(all_results, output_file):
    """Create heatmap of accuracy by region for top model."""
    # Get best model (by average accuracy)
    best_model = None
    best_avg = 0

    for model, tasks in all_results.items():
        accs = []
        for task_data in tasks.values():
            for region_data in task_data.values():
                if region_data['total'] >= 10:
                    accs.append(region_data['accuracy'])
        if accs:
            avg = sum(accs) / len(accs)
            if avg > best_avg:
                best_avg = avg
                best_model = model

    if not best_model:
        print("No model found for heatmap")
        return

    model_data = all_results[best_model]
    tasks = sorted(model_data.keys())
    regions = ['Europe', 'Asia', 'Africa', 'North America', 'South America', 'Oceania']

    # Build data matrix
    data = np.zeros((len(tasks), len(regions)))
    data[:] = np.nan

    for i, task in enumerate(tasks):
        for j, region in enumerate(regions):
            if region in model_data[task]:
                if model_data[task][region]['total'] >= 5:
                    data[i, j] = model_data[task][region]['accuracy']

    fig, ax = plt.subplots(figsize=(12, 10))

    # Mask NaN values
    masked_data = np.ma.masked_invalid(data)

    im = ax.imshow(masked_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(regions)))
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_xticklabels(regions, fontsize=13, rotation=45, ha='right')
    ax.set_yticklabels(tasks, fontsize=12)

    # Add value annotations
    for i in range(len(tasks)):
        for j in range(len(regions)):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f'{data[i, j]:.0f}%',
                              ha='center', va='center', fontsize=11,
                              color='white' if data[i, j] < 50 else 'black')

    ax.set_title(f'Accuracy by Region - {best_model}\n(Green = Better, Gray = Insufficient Data)',
                 fontsize=14, fontweight='bold')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_region_comparison_chart(all_results, output_file):
    """Create bar chart comparing regions across all models."""
    regions = ['Europe', 'Asia', 'Africa', 'North America', 'South America', 'Oceania']
    models = list(all_results.keys())

    # Calculate average accuracy per region for each model
    model_region_avgs = {}
    for model in models:
        model_region_avgs[model] = {}
        for region in regions:
            accs = []
            for task_data in all_results[model].values():
                if region in task_data and task_data[region]['total'] >= 10:
                    accs.append(task_data[region]['accuracy'])
            if accs:
                model_region_avgs[model][region] = sum(accs) / len(accs)

    x = np.arange(len(regions))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        accs = [model_region_avgs[model].get(r, 0) for r in regions]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=model, color=colors[i])

        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{acc:.0f}%', ha='center', va='bottom', fontsize=10, rotation=45)

    ax.set_xlabel('Region', fontsize=15)
    ax.set_ylabel('Average Accuracy (%)', fontsize=15)
    ax.set_title('Average Accuracy by Geographic Region (All Tasks)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=14)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze results by geographic region')
    parser.add_argument('--results-dir', type=Path, default=Path('results'),
                        help='Results directory')
    parser.add_argument('--min-samples', type=int, default=1000,
                        help='Minimum samples for model inclusion')
    parser.add_argument('--subregions', action='store_true',
                        help='Include Asia/Europe subregions')
    args = parser.parse_args()

    print("=" * 100)
    print("BENCHMARK RESULTS BY GEOGRAPHIC REGION")
    print("=" * 100)

    print("\nProcessing models and tasks...")
    all_results = process_all_models(args.results_dir, args.min_samples)

    if not all_results:
        print("No results found!")
        return

    print(f"\nFound {len(all_results)} models")

    # Print summary
    print_summary_by_task(all_results)

    # Generate charts
    print("\nGenerating charts...")
    create_region_heatmap(all_results, 'chart_region_heatmap.png')
    create_region_comparison_chart(all_results, 'chart_region_comparison.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
