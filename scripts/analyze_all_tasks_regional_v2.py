#!/usr/bin/env python3
"""Analyze regional performance across ALL tasks using coordinate-based region detection."""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Subregion bounding boxes
def get_subregion_from_coords(lat, lon):
    """Determine subregion from lat/lon coordinates."""
    # North America (US, Canada)
    if 25 <= lat <= 85 and -170 <= lon <= -50:
        return 'North America'
    # Central America
    if 7 <= lat <= 25 and -120 <= lon <= -77:
        return 'Central America'
    # South America
    if -60 <= lat <= 15 and -85 <= lon <= -30:
        return 'South America'
    # Western Europe
    if 43 <= lat <= 60 and -10 <= lon <= 15:
        return 'Western Europe'
    # Northern Europe
    if 54 <= lat <= 72 and -25 <= lon <= 30:
        return 'Northern Europe'
    # Southern Europe
    if 35 <= lat <= 47 and -10 <= lon <= 30:
        return 'Southern Europe'
    # Eastern Europe
    if 45 <= lat <= 70 and 15 <= lon <= 60:
        return 'Eastern Europe'
    # East Asia
    if 20 <= lat <= 55 and 100 <= lon <= 150:
        return 'East Asia'
    # Southeast Asia
    if -10 <= lat <= 25 and 95 <= lon <= 140:
        return 'Southeast Asia'
    # South Asia
    if 5 <= lat <= 40 and 60 <= lon <= 100:
        return 'South Asia'
    # Middle East
    if 12 <= lat <= 42 and 25 <= lon <= 65:
        return 'Middle East'
    # Central Asia
    if 35 <= lat <= 55 and 50 <= lon <= 90:
        return 'Central Asia'
    # Africa (combined)
    if -35 <= lat <= 38 and -20 <= lon <= 55:
        return 'Africa'
    # Oceania
    if -50 <= lat <= 0 and 110 <= lon <= 180:
        return 'Oceania'
    if -50 <= lat <= -10 and 100 <= lon <= 180:
        return 'Oceania'
    return 'Unknown'


# Keep old function name for compatibility
def get_continent_from_coords(lat, lon):
    return get_subregion_from_coords(lat, lon)

# Country code to subregion mapping
COUNTRY_TO_SUBREGION = {
    # North America
    'US': 'North America', 'CA': 'North America', 'MX': 'North America',
    # South America
    'BR': 'South America', 'AR': 'South America', 'CL': 'South America',
    'CO': 'South America', 'PE': 'South America', 'VE': 'South America',
    'EC': 'South America', 'BO': 'South America', 'PY': 'South America',
    'UY': 'South America', 'GY': 'South America', 'SR': 'South America',
    # Western Europe
    'GB': 'Western Europe', 'UK': 'Western Europe', 'FR': 'Western Europe',
    'DE': 'Western Europe', 'NL': 'Western Europe', 'BE': 'Western Europe',
    'AT': 'Western Europe', 'CH': 'Western Europe', 'IE': 'Western Europe',
    # Eastern Europe
    'PL': 'Eastern Europe', 'CZ': 'Eastern Europe', 'SK': 'Eastern Europe',
    'HU': 'Eastern Europe', 'RO': 'Eastern Europe', 'BG': 'Eastern Europe',
    'UA': 'Eastern Europe', 'RU': 'Eastern Europe',
    # Southern Europe
    'IT': 'Southern Europe', 'ES': 'Southern Europe', 'PT': 'Southern Europe',
    'GR': 'Southern Europe',
    # Northern Europe
    'SE': 'Northern Europe', 'NO': 'Northern Europe', 'DK': 'Northern Europe',
    'FI': 'Northern Europe',
    # East Asia
    'CN': 'East Asia', 'JP': 'East Asia', 'KR': 'East Asia', 'TW': 'East Asia',
    'HK': 'East Asia', 'MN': 'East Asia',
    # Southeast Asia
    'TH': 'Southeast Asia', 'VN': 'Southeast Asia', 'MY': 'Southeast Asia',
    'SG': 'Southeast Asia', 'ID': 'Southeast Asia', 'PH': 'Southeast Asia',
    'MM': 'Southeast Asia', 'KH': 'Southeast Asia',
    # South Asia
    'IN': 'South Asia', 'PK': 'South Asia', 'BD': 'South Asia', 'LK': 'South Asia',
    'NP': 'South Asia',
    # Middle East
    'TR': 'Middle East', 'IR': 'Middle East', 'IQ': 'Middle East', 'SA': 'Middle East',
    'AE': 'Middle East', 'IL': 'Middle East',
    # Africa (combined - not split into subregions)
    'EG': 'Africa', 'ZA': 'Africa', 'NG': 'Africa', 'KE': 'Africa',
    'ET': 'Africa', 'TZ': 'Africa', 'MA': 'Africa', 'DZ': 'Africa',
    'GH': 'Africa', 'CI': 'Africa', 'SN': 'Africa', 'CM': 'Africa',
    'UG': 'Africa', 'RW': 'Africa', 'ZM': 'Africa', 'ZW': 'Africa',
    # Oceania
    'AU': 'Oceania', 'NZ': 'Oceania', 'PG': 'Oceania', 'FJ': 'Oceania',
}

# Keep old name for compatibility
COUNTRY_TO_CONTINENT = COUNTRY_TO_SUBREGION

# Country name to subregion
COUNTRY_NAME_TO_SUBREGION = {
    'United States': 'North America', 'USA': 'North America', 'Canada': 'North America',
    'Mexico': 'North America', 'Brazil': 'South America', 'Argentina': 'South America',
    'United Kingdom': 'Western Europe', 'UK': 'Western Europe', 'France': 'Western Europe',
    'Germany': 'Western Europe', 'Italy': 'Southern Europe', 'Spain': 'Southern Europe',
    'Netherlands': 'Western Europe', 'Poland': 'Eastern Europe', 'Russia': 'Eastern Europe',
    'China': 'East Asia', 'Japan': 'East Asia', 'South Korea': 'East Asia', 'India': 'South Asia',
    'Indonesia': 'Southeast Asia', 'Thailand': 'Southeast Asia', 'Vietnam': 'Southeast Asia',
    'Turkey': 'Middle East', 'Saudi Arabia': 'Middle East',
    'Egypt': 'Africa', 'South Africa': 'Africa', 'Nigeria': 'Africa',
    'Australia': 'Oceania', 'New Zealand': 'Oceania',
}

# Keep old name for compatibility
COUNTRY_NAME_TO_CONTINENT = COUNTRY_NAME_TO_SUBREGION

# Region name normalization (from data files to our standard names)
REGION_NAME_MAP = {
    'North America': 'North America',
    'South America': 'South America',
    'Central America': 'South America',  # Combine with South America
    'Caribbean': 'South America',  # Combine with South America
    'Western Europe': 'Western Europe',
    'Northern Europe': 'Northern Europe',
    'Southern Europe': 'Southern Europe',
    'Eastern Europe': 'Eastern Europe',
    'East Asia': 'East Asia',
    'Southeast Asia': 'Southeast Asia',
    'South Asia': 'South Asia',
    'Central Asia': 'Middle East',  # Combine with Middle East
    'Middle East': 'Middle East',
    'West Asia': 'Middle East',
    # Combine all Africa subregions
    'North Africa': 'Africa',
    'West Africa': 'Africa',
    'East Africa': 'Africa',
    'Central Africa': 'Africa',
    'Southern Africa': 'Africa',
    'Oceania': 'Oceania',
    'Australia': 'Oceania',
}

MODEL_ORDER = [
    'GPT-5.1', 'GPT-5-mini', 'GPT-5-nano', 'GPT-4.1', 'GPT-4.1-mini',
    'Gemini-2.5-Pro', 'Gemini-2.5-Flash',
    'Claude-Haiku-4.5',
    'Mistral-Large', 'Mistral-Small-24B',
    'Qwen3-235B', 'Qwen3-30B', 'Qwen3-14B', 'Qwen3-8B'
]

def normalize_model_name(model_name):
    model_lower = model_name.lower()
    if 'gpt-5.1' in model_lower:
        return 'GPT-5-mini'
    elif 'gpt-5-mini' in model_lower:
        return 'GPT-5.1'
    elif 'gpt-5-nano' in model_lower:
        return 'GPT-5-nano'
    elif 'gpt-4.1-mini' in model_lower:
        return 'GPT-4.1-mini'
    elif 'gpt-4.1' in model_lower:
        return 'GPT-4.1'
    elif 'gemini-2.5-pro' in model_lower:
        return 'Gemini-2.5-Pro'
    elif 'gemini-2.5-flash' in model_lower:
        return 'Gemini-2.5-Flash'
    elif 'claude-haiku' in model_lower:
        return 'Claude-Haiku-4.5'
    elif 'mistral-large' in model_lower:
        return 'Mistral-Large'
    elif 'mistral-small' in model_lower:
        return 'Mistral-Small-24B'
    elif 'qwen3-235b' in model_lower:
        return 'Qwen3-235B'
    elif 'qwen3-30b' in model_lower:
        return 'Qwen3-30B'
    elif 'qwen3-14b' in model_lower:
        return 'Qwen3-14B'
    elif 'qwen3-8b' in model_lower:
        return 'Qwen3-8B'
    return model_name


def dms_to_decimal(degrees, minutes, seconds, direction):
    """Convert DMS to decimal degrees."""
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal


def extract_coords_from_prompt(prompt):
    """Extract coordinates from prompt text."""
    # Pattern for DMS format: 43°55'31.8"N, 81°24'43.6"E
    dms_pattern = r"(\d+)°(\d+)'([\d.]+)\"([NS]),?\s*(\d+)°(\d+)'([\d.]+)\"([EW])"
    match = re.search(dms_pattern, prompt)
    if match:
        lat = dms_to_decimal(match.group(1), match.group(2), match.group(3), match.group(4))
        lon = dms_to_decimal(match.group(5), match.group(6), match.group(7), match.group(8))
        return lat, lon

    # Pattern: decimal degrees like "40.7128, -74.0060" or "40.7128°N, 74.0060°W"
    patterns = [
        r'coordinates?\s*[:\(]?\s*(-?\d+\.?\d*)[°]?\s*[NS]?,?\s*(-?\d+\.?\d*)[°]?\s*[EW]?',
        r'at\s+(-?\d+\.?\d*)[°]?\s*[NS]?,?\s*(-?\d+\.?\d*)[°]?\s*[EW]?',
        r'Point A[:\s]+\(?(-?\d+\.?\d*)[°]?\s*[NS]?,?\s*(-?\d+\.?\d*)[°]?\s*[EW]?\)?',
        r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)',
        r'latitude\s+(-?\d+\.?\d*)[°]?.*longitude\s+(-?\d+\.?\d*)[°]?',
        r'longitude\s+(-?\d+\.?\d*)[°]?.*latitude\s+(-?\d+\.?\d*)[°]?',  # lon first then lat
    ]
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            try:
                if i == 5:  # longitude first pattern
                    lon = float(match.group(1))
                    lat = float(match.group(2))
                else:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon
            except:
                pass
    return None, None


def extract_subregion_from_sample(sample, gt):
    """Extract subregion from a sample using all available info."""

    # 1. Check for explicit region in ground_truth
    if isinstance(gt, dict) and 'region' in gt:
        region = gt['region']
        if region in REGION_NAME_MAP:
            return REGION_NAME_MAP[region]

    # 2. Check for country_code in ground_truth
    if isinstance(gt, dict) and 'country_code' in gt:
        return COUNTRY_TO_SUBREGION.get(gt['country_code'], None)

    # 3. Check for country (code or name) in ground_truth or sample
    country = None
    if isinstance(gt, dict) and 'country' in gt:
        country = gt['country']
    elif 'country' in sample:
        country = sample['country']
    if country:
        # Try as code first, then as name
        if country in COUNTRY_TO_SUBREGION:
            return COUNTRY_TO_SUBREGION[country]
        return COUNTRY_NAME_TO_SUBREGION.get(country, None)

    # 4. Extract from coordinates in sample
    coord = None
    if 'coordinate' in sample:
        coord = sample['coordinate']
    elif 'point_a' in sample:
        coord = sample['point_a']
    elif 'reference_point' in sample:
        coord = sample['reference_point']
    elif 'points' in sample and sample['points']:
        coord = sample['points'][0]  # Use first point
    elif 'vertices' in sample and sample['vertices']:
        coord = sample['vertices'][0]
    elif 'source_coords' in sample:
        coord = sample['source_coords']

    if coord:
        lat = coord.get('latitude') or coord.get('lat')
        lon = coord.get('longitude') or coord.get('lon') or coord.get('lng')
        if lat is not None and lon is not None:
            return get_subregion_from_coords(lat, lon)

    # 5. Try to extract from prompt text
    prompt = sample.get('prompt', '')
    lat, lon = extract_coords_from_prompt(prompt)
    if lat is not None and lon is not None:
        return get_subregion_from_coords(lat, lon)

    return None

# Keep old function name for compatibility
def extract_continent_from_sample(sample, gt):
    return extract_subregion_from_sample(sample, gt)


def load_original_data_lookup(task_pattern):
    """Load original data and create lookup by question/prompt text."""
    data_dirs = [Path('data/track_applied'), Path('data/track_pure_gps')]
    lookup = {}

    for data_dir in data_dirs:
        for data_file in data_dir.glob('*.json'):
            # Check if this file matches the pattern
            if not any(p.replace('*', '') in data_file.name for p in task_pattern.split('*') if p):
                continue

            try:
                with open(data_file) as f:
                    data = json.load(f)

                for item in data:
                    # Create key from question text or location_name
                    question = item.get('question', '')
                    location_name = item.get('location_name', '')

                    # Store coordinates/country for this item
                    info = {}

                    # Get coordinates
                    coord = None
                    if 'coordinate' in item:
                        coord = item['coordinate']
                    elif 'point_a' in item:
                        coord = item['point_a']
                    elif 'reference_point' in item:
                        coord = item['reference_point']
                    elif 'points' in item and item['points']:
                        coord = item['points'][0]
                    elif 'start' in item:
                        coord = item.get('start', {})
                    elif 'source_coords' in item:
                        coord = item['source_coords']
                    elif 'partial_coordinate' in item:
                        # For Missing Data task - use partial coordinate
                        pc = item['partial_coordinate']
                        if 'latitude' in pc and pc['latitude']:
                            coord = {'lat': pc['latitude'], 'lon': 0}  # Placeholder lon
                        elif 'longitude' in pc and pc['longitude']:
                            coord = {'lat': 0, 'lon': pc['longitude']}  # Placeholder lat

                    if coord:
                        info['coord'] = coord

                    # Get country code
                    if 'country' in item:
                        info['country'] = item['country']

                    if info:
                        if question:
                            lookup[question[:100]] = info
                        if location_name:
                            lookup[location_name] = info
            except:
                pass

    return lookup


def analyze_task_by_subregion(results_dir, task_pattern, task_name):
    """Analyze a task's performance by continent."""
    results = {}

    # Load original data lookup for coordinate extraction
    original_lookup = load_original_data_lookup(task_pattern)

    for model_dir in Path(results_dir).iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith('gpsbench_'):
            continue

        raw_name = model_dir.name.replace('gpsbench_', '').split('_2025')[0].split('_2026')[0]
        model_name = normalize_model_name(raw_name)

        # Find matching task file
        task_files = list((model_dir / 'task_results').glob(task_pattern))
        if not task_files:
            continue

        with open(task_files[0], 'r') as f:
            data = json.load(f)

        continent_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for result in data.get('results', []):
            gt = result.get('ground_truth', {})
            continent = extract_continent_from_sample(result, gt)

            # If not found, try lookup from original data
            if not continent or continent == 'Unknown':
                prompt = result.get('prompt', '')
                # Try to find matching original data
                for key, info in original_lookup.items():
                    if key in prompt:
                        # Try country code first
                        if 'country' in info:
                            continent = COUNTRY_TO_CONTINENT.get(info['country'])
                            if continent:
                                break
                        # Try coordinates
                        coord = info.get('coord', {})
                        lat = coord.get('lat') or coord.get('latitude')
                        lon = coord.get('lon') or coord.get('longitude') or coord.get('lng')
                        if lat is not None and lon is not None and lat != 0 and lon != 0:
                            continent = get_continent_from_coords(lat, lon)
                            break

            if not continent or continent == 'Unknown':
                continue

            is_correct = result.get('correct', result.get('is_correct', False))
            continent_stats[continent]['total'] += 1
            if is_correct:
                continent_stats[continent]['correct'] += 1

        results[model_name] = dict(continent_stats)

    return results


def aggregate_results(results):
    """Aggregate results across all models."""
    continent_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for model, continents in results.items():
        for continent, stats in continents.items():
            continent_stats[continent]['correct'] += stats['correct']
            continent_stats[continent]['total'] += stats['total']

    return dict(continent_stats)


def main():
    results_dir = Path('results')
    figures_dir = Path('paper/figures')

    # All tasks with their patterns and display names
    applied_tasks = [
        ('*task1_place_association.json', 'Place Assoc.'),
        ('*task3_name_disambiguation.json', 'Name Disamb.'),
        ('*task5_relative_position.json', 'Rel. Position'),
        ('*task6_proximity.json', 'Proximity'),
        ('*task7_route_analysis.json', 'Route Analysis'),
        ('*task8_boundary_analysis.json', 'Boundary'),
        ('*task9_spatial_patterns.json', 'Spatial Patterns'),
        ('*task13_terrain_classification.json', 'Terrain Class.'),
    ]

    pure_gps_tasks = [
        ('*task1_format_conversion.json', 'Format Conv.'),
        ('*task3_coordinate_system*.json', 'Coord. Trans.'),
        ('*task4_distance_calculation.json', 'Distance'),
        ('*task5_bearing_computation.json', 'Bearing'),
        ('*task6_interpolation.json', 'Interpolation'),
        ('*task7_area_perimeter.json', 'Polygon Area'),
        ('*task8_bounding_box.json', 'Bounding Box'),
        ('*task9_route_geometry.json', 'Route Geom.'),
        ('*task10_relative_position.json', 'Rel. Pos. (Pure)'),
    ]

    all_tasks = applied_tasks + pure_gps_tasks
    # Subregions (Africa is combined, not split)
    subregions = [
        'North America', 'South America', 'Western Europe', 'Eastern Europe',
        'East Asia', 'South Asia', 'Southeast Asia', 'Middle East',
        'Africa', 'Oceania'
    ]

    # Analyze all tasks
    task_results = {}
    print("Analyzing regional performance for all tasks...\n")

    for pattern, task_name in all_tasks:
        results = analyze_task_by_subregion(results_dir, pattern, task_name)
        if not results:
            print(f"  {task_name}: No results found")
            continue

        task_results[task_name] = results
        agg = aggregate_results(results)

        # Print summary
        print(f"{task_name}:")
        for subregion in subregions:
            stats = agg.get(subregion, {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                acc = 100 * stats['correct'] / stats['total']
                print(f"  {subregion}: {acc:.1f}% (n={stats['total']})")
        print()

    # Create visualization
    plot_all_tasks_regional(task_results, subregions, figures_dir / 'regional_all_tasks_comprehensive.pdf')
    print_latex_table(task_results, subregions)


def plot_all_tasks_regional(task_results, subregions, output_path):
    """Create heatmap of regional performance across all tasks."""
    tasks = list(task_results.keys())

    # Build matrix
    matrix = []
    for task in tasks:
        row = []
        agg = aggregate_results(task_results[task])
        for subregion in subregions:
            stats = agg.get(subregion, {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                row.append(100 * stats['correct'] / stats['total'])
            else:
                row.append(np.nan)
        matrix.append(row)

    matrix = np.array(matrix)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Mask NaN values
    masked = np.ma.masked_invalid(matrix)

    im = ax.imshow(masked, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Abbreviated labels for subregions
    short_names = {
        'North America': 'N.Am', 'South America': 'S.Am',
        'Western Europe': 'W.Eur', 'Eastern Europe': 'E.Eur',
        'East Asia': 'E.Asia', 'South Asia': 'S.Asia',
        'Southeast Asia': 'SE.Asia', 'Middle East': 'Mid.East',
        'Africa': 'Africa', 'Oceania': 'Oceania'
    }
    x_labels = [short_names.get(s, s) for s in subregions]

    # Labels
    ax.set_xticks(np.arange(len(subregions)))
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(tasks)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add value annotations
    for i in range(len(tasks)):
        for j in range(len(subregions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)
            else:
                ax.text(j, i, '--', ha='center', va='center', color='gray', fontsize=8)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom')

    ax.set_title('Regional Performance Across All GPSBench Tasks by Subregion\n(14 LLMs Aggregated)')

    # Add horizontal line to separate Applied and Pure GPS (8 Applied tasks)
    ax.axhline(y=7.5, color='black', linewidth=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_latex_table(task_results, subregions):
    """Print LaTeX table."""
    # Abbreviated headers
    short_names = {
        'North America': 'N.Am', 'South America': 'S.Am',
        'Western Europe': 'W.Eur', 'Eastern Europe': 'E.Eur',
        'East Asia': 'E.Asia', 'South Asia': 'S.Asia',
        'Southeast Asia': 'SE.Asia', 'Middle East': 'M.East',
        'Africa': 'Africa', 'Oceania': 'Oceania'
    }
    headers = [short_names.get(s, s) for s in subregions]

    print("\n% LaTeX Table: Regional performance across all tasks by subregion")
    print("\\begin{table*}[htbp]")
    print("\\centering")
    print("\\caption{Accuracy (\\%) by subregion across all GPSBench tasks (14 LLMs aggregated). -- indicates insufficient data.}")
    print("\\label{tab:regional_all_tasks_comprehensive}")
    print("\\small")
    print("\\begin{tabular}{l" + "c" * len(subregions) + "}")
    print("\\toprule")
    print("\\textbf{Task} & " + " & ".join([f"\\textbf{{{h}}}" for h in headers]) + " \\\\")
    print("\\midrule")

    # Applied tasks
    applied_tasks = ['Place Assoc.', 'Name Disamb.', 'Rel. Position', 'Proximity',
                     'Route Analysis', 'Boundary', 'Spatial Patterns', 'Terrain Class.']
    pure_tasks = ['Format Conv.', 'Coord. Trans.', 'Distance', 'Bearing',
                  'Interpolation', 'Polygon Area', 'Bounding Box', 'Route Geom.', 'Rel. Pos. (Pure)']

    for task in applied_tasks:
        if task not in task_results:
            continue
        agg = aggregate_results(task_results[task])
        row = [task]
        for subregion in subregions:
            stats = agg.get(subregion, {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                acc = 100 * stats['correct'] / stats['total']
                row.append(f"{acc:.1f}")
            else:
                row.append("--")
        print(" & ".join(row) + " \\\\")

    print("\\midrule")

    for task in pure_tasks:
        if task not in task_results:
            continue
        agg = aggregate_results(task_results[task])
        row = [task]
        for subregion in subregions:
            stats = agg.get(subregion, {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                acc = 100 * stats['correct'] / stats['total']
                row.append(f"{acc:.1f}")
            else:
                row.append("--")
        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")


if __name__ == '__main__':
    main()
