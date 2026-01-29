#!/usr/bin/env python3
"""Generate summary regional performance plot for main paper content."""

import json
import re
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
from collections import defaultdict

# Continent bounding boxes (approximate)
def get_continent_from_coords(lat, lon):
    """Determine continent from lat/lon coordinates."""
    if 15 <= lat <= 85 and -170 <= lon <= -50:
        return 'Americas'
    if 7 <= lat <= 25 and -120 <= lon <= -60:
        return 'Americas'
    if -60 <= lat <= 15 and -85 <= lon <= -30:
        return 'Americas'
    if 35 <= lat <= 72 and -25 <= lon <= 65:
        return 'Europe'
    if -35 <= lat <= 38 and -20 <= lon <= 55:
        return 'Africa'
    if -10 <= lat <= 80 and 25 <= lon <= 180:
        return 'Asia'
    if -10 <= lat <= 80 and -180 <= lon <= -130:
        return 'Asia'
    if -50 <= lat <= 0 and 110 <= lon <= 180:
        return 'Oceania'
    if -50 <= lat <= -10 and 100 <= lon <= 180:
        return 'Oceania'
    return 'Unknown'

COUNTRY_TO_CONTINENT = {
    'US': 'Americas', 'CA': 'Americas', 'MX': 'Americas', 'BR': 'Americas',
    'AR': 'Americas', 'CL': 'Americas', 'CO': 'Americas', 'PE': 'Americas',
    'GB': 'Europe', 'UK': 'Europe', 'FR': 'Europe', 'DE': 'Europe',
    'IT': 'Europe', 'ES': 'Europe', 'NL': 'Europe', 'BE': 'Europe',
    'PL': 'Europe', 'RU': 'Europe', 'UA': 'Europe', 'SE': 'Europe',
    'NO': 'Europe', 'FI': 'Europe', 'DK': 'Europe', 'AT': 'Europe',
    'CH': 'Europe', 'PT': 'Europe', 'GR': 'Europe', 'CZ': 'Europe',
    'CN': 'Asia', 'JP': 'Asia', 'KR': 'Asia', 'IN': 'Asia',
    'ID': 'Asia', 'TH': 'Asia', 'VN': 'Asia', 'MY': 'Asia',
    'SG': 'Asia', 'PH': 'Asia', 'PK': 'Asia', 'BD': 'Asia',
    'TR': 'Asia', 'SA': 'Asia', 'AE': 'Asia', 'IL': 'Asia',
    'IR': 'Asia', 'IQ': 'Asia',
    'EG': 'Africa', 'ZA': 'Africa', 'NG': 'Africa', 'KE': 'Africa',
    'ET': 'Africa', 'TZ': 'Africa', 'MA': 'Africa', 'DZ': 'Africa',
    'AU': 'Oceania', 'NZ': 'Oceania',
}

COUNTRY_NAME_TO_CONTINENT = {
    'United States': 'Americas', 'USA': 'Americas', 'Canada': 'Americas',
    'Mexico': 'Americas', 'Brazil': 'Americas', 'Argentina': 'Americas',
    'United Kingdom': 'Europe', 'UK': 'Europe', 'France': 'Europe',
    'Germany': 'Europe', 'Italy': 'Europe', 'Spain': 'Europe',
    'Netherlands': 'Europe', 'Poland': 'Europe', 'Russia': 'Europe',
    'China': 'Asia', 'Japan': 'Asia', 'South Korea': 'Asia', 'India': 'Asia',
    'Indonesia': 'Asia', 'Thailand': 'Asia', 'Vietnam': 'Asia',
    'Turkey': 'Asia', 'Saudi Arabia': 'Asia',
    'Egypt': 'Africa', 'South Africa': 'Africa', 'Nigeria': 'Africa',
    'Australia': 'Oceania', 'New Zealand': 'Oceania',
}

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
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal


def extract_coords_from_prompt(prompt):
    dms_pattern = r"(\d+)°(\d+)'([\d.]+)\"([NS]),?\s*(\d+)°(\d+)'([\d.]+)\"([EW])"
    match = re.search(dms_pattern, prompt)
    if match:
        lat = dms_to_decimal(match.group(1), match.group(2), match.group(3), match.group(4))
        lon = dms_to_decimal(match.group(5), match.group(6), match.group(7), match.group(8))
        return lat, lon

    patterns = [
        r'coordinates?\s*[:\(]?\s*(-?\d+\.?\d*)[°]?\s*[NS]?,?\s*(-?\d+\.?\d*)[°]?\s*[EW]?',
        r'at\s+(-?\d+\.?\d*)[°]?\s*[NS]?,?\s*(-?\d+\.?\d*)[°]?\s*[EW]?',
        r'Point A[:\s]+\(?(-?\d+\.?\d*)[°]?\s*[NS]?,?\s*(-?\d+\.?\d*)[°]?\s*[EW]?\)?',
        r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)',
        r'latitude\s+(-?\d+\.?\d*)[°]?.*longitude\s+(-?\d+\.?\d*)[°]?',
        r'longitude\s+(-?\d+\.?\d*)[°]?.*latitude\s+(-?\d+\.?\d*)[°]?',
    ]
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            try:
                if i == 5:
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


def extract_continent_from_sample(sample, gt):
    if isinstance(gt, dict) and 'region' in gt:
        region = gt['region']
        region_to_continent = {
            'North America': 'Americas', 'South America': 'Americas',
            'Central America': 'Americas', 'Caribbean': 'Americas',
            'Western Europe': 'Europe', 'Eastern Europe': 'Europe',
            'Northern Europe': 'Europe', 'Southern Europe': 'Europe',
            'East Asia': 'Asia', 'Southeast Asia': 'Asia',
            'South Asia': 'Asia', 'Central Asia': 'Asia',
            'Middle East': 'Asia', 'West Asia': 'Asia',
            'North Africa': 'Africa', 'West Africa': 'Africa',
            'East Africa': 'Africa', 'Southern Africa': 'Africa',
            'Central Africa': 'Africa',
            'Oceania': 'Oceania', 'Australia': 'Oceania',
        }
        if region in region_to_continent:
            return region_to_continent[region]

    if isinstance(gt, dict) and 'country_code' in gt:
        return COUNTRY_TO_CONTINENT.get(gt['country_code'], None)

    country = None
    if isinstance(gt, dict) and 'country' in gt:
        country = gt['country']
    elif 'country' in sample:
        country = sample['country']
    if country:
        if country in COUNTRY_TO_CONTINENT:
            return COUNTRY_TO_CONTINENT[country]
        return COUNTRY_NAME_TO_CONTINENT.get(country, None)

    coord = None
    if 'coordinate' in sample:
        coord = sample['coordinate']
    elif 'point_a' in sample:
        coord = sample['point_a']
    elif 'reference_point' in sample:
        coord = sample['reference_point']
    elif 'points' in sample and sample['points']:
        coord = sample['points'][0]
    elif 'vertices' in sample and sample['vertices']:
        coord = sample['vertices'][0]
    elif 'source_coords' in sample:
        coord = sample['source_coords']

    if coord:
        lat = coord.get('latitude') or coord.get('lat')
        lon = coord.get('longitude') or coord.get('lon') or coord.get('lng')
        if lat is not None and lon is not None:
            return get_continent_from_coords(lat, lon)

    prompt = sample.get('prompt', '')
    lat, lon = extract_coords_from_prompt(prompt)
    if lat is not None and lon is not None:
        return get_continent_from_coords(lat, lon)

    return None


def load_original_data_lookup(task_pattern):
    data_dirs = [Path('data/track_applied'), Path('data/track_pure_gps')]
    lookup = {}

    for data_dir in data_dirs:
        for data_file in data_dir.glob('*.json'):
            if not any(p.replace('*', '') in data_file.name for p in task_pattern.split('*') if p):
                continue
            try:
                with open(data_file) as f:
                    data = json.load(f)
                for item in data:
                    question = item.get('question', '')
                    location_name = item.get('location_name', '')
                    info = {}
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
                        pc = item['partial_coordinate']
                        if 'latitude' in pc and pc['latitude']:
                            coord = {'lat': pc['latitude'], 'lon': 0}
                        elif 'longitude' in pc and pc['longitude']:
                            coord = {'lat': 0, 'lon': pc['longitude']}
                    if coord:
                        info['coord'] = coord
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


def analyze_all_tasks():
    """Analyze all tasks and return aggregated continent stats."""
    results_dir = Path('results')

    all_tasks = [
        ('*task1_place_association.json', 'Place Association', 'Applied'),
        ('*task3_name_disambiguation.json', 'Name Disambiguation', 'Applied'),
        ('*task5_relative_position.json', 'Relative Position', 'Applied'),
        ('*task6_proximity.json', 'Proximity', 'Applied'),
        ('*task7_route_analysis.json', 'Route Analysis', 'Applied'),
        ('*task8_boundary_analysis.json', 'Boundary Analysis', 'Applied'),
        ('*task9_spatial_patterns.json', 'Spatial Patterns', 'Applied'),
        ('*task11_missing_data.json', 'Missing Data', 'Applied'),
        ('*task13_terrain_classification.json', 'Terrain Classification', 'Applied'),
        ('*task1_format_conversion.json', 'Format Conversion', 'Pure GPS'),
        ('*task3_coordinate_system*.json', 'Coordinate Transformation', 'Pure GPS'),
        ('*task4_distance_calculation.json', 'Distance Calculation', 'Pure GPS'),
        ('*task5_bearing_computation.json', 'Bearing Computation', 'Pure GPS'),
        ('*task6_interpolation.json', 'Interpolation', 'Pure GPS'),
        ('*task7_area_perimeter.json', 'Polygon Area', 'Pure GPS'),
        ('*task8_bounding_box.json', 'Bounding Box', 'Pure GPS'),
        ('*task9_route_geometry.json', 'Route Geometry', 'Pure GPS'),
        ('*task10_relative_position.json', 'Relative Position (Pure)', 'Pure GPS'),
    ]

    continents = ['Americas', 'Europe', 'Asia', 'Africa', 'Oceania']

    # Aggregate stats by track and continent
    applied_stats = {c: {'correct': 0, 'total': 0} for c in continents}
    pure_gps_stats = {c: {'correct': 0, 'total': 0} for c in continents}
    overall_stats = {c: {'correct': 0, 'total': 0} for c in continents}

    for task_pattern, task_name, track in all_tasks:
        original_lookup = load_original_data_lookup(task_pattern)

        for model_dir in results_dir.iterdir():
            if not model_dir.is_dir() or not model_dir.name.startswith('gpsbench_'):
                continue

            task_files = list((model_dir / 'task_results').glob(task_pattern))
            if not task_files:
                continue

            with open(task_files[0], 'r') as f:
                data = json.load(f)

            for result in data.get('results', []):
                gt = result.get('ground_truth', {})
                continent = extract_continent_from_sample(result, gt)

                if not continent or continent == 'Unknown':
                    prompt = result.get('prompt', '')
                    for key, info in original_lookup.items():
                        if key in prompt:
                            if 'country' in info:
                                continent = COUNTRY_TO_CONTINENT.get(info['country'])
                                if continent:
                                    break
                            coord = info.get('coord', {})
                            lat = coord.get('lat') or coord.get('latitude')
                            lon = coord.get('lon') or coord.get('longitude') or coord.get('lng')
                            if lat is not None and lon is not None and lat != 0 and lon != 0:
                                continent = get_continent_from_coords(lat, lon)
                                break

                if not continent or continent == 'Unknown':
                    continue

                is_correct = result.get('correct', result.get('is_correct', False))

                overall_stats[continent]['total'] += 1
                if is_correct:
                    overall_stats[continent]['correct'] += 1

                if track == 'Applied':
                    applied_stats[continent]['total'] += 1
                    if is_correct:
                        applied_stats[continent]['correct'] += 1
                else:
                    pure_gps_stats[continent]['total'] += 1
                    if is_correct:
                        pure_gps_stats[continent]['correct'] += 1

    return overall_stats, applied_stats, pure_gps_stats


def main():
    print("Analyzing regional performance across all tasks...")
    overall_stats, applied_stats, pure_gps_stats = analyze_all_tasks()

    continents = ['Americas', 'Europe', 'Asia', 'Africa', 'Oceania']

    # Calculate accuracies
    overall_accs = []
    applied_accs = []
    pure_gps_accs = []

    print("\nOverall Performance by Continent:")
    for c in continents:
        if overall_stats[c]['total'] > 0:
            acc = 100 * overall_stats[c]['correct'] / overall_stats[c]['total']
            overall_accs.append(acc)
            print(f"  {c}: {acc:.1f}% (n={overall_stats[c]['total']})")
        else:
            overall_accs.append(0)

    print("\nApplied Track by Continent:")
    for c in continents:
        if applied_stats[c]['total'] > 0:
            acc = 100 * applied_stats[c]['correct'] / applied_stats[c]['total']
            applied_accs.append(acc)
            print(f"  {c}: {acc:.1f}% (n={applied_stats[c]['total']})")
        else:
            applied_accs.append(0)

    print("\nPure GPS Track by Continent:")
    for c in continents:
        if pure_gps_stats[c]['total'] > 0:
            acc = 100 * pure_gps_stats[c]['correct'] / pure_gps_stats[c]['total']
            pure_gps_accs.append(acc)
            print(f"  {c}: {acc:.1f}% (n={pure_gps_stats[c]['total']})")
        else:
            pure_gps_accs.append(0)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(continents))
    width = 0.25

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    bars1 = ax.bar(x - width, applied_accs, width, label='Applied Track', color=colors[0])
    bars2 = ax.bar(x, pure_gps_accs, width, label='Pure GPS Track', color=colors[1])
    bars3 = ax.bar(x + width, overall_accs, width, label='Overall', color=colors[2])

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Continent')
    ax.set_title('Regional Performance Across GPSBench (14 LLMs Aggregated)')
    ax.set_xticks(x)
    ax.set_xticklabels(continents)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 85)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = Path('paper/figures/regional_performance_summary.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    # Print gap analysis
    print("\n=== Regional Gap Analysis ===")
    max_overall = max(overall_accs)
    min_overall = min(overall_accs)
    max_continent = continents[overall_accs.index(max_overall)]
    min_continent = continents[overall_accs.index(min_overall)]
    print(f"Overall: {max_continent} ({max_overall:.1f}%) vs {min_continent} ({min_overall:.1f}%) = {max_overall - min_overall:.1f}% gap")

    max_applied = max(applied_accs)
    min_applied = min(applied_accs)
    max_continent = continents[applied_accs.index(max_applied)]
    min_continent = continents[applied_accs.index(min_applied)]
    print(f"Applied: {max_continent} ({max_applied:.1f}%) vs {min_continent} ({min_applied:.1f}%) = {max_applied - min_applied:.1f}% gap")

    max_pure = max(pure_gps_accs)
    min_pure = min(pure_gps_accs)
    max_continent = continents[pure_gps_accs.index(max_pure)]
    min_continent = continents[pure_gps_accs.index(min_pure)]
    print(f"Pure GPS: {max_continent} ({max_pure:.1f}%) vs {min_continent} ({min_pure:.1f}%) = {max_pure - min_pure:.1f}% gap")


if __name__ == '__main__':
    main()
