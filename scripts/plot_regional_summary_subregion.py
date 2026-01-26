#!/usr/bin/env python3
"""Generate summary regional performance plot at subregion level for main paper content."""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Country code to subregion mapping (Africa combined)
COUNTRY_TO_SUBREGION = {
    # North America
    'US': 'North America', 'CA': 'North America', 'MX': 'North America',
    # Central America & Caribbean -> South America
    'GT': 'South America', 'BZ': 'South America', 'HN': 'South America',
    'SV': 'South America', 'NI': 'South America', 'CR': 'South America',
    'PA': 'South America', 'CU': 'South America', 'JM': 'South America',
    'HT': 'South America', 'DO': 'South America', 'PR': 'South America', 'TT': 'South America',
    # South America
    'BR': 'South America', 'AR': 'South America', 'CL': 'South America',
    'CO': 'South America', 'PE': 'South America', 'VE': 'South America',
    'EC': 'South America', 'BO': 'South America', 'PY': 'South America',
    'UY': 'South America', 'GY': 'South America', 'SR': 'South America',
    # Western Europe
    'GB': 'Western Europe', 'UK': 'Western Europe', 'FR': 'Western Europe',
    'DE': 'Western Europe', 'NL': 'Western Europe', 'BE': 'Western Europe',
    'AT': 'Western Europe', 'CH': 'Western Europe', 'IE': 'Western Europe',
    'LU': 'Western Europe',
    # Northern Europe -> merge with appropriate
    'SE': 'Northern Europe', 'NO': 'Northern Europe', 'DK': 'Northern Europe',
    'FI': 'Northern Europe', 'IS': 'Northern Europe', 'EE': 'Northern Europe',
    'LV': 'Northern Europe', 'LT': 'Northern Europe',
    # Southern Europe
    'IT': 'Southern Europe', 'ES': 'Southern Europe', 'PT': 'Southern Europe',
    'GR': 'Southern Europe', 'HR': 'Southern Europe', 'SI': 'Southern Europe',
    'RS': 'Southern Europe', 'BA': 'Southern Europe', 'ME': 'Southern Europe',
    'MK': 'Southern Europe', 'AL': 'Southern Europe', 'MT': 'Southern Europe',
    # Eastern Europe
    'PL': 'Eastern Europe', 'CZ': 'Eastern Europe', 'SK': 'Eastern Europe',
    'HU': 'Eastern Europe', 'RO': 'Eastern Europe', 'BG': 'Eastern Europe',
    'UA': 'Eastern Europe', 'BY': 'Eastern Europe', 'MD': 'Eastern Europe',
    'RU': 'Eastern Europe',
    # East Asia
    'CN': 'East Asia', 'JP': 'East Asia', 'KR': 'East Asia', 'TW': 'East Asia',
    'HK': 'East Asia', 'MO': 'East Asia', 'MN': 'East Asia', 'KP': 'East Asia',
    # Southeast Asia
    'TH': 'Southeast Asia', 'VN': 'Southeast Asia', 'MY': 'Southeast Asia',
    'SG': 'Southeast Asia', 'ID': 'Southeast Asia', 'PH': 'Southeast Asia',
    'MM': 'Southeast Asia', 'KH': 'Southeast Asia', 'LA': 'Southeast Asia',
    'BN': 'Southeast Asia', 'TL': 'Southeast Asia',
    # South Asia
    'IN': 'South Asia', 'PK': 'South Asia', 'BD': 'South Asia', 'LK': 'South Asia',
    'NP': 'South Asia', 'BT': 'South Asia', 'MV': 'South Asia', 'AF': 'South Asia',
    # Middle East
    'TR': 'Middle East', 'IR': 'Middle East', 'IQ': 'Middle East', 'SA': 'Middle East',
    'AE': 'Middle East', 'IL': 'Middle East', 'JO': 'Middle East', 'LB': 'Middle East',
    'SY': 'Middle East', 'YE': 'Middle East', 'OM': 'Middle East', 'KW': 'Middle East',
    'QA': 'Middle East', 'BH': 'Middle East', 'PS': 'Middle East', 'CY': 'Middle East',
    # Central Asia -> Middle East
    'KZ': 'Middle East', 'UZ': 'Middle East', 'TM': 'Middle East',
    'TJ': 'Middle East', 'KG': 'Middle East',
    # Africa (all combined)
    'EG': 'Africa', 'LY': 'Africa', 'TN': 'Africa',
    'DZ': 'Africa', 'MA': 'Africa', 'SD': 'Africa',
    'NG': 'Africa', 'GH': 'Africa', 'CI': 'Africa',
    'SN': 'Africa', 'ML': 'Africa', 'BF': 'Africa',
    'NE': 'Africa', 'GN': 'Africa', 'BJ': 'Africa',
    'TG': 'Africa', 'SL': 'Africa', 'LR': 'Africa',
    'KE': 'Africa', 'TZ': 'Africa', 'UG': 'Africa',
    'ET': 'Africa', 'RW': 'Africa', 'SO': 'Africa',
    'MG': 'Africa', 'MW': 'Africa', 'MZ': 'Africa',
    'ZM': 'Africa', 'ZW': 'Africa',
    'CD': 'Africa', 'CG': 'Africa', 'CF': 'Africa',
    'CM': 'Africa', 'GA': 'Africa', 'AO': 'Africa',
    'ZA': 'Africa', 'NA': 'Africa', 'BW': 'Africa',
    # Oceania
    'AU': 'Oceania', 'NZ': 'Oceania', 'PG': 'Oceania', 'FJ': 'Oceania',
}

# Region name mapping (from data files to our standard names)
REGION_NAME_MAP = {
    'North America': 'North America',
    'South America': 'South America',
    'Central America': 'South America',
    'Caribbean': 'South America',
    'Western Europe': 'Western Europe',
    'Northern Europe': 'Northern Europe',
    'Southern Europe': 'Southern Europe',
    'Eastern Europe': 'Eastern Europe',
    'East Asia': 'East Asia',
    'Southeast Asia': 'Southeast Asia',
    'South Asia': 'South Asia',
    'Central Asia': 'Middle East',
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

# Subregion bounding boxes for coordinate-based detection
def get_subregion_from_coords(lat, lon):
    """Determine subregion from lat/lon coordinates."""
    # North America (US, Canada)
    if 25 <= lat <= 85 and -170 <= lon <= -50:
        return 'North America'
    # Central America & Caribbean -> South America
    if 7 <= lat <= 25 and -120 <= lon <= -77:
        return 'South America'
    if 10 <= lat <= 27 and -85 <= lon <= -59:
        return 'South America'
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
    # Middle East (including Central Asia)
    if 12 <= lat <= 42 and 25 <= lon <= 65:
        return 'Middle East'
    if 35 <= lat <= 55 and 50 <= lon <= 90:
        return 'Middle East'
    # Africa (all combined)
    if -35 <= lat <= 38 and -20 <= lon <= 55:
        return 'Africa'
    # Oceania
    if -50 <= lat <= 0 and 110 <= lon <= 180:
        return 'Oceania'
    return 'Unknown'


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


def extract_subregion_from_sample(sample, gt):
    """Extract subregion from a sample."""
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
        if country in COUNTRY_TO_SUBREGION:
            return COUNTRY_TO_SUBREGION[country]

    # 4. Extract from coordinates
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
            return get_subregion_from_coords(lat, lon)

    # 5. Try to extract from prompt text
    prompt = sample.get('prompt', '')
    lat, lon = extract_coords_from_prompt(prompt)
    if lat is not None and lon is not None:
        return get_subregion_from_coords(lat, lon)

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
    """Analyze all tasks and return aggregated subregion stats."""
    results_dir = Path('results')

    all_tasks = [
        ('*task1_place_association.json', 'Place Association', 'Applied'),
        ('*task3_name_disambiguation.json', 'Name Disambiguation', 'Applied'),
        ('*task5_relative_position.json', 'Relative Position', 'Applied'),
        ('*task6_proximity.json', 'Proximity', 'Applied'),
        ('*task7_route_analysis.json', 'Route Analysis', 'Applied'),
        ('*task8_boundary_analysis.json', 'Boundary Analysis', 'Applied'),
        ('*task9_spatial_patterns.json', 'Spatial Patterns', 'Applied'),
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

    # Key subregions to show (Africa combined, not split)
    subregions = [
        'North America', 'South America', 'Western Europe', 'Eastern Europe',
        'East Asia', 'South Asia', 'Southeast Asia', 'Middle East',
        'Africa', 'Oceania'
    ]

    # Aggregate stats by subregion
    overall_stats = {s: {'correct': 0, 'total': 0} for s in subregions}
    applied_stats = {s: {'correct': 0, 'total': 0} for s in subregions}
    pure_gps_stats = {s: {'correct': 0, 'total': 0} for s in subregions}

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
                subregion = extract_subregion_from_sample(result, gt)

                if not subregion or subregion == 'Unknown':
                    prompt = result.get('prompt', '')
                    for key, info in original_lookup.items():
                        if key in prompt:
                            if 'country' in info:
                                subregion = COUNTRY_TO_SUBREGION.get(info['country'])
                                if subregion:
                                    break
                            coord = info.get('coord', {})
                            lat = coord.get('lat') or coord.get('latitude')
                            lon = coord.get('lon') or coord.get('longitude') or coord.get('lng')
                            if lat is not None and lon is not None and lat != 0 and lon != 0:
                                subregion = get_subregion_from_coords(lat, lon)
                                break

                if not subregion or subregion == 'Unknown' or subregion not in subregions:
                    continue

                is_correct = result.get('correct', result.get('is_correct', False))

                overall_stats[subregion]['total'] += 1
                if is_correct:
                    overall_stats[subregion]['correct'] += 1

                if track == 'Applied':
                    applied_stats[subregion]['total'] += 1
                    if is_correct:
                        applied_stats[subregion]['correct'] += 1
                else:
                    pure_gps_stats[subregion]['total'] += 1
                    if is_correct:
                        pure_gps_stats[subregion]['correct'] += 1

    return overall_stats, applied_stats, pure_gps_stats, subregions


def main():
    print("Analyzing regional performance at subregion level...")
    overall_stats, applied_stats, pure_gps_stats, subregions = analyze_all_tasks()

    # Calculate accuracies
    overall_accs = []
    applied_accs = []
    pure_gps_accs = []

    print("\nOverall Performance by Subregion:")
    for s in subregions:
        if overall_stats[s]['total'] > 0:
            acc = 100 * overall_stats[s]['correct'] / overall_stats[s]['total']
            overall_accs.append(acc)
            print(f"  {s}: {acc:.1f}% (n={overall_stats[s]['total']})")
        else:
            overall_accs.append(0)

    print("\nApplied Track by Subregion:")
    for s in subregions:
        if applied_stats[s]['total'] > 0:
            acc = 100 * applied_stats[s]['correct'] / applied_stats[s]['total']
            applied_accs.append(acc)
            print(f"  {s}: {acc:.1f}% (n={applied_stats[s]['total']})")
        else:
            applied_accs.append(0)

    print("\nPure GPS Track by Subregion:")
    for s in subregions:
        if pure_gps_stats[s]['total'] > 0:
            acc = 100 * pure_gps_stats[s]['correct'] / pure_gps_stats[s]['total']
            pure_gps_accs.append(acc)
            print(f"  {s}: {acc:.1f}% (n={pure_gps_stats[s]['total']})")
        else:
            pure_gps_accs.append(0)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(subregions))
    width = 0.25

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    bars1 = ax.bar(x - width, applied_accs, width, label='Applied Track', color=colors[0])
    bars2 = ax.bar(x, pure_gps_accs, width, label='Pure GPS Track', color=colors[1])
    bars3 = ax.bar(x + width, overall_accs, width, label='Overall', color=colors[2])

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Subregion')
    ax.set_title('Regional Performance Across GPSBench by Subregion (14 LLMs Aggregated)')
    ax.set_xticks(x)
    ax.set_xticklabels(subregions, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 85)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = Path('paper/figures/regional_performance_subregion.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    # Print gap analysis
    print("\n=== Subregion Gap Analysis ===")
    valid_overall = [(s, a) for s, a in zip(subregions, overall_accs) if a > 0]
    if valid_overall:
        max_s, max_a = max(valid_overall, key=lambda x: x[1])
        min_s, min_a = min(valid_overall, key=lambda x: x[1])
        print(f"Overall: {max_s} ({max_a:.1f}%) vs {min_s} ({min_a:.1f}%) = {max_a - min_a:.1f}% gap")

    valid_applied = [(s, a) for s, a in zip(subregions, applied_accs) if a > 0]
    if valid_applied:
        max_s, max_a = max(valid_applied, key=lambda x: x[1])
        min_s, min_a = min(valid_applied, key=lambda x: x[1])
        print(f"Applied: {max_s} ({max_a:.1f}%) vs {min_s} ({min_a:.1f}%) = {max_a - min_a:.1f}% gap")

    valid_pure = [(s, a) for s, a in zip(subregions, pure_gps_accs) if a > 0]
    if valid_pure:
        max_s, max_a = max(valid_pure, key=lambda x: x[1])
        min_s, min_a = min(valid_pure, key=lambda x: x[1])
        print(f"Pure GPS: {max_s} ({max_a:.1f}%) vs {min_s} ({min_a:.1f}%) = {max_a - min_a:.1f}% gap")


if __name__ == '__main__':
    main()
