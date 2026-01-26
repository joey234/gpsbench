#!/usr/bin/env python3
"""
Geographic Distribution Analysis for GPSBench

Analyzes the geographic coverage and distribution of test locations across
the benchmark to identify potential biases and ensure global coverage.

Usage:
    python evaluation/analyze_geographic_distribution.py
    python evaluation/analyze_geographic_distribution.py --output stats.json
    python evaluation/analyze_geographic_distribution.py --verbose
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import argparse


def extract_coordinates_from_test(test):
    """Extract coordinate and location metadata from a test."""
    coords = []

    # Try different coordinate field patterns
    if 'coordinate' in test:
        coord = test['coordinate']
        if isinstance(coord, dict) and 'lat' in coord and 'lon' in coord:
            coords.append({
                'lat': coord['lat'],
                'lon': coord['lon'],
                'location': test.get('true_location') or test.get('location') or test.get('actual_location'),
                'country': test.get('ground_truth', {}).get('country') or test.get('ground_truth', {}).get('country_code'),
                'continent': test.get('ground_truth', {}).get('continent'),
            })

    # Check for true_coords (noise tolerance, etc.)
    if 'true_coords' in test:
        coord = test['true_coords']
        coords.append({
            'lat': coord['lat'],
            'lon': coord['lon'],
            'location': test.get('true_location'),
            'country': test.get('ground_truth', {}).get('country') or test.get('ground_truth', {}).get('country_code'),
            'continent': test.get('ground_truth', {}).get('continent'),
        })

    # Check for coordinates list (polygons, routes)
    if 'coordinates' in test and isinstance(test['coordinates'], list):
        for coord in test['coordinates']:
            if isinstance(coord, dict) and 'lat' in coord and 'lon' in coord:
                coords.append({
                    'lat': coord['lat'],
                    'lon': coord['lon'],
                    'location': None,
                    'country': None,
                    'continent': None,
                })

    # Check for points
    if 'points' in test and isinstance(test['points'], list):
        for point in test['points']:
            if isinstance(point, dict) and 'lat' in point and 'lon' in point:
                coords.append({
                    'lat': point['lat'],
                    'lon': point['lon'],
                    'location': point.get('name'),
                    'country': point.get('country') or point.get('country_code'),
                    'continent': point.get('continent'),
                })

    return coords


def analyze_hemisphere(lat, lon):
    """Determine hemisphere(s) for a coordinate."""
    hemispheres = []
    hemispheres.append('Northern' if lat >= 0 else 'Southern')
    hemispheres.append('Eastern' if lon >= 0 else 'Western')
    return hemispheres


def categorize_latitude(lat):
    """Categorize latitude into zones."""
    abs_lat = abs(lat)
    if abs_lat >= 66.5:
        return 'Polar'
    elif abs_lat >= 23.5:
        return 'Temperate'
    else:
        return 'Tropical'


def load_all_tasks(data_dir):
    """Load all task files from all layers."""
    tasks_data = {}

    for layer in ['layer_a', 'layer_b', 'layer_c', 'layer_d', 'layer_e']:
        layer_path = os.path.join(data_dir, layer)
        if not os.path.exists(layer_path):
            continue

        for filename in os.listdir(layer_path):
            if not filename.startswith('task') or not filename.endswith('.json'):
                continue
            if 'backup' in filename or 'summary' in filename:
                continue

            filepath = os.path.join(layer_path, filename)
            try:
                with open(filepath) as f:
                    data = json.load(f)
                tasks_data[f"{layer}/{filename}"] = data
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}", file=sys.stderr)

    return tasks_data


def analyze_distribution(tasks_data, verbose=False):
    """Analyze geographic distribution across all tasks."""

    # Collect all coordinates
    all_coords = []
    task_coords = defaultdict(list)

    for task_name, tests in tasks_data.items():
        for test in tests:
            coords = extract_coordinates_from_test(test)
            all_coords.extend(coords)
            task_coords[task_name].extend(coords)

    if verbose:
        print(f"Total coordinates extracted: {len(all_coords)}")

    # Continent distribution
    continent_counts = Counter()
    country_counts = Counter()
    location_counts = Counter()
    hemisphere_counts = defaultdict(int)
    latitude_zones = Counter()

    # Coordinate ranges
    lats = []
    lons = []

    for coord in all_coords:
        lat, lon = coord['lat'], coord['lon']
        lats.append(lat)
        lons.append(lon)

        # Continent
        if coord['continent']:
            continent_counts[coord['continent']] += 1

        # Country
        if coord['country']:
            country_counts[coord['country']] += 1

        # Location
        if coord['location']:
            location_counts[coord['location']] += 1

        # Hemisphere
        for h in analyze_hemisphere(lat, lon):
            hemisphere_counts[h] += 1

        # Latitude zones
        latitude_zones[categorize_latitude(lat)] += 1

    # Calculate statistics
    stats = {
        'total_coordinates': len(all_coords),
        'unique_locations': len(location_counts),
        'unique_countries': len(country_counts),
        'unique_continents': len(continent_counts),

        'continents': dict(continent_counts.most_common()),
        'top_20_countries': dict(country_counts.most_common(20)),
        'all_countries': dict(country_counts.most_common()),

        'hemispheres': dict(hemisphere_counts),
        'latitude_zones': dict(latitude_zones),

        'coordinate_ranges': {
            'latitude': {
                'min': min(lats) if lats else None,
                'max': max(lats) if lats else None,
                'mean': sum(lats) / len(lats) if lats else None,
            },
            'longitude': {
                'min': min(lons) if lons else None,
                'max': max(lons) if lons else None,
                'mean': sum(lons) / len(lons) if lons else None,
            }
        },

        'coverage_by_task': {
            task: {
                'total_coords': len(coords),
                'continents': len(set(c['continent'] for c in coords if c['continent'])),
                'countries': len(set(c['country'] for c in coords if c['country'])),
            }
            for task, coords in task_coords.items()
        }
    }

    return stats


def print_report(stats):
    """Print a human-readable distribution report."""

    print("=" * 80)
    print("GPSBENCH GEOGRAPHIC DISTRIBUTION ANALYSIS")
    print("=" * 80)

    print(f"\n📊 OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total coordinate points:    {stats['total_coordinates']:,}")
    print(f"Unique locations:           {stats['unique_locations']:,}")
    print(f"Unique countries:           {stats['unique_countries']:,}")
    print(f"Unique continents:          {stats['unique_continents']:,}")

    # Continent distribution
    print(f"\n🌍 CONTINENT DISTRIBUTION")
    print("-" * 80)
    total_continent = sum(stats['continents'].values())
    for continent, count in sorted(stats['continents'].items(), key=lambda x: -x[1]):
        pct = (count / total_continent * 100) if total_continent > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"{continent:20s} {count:5,} ({pct:5.1f}%) {bar}")

    # Hemisphere distribution
    print(f"\n🌐 HEMISPHERE DISTRIBUTION")
    print("-" * 80)
    total_coords = stats['total_coordinates']
    for hemisphere, count in sorted(stats['hemispheres'].items()):
        pct = (count / total_coords * 100) if total_coords > 0 else 0
        print(f"{hemisphere:20s} {count:5,} ({pct:5.1f}%)")

    # Latitude zones
    print(f"\n🌡️  LATITUDE ZONES")
    print("-" * 80)
    for zone, count in sorted(stats['latitude_zones'].items(), key=lambda x: -x[1]):
        pct = (count / total_coords * 100) if total_coords > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"{zone:20s} {count:5,} ({pct:5.1f}%) {bar}")

    # Top 20 countries
    print(f"\n🏳️  TOP 20 COUNTRIES")
    print("-" * 80)
    for i, (country, count) in enumerate(stats['top_20_countries'].items(), 1):
        pct = (count / total_coords * 100) if total_coords > 0 else 0
        print(f"{i:2d}. {country:30s} {count:5,} ({pct:4.1f}%)")

    # Coordinate ranges
    print(f"\n📍 COORDINATE RANGES")
    print("-" * 80)
    lat_range = stats['coordinate_ranges']['latitude']
    lon_range = stats['coordinate_ranges']['longitude']
    print(f"Latitude:  [{lat_range['min']:7.2f}°, {lat_range['max']:7.2f}°] (mean: {lat_range['mean']:7.2f}°)")
    print(f"Longitude: [{lon_range['min']:7.2f}°, {lon_range['max']:7.2f}°] (mean: {lon_range['mean']:7.2f}°)")

    # Geographic biases
    print(f"\n⚠️  POTENTIAL BIASES")
    print("-" * 80)

    # Check for continent imbalance
    continent_pcts = {c: (count / total_continent * 100) for c, count in stats['continents'].items()}
    if continent_pcts:
        max_continent = max(continent_pcts, key=continent_pcts.get)
        min_continent = min(continent_pcts, key=continent_pcts.get)
        ratio = continent_pcts[max_continent] / continent_pcts[min_continent] if continent_pcts[min_continent] > 0 else float('inf')

        if ratio > 10:
            print(f"⚠️  High continent imbalance: {max_continent} ({continent_pcts[max_continent]:.1f}%) vs {min_continent} ({continent_pcts[min_continent]:.1f}%)")
        else:
            print(f"✅ Reasonable continent balance (max ratio: {ratio:.1f}x)")

    # Check hemisphere balance
    if 'Northern' in stats['hemispheres'] and 'Southern' in stats['hemispheres']:
        north_pct = stats['hemispheres']['Northern'] / total_coords * 100
        south_pct = stats['hemispheres']['Southern'] / total_coords * 100
        if abs(north_pct - south_pct) > 30:
            print(f"⚠️  Hemisphere imbalance: N={north_pct:.1f}% vs S={south_pct:.1f}%")
        else:
            print(f"✅ Reasonable hemisphere balance: N={north_pct:.1f}% vs S={south_pct:.1f}%")

    # Check if polar regions are covered
    polar_pct = (stats['latitude_zones'].get('Polar', 0) / total_coords * 100) if total_coords > 0 else 0
    if polar_pct < 1:
        print(f"⚠️  Low polar coverage: {polar_pct:.1f}% (may be intentional)")

    # Country concentration
    if stats['all_countries']:
        top_5_pct = sum(count for _, count in list(stats['top_20_countries'].items())[:5]) / total_coords * 100
        if top_5_pct > 50:
            print(f"⚠️  High country concentration: Top 5 countries = {top_5_pct:.1f}%")
        else:
            print(f"✅ Good country diversity: Top 5 countries = {top_5_pct:.1f}%")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze geographic distribution of GPSBench test locations'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--output',
        help='Save statistics to JSON file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Find data directory
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        # Try relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_paths = [
            os.path.join(script_dir, '..', data_dir),
            os.path.join(os.getcwd(), data_dir),
        ]
        for path in potential_paths:
            if os.path.exists(path):
                data_dir = path
                break

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Loading tasks from: {data_dir}")

    # Load all tasks
    tasks_data = load_all_tasks(data_dir)

    if args.verbose:
        print(f"Loaded {len(tasks_data)} task files")

    # Analyze distribution
    stats = analyze_distribution(tasks_data, verbose=args.verbose)

    # Print report
    print_report(stats)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✅ Statistics saved to: {args.output}")


if __name__ == '__main__':
    main()
