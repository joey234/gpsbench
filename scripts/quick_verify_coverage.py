#!/usr/bin/env python3
"""
Quick Geographic Coverage Verification
Verifies coverage by looking up cities in GeoNames database
"""

import json
import sys
from pathlib import Path
from collections import Counter
import argparse

# Continent mapping
CONTINENT_MAP = {
    'Africa': ['DZ', 'AO', 'BJ', 'BW', 'BF', 'BI', 'CM', 'CV', 'CF', 'TD', 'KM', 'CG', 'CD', 'CI', 'DJ', 'EG', 'GQ', 'ER', 'ET', 'GA', 'GM', 'GH', 'GN', 'GW', 'KE', 'LS', 'LR', 'LY', 'MG', 'MW', 'ML', 'MR', 'MU', 'YT', 'MA', 'MZ', 'NA', 'NE', 'NG', 'RE', 'RW', 'SH', 'ST', 'SN', 'SC', 'SL', 'SO', 'ZA', 'SS', 'SD', 'SZ', 'TZ', 'TG', 'TN', 'UG', 'ZM', 'ZW'],
    'Asia': ['AF', 'AM', 'AZ', 'BH', 'BD', 'BT', 'BN', 'KH', 'CN', 'CX', 'CC', 'IO', 'GE', 'HK', 'IN', 'ID', 'IR', 'IQ', 'IL', 'JP', 'JO', 'KZ', 'KW', 'KG', 'LA', 'LB', 'MO', 'MY', 'MV', 'MN', 'MM', 'NP', 'KP', 'OM', 'PK', 'PS', 'PH', 'QA', 'SA', 'SG', 'KR', 'LK', 'SY', 'TW', 'TJ', 'TH', 'TL', 'TR', 'TM', 'AE', 'UZ', 'VN', 'YE'],
    'Europe': ['AX', 'AL', 'AD', 'AT', 'BY', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FO', 'FI', 'FR', 'DE', 'GI', 'GR', 'GG', 'VA', 'HU', 'IS', 'IE', 'IM', 'IT', 'JE', 'XK', 'LV', 'LI', 'LT', 'LU', 'MK', 'MT', 'MD', 'MC', 'ME', 'NL', 'NO', 'PL', 'PT', 'RO', 'RU', 'SM', 'RS', 'SK', 'SI', 'ES', 'SJ', 'SE', 'CH', 'UA', 'GB'],
    'North America': ['AI', 'AG', 'AW', 'BS', 'BB', 'BZ', 'BM', 'BQ', 'CA', 'KY', 'CR', 'CU', 'CW', 'DM', 'DO', 'SV', 'GL', 'GD', 'GP', 'GT', 'HT', 'HN', 'JM', 'MQ', 'MX', 'MS', 'NI', 'PA', 'PM', 'PR', 'BL', 'KN', 'LC', 'MF', 'VC', 'SX', 'TT', 'TC', 'US', 'VG', 'VI'],
    'South America': ['AR', 'BO', 'BV', 'BR', 'CL', 'CO', 'EC', 'FK', 'GF', 'GY', 'PY', 'PE', 'GS', 'SR', 'UY', 'VE'],
    'Oceania': ['AS', 'AU', 'CK', 'FJ', 'PF', 'GU', 'KI', 'MH', 'FM', 'NR', 'NC', 'NZ', 'NU', 'NF', 'MP', 'PW', 'PG', 'PN', 'WS', 'SB', 'TK', 'TO', 'TV', 'UM', 'VU', 'WF'],
}

def get_continent(country_code):
    """Get continent from country code"""
    for continent, codes in CONTINENT_MAP.items():
        if country_code in codes:
            return continent
    return 'Unknown'

def load_geonames_lookup(geonames_path):
    """Load GeoNames and create lookup by city name"""
    print("Loading GeoNames database...")
    city_lookup = {}

    with open(geonames_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 15:
                try:
                    name = parts[1]
                    lat = float(parts[4])
                    lon = float(parts[5])
                    country = parts[8]

                    # Create key with rounded coords for approximate matching
                    key = (name, round(lat, 2), round(lon, 2))
                    city_lookup[key] = country
                except (ValueError, IndexError):
                    continue

    print(f"Loaded {len(city_lookup)} cities")
    return city_lookup

def extract_cities_from_task(task_file, city_lookup):
    """Extract cities from task file and look up continents"""
    continent_counts = Counter()
    matched = 0
    total = 0

    with open(task_file, 'r') as f:
        data = json.load(f)

    for item in data:
        # Check different possible fields for city data
        for field in ['point_a', 'point_b', 'reference_point', 'start', 'end', 'waypoint']:
            if field in item and isinstance(item[field], dict):
                total += 1
                name = item[field].get('name', '')
                lat = item[field].get('lat')
                lon = item[field].get('lon')

                if name and lat is not None and lon is not None:
                    # Try to find in lookup
                    key = (name, round(lat, 2), round(lon, 2))
                    if key in city_lookup:
                        country = city_lookup[key]
                        continent = get_continent(country)
                        if continent != 'Unknown':
                            continent_counts[continent] += 1
                            matched += 1

        # For points arrays
        if 'points' in item:
            for point in item['points']:
                if isinstance(point, dict):
                    total += 1
                    name = point.get('name', '')
                    lat = point.get('lat')
                    lon = point.get('lon')

                    if name and lat is not None and lon is not None:
                        key = (name, round(lat, 2), round(lon, 2))
                        if key in city_lookup:
                            country = city_lookup[key]
                            continent = get_continent(country)
                            if continent != 'Unknown':
                                continent_counts[continent] += 1
                                matched += 1

    total_matched = sum(continent_counts.values())
    coverage = {cont: (count / total_matched * 100) if total_matched > 0 else 0
                for cont, count in continent_counts.items()}

    return coverage, continent_counts, total_matched, matched, total

def main():
    parser = argparse.ArgumentParser(description='Quick verify geographic coverage')
    parser.add_argument('--data-dir', default='data_v2', help='Data directory')
    parser.add_argument('--geonames', default='data/raw/cities15000.txt', help='GeoNames file')
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    data_path = base_path / args.data_dir
    geonames_path = base_path / args.geonames

    if not data_path.exists():
        print(f"❌ Error: {data_path} does not exist")
        sys.exit(1)

    if not geonames_path.exists():
        print(f"❌ Error: {geonames_path} does not exist")
        sys.exit(1)

    # Load GeoNames lookup
    city_lookup = load_geonames_lookup(geonames_path)

    print("\n" + "="*80)
    print("GEOGRAPHIC COVERAGE VERIFICATION")
    print("="*80)
    print(f"Data directory: {data_path}")
    print("="*80)

    # Target distribution
    target = {
        'Africa': 20,
        'Asia': 25,
        'Europe': 20,
        'North America': 15,
        'South America': 15,
        'Oceania': 5
    }

    print("\nTarget Distribution:")
    for cont in sorted(target.keys()):
        print(f"  {cont:20} {target[cont]:5.1f}%")

    # Analyze sample tasks
    tasks_to_check = [
        'track_pure_gps/task4_distance_calculation.json',
        'track_pure_gps/task5_bearing_computation.json',
        'track_applied/task7_route_analysis.json',
    ]

    print("\n" + "="*80)
    print("SAMPLE TASK ANALYSIS")
    print("="*80)

    for task_file in tasks_to_check:
        filepath = data_path / task_file

        if not filepath.exists():
            print(f"\n{task_file}: File not found")
            continue

        coverage, counts, total_matched, matched, total = extract_cities_from_task(filepath, city_lookup)

        print(f"\n{task_file}:")
        print(f"  Total city references: {total}")
        print(f"  Matched in GeoNames: {matched} ({matched/total*100:.1f}%)")
        print(f"  With continent data: {total_matched}")

        if total_matched > 0:
            print(f"  Continent distribution:")
            for cont in sorted(target.keys()):
                actual = coverage.get(cont, 0)
                target_val = target[cont]
                diff = actual - target_val
                count = counts.get(cont, 0)

                if abs(diff) <= 10:
                    status = "✅"
                else:
                    status = "⚠️ "

                print(f"    {status} {cont:20} {actual:5.1f}% (target: {target_val:5.1f}%, diff: {diff:+6.1f}%, n={count})")
        else:
            print(f"  ⚠️  No continent data extracted")

    print("\n" + "="*80)
    print("Notes:")
    print("  ✅ = Within 10% of target")
    print("  ⚠️  = More than 10% off target")
    print("="*80)

if __name__ == "__main__":
    main()
