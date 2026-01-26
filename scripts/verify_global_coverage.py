#!/usr/bin/env python3
"""
Verify Global Geographic Coverage in Generated Data

Analyzes the geographic distribution of cities in generated task data
to ensure balanced representation across all continents.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Continent mapping (ISO country codes)
CONTINENT_MAP = {
    'Africa': ['DZ', 'AO', 'BJ', 'BW', 'BF', 'BI', 'CM', 'CV', 'CF', 'TD', 'KM', 'CG', 'CD', 'CI', 'DJ', 'EG', 'GQ', 'ER', 'ET', 'GA', 'GM', 'GH', 'GN', 'GW', 'KE', 'LS', 'LR', 'LY', 'MG', 'MW', 'ML', 'MR', 'MU', 'YT', 'MA', 'MZ', 'NA', 'NE', 'NG', 'RE', 'RW', 'SH', 'ST', 'SN', 'SC', 'SL', 'SO', 'ZA', 'SS', 'SD', 'SZ', 'TZ', 'TG', 'TN', 'UG', 'ZM', 'ZW'],
    'Asia': ['AF', 'AM', 'AZ', 'BH', 'BD', 'BT', 'BN', 'KH', 'CN', 'CX', 'CC', 'IO', 'GE', 'HK', 'IN', 'ID', 'IR', 'IQ', 'IL', 'JP', 'JO', 'KZ', 'KW', 'KG', 'LA', 'LB', 'MO', 'MY', 'MV', 'MN', 'MM', 'NP', 'KP', 'OM', 'PK', 'PS', 'PH', 'QA', 'SA', 'SG', 'KR', 'LK', 'SY', 'TW', 'TJ', 'TH', 'TL', 'TR', 'TM', 'AE', 'UZ', 'VN', 'YE'],
    'Europe': ['AX', 'AL', 'AD', 'AT', 'BY', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FO', 'FI', 'FR', 'DE', 'GI', 'GR', 'GG', 'VA', 'HU', 'IS', 'IE', 'IM', 'IT', 'JE', 'XK', 'LV', 'LI', 'LT', 'LU', 'MK', 'MT', 'MD', 'MC', 'ME', 'NL', 'NO', 'PL', 'PT', 'RO', 'RU', 'SM', 'RS', 'SK', 'SI', 'ES', 'SJ', 'SE', 'CH', 'UA', 'GB'],
    'North America': ['AI', 'AG', 'AW', 'BS', 'BB', 'BZ', 'BM', 'BQ', 'CA', 'KY', 'CR', 'CU', 'CW', 'DM', 'DO', 'SV', 'GL', 'GD', 'GP', 'GT', 'HT', 'HN', 'JM', 'MQ', 'MX', 'MS', 'NI', 'PA', 'PM', 'PR', 'BL', 'KN', 'LC', 'MF', 'VC', 'SX', 'TT', 'TC', 'US', 'VG', 'VI'],
    'South America': ['AR', 'BO', 'BV', 'BR', 'CL', 'CO', 'EC', 'FK', 'GF', 'GY', 'PY', 'PE', 'GS', 'SR', 'UY', 'VE'],
    'Oceania': ['AS', 'AU', 'CK', 'FJ', 'PF', 'GU', 'KI', 'MH', 'FM', 'NR', 'NC', 'NZ', 'NU', 'NF', 'MP', 'PW', 'PG', 'PN', 'WS', 'SB', 'TK', 'TO', 'TV', 'UM', 'VU', 'WF'],
    'Antarctica': ['AQ']
}

# Country name to code mapping (simplified - covers most common countries in data)
COUNTRY_NAME_TO_CODE = {
    'Kenya': 'KE', 'India': 'IN', 'China': 'CN', 'Japan': 'JP', 'USA': 'US', 'Brazil': 'BR',
    'Nigeria': 'NG', 'Egypt': 'EG', 'South Africa': 'ZA', 'France': 'FR', 'Germany': 'DE',
    'United Kingdom': 'GB', 'UK': 'GB', 'Spain': 'ES', 'Italy': 'IT', 'Russia': 'RU',
    'Canada': 'CA', 'Mexico': 'MX', 'Argentina': 'AR', 'Australia': 'AU', 'Indonesia': 'ID',
    'Pakistan': 'PK', 'Bangladesh': 'BD', 'Iran': 'IR', 'Turkey': 'TR', 'Thailand': 'TH',
    'Vietnam': 'VN', 'Philippines': 'PH', 'Colombia': 'CO', 'Poland': 'PL', 'Ukraine': 'UA',
    'Algeria': 'DZ', 'Morocco': 'MA', 'Peru': 'PE', 'Venezuela': 'VE', 'Malaysia': 'MY',
    'Ghana': 'GH', 'Ethiopia': 'ET', 'Tanzania': 'TZ', 'Uganda': 'UG', 'Angola': 'AO',
    'Chile': 'CL', 'Ecuador': 'EC', 'Netherlands': 'NL', 'Portugal': 'PT', 'Greece': 'GR',
    'Belgium': 'BE', 'Sweden': 'SE', 'Austria': 'AT', 'Switzerland': 'CH', 'Israel': 'IL',
    'Singapore': 'SG', 'Hong Kong': 'HK', 'New Zealand': 'NZ', 'Norway': 'NO', 'Denmark': 'DK',
    'Finland': 'FI', 'Ireland': 'IE', 'South Korea': 'KR', 'North Korea': 'KP', 'Taiwan': 'TW',
}

def get_continent(country_code):
    """Get continent from country code"""
    for continent, codes in CONTINENT_MAP.items():
        if country_code in codes:
            return continent
    return 'Unknown'

def extract_countries_from_geonames():
    """Extract country to name mapping from GeoNames"""
    country_to_name = {}
    with open('data/raw/cities15000.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 9:
                country_code = parts[8]
                city_name = parts[1]
                # Store first city name as representative
                if country_code not in country_to_name:
                    country_to_name[country_code] = city_name
    return country_to_name

def extract_countries_from_task(task_file):
    """Extract country codes from a task file"""
    countries = []

    with open(task_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract from various fields
    for item in data:
        # Check different possible fields
        for field in ['point_a', 'point_b', 'reference_point', 'start', 'end', 'waypoint']:
            if field in item and isinstance(item[field], dict):
                name = item[field].get('name', '')
                # Try to extract country from name (e.g., "Tehran, Iran")
                if ', ' in name:
                    country_part = name.split(', ')[-1]
                    countries.append(country_part)

        # For points arrays
        if 'points' in item:
            for point in item['points']:
                if isinstance(point, dict):
                    name = point.get('name', '')
                    if ', ' in name:
                        country_part = name.split(', ')[-1]
                        countries.append(country_part)

        # For candidates
        if 'candidates' in item:
            for cand in item['candidates']:
                if isinstance(cand, dict) and 'country' in cand:
                    countries.append(cand['country'])

        # Direct country field
        if 'country' in item:
            countries.append(item['country'])

    return countries

def analyze_task_coverage(task_file):
    """Analyze geographic coverage of a single task"""
    countries = extract_countries_from_task(task_file)

    if not countries:
        return None

    # Map to country codes
    continent_counts = Counter()

    for country in countries:
        # Try direct code lookup
        continent = get_continent(country)
        if continent != 'Unknown':
            continent_counts[continent] += 1
        # Try country name lookup
        elif country in COUNTRY_NAME_TO_CODE:
            code = COUNTRY_NAME_TO_CODE[country]
            continent = get_continent(code)
            if continent != 'Unknown':
                continent_counts[continent] += 1

    total = sum(continent_counts.values())
    coverage = {cont: (count / total * 100) if total > 0 else 0
                for cont, count in continent_counts.items()}

    return coverage, continent_counts, total

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify geographic coverage in GPSBench data')
    parser.add_argument('--data-dir', default='data', help='Data directory to analyze (default: data)')
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    data_path = base_path / args.data_dir

    if not data_path.exists():
        print(f"❌ Error: {data_path} does not exist")
        print(f"   Run ./generate_v2_safe.sh first to generate data_v2/")
        sys.exit(1)

    print("="*80)
    print("GEOGRAPHIC COVERAGE VERIFICATION")
    print("="*80)
    print(f"Analyzing: {data_path}")
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

    # Analyze a sample task
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
        task_name = task_file.split('/')[-1].replace('.json', '')

        if not filepath.exists():
            print(f"\n{task_name}: File not found")
            continue

        result = analyze_task_coverage(filepath)
        if result is None:
            print(f"\n{task_name}: No country data found")
            continue

        coverage, counts, total = result

        print(f"\n{task_name}:")
        print(f"  Total samples analyzed: {total}")
        print(f"  Continent distribution:")

        for cont in sorted(target.keys()):
            actual = coverage.get(cont, 0)
            expected = target[cont]
            diff = actual - expected
            status = "✅" if abs(diff) < 10 else "⚠️"
            count = counts.get(cont, 0)
            print(f"    {status} {cont:20} {actual:5.1f}% (target: {expected:5.1f}%, diff: {diff:+6.1f}%, n={count})")

    print("\n" + "="*80)
    print("Notes:")
    print("  ✅ = Within 10% of target")
    print("  ⚠️  = More than 10% off target")
    print("="*80)

if __name__ == "__main__":
    main()
