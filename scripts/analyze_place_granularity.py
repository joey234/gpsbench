#!/usr/bin/env python3
"""
Analyze Place Association at multiple granularity levels:
- Continent level: Did the model identify the correct continent?
- Country level: Did the model identify the correct country?
- City level: Did the model identify the correct city?

Uses existing model outputs without re-running.
"""

import json
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse

# Continent aliases for matching - includes common country names in each continent
CONTINENT_ALIASES = {
    'Asia': ['Asia', 'Asian', 'Middle East', 'Southeast Asia', 'East Asia', 'South Asia', 'Central Asia',
             'China', 'Japan', 'India', 'Indonesia', 'Philippines', 'Vietnam', 'Thailand', 'Malaysia',
             'Singapore', 'Korea', 'Taiwan', 'Hong Kong', 'Pakistan', 'Bangladesh', 'Myanmar', 'Cambodia',
             'Iran', 'Iraq', 'Saudi', 'Turkey', 'Israel', 'UAE', 'Emirates', 'Qatar', 'Kuwait', 'Jordan',
             'Lebanon', 'Syria', 'Yemen', 'Oman', 'Kazakhstan', 'Uzbekistan', 'Turkmenistan', 'Kyrgyzstan',
             'Tajikistan', 'Afghanistan', 'Nepal', 'Sri Lanka', 'Mongolia', 'Azerbaijan', 'Armenia', 'Georgia'],
    'Europe': ['Europe', 'European', 'Western Europe', 'Eastern Europe', 'EU',
               'France', 'Germany', 'Italy', 'Spain', 'UK', 'Britain', 'England', 'Scotland', 'Wales',
               'Netherlands', 'Belgium', 'Switzerland', 'Austria', 'Portugal', 'Greece', 'Poland',
               'Sweden', 'Norway', 'Denmark', 'Finland', 'Ireland', 'Czech', 'Hungary', 'Romania',
               'Bulgaria', 'Croatia', 'Serbia', 'Slovenia', 'Slovakia', 'Ukraine', 'Russia', 'Belarus',
               'Lithuania', 'Latvia', 'Estonia', 'Iceland', 'Cyprus', 'Malta', 'Luxembourg', 'Monaco',
               'Albania', 'Bosnia', 'Montenegro', 'Macedonia', 'Kosovo', 'Moldova'],
    'America': ['America', 'Americas', 'North America', 'South America', 'Central America', 'Latin America',
                'USA', 'United States', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Colombia', 'Peru',
                'Venezuela', 'Chile', 'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay', 'Guyana', 'Suriname',
                'Cuba', 'Dominican', 'Haiti', 'Jamaica', 'Puerto Rico', 'Trinidad', 'Panama', 'Costa Rica',
                'Guatemala', 'Honduras', 'El Salvador', 'Nicaragua', 'Belize'],
    'Africa': ['Africa', 'African', 'Sub-Saharan', 'North Africa',
               'Nigeria', 'Egypt', 'South Africa', 'Kenya', 'Ethiopia', 'Tanzania', 'Ghana', 'Morocco',
               'Algeria', 'Tunisia', 'Libya', 'Sudan', 'Uganda', 'Cameroon', 'Ivory Coast', 'Senegal',
               'Mali', 'Niger', 'Chad', 'Burkina', 'Benin', 'Togo', 'Rwanda', 'Madagascar', 'Zimbabwe',
               'Zambia', 'Malawi', 'Mozambique', 'Angola', 'Namibia', 'Botswana', 'Congo', 'Somalia',
               'Eritrea', 'Mauritius', 'Liberia', 'Sierra Leone', 'Gambia', 'Guinea'],
    'Australia': ['Australia', 'Oceania', 'Australian', 'Australasia', 'New Zealand', 'Papua New Guinea',
                  'Fiji', 'Solomon Islands', 'Vanuatu', 'Samoa', 'Tonga'],
    'Pacific': ['Pacific', 'Oceania', 'Pacific Islands', 'Guam', 'Hawaii', 'Palau', 'Micronesia'],
    'Indian': ['Indian Ocean', 'Maldives', 'Seychelles', 'Mauritius', 'Reunion'],
    'Atlantic': ['Atlantic', 'Bermuda', 'Azores', 'Canary Islands', 'Cape Verde'],
    'Arctic': ['Arctic', 'Greenland', 'Svalbard'],
}

# Complete country code to name mapping (all 159 countries in dataset)
COUNTRY_NAMES = {
    'AE': 'United Arab Emirates', 'AF': 'Afghanistan', 'AI': 'Anguilla',
    'AL': 'Albania', 'AM': 'Armenia', 'AO': 'Angola', 'AR': 'Argentina',
    'AT': 'Austria', 'AU': 'Australia', 'AZ': 'Azerbaijan', 'BA': 'Bosnia',
    'BD': 'Bangladesh', 'BE': 'Belgium', 'BF': 'Burkina Faso', 'BG': 'Bulgaria',
    'BJ': 'Benin', 'BO': 'Bolivia', 'BR': 'Brazil', 'BT': 'Bhutan',
    'BW': 'Botswana', 'BY': 'Belarus', 'CA': 'Canada', 'CD': 'Congo',
    'CF': 'Central African Republic', 'CG': 'Congo', 'CH': 'Switzerland',
    'CI': 'Ivory Coast', 'CL': 'Chile', 'CM': 'Cameroon', 'CN': 'China',
    'CO': 'Colombia', 'CR': 'Costa Rica', 'CU': 'Cuba', 'CY': 'Cyprus',
    'CZ': 'Czech', 'DE': 'Germany', 'DK': 'Denmark', 'DO': 'Dominican',
    'DZ': 'Algeria', 'EC': 'Ecuador', 'EG': 'Egypt', 'ES': 'Spain',
    'ET': 'Ethiopia', 'FI': 'Finland', 'FR': 'France', 'GB': 'United Kingdom',
    'GE': 'Georgia', 'GH': 'Ghana', 'GM': 'Gambia', 'GN': 'Guinea',
    'GP': 'Guadeloupe', 'GR': 'Greece', 'GT': 'Guatemala', 'GU': 'Guam',
    'GY': 'Guyana', 'HK': 'Hong Kong', 'HN': 'Honduras', 'HR': 'Croatia',
    'HT': 'Haiti', 'HU': 'Hungary', 'ID': 'Indonesia', 'IE': 'Ireland',
    'IL': 'Israel', 'IN': 'India', 'IQ': 'Iraq', 'IR': 'Iran', 'IS': 'Iceland',
    'IT': 'Italy', 'JO': 'Jordan', 'JP': 'Japan', 'KE': 'Kenya',
    'KG': 'Kyrgyzstan', 'KH': 'Cambodia', 'KP': 'North Korea', 'KR': 'South Korea',
    'KW': 'Kuwait', 'KZ': 'Kazakhstan', 'LB': 'Lebanon', 'LK': 'Sri Lanka',
    'LR': 'Liberia', 'LS': 'Lesotho', 'LT': 'Lithuania', 'LV': 'Latvia',
    'LY': 'Libya', 'MA': 'Morocco', 'ME': 'Montenegro', 'MG': 'Madagascar',
    'MK': 'North Macedonia', 'ML': 'Mali', 'MM': 'Myanmar', 'MN': 'Mongolia',
    'MO': 'Macau', 'MR': 'Mauritania', 'MT': 'Malta', 'MU': 'Mauritius',
    'MW': 'Malawi', 'MX': 'Mexico', 'MY': 'Malaysia', 'MZ': 'Mozambique',
    'NA': 'Namibia', 'NE': 'Niger', 'NF': 'Norfolk Island', 'NG': 'Nigeria',
    'NI': 'Nicaragua', 'NL': 'Netherlands', 'NO': 'Norway', 'NP': 'Nepal',
    'NZ': 'New Zealand', 'OM': 'Oman', 'PA': 'Panama', 'PE': 'Peru',
    'PG': 'Papua New Guinea', 'PH': 'Philippines', 'PK': 'Pakistan', 'PL': 'Poland',
    'PS': 'Palestine', 'PT': 'Portugal', 'PY': 'Paraguay', 'QA': 'Qatar',
    'RE': 'Reunion', 'RO': 'Romania', 'RS': 'Serbia', 'RU': 'Russia',
    'RW': 'Rwanda', 'SA': 'Saudi Arabia', 'SD': 'Sudan', 'SE': 'Sweden',
    'SG': 'Singapore', 'SI': 'Slovenia', 'SK': 'Slovakia', 'SN': 'Senegal',
    'SO': 'Somalia', 'SS': 'South Sudan', 'SV': 'El Salvador', 'SY': 'Syria',
    'TD': 'Chad', 'TG': 'Togo', 'TH': 'Thailand', 'TJ': 'Tajikistan',
    'TL': 'East Timor', 'TM': 'Turkmenistan', 'TN': 'Tunisia', 'TR': 'Turkey',
    'TT': 'Trinidad', 'TV': 'Tuvalu', 'TW': 'Taiwan', 'TZ': 'Tanzania',
    'UA': 'Ukraine', 'UG': 'Uganda', 'US': 'United States', 'UY': 'Uruguay',
    'UZ': 'Uzbekistan', 'VE': 'Venezuela', 'VN': 'Vietnam', 'XK': 'Kosovo',
    'YE': 'Yemen', 'ZA': 'South Africa', 'ZM': 'Zambia', 'ZW': 'Zimbabwe',
}

# Alternative country names for better matching
COUNTRY_ALIASES = {
    'United States': ['USA', 'US', 'United States of America', 'America', 'U.S.', 'U.S.A.'],
    'United Kingdom': ['UK', 'Britain', 'Great Britain', 'England', 'Scotland', 'Wales', 'U.K.'],
    'United Arab Emirates': ['UAE', 'Emirates', 'Dubai', 'Abu Dhabi'],
    'Russia': ['Russian Federation', 'USSR', 'Soviet'],
    'China': ['PRC', "People's Republic of China", 'Zhongguo', 'Chinese'],
    'South Korea': ['Korea', 'Republic of Korea', 'ROK', 'Korean'],
    'North Korea': ['DPRK', 'Democratic People\'s Republic of Korea'],
    'Czech': ['Czech Republic', 'Czechia'],
    'Myanmar': ['Burma', 'Burmese'],
    'Iran': ['Persia', 'Persian', 'Islamic Republic of Iran'],
    'Taiwan': ['Chinese Taipei', 'Republic of China', 'ROC'],
    'Vietnam': ['Viet Nam'],
    'Ivory Coast': ['Cote d\'Ivoire', 'Côte d\'Ivoire'],
    'Bosnia': ['Bosnia and Herzegovina', 'Bosnia Herzegovina'],
    'Dominican': ['Dominican Republic'],
    'Congo': ['Democratic Republic of the Congo', 'DRC', 'Republic of the Congo'],
    'Palestine': ['Palestinian', 'West Bank', 'Gaza'],
    'Hong Kong': ['HK'],
    'Macau': ['Macao'],
    'North Macedonia': ['Macedonia', 'FYROM'],
    'East Timor': ['Timor-Leste', 'Timor Leste'],
    'Trinidad': ['Trinidad and Tobago'],
}


def load_admin1_names(admin1_file):
    """Load admin1 (region/state) names from admin1CodesASCII.txt."""
    admin1_names = {}
    with open(admin1_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            try:
                code = row[0]  # e.g., "CN.22"
                name = row[1]  # e.g., "Shandong"
                admin1_names[code] = name
            except:
                continue
    return admin1_names


def load_city_data(cities_file, admin1_names):
    """Load city data including continent and region info."""
    city_data = {}
    with open(cities_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            try:
                city_name = row[1]
                country_code = row[8]
                admin1_code = row[10]
                timezone = row[17]

                # Extract continent from timezone
                continent = timezone.split('/')[0] if '/' in timezone else 'Unknown'

                # Get region name from admin1 code
                admin1_key = f"{country_code}.{admin1_code}"
                region = admin1_names.get(admin1_key, '')

                # Use city+country as key
                key = f"{city_name.lower()}_{country_code.lower()}"
                city_data[key] = {
                    'continent': continent,
                    'region': region,
                }
            except:
                continue
    return city_data


def get_country_name(code):
    """Get country name from code."""
    return COUNTRY_NAMES.get(code, code)


def normalize_text(text):
    """Normalize text for comparison."""
    if not text:
        return ''
    return re.sub(r'[^a-z0-9]', '', text.lower())


def check_continent_match(response, continent):
    """Check if response mentions the correct continent."""
    response_norm = normalize_text(response)

    # Check direct match
    if normalize_text(continent) in response_norm:
        return True

    # Check aliases
    aliases = CONTINENT_ALIASES.get(continent, [])
    for alias in aliases:
        if normalize_text(alias) in response_norm:
            return True

    return False


def check_region_match(response, region):
    """Check if response mentions the correct region/state/province."""
    if not region:
        return False

    response_norm = normalize_text(response)
    region_norm = normalize_text(region)

    # Check if region name appears in response
    if region_norm and region_norm in response_norm:
        return True

    return False


def check_country_match(response, country_code):
    """Check if response mentions the correct country."""
    response_lower = response.lower()

    country_name = get_country_name(country_code)

    # Check main name
    if normalize_text(country_name) in normalize_text(response):
        return True

    # Check code
    if f' {country_code.lower()} ' in f' {response_lower} ':
        return True
    if response_lower.endswith(f', {country_code.lower()}'):
        return True

    # Check aliases
    for main, aliases in COUNTRY_ALIASES.items():
        if normalize_text(main) == normalize_text(country_name):
            for alias in aliases:
                if normalize_text(alias) in normalize_text(response):
                    return True

    return False


def check_city_match(response, city_name, alternate_names):
    """Check if response mentions the correct city."""
    response_norm = normalize_text(response)

    # Check main name
    if normalize_text(city_name) in response_norm:
        return True

    # Check alternate names
    for alt in (alternate_names or []):
        if normalize_text(alt) in response_norm:
            return True

    return False


def extract_answer(response):
    """Extract the FINAL ANSWER from response."""
    match = re.search(r'FINAL ANSWER:?\s*([^\n]+)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response


def analyze_model(result_file, task_data, city_data):
    """Analyze a model's performance at different granularity levels."""
    with open(result_file) as f:
        results = json.load(f)

    stats = {
        'continent': {'correct': 0, 'total': 0},
        'country': {'correct': 0, 'total': 0},
        'region': {'correct': 0, 'total': 0},
        'city': {'correct': 0, 'total': 0},
        'all': {'correct': 0, 'total': 0},
    }

    for result in results.get('results', []):
        response = result.get('response', '')
        answer = extract_answer(response)

        # Use ground truth from result file (handle both formats)
        gt = result.get('ground_truth', {})
        # Format 1: {'name': ..., 'country': ...}
        # Format 2: {'location_name': ..., 'country_code': ...}
        country_code = gt.get('country') or gt.get('country_code', '')
        city_name = gt.get('name') or gt.get('location_name', '')

        # Get continent and region from lookup
        city_key = f"{city_name.lower()}_{country_code.lower()}"
        city_info = city_data.get(city_key, {})
        continent = city_info.get('continent', 'Unknown')
        region = city_info.get('region', '')

        # Check each granularity level
        continent_correct = check_continent_match(answer, continent)
        country_correct = check_country_match(answer, country_code)
        region_correct = check_region_match(answer, region) if region else False

        # Use ORIGINAL evaluation for city correctness (more accurate)
        city_correct = result.get('correct', False)

        stats['continent']['total'] += 1
        stats['country']['total'] += 1
        stats['city']['total'] += 1
        stats['all']['total'] += 1

        if region:
            stats['region']['total'] += 1
            if region_correct:
                stats['region']['correct'] += 1

        if continent_correct:
            stats['continent']['correct'] += 1
        if country_correct:
            stats['country']['correct'] += 1
        if city_correct:
            stats['city']['correct'] += 1
        if city_correct and country_correct and continent_correct:
            stats['all']['correct'] += 1

    return stats


def get_model_name(dirname):
    """Extract clean model name."""
    name = dirname.replace('gpsbench_', '').split('_20')[0]
    if len(name) > 30:
        name = name[:27] + '...'
    return name


def create_chart(all_results, output_file):
    """Create grouped bar chart."""
    models = list(all_results.keys())
    granularities = ['Continent', 'Country', 'Region', 'City', 'All']

    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(15, 7))

    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']

    for i, gran in enumerate(granularities):
        key = gran.lower()
        accs = []
        for m in models:
            stats = all_results[m][key]
            if stats['total'] > 0:
                accs.append(stats['correct'] / stats['total'] * 100)
            else:
                accs.append(0)
        bars = ax.bar(x + i*width, accs, width, label=gran, color=colors[i])

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.0f}%', ha='center', va='bottom', fontsize=9, rotation=90)

    ax.set_xlabel('Model', fontsize=15)
    ax.set_ylabel('Accuracy (%)', fontsize=15)
    ax.set_title('Place Association: Accuracy by Granularity Level\n(Continent → Country → Region → City)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=12, rotation=15, ha='right')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def print_results(all_results):
    """Print results table."""
    print("\n" + "=" * 100)
    print("PLACE ASSOCIATION - ACCURACY BY GRANULARITY LEVEL")
    print("=" * 100)
    print(f"\n{'Model':<30} {'Continent':>12} {'Country':>12} {'Region':>12} {'City':>12} {'All':>10}")
    print("-" * 100)

    for model in sorted(all_results.keys(),
                       key=lambda m: all_results[m]['city']['correct']/all_results[m]['city']['total'],
                       reverse=True):
        stats = all_results[model]
        cont_acc = stats['continent']['correct'] / stats['continent']['total'] * 100
        country_acc = stats['country']['correct'] / stats['country']['total'] * 100
        region_acc = stats['region']['correct'] / stats['region']['total'] * 100 if stats['region']['total'] > 0 else 0
        city_acc = stats['city']['correct'] / stats['city']['total'] * 100
        all_acc = stats['all']['correct'] / stats['all']['total'] * 100 if stats['all']['total'] > 0 else 0
        print(f"{model:<30} {cont_acc:>10.1f}% {country_acc:>10.1f}% {region_acc:>10.1f}% {city_acc:>10.1f}% {all_acc:>8.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--task-file', type=Path,
                       default=Path('data/track_applied/task1_place_association.json'))
    parser.add_argument('--cities-file', type=Path,
                       default=Path('data/raw/cities15000.txt'))
    parser.add_argument('--admin1-file', type=Path,
                       default=Path('data/raw/admin1CodesASCII.txt'))
    parser.add_argument('--min-samples', type=int, default=5000)
    args = parser.parse_args()

    print("Loading admin1 (region/state) names...")
    admin1_names = load_admin1_names(args.admin1_file)
    print(f"Loaded {len(admin1_names)} region names")

    print("\nLoading city data (continent + region)...")
    city_data = load_city_data(args.cities_file, admin1_names)
    print(f"Loaded data for {len(city_data)} cities")

    print("\nLoading task data...")
    with open(args.task_file) as f:
        task_data = json.load(f)
    print(f"Loaded {len(task_data)} examples")

    print("\nAnalyzing models...")
    all_results = {}

    for model_dir in sorted(args.results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue

        result_file = model_dir / 'task_results' / 'applied_task1_place_association.json'
        if not result_file.exists():
            continue

        summary_file = model_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            if summary.get('overall', {}).get('total_cases', 0) < args.min_samples:
                continue

        model_name = get_model_name(model_dir.name)
        print(f"  Processing: {model_name}")

        stats = analyze_model(result_file, task_data, city_data)
        all_results[model_name] = stats

    print_results(all_results)

    print("\nGenerating chart...")
    create_chart(all_results, 'chart_place_granularity.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
