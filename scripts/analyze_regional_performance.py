#!/usr/bin/env python3
"""Analyze model performance by geographic region."""

import json
import os
from collections import defaultdict
from pathlib import Path

# Map country codes to regions (UN geoscheme)
COUNTRY_TO_REGION = {
    # Northern Europe
    'DK': 'Northern Europe', 'EE': 'Northern Europe', 'FI': 'Northern Europe',
    'IS': 'Northern Europe', 'IE': 'Northern Europe', 'LV': 'Northern Europe',
    'LT': 'Northern Europe', 'NO': 'Northern Europe', 'SE': 'Northern Europe',
    'GB': 'Northern Europe', 'UK': 'Northern Europe',
    # Western Europe
    'AT': 'Western Europe', 'BE': 'Western Europe', 'FR': 'Western Europe',
    'DE': 'Western Europe', 'LI': 'Western Europe', 'LU': 'Western Europe',
    'MC': 'Western Europe', 'NL': 'Western Europe', 'CH': 'Western Europe',
    # Eastern Europe
    'BY': 'Eastern Europe', 'BG': 'Eastern Europe', 'CZ': 'Eastern Europe',
    'HU': 'Eastern Europe', 'MD': 'Eastern Europe', 'PL': 'Eastern Europe',
    'RO': 'Eastern Europe', 'RU': 'Eastern Europe', 'SK': 'Eastern Europe',
    'UA': 'Eastern Europe',
    # Southern Europe
    'AL': 'Southern Europe', 'AD': 'Southern Europe', 'BA': 'Southern Europe',
    'HR': 'Southern Europe', 'GR': 'Southern Europe', 'IT': 'Southern Europe',
    'MT': 'Southern Europe', 'ME': 'Southern Europe', 'MK': 'Southern Europe',
    'PT': 'Southern Europe', 'SM': 'Southern Europe', 'RS': 'Southern Europe',
    'SI': 'Southern Europe', 'ES': 'Southern Europe', 'VA': 'Southern Europe',
    # Northern America
    'CA': 'Northern America', 'US': 'Northern America', 'MX': 'Northern America',
    # Central America
    'BZ': 'Central America', 'CR': 'Central America', 'SV': 'Central America',
    'GT': 'Central America', 'HN': 'Central America', 'NI': 'Central America',
    'PA': 'Central America',
    # South America
    'AR': 'South America', 'BO': 'South America', 'BR': 'South America',
    'CL': 'South America', 'CO': 'South America', 'EC': 'South America',
    'GY': 'South America', 'PY': 'South America', 'PE': 'South America',
    'SR': 'South America', 'UY': 'South America', 'VE': 'South America',
    # Caribbean
    'CU': 'Caribbean', 'DO': 'Caribbean', 'HT': 'Caribbean', 'JM': 'Caribbean',
    'PR': 'Caribbean', 'TT': 'Caribbean',
    # Eastern Asia
    'CN': 'Eastern Asia', 'HK': 'Eastern Asia', 'JP': 'Eastern Asia',
    'KP': 'Eastern Asia', 'KR': 'Eastern Asia', 'MO': 'Eastern Asia',
    'MN': 'Eastern Asia', 'TW': 'Eastern Asia',
    # South-Eastern Asia
    'BN': 'South-Eastern Asia', 'KH': 'South-Eastern Asia', 'ID': 'South-Eastern Asia',
    'LA': 'South-Eastern Asia', 'MY': 'South-Eastern Asia', 'MM': 'South-Eastern Asia',
    'PH': 'South-Eastern Asia', 'SG': 'South-Eastern Asia', 'TH': 'South-Eastern Asia',
    'TL': 'South-Eastern Asia', 'VN': 'South-Eastern Asia',
    # Southern Asia
    'AF': 'Southern Asia', 'BD': 'Southern Asia', 'BT': 'Southern Asia',
    'IN': 'Southern Asia', 'IR': 'Southern Asia', 'MV': 'Southern Asia',
    'NP': 'Southern Asia', 'PK': 'Southern Asia', 'LK': 'Southern Asia',
    # Western Asia (Middle East)
    'AM': 'Western Asia', 'AZ': 'Western Asia', 'BH': 'Western Asia',
    'CY': 'Western Asia', 'GE': 'Western Asia', 'IQ': 'Western Asia',
    'IL': 'Western Asia', 'JO': 'Western Asia', 'KW': 'Western Asia',
    'LB': 'Western Asia', 'OM': 'Western Asia', 'PS': 'Western Asia',
    'QA': 'Western Asia', 'SA': 'Western Asia', 'SY': 'Western Asia',
    'TR': 'Western Asia', 'AE': 'Western Asia', 'YE': 'Western Asia',
    # Central Asia
    'KZ': 'Central Asia', 'KG': 'Central Asia', 'TJ': 'Central Asia',
    'TM': 'Central Asia', 'UZ': 'Central Asia',
    # Northern Africa
    'DZ': 'Northern Africa', 'EG': 'Northern Africa', 'LY': 'Northern Africa',
    'MA': 'Northern Africa', 'SD': 'Northern Africa', 'TN': 'Northern Africa',
    # Western Africa
    'BJ': 'Western Africa', 'BF': 'Western Africa', 'CV': 'Western Africa',
    'CI': 'Western Africa', 'GM': 'Western Africa', 'GH': 'Western Africa',
    'GN': 'Western Africa', 'GW': 'Western Africa', 'LR': 'Western Africa',
    'ML': 'Western Africa', 'MR': 'Western Africa', 'NE': 'Western Africa',
    'NG': 'Western Africa', 'SN': 'Western Africa', 'SL': 'Western Africa',
    'TG': 'Western Africa',
    # Eastern Africa
    'BI': 'Eastern Africa', 'KM': 'Eastern Africa', 'DJ': 'Eastern Africa',
    'ER': 'Eastern Africa', 'ET': 'Eastern Africa', 'KE': 'Eastern Africa',
    'MG': 'Eastern Africa', 'MW': 'Eastern Africa', 'MU': 'Eastern Africa',
    'MZ': 'Eastern Africa', 'RE': 'Eastern Africa', 'RW': 'Eastern Africa',
    'SC': 'Eastern Africa', 'SO': 'Eastern Africa', 'SS': 'Eastern Africa',
    'TZ': 'Eastern Africa', 'UG': 'Eastern Africa', 'ZM': 'Eastern Africa',
    'ZW': 'Eastern Africa',
    # Middle Africa
    'AO': 'Middle Africa', 'CM': 'Middle Africa', 'CF': 'Middle Africa',
    'TD': 'Middle Africa', 'CG': 'Middle Africa', 'CD': 'Middle Africa',
    'GQ': 'Middle Africa', 'GA': 'Middle Africa', 'ST': 'Middle Africa',
    # Southern Africa
    'BW': 'Southern Africa', 'SZ': 'Southern Africa', 'LS': 'Southern Africa',
    'NA': 'Southern Africa', 'ZA': 'Southern Africa',
    # Australia and New Zealand
    'AU': 'Australia/NZ', 'NZ': 'Australia/NZ',
    # Melanesia, Micronesia, Polynesia
    'FJ': 'Oceania', 'NC': 'Oceania', 'PG': 'Oceania', 'SB': 'Oceania',
    'VU': 'Oceania', 'GU': 'Oceania', 'KI': 'Oceania', 'MH': 'Oceania',
    'FM': 'Oceania', 'NR': 'Oceania', 'PW': 'Oceania', 'AS': 'Oceania',
    'CK': 'Oceania', 'PF': 'Oceania', 'NU': 'Oceania', 'PN': 'Oceania',
    'WS': 'Oceania', 'TK': 'Oceania', 'TO': 'Oceania', 'TV': 'Oceania',
    'WF': 'Oceania',
}

# Broader continent mapping
REGION_TO_CONTINENT = {
    'Northern Europe': 'Europe', 'Western Europe': 'Europe',
    'Eastern Europe': 'Europe', 'Southern Europe': 'Europe',
    'Northern America': 'Americas', 'Central America': 'Americas',
    'South America': 'Americas', 'Caribbean': 'Americas',
    'Eastern Asia': 'Asia', 'South-Eastern Asia': 'Asia',
    'Southern Asia': 'Asia', 'Western Asia': 'Asia', 'Central Asia': 'Asia',
    'Northern Africa': 'Africa', 'Western Africa': 'Africa',
    'Eastern Africa': 'Africa', 'Middle Africa': 'Africa', 'Southern Africa': 'Africa',
    'Australia/NZ': 'Oceania', 'Oceania': 'Oceania',
}


def get_region_from_country_code(country_code):
    """Get region from country code."""
    return COUNTRY_TO_REGION.get(country_code, 'Unknown')


def analyze_place_association_by_region(results_dir):
    """Analyze Place Association performance by region."""
    results = {}

    # Find all Place Association result files
    for model_dir in Path(results_dir).iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith('gpsbench_'):
            continue

        # Extract model name
        model_name = model_dir.name.replace('gpsbench_', '').split('_2025')[0].split('_2026')[0]

        # Fix name swaps
        if 'gpt-5.1' in model_name.lower():
            model_name = 'GPT-5-mini'
        elif 'gpt-5-mini' in model_name.lower():
            model_name = 'GPT-5.1'
        elif 'gpt-5-nano' in model_name.lower():
            model_name = 'GPT-5-nano'
        elif 'gpt-4.1-mini' in model_name.lower():
            model_name = 'GPT-4.1-mini'
        elif 'gpt-4.1' in model_name.lower():
            model_name = 'GPT-4.1'
        elif 'gemini-2.5-pro' in model_name.lower():
            model_name = 'Gemini-2.5-Pro'
        elif 'gemini-2.5-flash' in model_name.lower():
            model_name = 'Gemini-2.5-Flash'
        elif 'claude-haiku' in model_name.lower():
            model_name = 'Claude-Haiku-4.5'
        elif 'mistral-large' in model_name.lower():
            model_name = 'Mistral-Large'
        elif 'mistral-small' in model_name.lower():
            model_name = 'Mistral-Small-24B'
        elif 'qwen3-235b' in model_name.lower():
            model_name = 'Qwen3-235B'
        elif 'qwen3-30b' in model_name.lower():
            model_name = 'Qwen3-30B'
        elif 'qwen3-14b' in model_name.lower():
            model_name = 'Qwen3-14B'
        elif 'qwen3-8b' in model_name.lower():
            model_name = 'Qwen3-8B'

        # Load Place Association results
        task_file = model_dir / 'task_results' / 'applied_task1_place_association.json'
        if not task_file.exists():
            continue

        with open(task_file, 'r') as f:
            data = json.load(f)

        # Analyze by region
        region_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        continent_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for result in data.get('results', []):
            gt = result.get('ground_truth', {})
            country_code = gt.get('country_code', '')
            region = gt.get('region', get_region_from_country_code(country_code))
            continent = REGION_TO_CONTINENT.get(region, 'Unknown')

            is_correct = result.get('correct', False)

            region_stats[region]['total'] += 1
            continent_stats[continent]['total'] += 1

            if is_correct:
                region_stats[region]['correct'] += 1
                continent_stats[continent]['correct'] += 1

        results[model_name] = {
            'by_region': dict(region_stats),
            'by_continent': dict(continent_stats)
        }

    return results


def print_regional_analysis(results):
    """Print regional performance analysis."""
    # Aggregate across models
    continent_totals = defaultdict(lambda: {'correct': 0, 'total': 0})
    region_totals = defaultdict(lambda: {'correct': 0, 'total': 0})

    for model, data in results.items():
        for continent, stats in data['by_continent'].items():
            continent_totals[continent]['correct'] += stats['correct']
            continent_totals[continent]['total'] += stats['total']
        for region, stats in data['by_region'].items():
            region_totals[region]['correct'] += stats['correct']
            region_totals[region]['total'] += stats['total']

    print("\n" + "="*60)
    print("PERFORMANCE BY CONTINENT (aggregated across all models)")
    print("="*60)

    continent_accs = []
    for continent in sorted(continent_totals.keys()):
        stats = continent_totals[continent]
        acc = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        continent_accs.append((continent, acc, stats['total']))

    # Sort by accuracy
    continent_accs.sort(key=lambda x: x[1], reverse=True)

    for continent, acc, total in continent_accs:
        print(f"{continent:20s}: {acc:5.1f}% (n={total})")

    print("\n" + "="*60)
    print("PERFORMANCE BY REGION (aggregated across all models)")
    print("="*60)

    region_accs = []
    for region in sorted(region_totals.keys()):
        stats = region_totals[region]
        acc = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        region_accs.append((region, acc, stats['total']))

    # Sort by accuracy
    region_accs.sort(key=lambda x: x[1], reverse=True)

    for region, acc, total in region_accs:
        if total >= 50:  # Only show regions with sufficient samples
            print(f"{region:25s}: {acc:5.1f}% (n={total})")

    # Per-model continent breakdown
    print("\n" + "="*60)
    print("PER-MODEL CONTINENT ACCURACY")
    print("="*60)

    continents = ['Europe', 'Americas', 'Asia', 'Africa', 'Oceania']

    # Print header
    print(f"{'Model':20s}", end='')
    for c in continents:
        print(f"{c:>10s}", end='')
    print()
    print("-" * 70)

    for model in sorted(results.keys()):
        print(f"{model:20s}", end='')
        for continent in continents:
            stats = results[model]['by_continent'].get(continent, {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                acc = 100 * stats['correct'] / stats['total']
                print(f"{acc:9.1f}%", end='')
            else:
                print(f"{'N/A':>10s}", end='')
        print()

    return continent_accs, region_accs


def generate_latex_tables(results):
    """Generate LaTeX tables for the paper."""
    # Aggregate across models
    continent_totals = defaultdict(lambda: {'correct': 0, 'total': 0})

    for model, data in results.items():
        for continent, stats in data['by_continent'].items():
            continent_totals[continent]['correct'] += stats['correct']
            continent_totals[continent]['total'] += stats['total']

    # Per-model continent table
    continents = ['Europe', 'Americas', 'Asia', 'Africa', 'Oceania']

    print("\n% LaTeX Table: Per-model continent accuracy")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Place Association accuracy (\\%) by continent.}")
    print("\\label{tab:regional_performance}")
    print("\\small")
    print("\\begin{tabular}{l" + "c" * len(continents) + "}")
    print("\\toprule")
    print("\\textbf{Model} & " + " & ".join([f"\\textbf{{{c}}}" for c in continents]) + " \\\\")
    print("\\midrule")

    for model in sorted(results.keys()):
        row = [model]
        for continent in continents:
            stats = results[model]['by_continent'].get(continent, {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                acc = 100 * stats['correct'] / stats['total']
                row.append(f"{acc:.1f}")
            else:
                row.append("--")
        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == '__main__':
    results_dir = Path(__file__).parent.parent / 'results'
    results = analyze_place_association_by_region(results_dir)

    if results:
        continent_accs, region_accs = print_regional_analysis(results)
        generate_latex_tables(results)
    else:
        print("No results found")
