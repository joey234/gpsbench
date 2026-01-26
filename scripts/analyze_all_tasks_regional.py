#!/usr/bin/env python3
"""Analyze regional performance across all tasks with location data."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Country code to subregion mapping
COUNTRY_TO_SUBREGION = {
    # North America
    'US': 'North America', 'CA': 'North America', 'MX': 'North America',
    # Central America & Caribbean
    'GT': 'Central America', 'BZ': 'Central America', 'HN': 'Central America',
    'SV': 'Central America', 'NI': 'Central America', 'CR': 'Central America',
    'PA': 'Central America', 'CU': 'Caribbean', 'JM': 'Caribbean',
    'HT': 'Caribbean', 'DO': 'Caribbean', 'PR': 'Caribbean', 'TT': 'Caribbean',
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
    # Northern Europe
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
    # Central Asia
    'KZ': 'Central Asia', 'UZ': 'Central Asia', 'TM': 'Central Asia',
    'TJ': 'Central Asia', 'KG': 'Central Asia',
    # Africa
    'EG': 'North Africa', 'LY': 'North Africa', 'TN': 'North Africa',
    'DZ': 'North Africa', 'MA': 'North Africa', 'SD': 'North Africa',
    'NG': 'West Africa', 'GH': 'West Africa', 'CI': 'West Africa',
    'SN': 'West Africa', 'ML': 'West Africa', 'BF': 'West Africa',
    'NE': 'West Africa', 'GN': 'West Africa', 'BJ': 'West Africa',
    'TG': 'West Africa', 'SL': 'West Africa', 'LR': 'West Africa',
    'KE': 'East Africa', 'TZ': 'East Africa', 'UG': 'East Africa',
    'ET': 'East Africa', 'RW': 'East Africa', 'SO': 'East Africa',
    'MG': 'East Africa', 'MW': 'East Africa', 'MZ': 'East Africa',
    'ZM': 'East Africa', 'ZW': 'East Africa',
    'CD': 'Central Africa', 'CG': 'Central Africa', 'CF': 'Central Africa',
    'CM': 'Central Africa', 'GA': 'Central Africa', 'AO': 'Central Africa',
    'ZA': 'Southern Africa', 'NA': 'Southern Africa', 'BW': 'Southern Africa',
    # Oceania
    'AU': 'Oceania', 'NZ': 'Oceania', 'PG': 'Oceania', 'FJ': 'Oceania',
}

SUBREGION_TO_CONTINENT = {
    'North America': 'Americas', 'Central America': 'Americas',
    'South America': 'Americas', 'Caribbean': 'Americas',
    'Western Europe': 'Europe', 'Northern Europe': 'Europe',
    'Southern Europe': 'Europe', 'Eastern Europe': 'Europe',
    'East Asia': 'Asia', 'Southeast Asia': 'Asia',
    'South Asia': 'Asia', 'Middle East': 'Asia', 'Central Asia': 'Asia',
    'North Africa': 'Africa', 'West Africa': 'Africa',
    'East Africa': 'Africa', 'Central Africa': 'Africa', 'Southern Africa': 'Africa',
    'Oceania': 'Oceania',
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


def get_region_from_result(result, task_name):
    """Extract region from result based on task type."""
    gt = result.get('ground_truth', {})

    if not isinstance(gt, dict):
        return None

    # Direct region field
    if 'region' in gt:
        return gt['region']

    # Country code
    if 'country_code' in gt:
        return COUNTRY_TO_SUBREGION.get(gt['country_code'])

    # Country name (for name_disambiguation)
    if 'country' in gt:
        country = gt['country']
        # Try to map country name to code
        country_name_to_code = {
            'United States': 'US', 'USA': 'US', 'United Kingdom': 'GB', 'UK': 'GB',
            'Germany': 'DE', 'France': 'FR', 'Italy': 'IT', 'Spain': 'ES',
            'China': 'CN', 'Japan': 'JP', 'India': 'IN', 'Brazil': 'BR',
            'Australia': 'AU', 'Canada': 'CA', 'Mexico': 'MX', 'Russia': 'RU',
        }
        code = country_name_to_code.get(country, country[:2].upper() if len(country) >= 2 else None)
        if code:
            return COUNTRY_TO_SUBREGION.get(code)

    return None


def analyze_task_by_region(results_dir, task_file_pattern, task_display_name):
    """Analyze a specific task by region."""
    results = {}

    for model_dir in Path(results_dir).iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith('gpsbench_'):
            continue

        raw_name = model_dir.name.replace('gpsbench_', '').split('_2025')[0].split('_2026')[0]
        model_name = normalize_model_name(raw_name)

        # Find matching task file
        task_files = list((model_dir / 'task_results').glob(task_file_pattern))
        if not task_files:
            continue

        with open(task_files[0], 'r') as f:
            data = json.load(f)

        region_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for result in data.get('results', []):
            region = get_region_from_result(result, task_display_name)
            if not region or region == 'Unknown':
                continue

            is_correct = result.get('correct', result.get('is_correct', False))
            region_stats[region]['total'] += 1
            if is_correct:
                region_stats[region]['correct'] += 1

        results[model_name] = dict(region_stats)

    return results


def aggregate_by_continent(results):
    """Aggregate region stats to continent level."""
    continent_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for model, regions in results.items():
        for region, stats in regions.items():
            continent = SUBREGION_TO_CONTINENT.get(region, 'Unknown')
            if continent != 'Unknown':
                continent_stats[continent]['correct'] += stats['correct']
                continent_stats[continent]['total'] += stats['total']

    return dict(continent_stats)


def main():
    results_dir = Path('results')
    figures_dir = Path('paper/figures')

    # Tasks to analyze with their file patterns
    tasks = [
        ('*task1_place_association.json', 'Place Association'),
        ('*task3_name_disambiguation.json', 'Name Disambiguation'),
        ('*task13_terrain_classification.json', 'Terrain Classification'),
    ]

    all_task_results = {}

    for pattern, task_name in tasks:
        print(f"\nAnalyzing {task_name}...")
        results = analyze_task_by_region(results_dir, pattern, task_name)
        all_task_results[task_name] = results

        # Print summary
        continent_stats = aggregate_by_continent(results)
        print(f"  Models analyzed: {len(results)}")
        for continent in ['Americas', 'Europe', 'Asia', 'Africa', 'Oceania']:
            stats = continent_stats.get(continent, {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                acc = 100 * stats['correct'] / stats['total']
                print(f"  {continent}: {acc:.1f}% (n={stats['total']})")

    # Create combined visualization
    plot_combined_regional(all_task_results, figures_dir / 'regional_all_tasks.pdf')
    print_latex_summary(all_task_results)


def plot_combined_regional(all_task_results, output_path):
    """Plot combined regional performance across tasks."""
    continents = ['Americas', 'Europe', 'Asia', 'Africa', 'Oceania']
    tasks = list(all_task_results.keys())

    # Aggregate by continent for each task
    task_continent_accs = {}
    for task_name, results in all_task_results.items():
        continent_stats = aggregate_by_continent(results)
        accs = []
        for continent in continents:
            stats = continent_stats.get(continent, {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                accs.append(100 * stats['correct'] / stats['total'])
            else:
                accs.append(0)
        task_continent_accs[task_name] = accs

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(continents))
    width = 0.25
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for i, (task_name, accs) in enumerate(task_continent_accs.items()):
        offset = (i - len(tasks)/2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=task_name, color=colors[i % len(colors)])

        # Add value labels
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                       f'{acc:.0f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Continent')
    ax.set_title('Regional Performance by Task (14 LLMs aggregated)')
    ax.set_xticks(x)
    ax.set_xticklabels(continents)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")


def print_latex_summary(all_task_results):
    """Print LaTeX summary table."""
    continents = ['Americas', 'Europe', 'Asia', 'Africa', 'Oceania']

    print("\n% LaTeX Table: Regional performance by task")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Accuracy (\\%) by continent across tasks (14 LLMs aggregated).}")
    print("\\label{tab:regional_all_tasks}")
    print("\\small")
    print("\\begin{tabular}{l" + "c" * len(continents) + "}")
    print("\\toprule")
    print("\\textbf{Task} & " + " & ".join([f"\\textbf{{{c}}}" for c in continents]) + " \\\\")
    print("\\midrule")

    for task_name, results in all_task_results.items():
        continent_stats = aggregate_by_continent(results)
        row = [task_name]
        for continent in continents:
            stats = continent_stats.get(continent, {'correct': 0, 'total': 0})
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
    main()
