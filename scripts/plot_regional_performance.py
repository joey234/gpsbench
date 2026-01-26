#!/usr/bin/env python3
"""Plot regional performance breakdown by subregion."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Continent colors
CONTINENT_COLORS = {
    'Europe': '#2E86AB',
    'Americas': '#A23B72',
    'Asia': '#F18F01',
    'Africa': '#C73E1D',
    'Oceania': '#3B1F2B'
}

# Map country codes to subregions
COUNTRY_TO_SUBREGION = {
    # North America
    'US': 'North America', 'CA': 'North America', 'MX': 'North America',
    # Central America
    'GT': 'Central America', 'BZ': 'Central America', 'HN': 'Central America',
    'SV': 'Central America', 'NI': 'Central America', 'CR': 'Central America', 'PA': 'Central America',
    # Caribbean
    'CU': 'Caribbean', 'JM': 'Caribbean', 'HT': 'Caribbean', 'DO': 'Caribbean',
    'PR': 'Caribbean', 'TT': 'Caribbean', 'BB': 'Caribbean', 'BS': 'Caribbean',
    # South America
    'BR': 'South America', 'AR': 'South America', 'CL': 'South America', 'CO': 'South America',
    'PE': 'South America', 'VE': 'South America', 'EC': 'South America', 'BO': 'South America',
    'PY': 'South America', 'UY': 'South America', 'GY': 'South America', 'SR': 'South America',
    # Western Europe
    'GB': 'Western Europe', 'UK': 'Western Europe', 'FR': 'Western Europe', 'DE': 'Western Europe',
    'NL': 'Western Europe', 'BE': 'Western Europe', 'AT': 'Western Europe', 'CH': 'Western Europe',
    'IE': 'Western Europe', 'LU': 'Western Europe',
    # Northern Europe
    'SE': 'Northern Europe', 'NO': 'Northern Europe', 'DK': 'Northern Europe', 'FI': 'Northern Europe',
    'IS': 'Northern Europe', 'EE': 'Northern Europe', 'LV': 'Northern Europe', 'LT': 'Northern Europe',
    # Southern Europe
    'IT': 'Southern Europe', 'ES': 'Southern Europe', 'PT': 'Southern Europe', 'GR': 'Southern Europe',
    'HR': 'Southern Europe', 'SI': 'Southern Europe', 'RS': 'Southern Europe', 'BA': 'Southern Europe',
    'ME': 'Southern Europe', 'MK': 'Southern Europe', 'AL': 'Southern Europe', 'MT': 'Southern Europe',
    # Eastern Europe
    'PL': 'Eastern Europe', 'CZ': 'Eastern Europe', 'SK': 'Eastern Europe', 'HU': 'Eastern Europe',
    'RO': 'Eastern Europe', 'BG': 'Eastern Europe', 'UA': 'Eastern Europe', 'BY': 'Eastern Europe',
    'MD': 'Eastern Europe', 'RU': 'Eastern Europe',
    # East Asia
    'CN': 'East Asia', 'JP': 'East Asia', 'KR': 'East Asia', 'TW': 'East Asia',
    'HK': 'East Asia', 'MO': 'East Asia', 'MN': 'East Asia', 'KP': 'East Asia',
    # Southeast Asia
    'TH': 'Southeast Asia', 'VN': 'Southeast Asia', 'MY': 'Southeast Asia', 'SG': 'Southeast Asia',
    'ID': 'Southeast Asia', 'PH': 'Southeast Asia', 'MM': 'Southeast Asia', 'KH': 'Southeast Asia',
    'LA': 'Southeast Asia', 'BN': 'Southeast Asia', 'TL': 'Southeast Asia',
    # South Asia
    'IN': 'South Asia', 'PK': 'South Asia', 'BD': 'South Asia', 'LK': 'South Asia',
    'NP': 'South Asia', 'BT': 'South Asia', 'MV': 'South Asia', 'AF': 'South Asia',
    # Middle East
    'TR': 'Middle East', 'IR': 'Middle East', 'IQ': 'Middle East', 'SA': 'Middle East',
    'AE': 'Middle East', 'IL': 'Middle East', 'JO': 'Middle East', 'LB': 'Middle East',
    'SY': 'Middle East', 'YE': 'Middle East', 'OM': 'Middle East', 'KW': 'Middle East',
    'QA': 'Middle East', 'BH': 'Middle East', 'PS': 'Middle East', 'CY': 'Middle East',
    # Central Asia
    'KZ': 'Central Asia', 'UZ': 'Central Asia', 'TM': 'Central Asia', 'TJ': 'Central Asia', 'KG': 'Central Asia',
    # Caucasus
    'GE': 'Caucasus', 'AM': 'Caucasus', 'AZ': 'Caucasus',
    # North Africa
    'EG': 'North Africa', 'LY': 'North Africa', 'TN': 'North Africa', 'DZ': 'North Africa',
    'MA': 'North Africa', 'SD': 'North Africa',
    # West Africa
    'NG': 'West Africa', 'GH': 'West Africa', 'CI': 'West Africa', 'SN': 'West Africa',
    'ML': 'West Africa', 'BF': 'West Africa', 'NE': 'West Africa', 'GN': 'West Africa',
    'BJ': 'West Africa', 'TG': 'West Africa', 'SL': 'West Africa', 'LR': 'West Africa',
    'MR': 'West Africa', 'GM': 'West Africa', 'GW': 'West Africa', 'CV': 'West Africa',
    # East Africa
    'KE': 'East Africa', 'TZ': 'East Africa', 'UG': 'East Africa', 'ET': 'East Africa',
    'RW': 'East Africa', 'BI': 'East Africa', 'SO': 'East Africa', 'DJ': 'East Africa',
    'ER': 'East Africa', 'SS': 'East Africa', 'MG': 'East Africa', 'MU': 'East Africa',
    'SC': 'East Africa', 'KM': 'East Africa', 'MW': 'East Africa', 'MZ': 'East Africa',
    'ZM': 'East Africa', 'ZW': 'East Africa',
    # Central Africa
    'CD': 'Central Africa', 'CG': 'Central Africa', 'CF': 'Central Africa', 'CM': 'Central Africa',
    'GA': 'Central Africa', 'GQ': 'Central Africa', 'TD': 'Central Africa', 'AO': 'Central Africa',
    # Southern Africa
    'ZA': 'Southern Africa', 'NA': 'Southern Africa', 'BW': 'Southern Africa', 'SZ': 'Southern Africa', 'LS': 'Southern Africa',
    # Oceania
    'AU': 'Oceania', 'NZ': 'Oceania', 'PG': 'Oceania', 'FJ': 'Oceania',
    'SB': 'Oceania', 'VU': 'Oceania', 'NC': 'Oceania', 'PF': 'Oceania',
    'WS': 'Oceania', 'TO': 'Oceania', 'KI': 'Oceania', 'FM': 'Oceania',
}

# Map subregions to continents
SUBREGION_TO_CONTINENT = {
    'North America': 'Americas', 'Central America': 'Americas',
    'South America': 'Americas', 'Caribbean': 'Americas',
    'Western Europe': 'Europe', 'Northern Europe': 'Europe',
    'Southern Europe': 'Europe', 'Eastern Europe': 'Europe',
    'East Asia': 'Asia', 'Southeast Asia': 'Asia',
    'South Asia': 'Asia', 'Middle East': 'Asia',
    'Central Asia': 'Asia', 'Caucasus': 'Asia',
    'North Africa': 'Africa', 'West Africa': 'Africa',
    'East Africa': 'Africa', 'Central Africa': 'Africa', 'Southern Africa': 'Africa',
    'Oceania': 'Oceania',
}

# Model display names with proper ordering
MODEL_ORDER = [
    'GPT-5.1', 'GPT-5-mini', 'GPT-5-nano', 'GPT-4.1', 'GPT-4.1-mini',
    'Gemini-2.5-Pro', 'Gemini-2.5-Flash',
    'Claude-Haiku-4.5',
    'Mistral-Large', 'Mistral-Small-24B',
    'Qwen3-235B', 'Qwen3-30B', 'Qwen3-14B', 'Qwen3-8B'
]


def normalize_model_name(model_name):
    """Normalize model name from directory."""
    model_lower = model_name.lower()

    if 'gpt-5.1' in model_lower:
        return 'GPT-5-mini'  # Swapped
    elif 'gpt-5-mini' in model_lower:
        return 'GPT-5.1'     # Swapped
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


def analyze_place_association_by_region(results_dir):
    """Analyze Place Association performance by region for all 14 models."""
    results = {}

    for model_dir in Path(results_dir).iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith('gpsbench_'):
            continue

        raw_name = model_dir.name.replace('gpsbench_', '').split('_2025')[0].split('_2026')[0]
        model_name = normalize_model_name(raw_name)

        task_file = model_dir / 'task_results' / 'applied_task1_place_association.json'
        if not task_file.exists():
            print(f"Warning: No Place Association results for {model_name}")
            continue

        with open(task_file, 'r') as f:
            data = json.load(f)

        # Analyze by region using country_code
        region_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for result in data.get('results', []):
            gt = result.get('ground_truth', {})
            country_code = gt.get('country_code', '')

            # Get region from country code, or from region field if available
            region = gt.get('region', COUNTRY_TO_SUBREGION.get(country_code, 'Unknown'))
            is_correct = result.get('correct', False)

            region_stats[region]['total'] += 1
            if is_correct:
                region_stats[region]['correct'] += 1

        results[model_name] = dict(region_stats)

    print(f"Analyzed {len(results)} models: {sorted(results.keys())}")
    return results


def plot_subregion_heatmap(results, output_path):
    """Plot heatmap of model performance by subregion."""
    subregions = [
        'North America', 'South America', 'Central America',
        'Western Europe', 'Northern Europe', 'Eastern Europe', 'Southern Europe',
        'East Asia', 'South Asia', 'Southeast Asia', 'Middle East',
        'West Africa', 'East Africa', 'North Africa',
    ]

    models = [m for m in MODEL_ORDER if m in results]

    # Build accuracy matrix
    acc_matrix = np.zeros((len(models), len(subregions)))
    for i, model in enumerate(models):
        for j, region in enumerate(subregions):
            stats = results[model].get(region, {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                acc_matrix[i, j] = 100 * stats['correct'] / stats['total']
            else:
                acc_matrix[i, j] = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=40)

    ax.set_xticks(np.arange(len(subregions)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(subregions, fontsize=9)
    ax.set_yticklabels(models, fontsize=9)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(len(models)):
        for j in range(len(subregions)):
            if not np.isnan(acc_matrix[i, j]):
                text_color = 'white' if acc_matrix[i, j] < 15 else 'black'
                ax.text(j, i, f'{acc_matrix[i, j]:.0f}',
                       ha='center', va='center', fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom')

    # Add continent separators
    for x in [2.5, 6.5, 10.5]:
        ax.axvline(x, color='white', linewidth=2)

    ax.text(1, -1.5, 'Americas', ha='center', fontsize=10, fontweight='bold', color=CONTINENT_COLORS['Americas'])
    ax.text(4.5, -1.5, 'Europe', ha='center', fontsize=10, fontweight='bold', color=CONTINENT_COLORS['Europe'])
    ax.text(8.5, -1.5, 'Asia', ha='center', fontsize=10, fontweight='bold', color=CONTINENT_COLORS['Asia'])
    ax.text(12, -1.5, 'Africa', ha='center', fontsize=10, fontweight='bold', color=CONTINENT_COLORS['Africa'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_subregion_bar(results, output_path):
    """Plot bar chart of subregion performance aggregated across models."""
    region_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for model, regions in results.items():
        for region, stats in regions.items():
            if region != 'Unknown':
                region_stats[region]['correct'] += stats['correct']
                region_stats[region]['total'] += stats['total']

    region_accs = []
    for region, stats in region_stats.items():
        if stats['total'] >= 100:
            acc = 100 * stats['correct'] / stats['total']
            continent = SUBREGION_TO_CONTINENT.get(region, 'Unknown')
            region_accs.append((region, acc, stats['total'], continent))

    region_accs.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    regions = [r[0] for r in region_accs]
    accs = [r[1] for r in region_accs]
    totals = [r[2] for r in region_accs]
    continents = [r[3] for r in region_accs]
    colors = [CONTINENT_COLORS.get(c, '#666666') for c in continents]

    y_pos = np.arange(len(regions))
    bars = ax.barh(y_pos, accs, color=colors, edgecolor='black', linewidth=0.5)

    for bar, acc, total in zip(bars, accs, totals):
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                f'{acc:.1f}% (n={total})',
                ha='left', va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(regions)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Place Association by Subregion (14 LLMs aggregated, n≥100)')
    ax.set_xlim(0, max(accs) * 1.4)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CONTINENT_COLORS[c], label=c)
                      for c in ['Americas', 'Europe', 'Asia', 'Africa']]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_latex_table(results):
    """Print LaTeX table for subregion performance."""
    subregions = [
        'North America', 'South America',
        'Western Europe', 'Eastern Europe',
        'East Asia', 'South Asia',
        'West Africa', 'East Africa',
    ]

    models = [m for m in MODEL_ORDER if m in results]

    print("\n% LaTeX Table: Subregion performance")
    print("\\begin{table*}[htbp]")
    print("\\centering")
    print("\\caption{Place Association accuracy (\\%) by subregion across 14 LLMs.}")
    print("\\label{tab:subregion_performance}")
    print("\\small")
    print("\\begin{tabular}{l" + "c" * len(subregions) + "}")
    print("\\toprule")

    abbrev = {
        'North America': 'N.Am', 'South America': 'S.Am',
        'Western Europe': 'W.Eur', 'Eastern Europe': 'E.Eur',
        'East Asia': 'E.Asia', 'South Asia': 'S.Asia',
        'West Africa': 'W.Afr', 'East Africa': 'E.Afr',
    }
    print("\\textbf{Model} & " + " & ".join([f"\\textbf{{{abbrev[r]}}}" for r in subregions]) + " \\\\")
    print("\\midrule")

    for model in models:
        row = [model]
        for region in subregions:
            stats = results[model].get(region, {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                acc = 100 * stats['correct'] / stats['total']
                row.append(f"{acc:.1f}")
            else:
                row.append("--")
        print(" & ".join(row) + " \\\\")

    print("\\midrule")
    row = ["\\textit{Mean}"]
    for region in subregions:
        total_correct = sum(results[m].get(region, {'correct': 0})['correct'] for m in models)
        total_samples = sum(results[m].get(region, {'total': 0})['total'] for m in models)
        if total_samples > 0:
            acc = 100 * total_correct / total_samples
            row.append(f"\\textit{{{acc:.1f}}}")
        else:
            row.append("--")
    print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")


def print_summary_stats(results):
    """Print summary statistics."""
    region_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for model, regions in results.items():
        for region, stats in regions.items():
            region_stats[region]['correct'] += stats['correct']
            region_stats[region]['total'] += stats['total']

    print("\n" + "="*60)
    print(f"SUBREGION PERFORMANCE (aggregated across {len(results)} models)")
    print("="*60)

    region_accs = []
    for region, stats in region_stats.items():
        if stats['total'] >= 50 and region != 'Unknown':
            acc = 100 * stats['correct'] / stats['total']
            region_accs.append((region, acc, stats['total']))

    region_accs.sort(key=lambda x: x[1], reverse=True)

    for region, acc, total in region_accs:
        print(f"{region:25s}: {acc:5.1f}% (n={total})")


if __name__ == '__main__':
    results_dir = Path(__file__).parent.parent / 'results'
    figures_dir = Path(__file__).parent.parent / 'paper' / 'figures'

    results = analyze_place_association_by_region(results_dir)

    if results:
        print_summary_stats(results)
        plot_subregion_heatmap(results, figures_dir / 'regional_performance_heatmap.pdf')
        plot_subregion_bar(results, figures_dir / 'regional_performance_bar.pdf')
        print_latex_table(results)
    else:
        print("No results found")
