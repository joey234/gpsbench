#!/usr/bin/env python3
"""
Analyze Place Association task performance by granularity level (city population).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse


def load_task_data(task_file: Path):
    """Load the task data file with granularity information."""
    with open(task_file, 'r') as f:
        return json.load(f)


def get_population_bucket(population: int) -> str:
    """Categorize population into buckets."""
    if population < 50000:
        return '<50k'
    elif population < 100000:
        return '50k-100k'
    elif population < 500000:
        return '100k-500k'
    elif population < 1000000:
        return '500k-1M'
    else:
        return '>1M'


def analyze_model_by_granularity(result_file: Path, task_data: list) -> dict:
    """Analyze a model's performance by population granularity."""
    with open(result_file, 'r') as f:
        results = json.load(f)

    # Build lookup for task data by index
    task_lookup = {i: item for i, item in enumerate(task_data)}

    # Track accuracy by population
    by_population = defaultdict(lambda: {'correct': 0, 'total': 0})

    for result in results.get('results', []):
        example_id = result['example_id']
        is_correct = result.get('correct', False)

        if example_id not in task_lookup:
            continue

        task_item = task_lookup[example_id]
        population = task_item.get('ground_truth', {}).get('population', 0)
        pop_bucket = get_population_bucket(population)

        by_population[pop_bucket]['total'] += 1
        if is_correct:
            by_population[pop_bucket]['correct'] += 1

    # Calculate accuracies
    result = {'by_population': {}}

    for pop, stats in by_population.items():
        if stats['total'] > 0:
            result['by_population'][pop] = {
                'accuracy': stats['correct'] / stats['total'] * 100,
                'correct': stats['correct'],
                'total': stats['total']
            }

    return result


def get_model_name(dirname: str) -> str:
    """Extract clean model name from directory name."""
    name = dirname.replace('gpsbench_', '').split('_20')[0]
    if len(name) > 25:
        name = name[:22] + '...'
    return name


def create_population_chart(all_results: dict, output_file: str):
    """Create bar chart showing accuracy by population bucket."""
    pop_buckets = ['<50k', '50k-100k', '100k-500k', '500k-1M', '>1M']
    models = list(all_results.keys())

    x = np.arange(len(pop_buckets))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, (model, data) in enumerate(all_results.items()):
        accuracies = []
        for pop in pop_buckets:
            acc = data['by_population'].get(pop, {}).get('accuracy', 0)
            accuracies.append(acc)

        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, accuracies, width, label=model, color=colors[i])

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{acc:.0f}%', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel('City Population', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Place Association: Accuracy by City Population\n(Larger cities are easier to identify)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pop_buckets, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_heatmap(all_results: dict, output_file: str):
    """Create heatmap showing model performance by population granularity."""
    models = list(all_results.keys())
    pop_buckets = ['<50k', '50k-100k', '100k-500k', '500k-1M', '>1M']

    # Build data matrix
    data = np.zeros((len(models), len(pop_buckets)))

    for i, model in enumerate(models):
        model_data = all_results[model]
        for j, pop in enumerate(pop_buckets):
            data[i, j] = model_data['by_population'].get(pop, {}).get('accuracy', 0)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(pop_buckets)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(pop_buckets, fontsize=11)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel('City Population', fontsize=12)

    # Add value annotations
    for i in range(len(models)):
        for j in range(len(pop_buckets)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%',
                          ha='center', va='center', fontsize=10,
                          color='white' if data[i, j] < 50 else 'black')

    ax.set_title('Place Association: Accuracy by City Population\n(Green = Better)',
                 fontsize=14, fontweight='bold')

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def print_summary(all_results: dict):
    """Print text summary of results."""
    print("\n" + "=" * 90)
    print("PLACE ASSOCIATION - ACCURACY BY CITY POPULATION (GRANULARITY)")
    print("=" * 90)

    print(f"\n{'Model':<30} {'<50k':>10} {'50k-100k':>10} {'100k-500k':>12} {'500k-1M':>10} {'>1M':>10}")
    print("-" * 90)

    # Sort by overall accuracy (average across buckets)
    def avg_acc(data):
        accs = [data['by_population'].get(b, {}).get('accuracy', 0)
                for b in ['<50k', '50k-100k', '100k-500k', '500k-1M', '>1M']]
        return sum(accs) / len(accs)

    for model, data in sorted(all_results.items(), key=lambda x: avg_acc(x[1]), reverse=True):
        p1 = data['by_population'].get('<50k', {}).get('accuracy', 0)
        p2 = data['by_population'].get('50k-100k', {}).get('accuracy', 0)
        p3 = data['by_population'].get('100k-500k', {}).get('accuracy', 0)
        p4 = data['by_population'].get('500k-1M', {}).get('accuracy', 0)
        p5 = data['by_population'].get('>1M', {}).get('accuracy', 0)
        print(f"{model:<30} {p1:>8.1f}% {p2:>8.1f}% {p3:>10.1f}% {p4:>8.1f}% {p5:>8.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Analyze Place Association by granularity')
    parser.add_argument('--results-dir', type=Path, default=Path('results'),
                        help='Results directory')
    parser.add_argument('--task-file', type=Path,
                        default=Path('data/track_applied/task1_place_association.json'),
                        help='Task data file')
    parser.add_argument('--min-samples', type=int, default=5000,
                        help='Minimum samples for model inclusion')
    args = parser.parse_args()

    print("Loading task data...")
    task_data = load_task_data(args.task_file)
    print(f"Loaded {len(task_data)} task examples")

    print("\nAnalyzing model results...")
    all_results = {}

    for model_dir in sorted(args.results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue

        result_file = model_dir / 'task_results' / 'applied_task1_place_association.json'
        if not result_file.exists():
            continue

        # Check sample count
        summary_file = model_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            total = summary.get('overall', {}).get('total_cases', 0)
            if total < args.min_samples:
                continue

        model_name = get_model_name(model_dir.name)
        print(f"  Processing: {model_name}")

        results = analyze_model_by_granularity(result_file, task_data)
        all_results[model_name] = results

    if not all_results:
        print("No results found!")
        return

    # Print summary
    print_summary(all_results)

    # Generate charts
    print("\nGenerating charts...")
    create_population_chart(all_results, 'chart_place_by_population.png')
    create_heatmap(all_results, 'chart_place_heatmap.png')

    print("\nDone! Generated:")
    print("  - chart_place_by_population.png  (Bar chart: accuracy by city population)")
    print("  - chart_place_heatmap.png        (Heatmap: all models x population buckets)")


if __name__ == '__main__':
    main()
