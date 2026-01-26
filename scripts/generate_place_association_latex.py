#!/usr/bin/env python3
"""
Generate LaTeX tables for place_association task data distribution.

Shows breakdown by:
- Geographic region
- Country (top N)
- Admin1/Province coverage
- Noise level
- Difficulty

Usage:
    python scripts/generate_place_association_latex.py
"""

import json
from pathlib import Path
from collections import Counter
import argparse


def load_data(data_path: Path):
    """Load place_association data."""
    file_path = data_path / "track_applied" / "task1_place_association.json"
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_region_table(data: list) -> str:
    """Generate LaTeX table for geographic region distribution."""
    regions = Counter(s['ground_truth'].get('region', 'Unknown') for s in data)
    total = len(data)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Place Association: Geographic Region Distribution}",
        r"\label{tab:place_association_regions}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Region} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]

    for region, count in sorted(regions.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        lines.append(f"{region} & {count} & {pct:.1f} \\\\")

    lines.extend([
        r"\midrule",
        f"\\textbf{{Total}} & \\textbf{{{total}}} & \\textbf{{100.0}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_country_table(data: list, top_n: int = 15) -> str:
    """Generate LaTeX table for top N countries."""
    countries = Counter(s['ground_truth'].get('country_name', 'Unknown') for s in data)
    total = len(data)
    top_countries = countries.most_common(top_n)
    other_count = sum(c for _, c in countries.most_common()[top_n:])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Place Association: Top {top_n} Countries}}",
        r"\label{tab:place_association_countries}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Country} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]

    for country, count in top_countries:
        pct = count / total * 100
        lines.append(f"{country} & {count} & {pct:.1f} \\\\")

    if other_count > 0:
        other_pct = other_count / total * 100
        lines.append(r"\midrule")
        lines.append(f"Other ({len(countries) - top_n} countries) & {other_count} & {other_pct:.1f} \\\\")

    lines.extend([
        r"\midrule",
        f"\\textbf{{Total}} & \\textbf{{{total}}} & \\textbf{{100.0}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_noise_table(data: list) -> str:
    """Generate LaTeX table for noise level distribution."""
    noise_types = Counter(s['ground_truth'].get('noise_type', 'unknown') for s in data)
    total = len(data)

    # Define order for noise types
    noise_order = ['clean', 'gps_10m', 'gps_50m', 'gps_100m', 'gps_500m', 'gps_1000m']
    noise_labels = {
        'clean': 'Clean (no noise)',
        'gps_10m': 'GPS 10m ($\\sigma$ = 10m)',
        'gps_50m': 'GPS 50m ($\\sigma$ = 50m)',
        'gps_100m': 'GPS 100m ($\\sigma$ = 100m)',
        'gps_500m': 'GPS 500m ($\\sigma$ = 500m)',
        'gps_1000m': 'GPS 1000m ($\\sigma$ = 1km)',
    }

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Place Association: Coordinate Noise Level Distribution}",
        r"\label{tab:place_association_noise}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Noise Level} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]

    for noise_type in noise_order:
        if noise_type in noise_types:
            count = noise_types[noise_type]
            pct = count / total * 100
            label = noise_labels.get(noise_type, noise_type)
            lines.append(f"{label} & {count} & {pct:.1f} \\\\")

    # Add any unknown noise types
    for noise_type, count in noise_types.items():
        if noise_type not in noise_order:
            pct = count / total * 100
            lines.append(f"{noise_type} & {count} & {pct:.1f} \\\\")

    lines.extend([
        r"\midrule",
        f"\\textbf{{Total}} & \\textbf{{{total}}} & \\textbf{{100.0}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_difficulty_table(data: list) -> str:
    """Generate LaTeX table for difficulty distribution."""
    difficulties = Counter(s.get('difficulty', 'unknown') for s in data)
    total = len(data)

    difficulty_order = ['easy', 'medium', 'hard']

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Place Association: Difficulty Distribution}",
        r"\label{tab:place_association_difficulty}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Difficulty} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]

    for diff in difficulty_order:
        if diff in difficulties:
            count = difficulties[diff]
            pct = count / total * 100
            lines.append(f"{diff.capitalize()} & {count} & {pct:.1f} \\\\")

    lines.extend([
        r"\midrule",
        f"\\textbf{{Total}} & \\textbf{{{total}}} & \\textbf{{100.0}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_summary_table(data: list) -> str:
    """Generate a combined summary table."""
    total = len(data)

    # Counts
    regions = Counter(s['ground_truth'].get('region', 'Unknown') for s in data)
    countries = Counter(s['ground_truth'].get('country_name', 'Unknown') for s in data)
    noise_types = Counter(s['ground_truth'].get('noise_type', 'unknown') for s in data)
    difficulties = Counter(s.get('difficulty', 'unknown') for s in data)
    with_admin1 = sum(1 for s in data if s['ground_truth'].get('admin1_name'))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Place Association Task: Dataset Summary}",
        r"\label{tab:place_association_summary}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"\textbf{Attribute} & \textbf{Value} \\",
        r"\midrule",
        f"Total samples & {total:,} \\\\",
        f"Unique countries & {len(countries)} \\\\",
        f"Geographic regions & {len(regions)} \\\\",
        f"Admin1 coverage & {with_admin1:,} ({with_admin1/total*100:.1f}\\%) \\\\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{By Difficulty}} \\",
        f"\\quad Easy & {difficulties.get('easy', 0):,} ({difficulties.get('easy', 0)/total*100:.1f}\\%) \\\\",
        f"\\quad Medium & {difficulties.get('medium', 0):,} ({difficulties.get('medium', 0)/total*100:.1f}\\%) \\\\",
        f"\\quad Hard & {difficulties.get('hard', 0):,} ({difficulties.get('hard', 0)/total*100:.1f}\\%) \\\\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{By Noise Level}} \\",
        f"\\quad Clean & {noise_types.get('clean', 0):,} ({noise_types.get('clean', 0)/total*100:.1f}\\%) \\\\",
        f"\\quad GPS 10m & {noise_types.get('gps_10m', 0):,} ({noise_types.get('gps_10m', 0)/total*100:.1f}\\%) \\\\",
        f"\\quad GPS 50m & {noise_types.get('gps_50m', 0):,} ({noise_types.get('gps_50m', 0)/total*100:.1f}\\%) \\\\",
        f"\\quad GPS 100m & {noise_types.get('gps_100m', 0):,} ({noise_types.get('gps_100m', 0)/total*100:.1f}\\%) \\\\",
        f"\\quad GPS 500m & {noise_types.get('gps_500m', 0):,} ({noise_types.get('gps_500m', 0)/total*100:.1f}\\%) \\\\",
        f"\\quad GPS 1000m & {noise_types.get('gps_1000m', 0):,} ({noise_types.get('gps_1000m', 0)/total*100:.1f}\\%) \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def generate_combined_table(data: list, top_countries: int = 10) -> str:
    """Generate a single comprehensive table with regions and noise breakdown."""
    total = len(data)

    regions = Counter(s['ground_truth'].get('region', 'Unknown') for s in data)
    countries = Counter(s['ground_truth'].get('country_name', 'Unknown') for s in data)
    noise_types = Counter(s['ground_truth'].get('noise_type', 'unknown') for s in data)

    noise_order = ['clean', 'gps_10m', 'gps_50m', 'gps_100m', 'gps_500m', 'gps_1000m']

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Place Association Task: Data Distribution by Region and Noise Level}",
        r"\label{tab:place_association_combined}",
        r"\small",
        r"\begin{tabular}{l|rr|rrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c|}{\textbf{Overall}} & \multicolumn{6}{c}{\textbf{By Noise Level}} \\",
        r"\textbf{Region} & \textbf{N} & \textbf{\%} & \textbf{Clean} & \textbf{10m} & \textbf{50m} & \textbf{100m} & \textbf{500m} & \textbf{1km} \\",
        r"\midrule",
    ]

    # Group data by region and noise
    region_noise = {}
    for s in data:
        region = s['ground_truth'].get('region', 'Unknown')
        noise = s['ground_truth'].get('noise_type', 'unknown')
        if region not in region_noise:
            region_noise[region] = Counter()
        region_noise[region][noise] += 1

    for region, count in sorted(regions.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        noise_counts = region_noise.get(region, Counter())
        noise_vals = [str(noise_counts.get(nt, 0)) for nt in noise_order]
        lines.append(f"{region} & {count} & {pct:.1f} & " + " & ".join(noise_vals) + r" \\")

    # Totals
    noise_totals = [str(noise_types.get(nt, 0)) for nt in noise_order]
    lines.extend([
        r"\midrule",
        f"\\textbf{{Total}} & \\textbf{{{total}}} & \\textbf{{100}} & " + " & ".join(f"\\textbf{{{v}}}" for v in noise_totals) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for place_association data")
    parser.add_argument('--data-path', type=str, default='data', help='Path to data directory')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: print to stdout)')
    parser.add_argument('--top-countries', type=int, default=15, help='Number of top countries to show')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    data = load_data(data_path)

    print(f"Loaded {len(data)} samples\n")

    # Generate all tables
    tables = []

    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    summary = generate_summary_table(data)
    print(summary)
    tables.append(summary)

    print("\n" + "=" * 70)
    print("COMBINED REGION + NOISE TABLE")
    print("=" * 70)
    combined = generate_combined_table(data)
    print(combined)
    tables.append(combined)

    print("\n" + "=" * 70)
    print("REGION TABLE")
    print("=" * 70)
    region_table = generate_region_table(data)
    print(region_table)
    tables.append(region_table)

    print("\n" + "=" * 70)
    print("TOP COUNTRIES TABLE")
    print("=" * 70)
    country_table = generate_country_table(data, args.top_countries)
    print(country_table)
    tables.append(country_table)

    print("\n" + "=" * 70)
    print("NOISE LEVEL TABLE")
    print("=" * 70)
    noise_table = generate_noise_table(data)
    print(noise_table)
    tables.append(noise_table)

    print("\n" + "=" * 70)
    print("DIFFICULTY TABLE")
    print("=" * 70)
    difficulty_table = generate_difficulty_table(data)
    print(difficulty_table)
    tables.append(difficulty_table)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write("% Place Association Task LaTeX Tables\n")
            f.write("% Generated by scripts/generate_place_association_latex.py\n\n")
            f.write("\n\n".join(tables))
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
