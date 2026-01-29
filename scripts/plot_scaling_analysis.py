#!/usr/bin/env python3
"""
Scaling analysis for GPSBench using Mistral and Qwen models with known parameter counts.
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['figure.dpi'] = 150
import numpy as np
import json

# Model parameter sizes (in billions)
MODEL_SIZES = {
    'mistral-small-24b-instruct-2501': 24,
    'mistral-large-2512': 123,
    'qwen3-8b': 8,
    'qwen3-14b': 14,
    'qwen3-30b-a3b-instruct-2507': 30,  # 30B total, 3B active (MoE)
    'qwen3-235b-a22b-2507': 235,  # 235B total, 22B active (MoE)
}

# Whether model is MoE
MODEL_IS_MOE = {
    'mistral-small-24b-instruct-2501': False,
    'mistral-large-2512': False,
    'qwen3-8b': False,
    'qwen3-14b': False,
    'qwen3-30b-a3b-instruct-2507': True,
    'qwen3-235b-a22b-2507': True,
}

MODEL_FAMILY = {
    'mistral-small-24b-instruct-2501': 'Mistral',
    'mistral-large-2512': 'Mistral',
    'qwen3-8b': 'Qwen3',
    'qwen3-14b': 'Qwen3',
    'qwen3-30b-a3b-instruct-2507': 'Qwen3',
    'qwen3-235b-a22b-2507': 'Qwen3',
}

MODEL_DISPLAY_NAMES = {
    'mistral-small-24b-instruct-2501': '24B',
    'mistral-large-2512': '123B',
    'qwen3-8b': '8B',
    'qwen3-14b': '14B',
    'qwen3-30b-a3b-instruct-2507': '30B*',
    'qwen3-235b-a22b-2507': '235B*',
}

def load_results_from_csv():
    """Load results from CSV excluding Missing Data task."""
    df = pd.read_csv('/Users/hungthinh/workspace/GPSBench/analysis/results.csv')
    # Exclude Missing Data task
    df = df[~df['Task'].str.contains('Missing Data', case=False, na=False)]
    return df

def load_qwen_from_json():
    """Load Qwen 8B and 14B results from JSON files."""
    results = []

    json_files = {
        'qwen3-8b': '/Users/hungthinh/workspace/GPSBench/results/gpsbench_qwen_qwen3-8b_20260114_154454/summary.json',
        'qwen3-14b': '/Users/hungthinh/workspace/GPSBench/results/gpsbench_qwen_qwen3-14b_20260114_150050/summary.json',
    }

    for model, path in json_files.items():
        with open(path) as f:
            data = json.load(f)

        # Calculate accuracies excluding Missing Data
        pure_gps_correct = 0
        pure_gps_total = 0
        applied_correct = 0
        applied_total = 0

        for task in data['tracks']['pure_gps']['tasks_summary']:
            pure_gps_correct += task['correct']
            pure_gps_total += task['total']

        for task in data['tracks']['applied']['tasks_summary']:
            if 'Missing Data' not in task['task_name']:
                applied_correct += task['correct']
                applied_total += task['total']

        results.append({
            'Model': model,
            'Pure_GPS': pure_gps_correct / pure_gps_total * 100,
            'Applied': applied_correct / applied_total * 100,
            'Overall': (pure_gps_correct + applied_correct) / (pure_gps_total + applied_total) * 100,
        })

    return results

def calculate_track_accuracies():
    """Calculate accuracies for each model and track, excluding Missing Data."""
    df = load_results_from_csv()
    results = []

    # Process models from CSV
    csv_models = ['mistral-small-24b-instruct-2501', 'mistral-large-2512',
                  'qwen3-30b-a3b-instruct-2507', 'qwen3-235b-a22b-2507']

    for model in csv_models:
        model_df = df[df['Model'] == model]
        task_df = model_df[model_df['Task'] != 'ALL']

        pure_gps = task_df[task_df['Track'] == 'PURE_GPS']
        applied = task_df[task_df['Track'] == 'APPLIED']

        pure_gps_acc = pure_gps['Correct'].sum() / pure_gps['Total'].sum() * 100 if len(pure_gps) > 0 else None
        applied_acc = applied['Correct'].sum() / applied['Total'].sum() * 100 if len(applied) > 0 else None
        overall_acc = task_df['Correct'].sum() / task_df['Total'].sum() * 100 if len(task_df) > 0 else None

        results.append({
            'Model': model,
            'Pure_GPS': pure_gps_acc,
            'Applied': applied_acc,
            'Overall': overall_acc,
        })

    # Add Qwen 8B and 14B from JSON
    results.extend(load_qwen_from_json())

    # Convert to DataFrame and add metadata
    results_df = pd.DataFrame(results)
    results_df['Family'] = results_df['Model'].map(MODEL_FAMILY)
    results_df['Size_B'] = results_df['Model'].map(MODEL_SIZES)
    results_df['Is_MoE'] = results_df['Model'].map(MODEL_IS_MOE)
    results_df['Display'] = results_df['Model'].map(MODEL_DISPLAY_NAMES)

    return results_df

def plot_scaling(results_df):
    """Create scaling analysis plot."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    tracks = [('Pure_GPS', 'Pure GPS Track'), ('Applied', 'Applied Track'), ('Overall', 'Overall')]
    colors = {'Mistral': '#E74C3C', 'Qwen3': '#3498DB'}
    markers = {'Mistral': 'o', 'Qwen3': 's'}

    for ax, (track, title) in zip(axes, tracks):
        for family in ['Mistral', 'Qwen3']:
            family_df = results_df[results_df['Family'] == family].sort_values('Size_B')

            ax.plot(family_df['Size_B'], family_df[track],
                   color=colors[family], marker=markers[family],
                   markersize=8, linewidth=2, label=family)

            # Add labels
            for _, row in family_df.iterrows():
                ax.annotate(row['Display'],
                           (row['Size_B'], row[track]),
                           textcoords="offset points",
                           xytext=(0, 8),
                           ha='center',
                           fontsize=10)

        ax.set_xlabel('Parameters (B)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.set_xscale('log')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(35, 85)

    plt.tight_layout()
    plt.savefig('/Users/hungthinh/workspace/GPSBench/paper/figures/scaling_analysis.pdf',
                bbox_inches='tight', dpi=300)
    plt.savefig('/Users/hungthinh/workspace/GPSBench/paper/figures/scaling_analysis.png',
                bbox_inches='tight', dpi=300)
    print("Saved scaling_analysis.pdf/png")

    return fig

def main():
    # Load and process data
    results_df = calculate_track_accuracies()
    results_df = results_df.sort_values(['Family', 'Size_B'])

    print("Scaling Analysis Results (excluding Missing Data):")
    print("=" * 80)
    print(results_df[['Model', 'Family', 'Size_B', 'Is_MoE', 'Pure_GPS', 'Applied', 'Overall']].to_string(index=False))
    print()

    # Print scaling analysis
    print("\nScaling Analysis:")
    print("-" * 60)

    # Mistral scaling
    mistral_df = results_df[results_df['Family'] == 'Mistral'].sort_values('Size_B')
    if len(mistral_df) >= 2:
        small, large = mistral_df.iloc[0], mistral_df.iloc[1]
        print(f"\nMistral: {small['Size_B']:.0f}B → {large['Size_B']:.0f}B ({large['Size_B']/small['Size_B']:.1f}x)")
        for track in ['Pure_GPS', 'Applied', 'Overall']:
            print(f"  {track}: {small[track]:.1f}% → {large[track]:.1f}% ({large[track]-small[track]:+.1f}%)")

    # Qwen dense scaling
    qwen_dense = results_df[(results_df['Family'] == 'Qwen3') & (~results_df['Is_MoE'])].sort_values('Size_B')
    if len(qwen_dense) >= 2:
        small, large = qwen_dense.iloc[0], qwen_dense.iloc[-1]
        print(f"\nQwen3 (dense): {small['Size_B']:.0f}B → {large['Size_B']:.0f}B ({large['Size_B']/small['Size_B']:.1f}x)")
        for track in ['Pure_GPS', 'Applied', 'Overall']:
            print(f"  {track}: {small[track]:.1f}% → {large[track]:.1f}% ({large[track]-small[track]:+.1f}%)")

    # Qwen MoE scaling
    qwen_moe = results_df[(results_df['Family'] == 'Qwen3') & (results_df['Is_MoE'])].sort_values('Size_B')
    if len(qwen_moe) >= 2:
        small, large = qwen_moe.iloc[0], qwen_moe.iloc[-1]
        print(f"\nQwen3 (MoE): {small['Size_B']:.0f}B → {large['Size_B']:.0f}B ({large['Size_B']/small['Size_B']:.1f}x)")
        for track in ['Pure_GPS', 'Applied', 'Overall']:
            print(f"  {track}: {small[track]:.1f}% → {large[track]:.1f}% ({large[track]-small[track]:+.1f}%)")

    # Full Qwen scaling (8B to 235B)
    qwen_all = results_df[results_df['Family'] == 'Qwen3'].sort_values('Size_B')
    if len(qwen_all) >= 2:
        small, large = qwen_all.iloc[0], qwen_all.iloc[-1]
        print(f"\nQwen3 (all): {small['Size_B']:.0f}B → {large['Size_B']:.0f}B ({large['Size_B']/small['Size_B']:.1f}x)")
        for track in ['Pure_GPS', 'Applied', 'Overall']:
            print(f"  {track}: {small[track]:.1f}% → {large[track]:.1f}% ({large[track]-small[track]:+.1f}%)")

    # Create plot
    plot_scaling(results_df)

    print("\nDone!")

if __name__ == '__main__':
    main()
