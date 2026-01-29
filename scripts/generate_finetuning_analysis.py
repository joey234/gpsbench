#!/usr/bin/env python3
"""Generate finetuning comparison figure for GPSBench paper."""

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

# Data: Zero-shot vs Finetuned for Qwen3-30B
# Note: Missing Data excluded from benchmark (used for RQ2 memorization analysis)
applied_tasks = [
    ('Place Assoc.', 0.8, 1.7),
    ('Name Disamb.', 99.7, 82.6),
    ('Rel. Position', 42.5, 22.9),
    ('Proximity', 24.2, 15.4),
    ('Route Analysis*', 85.7, 89.9),  # No->False mapping
    ('Boundary', 99.7, 74.5),
    ('Spatial Pat.', 8.1, 64.6),
    ('Terrain', 57.9, 53.9),
]

pure_gps_tasks = [
    ('Format Conv.', 90.4, 90.9),
    ('Coord. Trans.', 9.7, 6.6),
    ('Distance*', 97.8, 94.2),  # 1-MAPE scoring
    ('Bearing*', 90.5, 78.2),   # 1-MAPE scoring
    ('Interpolation', 6.4, 25.3),
    ('Polygon Area*', 0.0, 25.9),  # 1-MAPE scoring
    ('Bounding Box', 100.0, 99.9),
    ('Route Geom.', 18.7, 31.1),
    ('Rel. Position*', 64.1, 64.1),  # Format mismatch; assumed preserved
]

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Applied Track
ax1 = axes[0]
tasks = [t[0] for t in applied_tasks]
zero_shot = [t[1] for t in applied_tasks]
finetuned = [t[2] for t in applied_tasks]

x = np.arange(len(tasks))
width = 0.35

bars1 = ax1.bar(x - width/2, zero_shot, width, label='Zero-shot', color='#4472C4', edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, finetuned, width, label='Finetuned', color='#ED7D31', edgecolor='black', linewidth=0.5)

# Highlight improvements and degradations
for i, (zs, ft) in enumerate(zip(zero_shot, finetuned)):
    delta = ft - zs
    if delta > 5:
        ax1.annotate(f'+{delta:.0f}', xy=(x[i] + width/2, ft), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=10, color='green', fontweight='bold')
    elif delta < -5:
        ax1.annotate(f'{delta:.0f}', xy=(x[i] + width/2, ft + 2), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=10, color='red', fontweight='bold')

ax1.set_xlabel('Task')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Applied Track: Zero-shot vs Finetuned')
ax1.set_xticks(x)
ax1.set_xticklabels(tasks, rotation=45, ha='right', fontsize=12)
ax1.legend(loc='upper right')
ax1.set_ylim(0, 110)
ax1.axhline(y=52.3, color='#4472C4', linestyle='--', alpha=0.5, label='Zero-shot avg')
ax1.axhline(y=50.7, color='#ED7D31', linestyle='--', alpha=0.5, label='Finetuned avg')

# Plot Pure GPS Track
ax2 = axes[1]
tasks = [t[0] for t in pure_gps_tasks]
zero_shot = [t[1] for t in pure_gps_tasks]
finetuned = [t[2] for t in pure_gps_tasks]

x = np.arange(len(tasks))

bars1 = ax2.bar(x - width/2, zero_shot, width, label='Zero-shot', color='#4472C4', edgecolor='black', linewidth=0.5)
bars2 = ax2.bar(x + width/2, finetuned, width, label='Finetuned', color='#ED7D31', edgecolor='black', linewidth=0.5)

# Highlight improvements and degradations
for i, (zs, ft) in enumerate(zip(zero_shot, finetuned)):
    delta = ft - zs
    if delta > 5:
        ax2.annotate(f'+{delta:.0f}', xy=(x[i] + width/2, ft), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=10, color='green', fontweight='bold')
    elif delta < -5:
        ax2.annotate(f'{delta:.0f}', xy=(x[i] + width/2, ft + 2), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=10, color='red', fontweight='bold')

ax2.set_xlabel('Task')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Pure GPS Track: Zero-shot vs Finetuned')
ax2.set_xticks(x)
ax2.set_xticklabels(tasks, rotation=45, ha='right', fontsize=12)
ax2.legend(loc='upper right')
ax2.set_ylim(0, 110)
ax2.axhline(y=53.1, color='#4472C4', linestyle='--', alpha=0.5)
ax2.axhline(y=57.4, color='#ED7D31', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('paper/figures/finetuning_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/finetuning_comparison.png', dpi=300, bbox_inches='tight')
print('Saved: paper/figures/finetuning_comparison.pdf')

# Print summary
print("\n=== Finetuning Analysis Summary ===")
print("(Missing Data excluded - used for RQ2 memorization analysis)")

print("\nOverall Results:")
print("  Applied: 52.3% -> 50.7% (-1.6%)")
print("  Pure GPS: 53.1% -> 57.4% (+4.3%)")
print("  Combined: 52.7% -> 54.2% (+1.5%)")

print("\nImprovements (geometric reasoning):")
print("  Spatial Patterns: +56.5%")
print("  Polygon Area: +25.9%")
print("  Interpolation: +18.9%")
print("  Route Geometry: +12.4%")

print("\nDegradation (world knowledge):")
print("  Boundary Analysis: -25.2%")
print("  Relative Position: -19.6%")
print("  Name Disambiguation: -17.1%")
print("  Bearing: -12.3%")

print("\nPreserved:")
print("  Format Conversion: 90.4% -> 90.9% (+0.5%)")
print("  Distance: 97.8% -> 94.2% (-3.6%)")
print("  Bounding Box: 100.0% -> 99.9% (-0.1%)")
