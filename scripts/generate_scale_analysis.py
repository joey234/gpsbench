#!/usr/bin/env python3
"""Generate scale analysis figure for GPSBench paper."""

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
from scipy import stats

# Models with known parameter counts
# Format: (model_name, total_params_B, active_params_B, applied_acc, pure_gps_acc, is_moe)
models_with_params = [
    ("Qwen3-8B", 8, 8, 62.2, 43.9, False),
    ("Qwen3-14B", 14, 14, 59.5, 41.0, False),
    ("Mistral-Small-24B", 24, 24, 59.8, 42.1, False),
    ("Qwen3-30B", 30, 3, 53.9, 53.1, True),  # MoE: 30B total, 3B active
    ("Mistral-Large", 123, 123, 72.5, 54.3, False),
    ("Qwen3-235B", 235, 22, 69.8, 55.2, True),  # MoE: 235B total, 22B active
]

# Extract data
names = [m[0] for m in models_with_params]
total_params = np.array([m[1] for m in models_with_params])
active_params = np.array([m[2] for m in models_with_params])
applied_acc = np.array([m[3] for m in models_with_params])
pure_gps_acc = np.array([m[4] for m in models_with_params])
is_moe = np.array([m[5] for m in models_with_params])

# Calculate correlations
log_total = np.log10(total_params)
log_active = np.log10(active_params)

# Correlation with total params
r_total_applied, p_total_applied = stats.pearsonr(log_total, applied_acc)
r_total_pure, p_total_pure = stats.pearsonr(log_total, pure_gps_acc)

# Correlation with active params
r_active_applied, p_active_applied = stats.pearsonr(log_active, applied_acc)
r_active_pure, p_active_pure = stats.pearsonr(log_active, pure_gps_acc)

print("=== Scale Analysis Results ===")
print(f"\nCorrelation with log(Total Params):")
print(f"  Applied Track:  r={r_total_applied:.3f}, p={p_total_applied:.3f}")
print(f"  Pure GPS Track: r={r_total_pure:.3f}, p={p_total_pure:.3f}")
print(f"\nCorrelation with log(Active Params):")
print(f"  Applied Track:  r={r_active_applied:.3f}, p={p_active_applied:.3f}")
print(f"  Pure GPS Track: r={r_active_pure:.3f}, p={p_active_pure:.3f}")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Colors for dense vs MoE
colors = ['#2ecc71' if not moe else '#e74c3c' for moe in is_moe]
markers = ['o' if not moe else 's' for moe in is_moe]

# Plot 1: Total Params vs Accuracy
ax1 = axes[0]
for i, (name, tp, ap, pure, moe) in enumerate(zip(names, total_params, active_params, pure_gps_acc, is_moe)):
    marker = 's' if moe else 'o'
    color = '#e74c3c' if moe else '#2ecc71'
    label = 'MoE' if moe and i == 3 else ('Dense' if not moe and i == 0 else None)
    ax1.scatter(tp, pure, c=color, s=100, marker=marker, label=label, edgecolors='black', linewidths=0.5, zorder=5)

    # Add model name labels
    offset_x = 1.1
    offset_y = 1.5
    if name == "Qwen3-8B":
        offset_y = -2.5
    elif name == "Qwen3-14B":
        offset_y = -2.5
    elif name == "Mistral-Small-24B":
        offset_y = -2.5
    elif name == "Qwen3-30B":
        offset_x = 1.15
        offset_y = 1.5
    ax1.annotate(name, (tp, pure), xytext=(tp*offset_x, pure+offset_y), fontsize=11)

# Regression line
slope, intercept, _, _, _ = stats.linregress(log_total, pure_gps_acc)
x_line = np.logspace(np.log10(5), np.log10(300), 100)
y_line = slope * np.log10(x_line) + intercept
ax1.plot(x_line, y_line, 'k--', alpha=0.5, label=f'Fit (r={r_total_pure:.2f})')

ax1.set_xscale('log')
ax1.set_xlabel('Total Parameters (B)', fontsize=14)
ax1.set_ylabel('Pure GPS Accuracy (%)', fontsize=14)
ax1.set_title('Model Scale vs Pure GPS Performance', fontsize=15)
ax1.legend(loc='lower right', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(5, 300)
ax1.set_ylim(35, 60)

# Plot 2: Active Params vs Accuracy (shows MoE efficiency)
ax2 = axes[1]
for i, (name, tp, ap, pure, moe) in enumerate(zip(names, total_params, active_params, pure_gps_acc, is_moe)):
    marker = 's' if moe else 'o'
    color = '#e74c3c' if moe else '#2ecc71'
    label = 'MoE' if moe and i == 3 else ('Dense' if not moe and i == 0 else None)
    ax2.scatter(ap, pure, c=color, s=100, marker=marker, label=label, edgecolors='black', linewidths=0.5, zorder=5)

    # Add model name labels
    offset_x = 1.1
    offset_y = 1.5
    if name == "Qwen3-8B":
        offset_y = -2.5
    elif name == "Qwen3-14B":
        offset_y = -2.5
    elif name == "Mistral-Small-24B":
        offset_y = -2.5
    elif name == "Qwen3-30B":
        offset_x = 1.15
    ax2.annotate(name, (ap, pure), xytext=(ap*offset_x, pure+offset_y), fontsize=11)

# Regression line
slope_active, intercept_active, _, _, _ = stats.linregress(log_active, pure_gps_acc)
x_line_active = np.logspace(np.log10(2), np.log10(150), 100)
y_line_active = slope_active * np.log10(x_line_active) + intercept_active
ax2.plot(x_line_active, y_line_active, 'k--', alpha=0.5, label=f'Fit (r={r_active_pure:.2f})')

ax2.set_xscale('log')
ax2.set_xlabel('Active Parameters (B)', fontsize=14)
ax2.set_ylabel('Pure GPS Accuracy (%)', fontsize=14)
ax2.set_title('Active Parameters vs Pure GPS Performance', fontsize=15)
ax2.legend(loc='lower right', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(2, 150)
ax2.set_ylim(35, 60)

plt.tight_layout()
plt.savefig('/Users/hungthinh/workspace/GPSBench/paper/figures/scale_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/hungthinh/workspace/GPSBench/paper/figures/scale_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved figures to paper/figures/scale_analysis.pdf and .png")

# Print summary table
print("\n=== Model Data Summary ===")
print(f"{'Model':<20} {'Total (B)':<12} {'Active (B)':<12} {'Applied':<10} {'Pure GPS':<10} {'Type':<8}")
print("-" * 72)
for name, tp, ap, app, pure, moe in models_with_params:
    mtype = "MoE" if moe else "Dense"
    print(f"{name:<20} {tp:<12} {ap:<12} {app:<10.1f} {pure:<10.1f} {mtype:<8}")

# MoE efficiency analysis
print("\n=== MoE Efficiency Analysis ===")
dense_8b = models_with_params[0]  # Qwen3-8B
moe_30b = models_with_params[3]   # Qwen3-30B (3B active)
print(f"Qwen3-30B (3B active) vs Qwen3-8B (8B dense):")
print(f"  Pure GPS: {moe_30b[4]:.1f}% vs {dense_8b[4]:.1f}% (+{moe_30b[4]-dense_8b[4]:.1f}%)")
print(f"  Applied:  {moe_30b[3]:.1f}% vs {dense_8b[3]:.1f}% ({moe_30b[3]-dense_8b[3]:+.1f}%)")
print(f"  With only {moe_30b[2]/dense_8b[1]*100:.0f}% of active parameters, MoE achieves +{moe_30b[4]-dense_8b[4]:.1f}% Pure GPS")
