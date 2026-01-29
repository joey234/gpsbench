#!/usr/bin/env python3
"""
Generate task taxonomy diagram for GPSBench.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np


def create_taxonomy_diagram(output_path):
    """Create a hierarchical taxonomy diagram."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Colors
    root_color = '#3498db'
    pure_color = '#1abc9c'
    applied_color = '#e67e22'
    pure_task_color = '#d5f4f0'
    applied_task_color = '#fdebd0'

    # Helper function to draw rounded rectangle
    def draw_box(x, y, w, h, color, text, fontsize=10, bold=False, text_color='white'):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor=color, edgecolor='gray', linewidth=1)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight=weight, color=text_color)

    # Root node
    draw_box(7, 8.2, 3, 0.7, root_color, 'GPSBench', fontsize=14, bold=True)

    # Track nodes
    draw_box(3.5, 6.8, 2.8, 0.6, pure_color, 'Pure GPS Track', fontsize=14, bold=True)
    draw_box(10.5, 6.8, 2.8, 0.6, applied_color, 'Applied Track', fontsize=14, bold=True)

    # Connecting lines from root to tracks
    ax.plot([7, 7], [7.85, 7.4], color='gray', linewidth=1.5)
    ax.plot([3.5, 7], [7.4, 7.4], color='gray', linewidth=1.5)
    ax.plot([7, 10.5], [7.4, 7.4], color='gray', linewidth=1.5)
    ax.plot([3.5, 3.5], [7.4, 7.1], color='gray', linewidth=1.5)
    ax.plot([10.5, 10.5], [7.4, 7.1], color='gray', linewidth=1.5)

    # Pure GPS tasks
    pure_tasks_left = [
        'Format Conversion',
        'Coord. Transform',
        'Distance Calculation',
        'Bearing Computation',
        'Interpolation'
    ]
    pure_tasks_right = [
        'Area & Perimeter',
        'Bounding Box',
        'Route Geometry',
        'Relative Position'
    ]

    # Draw Pure GPS tasks (left column)
    for i, task in enumerate(pure_tasks_left):
        y = 5.8 - i * 0.75
        draw_box(1.8, y, 2.3, 0.55, pure_task_color, task, fontsize=12, text_color='#1a1a1a')

    # Draw Pure GPS tasks (right column)
    for i, task in enumerate(pure_tasks_right):
        y = 5.8 - i * 0.75
        draw_box(5.2, y, 2.3, 0.55, pure_task_color, task, fontsize=12, text_color='#1a1a1a')

    # Connecting lines for Pure GPS
    ax.plot([3.5, 3.5], [6.5, 6.2], color='gray', linewidth=1)
    ax.plot([1.8, 3.5], [6.2, 6.2], color='gray', linewidth=1)
    ax.plot([3.5, 5.2], [6.2, 6.2], color='gray', linewidth=1)
    ax.plot([1.8, 1.8], [6.2, 6.05], color='gray', linewidth=1)
    ax.plot([5.2, 5.2], [6.2, 6.05], color='gray', linewidth=1)

    # Vertical lines for task lists
    ax.plot([1.8, 1.8], [5.55, 2.9], color='#cccccc', linewidth=0.8, linestyle='-')
    ax.plot([5.2, 5.2], [5.55, 3.65], color='#cccccc', linewidth=0.8, linestyle='-')

    # Applied tasks
    applied_tasks_left = [
        'Place Association',
        'Name Disambiguation',
        'Relative Position',
        'Proximity',
        'Route Analysis'
    ]
    applied_tasks_right = [
        'Spatial Patterns',
        'Boundary Analysis',
        'Terrain Classification'
    ]

    # Draw Applied tasks (left column)
    for i, task in enumerate(applied_tasks_left):
        y = 5.8 - i * 0.75
        draw_box(8.8, y, 2.3, 0.55, applied_task_color, task, fontsize=12, text_color='#1a1a1a')

    # Draw Applied tasks (right column)
    for i, task in enumerate(applied_tasks_right):
        y = 5.8 - i * 0.75
        draw_box(12.2, y, 2.3, 0.55, applied_task_color, task, fontsize=12, text_color='#1a1a1a')

    # Connecting lines for Applied
    ax.plot([10.5, 10.5], [6.5, 6.2], color='gray', linewidth=1)
    ax.plot([8.8, 10.5], [6.2, 6.2], color='gray', linewidth=1)
    ax.plot([10.5, 12.2], [6.2, 6.2], color='gray', linewidth=1)
    ax.plot([8.8, 8.8], [6.2, 6.05], color='gray', linewidth=1)
    ax.plot([12.2, 12.2], [6.2, 6.05], color='gray', linewidth=1)

    # Vertical lines for task lists
    ax.plot([8.8, 8.8], [5.55, 2.9], color='#cccccc', linewidth=0.8, linestyle='-')
    ax.plot([12.2, 12.2], [5.55, 3.65], color='#cccccc', linewidth=0.8, linestyle='-')

    # Category labels
    ax.text(1.8, 6.35, 'Coordinate Ops', ha='center', va='bottom', fontsize=11,
            style='italic', color='gray')
    ax.text(5.2, 6.35, 'Spatial Analysis', ha='center', va='bottom', fontsize=11,
            style='italic', color='gray')
    ax.text(8.8, 6.35, 'Geo Knowledge', ha='center', va='bottom', fontsize=11,
            style='italic', color='gray')
    ax.text(12.2, 6.35, 'Complex Reasoning', ha='center', va='bottom', fontsize=11,
            style='italic', color='gray')

    # Statistics
    ax.text(3.5, 2.3, '9 tasks × 3,400 = 30,600 samples', ha='center', va='center',
            fontsize=12, color=pure_color, fontweight='bold')
    ax.text(10.5, 2.3, '8 tasks × 3,400 = 27,200 samples', ha='center', va='center',
            fontsize=12, color=applied_color, fontweight='bold')

    # Total
    ax.text(7, 1.5, 'Total: 17 Tasks, 57,800 Samples', ha='center', va='center',
            fontsize=15, color=root_color, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=pure_color, edgecolor='gray', label='Pure GPS Track'),
        mpatches.Patch(facecolor=applied_color, edgecolor='gray', label='Applied Track'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    from pathlib import Path
    output_path = str(Path(__file__).parent / "task_taxonomy.png")
    create_taxonomy_diagram(output_path)
