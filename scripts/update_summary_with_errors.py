#!/usr/bin/env python3
"""
Update summary and README with error metrics.

This script reads error_metrics.json and updates:
1. summary.json - adds error_metrics section
2. README.md - adds error metrics tables

Usage:
    python scripts/update_summary_with_errors.py <results_folder>
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def update_summary_json(results_path: Path):
    """Add error metrics to summary.json"""
    summary_file = results_path / "summary.json"
    error_metrics_file = results_path / "error_metrics.json"

    if not error_metrics_file.exists():
        print(f"No error_metrics.json found. Run compute_error_metrics.py first.")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    with open(error_metrics_file) as f:
        error_metrics = json.load(f)

    # Add error metrics to summary
    summary['error_metrics'] = error_metrics

    # Save updated summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Updated {summary_file}")


def generate_error_metrics_section(error_metrics: Dict[str, Any]) -> str:
    """Generate markdown section for error metrics"""
    if not error_metrics:
        return ""

    sections = ["\n## Error Metrics\n"]
    sections.append("Detailed error analysis for tasks with numeric outputs:\n")

    for task_key, metrics in error_metrics.items():
        # Parse task name
        task_name = task_key.replace('_', ' ').title()

        sections.append(f"\n### {task_name}\n")

        # Distance metrics
        if 'error_km' in metrics:
            err_km = metrics['error_km']
            err_pct = metrics['error_pct']

            sections.append("**Absolute Error (km):**\n")
            sections.append(f"- Mean: {err_km['mean']:.2f} km\n")
            sections.append(f"- Median: {err_km['median']:.2f} km\n")
            sections.append(f"- 95th percentile: {err_km['p95']:.2f} km\n")

            sections.append("\n**Relative Error:**\n")
            sections.append(f"- Mean: {err_pct['mean']:.3f}%\n")
            sections.append(f"- Median: {err_pct['median']:.3f}%\n")

            sections.append("\n**Accuracy at Different Tolerances:**\n\n")
            sections.append("| Tolerance | Accuracy | Examples |\n")
            sections.append("|-----------|----------|----------|\n")

            for tol, data in sorted(metrics['accuracy_at_tolerance'].items()):
                tol_label = tol.replace('%', '% of distance')
                sections.append(f"| ±{tol_label} | {data['accuracy']:.1f}% | {data['count']}/{data['total']} |\n")

        # Bearing metrics
        elif 'error_deg' in metrics:
            err_deg = metrics['error_deg']

            sections.append("**Absolute Error (degrees):**\n")
            sections.append(f"- Mean: {err_deg['mean']:.3f}°\n")
            sections.append(f"- Median: {err_deg['median']:.3f}°\n")
            sections.append(f"- 95th percentile: {err_deg['p95']:.3f}°\n")

            sections.append("\n**Accuracy at Different Tolerances:**\n\n")
            sections.append("| Tolerance | Accuracy | Examples |\n")
            sections.append("|-----------|----------|----------|\n")

            for tol, data in sorted(metrics['accuracy_at_tolerance'].items()):
                tol_label = tol.replace('deg', '°')
                sections.append(f"| ±{tol_label} | {data['accuracy']:.1f}% | {data['count']}/{data['total']} |\n")

    return ''.join(sections)


def update_readme(results_path: Path):
    """Add error metrics section to README.md"""
    readme_file = results_path / "README.md"
    error_metrics_file = results_path / "error_metrics.json"

    if not error_metrics_file.exists():
        print(f"No error_metrics.json found. Run compute_error_metrics.py first.")
        return

    with open(error_metrics_file) as f:
        error_metrics = json.load(f)

    with open(readme_file) as f:
        readme_content = f.read()

    # Check if error metrics section already exists
    if "## Error Metrics" in readme_content:
        # Remove old section
        start_idx = readme_content.find("## Error Metrics")
        # Find next ## section or end of file
        end_idx = readme_content.find("\n## ", start_idx + 1)
        if end_idx == -1:
            # No next section, remove to end
            readme_content = readme_content[:start_idx]
        else:
            readme_content = readme_content[:start_idx] + readme_content[end_idx:]

    # Generate new error metrics section
    error_section = generate_error_metrics_section(error_metrics)

    # Insert before "## File Structure" section
    insert_marker = "## File Structure"
    if insert_marker in readme_content:
        idx = readme_content.find(insert_marker)
        readme_content = readme_content[:idx] + error_section + "\n" + readme_content[idx:]
    else:
        # Append to end
        readme_content += "\n" + error_section

    # Save updated README
    with open(readme_file, 'w') as f:
        f.write(readme_content)

    print(f"✅ Updated {readme_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/update_summary_with_errors.py <results_folder>")
        print("\nExample:")
        print("  python scripts/update_summary_with_errors.py results/gpsbench_gpt-5-mini_20251205_142538")
        sys.exit(1)

    results_folder = sys.argv[1]
    results_path = Path(results_folder)

    if not results_path.exists():
        print(f"Error: Results folder not found: {results_folder}")
        sys.exit(1)

    print(f"Updating summary and README in: {results_path}")
    print("=" * 80)

    update_summary_json(results_path)
    update_readme(results_path)

    print("=" * 80)
    print("✅ Done! Summary and README updated with error metrics.")


if __name__ == "__main__":
    main()
