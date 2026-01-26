#!/usr/bin/env python3
"""
Generate LaTeX tables from GPSBench evaluation results.

Usage:
    python scripts/generate_latex_tables.py
    python scripts/generate_latex_tables.py --output paper/tables/
    python scripts/generate_latex_tables.py --format booktabs
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional


def load_all_results(results_dir: Path) -> Dict[str, Dict]:
    """Load results from all model directories (most recent per model)."""
    results = {}

    # Sort by folder name (timestamp) in reverse to get newest first
    for model_dir in sorted(results_dir.glob("gpsbench_*"), reverse=True):
        summary_file = model_dir / "summary.json"
        if not summary_file.exists():
            continue

        with open(summary_file) as f:
            data = json.load(f)

        model_name = data.get('model', model_dir.name)
        # Only keep most recent result per model
        if model_name not in results:
            results[model_name] = data

    return results


def get_short_model_name(model: str) -> str:
    """Get shortened model name for table display."""
    name_map = {
        'gpt-4.1': 'GPT-4.1',
        'gpt-4.1-mini': 'GPT-4.1-mini',
        'gpt-5.1': 'GPT-5.1',
        'gpt-5-mini-2025-08-07': 'GPT-5-mini',
        'gpt-5-nano-2025-08-07': 'GPT-5-nano',
        'gemini-2.5-flash': 'Gemini-2.5-Flash',
        'gemini-2.5-pro': 'Gemini-2.5-Pro',
        'gemini-3-pro-preview': 'Gemini-3-Pro',
        'anthropic/claude-haiku-4.5': 'Claude-Haiku-4.5',
        'mistralai/mistral-large-2512': 'Mistral-Large',
        'mistralai/mistral-small-24b-instruct-2501': 'Mistral-Small-24B',
        'qwen/qwen3-235b-a22b-2507': 'Qwen3-235B',
        'qwen/qwen3-30b-a3b-instruct-2507': 'Qwen3-30B',
    }
    return name_map.get(model, model)


def get_short_task_name(task: str) -> str:
    """Get shortened task name for table display."""
    name_map = {
        'Format Conversion': 'Format Conv.',
        'Coordinate System Transformation': 'Coord. Trans.',
        'Distance Calculation': 'Distance$^*$',
        'Bearing Computation': 'Bearing$^*$',
        'Coordinate Interpolation': 'Interpolation',
        'Polygon Area': 'Area/Perim.$^*$',
        'Bounding Box': 'Bbox',
        'Route Geometry': 'Route Geom.',
        'Relative Position': 'Rel. Position',
        'Place Association': 'Place Assoc.',
        'Name Disambiguation': 'Name Disamb.',
        'Proximity & Nearest Neighbor': 'Proximity',
        'Route Analysis': 'Route Anal.',
        'Boundary Analysis': 'Boundary',
        'Spatial Patterns': 'Spatial Pat.',
        'Missing Data Inference': 'Missing Data',
        'Terrain Classification': 'Terrain',
    }
    return name_map.get(task, task)


# Tasks that use error metrics (100 - MAPE) instead of accuracy
ERROR_METRIC_TASKS = {
    'Distance Calculation',
    'Bearing Computation',
    'Polygon Area',
}


def get_task_score(model_data: Dict, task_name: str, task_acc: float) -> float:
    """Get task score - use 100-MAPE for numerical tasks, accuracy for others."""
    if task_name in ERROR_METRIC_TASKS:
        # Get MAPE from error_metrics
        error_metrics = model_data.get('error_metrics', {})
        task_metrics = error_metrics.get(task_name, {})
        mape = task_metrics.get('metrics', {}).get('mape', None)
        if mape is not None:
            # Cap MAPE at 100 to avoid negative scores
            return max(0, 100 - min(mape, 100))
    return task_acc


def generate_main_results_table(results: Dict[str, Dict], use_booktabs: bool = True) -> str:
    """Generate main results table with overall accuracy per track."""

    # Calculate scores with error metrics for numerical tasks
    model_data = []
    for folder, data in results.items():
        model = data.get('model', folder)
        provider = data.get('provider', 'unknown')

        # Get task-level data and recalculate scores
        pure_tasks = data.get('tracks', {}).get('pure_gps', {}).get('tasks_summary', [])
        applied_tasks = data.get('tracks', {}).get('applied', {}).get('tasks_summary', [])

        pure_scores = []
        for t in pure_tasks:
            score = get_task_score(data, t['task_name'], t['accuracy'])
            pure_scores.append(score)

        applied_scores = []
        for t in applied_tasks:
            score = get_task_score(data, t['task_name'], t['accuracy'])
            applied_scores.append(score)

        pure_gps = sum(pure_scores) / len(pure_scores) if pure_scores else 0
        applied = sum(applied_scores) / len(applied_scores) if applied_scores else 0
        overall = sum(pure_scores + applied_scores) / len(pure_scores + applied_scores) if (pure_scores or applied_scores) else 0

        model_data.append({
            'model': model,
            'provider': provider,
            'overall': overall,
            'pure_gps': pure_gps,
            'applied': applied,
        })

    # Sort by overall accuracy descending
    model_data.sort(key=lambda x: x['overall'], reverse=True)

    # Generate LaTeX
    lines = []

    if use_booktabs:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{GPSBench Results: Overall Accuracy (\%) by Model}")
        lines.append(r"\label{tab:main_results}")
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")
        lines.append(r"Model & Pure GPS & Applied & Overall \\")
        lines.append(r"\midrule")
    else:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{GPSBench Results: Overall Accuracy (\%) by Model}")
        lines.append(r"\label{tab:main_results}")
        lines.append(r"\begin{tabular}{|l|c|c|c|}")
        lines.append(r"\hline")
        lines.append(r"Model & Pure GPS & Applied & Overall \\")
        lines.append(r"\hline")

    for m in model_data:
        short_name = get_short_model_name(m['model'])
        row = f"{short_name} & {m['pure_gps']:.1f} & {m['applied']:.1f} & {m['overall']:.1f} \\\\"
        lines.append(row)

    if use_booktabs:
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    else:
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")

    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_pure_gps_table(results: Dict[str, Dict], use_booktabs: bool = True) -> str:
    """Generate detailed Pure GPS track results table."""

    # Get task names from first result
    task_names = []
    for data in results.values():
        tasks = data.get('tracks', {}).get('pure_gps', {}).get('tasks_summary', [])
        if tasks:
            task_names = [t['task_name'] for t in tasks]
            break

    if not task_names:
        return "% No Pure GPS results found"

    # Collect data
    model_data = []
    for folder, data in results.items():
        model = data.get('model', folder)
        tasks = data.get('tracks', {}).get('pure_gps', {}).get('tasks_summary', [])
        task_acc = {t['task_name']: t['accuracy'] for t in tasks}

        # Convert to scores (use 100-MAPE for numerical tasks)
        task_scores = {}
        for task_name, acc in task_acc.items():
            task_scores[task_name] = get_task_score(data, task_name, acc)

        # Recalculate overall as average of scores
        if task_scores:
            overall = sum(task_scores.values()) / len(task_scores)
        else:
            overall = 0

        model_data.append({
            'model': model,
            'tasks': task_scores,
            'overall': overall,
            'raw_data': data,
        })

    model_data.sort(key=lambda x: x['overall'], reverse=True)

    # Generate LaTeX
    lines = []
    n_tasks = len(task_names)
    col_spec = "l" + "c" * n_tasks + "c"

    if use_booktabs:
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Pure GPS Track: Per-Task Accuracy (\%)}")
        lines.append(r"\label{tab:pure_gps_results}")
        lines.append(r"\small")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
    else:
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Pure GPS Track: Per-Task Accuracy (\%)}")
        lines.append(r"\label{tab:pure_gps_results}")
        lines.append(r"\small")
        lines.append(f"\\begin{{tabular}}{{|{col_spec}|}}")
        lines.append(r"\hline")

    # Header row
    header = "Model"
    for task in task_names:
        header += f" & {get_short_task_name(task)}"
    header += r" & Avg. \\"
    lines.append(header)

    if use_booktabs:
        lines.append(r"\midrule")
    else:
        lines.append(r"\hline")

    # Data rows
    for m in model_data:
        row = get_short_model_name(m['model'])
        for task in task_names:
            acc = m['tasks'].get(task, 0)
            row += f" & {acc:.1f}"
        row += f" & {m['overall']:.1f} \\\\"
        lines.append(row)

    if use_booktabs:
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    else:
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")

    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{$^*$Tasks marked with asterisk use 100$-$MAPE instead of accuracy.}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_applied_table(results: Dict[str, Dict], use_booktabs: bool = True) -> str:
    """Generate detailed Applied track results table."""

    # Get task names from first result
    task_names = []
    for data in results.values():
        tasks = data.get('tracks', {}).get('applied', {}).get('tasks_summary', [])
        if tasks:
            task_names = [t['task_name'] for t in tasks]
            break

    if not task_names:
        return "% No Applied results found"

    # Collect data
    model_data = []
    for folder, data in results.items():
        model = data.get('model', folder)
        tasks = data.get('tracks', {}).get('applied', {}).get('tasks_summary', [])
        task_acc = {t['task_name']: t['accuracy'] for t in tasks}

        # Convert to scores (use 100-MAPE for numerical tasks)
        task_scores = {}
        for task_name, acc in task_acc.items():
            task_scores[task_name] = get_task_score(data, task_name, acc)

        # Recalculate overall as average of scores
        if task_scores:
            overall = sum(task_scores.values()) / len(task_scores)
        else:
            overall = 0

        model_data.append({
            'model': model,
            'tasks': task_scores,
            'overall': overall,
            'raw_data': data,
        })

    model_data.sort(key=lambda x: x['overall'], reverse=True)

    # Generate LaTeX
    lines = []
    n_tasks = len(task_names)
    col_spec = "l" + "c" * n_tasks + "c"

    if use_booktabs:
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Applied Track: Per-Task Accuracy (\%)}")
        lines.append(r"\label{tab:applied_results}")
        lines.append(r"\small")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
    else:
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Applied Track: Per-Task Accuracy (\%)}")
        lines.append(r"\label{tab:applied_results}")
        lines.append(r"\small")
        lines.append(f"\\begin{{tabular}}{{|{col_spec}|}}")
        lines.append(r"\hline")

    # Header row
    header = "Model"
    for task in task_names:
        header += f" & {get_short_task_name(task)}"
    header += r" & Avg. \\"
    lines.append(header)

    if use_booktabs:
        lines.append(r"\midrule")
    else:
        lines.append(r"\hline")

    # Data rows
    for m in model_data:
        row = get_short_model_name(m['model'])
        for task in task_names:
            acc = m['tasks'].get(task, 0)
            row += f" & {acc:.1f}"
        row += f" & {m['overall']:.1f} \\\\"
        lines.append(row)

    if use_booktabs:
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    else:
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")

    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_combined_table(results: Dict[str, Dict], use_booktabs: bool = True) -> str:
    """Generate a combined table with all tasks (rotated headers for compactness)."""

    # Get all task names
    pure_gps_tasks = []
    applied_tasks = []

    for data in results.values():
        pure = data.get('tracks', {}).get('pure_gps', {}).get('tasks_summary', [])
        applied = data.get('tracks', {}).get('applied', {}).get('tasks_summary', [])
        if pure:
            pure_gps_tasks = [t['task_name'] for t in pure]
        if applied:
            applied_tasks = [t['task_name'] for t in applied]
        if pure_gps_tasks and applied_tasks:
            break

    all_tasks = pure_gps_tasks + applied_tasks

    # Collect data
    model_data = []
    for folder, data in results.items():
        model = data.get('model', folder)

        pure_tasks = data.get('tracks', {}).get('pure_gps', {}).get('tasks_summary', [])
        applied_tasks_data = data.get('tracks', {}).get('applied', {}).get('tasks_summary', [])

        task_acc = {}
        for t in pure_tasks:
            task_acc[t['task_name']] = t['accuracy']
        for t in applied_tasks_data:
            task_acc[t['task_name']] = t['accuracy']

        # Convert to scores (use 100-MAPE for numerical tasks)
        task_scores = {}
        for task_name, acc in task_acc.items():
            task_scores[task_name] = get_task_score(data, task_name, acc)

        # Recalculate overall as average of scores
        if task_scores:
            overall = sum(task_scores.values()) / len(task_scores)
        else:
            overall = 0

        model_data.append({
            'model': model,
            'tasks': task_scores,
            'overall': overall,
            'raw_data': data,
        })

    model_data.sort(key=lambda x: x['overall'], reverse=True)

    # Generate LaTeX with rotated headers
    lines = []
    n_tasks = len(all_tasks)
    col_spec = "l" + "c" * n_tasks + "c"

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{GPSBench Complete Results: Per-Task Accuracy (\%)}")
    lines.append(r"\label{tab:complete_results}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")

    if use_booktabs:
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
    else:
        lines.append(f"\\begin{{tabular}}{{|{col_spec}|}}")
        lines.append(r"\hline")

    # Header row with rotated text
    header = "Model"
    for task in all_tasks:
        short = get_short_task_name(task)
        header += f" & \\rotatebox{{90}}{{{short}}}"
    header += r" & \rotatebox{90}{Overall} \\"
    lines.append(header)

    if use_booktabs:
        lines.append(r"\midrule")
    else:
        lines.append(r"\hline")

    # Data rows
    for m in model_data:
        row = get_short_model_name(m['model'])
        for task in all_tasks:
            acc = m['tasks'].get(task, 0)
            # Bold the best score in each column
            row += f" & {acc:.1f}"
        row += f" & \\textbf{{{m['overall']:.1f}}} \\\\"
        lines.append(row)

    if use_booktabs:
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    else:
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")

    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from GPSBench results")
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing result folders')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for LaTeX files (default: print to stdout)')
    parser.add_argument('--format', type=str, choices=['booktabs', 'basic'], default='booktabs',
                        help='Table format style')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    use_booktabs = args.format == 'booktabs'

    print(f"Loading results from {results_dir}...")
    results = load_all_results(results_dir)
    print(f"Found {len(results)} model results\n")

    # Generate tables
    tables = {
        'main_results.tex': generate_main_results_table(results, use_booktabs),
        'pure_gps_results.tex': generate_pure_gps_table(results, use_booktabs),
        'applied_results.tex': generate_applied_table(results, use_booktabs),
        'complete_results.tex': generate_combined_table(results, use_booktabs),
    }

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for filename, content in tables.items():
            output_path = output_dir / filename
            with open(output_path, 'w') as f:
                f.write(content)
            print(f"Saved: {output_path}")
    else:
        # Print to stdout
        for filename, content in tables.items():
            print(f"\n{'='*60}")
            print(f"% {filename}")
            print('='*60)
            print(content)


if __name__ == "__main__":
    main()
