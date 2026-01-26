#!/usr/bin/env python3
"""
Analyze and visualize GPSBench evaluation results
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import pandas as pd
from tabulate import tabulate


class ResultsAnalyzer:
    """Analyzer for GPSBench evaluation results"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def load_result(self, filename: str) -> Dict[str, Any]:
        """Load a single result file"""
        filepath = self.results_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_all_results(self) -> List[Dict[str, Any]]:
        """Load all result files from results directory"""
        results = []
        for filepath in self.results_dir.glob("eval_*.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
        return results

    def generate_summary_table(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate summary table comparing models"""
        data = []

        for result in results:
            row = {
                'Model': result.get('model', 'Unknown'),
                'Provider': result.get('provider', 'Unknown'),
                'Timestamp': result.get('timestamp', 'Unknown')
            }

            # Add overall accuracy if available
            if 'overall_metrics' in result:
                row['Overall'] = f"{result['overall_metrics']['overall_accuracy']:.2f}%"

            # Add task-specific accuracies
            if 'tasks' in result:
                for task_name, task_result in result['tasks'].items():
                    if 'metrics' in task_result:
                        short_name = self._shorten_task_name(task_name)
                        row[short_name] = f"{task_result['metrics']['accuracy']:.2f}%"

            # Add token usage if available
            total_tokens = sum(
                task_result['metrics'].get('total_tokens', 0)
                for task_result in result.get('tasks', {}).values()
                if 'metrics' in task_result
            )
            if total_tokens > 0:
                row['Tokens'] = f"{total_tokens:,}"

            data.append(row)

        return pd.DataFrame(data)

    def _shorten_task_name(self, task_name: str) -> str:
        """Shorten task name for display"""
        mapping = {
            'format_recognition': 'Format Recog',
            'format_conversion': 'Format Conv',
            'validation': 'Validation',
            'coordinate_system_recognition': 'Coord System',
            'precision_understanding': 'Precision'
        }
        return mapping.get(task_name, task_name)

    def analyze_single_result(self, result: Dict[str, Any]):
        """Print detailed analysis of a single result"""
        print(f"\n{'='*80}")
        print(f"Model: {result.get('model', 'Unknown')}")
        print(f"Provider: {result.get('provider', 'Unknown')}")
        print(f"Timestamp: {result.get('timestamp', 'Unknown')}")
        print(f"{'='*80}")

        if 'overall_metrics' in result:
            metrics = result['overall_metrics']
            print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.2f}%")
            print(f"Total Correct: {metrics['total_correct']}/{metrics['total_cases']}")

        print(f"\n{'─'*80}")
        print("Task-by-Task Breakdown:")
        print(f"{'─'*80}")

        if 'tasks' in result:
            for task_name, task_result in result['tasks'].items():
                if 'metrics' in task_result:
                    metrics = task_result['metrics']
                    print(f"\n{task_name}:")
                    print(f"  Accuracy:      {metrics['accuracy']:.2f}%")
                    print(f"  Correct:       {metrics['correct_count']}/{metrics['total_count']}")
                    print(f"  Errors:        {metrics.get('error_count', 0)}")
                    print(f"  Avg Latency:   {metrics.get('avg_latency_ms', 0):.0f}ms")
                    print(f"  Total Tokens:  {metrics.get('total_tokens', 0):,}")

        # Error analysis
        print(f"\n{'─'*80}")
        print("Error Analysis:")
        print(f"{'─'*80}")

        for task_name, task_result in result.get('tasks', {}).items():
            if 'test_results' in task_result:
                failures = [
                    tr for tr in task_result['test_results']
                    if not tr.get('correct', False)
                ]

                if failures:
                    print(f"\n{task_name}: {len(failures)} failures")
                    # Show first few failures
                    for i, failure in enumerate(failures[:3]):
                        print(f"  [{i+1}] Input: {failure.get('input', 'N/A')[:50]}")
                        if 'evaluation' in failure:
                            eval_result = failure['evaluation']
                            if 'expected' in eval_result:
                                print(f"      Expected: {eval_result.get('expected', 'N/A')}")
                            if 'predicted' in eval_result:
                                print(f"      Predicted: {eval_result.get('predicted', 'N/A')}")

    def generate_comparison_report(self, results: List[Dict[str, Any]]):
        """Generate comparison report for multiple models"""
        print(f"\n{'='*80}")
        print("GPSBench Layer A - Model Comparison")
        print(f"{'='*80}")

        # Generate summary table
        df = self.generate_summary_table(results)

        # Print table
        print("\n" + tabulate(
            df,
            headers='keys',
            tablefmt='grid',
            showindex=False
        ))

        # Find best model per task
        print(f"\n{'─'*80}")
        print("Best Models by Task:")
        print(f"{'─'*80}")

        task_best = defaultdict(lambda: {'model': None, 'accuracy': 0})

        for result in results:
            model = result.get('model', 'Unknown')
            for task_name, task_result in result.get('tasks', {}).items():
                if 'metrics' in task_result:
                    accuracy = task_result['metrics']['accuracy']
                    if accuracy > task_best[task_name]['accuracy']:
                        task_best[task_name] = {
                            'model': model,
                            'accuracy': accuracy
                        }

        for task_name, best in task_best.items():
            print(f"{self._shorten_task_name(task_name):20s}: {best['model']:30s} ({best['accuracy']:.2f}%)")

        # Overall best model
        print(f"\n{'─'*80}")
        print("Overall Best Model:")
        print(f"{'─'*80}")

        best_overall = max(
            results,
            key=lambda r: r.get('overall_metrics', {}).get('overall_accuracy', 0)
        )
        print(f"Model: {best_overall.get('model', 'Unknown')}")
        print(f"Overall Accuracy: {best_overall.get('overall_metrics', {}).get('overall_accuracy', 0):.2f}%")

    def export_to_csv(self, results: List[Dict[str, Any]], output_file: str = "results_summary.csv"):
        """Export results to CSV"""
        df = self.generate_summary_table(results)
        output_path = self.results_dir / output_file
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze GPSBench evaluation results'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Results directory (default: results)'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Specific result file to analyze'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all results in directory'
    )
    parser.add_argument(
        '--export-csv',
        type=str,
        default=None,
        help='Export comparison to CSV file'
    )

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(results_dir=args.results_dir)

    if args.file:
        # Analyze single file
        result = analyzer.load_result(args.file)
        analyzer.analyze_single_result(result)

    elif args.compare or args.export_csv:
        # Compare all results
        results = analyzer.load_all_results()

        if not results:
            print(f"No result files found in {args.results_dir}")
            return

        print(f"Found {len(results)} result files")
        analyzer.generate_comparison_report(results)

        if args.export_csv:
            analyzer.export_to_csv(results, args.export_csv)

    else:
        # Show latest result
        result_files = list(Path(args.results_dir).glob("eval_*.json"))
        if not result_files:
            print(f"No result files found in {args.results_dir}")
            return

        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
        print(f"Analyzing latest result: {latest_file.name}")

        result = analyzer.load_result(latest_file.name)
        analyzer.analyze_single_result(result)


if __name__ == "__main__":
    main()
