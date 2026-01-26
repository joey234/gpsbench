#!/usr/bin/env python3
"""
GPSBench Results Analysis and Comparison Tool

This script analyzes and compares evaluation results across different models.
It loads all result files from the results directory and generates comprehensive
comparison reports.

Features:
- Overall model comparison
- Track-level comparison (Pure GPS vs Applied)
- Task-level comparison
- Error metrics analysis (MAE, MAPE for numerical tasks)
- Place Association granularity analysis (Continent → Country → Region → City)
- Geographic region breakdown (Europe, Asia, Africa, Americas, Oceania)
"""

import json
import os
import sys
import math
import re
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import argparse

try:
    import numpy as np
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Tasks to exclude from analysis (e.g., tasks being regenerated)
# Include both task IDs and display names for matching
EXCLUDED_TASKS = {
    'area_perimeter',   # Task ID
    'Polygon Area',     # Display name in error_metrics
}

# Country code to continent mapping
COUNTRY_TO_CONTINENT = {
    # Europe
    'AL': 'Europe', 'AD': 'Europe', 'AT': 'Europe', 'BY': 'Europe', 'BE': 'Europe',
    'BA': 'Europe', 'BG': 'Europe', 'HR': 'Europe', 'CY': 'Europe', 'CZ': 'Europe',
    'DK': 'Europe', 'EE': 'Europe', 'FI': 'Europe', 'FR': 'Europe', 'DE': 'Europe',
    'GR': 'Europe', 'HU': 'Europe', 'IS': 'Europe', 'IE': 'Europe', 'IT': 'Europe',
    'XK': 'Europe', 'LV': 'Europe', 'LI': 'Europe', 'LT': 'Europe', 'LU': 'Europe',
    'MT': 'Europe', 'MD': 'Europe', 'MC': 'Europe', 'ME': 'Europe', 'NL': 'Europe',
    'MK': 'Europe', 'NO': 'Europe', 'PL': 'Europe', 'PT': 'Europe', 'RO': 'Europe',
    'RU': 'Europe', 'SM': 'Europe', 'RS': 'Europe', 'SK': 'Europe', 'SI': 'Europe',
    'ES': 'Europe', 'SE': 'Europe', 'CH': 'Europe', 'UA': 'Europe', 'GB': 'Europe',
    # Asia
    'AF': 'Asia', 'AM': 'Asia', 'AZ': 'Asia', 'BH': 'Asia', 'BD': 'Asia',
    'BT': 'Asia', 'BN': 'Asia', 'KH': 'Asia', 'CN': 'Asia', 'GE': 'Asia',
    'HK': 'Asia', 'IN': 'Asia', 'ID': 'Asia', 'IR': 'Asia', 'IQ': 'Asia',
    'IL': 'Asia', 'JP': 'Asia', 'JO': 'Asia', 'KZ': 'Asia', 'KW': 'Asia',
    'KG': 'Asia', 'LA': 'Asia', 'LB': 'Asia', 'MO': 'Asia', 'MY': 'Asia',
    'MV': 'Asia', 'MN': 'Asia', 'MM': 'Asia', 'NP': 'Asia', 'KP': 'Asia',
    'OM': 'Asia', 'PK': 'Asia', 'PS': 'Asia', 'PH': 'Asia', 'QA': 'Asia',
    'SA': 'Asia', 'SG': 'Asia', 'KR': 'Asia', 'LK': 'Asia', 'SY': 'Asia',
    'TW': 'Asia', 'TJ': 'Asia', 'TH': 'Asia', 'TL': 'Asia', 'TR': 'Asia',
    'TM': 'Asia', 'AE': 'Asia', 'UZ': 'Asia', 'VN': 'Asia', 'YE': 'Asia',
    # Africa
    'DZ': 'Africa', 'AO': 'Africa', 'BJ': 'Africa', 'BW': 'Africa', 'BF': 'Africa',
    'BI': 'Africa', 'CV': 'Africa', 'CM': 'Africa', 'CF': 'Africa', 'TD': 'Africa',
    'KM': 'Africa', 'CG': 'Africa', 'CD': 'Africa', 'CI': 'Africa', 'DJ': 'Africa',
    'EG': 'Africa', 'GQ': 'Africa', 'ER': 'Africa', 'SZ': 'Africa', 'ET': 'Africa',
    'GA': 'Africa', 'GM': 'Africa', 'GH': 'Africa', 'GN': 'Africa', 'GW': 'Africa',
    'KE': 'Africa', 'LS': 'Africa', 'LR': 'Africa', 'LY': 'Africa', 'MG': 'Africa',
    'MW': 'Africa', 'ML': 'Africa', 'MR': 'Africa', 'MU': 'Africa', 'MA': 'Africa',
    'MZ': 'Africa', 'NA': 'Africa', 'NE': 'Africa', 'NG': 'Africa', 'RW': 'Africa',
    'ST': 'Africa', 'SN': 'Africa', 'SC': 'Africa', 'SL': 'Africa', 'SO': 'Africa',
    'ZA': 'Africa', 'SS': 'Africa', 'SD': 'Africa', 'TZ': 'Africa', 'TG': 'Africa',
    'TN': 'Africa', 'UG': 'Africa', 'ZM': 'Africa', 'ZW': 'Africa', 'RE': 'Africa',
    # North America
    'AG': 'N. America', 'BS': 'N. America', 'BB': 'N. America', 'BZ': 'N. America',
    'CA': 'N. America', 'CR': 'N. America', 'CU': 'N. America', 'DM': 'N. America',
    'DO': 'N. America', 'SV': 'N. America', 'GD': 'N. America', 'GT': 'N. America',
    'HT': 'N. America', 'HN': 'N. America', 'JM': 'N. America', 'MX': 'N. America',
    'NI': 'N. America', 'PA': 'N. America', 'KN': 'N. America', 'LC': 'N. America',
    'VC': 'N. America', 'TT': 'N. America', 'US': 'N. America', 'AI': 'N. America',
    'GP': 'N. America', 'GU': 'N. America', 'PR': 'N. America', 'VI': 'N. America',
    # South America
    'AR': 'S. America', 'BO': 'S. America', 'BR': 'S. America', 'CL': 'S. America',
    'CO': 'S. America', 'EC': 'S. America', 'GY': 'S. America', 'PY': 'S. America',
    'PE': 'S. America', 'SR': 'S. America', 'UY': 'S. America', 'VE': 'S. America',
    # Oceania
    'AU': 'Oceania', 'FJ': 'Oceania', 'KI': 'Oceania', 'MH': 'Oceania', 'FM': 'Oceania',
    'NR': 'Oceania', 'NZ': 'Oceania', 'PW': 'Oceania', 'PG': 'Oceania', 'WS': 'Oceania',
    'SB': 'Oceania', 'TO': 'Oceania', 'TV': 'Oceania', 'VU': 'Oceania', 'NF': 'Oceania',
}

# Country code to full name mapping
COUNTRY_NAMES = {
    'AE': 'United Arab Emirates', 'AF': 'Afghanistan', 'AI': 'Anguilla',
    'AL': 'Albania', 'AM': 'Armenia', 'AO': 'Angola', 'AR': 'Argentina',
    'AT': 'Austria', 'AU': 'Australia', 'AZ': 'Azerbaijan', 'BA': 'Bosnia',
    'BD': 'Bangladesh', 'BE': 'Belgium', 'BF': 'Burkina Faso', 'BG': 'Bulgaria',
    'BJ': 'Benin', 'BO': 'Bolivia', 'BR': 'Brazil', 'BT': 'Bhutan',
    'BW': 'Botswana', 'BY': 'Belarus', 'CA': 'Canada', 'CD': 'Congo',
    'CF': 'Central African Republic', 'CG': 'Congo', 'CH': 'Switzerland',
    'CI': 'Ivory Coast', 'CL': 'Chile', 'CM': 'Cameroon', 'CN': 'China',
    'CO': 'Colombia', 'CR': 'Costa Rica', 'CU': 'Cuba', 'CY': 'Cyprus',
    'CZ': 'Czech', 'DE': 'Germany', 'DK': 'Denmark', 'DO': 'Dominican',
    'DZ': 'Algeria', 'EC': 'Ecuador', 'EG': 'Egypt', 'ES': 'Spain',
    'ET': 'Ethiopia', 'FI': 'Finland', 'FR': 'France', 'GB': 'United Kingdom',
    'GE': 'Georgia', 'GH': 'Ghana', 'GM': 'Gambia', 'GN': 'Guinea',
    'GP': 'Guadeloupe', 'GR': 'Greece', 'GT': 'Guatemala', 'GU': 'Guam',
    'GY': 'Guyana', 'HK': 'Hong Kong', 'HN': 'Honduras', 'HR': 'Croatia',
    'HT': 'Haiti', 'HU': 'Hungary', 'ID': 'Indonesia', 'IE': 'Ireland',
    'IL': 'Israel', 'IN': 'India', 'IQ': 'Iraq', 'IR': 'Iran', 'IS': 'Iceland',
    'IT': 'Italy', 'JO': 'Jordan', 'JP': 'Japan', 'KE': 'Kenya',
    'KG': 'Kyrgyzstan', 'KH': 'Cambodia', 'KP': 'North Korea', 'KR': 'South Korea',
    'KW': 'Kuwait', 'KZ': 'Kazakhstan', 'LB': 'Lebanon', 'LK': 'Sri Lanka',
    'LR': 'Liberia', 'LS': 'Lesotho', 'LT': 'Lithuania', 'LV': 'Latvia',
    'LY': 'Libya', 'MA': 'Morocco', 'ME': 'Montenegro', 'MG': 'Madagascar',
    'MK': 'North Macedonia', 'ML': 'Mali', 'MM': 'Myanmar', 'MN': 'Mongolia',
    'MO': 'Macau', 'MR': 'Mauritania', 'MT': 'Malta', 'MU': 'Mauritius',
    'MW': 'Malawi', 'MX': 'Mexico', 'MY': 'Malaysia', 'MZ': 'Mozambique',
    'NA': 'Namibia', 'NE': 'Niger', 'NF': 'Norfolk Island', 'NG': 'Nigeria',
    'NI': 'Nicaragua', 'NL': 'Netherlands', 'NO': 'Norway', 'NP': 'Nepal',
    'NZ': 'New Zealand', 'OM': 'Oman', 'PA': 'Panama', 'PE': 'Peru',
    'PG': 'Papua New Guinea', 'PH': 'Philippines', 'PK': 'Pakistan', 'PL': 'Poland',
    'PS': 'Palestine', 'PT': 'Portugal', 'PY': 'Paraguay', 'QA': 'Qatar',
    'RE': 'Reunion', 'RO': 'Romania', 'RS': 'Serbia', 'RU': 'Russia',
    'RW': 'Rwanda', 'SA': 'Saudi Arabia', 'SD': 'Sudan', 'SE': 'Sweden',
    'SG': 'Singapore', 'SI': 'Slovenia', 'SK': 'Slovakia', 'SN': 'Senegal',
    'SO': 'Somalia', 'SS': 'South Sudan', 'SV': 'El Salvador', 'SY': 'Syria',
    'TD': 'Chad', 'TG': 'Togo', 'TH': 'Thailand', 'TJ': 'Tajikistan',
    'TL': 'East Timor', 'TM': 'Turkmenistan', 'TN': 'Tunisia', 'TR': 'Turkey',
    'TT': 'Trinidad', 'TV': 'Tuvalu', 'TW': 'Taiwan', 'TZ': 'Tanzania',
    'UA': 'Ukraine', 'UG': 'Uganda', 'US': 'United States', 'UY': 'Uruguay',
    'UZ': 'Uzbekistan', 'VE': 'Venezuela', 'VN': 'Vietnam', 'XK': 'Kosovo',
    'YE': 'Yemen', 'ZA': 'South Africa', 'ZM': 'Zambia', 'ZW': 'Zimbabwe',
}

# Continent aliases for matching country names in responses
CONTINENT_ALIASES = {
    'Asia': ['Asia', 'China', 'Japan', 'India', 'Indonesia', 'Philippines', 'Vietnam',
             'Thailand', 'Malaysia', 'Singapore', 'Korea', 'Taiwan', 'Hong Kong',
             'Pakistan', 'Bangladesh', 'Myanmar', 'Cambodia', 'Iran', 'Iraq', 'Saudi',
             'Turkey', 'Israel', 'UAE', 'Emirates', 'Qatar', 'Kuwait', 'Jordan',
             'Lebanon', 'Syria', 'Yemen', 'Oman', 'Kazakhstan', 'Uzbekistan'],
    'Europe': ['Europe', 'France', 'Germany', 'Italy', 'Spain', 'UK', 'Britain',
               'England', 'Netherlands', 'Belgium', 'Switzerland', 'Austria',
               'Portugal', 'Greece', 'Poland', 'Sweden', 'Norway', 'Denmark',
               'Finland', 'Ireland', 'Czech', 'Hungary', 'Romania', 'Bulgaria',
               'Croatia', 'Serbia', 'Slovenia', 'Slovakia', 'Ukraine', 'Russia'],
    'Africa': ['Africa', 'Nigeria', 'Egypt', 'South Africa', 'Kenya', 'Ethiopia',
               'Tanzania', 'Ghana', 'Morocco', 'Algeria', 'Tunisia', 'Libya',
               'Sudan', 'Uganda', 'Cameroon', 'Senegal', 'Mali', 'Niger', 'Chad'],
    'N. America': ['America', 'United States', 'USA', 'Canada', 'Mexico', 'Cuba',
                   'Dominican', 'Haiti', 'Jamaica', 'Panama', 'Costa Rica',
                   'Guatemala', 'Honduras', 'Nicaragua'],
    'S. America': ['Brazil', 'Argentina', 'Colombia', 'Peru', 'Venezuela', 'Chile',
                   'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay'],
    'Oceania': ['Australia', 'New Zealand', 'Papua', 'Fiji', 'Oceania'],
}


class ResultsAnalyzer:
    """Analyzer for GPSBench evaluation results"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results: List[Dict[str, Any]] = []

    def load_all_results(self) -> None:
        """Load all result files from the results directory"""
        if not self.results_dir.exists():
            print(f"Error: Results directory '{self.results_dir}' not found")
            sys.exit(1)

        # Look for both .json files and directories with summary.json
        for item in self.results_dir.iterdir():
            if item.name.startswith('.') or item.name == 'README.md':
                continue

            result_data = None
            source_path = None

            # Check if it's a directory with summary.json
            if item.is_dir():
                summary_file = item / "summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            result_data = json.load(f)
                        source_path = str(summary_file)
                    except Exception as e:
                        print(f"Warning: Failed to load {summary_file}: {e}")
                        continue
            # Check if it's a .json file
            elif item.suffix == '.json':
                try:
                    with open(item, 'r') as f:
                        result_data = json.load(f)
                    source_path = str(item)
                except Exception as e:
                    print(f"Warning: Failed to load {item}: {e}")
                    continue

            if result_data:
                result_data['_source_path'] = source_path
                result_data['_source_name'] = item.name
                self.results.append(result_data)

        print(f"Loaded {len(self.results)} result files")

    def get_model_name(self, result: Dict[str, Any]) -> str:
        """Extract a clean model name from result data"""
        model = result.get('model', 'unknown')
        # Clean up model name
        if '/' in model:
            model = model.split('/')[-1]
        return model

    def generate_overall_comparison(self) -> str:
        """Generate overall accuracy comparison across models"""
        lines = []
        lines.append("=" * 80)
        lines.append("OVERALL MODEL COMPARISON")
        lines.append("=" * 80)
        lines.append("")

        # Sort by overall accuracy
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get('overall', {}).get('overall_accuracy', 0),
            reverse=True
        )

        lines.append(f"{'Rank':<6} {'Model':<40} {'Accuracy':<12} {'Correct/Total':<20}")
        lines.append("-" * 80)

        for rank, result in enumerate(sorted_results, 1):
            model = self.get_model_name(result)
            overall = result.get('overall', {})
            accuracy = overall.get('overall_accuracy', 0)
            correct = overall.get('total_correct', 0)
            total = overall.get('total_cases', 0)

            lines.append(
                f"{rank:<6} {model:<40} {accuracy:>10.2f}% {correct:>8}/{total:<10}"
            )

        lines.append("")
        return "\n".join(lines)

    def generate_track_comparison(self) -> str:
        """Generate track-level comparison across models"""
        lines = []
        lines.append("=" * 80)
        lines.append("TRACK-LEVEL COMPARISON")
        lines.append("=" * 80)
        lines.append("")

        # Get all unique tracks
        tracks = set()
        for result in self.results:
            if 'tracks' in result:
                tracks.update(result['tracks'].keys())

        for track in sorted(tracks):
            lines.append(f"\n{track.upper()} TRACK")
            lines.append("-" * 80)
            lines.append(f"{'Model':<40} {'Accuracy':<12} {'Correct/Total':<20}")
            lines.append("-" * 80)

            # Collect and sort results for this track
            track_results = []
            for result in self.results:
                if 'tracks' in result and track in result['tracks']:
                    track_data = result['tracks'][track]
                    track_results.append({
                        'model': self.get_model_name(result),
                        'accuracy': track_data.get('overall_accuracy', 0),
                        'correct': track_data.get('total_correct', 0),
                        'total': track_data.get('total_cases', 0)
                    })

            track_results.sort(key=lambda x: x['accuracy'], reverse=True)

            for tr in track_results:
                lines.append(
                    f"{tr['model']:<40} {tr['accuracy']:>10.2f}% "
                    f"{tr['correct']:>8}/{tr['total']:<10}"
                )

        lines.append("")
        return "\n".join(lines)

    def generate_task_comparison(self, track: Optional[str] = None) -> str:
        """Generate task-level comparison across models"""
        lines = []
        lines.append("=" * 80)
        lines.append("TASK-LEVEL COMPARISON")
        lines.append("=" * 80)
        lines.append("")

        # Collect all tasks organized by track
        tasks_by_track = defaultdict(set)
        for result in self.results:
            if 'tracks' in result:
                for track_name, track_data in result['tracks'].items():
                    if track and track_name != track:
                        continue
                    # Handle both 'tasks_summary' and 'tasks' keys
                    tasks_list = track_data.get('tasks_summary') or track_data.get('tasks', [])
                    for task in tasks_list:
                        # Skip excluded tasks
                        if task['task_name'] in EXCLUDED_TASKS:
                            continue
                        tasks_by_track[track_name].add(task['task_name'])

        # For each track and task, compare models
        for track_name in sorted(tasks_by_track.keys()):
            lines.append(f"\n{track_name.upper()} TRACK TASKS")
            lines.append("=" * 80)

            for task_name in sorted(tasks_by_track[track_name]):
                lines.append(f"\n{task_name}")
                lines.append("-" * 80)
                lines.append(f"{'Model':<40} {'Accuracy':<12} {'Correct/Total':<20}")
                lines.append("-" * 80)

                # Collect results for this task
                task_results = []
                for result in self.results:
                    if 'tracks' in result and track_name in result['tracks']:
                        track_data = result['tracks'][track_name]
                        # Handle both 'tasks_summary' and 'tasks' keys
                        tasks_list = track_data.get('tasks_summary') or track_data.get('tasks', [])
                        for task in tasks_list:
                            if task['task_name'] == task_name:
                                task_results.append({
                                    'model': self.get_model_name(result),
                                    'accuracy': task.get('accuracy', 0),
                                    'correct': task.get('correct', 0),
                                    'total': task.get('total', 0)
                                })
                                break

                task_results.sort(key=lambda x: x['accuracy'], reverse=True)

                for tr in task_results:
                    lines.append(
                        f"{tr['model']:<40} {tr['accuracy']:>10.2f}% "
                        f"{tr['correct']:>8}/{tr['total']:<10}"
                    )

        lines.append("")
        return "\n".join(lines)

    def generate_csv_export(self, output_file: str) -> None:
        """Export results to CSV format"""
        import csv

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Model', 'Track', 'Task', 'Accuracy', 'Correct', 'Total', 'Timestamp'
            ])

            # Data rows
            for result in self.results:
                model = self.get_model_name(result)
                timestamp = result.get('timestamp', '')

                # Overall row
                overall = result.get('overall', {})
                writer.writerow([
                    model, 'OVERALL', 'ALL',
                    f"{overall.get('overall_accuracy', 0):.2f}",
                    overall.get('total_correct', 0),
                    overall.get('total_cases', 0),
                    timestamp
                ])

                # Track and task rows
                if 'tracks' in result:
                    for track_name, track_data in result['tracks'].items():
                        # Track row
                        writer.writerow([
                            model, track_name.upper(), 'ALL',
                            f"{track_data.get('overall_accuracy', 0):.2f}",
                            track_data.get('total_correct', 0),
                            track_data.get('total_cases', 0),
                            timestamp
                        ])

                        # Task rows - handle both 'tasks_summary' and 'tasks' keys
                        tasks_list = track_data.get('tasks_summary') or track_data.get('tasks', [])
                        for task in tasks_list:
                            # Skip excluded tasks
                            if task['task_name'] in EXCLUDED_TASKS:
                                continue
                            writer.writerow([
                                model, track_name.upper(), task['task_name'],
                                f"{task.get('accuracy', 0):.2f}",
                                task.get('correct', 0),
                                task.get('total', 0),
                                timestamp
                            ])

        print(f"CSV export saved to: {output_file}")

    def generate_markdown_table(self) -> str:
        """Generate markdown table for overall comparison"""
        lines = []
        lines.append("# GPSBench Model Comparison")
        lines.append("")
        lines.append("## Overall Results")
        lines.append("")

        # Sort by overall accuracy
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get('overall', {}).get('overall_accuracy', 0),
            reverse=True
        )

        lines.append("| Rank | Model | Overall Accuracy | Pure GPS | Applied | Correct/Total |")
        lines.append("|------|-------|------------------|----------|---------|---------------|")

        for rank, result in enumerate(sorted_results, 1):
            model = self.get_model_name(result)
            overall = result.get('overall', {})
            accuracy = overall.get('overall_accuracy', 0)
            correct = overall.get('total_correct', 0)
            total = overall.get('total_cases', 0)

            # Get track accuracies
            pure_gps_acc = 0
            applied_acc = 0
            if 'tracks' in result:
                if 'pure_gps' in result['tracks']:
                    pure_gps_acc = result['tracks']['pure_gps'].get('overall_accuracy', 0)
                if 'applied' in result['tracks']:
                    applied_acc = result['tracks']['applied'].get('overall_accuracy', 0)

            lines.append(
                f"| {rank} | {model} | {accuracy:.2f}% | {pure_gps_acc:.2f}% | "
                f"{applied_acc:.2f}% | {correct}/{total} |"
            )

        lines.append("")
        return "\n".join(lines)

    def find_strengths_weaknesses(self) -> str:
        """Identify strengths and weaknesses for each model"""
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL STRENGTHS & WEAKNESSES")
        lines.append("=" * 80)
        lines.append("")

        for result in self.results:
            model = self.get_model_name(result)
            lines.append(f"\n{model}")
            lines.append("-" * 80)

            # Collect all task performances
            task_performances = []
            if 'tracks' in result:
                for track_name, track_data in result['tracks'].items():
                    # Handle both 'tasks_summary' and 'tasks' keys
                    tasks_list = track_data.get('tasks_summary') or track_data.get('tasks', [])
                    for task in tasks_list:
                        # Skip excluded tasks
                        if task['task_name'] in EXCLUDED_TASKS:
                            continue
                        task_performances.append({
                            'track': track_name,
                            'task': task['task_name'],
                            'accuracy': task.get('accuracy', 0)
                        })

            if task_performances:
                # Sort by accuracy
                task_performances.sort(key=lambda x: x['accuracy'], reverse=True)

                # Top 3 strengths
                lines.append("\nTop Strengths:")
                for i, perf in enumerate(task_performances[:3], 1):
                    lines.append(
                        f"  {i}. {perf['task']} ({perf['track']}): {perf['accuracy']:.2f}%"
                    )

                # Bottom 3 weaknesses
                lines.append("\nTop Weaknesses:")
                for i, perf in enumerate(task_performances[-3:], 1):
                    lines.append(
                        f"  {i}. {perf['task']} ({perf['track']}): {perf['accuracy']:.2f}%"
                    )

        lines.append("")
        return "\n".join(lines)

    def generate_summary_stats(self) -> str:
        """Generate summary statistics"""
        lines = []
        lines.append("=" * 80)
        lines.append("SUMMARY STATISTICS")
        lines.append("=" * 80)
        lines.append("")

        if not self.results:
            lines.append("No results found.")
            return "\n".join(lines)

        # Overall accuracy stats
        accuracies = [r.get('overall', {}).get('overall_accuracy', 0) for r in self.results]
        lines.append(f"Number of models evaluated: {len(self.results)}")
        lines.append(f"Best overall accuracy: {max(accuracies):.2f}%")
        lines.append(f"Worst overall accuracy: {min(accuracies):.2f}%")
        lines.append(f"Average overall accuracy: {sum(accuracies)/len(accuracies):.2f}%")
        lines.append("")

        # Track stats
        for track in ['pure_gps', 'applied']:
            track_accs = []
            for result in self.results:
                if 'tracks' in result and track in result['tracks']:
                    track_accs.append(result['tracks'][track].get('overall_accuracy', 0))

            if track_accs:
                lines.append(f"{track.upper()} Track:")
                lines.append(f"  Best: {max(track_accs):.2f}%")
                lines.append(f"  Worst: {min(track_accs):.2f}%")
                lines.append(f"  Average: {sum(track_accs)/len(track_accs):.2f}%")
                lines.append("")

        return "\n".join(lines)

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def analyze_error_metrics(self) -> str:
        """Analyze error metrics (distance, bearing, area, etc.) across models"""
        lines = []
        lines.append("=" * 90)
        lines.append("ERROR METRICS COMPARISON")
        lines.append("Using MAE (Mean Absolute Error) and MAPE instead of binary accuracy")
        lines.append("=" * 90)
        lines.append("")

        # Collect all tasks with error metrics across all results
        # New format: error_metrics -> {task_name: {metrics: {...}, unit: ..., type: ...}}
        tasks = set()
        for result in self.results:
            if 'error_metrics' in result:
                for task in result['error_metrics'].keys():
                    # Skip excluded tasks
                    if task not in EXCLUDED_TASKS:
                        tasks.add(task)

        if not tasks:
            lines.append("No error metrics found in results.")
            lines.append("Run: python scripts/compute_error_metrics.py --save")
            return "\n".join(lines)

        # For each task, compare across models
        for task_name in sorted(tasks):
            lines.append(f"\n{'='*90}")
            lines.append(f"TASK: {task_name}")
            lines.append("=" * 90)

            # Get unit from first model that has this task
            unit = ''
            for result in self.results:
                if 'error_metrics' in result and task_name in result['error_metrics']:
                    task_data = result['error_metrics'][task_name]
                    unit = task_data.get('unit', '')
                    break

            # Collect model data
            model_metrics = []
            for result in self.results:
                if 'error_metrics' in result and task_name in result['error_metrics']:
                    task_data = result['error_metrics'][task_name]
                    metrics = task_data.get('metrics', {})
                    if metrics.get('count', 0) > 0:
                        model_metrics.append({
                            'model': self.get_model_name(result),
                            'mae': metrics.get('mae', 0),
                            'median': metrics.get('median', 0),
                            'mape': metrics.get('mape', 0),
                            'p90': metrics.get('p90', 0),
                            'p95': metrics.get('p95', 0),
                            'count': metrics.get('count', 0),
                            'accuracy_at': metrics.get('accuracy_at', {}),
                        })

            if not model_metrics:
                lines.append("  No data available")
                continue

            # Sort by MAE (lower is better)
            model_metrics.sort(key=lambda x: x['mae'])

            lines.append(f"\n{'Model':<35} {'MAE':>12} {'Median':>12} {'MAPE':>10} {'P90':>12} {'Acc@5%':>10}")
            lines.append("-" * 95)

            for m in model_metrics:
                acc5 = m['accuracy_at'].get('5%', 0)
                lines.append(
                    f"{m['model']:<35} {m['mae']:>10.2f}{unit:<2} {m['median']:>10.2f}{unit:<2} "
                    f"{m['mape']:>8.1f}% {m['p90']:>10.2f}{unit:<2} {acc5:>8.1f}%"
                )

            # Show accuracy at various tolerance levels
            lines.append("")
            lines.append("Accuracy at Different Tolerance Levels:")
            lines.append("-" * 95)

            # Get all tolerance levels
            all_tols = set()
            for m in model_metrics:
                all_tols.update(m['accuracy_at'].keys())

            if all_tols:
                # Sort numerically
                def tol_key(t):
                    num = float(re.search(r'[\d.]+', t).group())
                    return num
                sorted_tols = sorted(all_tols, key=tol_key)[:7]

                header = f"{'Model':<35}"
                for tol in sorted_tols:
                    header += f" {tol:>10}"
                lines.append(header)
                lines.append("-" * 95)

                for m in model_metrics:
                    row = f"{m['model']:<35}"
                    for tol in sorted_tols:
                        acc = m['accuracy_at'].get(tol, 0)
                        row += f" {acc:>9.1f}%"
                    lines.append(row)

        # Summary: best model per task
        lines.append("")
        lines.append("=" * 90)
        lines.append("BEST MODELS BY TASK (Lowest MAE)")
        lines.append("=" * 90)

        for task_name in sorted(tasks):
            best_model = None
            best_mae = float('inf')
            best_unit = ''

            for result in self.results:
                if 'error_metrics' in result and task_name in result['error_metrics']:
                    task_data = result['error_metrics'][task_name]
                    metrics = task_data.get('metrics', {})
                    mae = metrics.get('mae', float('inf'))
                    if mae < best_mae:
                        best_mae = mae
                        best_model = self.get_model_name(result)
                        best_unit = task_data.get('unit', '')

            if best_model:
                lines.append(f"  {task_name:<25} {best_model:<35} MAE: {best_mae:>10.2f}{best_unit}")

        lines.append("")
        return "\n".join(lines)

    def analyze_place_granularity(self) -> str:
        """Analyze Place Association task at different granularity levels."""
        lines = []
        lines.append("=" * 100)
        lines.append("PLACE ASSOCIATION - ACCURACY BY GRANULARITY LEVEL")
        lines.append("Evaluates how well models identify: Continent → Country → Region → City")
        lines.append("=" * 100)
        lines.append("")

        # Load city data for region lookup
        cities_file = Path('data/raw/cities15000.txt')
        admin1_file = Path('data/raw/admin1CodesASCII.txt')

        if not cities_file.exists() or not admin1_file.exists():
            lines.append("Required data files not found:")
            lines.append("  - data/raw/cities15000.txt")
            lines.append("  - data/raw/admin1CodesASCII.txt")
            lines.append("Download from: https://download.geonames.org/export/dump/")
            return "\n".join(lines)

        # Load admin1 names
        admin1_names = {}
        with open(admin1_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                try:
                    admin1_names[row[0]] = row[1]
                except:
                    continue

        # Load city data
        city_data = {}
        with open(cities_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                try:
                    city_name = row[1]
                    country_code = row[8]
                    admin1_code = row[10]
                    timezone = row[17]
                    continent = timezone.split('/')[0] if '/' in timezone else 'Unknown'
                    admin1_key = f"{country_code}.{admin1_code}"
                    region = admin1_names.get(admin1_key, '')
                    key = f"{city_name.lower()}_{country_code.lower()}"
                    city_data[key] = {'continent': continent, 'region': region}
                except:
                    continue

        # Process each model
        all_results = {}
        for result in self.results:
            model_name = self.get_model_name(result)
            source_path = Path(result.get('_source_path', ''))
            model_dir = source_path.parent

            result_file = model_dir / 'task_results' / 'applied_task1_place_association.json'
            if not result_file.exists():
                continue

            stats = self._analyze_granularity_for_model(result_file, city_data)
            if stats:
                all_results[model_name] = stats

        if not all_results:
            lines.append("No Place Association results found.")
            return "\n".join(lines)

        # Print results table
        lines.append(f"{'Model':<30} {'Continent':>12} {'Country':>12} {'Region':>12} {'City':>12} {'All':>10}")
        lines.append("-" * 100)

        for model in sorted(all_results.keys(),
                           key=lambda m: all_results[m]['city']['correct']/max(all_results[m]['city']['total'],1),
                           reverse=True):
            stats = all_results[model]
            cont_acc = stats['continent']['correct'] / max(stats['continent']['total'], 1) * 100
            country_acc = stats['country']['correct'] / max(stats['country']['total'], 1) * 100
            region_acc = stats['region']['correct'] / max(stats['region']['total'], 1) * 100
            city_acc = stats['city']['correct'] / max(stats['city']['total'], 1) * 100
            all_acc = stats['all']['correct'] / max(stats['all']['total'], 1) * 100
            lines.append(f"{model:<30} {cont_acc:>10.1f}% {country_acc:>10.1f}% {region_acc:>10.1f}% {city_acc:>10.1f}% {all_acc:>8.1f}%")

        lines.append("")
        return "\n".join(lines)

    def _analyze_granularity_for_model(self, result_file: Path, city_data: dict) -> dict:
        """Analyze granularity for a single model."""
        with open(result_file) as f:
            results = json.load(f)

        stats = {
            'continent': {'correct': 0, 'total': 0},
            'country': {'correct': 0, 'total': 0},
            'region': {'correct': 0, 'total': 0},
            'city': {'correct': 0, 'total': 0},
            'all': {'correct': 0, 'total': 0},
        }

        def normalize_text(text):
            if not text:
                return ''
            return re.sub(r'[^a-z0-9]', '', text.lower())

        def check_continent_match(response, continent):
            response_norm = normalize_text(response)
            if normalize_text(continent) in response_norm:
                return True
            for alias in CONTINENT_ALIASES.get(continent, []):
                if normalize_text(alias) in response_norm:
                    return True
            return False

        def check_country_match(response, country_code):
            response_norm = normalize_text(response)
            country_name = COUNTRY_NAMES.get(country_code, country_code)
            if normalize_text(country_name) in response_norm:
                return True
            if normalize_text(country_code) in response_norm:
                return True
            return False

        def check_region_match(response, region):
            if not region:
                return False
            return normalize_text(region) in normalize_text(response)

        for result in results.get('results', []):
            response = result.get('response', '')
            match = re.search(r'FINAL ANSWER:?\s*([^\n]+)', response, re.IGNORECASE)
            answer = match.group(1).strip() if match else response

            gt = result.get('ground_truth', {})
            country_code = gt.get('country') or gt.get('country_code', '')
            city_name = gt.get('name') or gt.get('location_name', '')

            city_key = f"{city_name.lower()}_{country_code.lower()}"
            city_info = city_data.get(city_key, {})
            continent = city_info.get('continent', 'Unknown')
            region = city_info.get('region', '')

            continent_correct = check_continent_match(answer, continent)
            country_correct = check_country_match(answer, country_code)
            region_correct = check_region_match(answer, region) if region else False
            city_correct = result.get('correct', False)

            stats['continent']['total'] += 1
            stats['country']['total'] += 1
            stats['city']['total'] += 1
            stats['all']['total'] += 1

            if region:
                stats['region']['total'] += 1
                if region_correct:
                    stats['region']['correct'] += 1

            if continent_correct:
                stats['continent']['correct'] += 1
            if country_correct:
                stats['country']['correct'] += 1
            if city_correct:
                stats['city']['correct'] += 1
            if city_correct and country_correct and continent_correct:
                stats['all']['correct'] += 1

        return stats

    def analyze_by_noise_level(self) -> str:
        """Analyze Place Association results by GPS noise level."""
        lines = []
        lines.append("=" * 110)
        lines.append("PLACE ASSOCIATION: ACCURACY BY GPS NOISE LEVEL")
        lines.append("Noise levels: clean (0m), 10m, 50m, 100m, 500m, 1000m (Gaussian σ)")
        lines.append("=" * 110)
        lines.append("")

        # Collect results by noise level for each model
        all_results = {}
        noise_levels_found = set()

        for result in self.results:
            model_name = self.get_model_name(result)
            source_path = Path(result.get('_source_path', ''))
            model_dir = source_path.parent

            result_file = model_dir / 'task_results' / 'applied_task1_place_association.json'
            if not result_file.exists():
                continue

            with open(result_file) as f:
                task_data = json.load(f)

            results_list = task_data.get('results', [])
            if not results_list:
                continue

            # Group by noise level
            noise_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

            for item in results_list:
                gt = item.get('ground_truth', {})
                noise_type = gt.get('noise_type', 'unknown')
                noise_sigma = gt.get('noise_sigma_m', -1)

                # Use noise_sigma if available, else parse from noise_type
                if noise_sigma >= 0:
                    noise_key = noise_sigma
                elif noise_type == 'clean':
                    noise_key = 0
                elif noise_type.startswith('gps_'):
                    try:
                        noise_key = int(noise_type.replace('gps_', '').replace('m', ''))
                    except:
                        noise_key = -1
                else:
                    noise_key = -1

                if noise_key >= 0:
                    noise_levels_found.add(noise_key)
                    noise_stats[noise_key]['total'] += 1
                    if item.get('correct', False):
                        noise_stats[noise_key]['correct'] += 1

            if noise_stats:
                all_results[model_name] = dict(noise_stats)

        if not all_results or not noise_levels_found:
            lines.append("No noise level data found in Place Association results.")
            lines.append("Regenerate data with: python scripts/generate_all_samples.py")
            return "\n".join(lines)

        # Sort noise levels
        sorted_noise = sorted(noise_levels_found)

        # Build header
        header = f"{'Model':<35}"
        for nl in sorted_noise:
            label = 'clean' if nl == 0 else f'{nl}m'
            header += f" {label:>10}"
        header += f" {'Δ clean→1km':>12}"
        lines.append(header)
        lines.append("-" * (35 + 11 * len(sorted_noise) + 13))

        # Sort models by clean accuracy
        def get_clean_acc(model):
            stats = all_results.get(model, {}).get(0, {'correct': 0, 'total': 1})
            return stats['correct'] / max(stats['total'], 1)

        for model in sorted(all_results.keys(), key=get_clean_acc, reverse=True):
            noise_stats = all_results[model]
            row = f"{model:<35}"

            clean_acc = None
            max_noise_acc = None

            for nl in sorted_noise:
                stats = noise_stats.get(nl, {'correct': 0, 'total': 0})
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total'] * 100
                    row += f" {acc:>9.1f}%"
                    if nl == 0:
                        clean_acc = acc
                    if nl == max(sorted_noise):
                        max_noise_acc = acc
                else:
                    row += f" {'N/A':>10}"

            # Calculate degradation from clean to max noise
            if clean_acc is not None and max_noise_acc is not None:
                delta = max_noise_acc - clean_acc
                row += f" {delta:>+10.1f}%"
            else:
                row += f" {'N/A':>12}"

            lines.append(row)

        # Add summary
        lines.append("")
        lines.append("Note: Δ clean→1km shows accuracy change from clean to 1000m noise (negative = degradation)")

        lines.append("")
        return "\n".join(lines)

    def analyze_by_region(self) -> str:
        """Analyze results broken down by geographic region."""
        lines = []
        lines.append("=" * 110)
        lines.append("RESULTS BY GEOGRAPHIC REGION")
        lines.append("Breakdown by: Europe, Asia, Africa, N. America, S. America, Oceania")
        lines.append("=" * 110)
        lines.append("")

        regions = ['Europe', 'Asia', 'Africa', 'N. America', 'S. America', 'Oceania']

        # Process each model
        all_results = {}
        for result in self.results:
            model_name = self.get_model_name(result)
            source_path = Path(result.get('_source_path', ''))
            model_dir = source_path.parent

            task_results_dir = model_dir / 'task_results'
            if not task_results_dir.exists():
                continue

            model_data = {}
            for result_file in task_results_dir.glob('*.json'):
                task_name = self._get_task_display_name(result_file.stem)
                region_stats = self._analyze_regions_for_task(result_file)
                if region_stats:
                    model_data[task_name] = region_stats

            if model_data:
                all_results[model_name] = model_data

        if not all_results:
            lines.append("No results found for regional analysis.")
            return "\n".join(lines)

        # Get all tasks
        all_tasks = set()
        for model_data in all_results.values():
            all_tasks.update(model_data.keys())

        # Print results by task
        for task in sorted(all_tasks):
            lines.append(f"\n{'='*110}")
            lines.append(f"TASK: {task}")
            lines.append("=" * 110)

            header = f"{'Model':<28}"
            for r in regions:
                header += f"{r:>14}"
            header += f"{'Overall':>12}"
            lines.append(header)
            lines.append("-" * 110)

            for model in sorted(all_results.keys()):
                if task not in all_results[model]:
                    continue

                task_data = all_results[model][task]
                row = f"{model:<28}"

                total_correct = 0
                total_count = 0

                for region in regions:
                    if region in task_data and task_data[region]['total'] > 0:
                        acc = task_data[region]['correct'] / task_data[region]['total'] * 100
                        count = task_data[region]['total']
                        row += f"{acc:>10.1f}%({count:>3})"
                        total_correct += task_data[region]['correct']
                        total_count += task_data[region]['total']
                    else:
                        row += f"{'--':>14}"

                if total_count > 0:
                    overall = total_correct / total_count * 100
                    row += f"{overall:>10.1f}%"
                else:
                    row += f"{'--':>12}"

                lines.append(row)

        # Summary: Average accuracy by region across all tasks
        lines.append("")
        lines.append("=" * 110)
        lines.append("SUMMARY: Average Accuracy by Region (All Tasks)")
        lines.append("=" * 110)

        header = f"{'Model':<28}"
        for r in regions:
            header += f"{r:>14}"
        lines.append(header)
        lines.append("-" * 110)

        for model in sorted(all_results.keys()):
            row = f"{model:<28}"
            for region in regions:
                accs = []
                for task_data in all_results[model].values():
                    if region in task_data and task_data[region]['total'] >= 10:
                        acc = task_data[region]['correct'] / task_data[region]['total'] * 100
                        accs.append(acc)
                if accs:
                    avg = sum(accs) / len(accs)
                    row += f"{avg:>12.1f}%"
                else:
                    row += f"{'--':>14}"
            lines.append(row)

        lines.append("")
        return "\n".join(lines)

    def _analyze_regions_for_task(self, result_file: Path) -> dict:
        """Analyze regional breakdown for a single task."""
        try:
            with open(result_file) as f:
                results = json.load(f)
        except:
            return {}

        by_region = defaultdict(lambda: {'correct': 0, 'total': 0})

        for result in results.get('results', []):
            is_correct = result.get('correct', False)
            gt = result.get('ground_truth', {})

            # Try to get region from country code
            country_code = gt.get('country') or gt.get('country_code') or gt.get('start_country', '')
            region = COUNTRY_TO_CONTINENT.get(country_code, '')

            # If no country code, try to extract from coordinates
            if not region:
                region = self._get_region_from_coords(result, gt)

            if region:
                by_region[region]['total'] += 1
                if is_correct:
                    by_region[region]['correct'] += 1

        return dict(by_region)

    def _get_region_from_coords(self, result: dict, gt: dict) -> str:
        """Get region from coordinates using simple bounds."""
        lat, lon = None, None

        # Try to get country code first (most reliable)
        country_code = gt.get('country_code') or gt.get('country')
        if country_code and isinstance(country_code, str) and len(country_code) == 2:
            continent = COUNTRY_TO_CONTINENT.get(country_code.upper())
            if continent:
                return CONTINENT_ALIASES.get(continent, continent)

        # Try various coordinate sources
        if 'point_a' in result:
            p = result['point_a']
            lat, lon = p.get('lat'), p.get('lon')
        elif 'source_coords' in result:
            c = result['source_coords']
            lat, lon = c.get('lat'), c.get('lon')
        elif 'true_coordinates' in gt:
            c = gt['true_coordinates']
            lat, lon = c.get('lat'), c.get('lon')
        elif 'complete_coordinate' in gt:
            c = gt['complete_coordinate']
            lat, lon = c.get('latitude'), c.get('longitude')
        elif 'latitude' in gt:
            lat, lon = gt.get('latitude'), gt.get('longitude')
        elif 'lat' in gt:
            lat, lon = gt.get('lat'), gt.get('lon')

        # Try to extract from prompt
        if lat is None or lon is None:
            prompt = result.get('prompt', '')
            import re
            # Match patterns like "coordinates 11.974, -86.094" or "at -37.48, -59.59"
            coord_match = re.search(r'coordinates?\s+(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)', prompt, re.IGNORECASE)
            if coord_match:
                try:
                    lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
                except:
                    pass

        if lat is None or lon is None:
            return ''

        # Simple bounding box approach
        if 35 <= lat <= 72 and -25 <= lon <= 40:
            return 'Europe'
        if -10 <= lat <= 80 and 25 <= lon <= 180:
            return 'Asia'
        if -35 <= lat <= 37 and -20 <= lon <= 55:
            return 'Africa'
        if 5 <= lat <= 85 and -170 <= lon <= -50:
            return 'N. America'
        if -60 <= lat <= 15 and -85 <= lon <= -30:
            return 'S. America'
        if -50 <= lat <= 0 and 110 <= lon <= 180:
            return 'Oceania'

        return ''

    def _get_task_display_name(self, task_filename: str) -> str:
        """Get display name for task."""
        name_map = {
            'pure_gps_task1_format_conversion': 'Format Conversion',
            'pure_gps_task3_coordinate_system': 'Coordinate Transform',
            'pure_gps_task4_distance_calculation': 'Distance Calculation',
            'pure_gps_task5_bearing_computation': 'Bearing Computation',
            'pure_gps_task6_interpolation': 'Interpolation',
            'pure_gps_task7_area_perimeter': 'Polygon Area',
            'pure_gps_task8_bounding_box': 'Bounding Box',
            'pure_gps_task9_route_geometry': 'Route Geometry',
            'pure_gps_task10_relative_position': 'Relative Position',
            'applied_task1_place_association': 'Place Association',
            'applied_task3_name_disambiguation': 'Name Disambiguation',
            'applied_task5_relative_position': 'Relative Position (App)',
            'applied_task6_proximity': 'Proximity',
            'applied_task7_route_analysis': 'Route Analysis',
            'applied_task8_boundary_analysis': 'Boundary Analysis',
            'applied_task9_spatial_patterns': 'Spatial Patterns',
            'applied_task11_missing_data': 'Missing Data',
            'applied_task13_terrain': 'Terrain Classification',
        }
        for key, display in name_map.items():
            if key in task_filename:
                return display
        return task_filename.replace('.json', '').replace('_', ' ').title()

    def generate_charts(self, output_dir: Path) -> List[str]:
        """Generate all analysis charts and save to output directory."""
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping chart generation")
            return []

        generated = []

        # 1. Overall model comparison bar chart
        chart_path = self._generate_overall_comparison_chart(output_dir)
        if chart_path:
            generated.append(chart_path)

        # 2. Track comparison chart
        chart_path = self._generate_track_comparison_chart(output_dir)
        if chart_path:
            generated.append(chart_path)

        # 3. Regional heatmap
        chart_path = self._generate_regional_heatmap(output_dir)
        if chart_path:
            generated.append(chart_path)

        # 4. Place association by population chart
        chart_path = self._generate_place_population_chart(output_dir)
        if chart_path:
            generated.append(chart_path)

        return generated

    def _generate_overall_comparison_chart(self, output_dir: Path) -> Optional[str]:
        """Generate overall accuracy comparison bar chart."""
        if not self.results:
            return None

        models = []
        accuracies = []

        for result in sorted(self.results,
                            key=lambda x: x.get('overall', {}).get('overall_accuracy', 0),
                            reverse=True):
            models.append(self.get_model_name(result))
            accuracies.append(result.get('overall', {}).get('overall_accuracy', 0))

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        bars = ax.barh(models[::-1], accuracies[::-1], color=colors[::-1])

        ax.set_xlabel('Accuracy (%)', fontsize=12)
        ax.set_title('GPSBench Overall Model Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)

        # Add value labels
        for bar, acc in zip(bars, accuracies[::-1]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{acc:.1f}%', va='center', fontsize=10)

        plt.tight_layout()
        chart_path = output_dir / 'chart_overall_comparison.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(chart_path)

    def _generate_track_comparison_chart(self, output_dir: Path) -> Optional[str]:
        """Generate track comparison grouped bar chart."""
        if not self.results:
            return None

        models = []
        pure_gps = []
        applied = []

        for result in sorted(self.results,
                            key=lambda x: x.get('overall', {}).get('overall_accuracy', 0),
                            reverse=True):
            models.append(self.get_model_name(result))
            tracks = result.get('tracks', {})
            pure_gps.append(tracks.get('pure_gps', {}).get('overall_accuracy', 0))
            applied.append(tracks.get('applied', {}).get('overall_accuracy', 0))

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 7))
        bars1 = ax.bar(x - width/2, pure_gps, width, label='Pure GPS', color='steelblue')
        bars2 = ax.bar(x + width/2, applied, width, label='Applied', color='coral')

        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('GPSBench Track Comparison (Pure GPS vs Applied)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_path = output_dir / 'chart_track_comparison.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(chart_path)

    def _generate_regional_heatmap(self, output_dir: Path) -> Optional[str]:
        """Generate regional breakdown heatmap."""
        if not self.results:
            return None

        regions = ['Europe', 'Asia', 'Africa', 'N. America', 'S. America', 'Oceania']
        models = [self.get_model_name(r) for r in sorted(
            self.results,
            key=lambda x: x.get('overall', {}).get('overall_accuracy', 0),
            reverse=True
        )]

        # Collect regional accuracy data
        data = np.zeros((len(models), len(regions)))

        for i, result in enumerate(sorted(self.results,
                                          key=lambda x: x.get('overall', {}).get('overall_accuracy', 0),
                                          reverse=True)):
            model_dir = self._find_model_dir(result)
            if not model_dir:
                continue

            task_results_dir = model_dir / 'task_results'
            if not task_results_dir.exists():
                continue

            # Aggregate across all tasks
            region_stats = {r: {'correct': 0, 'total': 0} for r in regions}

            for task_file in task_results_dir.glob('*.json'):
                try:
                    with open(task_file, 'r') as f:
                        task_data = json.load(f)

                    for item in task_data.get('results', []):
                        gt = item.get('ground_truth', {})
                        region = self._get_region_from_coords(item, gt)
                        if region in region_stats:
                            region_stats[region]['total'] += 1
                            if item.get('correct', False):
                                region_stats[region]['correct'] += 1
                except:
                    continue

            for j, region in enumerate(regions):
                stats = region_stats[region]
                if stats['total'] > 0:
                    data[i, j] = stats['correct'] / stats['total'] * 100

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        ax.set_xticks(np.arange(len(regions)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(regions, fontsize=11)
        ax.set_yticklabels(models, fontsize=10)

        # Add value annotations
        for i in range(len(models)):
            for j in range(len(regions)):
                color = 'white' if data[i, j] < 50 else 'black'
                ax.text(j, i, f'{data[i, j]:.1f}%', ha='center', va='center',
                       fontsize=9, color=color)

        ax.set_title('GPSBench Accuracy by Geographic Region', fontsize=14, fontweight='bold')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom', fontsize=11)

        plt.tight_layout()
        chart_path = output_dir / 'chart_regional_heatmap.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(chart_path)

    def _load_cities_database(self) -> list:
        """Load cities database for population lookup."""
        cities_file = Path('data/raw/cities15000.txt')
        if not cities_file.exists():
            return []

        cities = []
        try:
            with open(cities_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 15:
                        try:
                            lat = float(parts[4])
                            lon = float(parts[5])
                            pop = int(parts[14]) if parts[14] else 0
                            cities.append((lat, lon, pop))
                        except:
                            continue
        except:
            return []

        return cities

    def _load_source_task_data(self) -> Dict[str, List[dict]]:
        """Load all source task data files for coordinate lookup.

        Prefers split files (used in evaluation) over full source files.
        """
        if hasattr(self, '_source_task_cache'):
            return self._source_task_cache

        self._source_task_cache = {}

        # Map result file names to source data paths (split file, then full source as fallback)
        task_mappings = {
            # Applied tasks
            'applied_task1_place_association': [
                'data/track_applied/splits/task1_place_association_test.json',
                'data/track_applied/task1_place_association.json'
            ],
            'applied_task3_name_disambiguation': [
                'data/track_applied/splits/task3_name_disambiguation_test.json',
                'data/track_applied/task3_name_disambiguation.json'
            ],
            'applied_task5_relative_position': [
                'data/track_applied/splits/task5_relative_position_test.json',
                'data/track_applied/task5_relative_position.json'
            ],
            'applied_task6_proximity': [
                'data/track_applied/splits/task6_proximity_test.json',
                'data/track_applied/task6_proximity.json'
            ],
            'applied_task7_route_analysis': [
                'data/track_applied/splits/task7_route_analysis_test.json',
                'data/track_applied/task7_route_analysis.json'
            ],
            'applied_task8_boundary_analysis': [
                'data/track_applied/splits/task8_boundary_analysis_test.json',
                'data/track_applied/task8_boundary_analysis.json'
            ],
            'applied_task9_spatial_patterns': [
                'data/track_applied/splits/task9_spatial_patterns_test.json',
                'data/track_applied/task9_spatial_patterns.json'
            ],
            'applied_task11_missing_data': [
                'data/track_applied/splits/task11_missing_data_test.json',
                'data/track_applied/task11_missing_data.json'
            ],
            'applied_task13_terrain_classification': [
                'data/track_applied/splits/task13_terrain_classification_test.json',
                'data/track_applied/task13_terrain_classification.json'
            ],
            # Pure GPS tasks
            'pure_gps_task1_format_conversion': [
                'data/track_pure_gps/splits/task1_format_conversion_test.json',
                'data/track_pure_gps/task1_format_conversion.json'
            ],
            'pure_gps_task3_coordinate_system_transformation': [
                'data/track_pure_gps/splits/task3_coordinate_system_transformation_test.json',
                'data/track_pure_gps/task3_coordinate_system_transformation.json'
            ],
            'pure_gps_task4_distance_calculation': [
                'data/track_pure_gps/splits/task4_distance_calculation_test.json',
                'data/track_pure_gps/task4_distance_calculation.json'
            ],
            'pure_gps_task5_bearing_computation': [
                'data/track_pure_gps/splits/task5_bearing_computation_test.json',
                'data/track_pure_gps/task5_bearing_computation.json'
            ],
            'pure_gps_task6_interpolation': [
                'data/track_pure_gps/splits/task6_interpolation_test.json',
                'data/track_pure_gps/task6_interpolation.json'
            ],
            'pure_gps_task7_area_perimeter': [
                'data/track_pure_gps/splits/task7_area_perimeter_test.json',
                'data/track_pure_gps/task7_area_perimeter.json'
            ],
            'pure_gps_task8_bounding_box': [
                'data/track_pure_gps/splits/task8_bounding_box_test.json',
                'data/track_pure_gps/task8_bounding_box.json'
            ],
            'pure_gps_task9_route_geometry': [
                'data/track_pure_gps/splits/task9_route_geometry_test.json',
                'data/track_pure_gps/task9_route_geometry.json'
            ],
            'pure_gps_task10_relative_position': [
                'data/track_pure_gps/splits/task10_relative_position_test.json',
                'data/track_pure_gps/task10_relative_position.json'
            ],
        }

        for task_key, source_paths in task_mappings.items():
            for source_path in source_paths:
                path = Path(source_path)
                if path.exists():
                    try:
                        with open(path, 'r') as f:
                            self._source_task_cache[task_key] = json.load(f)
                        break  # Use first available file
                    except:
                        pass

        return self._source_task_cache

    def _extract_coordinates_from_source(self, task_key: str, example_id: int, source_data: List[dict]) -> Tuple[Optional[float], Optional[float]]:
        """Extract coordinates from source task data based on task type.

        Returns a representative coordinate for the task item to determine
        which population bucket it belongs to.
        """
        if not source_data or example_id >= len(source_data):
            return None, None

        item = source_data[example_id]

        # Task-specific coordinate extraction
        if 'place_association' in task_key or 'terrain_classification' in task_key:
            # Has coordinate: {lat, lon} or coordinate_string: "lat, lon"
            coord = item.get('coordinate', {})
            if coord and coord.get('lat') is not None:
                return coord.get('lat'), coord.get('lon')
            # Fallback to coordinate_string
            coord_str = item.get('coordinate_string', '')
            if coord_str:
                parts = coord_str.split(',')
                if len(parts) == 2:
                    try:
                        return float(parts[0].strip()), float(parts[1].strip())
                    except:
                        pass
            return None, None

        elif 'name_disambiguation' in task_key:
            # Has coordinate_string or candidates with coords
            # Use the correct answer's coordinates
            gt = item.get('ground_truth', {})
            for candidate in item.get('candidates', []):
                if candidate.get('name') == gt.get('name') and candidate.get('country') == gt.get('country'):
                    return candidate.get('lat'), candidate.get('lon')
            # Fallback: parse coordinate_string
            coord_str = item.get('coordinate_string', '')
            if coord_str:
                parts = coord_str.split(',')
                if len(parts) == 2:
                    try:
                        return float(parts[0].strip()), float(parts[1].strip())
                    except:
                        pass
            return None, None

        elif 'missing_data' in task_key:
            # Has ground_truth.complete_coordinate
            gt = item.get('ground_truth', {})
            complete = gt.get('complete_coordinate', {})
            return complete.get('latitude'), complete.get('longitude')

        elif 'relative_position' in task_key or 'distance_calculation' in task_key or \
             'bearing_computation' in task_key or 'interpolation' in task_key:
            # Has point_a: {lat, lon}
            point_a = item.get('point_a', {})
            return point_a.get('lat'), point_a.get('lon')

        elif 'proximity' in task_key:
            # Has reference_point: {lat, lon}
            ref = item.get('reference_point', {})
            return ref.get('lat'), ref.get('lon')

        elif 'route_analysis' in task_key:
            # Has start: {lat, lon}
            start = item.get('start', {})
            return start.get('lat'), start.get('lon')

        elif 'format_conversion' in task_key:
            # Has ground_truth: {latitude, longitude}
            gt = item.get('ground_truth', {})
            return gt.get('latitude'), gt.get('longitude')

        elif 'coordinate_system_transformation' in task_key:
            # Has source_coords: {lat, lon}
            coords = item.get('source_coords', {})
            return coords.get('lat'), coords.get('lon')

        elif 'boundary_analysis' in task_key or 'spatial_patterns' in task_key or \
             'bounding_box' in task_key:
            # Has points: [{lat, lon}, ...] - use centroid
            points = item.get('points', [])
            if points:
                lats = [p.get('lat', 0) for p in points if p.get('lat') is not None]
                lons = [p.get('lon', 0) for p in points if p.get('lon') is not None]
                if lats and lons:
                    return sum(lats) / len(lats), sum(lons) / len(lons)
            return None, None

        elif 'area_perimeter' in task_key:
            # Has vertices: [{lat, lon}, ...] - use centroid
            vertices = item.get('vertices', [])
            if vertices:
                lats = [v.get('lat', 0) for v in vertices if v.get('lat') is not None]
                lons = [v.get('lon', 0) for v in vertices if v.get('lon') is not None]
                if lats and lons:
                    return sum(lats) / len(lats), sum(lons) / len(lons)
            return None, None

        elif 'route_geometry' in task_key:
            # Has points: [{index, lat, lon}, ...] - use first point
            points = item.get('points', [])
            if points:
                return points[0].get('lat'), points[0].get('lon')
            return None, None

        return None, None

    def _find_nearest_city_population(self, lat: float, lon: float, cities: list) -> int:
        """Find population of nearest city using simple distance."""
        if not cities:
            return 0

        min_dist = float('inf')
        nearest_pop = 0

        for city_lat, city_lon, pop in cities:
            # Simple squared distance (good enough for nearest neighbor)
            dist = (lat - city_lat) ** 2 + (lon - city_lon) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_pop = pop

        return nearest_pop

    def _extract_coordinates_from_result(self, item: dict) -> Optional[tuple]:
        """Extract lat/lon from a result item."""
        gt = item.get('ground_truth', {})

        # Try various coordinate sources
        if 'true_coordinates' in gt:
            c = gt['true_coordinates']
            return c.get('lat'), c.get('lon')
        elif 'complete_coordinate' in gt:
            c = gt['complete_coordinate']
            return c.get('latitude'), c.get('longitude')
        elif 'latitude' in gt and 'longitude' in gt:
            return gt.get('latitude'), gt.get('longitude')
        elif 'lat' in gt and 'lon' in gt:
            return gt.get('lat'), gt.get('lon')
        elif 'point_a' in item:
            p = item['point_a']
            return p.get('lat'), p.get('lon')
        elif 'source_coords' in item:
            c = item['source_coords']
            return c.get('lat'), c.get('lon')

        # Try to extract from prompt
        prompt = item.get('prompt', '')
        coord_match = re.search(r'coordinates?\s+(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)', prompt, re.IGNORECASE)
        if coord_match:
            try:
                return float(coord_match.group(1)), float(coord_match.group(2))
            except:
                pass

        return None, None

    def _generate_place_population_chart(self, output_dir: Path) -> Optional[str]:
        """Generate accuracy by nearest city population for all tasks."""
        if not self.results:
            return None

        # Load cities database
        print("    Loading cities database...")
        cities = self._load_cities_database()
        if not cities:
            print("    Warning: Could not load cities database")
            return None

        # Load source task data for coordinate lookup
        print("    Loading source task data...")
        source_task_data = self._load_source_task_data()

        pop_buckets = ['<50k', '50k-100k', '100k-500k', '500k-1M', '>1M']

        def get_pop_bucket(pop):
            if pop < 50000:
                return '<50k'
            elif pop < 100000:
                return '50k-100k'
            elif pop < 500000:
                return '100k-500k'
            elif pop < 1000000:
                return '500k-1M'
            return '>1M'

        models = []
        model_data = {}

        for result in sorted(self.results,
                            key=lambda x: x.get('overall', {}).get('overall_accuracy', 0),
                            reverse=True):
            model_name = self.get_model_name(result)
            model_dir = self._find_model_dir(result)
            if not model_dir:
                continue

            task_results_dir = model_dir / 'task_results'
            if not task_results_dir.exists():
                continue

            pop_stats = {b: {'correct': 0, 'total': 0} for b in pop_buckets}

            # Process all task files
            for task_file in task_results_dir.glob('*.json'):
                try:
                    with open(task_file, 'r') as f:
                        task_data = json.load(f)

                    # Get task key from filename (e.g., "applied_task1_place_association.json" -> "applied_task1_place_association")
                    task_key = task_file.stem
                    source_data = source_task_data.get(task_key, [])

                    for item in task_data.get('results', []):
                        example_id = item.get('example_id', -1)

                        # Extract coordinates from source task data
                        lat, lon = self._extract_coordinates_from_source(task_key, example_id, source_data)
                        if lat is None or lon is None:
                            continue

                        # Find nearest city population
                        pop = self._find_nearest_city_population(lat, lon, cities)
                        bucket = get_pop_bucket(pop)

                        pop_stats[bucket]['total'] += 1
                        if item.get('correct', False):
                            pop_stats[bucket]['correct'] += 1
                except:
                    continue

            models.append(model_name)
            model_data[model_name] = {
                b: (pop_stats[b]['correct'] / pop_stats[b]['total'] * 100
                    if pop_stats[b]['total'] > 0 else 0)
                for b in pop_buckets
            }

        if not models:
            return None

        x = np.arange(len(pop_buckets))
        width = 0.8 / len(models)
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

        fig, ax = plt.subplots(figsize=(14, 7))

        for i, model in enumerate(models):
            accuracies = [model_data[model][b] for b in pop_buckets]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, accuracies, width, label=model, color=colors[i])

        ax.set_xlabel('Nearest City Population', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('All Tasks: Accuracy by Nearest City Population', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pop_buckets, fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_path = output_dir / 'chart_accuracy_by_population.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(chart_path)

    def _find_model_dir(self, result: Dict) -> Optional[Path]:
        """Find the directory for a model's results."""
        model_name = result.get('model', '')
        model_safe = model_name.replace('/', '_').replace(':', '_')

        results_dir = Path(self.results_dir)
        matches = list(results_dir.glob(f'gpsbench_{model_safe}_*'))

        if matches:
            return sorted(matches, reverse=True)[0]
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and compare GPSBench evaluation results'
    )
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directory containing result files (default: results)'
    )
    parser.add_argument(
        '--output-dir',
        default='analysis',
        help='Output directory for analysis results (default: analysis)'
    )
    parser.add_argument(
        '--track',
        choices=['pure_gps', 'applied'],
        help='Filter to specific track for task comparison'
    )
    parser.add_argument(
        '--minimal',
        action='store_true',
        help='Only show summary and overall comparison (skip detailed analysis)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize analyzer
    analyzer = ResultsAnalyzer(args.results_dir)
    analyzer.load_all_results()

    if not analyzer.results:
        print("No results found to analyze.")
        sys.exit(1)

    print(f"\nRunning comprehensive analysis...")
    print(f"Output directory: {output_dir}/\n")

    # Collect all reports
    reports = []

    # Basic reports (always included)
    reports.append(("Summary Statistics", analyzer.generate_summary_stats()))
    reports.append(("Overall Comparison", analyzer.generate_overall_comparison()))
    reports.append(("Track Comparison", analyzer.generate_track_comparison()))

    if not args.minimal:
        # Detailed reports
        reports.append(("Task Comparison", analyzer.generate_task_comparison(args.track)))
        reports.append(("Strengths & Weaknesses", analyzer.find_strengths_weaknesses()))
        reports.append(("Error Metrics", analyzer.analyze_error_metrics()))
        reports.append(("Place Granularity", analyzer.analyze_place_granularity()))
        reports.append(("Noise Level Analysis", analyzer.analyze_by_noise_level()))
        reports.append(("Regional Breakdown", analyzer.analyze_by_region()))

    # Print to console
    for title, content in reports:
        print(content)

    # Save full report to text file
    full_report = "\n".join([content for _, content in reports])
    report_file = output_dir / "full_report.txt"
    with open(report_file, 'w') as f:
        f.write(full_report)
    print(f"\nSaved: {report_file}")

    # Save individual reports
    report_files = {
        "Summary Statistics": "01_summary.txt",
        "Overall Comparison": "02_overall_comparison.txt",
        "Track Comparison": "03_track_comparison.txt",
        "Task Comparison": "04_task_comparison.txt",
        "Strengths & Weaknesses": "05_strengths_weaknesses.txt",
        "Error Metrics": "06_error_metrics.txt",
        "Place Granularity": "07_place_granularity.txt",
        "Noise Level Analysis": "08_noise_level.txt",
        "Regional Breakdown": "09_regional_breakdown.txt",
    }

    for title, content in reports:
        if title in report_files:
            filepath = output_dir / report_files[title]
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Saved: {filepath}")

    # Save CSV export
    csv_file = output_dir / "results.csv"
    analyzer.generate_csv_export(str(csv_file))

    # Save Markdown export
    md_file = output_dir / "results.md"
    markdown_content = analyzer.generate_markdown_table()
    with open(md_file, 'w') as f:
        f.write(markdown_content)
    print(f"Saved: {md_file}")

    # Generate summary JSON
    summary_data = {
        'models': [],
        'generated_at': datetime.now().isoformat(),
    }
    for result in sorted(analyzer.results,
                        key=lambda x: x.get('overall', {}).get('overall_accuracy', 0),
                        reverse=True):
        model_info = {
            'name': analyzer.get_model_name(result),
            'overall_accuracy': result.get('overall', {}).get('overall_accuracy', 0),
            'total_correct': result.get('overall', {}).get('total_correct', 0),
            'total_cases': result.get('overall', {}).get('total_cases', 0),
        }
        if 'tracks' in result:
            model_info['tracks'] = {}
            for track_name, track_data in result['tracks'].items():
                model_info['tracks'][track_name] = {
                    'accuracy': track_data.get('overall_accuracy', 0),
                    'correct': track_data.get('total_correct', 0),
                    'total': track_data.get('total_cases', 0),
                }
        summary_data['models'].append(model_info)

    json_file = output_dir / "summary.json"
    with open(json_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved: {json_file}")

    # Generate charts
    if not args.minimal:
        print("\nGenerating charts...")
        chart_files = analyzer.generate_charts(output_dir)
        for chart in chart_files:
            print(f"Saved: {chart}")

    print(f"\n{'='*60}")
    print(f"Analysis complete! All files saved to: {output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
