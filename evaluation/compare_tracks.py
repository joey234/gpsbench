#!/usr/bin/env python3
"""
Compare Pure GPS vs Applied Track Performance
Analyzes model strengths across different types of GPS understanding
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import glob

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import LLMClient
from track_config import get_tier_display_name, get_total_tests


class TrackComparator:
    """Compare Pure GPS vs Applied track performance"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def find_result_files(self, model_name: Optional[str] = None) -> Dict[str, List[Path]]:
        """Find result files for Pure GPS and Applied tracks"""

        pure_pattern = "eval_pure_gps_*.json"
        applied_pattern = "eval_applied_*.json"

        if model_name:
            model_safe = model_name.replace('/', '_')
            pure_pattern = f"eval_pure_gps_{model_safe}_*.json"
            applied_pattern = f"eval_applied_{model_safe}_*.json"

        pure_files = sorted(self.results_dir.glob(pure_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        applied_files = sorted(self.results_dir.glob(applied_pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        return {
            'pure_gps': pure_files,
            'applied': applied_files
        }

    def load_result(self, filepath: Path) -> Dict:
        """Load a result file"""
        with open(filepath, 'r') as f:
            return json.load(f)

    def compare_latest(self, model_name: Optional[str] = None) -> Dict:
        """Compare latest results for each track"""

        files = self.find_result_files(model_name)

        if not files['pure_gps']:
            raise FileNotFoundError(f"No Pure GPS results found for model: {model_name}")
        if not files['applied']:
            raise FileNotFoundError(f"No Applied results found for model: {model_name}")

        pure_result = self.load_result(files['pure_gps'][0])
        applied_result = self.load_result(files['applied'][0])

        comparison = {
            'comparison_timestamp': datetime.now().isoformat(),
            'model': pure_result['model'],
            'provider': pure_result['provider'],
            'pure_gps': {
                'file': files['pure_gps'][0].name,
                'timestamp': pure_result['timestamp'],
                'overall_accuracy': pure_result['overall_metrics']['overall_accuracy'],
                'total_correct': pure_result['overall_metrics']['total_correct'],
                'total_cases': pure_result['overall_metrics']['total_cases'],
                'tier_accuracies': pure_result['overall_metrics']['tier_accuracies']
            },
            'applied': {
                'file': files['applied'][0].name,
                'timestamp': applied_result['timestamp'],
                'overall_accuracy': applied_result['overall_metrics']['overall_accuracy'],
                'total_correct': applied_result['overall_metrics']['total_correct'],
                'total_cases': applied_result['overall_metrics']['total_cases'],
                'tier_accuracies': applied_result['overall_metrics']['tier_accuracies']
            }
        }

        # Calculate differences
        comparison['analysis'] = self._analyze_comparison(comparison)

        return comparison

    def _analyze_comparison(self, comparison: Dict) -> Dict:
        """Analyze comparison and generate insights"""

        pure_acc = comparison['pure_gps']['overall_accuracy']
        applied_acc = comparison['applied']['overall_accuracy']

        diff = applied_acc - pure_acc
        diff_percent = (diff / pure_acc * 100) if pure_acc > 0 else 0

        # Determine strength
        if abs(diff) < 2:
            strength = "balanced"
            insight = "Model performs equally well on both pure coordinate understanding and geographic knowledge"
        elif diff > 0:
            strength = "geographic_knowledge"
            insight = f"Model is {diff:.1f}% stronger in geographic knowledge than pure coordinate math"
        else:
            strength = "coordinate_math"
            insight = f"Model is {abs(diff):.1f}% stronger in pure coordinate math than geographic knowledge"

        # Tier-level analysis
        tier_insights = []

        for tier_name, pure_tier_acc in comparison['pure_gps']['tier_accuracies'].items():
            tier_display = get_tier_display_name(tier_name)
            tier_insights.append({
                'tier': tier_display,
                'accuracy': pure_tier_acc,
                'track': 'pure_gps'
            })

        for tier_name, applied_tier_acc in comparison['applied']['tier_accuracies'].items():
            tier_display = get_tier_display_name(tier_name)
            tier_insights.append({
                'tier': tier_display,
                'accuracy': applied_tier_acc,
                'track': 'applied'
            })

        # Sort by accuracy
        tier_insights.sort(key=lambda x: x['accuracy'], reverse=True)

        return {
            'accuracy_difference': diff,
            'accuracy_difference_percent': diff_percent,
            'model_strength': strength,
            'insight': insight,
            'tier_rankings': tier_insights,
            'strongest_tier': tier_insights[0] if tier_insights else None,
            'weakest_tier': tier_insights[-1] if tier_insights else None
        }

    def print_comparison(self, comparison: Dict):
        """Print formatted comparison"""

        print(f"\n{'='*80}")
        print(f"Track Performance Comparison")
        print(f"Model: {comparison['model']}")
        print(f"Provider: {comparison['provider']}")
        print(f"{'='*80}\n")

        # Overall scores
        pure = comparison['pure_gps']
        applied = comparison['applied']

        print(f"Pure GPS Knowledge Track:")
        print(f"  Accuracy: {pure['overall_accuracy']:.2f}%")
        print(f"  Correct: {pure['total_correct']}/{pure['total_cases']}")
        print(f"  File: {pure['file']}")

        print(f"\nApplied Geographic Reasoning Track:")
        print(f"  Accuracy: {applied['overall_accuracy']:.2f}%")
        print(f"  Correct: {applied['total_correct']}/{applied['total_cases']}")
        print(f"  File: {applied['file']}")

        # Analysis
        analysis = comparison['analysis']
        print(f"\n{'─'*80}")
        print(f"Analysis:")
        print(f"  Difference: {analysis['accuracy_difference']:+.2f}%")
        print(f"  Model Strength: {analysis['model_strength'].replace('_', ' ').title()}")
        print(f"  Insight: {analysis['insight']}")

        # Tier rankings
        print(f"\n{'─'*80}")
        print(f"Tier Rankings (by accuracy):")
        for i, tier_info in enumerate(analysis['tier_rankings'], 1):
            track_label = "Pure GPS" if tier_info['track'] == 'pure_gps' else "Applied"
            print(f"  {i}. {tier_info['tier']} ({track_label}): {tier_info['accuracy']:.2f}%")

        # Strengths and weaknesses
        if analysis['strongest_tier']:
            strongest = analysis['strongest_tier']
            print(f"\nStrongest Area:")
            print(f"  {strongest['tier']} ({strongest['track'].replace('_', ' ').title()}): {strongest['accuracy']:.2f}%")

        if analysis['weakest_tier']:
            weakest = analysis['weakest_tier']
            print(f"\nWeakest Area:")
            print(f"  {weakest['tier']} ({weakest['track'].replace('_', ' ').title()}): {weakest['accuracy']:.2f}%")

        print(f"\n{'='*80}\n")

    def save_comparison(self, comparison: Dict, output_file: Optional[str] = None):
        """Save comparison to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = comparison['model'].replace('/', '_')
            output_file = f"comparison_tracks_{model_name}_{timestamp}.json"

        output_path = self.results_dir / output_file

        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"✓ Comparison saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Compare Pure GPS vs Applied track performance'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name to compare (default: latest results)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Results directory (default: results)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output filename for comparison (default: auto-generated)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available result files'
    )

    args = parser.parse_args()

    comparator = TrackComparator(results_dir=args.results_dir)

    if args.list:
        # List available files
        files = comparator.find_result_files(args.model)
        print("\nAvailable Pure GPS results:")
        for f in files['pure_gps'][:5]:
            print(f"  - {f.name}")
        print("\nAvailable Applied results:")
        for f in files['applied'][:5]:
            print(f"  - {f.name}")
        return

    # Compare tracks
    try:
        comparison = comparator.compare_latest(args.model)
        comparator.print_comparison(comparison)

        if args.output or not args.output:  # Always save
            comparator.save_comparison(comparison, args.output)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run evaluations first:")
        print("  python evaluation/evaluate_pure_gps.py --model <model_name>")
        print("  python evaluation/evaluate_applied.py --model <model_name>")
        sys.exit(1)


if __name__ == "__main__":
    main()
