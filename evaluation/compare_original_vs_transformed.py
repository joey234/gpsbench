#!/usr/bin/env python3
"""
Comparison evaluation: Original (clean) vs Transformed (Layer E) coordinates

This script tests the SAME locations with:
1. Original clean coordinates (from Layers A-D)
2. Transformed coordinates (Layer E - with noise, low precision, etc.)

The comparison reveals whether models truly understand coordinates or just memorized them:
- If model understands → performance should be similar on both
- If model only memorized → performance drops significantly with transformations

Features (Updated with Hierarchical Evaluation):
- Compares performance across 5 geographic levels (exact, city, region, country, continent)
- Tracks degradation patterns (e.g., exact → city, exact → none)
- Identifies which levels are most robust to transformations
- Provides detailed memorization failure analysis

Metrics:
Overall:
- Original accuracy: Performance on clean coordinates
- Transformed accuracy: Performance on noisy/degraded coordinates
- Robustness score: transformed_acc / original_acc (1.0 = perfect robustness)
- Performance drop: original_acc - transformed_acc (0.0 = no drop)

Hierarchical (NEW):
- Per-level accuracy: Separate accuracy for each geographic level
- Per-level robustness: Robustness score for each level independently
- Degradation patterns: How transformations affect different levels
  * Stable matches (exact/city/country remain stable)
  * Precision loss (exact → city, exact → country)
  * Complete failures (exact → none)

Example Output:
Hierarchical Level Comparison:
Level                   Original  Transformed   Robustness     Drop
────────────────────────────────────────────────────────────────────
Exact Location              52.3%        38.7%        0.740    13.6%
City                        68.3%        62.1%        0.909     6.2%
Region                      45.2%        41.3%        0.914     3.9%
Country                     82.7%        79.8%        0.965     2.9%
Continent                   91.4%        90.1%        0.986     1.3%

Interpretation:
- Country/continent knowledge is robust (>0.96 robustness)
- Exact location memory is fragile (0.74 robustness)
- Transformations degrade precision more than general knowledge
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from tqdm import tqdm
import time

import sys
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import LLMClient
from layer_e_evaluators import get_evaluator


class ComparisonEvaluator:
    """
    Evaluates models on original vs transformed coordinates
    """

    def __init__(
        self,
        llm_client: LLMClient,
        data_dir: str = "data",
        results_dir: str = "results"
    ):
        self.llm_client = llm_client
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_task1_comparison(
        self,
        max_samples: int = None,
        delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Compare performance on Task 1 (Noise Tolerance):
        - Original: Clean coordinates
        - Transformed: Noisy/degraded coordinates
        """
        print(f"\n{'='*80}")
        print(f"COMPARISON EVALUATION: Task 1 Noise Tolerance")
        print(f"Testing SAME locations: Original (clean) vs Transformed (noisy)")
        print(f"{'='*80}")

        # Load Layer E Task 1 tests (these have source tracking)
        task_file = self.data_dir / "layer_e" / "task1_noise_tolerance.json"
        with open(task_file, 'r', encoding='utf-8') as f:
            layer_e_tests = json.load(f)

        if max_samples:
            layer_e_tests = layer_e_tests[:max_samples]

        print(f"\nTest cases: {len(layer_e_tests)}")
        print(f"Model: {self.llm_client.model}")

        # Get evaluator for location identification
        from layer_e_evaluators import NoiseToleranceEvaluator
        evaluator = NoiseToleranceEvaluator()

        results = {
            'model': self.llm_client.model,
            'provider': self.llm_client.provider,
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(layer_e_tests),
            'comparisons': []
        }

        # Overall counters
        original_correct = 0
        transformed_correct = 0
        both_correct = 0
        both_wrong = 0
        only_original_correct = 0
        only_transformed_correct = 0

        # Hierarchical level counters
        hierarchical_stats = {
            'original': {
                'exact_location_match': 0,
                'city_match': 0,
                'region_match': 0,
                'country_match': 0,
                'continent_match': 0
            },
            'transformed': {
                'exact_location_match': 0,
                'city_match': 0,
                'region_match': 0,
                'country_match': 0,
                'continent_match': 0
            },
            'degradation_patterns': {
                'exact_to_city': 0,  # Original exact, transformed only city
                'exact_to_country': 0,  # Original exact, transformed only country
                'exact_to_none': 0,  # Original exact, transformed nothing
                'city_to_country': 0,  # Original city, transformed only country
                'stable_exact': 0,  # Both exact
                'stable_city': 0,  # Both city (but not exact)
                'stable_country': 0  # Both country (but not exact/city)
            }
        }

        # Evaluate each test on BOTH original and transformed
        for i, test in enumerate(tqdm(layer_e_tests, desc="Comparing")):
            # 1. TEST WITH ORIGINAL (CLEAN) COORDINATES
            original_question = test['source']['original_question']
            original_test = {
                'subtask': test['subtask'],
                'question': original_question,
                'ground_truth': test['ground_truth']
            }

            system_prompt_orig = "You are a geospatial AI assistant. Answer questions about locations based on coordinates."
            user_prompt_orig = f"""{original_question}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location name]

Example: FINAL ANSWER: New York City"""

            response_orig = self.llm_client.generate(user_prompt_orig, system_prompt_orig)
            eval_orig = evaluator.evaluate_response(response_orig.text, test)

            # 2. TEST WITH TRANSFORMED (NOISY) COORDINATES
            system_prompt_trans, user_prompt_trans = evaluator.create_prompt(test)
            response_trans = self.llm_client.generate(user_prompt_trans, system_prompt_trans)
            eval_trans = evaluator.evaluate_response(response_trans.text, test)

            # Track overall results
            orig_correct = eval_orig['correct']
            trans_correct = eval_trans['correct']

            if orig_correct:
                original_correct += 1
            if trans_correct:
                transformed_correct += 1

            if orig_correct and trans_correct:
                both_correct += 1
            elif not orig_correct and not trans_correct:
                both_wrong += 1
            elif orig_correct and not trans_correct:
                only_original_correct += 1
            elif not orig_correct and trans_correct:
                only_transformed_correct += 1

            # Track hierarchical level matches
            for level in ['exact_location_match', 'city_match', 'region_match', 'country_match', 'continent_match']:
                if eval_orig.get(level):
                    hierarchical_stats['original'][level] += 1
                if eval_trans.get(level):
                    hierarchical_stats['transformed'][level] += 1

            # Track degradation patterns
            orig_exact = eval_orig.get('exact_location_match', False)
            orig_city = eval_orig.get('city_match', False)
            orig_country = eval_orig.get('country_match', False)
            trans_exact = eval_trans.get('exact_location_match', False)
            trans_city = eval_trans.get('city_match', False)
            trans_country = eval_trans.get('country_match', False)

            if orig_exact and trans_exact:
                hierarchical_stats['degradation_patterns']['stable_exact'] += 1
            elif orig_exact and trans_city and not trans_exact:
                hierarchical_stats['degradation_patterns']['exact_to_city'] += 1
            elif orig_exact and trans_country and not trans_city:
                hierarchical_stats['degradation_patterns']['exact_to_country'] += 1
            elif orig_exact and not trans_city and not trans_country:
                hierarchical_stats['degradation_patterns']['exact_to_none'] += 1
            elif orig_city and not orig_exact and trans_city and not trans_exact:
                hierarchical_stats['degradation_patterns']['stable_city'] += 1
            elif orig_city and trans_country and not trans_city:
                hierarchical_stats['degradation_patterns']['city_to_country'] += 1
            elif orig_country and not orig_city and trans_country and not trans_city:
                hierarchical_stats['degradation_patterns']['stable_country'] += 1

            comparison = {
                'test_index': i,
                'location': test['true_location'],
                'subtask': test['subtask'],
                'transformation': test['source']['transformation'],
                'original': {
                    'question': original_question,
                    'correct': orig_correct,
                    'predicted': eval_orig.get('predicted', 'N/A'),
                    'tokens': response_orig.total_tokens,
                    'latency_ms': response_orig.latency_ms,
                    # Add hierarchical scores
                    'exact_location_match': eval_orig.get('exact_location_match', False),
                    'city_match': eval_orig.get('city_match', False),
                    'region_match': eval_orig.get('region_match', False),
                    'country_match': eval_orig.get('country_match', False),
                    'continent_match': eval_orig.get('continent_match', False),
                    'match_notes': eval_orig.get('match_notes', '')
                },
                'transformed': {
                    'question': test['question'],
                    'correct': trans_correct,
                    'predicted': eval_trans.get('predicted', 'N/A'),
                    'tokens': response_trans.total_tokens,
                    'latency_ms': response_trans.latency_ms,
                    # Add hierarchical scores
                    'exact_location_match': eval_trans.get('exact_location_match', False),
                    'city_match': eval_trans.get('city_match', False),
                    'region_match': eval_trans.get('region_match', False),
                    'country_match': eval_trans.get('country_match', False),
                    'continent_match': eval_trans.get('continent_match', False),
                    'match_notes': eval_trans.get('match_notes', '')
                },
                'expected': test['ground_truth']['location']
            }

            results['comparisons'].append(comparison)

            # Rate limiting
            if delay > 0 and i < len(layer_e_tests) - 1:
                time.sleep(delay)

        # Calculate metrics
        total = len(layer_e_tests)
        original_acc = (original_correct / total) * 100
        transformed_acc = (transformed_correct / total) * 100
        robustness_score = transformed_acc / original_acc if original_acc > 0 else 0
        performance_drop = original_acc - transformed_acc

        # Calculate hierarchical level accuracies
        hierarchical_accuracies = {
            'original': {},
            'transformed': {},
            'robustness_by_level': {}
        }

        for level in ['exact_location_match', 'city_match', 'region_match', 'country_match', 'continent_match']:
            orig_level_acc = (hierarchical_stats['original'][level] / total) * 100
            trans_level_acc = (hierarchical_stats['transformed'][level] / total) * 100
            level_robustness = trans_level_acc / orig_level_acc if orig_level_acc > 0 else 0

            hierarchical_accuracies['original'][level] = orig_level_acc
            hierarchical_accuracies['transformed'][level] = trans_level_acc
            hierarchical_accuracies['robustness_by_level'][level] = level_robustness

        results['metrics'] = {
            'original_accuracy': original_acc,
            'transformed_accuracy': transformed_acc,
            'robustness_score': robustness_score,
            'performance_drop_pct': performance_drop,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'only_original_correct': only_original_correct,
            'only_transformed_correct': only_transformed_correct,
            'breakdown': {
                'both_correct_pct': (both_correct / total) * 100,
                'both_wrong_pct': (both_wrong / total) * 100,
                'memorization_failure_pct': (only_original_correct / total) * 100,
                'unexpected_success_pct': (only_transformed_correct / total) * 100
            },
            'hierarchical_stats': hierarchical_stats,
            'hierarchical_accuracies': hierarchical_accuracies
        }

        # Print summary
        print(f"\n{'─'*80}")
        print(f"COMPARISON RESULTS")
        print(f"{'─'*80}")
        print(f"Overall Accuracy (Exact Location):")
        print(f"  Original (clean):             {original_acc:.2f}% ({original_correct}/{total})")
        print(f"  Transformed (noisy):          {transformed_acc:.2f}% ({transformed_correct}/{total})")
        print(f"  Robustness score:             {robustness_score:.3f} (1.0 = perfect)")
        print(f"  Performance drop:             {performance_drop:.2f}%")

        print(f"\nHierarchical Level Comparison:")
        print(f"{'Level':<20s} {'Original':>12s} {'Transformed':>12s} {'Robustness':>12s} {'Drop':>8s}")
        print(f"{'─'*68}")

        level_names = {
            'exact_location_match': 'Exact Location',
            'city_match': 'City',
            'region_match': 'Region',
            'country_match': 'Country',
            'continent_match': 'Continent'
        }

        for level, display_name in level_names.items():
            orig_acc_level = hierarchical_accuracies['original'][level]
            trans_acc_level = hierarchical_accuracies['transformed'][level]
            robust_level = hierarchical_accuracies['robustness_by_level'][level]
            drop_level = orig_acc_level - trans_acc_level

            print(f"{display_name:<20s} {orig_acc_level:>11.1f}% {trans_acc_level:>11.1f}% {robust_level:>11.3f} {drop_level:>7.1f}%")

        print(f"\nDegradation Patterns:")
        print(f"  Stable exact match:         {hierarchical_stats['degradation_patterns']['stable_exact']} ({hierarchical_stats['degradation_patterns']['stable_exact']/total*100:.1f}%)")
        print(f"  Stable city match:          {hierarchical_stats['degradation_patterns']['stable_city']} ({hierarchical_stats['degradation_patterns']['stable_city']/total*100:.1f}%)")
        print(f"  Stable country match:       {hierarchical_stats['degradation_patterns']['stable_country']} ({hierarchical_stats['degradation_patterns']['stable_country']/total*100:.1f}%)")
        print(f"  Exact → City:               {hierarchical_stats['degradation_patterns']['exact_to_city']} ({hierarchical_stats['degradation_patterns']['exact_to_city']/total*100:.1f}%) ← Precision loss")
        print(f"  Exact → Country:            {hierarchical_stats['degradation_patterns']['exact_to_country']} ({hierarchical_stats['degradation_patterns']['exact_to_country']/total*100:.1f}%) ← Severe degradation")
        print(f"  Exact → None:               {hierarchical_stats['degradation_patterns']['exact_to_none']} ({hierarchical_stats['degradation_patterns']['exact_to_none']/total*100:.1f}%) ← Complete failure")
        print(f"  City → Country:             {hierarchical_stats['degradation_patterns']['city_to_country']} ({hierarchical_stats['degradation_patterns']['city_to_country']/total*100:.1f}%)")

        print(f"\nOverall Breakdown:")
        print(f"  Both correct:               {both_correct} ({results['metrics']['breakdown']['both_correct_pct']:.1f}%)")
        print(f"  Both wrong:                 {both_wrong} ({results['metrics']['breakdown']['both_wrong_pct']:.1f}%)")
        print(f"  Only original correct:      {only_original_correct} ({results['metrics']['breakdown']['memorization_failure_pct']:.1f}%) ← Memorization failure")
        print(f"  Only transformed correct:   {only_transformed_correct} ({results['metrics']['breakdown']['unexpected_success_pct']:.1f}%)")
        print(f"{'─'*80}")

        # Add interpretation guidance
        print(f"\n{'='*80}")
        print(f"INTERPRETATION GUIDE")
        print(f"{'='*80}")
        print(f"Robustness Score Interpretation:")
        print(f"  > 0.90: Excellent robustness (model understands coordinates)")
        print(f"  0.70-0.90: Good robustness (some memorization dependency)")
        print(f"  0.50-0.70: Moderate robustness (significant memorization)")
        print(f"  < 0.50: Poor robustness (heavy memorization dependency)")
        print(f"\nCurrent Score: {robustness_score:.3f} - ", end="")
        if robustness_score > 0.90:
            print("Excellent (model understands coordinates)")
        elif robustness_score > 0.70:
            print("Good (some memorization dependency)")
        elif robustness_score > 0.50:
            print("Moderate (significant memorization)")
        else:
            print("Poor (heavy memorization dependency)")
        print(f"{'='*80}")

        return results

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save comparison results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results['model'].replace('/', '_')
            filename = f"comparison_original_vs_transformed_{model_name}_{timestamp}.json"

        filepath = self.results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(
        description='Compare model performance on original vs transformed coordinates'
    )

    # Model configuration
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'openrouter'],
        default='openai',
        help='API provider (default: openai)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-3.5-turbo',
        help='Model name (default: gpt-3.5-turbo)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for generation (default: 0.0)'
    )

    # Evaluation configuration
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to test (default: all)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )

    # Output configuration
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory (default: data)'
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
        help='Output filename (default: auto-generated)'
    )

    args = parser.parse_args()

    # Initialize LLM client
    print(f"Initializing {args.provider} client with model: {args.model}")
    llm_client = LLMClient(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature
    )

    # Initialize evaluator
    evaluator = ComparisonEvaluator(
        llm_client=llm_client,
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )

    # Run comparison
    results = evaluator.evaluate_task1_comparison(
        max_samples=args.max_samples,
        delay=args.delay
    )

    # Save results
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
