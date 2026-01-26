#!/usr/bin/env python3
"""
Unified Comparison Evaluation: Original vs Transformed - All Tasks

This script provides comprehensive robustness testing across multiple GPSBench tasks:
- Layer E Task 1: Location Identification (Memorization Robustness)
- Layer C Task 1: Distance Calculation (Computational Robustness)
- Layer C Task 2: Bearing Computation (Computational Robustness)
- Layer D Task 1: Spatial Ordering (Reasoning Robustness)
- Layer E Task 2: Missing Data (Memorization Robustness)

The comparison reveals whether models:
- Truly understand geospatial concepts (robust performance)
- Rely on memorized patterns (fragile performance)
- Use computational formulas vs pattern matching

Usage:
    # Single task
    python evaluation/unified_comparison.py --model gpt-4 --task distance_calculation

    # All tasks
    python evaluation/unified_comparison.py --model gpt-4 --task all

    # Generate comprehensive report
    python evaluation/unified_comparison.py --model gpt-4 --task all --generate-report
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tqdm import tqdm
import time
import sys

sys.path.insert(0, str(Path(__file__).parent))

from llm_client import LLMClient
from layer_e_evaluators import NoiseToleranceEvaluator
from layer_c_evaluators import DistanceCalculationEvaluator, BearingComputationEvaluator
from layer_d_evaluators import RelativePositionEvaluator


class UnifiedComparisonEvaluator:
    """
    Unified evaluator for original vs transformed comparison across all tasks
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

    def compare_location_identification(
        self,
        max_samples: int = None,
        delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Layer E Task 1: Location Identification
        Tests memorization robustness to coordinate noise
        """
        print(f"\n{'='*80}")
        print(f"TASK 1: LOCATION IDENTIFICATION (Layer E Task 1)")
        print(f"Testing: Memorization robustness to coordinate noise")
        print(f"{'='*80}")

        # Load Layer E Task 1 tests
        task_file = self.data_dir / "layer_e" / "task1_noise_tolerance.json"
        with open(task_file, 'r', encoding='utf-8') as f:
            tests = json.load(f)

        if max_samples:
            tests = tests[:max_samples]

        print(f"\nTest cases: {len(tests)}")
        print(f"Model: {self.llm_client.model}")

        evaluator = NoiseToleranceEvaluator()
        results = self._run_hierarchical_comparison(
            tests, evaluator, "location_identification", delay
        )

        return results

    def compare_distance_calculation(
        self,
        max_samples: int = None,
        delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Layer C Task 1: Distance Calculation
        Tests computational robustness (Haversine formula)
        """
        print(f"\n{'='*80}")
        print(f"TASK 2: DISTANCE CALCULATION (Layer C Task 1)")
        print(f"Testing: Computational robustness to coordinate noise")
        print(f"{'='*80}")

        # Load Layer C Task 1 tests
        task_file = self.data_dir / "layer_c" / "task1_distance_calculation.json"
        with open(task_file, 'r', encoding='utf-8') as f:
            tests = json.load(f)

        if max_samples:
            tests = tests[:max_samples]

        print(f"\nTest cases: {len(tests)}")
        print(f"Model: {self.llm_client.model}")

        evaluator = DistanceCalculationEvaluator()
        results = self._run_numerical_comparison(
            tests, evaluator, "distance_calculation", delay
        )

        return results

    def compare_bearing_computation(
        self,
        max_samples: int = None,
        delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Layer C Task 2: Bearing Computation
        Tests computational robustness (bearing formulas)
        """
        print(f"\n{'='*80}")
        print(f"TASK 3: BEARING COMPUTATION (Layer C Task 2)")
        print(f"Testing: Computational robustness to coordinate noise")
        print(f"{'='*80}")

        # Load Layer C Task 2 tests
        task_file = self.data_dir / "layer_c" / "task2_bearing_computation.json"
        with open(task_file, 'r', encoding='utf-8') as f:
            tests = json.load(f)

        if max_samples:
            tests = tests[:max_samples]

        print(f"\nTest cases: {len(tests)}")
        print(f"Model: {self.llm_client.model}")

        evaluator = BearingComputationEvaluator()
        results = self._run_numerical_comparison(
            tests, evaluator, "bearing_computation", delay
        )

        return results

    def compare_spatial_ordering(
        self,
        max_samples: int = None,
        delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Layer D Task 1: Spatial Ordering
        Tests reasoning robustness to coordinate noise
        """
        print(f"\n{'='*80}")
        print(f"TASK 4: SPATIAL ORDERING (Layer D Task 1)")
        print(f"Testing: Reasoning robustness to coordinate noise")
        print(f"{'='*80}")

        # Load Layer D Task 1 tests (ordering subtask only)
        task_file = self.data_dir / "layer_d" / "task1_relative_position.json"
        with open(task_file, 'r', encoding='utf-8') as f:
            all_tests = json.load(f)

        # Filter to ordering subtask with source tracking
        tests = [t for t in all_tests if t.get('subtask') == 'ordering' and 'source' in t]

        if max_samples:
            tests = tests[:max_samples]

        print(f"\nTest cases: {len(tests)}")
        print(f"Model: {self.llm_client.model}")

        evaluator = RelativePositionEvaluator()
        results = self._run_ordering_comparison(
            tests, evaluator, "spatial_ordering", delay
        )

        return results

    def compare_missing_data(
        self,
        max_samples: int = None,
        delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Layer E Task 2: Missing Data
        Tests memorization robustness to incomplete information
        """
        print(f"\n{'='*80}")
        print(f"TASK 5: MISSING DATA (Layer E Task 2)")
        print(f"Testing: Memorization robustness to incomplete information")
        print(f"{'='*80}")

        # Load Layer E Task 2 tests
        task_file = self.data_dir / "layer_e" / "task2_missing_data.json"
        with open(task_file, 'r', encoding='utf-8') as f:
            tests = json.load(f)

        if max_samples:
            tests = tests[:max_samples]

        print(f"\nTest cases: {len(tests)}")
        print(f"Model: {self.llm_client.model}")

        # Import MissingDataEvaluator
        from layer_e_evaluators import MissingDataEvaluator
        evaluator = MissingDataEvaluator()

        results = self._run_missing_data_comparison(
            tests, evaluator, "missing_data", delay
        )

        return results

    def _run_hierarchical_comparison(
        self,
        tests: List[Dict],
        evaluator: Any,
        task_name: str,
        delay: float
    ) -> Dict[str, Any]:
        """
        Run comparison for hierarchical tasks (location identification)
        """
        results = {
            'model': self.llm_client.model,
            'provider': self.llm_client.provider,
            'task': task_name,
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(tests),
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
                'stable_exact': 0,
                'stable_city': 0,
                'stable_country': 0,
                'exact_to_city': 0,
                'exact_to_country': 0,
                'exact_to_none': 0,
                'city_to_country': 0
            }
        }

        # Evaluate each test
        for i, test in enumerate(tqdm(tests, desc="Comparing")):
            # Test with original (clean) coordinates
            original_question = test['source']['original_question']
            system_prompt = "You are a geospatial AI assistant. Answer questions about locations based on coordinates."
            user_prompt_orig = f"""{original_question}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location name]

Example: FINAL ANSWER: New York City"""

            response_orig = self.llm_client.generate(user_prompt_orig, system_prompt)
            eval_orig = evaluator.evaluate_response(response_orig.text, test)

            # Test with transformed (noisy) coordinates
            system_prompt_trans, user_prompt_trans = evaluator.create_prompt(test)
            response_trans = self.llm_client.generate(user_prompt_trans, system_prompt_trans)
            eval_trans = evaluator.evaluate_response(response_trans.text, test)

            # Track results
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

            # Track hierarchical levels
            for level in ['exact_location_match', 'city_match', 'region_match', 'country_match', 'continent_match']:
                if eval_orig.get(level):
                    hierarchical_stats['original'][level] += 1
                if eval_trans.get(level):
                    hierarchical_stats['transformed'][level] += 1

            # Track degradation patterns
            self._update_degradation_patterns(
                hierarchical_stats['degradation_patterns'],
                eval_orig, eval_trans
            )

            comparison = {
                'test_index': i,
                'location': test.get('true_location', 'N/A'),
                'subtask': test.get('subtask', 'N/A'),
                'transformation': test['source']['transformation'],
                'original': {
                    'question': original_question,
                    'correct': orig_correct,
                    'predicted': eval_orig.get('predicted', 'N/A'),
                    'exact_location_match': eval_orig.get('exact_location_match', False),
                    'city_match': eval_orig.get('city_match', False),
                    'country_match': eval_orig.get('country_match', False),
                    'continent_match': eval_orig.get('continent_match', False),
                },
                'transformed': {
                    'question': test['question'],
                    'correct': trans_correct,
                    'predicted': eval_trans.get('predicted', 'N/A'),
                    'exact_location_match': eval_trans.get('exact_location_match', False),
                    'city_match': eval_trans.get('city_match', False),
                    'country_match': eval_trans.get('country_match', False),
                    'continent_match': eval_trans.get('continent_match', False),
                },
                'expected': test['ground_truth'].get('location', 'N/A')
            }

            results['comparisons'].append(comparison)

            if delay > 0 and i < len(tests) - 1:
                time.sleep(delay)

        # Calculate metrics
        results['metrics'] = self._calculate_hierarchical_metrics(
            len(tests), original_correct, transformed_correct,
            both_correct, both_wrong, only_original_correct, only_transformed_correct,
            hierarchical_stats
        )

        self._print_hierarchical_results(results['metrics'], task_name)

        return results

    def _run_numerical_comparison(
        self,
        tests: List[Dict],
        evaluator: Any,
        task_name: str,
        delay: float
    ) -> Dict[str, Any]:
        """
        Run comparison for numerical tasks (distance, bearing)
        """
        results = {
            'model': self.llm_client.model,
            'provider': self.llm_client.provider,
            'task': task_name,
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(tests),
            'comparisons': []
        }

        # Counters
        original_correct = 0
        transformed_correct = 0
        both_correct = 0
        both_wrong = 0
        only_original_correct = 0
        only_transformed_correct = 0

        # Evaluate each test
        for i, test in enumerate(tqdm(tests, desc="Comparing")):
            # Test with original (clean) coordinates
            original_test = {
                'task': test.get('task', task_name),
                'point_a': test['source']['original_point_a'],
                'point_b': test['source']['original_point_b'],
                'ground_truth': test['ground_truth']
            }

            system_prompt_orig, user_prompt_orig = evaluator.create_prompt(original_test)
            response_orig = self.llm_client.generate(user_prompt_orig, system_prompt_orig)
            eval_orig = evaluator.evaluate_response(response_orig.text, original_test)

            # Test with transformed (noisy) coordinates
            transformed_test = {
                'task': test.get('task', task_name),
                'point_a': test['noisy_version']['point_a'],
                'point_b': test['noisy_version']['point_b'],
                'ground_truth': test['noisy_version']['ground_truth']
            }

            system_prompt_trans, user_prompt_trans = evaluator.create_prompt(transformed_test)
            response_trans = self.llm_client.generate(user_prompt_trans, system_prompt_trans)
            eval_trans = evaluator.evaluate_response(response_trans.text, transformed_test)

            # Track results
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

            comparison = {
                'test_index': i,
                'transformation': test['source']['transformation'],
                'noise_meters': test['noisy_version'].get('noise_meters', 'N/A'),
                'original': {
                    'correct': orig_correct,
                    'predicted': eval_orig.get('predicted', 'N/A'),
                    'expected': eval_orig.get('expected', 'N/A'),
                },
                'transformed': {
                    'correct': trans_correct,
                    'predicted': eval_trans.get('predicted', 'N/A'),
                    'expected': eval_trans.get('expected', 'N/A'),
                }
            }

            results['comparisons'].append(comparison)

            if delay > 0 and i < len(tests) - 1:
                time.sleep(delay)

        # Calculate metrics
        results['metrics'] = self._calculate_basic_metrics(
            len(tests), original_correct, transformed_correct,
            both_correct, both_wrong, only_original_correct, only_transformed_correct
        )

        self._print_basic_results(results['metrics'], task_name)

        return results

    def _run_ordering_comparison(
        self,
        tests: List[Dict],
        evaluator: Any,
        task_name: str,
        delay: float
    ) -> Dict[str, Any]:
        """
        Run comparison for ordering tasks
        """
        results = {
            'model': self.llm_client.model,
            'provider': self.llm_client.provider,
            'task': task_name,
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(tests),
            'comparisons': []
        }

        # Counters
        original_correct = 0
        transformed_correct = 0
        both_correct = 0
        both_wrong = 0
        only_original_correct = 0
        only_transformed_correct = 0

        # Evaluate each test
        for i, test in enumerate(tqdm(tests, desc="Comparing")):
            # Test with original (clean) coordinates
            original_test = {
                'subtask': 'ordering',
                'question': test['question'],  # Add question field
                'points': test['source']['original_points'],
                'ground_truth': {
                    'correct_order': test['source']['original_order'],
                    'dimension': test['source']['dimension']
                }
            }

            system_prompt_orig, user_prompt_orig = evaluator.create_prompt(original_test)
            response_orig = self.llm_client.generate(user_prompt_orig, system_prompt_orig)
            eval_orig = evaluator.evaluate_response(response_orig.text, original_test)

            # Test with transformed (noisy) coordinates
            transformed_test = {
                'subtask': 'ordering',
                'question': test['question'],  # Add question field
                'points': test['noisy_version']['points'],
                'ground_truth': test['noisy_version']['ground_truth']
            }

            system_prompt_trans, user_prompt_trans = evaluator.create_prompt(transformed_test)
            response_trans = self.llm_client.generate(user_prompt_trans, system_prompt_trans)
            eval_trans = evaluator.evaluate_response(response_trans.text, transformed_test)

            # Track results
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

            comparison = {
                'test_index': i,
                'transformation': test['source']['transformation'],
                'noise_meters': test['noisy_version'].get('noise_meters', 'N/A'),
                'dimension': test['ground_truth']['dimension'],
                'original': {
                    'correct': orig_correct,
                    'predicted': eval_orig.get('predicted', 'N/A'),
                    'expected': test['source']['original_order'],
                },
                'transformed': {
                    'correct': trans_correct,
                    'predicted': eval_trans.get('predicted', 'N/A'),
                    'expected': test['noisy_version']['ground_truth']['correct_order'],
                }
            }

            results['comparisons'].append(comparison)

            if delay > 0 and i < len(tests) - 1:
                time.sleep(delay)

        # Calculate metrics
        results['metrics'] = self._calculate_basic_metrics(
            len(tests), original_correct, transformed_correct,
            both_correct, both_wrong, only_original_correct, only_transformed_correct
        )

        self._print_basic_results(results['metrics'], task_name)

        return results

    def _run_missing_data_comparison(
        self,
        tests: List[Dict],
        evaluator: Any,
        task_name: str,
        delay: float
    ) -> Dict[str, Any]:
        """
        Run comparison for missing data tasks
        """
        results = {
            'model': self.llm_client.model,
            'provider': self.llm_client.provider,
            'task': task_name,
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(tests),
            'comparisons': []
        }

        # Counters
        complete_correct = 0
        missing_correct = 0
        both_correct = 0
        both_wrong = 0
        only_complete_correct = 0
        only_missing_correct = 0

        # Evaluate each test
        for i, test in enumerate(tqdm(tests, desc="Comparing")):
            # Test with complete data (original)
            # Use 'coordinate_component_recovery' subtask since it expects lat/lon in ground_truth
            complete_test = {
                'subtask': 'coordinate_component_recovery',
                'question': test['complete_version']['question'],
                'ground_truth': test['complete_version']['ground_truth']
            }

            system_prompt_comp, user_prompt_comp = evaluator.create_prompt(complete_test)
            response_comp = self.llm_client.generate(user_prompt_comp, system_prompt_comp)
            eval_comp = evaluator.evaluate_response(response_comp.text, complete_test)

            # Test with missing/corrupted data (transformed)
            system_prompt_miss, user_prompt_miss = evaluator.create_prompt(test)
            response_miss = self.llm_client.generate(user_prompt_miss, system_prompt_miss)
            eval_miss = evaluator.evaluate_response(response_miss.text, test)

            # Track results
            comp_correct = eval_comp['correct']
            miss_correct = eval_miss['correct']

            if comp_correct:
                complete_correct += 1
            if miss_correct:
                missing_correct += 1

            if comp_correct and miss_correct:
                both_correct += 1
            elif not comp_correct and not miss_correct:
                both_wrong += 1
            elif comp_correct and not miss_correct:
                only_complete_correct += 1
            elif not comp_correct and miss_correct:
                only_missing_correct += 1

            comparison = {
                'test_index': i,
                'subtask': test['subtask'],
                'transformation_type': test.get('transformation_type', 'N/A'),
                'complete': {
                    'correct': comp_correct,
                    'predicted': eval_comp.get('predicted', 'N/A'),
                },
                'missing': {
                    'correct': miss_correct,
                    'predicted': eval_miss.get('predicted', 'N/A'),
                }
            }

            results['comparisons'].append(comparison)

            if delay > 0 and i < len(tests) - 1:
                time.sleep(delay)

        # Calculate metrics
        results['metrics'] = self._calculate_basic_metrics(
            len(tests), complete_correct, missing_correct,
            both_correct, both_wrong, only_complete_correct, only_missing_correct
        )

        self._print_basic_results(results['metrics'], task_name,
                                 label_orig="Complete", label_trans="Missing",
                                 is_memorization_task=True)

        return results

    def _update_degradation_patterns(
        self,
        patterns: Dict,
        eval_orig: Dict,
        eval_trans: Dict
    ):
        """Update degradation pattern counters"""
        orig_exact = eval_orig.get('exact_location_match', False)
        orig_city = eval_orig.get('city_match', False)
        orig_country = eval_orig.get('country_match', False)
        trans_exact = eval_trans.get('exact_location_match', False)
        trans_city = eval_trans.get('city_match', False)
        trans_country = eval_trans.get('country_match', False)

        if orig_exact and trans_exact:
            patterns['stable_exact'] += 1
        elif orig_exact and trans_city and not trans_exact:
            patterns['exact_to_city'] += 1
        elif orig_exact and trans_country and not trans_city:
            patterns['exact_to_country'] += 1
        elif orig_exact and not trans_city and not trans_country:
            patterns['exact_to_none'] += 1
        elif orig_city and not orig_exact and trans_city and not trans_exact:
            patterns['stable_city'] += 1
        elif orig_city and trans_country and not trans_city:
            patterns['city_to_country'] += 1
        elif orig_country and not orig_city and trans_country and not trans_city:
            patterns['stable_country'] += 1

    def _calculate_hierarchical_metrics(
        self,
        total: int,
        original_correct: int,
        transformed_correct: int,
        both_correct: int,
        both_wrong: int,
        only_original_correct: int,
        only_transformed_correct: int,
        hierarchical_stats: Dict
    ) -> Dict[str, Any]:
        """Calculate metrics for hierarchical tasks"""
        original_acc = (original_correct / total) * 100
        transformed_acc = (transformed_correct / total) * 100
        robustness_score = transformed_acc / original_acc if original_acc > 0 else 0
        performance_drop = original_acc - transformed_acc

        # Calculate hierarchical accuracies
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

        return {
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

    def _calculate_basic_metrics(
        self,
        total: int,
        original_correct: int,
        transformed_correct: int,
        both_correct: int,
        both_wrong: int,
        only_original_correct: int,
        only_transformed_correct: int
    ) -> Dict[str, Any]:
        """Calculate metrics for non-hierarchical tasks"""
        original_acc = (original_correct / total) * 100
        transformed_acc = (transformed_correct / total) * 100
        robustness_score = transformed_acc / original_acc if original_acc > 0 else 0
        performance_drop = original_acc - transformed_acc

        return {
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
                'computation_failure_pct': (only_original_correct / total) * 100,
                'unexpected_success_pct': (only_transformed_correct / total) * 100
            }
        }

    def _print_hierarchical_results(self, metrics: Dict, task_name: str):
        """Print results for hierarchical tasks"""
        print(f"\n{'─'*80}")
        print(f"COMPARISON RESULTS: {task_name.upper()}")
        print(f"{'─'*80}")
        print(f"Overall Accuracy (Exact Location):")
        print(f"  Original (clean):             {metrics['original_accuracy']:.2f}%")
        print(f"  Transformed (noisy):          {metrics['transformed_accuracy']:.2f}%")
        print(f"  Robustness score:             {metrics['robustness_score']:.3f}")
        print(f"  Performance drop:             {metrics['performance_drop_pct']:.2f}%")

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
            orig_acc = metrics['hierarchical_accuracies']['original'][level]
            trans_acc = metrics['hierarchical_accuracies']['transformed'][level]
            robust = metrics['hierarchical_accuracies']['robustness_by_level'][level]
            drop = orig_acc - trans_acc

            print(f"{display_name:<20s} {orig_acc:>11.1f}% {trans_acc:>11.1f}% {robust:>11.3f} {drop:>7.1f}%")

        print(f"{'─'*80}")

    def _print_basic_results(self, metrics: Dict, task_name: str,
                           label_orig: str = "Original", label_trans: str = "Transformed",
                           is_memorization_task: bool = False):
        """Print results for non-hierarchical tasks"""
        print(f"\n{'─'*80}")
        print(f"COMPARISON RESULTS: {task_name.upper()}")
        print(f"{'─'*80}")
        print(f"Accuracy:")
        print(f"  {label_orig:20s}:     {metrics['original_accuracy']:.2f}%")
        print(f"  {label_trans:20s}:     {metrics['transformed_accuracy']:.2f}%")

        # Use different label for memorization tasks
        if is_memorization_task:
            print(f"  Memorization score:           {metrics['robustness_score']:.3f}")
            print(f"  Performance drop:             {metrics['performance_drop_pct']:.2f}%")
            print(f"\n  Interpretation:")
            score = metrics['robustness_score']
            if score > 0.8:
                print(f"    → Strong memorization: Model memorized these coordinates")
            elif score > 0.5:
                print(f"    → Moderate memorization: Model has partial knowledge")
            else:
                print(f"    → Weak memorization: Model doesn't know these locations")
        else:
            print(f"  Robustness score:             {metrics['robustness_score']:.3f}")
            print(f"  Performance drop:             {metrics['performance_drop_pct']:.2f}%")

        print(f"\nBreakdown:")
        print(f"  Both correct:                 {metrics['breakdown']['both_correct_pct']:.1f}%")
        print(f"  Both wrong:                   {metrics['breakdown']['both_wrong_pct']:.1f}%")
        print(f"  Only {label_orig.lower()} correct:       {metrics['breakdown']['computation_failure_pct']:.1f}%")
        print(f"  Only {label_trans.lower()} correct:   {metrics['breakdown']['unexpected_success_pct']:.1f}%")
        print(f"{'─'*80}")

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save comparison results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results['model'].replace('/', '_')
            task = results['task']
            filename = f"comparison_{task}_{model_name}_{timestamp}.json"

        filepath = self.results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(
        description='Unified comparison evaluation across all GPSBench tasks'
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

    # Task selection
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'all',
            'location_identification',
            'distance_calculation',
            'bearing_computation',
            'spatial_ordering',
            'missing_data'
        ],
        default='all',
        help='Task to evaluate (default: all)'
    )

    # Evaluation configuration
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to test per task (default: all)'
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
        '--generate-report',
        action='store_true',
        help='Generate comprehensive report with all tasks'
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
    evaluator = UnifiedComparisonEvaluator(
        llm_client=llm_client,
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )

    # Run comparisons based on task selection
    all_results = {}

    if args.task == 'all' or args.task == 'location_identification':
        results = evaluator.compare_location_identification(
            max_samples=args.max_samples,
            delay=args.delay
        )
        evaluator.save_results(results)
        all_results['location_identification'] = results

    if args.task == 'all' or args.task == 'distance_calculation':
        results = evaluator.compare_distance_calculation(
            max_samples=args.max_samples,
            delay=args.delay
        )
        evaluator.save_results(results)
        all_results['distance_calculation'] = results

    if args.task == 'all' or args.task == 'bearing_computation':
        results = evaluator.compare_bearing_computation(
            max_samples=args.max_samples,
            delay=args.delay
        )
        evaluator.save_results(results)
        all_results['bearing_computation'] = results

    if args.task == 'all' or args.task == 'spatial_ordering':
        results = evaluator.compare_spatial_ordering(
            max_samples=args.max_samples,
            delay=args.delay
        )
        evaluator.save_results(results)
        all_results['spatial_ordering'] = results

    if args.task == 'all' or args.task == 'missing_data':
        results = evaluator.compare_missing_data(
            max_samples=args.max_samples,
            delay=args.delay
        )
        evaluator.save_results(results)
        all_results['missing_data'] = results

    # Generate comprehensive report if requested
    if args.generate_report and len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE ROBUSTNESS REPORT")
        print(f"{'='*80}")
        print(f"\nModel: {args.model}")
        print(f"Tasks evaluated: {len(all_results)}")
        print()

        print(f"{'Task':<30s} {'Original':>12s} {'Transformed':>12s} {'Robustness':>12s}")
        print(f"{'─'*70}")

        for task_name, task_results in all_results.items():
            metrics = task_results['metrics']
            print(f"{task_name.replace('_', ' ').title():<30s} "
                  f"{metrics['original_accuracy']:>11.1f}% "
                  f"{metrics['transformed_accuracy']:>11.1f}% "
                  f"{metrics['robustness_score']:>11.3f}")

        print(f"{'='*80}")


if __name__ == "__main__":
    main()
