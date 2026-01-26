#!/usr/bin/env python3
"""
Evaluate LLMs on Applied Geographic Reasoning Track
Tests real-world geographic knowledge and location understanding
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import LLMClient
from track_config import get_track_info, get_total_tests, get_tier_display_name
from tqdm import tqdm

# Import evaluators from all layers
from layer_b_evaluators import get_evaluator as get_layer_b_evaluator
from layer_d_evaluators import get_evaluator as get_layer_d_evaluator
from layer_e_evaluators import get_evaluator as get_layer_e_evaluator


class AppliedGPSEvaluator:
    """Evaluator for Applied Geographic Reasoning Track"""

    def __init__(
        self,
        llm_client: LLMClient,
        data_dir: str = "../data",
        results_dir: str = "results"
    ):
        self.llm_client = llm_client
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Load track configuration
        self.track_config = get_track_info("applied")

    def load_task_data(self, layer: str, filename: str) -> list:
        """Load test cases from test split (not full dataset)"""
        # CRITICAL FIX: Use test split instead of full dataset
        task_name = filename.replace('.json', '')
        filepath = self.data_dir / layer / "splits" / f"{task_name}_test.json"

        if not filepath.exists():
            print(f"⚠️  Test split not found: {filepath}")
            print(f"⚠️  Falling back to full dataset (NOT RECOMMENDED - data leakage!)")
            filepath = self.data_dir / layer / filename

            if not filepath.exists():
                raise FileNotFoundError(f"Task file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Verify we're using test split (not full dataset)
        # Note: Legitimate test splits can have up to ~300 examples (30% of 1000)
        # Full datasets are typically 500-1000 examples
        if len(data) > 450:
            print(f"⚠️  WARNING: Loaded {len(data)} examples - this looks like full dataset!")
            print(f"⚠️  Test splits should have ~90-300 examples. Possible data leakage!")

        return data

    def evaluate_task(
        self,
        layer: str,
        task_name: str,
        filename: str,
        subtasks: Optional[list] = None,
        max_samples: Optional[int] = None,
        delay: float = 0.5
    ) -> Dict[str, Any]:
        """Evaluate a single task"""

        # Load test cases
        test_cases = self.load_task_data(layer, filename)

        # Filter by subtasks if specified
        if subtasks:
            test_cases = [
                tc for tc in test_cases
                if tc.get("subtask") in subtasks
            ]

        # Limit samples if requested
        if max_samples:
            test_cases = test_cases[:max_samples]

        print(f"\n{'='*80}")
        print(f"Task: {task_name}")
        print(f"Layer: {layer}")
        print(f"Test cases: {len(test_cases)}")
        if subtasks:
            print(f"Subtasks: {', '.join(subtasks)}")
        print(f"{'='*80}\n")

        # Get evaluator based on task (all tasks are in track_applied now)
        # Map task names to their evaluators
        tier1_tasks = ["place_association", "semantic_validation", "regional_knowledge", "context_grounding"]
        tier2_tasks = ["relative_position", "proximity", "route_analysis", "boundary_analysis", "spatial_patterns"]
        tier3_tasks = ["consistency", "hallucination_detection", "unit_confusion"]

        if task_name in tier1_tasks:
            evaluator = get_layer_b_evaluator(task_name)
        elif task_name in tier2_tasks:
            evaluator = get_layer_d_evaluator(task_name)
        elif task_name in tier3_tasks:
            evaluator = get_layer_e_evaluator(task_name)
        else:
            raise ValueError(f"Unknown task for Applied track: {task_name}")

        # Check if client supports batched inference
        supports_batching = hasattr(self.llm_client, 'query_batch')

        # Evaluate test cases
        results = []
        correct_count = 0
        error_count = 0
        total_tokens = 0
        total_latency = 0

        if supports_batching:
            # BATCHED PROCESSING (fast - for vLLM)
            print(f"Using batched inference for {len(test_cases)} examples...")

            # Create all prompts upfront
            all_prompts = []
            all_system_prompts = []
            for test_case in test_cases:
                system_prompt, user_prompt = evaluator.create_prompt(test_case)
                all_prompts.append(user_prompt)
                all_system_prompts.append(system_prompt)

            # SANITY CHECK: Print first prompt
            print(f"\n{'='*80}")
            print("SANITY CHECK: First Prompt Sample")
            print(f"{'='*80}")
            print(f"System Prompt:\n{all_system_prompts[0][:200]}...")
            print(f"\nUser Prompt:\n{all_prompts[0][:300]}...")
            print(f"{'='*80}\n")

            # Batch process all prompts at once
            import time
            start_time = time.time()

            # Check if all system prompts are the same (common case)
            if len(set(all_system_prompts)) == 1:
                # All same - pass single system prompt (FAST - single batch)
                responses = self.llm_client.query_batch(all_prompts, system_prompt=all_system_prompts[0])
            else:
                # Different system prompts - batch by system prompt (FAST - multiple batches)
                unique_prompts = list(set(all_system_prompts))
                print(f"Info: {len(unique_prompts)} different system prompts detected, batching by prompt type")

                # Group test cases by system prompt
                from collections import defaultdict
                prompt_groups = defaultdict(list)
                for idx, (user_prompt, sys_prompt) in enumerate(zip(all_prompts, all_system_prompts)):
                    prompt_groups[sys_prompt].append((idx, user_prompt))

                # Process each group in batch
                responses = [None] * len(all_prompts)  # Preallocate
                for sys_prompt, group in prompt_groups.items():
                    indices, prompts_in_group = zip(*group)
                    group_responses = self.llm_client.query_batch(list(prompts_in_group), system_prompt=sys_prompt)

                    # Place responses back in original order
                    for idx, response in zip(indices, group_responses):
                        responses[idx] = response

            batch_time = time.time() - start_time
            print(f"Batch inference completed in {batch_time:.2f}s ({len(test_cases)/batch_time:.1f} examples/sec)")

            # Evaluate all responses
            for idx, (test_case, response_text) in enumerate(zip(test_cases, responses)):
                try:
                    eval_result = evaluator.evaluate_response(response_text, test_case)

                    if eval_result['correct']:
                        correct_count += 1

                    # Save full prediction with ground truth for analysis
                    result_entry = {
                        'test_index': idx,
                        'correct': eval_result['correct'],
                        'prediction': response_text,  # FULL prediction
                        'ground_truth': test_case.get('ground_truth'),
                        'task': test_case.get('task'),
                        'tokens': 0,  # vLLM doesn't return token counts easily
                        'latency_ms': 0
                    }

                    # Add input information for context
                    if 'source_string' in test_case:
                        result_entry['input'] = test_case['source_string']
                    elif 'coordinates' in test_case:
                        result_entry['input'] = test_case['coordinates']
                    elif 'location' in test_case:
                        result_entry['input'] = test_case['location']
                    elif 'location_name' in test_case:
                        result_entry['input'] = test_case['location_name']

                    results.append(result_entry)

                except Exception as e:
                    error_count += 1
                    results.append({
                        'test_index': idx,
                        'error': str(e),
                        'correct': False,
                        'test_case_sample': str(test_case)[:200]  # Sample for debugging
                    })

        else:
            # SEQUENTIAL PROCESSING (slow - for API clients)
            print(f"Using sequential inference for {len(test_cases)} examples...")

            # SANITY CHECK: Print first prompt
            if test_cases:
                system_prompt, user_prompt = evaluator.create_prompt(test_cases[0])
                print(f"\n{'='*80}")
                print("SANITY CHECK: First Prompt Sample")
                print(f"{'='*80}")
                print(f"System Prompt:\n{system_prompt[:200]}...")
                print(f"\nUser Prompt:\n{user_prompt[:300]}...")
                print(f"{'='*80}\n")

            for test_case in tqdm(test_cases, desc=f"Evaluating {task_name}"):
                try:
                    # Create prompt
                    system_prompt, user_prompt = evaluator.create_prompt(test_case)

                    # Get LLM response
                    response = self.llm_client.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt
                    )

                    # Evaluate response
                    eval_result = evaluator.evaluate_response(response.text, test_case)

                    # Track metrics
                    if eval_result['correct']:
                        correct_count += 1

                    total_tokens += response.total_tokens
                    total_latency += response.latency_ms

                    # Save full prediction with ground truth for analysis
                    result_entry = {
                        'test_index': len(results),
                        'correct': eval_result['correct'],
                        'prediction': response.text,  # FULL prediction
                        'ground_truth': test_case.get('ground_truth'),
                        'task': test_case.get('task'),
                        'tokens': response.total_tokens,
                        'latency_ms': response.latency_ms
                    }

                    # Add input information for context
                    if 'source_string' in test_case:
                        result_entry['input'] = test_case['source_string']
                    elif 'coordinates' in test_case:
                        result_entry['input'] = test_case['coordinates']
                    elif 'location' in test_case:
                        result_entry['input'] = test_case['location']
                    elif 'location_name' in test_case:
                        result_entry['input'] = test_case['location_name']

                    results.append(result_entry)

                except Exception as e:
                    error_count += 1
                    results.append({
                        'test_index': len(results),
                        'error': str(e),
                        'correct': False,
                        'test_case_sample': str(test_case)[:200]  # Sample for debugging
                    })

                # Rate limiting
                if delay > 0:
                    import time
                    time.sleep(delay)

        # Calculate metrics
        accuracy = (correct_count / len(test_cases) * 100) if test_cases else 0
        avg_tokens = total_tokens / len(test_cases) if test_cases else 0
        avg_latency = total_latency / len(test_cases) if test_cases else 0

        task_results = {
            'task': task_name,
            'layer': layer,
            'file': filename,
            'subtasks': subtasks,
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(test_cases),
            'test_results': results,
            'metrics': {
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': len(test_cases),
                'error_count': error_count,
                'total_tokens': total_tokens,
                'avg_tokens_per_case': avg_tokens,
                'avg_latency_ms': avg_latency
            }
        }

        # Print summary
        print(f"\n{'─'*80}")
        print(f"Task: {task_name}")
        print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(test_cases)})")
        print(f"Errors: {error_count}")
        print(f"Avg Latency: {avg_latency:.0f}ms")
        print(f"Total Tokens: {total_tokens}")
        print(f"{'─'*80}\n")

        return task_results

    def evaluate_tier(
        self,
        tier_name: str,
        max_samples: Optional[int] = None,
        delay: float = 0.5
    ) -> Dict[str, Any]:
        """Evaluate all tasks in a tier"""

        tier_data = self.track_config[tier_name]
        tier_display = get_tier_display_name(tier_name)

        print(f"\n{'='*80}")
        print(f"Evaluating Tier: {tier_display}")
        print(f"Total tasks: {len(tier_data['tasks'])}")
        print(f"{'='*80}")

        tier_results = {
            'tier': tier_name,
            'tier_display': tier_display,
            'layer': tier_data['layer'],
            'timestamp': datetime.now().isoformat(),
            'tasks': {}
        }

        # Evaluate each task
        for task_name, task_info in tier_data['tasks'].items():
            try:
                task_results = self.evaluate_task(
                    layer=tier_data['layer'],
                    task_name=task_name,
                    filename=task_info['file'],
                    subtasks=task_info.get('subtasks'),
                    max_samples=max_samples,
                    delay=delay
                )
                tier_results['tasks'][task_name] = task_results
            except Exception as e:
                print(f"Error evaluating {task_name}: {e}")
                tier_results['tasks'][task_name] = {'error': str(e)}

        # Calculate tier metrics
        tier_results['metrics'] = self._calculate_tier_metrics(tier_results['tasks'])

        return tier_results

    def evaluate_all_tiers(
        self,
        max_samples: Optional[int] = None,
        delay: float = 0.5
    ) -> Dict[str, Any]:
        """Evaluate all tiers in Applied track"""

        print(f"\n{'='*80}")
        print(f"GPSBench Applied Geographic Reasoning Track Evaluation")
        print(f"Model: {self.llm_client.model}")
        print(f"Provider: {self.llm_client.provider}")
        print(f"Total tests: {get_total_tests('applied')}")
        print(f"{'='*80}")

        track_results = {
            'track': 'applied',
            'model': self.llm_client.model,
            'provider': self.llm_client.provider,
            'timestamp': datetime.now().isoformat(),
            'tiers': {}
        }

        # Evaluate each tier
        for tier_name in self.track_config.keys():
            try:
                tier_results = self.evaluate_tier(
                    tier_name=tier_name,
                    max_samples=max_samples,
                    delay=delay
                )
                track_results['tiers'][tier_name] = tier_results
            except Exception as e:
                print(f"Error evaluating tier {tier_name}: {e}")
                track_results['tiers'][tier_name] = {'error': str(e)}

        # Calculate overall metrics
        track_results['overall_metrics'] = self._calculate_overall_metrics(
            track_results['tiers']
        )

        return track_results

    def _calculate_tier_metrics(self, tasks: Dict) -> Dict:
        """Calculate aggregate metrics for a tier"""
        total_correct = sum(
            t['metrics']['correct_count']
            for t in tasks.values()
            if 'metrics' in t
        )
        total_cases = sum(
            t['metrics']['total_count']
            for t in tasks.values()
            if 'metrics' in t
        )

        return {
            'accuracy': (total_correct / total_cases * 100) if total_cases > 0 else 0,
            'total_correct': total_correct,
            'total_cases': total_cases
        }

    def _calculate_overall_metrics(self, tiers: Dict) -> Dict:
        """Calculate aggregate metrics for entire track"""
        total_correct = sum(
            tier['metrics']['total_correct']
            for tier in tiers.values()
            if 'metrics' in tier
        )
        total_cases = sum(
            tier['metrics']['total_cases']
            for tier in tiers.values()
            if 'metrics' in tier
        )

        tier_accuracies = {
            tier_name: tier['metrics']['accuracy']
            for tier_name, tier in tiers.items()
            if 'metrics' in tier
        }

        return {
            'overall_accuracy': (total_correct / total_cases * 100) if total_cases > 0 else 0,
            'total_correct': total_correct,
            'total_cases': total_cases,
            'tier_accuracies': tier_accuracies
        }

    def save_results(self, results: Dict, output_file: Optional[str] = None):
        """Save results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.llm_client.model.replace('/', '_')
            output_file = f"eval_applied_{model_name}_{timestamp}.json"

        output_path = self.results_dir / output_file

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLMs on Applied Geographic Reasoning Track'
    )

    # Model configuration
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'openrouter', 'auto'],
        default='auto',
        help='API provider (default: auto-detect from model name)'
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
        '--tier',
        type=str,
        choices=['tier_1_location_grounding', 'tier_2_spatial_relations', 'tier_3_robustness', 'all'],
        default='all',
        help='Tier to evaluate (default: all)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per task (default: all)'
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
    evaluator = AppliedGPSEvaluator(
        llm_client=llm_client,
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )

    # Run evaluation
    if args.tier == 'all':
        results = evaluator.evaluate_all_tiers(
            max_samples=args.max_samples,
            delay=args.delay
        )
    else:
        tier_results = evaluator.evaluate_tier(
            tier_name=args.tier,
            max_samples=args.max_samples,
            delay=args.delay
        )
        results = {
            'track': 'applied',
            'model': args.model,
            'provider': llm_client.provider,
            'timestamp': datetime.now().isoformat(),
            'tier': args.tier,
            'tier_results': tier_results
        }

    # Save results
    evaluator.save_results(results, args.output)

    # Print summary
    print(f"\n{'='*80}")
    print("Applied Track Evaluation Complete")
    print(f"{'='*80}")
    if 'overall_metrics' in results:
        metrics = results['overall_metrics']
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        print(f"Total Correct: {metrics['total_correct']}/{metrics['total_cases']}")
        print(f"\nBy Tier:")
        for tier_name, accuracy in metrics['tier_accuracies'].items():
            tier_display = get_tier_display_name(tier_name)
            print(f"  {tier_display}: {accuracy:.2f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
