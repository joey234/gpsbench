#!/usr/bin/env python3
"""
Run MapEval evaluation comparing base vs GPS-augmented contexts.

Usage:
    # Run both base and augmented evaluations
    python run_mapeval.py --model gpt-4.1

    # Run only base or augmented
    python run_mapeval.py --model gpt-4.1 --variant base
    python run_mapeval.py --model gpt-4.1 --variant augmented

    # Limit samples for testing
    python run_mapeval.py --model gpt-4.1 --max-samples 10
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
from llm_client import LLMClient, LLMResponse


SYSTEM_PROMPT = """You are an expert at answering questions about locations, navigation, and travel planning based on provided context.

Read the context carefully and answer the multiple choice question by selecting the correct option.

IMPORTANT: Your response must end with exactly one of these formats:
FINAL ANSWER: [A]
FINAL ANSWER: [B]
FINAL ANSWER: [C]
FINAL ANSWER: [D]

Where A, B, C, D correspond to the options provided (first option = A, second = B, etc.)."""


def create_prompt(sample: Dict[str, Any]) -> str:
    """Create a prompt for a MapEval sample."""
    context = sample['context']
    question = sample['question']
    options = sample['options']

    # Format options with letters
    options_text = ""
    for i, opt in enumerate(options):
        if opt:  # Skip empty options
            letter = chr(ord('A') + i)
            options_text += f"{letter}. {opt}\n"

    prompt = f"""Context:
{context}

Question: {question}

Options:
{options_text}
Think through this step by step, then provide your final answer."""

    return prompt


def parse_answer(response_text: str, num_options: int) -> Optional[int]:
    """Parse the model's response to extract the answer index."""
    import re

    # Look for FINAL ANSWER: [X] pattern
    match = re.search(r'FINAL ANSWER:\s*\[([A-D])\]', response_text, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        idx = ord(letter) - ord('A')
        if 0 <= idx < num_options:
            return idx

    # Fallback: look for just a letter at the end
    match = re.search(r'\b([A-D])\b\s*$', response_text.strip(), re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        idx = ord(letter) - ord('A')
        if 0 <= idx < num_options:
            return idx

    return None


def evaluate_sample(
    llm_client: LLMClient,
    sample: Dict[str, Any],
    use_augmented: bool = False
) -> Dict[str, Any]:
    """Evaluate a single sample."""
    # Use augmented context if available and requested
    if use_augmented and 'context_augmented' in sample:
        eval_sample = {**sample, 'context': sample['context_augmented']}
    else:
        eval_sample = sample

    prompt = create_prompt(eval_sample)
    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)

    # Parse answer
    num_options = len([o for o in sample['options'] if o])
    predicted_idx = parse_answer(response.text, num_options)

    # MapEval uses 1-indexed answers, convert to 0-indexed for comparison
    correct_idx_raw = sample['answer']
    correct_idx = correct_idx_raw - 1 if correct_idx_raw >= 1 else correct_idx_raw

    is_correct = predicted_idx == correct_idx

    return {
        'id': sample['id'],
        'classification': sample['classification'],
        'question': sample['question'],
        'options': sample['options'],
        'correct_answer': correct_idx,
        'predicted_answer': predicted_idx,
        'is_correct': is_correct,
        'response_text': response.text,
        'prompt_tokens': response.prompt_tokens,
        'completion_tokens': response.completion_tokens,
        'latency_ms': response.latency_ms,
        'error': response.error
    }


def run_evaluation(
    llm_client: LLMClient,
    samples: List[Dict[str, Any]],
    use_augmented: bool = False,
    max_workers: int = 5,
    output_dir: Path = None
) -> Dict[str, Any]:
    """Run evaluation on all samples."""
    results = []

    # Process samples with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(evaluate_sample, llm_client, sample, use_augmented): i
            for i, sample in enumerate(samples)
        }

        with tqdm(total=len(samples), desc=f"{'Augmented' if use_augmented else 'Base'} eval") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    results.append((idx, {
                        'id': samples[idx]['id'],
                        'error': str(e),
                        'is_correct': False
                    }))
                pbar.update(1)

    # Sort by original index
    results.sort(key=lambda x: x[0])
    results = [r[1] for r in results]

    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r.get('is_correct', False))
    errors = sum(1 for r in results if r.get('error'))

    # Metrics by classification
    by_class = {}
    for r in results:
        cls = r.get('classification', 'unknown')
        if cls not in by_class:
            by_class[cls] = {'total': 0, 'correct': 0}
        by_class[cls]['total'] += 1
        if r.get('is_correct', False):
            by_class[cls]['correct'] += 1

    for cls in by_class:
        by_class[cls]['accuracy'] = by_class[cls]['correct'] / by_class[cls]['total'] if by_class[cls]['total'] > 0 else 0

    metrics = {
        'total_samples': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'errors': errors,
        'by_classification': by_class
    }

    return {
        'metrics': metrics,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Run MapEval evaluation')
    parser.add_argument('--model', type=str, default='gpt-4.1', help='Model to use')
    parser.add_argument('--provider', type=str, default='auto', help='API provider')
    parser.add_argument('--variant', type=str, choices=['base', 'augmented', 'both'],
                        default='both', help='Which variant to evaluate')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit number of samples')
    parser.add_argument('--max-workers', type=int, default=5, help='Concurrent requests')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace("/", "_").replace(":", "_")

    # Load data
    data_dir = Path(__file__).parent

    # Check if augmented data exists
    augmented_file = data_dir / "mapeval_gps_augmented.json"
    if not augmented_file.exists():
        print("Error: GPS-augmented data not found. Run augment_mapeval_with_gps.py first.")
        sys.exit(1)

    print(f"Loading augmented dataset from {augmented_file}...")
    with open(augmented_file) as f:
        samples = json.load(f)

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Loaded {len(samples)} samples")

    # Initialize LLM client
    print(f"\nInitializing {args.model}...")
    llm_client = LLMClient(
        provider=args.provider,
        model=args.model,
        temperature=0.0,
        max_tokens=1500,
        timeout=60
    )

    results_summary = {}

    # Run base evaluation
    if args.variant in ['base', 'both']:
        print(f"\n{'='*60}")
        print("Running BASE evaluation (original context)")
        print('='*60)

        base_results = run_evaluation(
            llm_client,
            samples,
            use_augmented=False,
            max_workers=args.max_workers,
            output_dir=output_dir
        )

        # Save base results
        base_output = output_dir / f"mapeval_base_{model_safe}_{timestamp}.json"
        with open(base_output, 'w') as f:
            json.dump(base_results, f, indent=2)

        print(f"\nBase Results:")
        print(f"  Accuracy: {base_results['metrics']['accuracy']*100:.1f}% ({base_results['metrics']['correct']}/{base_results['metrics']['total_samples']})")
        print(f"  Saved to: {base_output}")

        results_summary['base'] = base_results['metrics']

    # Run augmented evaluation
    if args.variant in ['augmented', 'both']:
        print(f"\n{'='*60}")
        print("Running AUGMENTED evaluation (with GPS coordinates)")
        print('='*60)

        aug_results = run_evaluation(
            llm_client,
            samples,
            use_augmented=True,
            max_workers=args.max_workers,
            output_dir=output_dir
        )

        # Save augmented results
        aug_output = output_dir / f"mapeval_augmented_{model_safe}_{timestamp}.json"
        with open(aug_output, 'w') as f:
            json.dump(aug_results, f, indent=2)

        print(f"\nAugmented Results:")
        print(f"  Accuracy: {aug_results['metrics']['accuracy']*100:.1f}% ({aug_results['metrics']['correct']}/{aug_results['metrics']['total_samples']})")
        print(f"  Saved to: {aug_output}")

        results_summary['augmented'] = aug_results['metrics']

    # Print comparison if both were run
    if args.variant == 'both':
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print('='*60)

        base_acc = results_summary['base']['accuracy'] * 100
        aug_acc = results_summary['augmented']['accuracy'] * 100
        diff = aug_acc - base_acc

        print(f"\nOverall Accuracy:")
        print(f"  Base:      {base_acc:.1f}%")
        print(f"  Augmented: {aug_acc:.1f}%")
        print(f"  Difference: {diff:+.1f}%")

        print(f"\nBy Classification:")
        all_classes = set(results_summary['base']['by_classification'].keys()) | set(results_summary['augmented']['by_classification'].keys())

        print(f"  {'Class':<15} {'Base':>10} {'Augmented':>10} {'Diff':>10}")
        print(f"  {'-'*45}")

        for cls in sorted(all_classes):
            base_cls = results_summary['base']['by_classification'].get(cls, {})
            aug_cls = results_summary['augmented']['by_classification'].get(cls, {})

            base_cls_acc = base_cls.get('accuracy', 0) * 100
            aug_cls_acc = aug_cls.get('accuracy', 0) * 100
            cls_diff = aug_cls_acc - base_cls_acc

            print(f"  {cls:<15} {base_cls_acc:>9.1f}% {aug_cls_acc:>9.1f}% {cls_diff:>+9.1f}%")

        # Save comparison
        comparison_output = output_dir / f"mapeval_comparison_{model_safe}_{timestamp}.json"
        with open(comparison_output, 'w') as f:
            json.dump({
                'model': args.model,
                'timestamp': timestamp,
                'num_samples': len(samples),
                'base': results_summary['base'],
                'augmented': results_summary['augmented'],
                'improvement': {
                    'overall': diff,
                    'by_classification': {
                        cls: (results_summary['augmented']['by_classification'].get(cls, {}).get('accuracy', 0) -
                              results_summary['base']['by_classification'].get(cls, {}).get('accuracy', 0)) * 100
                        for cls in all_classes
                    }
                }
            }, f, indent=2)
        print(f"\nComparison saved to: {comparison_output}")


if __name__ == "__main__":
    main()
