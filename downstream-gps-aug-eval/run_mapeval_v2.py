#!/usr/bin/env python3
"""
MapEval evaluation with correct prompt format (v2).

Key improvements:
1. Uses official MapEval prompt format with numbered options
2. Better answer parsing
3. Support for v2 augmented dataset
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
from llm_client import LLMClient, LLMResponse


# Official MapEval prompt format
SYSTEM_PROMPT = """You are a highly intelligent assistant. Based on the given context, answer the multiple-choice question by selecting the correct option."""


def create_prompt(sample: Dict[str, Any]) -> str:
    """Create prompt using official MapEval format."""
    context = sample['context']
    question = sample['question']
    options = sample['options']

    prompt = f"""Context:
{context}

Question:
{question}

Options:
"""
    # Use numbered options (1-4) as per official format
    for i, opt in enumerate(options, start=1):
        if opt:
            prompt += f"{i}. {opt}\n"

    prompt += "\nSelect the best option by choosing its number."

    return prompt


def parse_answer(response_text: str, num_options: int) -> Optional[int]:
    """Parse the model's response to extract the answer (1-indexed)."""
    text = response_text.strip()

    # Pattern 1: Look for explicit "Option X" or "Answer: X" or just "X" at the end
    patterns = [
        r'(?:option|answer|choice)[:\s]*(\d)',  # "Option 1", "Answer: 2"
        r'\b([1-4])\b(?:\s*[.\)]|\s*$)',  # "1." or "1)" or "1" at end
        r'(?:select|choose|pick)[:\s]*(\d)',  # "select 1"
        r'^(\d)\s*$',  # Just a number alone
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Take the last match (usually the final answer)
            try:
                num = int(matches[-1])
                if 1 <= num <= num_options:
                    return num - 1  # Convert to 0-indexed
            except ValueError:
                continue

    # Pattern 2: Look for the option text being repeated
    # Sometimes models say "The answer is [full option text]"
    # This is harder to parse reliably, skip for now

    return None


def evaluate_sample(
    llm_client: LLMClient,
    sample: Dict[str, Any],
    use_augmented: bool = False
) -> Dict[str, Any]:
    """Evaluate a single sample."""
    if use_augmented and 'context_augmented' in sample:
        eval_sample = {**sample, 'context': sample['context_augmented']}
    else:
        eval_sample = sample

    prompt = create_prompt(eval_sample)
    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)

    num_options = len([o for o in sample['options'] if o])
    predicted_idx = parse_answer(response.text, num_options)

    # MapEval uses 1-indexed answers
    correct_idx_raw = sample['answer']
    correct_idx = correct_idx_raw - 1 if correct_idx_raw >= 1 else correct_idx_raw

    is_correct = predicted_idx == correct_idx

    return {
        'id': sample['id'],
        'classification': sample['classification'],
        'question': sample['question'][:100],
        'options': sample['options'],
        'correct_answer': correct_idx,
        'predicted_answer': predicted_idx,
        'is_correct': is_correct,
        'response_text': response.text,
        'error': response.error
    }


def run_evaluation(
    llm_client: LLMClient,
    samples: List[Dict[str, Any]],
    use_augmented: bool = False,
    max_workers: int = 5
) -> Dict[str, Any]:
    """Run evaluation on all samples."""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(evaluate_sample, llm_client, sample, use_augmented): i
            for i, sample in enumerate(samples)
        }

        desc = "Augmented" if use_augmented else "Base"
        with tqdm(total=len(samples), desc=f"{desc} eval") as pbar:
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

    results.sort(key=lambda x: x[0])
    results = [r[1] for r in results]

    # Metrics
    total = len(results)
    correct = sum(1 for r in results if r.get('is_correct', False))
    errors = sum(1 for r in results if r.get('error'))
    unparsed = sum(1 for r in results if r.get('predicted_answer') is None and not r.get('error'))

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
        'unparsed': unparsed,
        'by_classification': by_class
    }

    return {'metrics': metrics, 'results': results}


def main():
    parser = argparse.ArgumentParser(description='Run MapEval evaluation (v2)')
    parser.add_argument('--model', type=str, default='gpt-4.1', help='Model to use')
    parser.add_argument('--provider', type=str, default='auto', help='API provider')
    parser.add_argument('--variant', type=str, choices=['base', 'augmented', 'both'],
                        default='both', help='Which variant to evaluate')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit samples')
    parser.add_argument('--max-workers', type=int, default=5, help='Concurrent requests')
    parser.add_argument('--data-version', type=str, choices=['v1', 'v2', 'v3', 'v4'], default='v2',
                        help='Which augmented dataset version to use')
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace("/", "_").replace(":", "_")

    # Load data
    data_dir = Path(__file__).parent
    if args.data_version == 'v4':
        data_file = data_dir / "mapeval_gps_augmented_v4.json"
    elif args.data_version == 'v3':
        data_file = data_dir / "mapeval_gps_augmented_v3.json"
    elif args.data_version == 'v2':
        data_file = data_dir / "mapeval_gps_augmented_v2.json"
    else:
        data_file = data_dir / "mapeval_gps_augmented.json"

    if not data_file.exists():
        print(f"Error: {data_file} not found. Run the augmentation script first.")
        sys.exit(1)

    print(f"Loading dataset from {data_file}...")
    with open(data_file) as f:
        samples = json.load(f)

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Loaded {len(samples)} samples")

    # Initialize LLM
    print(f"\nInitializing {args.model}...")
    llm_client = LLMClient(
        provider=args.provider,
        model=args.model,
        temperature=0.0,
        max_tokens=1500,
        timeout=60
    )

    results_summary = {}

    # Base evaluation
    if args.variant in ['base', 'both']:
        print(f"\n{'='*60}")
        print("Running BASE evaluation (original context)")
        print('='*60)

        base_results = run_evaluation(
            llm_client, samples,
            use_augmented=False,
            max_workers=args.max_workers
        )

        base_output = output_dir / f"mapeval_v2_base_{model_safe}_{timestamp}.json"
        with open(base_output, 'w') as f:
            json.dump(base_results, f, indent=2)

        print(f"\nBase Results:")
        print(f"  Accuracy: {base_results['metrics']['accuracy']*100:.1f}% ({base_results['metrics']['correct']}/{base_results['metrics']['total_samples']})")
        print(f"  Unparsed: {base_results['metrics']['unparsed']}")
        print(f"  Saved to: {base_output}")

        results_summary['base'] = base_results['metrics']

    # Augmented evaluation
    if args.variant in ['augmented', 'both']:
        print(f"\n{'='*60}")
        print("Running AUGMENTED evaluation (with GPS coordinates)")
        print('='*60)

        aug_results = run_evaluation(
            llm_client, samples,
            use_augmented=True,
            max_workers=args.max_workers
        )

        aug_output = output_dir / f"mapeval_v2_augmented_{model_safe}_{timestamp}.json"
        with open(aug_output, 'w') as f:
            json.dump(aug_results, f, indent=2)

        print(f"\nAugmented Results:")
        print(f"  Accuracy: {aug_results['metrics']['accuracy']*100:.1f}% ({aug_results['metrics']['correct']}/{aug_results['metrics']['total_samples']})")
        print(f"  Unparsed: {aug_results['metrics']['unparsed']}")
        print(f"  Saved to: {aug_output}")

        results_summary['augmented'] = aug_results['metrics']

    # Comparison
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
        all_classes = set(results_summary['base']['by_classification'].keys()) | \
                      set(results_summary['augmented']['by_classification'].keys())

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
        comparison_output = output_dir / f"mapeval_v2_comparison_{model_safe}_{timestamp}.json"
        with open(comparison_output, 'w') as f:
            json.dump({
                'model': args.model,
                'timestamp': timestamp,
                'data_version': args.data_version,
                'num_samples': len(samples),
                'base': results_summary['base'],
                'augmented': results_summary['augmented'],
                'improvement': diff
            }, f, indent=2)
        print(f"\nComparison saved to: {comparison_output}")


if __name__ == "__main__":
    main()
