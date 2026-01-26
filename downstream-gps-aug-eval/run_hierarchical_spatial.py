#!/usr/bin/env python3
"""
Run hierarchical_spatial evaluation comparing base vs GPS-augmented prompts.

Usage:
    python run_hierarchical_spatial.py --model gpt-4.1
    python run_hierarchical_spatial.py --model gpt-4.1 --runs 10  # Multiple runs like the paper
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
from llm_client import LLMClient


SYSTEM_PROMPT = """You are a geography expert. Answer questions about directions between locations accurately and concisely.

When asked about intercardinal directions, respond with exactly one of these options:
- Northeast
- Northwest
- Southeast
- Southwest

Provide your reasoning briefly, then state your final answer clearly."""


INTERCARDINAL_DIRECTIONS = ['northeast', 'northwest', 'southeast', 'southwest']


def create_base_prompt(question: Dict[str, Any]) -> str:
    """Create base prompt without GPS coordinates."""
    return f"What is the intercardinal direction from {question['city_a']} to {question['city_b']}?"


def create_augmented_prompt(question: Dict[str, Any]) -> str:
    """Create prompt with GPS coordinates."""
    if question.get('gps_augmentation'):
        return (
            f"{question['gps_augmentation']}\n\n"
            f"What is the intercardinal direction from {question['city_a']} to {question['city_b']}?"
        )
    else:
        # Fallback to base if no GPS data
        return create_base_prompt(question)


def parse_direction(response_text: str) -> Optional[str]:
    """Parse the intercardinal direction from response."""
    text_lower = response_text.lower()

    # Look for direction words
    for direction in INTERCARDINAL_DIRECTIONS:
        if direction in text_lower:
            # Check if it's the final answer (usually last mention)
            pass

    # Find all mentions of directions
    found_directions = []
    for direction in INTERCARDINAL_DIRECTIONS:
        # Find all positions
        start = 0
        while True:
            pos = text_lower.find(direction, start)
            if pos == -1:
                break
            found_directions.append((pos, direction))
            start = pos + 1

    if found_directions:
        # Return the last mentioned direction (usually the final answer)
        found_directions.sort(key=lambda x: x[0])
        return found_directions[-1][1].capitalize()

    return None


def evaluate_question(
    llm_client: LLMClient,
    question: Dict[str, Any],
    use_augmented: bool = False
) -> Dict[str, Any]:
    """Evaluate a single question."""
    if use_augmented:
        prompt = create_augmented_prompt(question)
    else:
        prompt = create_base_prompt(question)

    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)

    predicted = parse_direction(response.text)
    correct = question['correct_answer']

    is_correct = predicted and predicted.lower() == correct.lower()

    return {
        'id': question['id'],
        'city_a': question['city_a'],
        'city_b': question['city_b'],
        'correct_answer': correct,
        'predicted_answer': predicted,
        'is_correct': is_correct,
        'bias_type': question['bias_type'],
        'response_text': response.text,
        'prompt': prompt,
        'error': response.error
    }


def run_evaluation(
    llm_client: LLMClient,
    questions: List[Dict[str, Any]],
    use_augmented: bool = False,
    num_runs: int = 1
) -> Dict[str, Any]:
    """Run evaluation, optionally multiple times like the paper."""
    all_results = []

    desc = "Augmented" if use_augmented else "Base"

    for run in range(num_runs):
        run_results = []
        for q in tqdm(questions, desc=f"{desc} run {run+1}/{num_runs}"):
            result = evaluate_question(llm_client, q, use_augmented)
            result['run'] = run + 1
            run_results.append(result)
        all_results.extend(run_results)

    # Calculate metrics
    total = len(all_results)
    correct = sum(1 for r in all_results if r['is_correct'])

    # Per-question accuracy (averaged across runs)
    question_scores = {}
    for r in all_results:
        qid = r['id']
        if qid not in question_scores:
            question_scores[qid] = {'correct': 0, 'total': 0}
        question_scores[qid]['total'] += 1
        if r['is_correct']:
            question_scores[qid]['correct'] += 1

    # Accuracy by bias type
    by_bias = {}
    for r in all_results:
        bias = r['bias_type']
        if bias not in by_bias:
            by_bias[bias] = {'correct': 0, 'total': 0}
        by_bias[bias]['total'] += 1
        if r['is_correct']:
            by_bias[bias]['correct'] += 1

    for bias in by_bias:
        by_bias[bias]['accuracy'] = by_bias[bias]['correct'] / by_bias[bias]['total']

    metrics = {
        'total_evaluations': total,
        'num_questions': len(questions),
        'num_runs': num_runs,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'by_bias_type': by_bias,
        'per_question': {
            qid: scores['correct'] / scores['total']
            for qid, scores in question_scores.items()
        }
    }

    return {
        'metrics': metrics,
        'results': all_results
    }


def main():
    parser = argparse.ArgumentParser(description='Run hierarchical_spatial evaluation')
    parser.add_argument('--model', type=str, default='gpt-4.1', help='Model to use')
    parser.add_argument('--provider', type=str, default='auto', help='API provider')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per question (paper uses 10)')
    parser.add_argument('--variant', type=str, choices=['base', 'augmented', 'both'],
                        default='both', help='Which variant to evaluate')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace("/", "_").replace(":", "_")

    # Load augmented dataset (has both base and GPS info)
    data_file = Path(__file__).parent / "hierarchical_spatial_gps.json"
    if not data_file.exists():
        print("Error: Run augment_hierarchical_spatial.py first")
        sys.exit(1)

    with open(data_file) as f:
        data = json.load(f)

    questions = data['questions']
    print(f"Loaded {len(questions)} questions")

    # Initialize LLM
    print(f"\nInitializing {args.model}...")
    llm_client = LLMClient(
        provider=args.provider,
        model=args.model,
        temperature=0.0,
        max_tokens=500,
        timeout=30
    )

    results_summary = {}

    # Run base evaluation
    if args.variant in ['base', 'both']:
        print(f"\n{'='*60}")
        print("Running BASE evaluation (no GPS coordinates)")
        print('='*60)

        base_results = run_evaluation(
            llm_client, questions,
            use_augmented=False,
            num_runs=args.runs
        )

        base_output = output_dir / f"hierarchical_base_{model_safe}_{timestamp}.json"
        with open(base_output, 'w') as f:
            json.dump(base_results, f, indent=2)

        print(f"\nBase Results:")
        print(f"  Accuracy: {base_results['metrics']['accuracy']*100:.1f}% ({base_results['metrics']['correct']}/{base_results['metrics']['total_evaluations']})")
        print(f"  Saved to: {base_output}")

        results_summary['base'] = base_results['metrics']

    # Run augmented evaluation
    if args.variant in ['augmented', 'both']:
        print(f"\n{'='*60}")
        print("Running AUGMENTED evaluation (with GPS coordinates)")
        print('='*60)

        aug_results = run_evaluation(
            llm_client, questions,
            use_augmented=True,
            num_runs=args.runs
        )

        aug_output = output_dir / f"hierarchical_augmented_{model_safe}_{timestamp}.json"
        with open(aug_output, 'w') as f:
            json.dump(aug_results, f, indent=2)

        print(f"\nAugmented Results:")
        print(f"  Accuracy: {aug_results['metrics']['accuracy']*100:.1f}% ({aug_results['metrics']['correct']}/{aug_results['metrics']['total_evaluations']})")
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

        print(f"\nBy Bias Type:")
        all_biases = set(results_summary['base']['by_bias_type'].keys()) | set(results_summary['augmented']['by_bias_type'].keys())

        print(f"  {'Bias Type':<15} {'Base':>10} {'Augmented':>10} {'Diff':>10}")
        print(f"  {'-'*45}")

        for bias in sorted(all_biases):
            base_bias = results_summary['base']['by_bias_type'].get(bias, {})
            aug_bias = results_summary['augmented']['by_bias_type'].get(bias, {})

            base_bias_acc = base_bias.get('accuracy', 0) * 100
            aug_bias_acc = aug_bias.get('accuracy', 0) * 100
            bias_diff = aug_bias_acc - base_bias_acc

            print(f"  {bias:<15} {base_bias_acc:>9.1f}% {aug_bias_acc:>9.1f}% {bias_diff:>+9.1f}%")

        # Per-question comparison for interesting cases
        print(f"\nPer-Question Accuracy Changes:")
        base_pq = results_summary['base']['per_question']
        aug_pq = results_summary['augmented']['per_question']

        changes = []
        for qid in base_pq:
            base_q_acc = base_pq[qid]
            aug_q_acc = aug_pq.get(qid, 0)
            change = aug_q_acc - base_q_acc
            q_info = next(q for q in questions if q['id'] == qid)
            changes.append((qid, q_info['city_a'], q_info['city_b'], base_q_acc, aug_q_acc, change))

        # Show biggest improvements and drops
        changes.sort(key=lambda x: x[5], reverse=True)

        print(f"\n  Top Improvements:")
        for qid, ca, cb, ba, aa, ch in changes[:5]:
            if ch > 0:
                print(f"    Q{qid}: {ca} -> {cb}: {ba*100:.0f}% -> {aa*100:.0f}% ({ch*100:+.0f}%)")

        print(f"\n  Top Drops:")
        for qid, ca, cb, ba, aa, ch in changes[-5:]:
            if ch < 0:
                print(f"    Q{qid}: {ca} -> {cb}: {ba*100:.0f}% -> {aa*100:.0f}% ({ch*100:+.0f}%)")

        # Save comparison
        comparison_output = output_dir / f"hierarchical_comparison_{model_safe}_{timestamp}.json"
        with open(comparison_output, 'w') as f:
            json.dump({
                'model': args.model,
                'timestamp': timestamp,
                'num_questions': len(questions),
                'num_runs': args.runs,
                'base': results_summary['base'],
                'augmented': results_summary['augmented'],
                'improvement': diff
            }, f, indent=2)
        print(f"\nComparison saved to: {comparison_output}")


if __name__ == "__main__":
    main()
