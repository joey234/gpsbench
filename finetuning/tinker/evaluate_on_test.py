#!/usr/bin/env python3
"""
Evaluate finetuned model on the place_association test set.

Usage:
    python evaluate_on_test.py --checkpoint final
    python evaluate_on_test.py --checkpoint 000800
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
from difflib import SequenceMatcher
from tqdm import tqdm

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import tinker
from tinker import types
from transformers import AutoTokenizer


def extract_final_answer(text: str) -> str:
    """Extract the FINAL ANSWER from model output."""
    import re
    patterns = [
        r'FINAL ANSWER:\s*(.+?)(?:\n|$)',
        r'Final Answer:\s*(.+?)(?:\n|$)',
        r'final answer:\s*(.+?)(?:\n|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip()


def evaluate_location_name(prediction: str, ground_truth: str) -> float:
    """Evaluate location name prediction with fuzzy matching."""
    pred_lower = prediction.lower().strip()
    gt_lower = ground_truth.lower().strip()

    # Exact match
    if pred_lower == gt_lower:
        return 1.0

    # Check if prediction contains ground truth or vice versa
    if gt_lower in pred_lower or pred_lower in gt_lower:
        return 0.9

    # Fuzzy match
    ratio = SequenceMatcher(None, pred_lower, gt_lower).ratio()
    if ratio > 0.8:
        return ratio

    # Check city name only (ignore country)
    pred_parts = [p.strip() for p in pred_lower.split(',')]
    gt_parts = [p.strip() for p in gt_lower.split(',')]

    if pred_parts[0] == gt_parts[0]:
        return 0.95

    # Fuzzy match on city name
    city_ratio = SequenceMatcher(None, pred_parts[0], gt_parts[0]).ratio()
    if city_ratio > 0.8:
        return city_ratio * 0.9

    return 0.0


def load_test_data(data_path: Path) -> List[Dict]:
    """Load the place_association test set."""
    test_file = data_path / "track_applied" / "splits" / "task1_place_association_test.json"

    with open(test_file) as f:
        data = json.load(f)

    return data


def format_test_example(example: Dict) -> str:
    """Format a test example as a prompt."""
    coord_str = example.get('coordinate_string', '')

    prompt = (
        f"What location is at coordinates {coord_str}?\n\n"
        f"Provide your answer in the format: City, Region/State, Country "
        f"(e.g., 'Paris, Île-de-France, France' or 'Tokyo, Tokyo, Japan').\n\n"
        f"FINAL ANSWER: [your answer]"
    )

    return prompt


def get_ground_truth(example: Dict) -> str:
    """Get the expected answer from an example."""
    gt = example.get('ground_truth', {})
    location_name = gt.get('location_name', gt.get('name', 'Unknown'))
    country_code = gt.get('country_code', '')

    # Country code to name mapping (subset)
    COUNTRY_NAMES = {
        'US': 'United States', 'GB': 'United Kingdom', 'CN': 'China', 'IN': 'India',
        'JP': 'Japan', 'DE': 'Germany', 'FR': 'France', 'BR': 'Brazil', 'IT': 'Italy',
        'CA': 'Canada', 'RU': 'Russia', 'MX': 'Mexico', 'AU': 'Australia', 'ES': 'Spain',
        'GT': 'Guatemala', 'PH': 'Philippines', 'ID': 'Indonesia', 'TH': 'Thailand',
        'VN': 'Vietnam', 'MY': 'Malaysia', 'PK': 'Pakistan', 'BD': 'Bangladesh',
        'NG': 'Nigeria', 'EG': 'Egypt', 'ZA': 'South Africa', 'KE': 'Kenya',
        'AR': 'Argentina', 'CO': 'Colombia', 'PE': 'Peru', 'CL': 'Chile',
    }

    country_name = COUNTRY_NAMES.get(country_code, country_code)

    if country_name:
        return f"{location_name}, {country_name}"
    return location_name


def run_inference(
    sampler_path: str,
    model_name: str,
    prompts: List[str],
    max_tokens: int = 256,
) -> List[str]:
    """Run inference using Tinker SamplingClient API."""

    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create sampling client
    print(f"Creating sampling client with checkpoint: {sampler_path}")
    client = tinker.ServiceClient()
    sampling_client = client.create_sampling_client(
        model_path=sampler_path,
        base_model=model_name
    )

    system_msg = "You are an expert in GPS coordinates, geographic information systems, and spatial reasoning."

    responses = []
    for prompt in tqdm(prompts, desc="Running inference"):
        # Format as chat messages
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]

        # Tokenize
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt_text)

        # Create input
        model_input = types.ModelInput(chunks=[
            types.EncodedTextChunk(tokens=tokens, type='encoded_text')
        ])

        # Sample
        sampling_params = types.SamplingParams(max_tokens=max_tokens, temperature=0.1)
        response = sampling_client.sample(
            model_input, num_samples=1, sampling_params=sampling_params
        ).result()

        # Decode
        output_tokens = response.sequences[0].tokens
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        responses.append(output_text)

    return responses


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned model on test set")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='final',
        help='Checkpoint name (e.g., "final", "000800", "000400")'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/place_association_augmented',
        help='Directory containing checkpoints.jsonl'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='../../data',
        help='Path to GPSBench data directory'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen3-30B-A3B-Instruct-2507',
        help='Model name'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to evaluate (for quick testing)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Evaluating Finetuned Model on Test Set")
    print("=" * 70)
    print()

    # Load checkpoint info
    output_dir = Path(args.output_dir)
    checkpoints_file = output_dir / "checkpoints.jsonl"

    if not checkpoints_file.exists():
        print(f"Error: {checkpoints_file} not found")
        return

    # Find the requested checkpoint
    sampler_path = None
    with open(checkpoints_file) as f:
        for line in f:
            ckpt = json.loads(line)
            if ckpt['name'] == args.checkpoint:
                sampler_path = ckpt['sampler_path']  # Use sampler_path for inference
                print(f"Using checkpoint: {args.checkpoint}")
                print(f"  Epoch: {ckpt['epoch']}, Batch: {ckpt['batch']}")
                print(f"  Sampler path: {sampler_path}")
                break

    if not sampler_path:
        print(f"Error: Checkpoint '{args.checkpoint}' not found")
        print("Available checkpoints:")
        with open(checkpoints_file) as f:
            for line in f:
                ckpt = json.loads(line)
                print(f"  - {ckpt['name']} (epoch {ckpt['epoch']})")
        return

    print()

    # Load test data
    data_path = Path(args.data_path)
    print(f"Loading test data from {data_path}...")
    test_data = load_test_data(data_path)

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    print(f"Loaded {len(test_data)} test examples")
    print()

    # Prepare prompts and ground truths
    prompts = [format_test_example(ex) for ex in test_data]
    ground_truths = [get_ground_truth(ex) for ex in test_data]

    # Run inference
    print("Running inference...")
    responses = run_inference(
        sampler_path=sampler_path,
        model_name=args.model_name,
        prompts=prompts,
    )

    print()
    print("Evaluating responses...")

    # Evaluate
    correct = 0
    partial_scores = []
    results = []

    for i, (response, gt) in enumerate(zip(responses, ground_truths)):
        prediction = extract_final_answer(response)
        score = evaluate_location_name(prediction, gt)

        partial_scores.append(score)
        if score >= 0.8:
            correct += 1

        results.append({
            'index': i,
            'prediction': prediction,
            'ground_truth': gt,
            'score': score,
            'full_response': response
        })

    # Calculate metrics
    accuracy = correct / len(test_data) * 100
    avg_score = sum(partial_scores) / len(partial_scores) * 100

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Test samples: {len(test_data)}")
    print(f"Exact/Near-Exact Match (≥0.8): {correct} ({accuracy:.1f}%)")
    print(f"Average Score: {avg_score:.1f}%")
    print()

    # Show some examples
    print("=" * 70)
    print("Sample Predictions")
    print("=" * 70)

    # Show some correct and incorrect examples
    correct_examples = [r for r in results if r['score'] >= 0.8][:3]
    incorrect_examples = [r for r in results if r['score'] < 0.8][:3]

    print("\n✓ Correct predictions:")
    for r in correct_examples:
        print(f"  Pred: {r['prediction']}")
        print(f"  GT:   {r['ground_truth']}")
        print(f"  Score: {r['score']:.2f}")
        print()

    print("\n✗ Incorrect predictions:")
    for r in incorrect_examples:
        print(f"  Pred: {r['prediction']}")
        print(f"  GT:   {r['ground_truth']}")
        print(f"  Score: {r['score']:.2f}")
        print()

    # Save results
    results_file = output_dir / f"test_results_{args.checkpoint}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'test_samples': len(test_data),
            'accuracy': accuracy,
            'avg_score': avg_score,
            'results': results
        }, f, indent=2)

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
