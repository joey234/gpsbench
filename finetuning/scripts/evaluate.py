"""
Evaluation script for fine-tuned location prediction model.

This script evaluates the model on validation data and computes metrics:
- Exact match accuracy
- Distance-based metrics
- Robustness to noise
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import torch

from inference import load_model, predict_location


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points in kilometers.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def parse_location(location_str: str) -> Dict[str, str]:
    """
    Parse location string into components.

    Expected format: "City, Country" or "City"

    Args:
        location_str: Location string

    Returns:
        Dictionary with parsed components
    """
    parts = [p.strip() for p in location_str.split(',')]

    if len(parts) >= 2:
        return {
            'city': parts[0],
            'country': parts[1],
            'full': location_str
        }
    else:
        return {
            'city': location_str,
            'country': '',
            'full': location_str
        }


def compute_metrics(predictions: List[str], ground_truths: List[str]) -> Dict:
    """
    Compute evaluation metrics.

    Args:
        predictions: List of predicted locations
        ground_truths: List of ground truth locations

    Returns:
        Dictionary of metrics
    """
    n = len(predictions)

    # Exact match
    exact_match = sum(
        p.lower() == g.lower()
        for p, g in zip(predictions, ground_truths)
    ) / n

    # City match (first component)
    city_match = 0
    country_match = 0

    for pred, truth in zip(predictions, ground_truths):
        pred_parsed = parse_location(pred)
        truth_parsed = parse_location(truth)

        if pred_parsed['city'].lower() == truth_parsed['city'].lower():
            city_match += 1

        if pred_parsed['country'].lower() == truth_parsed['country'].lower():
            country_match += 1

    city_match /= n
    country_match /= n

    return {
        'exact_match': exact_match,
        'city_match': city_match,
        'country_match': country_match,
        'total_samples': n,
    }


def evaluate_by_noise_level(
    model,
    tokenizer,
    val_data: List[Dict],
) -> pd.DataFrame:
    """
    Evaluate model performance by noise level.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        val_data: Validation dataset

    Returns:
        DataFrame with results by noise level
    """
    # Group by noise level
    noise_groups = {}
    for example in val_data:
        noise_std = example['metadata']['noise_std']
        if noise_std not in noise_groups:
            noise_groups[noise_std] = []
        noise_groups[noise_std].append(example)

    results = []

    for noise_std in sorted(noise_groups.keys()):
        examples = noise_groups[noise_std]

        print(f"\nEvaluating noise level {noise_std:.6f}° ({len(examples)} examples)...")

        predictions = []
        ground_truths = []

        for example in tqdm(examples, desc=f"Noise {noise_std:.6f}"):
            # Extract coordinates
            lat = example['metadata']['noisy_lat']
            lon = example['metadata']['noisy_lon']

            # Predict
            pred = predict_location(model, tokenizer, lat, lon, temperature=0.0)
            predictions.append(pred)
            ground_truths.append(example['output'])

        # Compute metrics
        metrics = compute_metrics(predictions, ground_truths)
        metrics['noise_std'] = noise_std
        metrics['noise_meters'] = noise_std * 111000  # Approximate conversion

        results.append(metrics)

    return pd.DataFrame(results)


def evaluate_full(
    model,
    tokenizer,
    val_file: str,
    output_dir: str = None,
) -> Dict:
    """
    Full evaluation on validation set.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        val_file: Path to validation dataset JSON
        output_dir: Optional output directory for detailed results

    Returns:
        Dictionary of evaluation results
    """
    print(f"Loading validation data from {val_file}...")
    with open(val_file, 'r') as f:
        val_data = json.load(f)

    print(f"Loaded {len(val_data)} validation examples")

    # Overall evaluation
    print("\n=== Overall Evaluation ===")
    predictions = []
    ground_truths = []

    for example in tqdm(val_data, desc="Evaluating"):
        lat = example['metadata']['noisy_lat']
        lon = example['metadata']['noisy_lon']

        pred = predict_location(model, tokenizer, lat, lon, temperature=0.0)
        predictions.append(pred)
        ground_truths.append(example['output'])

    overall_metrics = compute_metrics(predictions, ground_truths)

    print(f"\nOverall Results:")
    print(f"  Exact Match: {overall_metrics['exact_match']:.2%}")
    print(f"  City Match: {overall_metrics['city_match']:.2%}")
    print(f"  Country Match: {overall_metrics['country_match']:.2%}")

    # By noise level
    print("\n=== Evaluation by Noise Level ===")
    noise_results = evaluate_by_noise_level(model, tokenizer, val_data)

    print("\nResults by Noise Level:")
    print(noise_results.to_string(index=False))

    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save overall metrics
        with open(output_path / 'overall_metrics.json', 'w') as f:
            json.dump(overall_metrics, f, indent=2)

        # Save noise-level results
        noise_results.to_csv(output_path / 'noise_level_results.csv', index=False)
        noise_results.to_json(output_path / 'noise_level_results.json', orient='records', indent=2)

        # Save detailed predictions
        detailed_results = []
        for i, example in enumerate(val_data):
            detailed_results.append({
                'latitude': example['metadata']['noisy_lat'],
                'longitude': example['metadata']['noisy_lon'],
                'noise_std': example['metadata']['noise_std'],
                'ground_truth': ground_truths[i],
                'prediction': predictions[i],
                'exact_match': predictions[i].lower() == ground_truths[i].lower(),
            })

        with open(output_path / 'detailed_predictions.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"\nResults saved to {output_dir}")

    return {
        'overall': overall_metrics,
        'by_noise': noise_results.to_dict('records'),
    }


def compare_with_baseline(
    finetuned_results: Dict,
    baseline_results_file: str = None,
):
    """
    Compare fine-tuned model with baseline results.

    Args:
        finetuned_results: Results from fine-tuned model
        baseline_results_file: Optional JSON file with baseline results
    """
    if baseline_results_file and Path(baseline_results_file).exists():
        with open(baseline_results_file, 'r') as f:
            baseline = json.load(f)

        print("\n=== Comparison with Baseline ===")
        print(f"Metric                    | Baseline    | Fine-tuned  | Improvement")
        print(f"--------------------------|-------------|-------------|-------------")

        ft_exact = finetuned_results['overall']['exact_match']
        bl_exact = baseline.get('exact_match', 0)
        diff_exact = ft_exact - bl_exact

        print(f"Exact Match               | {bl_exact:>10.2%} | {ft_exact:>10.2%} | {diff_exact:>+10.2%}")

        ft_city = finetuned_results['overall']['city_match']
        bl_city = baseline.get('city_match', 0)
        diff_city = ft_city - bl_city

        print(f"City Match                | {bl_city:>10.2%} | {ft_city:>10.2%} | {diff_city:>+10.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned location prediction model"
    )

    parser.add_argument(
        '--base-model',
        type=str,
        default='Qwen/Qwen2.5-0.5B',
        help='Base model name'
    )
    parser.add_argument(
        '--lora-path',
        type=str,
        default='../models/qwen-lora-location',
        help='Path to LoRA adapter'
    )
    parser.add_argument(
        '--val-file',
        type=str,
        default='../data/val_dataset.json',
        help='Validation dataset JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--baseline-file',
        type=str,
        help='Baseline results JSON for comparison'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Evaluate on a sample (for quick testing)'
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.base_model, args.lora_path)

    # Load validation data
    with open(args.val_file, 'r') as f:
        val_data = json.load(f)

    # Sample if requested
    if args.sample_size:
        import random
        val_data = random.sample(val_data, min(args.sample_size, len(val_data)))
        print(f"Using {len(val_data)} samples for quick evaluation")

    # Evaluate
    results = evaluate_full(
        model,
        tokenizer,
        args.val_file,
        output_dir=args.output_dir,
    )

    # Compare with baseline if provided
    if args.baseline_file:
        compare_with_baseline(results, args.baseline_file)


if __name__ == '__main__':
    main()
