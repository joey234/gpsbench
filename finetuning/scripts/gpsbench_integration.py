"""
Integration script to evaluate fine-tuned model on GPSBench tasks.

This script allows you to run the fine-tuned LoRA model on GPSBench
benchmark tasks and compare with OpenAI/OpenRouter baselines.
"""

import json
import sys
from pathlib import Path
import argparse
from typing import Dict, List

# Add parent directory to path to import GPSBench modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from inference import load_model, predict_location


class FinetunedLocationProvider:
    """
    Provider class for fine-tuned location prediction model.

    This class wraps the fine-tuned model to be compatible with
    GPSBench's evaluation framework.
    """

    def __init__(self, base_model: str, lora_path: str):
        """
        Initialize the fine-tuned model provider.

        Args:
            base_model: Base model name
            lora_path: Path to LoRA adapter
        """
        print(f"Loading fine-tuned model: {base_model} + {lora_path}")
        self.model, self.tokenizer = load_model(base_model, lora_path)
        self.model_name = f"finetuned-{Path(base_model).name}"

    def predict(self, latitude: float, longitude: float) -> str:
        """
        Predict location from coordinates.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            Predicted location string
        """
        return predict_location(
            self.model,
            self.tokenizer,
            latitude,
            longitude,
            temperature=0.1
        )

    def __call__(self, latitude: float, longitude: float) -> str:
        """Allow provider to be called directly."""
        return self.predict(latitude, longitude)


def load_gpsbench_task(task_file: str) -> List[Dict]:
    """
    Load a GPSBench task file.

    Args:
        task_file: Path to task JSON file

    Returns:
        List of task examples
    """
    with open(task_file, 'r') as f:
        return json.load(f)


def evaluate_on_task(
    provider: FinetunedLocationProvider,
    task_data: List[Dict],
    task_name: str
) -> Dict:
    """
    Evaluate fine-tuned model on a GPSBench task.

    Args:
        provider: Fine-tuned model provider
        task_data: List of task examples
        task_name: Name of the task

    Returns:
        Evaluation results dictionary
    """
    print(f"\nEvaluating on {task_name} ({len(task_data)} examples)...")

    correct = 0
    results = []

    for i, example in enumerate(task_data):
        # Extract coordinates based on task format
        if 'latitude' in example and 'longitude' in example:
            lat = example['latitude']
            lon = example['longitude']
        elif 'ground_truth' in example:
            lat = example['ground_truth']['latitude']
            lon = example['ground_truth']['longitude']
        else:
            print(f"Warning: Could not extract coordinates from example {i}")
            continue

        # Get ground truth location
        if 'location_name' in example:
            ground_truth = example['location_name']
        elif 'ground_truth' in example and 'location_name' in example['ground_truth']:
            ground_truth = example['ground_truth']['location_name']
        else:
            ground_truth = "Unknown"

        # Predict
        prediction = provider.predict(lat, lon)

        # Check if correct (case-insensitive)
        is_correct = prediction.lower() == ground_truth.lower()
        if is_correct:
            correct += 1

        results.append({
            'latitude': lat,
            'longitude': lon,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'correct': is_correct,
        })

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(task_data)} ({correct/(i+1):.2%} accuracy)")

    accuracy = correct / len(results) if results else 0

    print(f"\nResults for {task_name}:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{len(results)})")

    return {
        'task_name': task_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': len(results),
        'detailed_results': results,
    }


def compare_with_baseline(
    finetuned_results: Dict,
    baseline_file: str
):
    """
    Compare fine-tuned results with baseline results.

    Args:
        finetuned_results: Results from fine-tuned model
        baseline_file: Path to baseline results JSON
    """
    if not Path(baseline_file).exists():
        print(f"Baseline file not found: {baseline_file}")
        return

    with open(baseline_file, 'r') as f:
        baseline = json.load(f)

    print("\n=== Comparison with Baseline ===")
    print(f"{'Metric':<30} | {'Baseline':>12} | {'Fine-tuned':>12} | {'Diff':>12}")
    print("-" * 75)

    ft_acc = finetuned_results['accuracy']
    bl_acc = baseline.get('accuracy', 0)
    diff = ft_acc - bl_acc

    print(f"{'Accuracy':<30} | {bl_acc:>11.2%} | {ft_acc:>11.2%} | {diff:>+11.2%}")


def run_robustness_test(
    provider: FinetunedLocationProvider,
    test_coordinates: List[Tuple[float, float]],
    noise_levels: List[float],
) -> Dict:
    """
    Test model robustness to coordinate noise.

    Args:
        provider: Fine-tuned model provider
        test_coordinates: List of (lat, lon) tuples
        noise_levels: List of noise standard deviations

    Returns:
        Robustness test results
    """
    import numpy as np

    print("\n=== Robustness Test ===")
    results = {}

    for noise_std in noise_levels:
        print(f"\nTesting with noise std = {noise_std:.6f}° (~{noise_std*111:.0f}m)")

        predictions = []
        for lat, lon in test_coordinates:
            # Add noise
            noisy_lat = lat + np.random.normal(0, noise_std)
            noisy_lon = lon + np.random.normal(0, noise_std)

            pred = provider.predict(noisy_lat, noisy_lon)
            predictions.append(pred)

        results[noise_std] = predictions
        print(f"Sample predictions: {predictions[:3]}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model on GPSBench tasks"
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
        '--task-file',
        type=str,
        help='GPSBench task JSON file to evaluate on'
    )
    parser.add_argument(
        '--task-dir',
        type=str,
        help='Directory containing multiple task files'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='../results/gpsbench_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--baseline-file',
        type=str,
        help='Baseline results for comparison'
    )
    parser.add_argument(
        '--robustness-test',
        action='store_true',
        help='Run robustness test with varying noise levels'
    )

    args = parser.parse_args()

    # Load model
    provider = FinetunedLocationProvider(args.base_model, args.lora_path)

    all_results = []

    # Evaluate on single task file
    if args.task_file:
        task_data = load_gpsbench_task(args.task_file)
        task_name = Path(args.task_file).stem

        results = evaluate_on_task(provider, task_data, task_name)
        all_results.append(results)

        # Compare with baseline if provided
        if args.baseline_file:
            compare_with_baseline(results, args.baseline_file)

    # Evaluate on multiple tasks in directory
    elif args.task_dir:
        task_dir = Path(args.task_dir)
        task_files = list(task_dir.glob("*.json"))

        print(f"Found {len(task_files)} task files in {task_dir}")

        for task_file in task_files:
            task_data = load_gpsbench_task(str(task_file))
            task_name = task_file.stem

            results = evaluate_on_task(provider, task_data, task_name)
            all_results.append(results)

    # Run robustness test
    if args.robustness_test:
        # Test coordinates (major cities)
        test_coords = [
            (40.7128, -74.0060),  # New York
            (48.8566, 2.3522),     # Paris
            (35.6762, 139.6503),   # Tokyo
            (51.5074, -0.1278),    # London
            (-33.8688, 151.2093),  # Sydney
        ]

        noise_levels = [0, 0.001, 0.005, 0.01, 0.05]

        robustness_results = run_robustness_test(
            provider,
            test_coords,
            noise_levels
        )

        all_results.append({
            'task_name': 'robustness_test',
            'results': robustness_results
        })

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'model': args.base_model,
            'lora_path': args.lora_path,
            'results': all_results,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    if len(all_results) > 1:
        print("\n=== Summary ===")
        for result in all_results:
            if 'accuracy' in result:
                print(f"{result['task_name']}: {result['accuracy']:.2%}")


if __name__ == '__main__':
    main()
