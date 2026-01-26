#!/usr/bin/env python3
"""
Evaluate finetuned model on ALL GPSBench test sets.

Usage:
    python evaluate_all_tasks.py --checkpoint final
    python evaluate_all_tasks.py --checkpoint 000800 --max-samples 50
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from collections import defaultdict

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import tinker
from tinker import types
from transformers import AutoTokenizer

# Import evaluation functions from gpsbench_evaluators
from gpsbench_evaluators import (
    extract_final_answer,
    evaluate_format_conversion,
    evaluate_numerical_task,
    evaluate_location_name,
)

# Import dataset conversion logic
from gpsbench_datasets import GPSBenchSplitDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def load_all_test_data(data_path: Path) -> Dict[str, List[Dict]]:
    """Load all test splits from both tracks."""
    test_data = {}

    tracks = [
        ("track_pure_gps", "pure_gps"),
        ("track_applied", "applied"),
    ]

    for track_dir, track_name in tracks:
        splits_path = data_path / track_dir / "splits"
        if not splits_path.exists():
            continue

        for test_file in splits_path.glob("*_test.json"):
            task_name = test_file.stem.replace("_test", "")
            with open(test_file) as f:
                examples = json.load(f)

            # Add track info
            for ex in examples:
                ex['_track'] = track_name
                ex['_task_file'] = task_name

            test_data[task_name] = examples
            print(f"  Loaded {len(examples)} examples from {task_name}")

    return test_data


def convert_example_to_prompt(example: Dict, dataset_builder) -> tuple[str, str]:
    """Convert raw example to instruction/output format using dataset builder."""
    instruction_data = dataset_builder.convert_to_instruction(example)
    return instruction_data['instruction'], instruction_data['output']


def evaluate_missing_data(prediction: str, ground_truth) -> float:
    """
    Evaluate Missing Data task with tolerance-based comparison.

    Args:
        prediction: Model's predicted coordinate value
        ground_truth: Either a dict with latitude/longitude and tolerance, or a string

    Returns:
        1.0 if within tolerance, 0.0 otherwise
    """
    # Extract expected value and tolerance from ground_truth
    if isinstance(ground_truth, dict):
        # Get the expected value (either latitude or longitude)
        if 'latitude' in ground_truth:
            expected = ground_truth['latitude']
        elif 'longitude' in ground_truth:
            expected = ground_truth['longitude']
        else:
            return 0.0
        tolerance = ground_truth.get('tolerance', 0.1)
    else:
        # If ground_truth is a string, try to parse it
        gt_numbers = extract_numbers(str(ground_truth))
        if not gt_numbers:
            return 0.0
        expected = gt_numbers[0]
        tolerance = 0.1  # Default tolerance

    # Extract predicted value
    pred_numbers = extract_numbers(str(prediction).replace('°', ''))
    if not pred_numbers:
        return 0.0

    predicted = pred_numbers[0]

    # Check if within tolerance
    return 1.0 if abs(predicted - expected) <= tolerance else 0.0


def evaluate_prediction(task: str, prediction: str, ground_truth: str) -> float:
    """Evaluate prediction based on task type."""
    if not prediction or not ground_truth:
        return 0.0

    # Extract task type from task name (e.g., "task1_format_conversion" -> "format_conversion")
    task_type = task.split('_', 1)[1] if '_' in task else task

    # Format conversion
    if 'format_conversion' in task_type:
        return 1.0 if evaluate_format_conversion(prediction, ground_truth) else 0.0

    # Numerical tasks
    elif any(t in task_type for t in ['distance', 'bearing', 'area', 'perimeter',
                                       'interpolation', 'coordinate_system', 'bounding_box']):
        return 1.0 if evaluate_numerical_task(prediction, ground_truth) else 0.0

    # Place association
    elif 'place_association' in task_type:
        return evaluate_location_name(prediction, ground_truth)

    # Yes/No tasks
    elif any(t in task_type for t in ['relative_position', 'route_analysis', 'semantic_validation']):
        pred_clean = prediction.lower().strip()
        gt_clean = ground_truth.lower().strip()
        # Check if yes/no matches
        if 'yes' in pred_clean and 'yes' in gt_clean:
            return 1.0
        elif 'no' in pred_clean and 'no' in gt_clean:
            return 1.0
        return 0.0

    # Multiple choice (letter answers)
    elif any(t in task_type for t in ['proximity', 'terrain', 'spatial_patterns', 'route_geometry']):
        pred_clean = prediction.upper().strip()
        gt_clean = ground_truth.upper().strip()
        # Extract just the letter
        import re
        pred_letter = re.search(r'[A-D]', pred_clean)
        gt_letter = re.search(r'[A-D]', gt_clean)
        if pred_letter and gt_letter:
            return 1.0 if pred_letter.group() == gt_letter.group() else 0.0
        return 1.0 if pred_clean == gt_clean else 0.0

    # Name disambiguation
    elif 'name_disambiguation' in task_type:
        return evaluate_location_name(prediction, ground_truth)

    # Missing Data - handle dict ground_truth with tolerance
    elif 'missing_data' in task_type:
        return evaluate_missing_data(prediction, ground_truth)

    # Default: exact string match (case-insensitive)
    else:
        pred_clean = prediction.lower().strip()
        gt_clean = ground_truth.lower().strip()
        return 1.0 if pred_clean == gt_clean else 0.0


def run_inference(
    sampler_path: str,
    model_name: str,
    prompts: List[str],
    max_tokens: int = 1024,  # Consistent with training expectations
) -> List[str]:
    """Run inference using Tinker SamplingClient API."""

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Creating sampling client with checkpoint: {sampler_path}")
    client = tinker.ServiceClient()
    sampling_client = client.create_sampling_client(
        model_path=sampler_path,
        base_model=model_name
    )

    system_msg = "You are an expert in GPS coordinates, geographic information systems, and spatial reasoning. Provide precise, accurate answers."

    responses = []
    for prompt in tqdm(prompts, desc="Running inference"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt_text)

        model_input = types.ModelInput(chunks=[
            types.EncodedTextChunk(tokens=tokens, type='encoded_text')
        ])

        sampling_params = types.SamplingParams(max_tokens=max_tokens, temperature=0.0)
        response = sampling_client.sample(
            model_input, num_samples=1, sampling_params=sampling_params
        ).result()

        output_tokens = response.sequences[0].tokens
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        responses.append(output_text)

    return responses


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned model on ALL GPSBench test sets")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='final',
        help='Checkpoint name (e.g., "final", "000800", "000400")'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/all_tasks_with_reasoning',
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
        help='Maximum samples per task to evaluate (for quick testing)'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=None,
        help='Specific tasks to evaluate (e.g., task1_format_conversion task4_distance_calculation)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Evaluating Finetuned Model on ALL Test Sets")
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
                sampler_path = ckpt['sampler_path']
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
    test_data = load_all_test_data(data_path)

    # Filter tasks if specified
    if args.tasks:
        test_data = {k: v for k, v in test_data.items() if k in args.tasks}

    print(f"\nTotal tasks to evaluate: {len(test_data)}")
    print()

    # Create dataset builder for conversion
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name="qwen_chat",
        max_length=16384,  # Consistent with training
        batch_size=1,
    )
    dataset_builder = GPSBenchSplitDataset(
        common_config=common_config,
        gpsbench_root=str(data_path.parent),
        track="both",
        include_reasoning=False,
    )

    # Evaluate each task
    all_results = {}
    task_accuracies = {}

    for task_name, examples in test_data.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {task_name}")
        print(f"{'='*70}")

        # Limit samples if requested
        if args.max_samples:
            examples = examples[:args.max_samples]

        print(f"Test samples: {len(examples)}")

        # Convert to prompts
        prompts = []
        ground_truths = []

        for ex in examples:
            try:
                instruction, output = convert_example_to_prompt(ex, dataset_builder)
                prompts.append(instruction)
                expected_answer = extract_final_answer(output)
                ground_truths.append(expected_answer or output)
            except Exception as e:
                print(f"  Warning: Failed to convert example: {e}")
                continue

        if not prompts:
            print(f"  No valid examples for {task_name}, skipping...")
            continue

        # Run inference
        print(f"Running inference on {len(prompts)} examples...")
        responses = run_inference(
            sampler_path=sampler_path,
            model_name=args.model_name,
            prompts=prompts,
        )

        # Evaluate
        correct = 0
        scores = []
        results = []

        for i, (response, gt) in enumerate(zip(responses, ground_truths)):
            prediction = extract_final_answer(response) or response.strip()
            score = evaluate_prediction(task_name, prediction, gt)

            scores.append(score)
            if score >= 0.5:
                correct += 1

            results.append({
                'index': i,
                'prediction': prediction,
                'ground_truth': gt,
                'score': score,
                'full_response': response[:500]  # Truncate for storage
            })

        # Calculate metrics
        accuracy = correct / len(examples) * 100 if examples else 0
        avg_score = sum(scores) / len(scores) * 100 if scores else 0

        task_accuracies[task_name] = {
            'accuracy': accuracy,
            'avg_score': avg_score,
            'correct': correct,
            'total': len(examples)
        }

        print(f"Results: {correct}/{len(examples)} correct ({accuracy:.1f}%)")

        all_results[task_name] = {
            'accuracy': accuracy,
            'avg_score': avg_score,
            'correct': correct,
            'total': len(examples),
            'results': results
        }

    # Print summary
    print("\n")
    print("=" * 70)
    print("SUMMARY - All Tasks")
    print("=" * 70)

    # Group by track
    pure_gps_tasks = [t for t in task_accuracies.keys() if 'track_pure_gps' in str(test_data.get(t, [{}])[0].get('_track', '')) or any(x in t for x in ['format', 'distance', 'bearing', 'interpolation', 'area', 'bounding', 'route_geometry', 'coordinate_system', 'relative_position'])]
    applied_tasks = [t for t in task_accuracies.keys() if t not in pure_gps_tasks]

    total_correct = sum(t['correct'] for t in task_accuracies.values())
    total_samples = sum(t['total'] for t in task_accuracies.values())
    overall_accuracy = total_correct / total_samples * 100 if total_samples else 0

    print(f"\nOverall: {total_correct}/{total_samples} ({overall_accuracy:.1f}%)")
    print()

    print("Per-task results:")
    print("-" * 50)
    for task_name in sorted(task_accuracies.keys()):
        stats = task_accuracies[task_name]
        print(f"  {task_name}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")

    # Save results
    results_file = output_dir / f"all_tasks_test_results_{args.checkpoint}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_samples': total_samples,
            'task_results': all_results,
            'task_accuracies': task_accuracies,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
