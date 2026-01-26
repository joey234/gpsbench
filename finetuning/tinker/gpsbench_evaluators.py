"""
Task-specific evaluators for GPSBench Tinker fine-tuning.

Integrates existing GPSBench evaluation logic with Tinker's training framework.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

import tinker
from tinker import types
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Import system prompt from task_prompt_mapping for consistency with run_benchmark.py
sys.path.append(str(Path(__file__).parent.parent))
from task_prompt_mapping import STANDARD_SYSTEM_PROMPT


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the FINAL ANSWER line from response

    Returns:
        The content after "FINAL ANSWER:" or None if not found
    """
    # Look for "FINAL ANSWER:" pattern (case-insensitive)
    match = re.search(r'FINAL\s+ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', text) if x]


def numbers_match(nums1: List[float], nums2: List[float], tolerance: float = 0.01) -> bool:
    """Check if two lists of numbers match within tolerance."""
    if len(nums1) != len(nums2):
        return False
    return all(abs(a - b) < tolerance for a, b in zip(nums1, nums2))


def evaluate_format_conversion(prediction: str, ground_truth: str) -> bool:
    """
    Evaluate format conversion task.

    Args:
        prediction: Model's predicted coordinate string
        ground_truth: Expected coordinate string

    Returns:
        True if coordinates match within tolerance
    """
    pred_numbers = extract_numbers(prediction)
    gt_numbers = extract_numbers(ground_truth)
    return numbers_match(pred_numbers, gt_numbers, tolerance=0.01)


def evaluate_numerical_task(prediction: str, ground_truth: str, rel_threshold: float = 0.05) -> bool:
    """
    Evaluate numerical tasks (distance, bearing, area, etc.).

    Args:
        prediction: Model's prediction
        ground_truth: Expected value
        rel_threshold: Relative error threshold (5% default)

    Returns:
        True if within tolerance
    """
    pred_numbers = extract_numbers(prediction)
    gt_numbers = extract_numbers(ground_truth)

    if not pred_numbers or not gt_numbers:
        return False

    # Use first number from each
    pred_val = pred_numbers[0]
    gt_val = gt_numbers[0]

    # Handle zero ground truth
    if abs(gt_val) < 1e-9:
        return abs(pred_val) < 1e-9

    # Relative error
    rel_error = abs(pred_val - gt_val) / abs(gt_val)
    return rel_error < rel_threshold


def evaluate_location_name(prediction: str, ground_truth: str) -> float:
    """
    Evaluate location name prediction with fuzzy matching.

    Returns:
        Score between 0.0 and 1.0
    """
    pred_lower = prediction.lower().strip()
    gt_lower = ground_truth.lower().strip()

    # Exact match
    if pred_lower == gt_lower:
        return 1.0

    # Contains match
    if pred_lower in gt_lower or gt_lower in pred_lower:
        return 0.5

    # Word overlap
    pred_words = set(pred_lower.split())
    gt_words = set(gt_lower.split())
    overlap = len(pred_words & gt_words)

    if overlap > 0:
        return 0.5 * (overlap / max(len(pred_words), len(gt_words)))

    return 0.0


class GPSBenchTaskEvaluator(SamplingClientEvaluator):
    """
    GPSBench task-specific evaluator for Tinker training.

    Computes accuracy metrics during training for different task types:
    - Format conversion: Coordinate accuracy
    - Distance/bearing/area: Numerical MAE
    - Place association: String matching
    """

    def __init__(
        self,
        eval_examples: List[Dict[str, Any]],
        model_name: str,
        renderer_name: str,
        name: str = "gpsbench",
        max_samples: int | None = None,
    ):
        """
        Args:
            eval_examples: List of evaluation examples with instruction/output/metadata
            model_name: Model name for tokenizer
            renderer_name: Renderer name (llama3, qwen_chat, etc.)
            name: Name prefix for metrics
            max_samples: Maximum samples to evaluate (None = use all samples)
        """
        self.name = name
        if max_samples is not None:
            self.eval_examples = eval_examples[:max_samples]
        else:
            self.eval_examples = eval_examples

        # Create renderer for building prompts and parsing responses
        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    async def __call__(self, sampling_client: tinker.SamplingClient) -> Dict[str, float]:
        """
        Evaluate model on GPSBench tasks.

        Returns:
            Dictionary of metric_name -> value
        """
        task_scores = {}
        all_scores = []

        # Set up sampling parameters
        sampling_params = types.SamplingParams(
            max_tokens=512,
            temperature=0.0,  # Greedy for evaluation
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )

        total_examples = len(self.eval_examples)
        print(f"\n{'='*60}")
        print(f"Evaluating {total_examples} examples...")
        print(f"{'='*60}")

        # Progress tracking
        progress_interval = max(1, total_examples // 20)  # Print progress every 5%

        for idx, example in enumerate(self.eval_examples, 1):
            try:
                # Print progress
                if idx % progress_interval == 0 or idx == total_examples:
                    pct = (idx / total_examples) * 100
                    current_acc = np.mean(all_scores) if all_scores else 0.0
                    print(f"Progress: {idx}/{total_examples} ({pct:.0f}%) | Current accuracy: {current_acc:.3f}")

                # Extract task name
                task = example.get('task', 'unknown')

                # Get ground truth output
                expected_output = example['output']
                expected_answer = extract_final_answer(expected_output)

                # Create prompt from instruction
                instruction = example['instruction']

                # Build model input using renderer
                # Use STANDARD_SYSTEM_PROMPT for consistency with run_benchmark.py
                messages = [
                    {"role": "system", "content": STANDARD_SYSTEM_PROMPT},
                    {"role": "user", "content": instruction}
                ]
                model_input = self.renderer.build_generation_prompt(messages)

                # Generate prediction
                response = await sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params
                )

                # Parse response to get text
                tokens = response.sequences[0].tokens
                parsed_message, _ = self.renderer.parse_response(tokens)
                prediction_text = parsed_message["content"]

                # Extract FINAL ANSWER
                prediction = extract_final_answer(prediction_text)

                # Task-specific evaluation
                score = self._evaluate_by_task(task, prediction, expected_answer)

                # Track scores by task type
                if task not in task_scores:
                    task_scores[task] = []
                task_scores[task].append(score)
                all_scores.append(score)

            except Exception as e:
                print(f"Error evaluating example {idx} (task={example.get('task', 'unknown')}): {e}")
                all_scores.append(0.0)

        print(f"{'='*60}")
        print(f"Evaluation complete! Final accuracy: {np.mean(all_scores):.3f}")
        print(f"{'='*60}\n")

        # Compute metrics
        metrics = {}

        # Overall accuracy
        if all_scores:
            metrics[f"{self.name}/accuracy"] = np.mean(all_scores)
            metrics[f"{self.name}/count"] = len(all_scores)

        # Per-task accuracy
        for task, scores in task_scores.items():
            if scores:
                metrics[f"{self.name}/{task}_accuracy"] = np.mean(scores)
                metrics[f"{self.name}/{task}_count"] = len(scores)

        return metrics

    def _evaluate_by_task(
        self,
        task: str,
        prediction: str,
        ground_truth: str
    ) -> float:
        """
        Evaluate prediction based on task type.

        Returns:
            Score between 0.0 and 1.0
        """
        if not prediction or not ground_truth:
            return 0.0

        # Format conversion
        if task == 'format_conversion':
            return 1.0 if evaluate_format_conversion(prediction, ground_truth) else 0.0

        # Numerical tasks
        elif task in ['distance_calculation', 'bearing_computation', 'area_perimeter',
                      'interpolation', 'coordinate_system_transformation']:
            return 1.0 if evaluate_numerical_task(prediction, ground_truth) else 0.0

        # Place association
        elif task == 'place_association':
            return evaluate_location_name(prediction, ground_truth)

        # Default: exact string match (case-insensitive)
        else:
            pred_clean = prediction.lower().strip()
            gt_clean = ground_truth.lower().strip()
            return 1.0 if pred_clean == gt_clean else 0.0


def create_gpsbench_evaluator(
    eval_dataset_builder,
    model_name: str,
    renderer_name: str,
    name: str = "eval",
    max_samples: int | None = None
):
    """
    Factory function to create GPSBench evaluator from dataset builder.

    Args:
        eval_dataset_builder: Dataset builder instance
        model_name: Model name for tokenizer
        renderer_name: Renderer name (llama3, qwen_chat, etc.)
        name: Name for the evaluator
        max_samples: Maximum samples to evaluate (None = use full dev set)

    Returns:
        Lambda that creates evaluator instance
    """
    def builder():
        # Get evaluation examples from dataset builder
        try:
            eval_conversations = eval_dataset_builder.load_conversations("dev")

            # STRATIFIED SAMPLING: Sample evenly across all tasks
            from collections import defaultdict
            import random

            # Group conversations by task
            task_groups = defaultdict(list)
            for conv in eval_conversations:
                task = conv.get('task', 'unknown')
                task_groups[task].append(conv)

            num_tasks = len(task_groups)

            print(f"\n{'='*60}")
            print(f"GPSBench Evaluation Setup (stratified sampling)")
            print(f"{'='*60}")
            print(f"Total dev examples: {len(eval_conversations)}")
            print(f"Number of tasks: {num_tasks}")

            # Sample from each task
            sampled_conversations = []
            if max_samples is None:
                # Use ALL dev examples
                print(f"Using FULL dev set (all {len(eval_conversations)} examples)")
                print(f"\nTask distribution in eval set:")
                for task in sorted(task_groups.keys()):
                    task_convs = task_groups[task]
                    sampled_conversations.extend(task_convs)
                    print(f"  {task}: {len(task_convs)} examples")
            else:
                # Stratified sampling: distribute max_samples evenly across tasks
                samples_per_task = max(1, max_samples // num_tasks)
                print(f"Samples per task: {samples_per_task}")
                print(f"Total samples to evaluate: ~{min(max_samples, num_tasks * samples_per_task)}")
                print(f"\nTask distribution in eval set:")

                for task in sorted(task_groups.keys()):
                    task_convs = task_groups[task]
                    # Shuffle and take samples_per_task examples
                    random.seed(42)  # Reproducible sampling
                    sampled = random.sample(task_convs, min(samples_per_task, len(task_convs)))
                    sampled_conversations.extend(sampled)
                    print(f"  {task}: {len(sampled)}/{len(task_convs)} examples")

            print(f"{'='*60}\n")

            # Convert conversations to evaluation format
            eval_examples = []
            for conv in sampled_conversations:
                # Extract instruction and output from messages
                messages = conv.get('messages', [])
                instruction = None
                output = None

                for msg in messages:
                    if msg['role'] == 'user':
                        instruction = msg['content']
                    elif msg['role'] == 'assistant':
                        output = msg['content']

                if instruction and output:
                    eval_examples.append({
                        'instruction': instruction,
                        'output': output,
                        'task': conv.get('task', 'unknown')  # Extract task name directly
                    })

            return GPSBenchTaskEvaluator(
                eval_examples=eval_examples,
                model_name=model_name,
                renderer_name=renderer_name,
                name=name,
                max_samples=max_samples
            )
        except Exception as e:
            print(f"Error creating GPSBench evaluator: {e}")
            # Return dummy evaluator that does nothing
            return GPSBenchTaskEvaluator(
                eval_examples=[],
                model_name=model_name,
                renderer_name=renderer_name,
                name=name,
                max_samples=0
            )

    return builder
