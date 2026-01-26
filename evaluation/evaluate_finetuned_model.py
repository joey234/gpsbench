#!/usr/bin/env python3
"""
Evaluate Fine-Tuned Model on GPSBench Test Set
Specifically for locally fine-tuned models with LoRA adapters
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class FineTunedModelEvaluator:
    """Evaluator for locally fine-tuned models"""

    def __init__(
        self,
        base_model_name: str,
        lora_path: str = None,
        device: str = "auto"
    ):
        print("=" * 80)
        print("LOADING MODEL")
        print("=" * 80)
        print(f"\nBase model: {base_model_name}")
        if lora_path:
            print(f"LoRA adapter: {lora_path}")
        else:
            print("No LoRA adapter (using base model)")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print("\nLoading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        )

        # Load LoRA adapter if provided
        if lora_path:
            print(f"Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            self.model = base_model

        self.model.eval()
        print("✓ Model loaded successfully\n")

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0
    ) -> str:
        """Generate response for a prompt"""
        messages = [
            {"role": "system", "content": "You are a geographic AI assistant. Given coordinates, identify the location name. Provide clear, concise answers with the place name and country if relevant."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            if temperature == 0.0:
                # Greedy decoding
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                # Sampling
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

        # Extract only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return generated_text.strip()

    def evaluate_place_association(
        self,
        test_file: str,
        max_samples: int = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Evaluate on place_association task"""

        # Load test data
        print("=" * 80)
        print("LOADING TEST DATA")
        print("=" * 80)
        print(f"\nTest file: {test_file}")

        with open(test_file) as f:
            test_data = json.load(f)

        if max_samples:
            test_data = test_data[:max_samples]

        print(f"Test cases: {len(test_data)}")
        print()

        # Evaluate each test case
        print("=" * 80)
        print("RUNNING EVALUATION")
        print("=" * 80)
        print()

        results = []
        correct = 0
        total = 0

        for i, test_case in enumerate(tqdm(test_data, desc="Evaluating")):
            # Create prompt
            coord_str = test_case['coordinate_string']
            prompt = f"What location is at coordinates {coord_str}?"

            # Generate response
            response = self.generate_response(prompt, temperature=temperature)

            # Extract ground truth
            ground_truth = test_case['ground_truth']['name']
            country = test_case['ground_truth'].get('country', '')

            # Simple matching (case-insensitive, check if name is in response)
            # More sophisticated matching could be added
            is_correct = self.check_answer(response, ground_truth, country)

            if is_correct:
                correct += 1
            total += 1

            # Store result
            result = {
                'test_case_id': i,
                'coordinate': coord_str,
                'prompt': prompt,
                'response': response,
                'ground_truth': ground_truth,
                'country': country,
                'correct': is_correct,
                'difficulty': test_case.get('difficulty', 'unknown'),
                'subtask': test_case.get('subtask', 'unknown')
            }
            results.append(result)

        # Calculate metrics
        accuracy = (correct / total * 100) if total > 0 else 0

        print()
        print("=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"\nTotal test cases: {total}")
        print(f"Correct: {correct}")
        print(f"Incorrect: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")

        # Breakdown by difficulty
        by_difficulty = {}
        for result in results:
            diff = result['difficulty']
            if diff not in by_difficulty:
                by_difficulty[diff] = {'correct': 0, 'total': 0}
            by_difficulty[diff]['total'] += 1
            if result['correct']:
                by_difficulty[diff]['correct'] += 1

        print("\nBy difficulty:")
        for diff, stats in sorted(by_difficulty.items()):
            diff_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {diff}: {stats['correct']}/{stats['total']} ({diff_acc:.1f}%)")

        # Breakdown by subtask
        by_subtask = {}
        for result in results:
            subtask = result['subtask']
            if subtask not in by_subtask:
                by_subtask[subtask] = {'correct': 0, 'total': 0}
            by_subtask[subtask]['total'] += 1
            if result['correct']:
                by_subtask[subtask]['correct'] += 1

        print("\nBy subtask:")
        for subtask, stats in sorted(by_subtask.items()):
            subtask_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {subtask}: {stats['correct']}/{stats['total']} ({subtask_acc:.1f}%)")

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'by_difficulty': by_difficulty,
            'by_subtask': by_subtask,
            'results': results
        }

    def check_answer(self, response: str, ground_truth: str, country: str = '') -> bool:
        """
        Check if response matches ground truth
        Uses fuzzy matching since the model may format answers differently
        """
        response_lower = response.lower()
        ground_truth_lower = ground_truth.lower()

        # Extract main location name (before comma if exists)
        gt_parts = ground_truth_lower.split(',')
        main_location = gt_parts[0].strip()

        # Check if main location is in response
        if main_location in response_lower:
            return True

        # Check if country code is mentioned
        if country:
            country_lower = country.lower()
            # Check for country code (e.g., "US", "GB")
            if f" {country[:2].upper()}" in response or f", {country[:2].upper()}" in response:
                return True

        # Additional fuzzy checks could go here
        # For now, strict matching
        return False

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to JSON"""
        print(f"\nSaving results to {output_file}...")

        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned model on GPSBench test set'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='Qwen/Qwen3-4B-Instruct-2507',
        help='Base model name'
    )
    parser.add_argument(
        '--lora-path',
        type=str,
        default=None,
        help='Path to LoRA adapter (if None, evaluates base model)'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default='../data/track_applied/task1_place_association.json',
        help='Test file path'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for generation (default: 0.0 for deterministic)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: auto-generated)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Results directory'
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = FineTunedModelEvaluator(
        base_model_name=args.base_model,
        lora_path=args.lora_path
    )

    # Run evaluation
    results = evaluator.evaluate_place_association(
        test_file=args.test_file,
        max_samples=args.max_samples,
        temperature=args.temperature
    )

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "finetuned" if args.lora_path else "base"
        args.output = f"{args.results_dir}/place_association_{model_name}_{timestamp}.json"

    # Save results
    evaluator.save_results(results, args.output)

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Accuracy: {results['accuracy']:.2f}%")
    print(f"✓ Results saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
