#!/usr/bin/env python3
"""
Test General Capabilities After Fine-Tuning
Ensures the model hasn't forgotten how to do general tasks
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# General capability test cases
TEST_CASES = [
    {
        "name": "Basic Math",
        "prompt": "What is 15 + 27?",
        "expected_keywords": ["42", "forty-two", "forty two"],
        "category": "reasoning"
    },
    {
        "name": "General Knowledge",
        "prompt": "What is the capital of France?",
        "expected_keywords": ["Paris"],
        "category": "knowledge"
    },
    {
        "name": "Instruction Following",
        "prompt": "List three primary colors.",
        "expected_keywords": ["red", "blue", "yellow"],
        "category": "instruction"
    },
    {
        "name": "Simple Reasoning",
        "prompt": "If it's raining, should I bring an umbrella? Answer yes or no.",
        "expected_keywords": ["yes"],
        "category": "reasoning"
    },
    {
        "name": "Text Generation",
        "prompt": "Write a friendly greeting.",
        "expected_keywords": ["hello", "hi", "greetings", "welcome"],
        "category": "generation"
    },
    {
        "name": "Counting",
        "prompt": "Count from 1 to 5.",
        "expected_keywords": ["1", "2", "3", "4", "5"],
        "category": "reasoning"
    },
    {
        "name": "Language Understanding",
        "prompt": "Is the sentence 'The cat sat on the mat' grammatically correct? Answer yes or no.",
        "expected_keywords": ["yes"],
        "category": "language"
    }
]


def load_model(base_model_name, lora_path=None):
    """Load base model or fine-tuned model with LoRA"""
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if lora_path:
        print(f"  Base: {base_model_name}")
        print(f"  LoRA: {lora_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        print(f"  Model: {base_model_name} (base model only)")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    """Generate response for a prompt"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only the generated part (after the prompt)
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def evaluate_response(response, expected_keywords):
    """Check if response contains expected keywords"""
    response_lower = response.lower()
    matches = [kw for kw in expected_keywords if kw.lower() in response_lower]
    return len(matches) > 0, matches


def run_tests(model, tokenizer, verbose=False):
    """Run all test cases"""
    results = {
        "total": len(TEST_CASES),
        "passed": 0,
        "failed": 0,
        "by_category": {}
    }

    print("\n" + "=" * 80)
    print("GENERAL CAPABILITY TESTS")
    print("=" * 80)

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {test['name']} ({test['category']})")
        print(f"  Prompt: {test['prompt']}")

        response = generate_response(model, tokenizer, test['prompt'])
        passed, matches = evaluate_response(response, test['expected_keywords'])

        if verbose or not passed:
            print(f"  Response: {response}")

        if passed:
            print(f"  ✅ PASS (found: {', '.join(matches)})")
            results['passed'] += 1
        else:
            print(f"  ❌ FAIL (expected any of: {', '.join(test['expected_keywords'])})")
            if not verbose:
                print(f"  Response: {response}")
            results['failed'] += 1

        # Track by category
        category = test['category']
        if category not in results['by_category']:
            results['by_category'][category] = {'passed': 0, 'failed': 0}

        if passed:
            results['by_category'][category]['passed'] += 1
        else:
            results['by_category'][category]['failed'] += 1

    return results


def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    accuracy = (results['passed'] / results['total']) * 100
    print(f"\nOverall: {results['passed']}/{results['total']} passed ({accuracy:.1f}%)")

    if accuracy >= 80:
        print("✅ EXCELLENT: Model retains strong general capabilities")
    elif accuracy >= 60:
        print("⚠️  GOOD: Model retains most general capabilities, minor degradation")
    elif accuracy >= 40:
        print("⚠️  MODERATE: Some capability loss, consider lower learning rate or LoRA rank")
    else:
        print("❌ POOR: Significant capability loss, catastrophic forgetting detected!")

    print("\nBy Category:")
    for category, stats in sorted(results['by_category'].items()):
        total = stats['passed'] + stats['failed']
        cat_accuracy = (stats['passed'] / total) * 100
        status = "✅" if cat_accuracy >= 70 else "⚠️" if cat_accuracy >= 50 else "❌"
        print(f"  {status} {category.capitalize()}: {stats['passed']}/{total} ({cat_accuracy:.0f}%)")

    print()


def main():
    parser = argparse.ArgumentParser(description='Test general capabilities after fine-tuning')
    parser.add_argument('--base-model', type=str, default='Qwen/Qwen3-4B-Instruct-2507',
                        help='Base model name')
    parser.add_argument('--lora-path', type=str, default=None,
                        help='Path to fine-tuned LoRA adapter (if None, tests base model)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show all responses (not just failures)')

    args = parser.parse_args()

    print("=" * 80)
    print("GENERAL CAPABILITY RETENTION TEST")
    print("=" * 80)
    print("\nPurpose: Verify that fine-tuning hasn't caused catastrophic forgetting")
    print("         of the model's general capabilities (math, reasoning, knowledge)")

    # Load model
    model, tokenizer = load_model(args.base_model, args.lora_path)

    # Run tests
    results = run_tests(model, tokenizer, verbose=args.verbose)

    # Print summary
    print_summary(results)

    # Recommendations
    if results['passed'] / results['total'] < 0.6:
        print("=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        print("\nSignificant capability loss detected. Consider:")
        print("  1. Lower learning rate: --learning-rate 5e-5 (instead of 1e-4)")
        print("  2. Lower LoRA rank: --lora-r 8 (instead of 16)")
        print("  3. Higher LoRA dropout: --lora-dropout 0.2 (instead of 0.1)")
        print("  4. Fewer epochs: --num-epochs 2 (instead of 3+)")
        print("  5. Stronger early stopping: --early-stopping-patience 1")
        print()


if __name__ == "__main__":
    main()
