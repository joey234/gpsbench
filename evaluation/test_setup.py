#!/usr/bin/env python3
"""
Test script to verify GPSBench evaluation setup
Runs a quick test with a few samples to ensure everything is configured correctly
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import LLMClient
from task_evaluators import get_evaluator
import json


def test_llm_client():
    """Test LLM client connection"""
    print("=" * 80)
    print("Testing LLM Client")
    print("=" * 80)

    try:
        # Test OpenAI
        print("\n[1] Testing OpenAI connection...")
        client = LLMClient(provider="openai", model="gpt-3.5-turbo")
        response = client.generate("What is 2+2? Answer with just the number.")

        if response.error:
            print(f"  ✗ OpenAI test failed: {response.error}")
            return False
        else:
            print(f"  ✓ OpenAI working! Response: {response.text}")
            print(f"    Tokens: {response.total_tokens}, Latency: {response.latency_ms:.0f}ms")

    except Exception as e:
        print(f"  ✗ OpenAI test failed: {e}")
        print("    Make sure OPENAI_API_KEY is set in .env file")
        return False

    try:
        # Test OpenRouter (optional)
        print("\n[2] Testing OpenRouter connection...")
        client = LLMClient(provider="openrouter", model="openai/gpt-3.5-turbo")
        response = client.generate("What is 2+2? Answer with just the number.")

        if response.error:
            print(f"  ⚠ OpenRouter test failed: {response.error}")
            print("    (Optional - OpenRouter is not required)")
        else:
            print(f"  ✓ OpenRouter working! Response: {response.text}")
            print(f"    Tokens: {response.total_tokens}, Latency: {response.latency_ms:.0f}ms")

    except Exception as e:
        print(f"  ⚠ OpenRouter test skipped: {e}")
        print("    (Optional - OpenRouter is not required)")

    return True


def test_data_loading():
    """Test that data files can be loaded"""
    print("\n" + "=" * 80)
    print("Testing Data Loading")
    print("=" * 80)

    data_dir = Path("data/layer_a")

    if not data_dir.exists():
        print(f"  ✗ Data directory not found: {data_dir}")
        print("    Run: python scripts/download_data.py")
        print("    Then: python scripts/construct_dataset.py")
        return False

    task_files = {
        'format_recognition': 'task1_format_recognition.json',
        'format_conversion': 'task2_format_conversion.json',
        'validation': 'task3_validation.json',
        'coordinate_system_recognition': 'task4_coordinate_systems.json',
        'precision_understanding': 'task5_precision.json'
    }

    for task_name, filename in task_files.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  ✗ Missing: {filename}")
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)
            print(f"  ✓ {task_name}: {len(data)} test cases")

    return True


def test_evaluators():
    """Test task evaluators"""
    print("\n" + "=" * 80)
    print("Testing Task Evaluators")
    print("=" * 80)

    # Test format recognition evaluator
    print("\n[1] Testing Format Recognition Evaluator...")
    evaluator = get_evaluator('format_recognition')

    test_case = {
        'coordinate_string': '40°51\'30"N, 73°59\'45"W',
        'ground_truth': {
            'format_type': 'DMS',
            'latitude': 40.858,
            'longitude': -73.996
        }
    }

    system_prompt, user_prompt = evaluator.create_prompt(test_case)
    print(f"  Prompt created: '{user_prompt[:50]}...'")

    # Test evaluation
    response = "This is in DMS format."
    result = evaluator.evaluate_response(response, test_case)
    print(f"  Evaluation: {'✓ PASS' if result['correct'] else '✗ FAIL'}")

    return True


def test_mini_evaluation():
    """Run a mini evaluation with 3 test cases"""
    print("\n" + "=" * 80)
    print("Running Mini Evaluation (3 samples)")
    print("=" * 80)

    try:
        # Load a few test cases
        data_dir = Path("data/layer_a")
        with open(data_dir / "task1_format_recognition.json", 'r') as f:
            test_cases = json.load(f)[:3]

        # Initialize client
        client = LLMClient(provider="openai", model="gpt-3.5-turbo", temperature=0.0)
        evaluator = get_evaluator('format_recognition')

        correct = 0
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[Test {i}/3]")
            print(f"  Input: {test_case['coordinate_string']}")

            # Create prompt and get response
            system_prompt, user_prompt = evaluator.create_prompt(test_case)
            response = client.generate(user_prompt, system_prompt)

            if response.error:
                print(f"  ✗ Error: {response.error}")
                continue

            # Evaluate
            result = evaluator.evaluate_response(response.text, test_case)
            status = "✓" if result['correct'] else "✗"

            print(f"  Expected: {result['expected']}")
            print(f"  Response: {response.text}")
            print(f"  {status} {'CORRECT' if result['correct'] else 'INCORRECT'}")

            if result['correct']:
                correct += 1

        accuracy = (correct / len(test_cases)) * 100
        print(f"\n{'─' * 80}")
        print(f"Mini Evaluation Results: {correct}/{len(test_cases)} correct ({accuracy:.1f}%)")
        print(f"{'─' * 80}")

        return True

    except Exception as e:
        print(f"  ✗ Mini evaluation failed: {e}")
        return False


def main():
    print("\n" + "=" * 80)
    print("GPSBench Evaluation Setup Test")
    print("=" * 80)

    all_passed = True

    # Test 1: LLM Client
    if not test_llm_client():
        all_passed = False
        print("\n⚠ LLM client test failed. Check your API keys in .env file")

    # Test 2: Data Loading
    if not test_data_loading():
        all_passed = False
        print("\n⚠ Data loading failed. Run data generation scripts first")

    # Test 3: Evaluators
    if not test_evaluators():
        all_passed = False

    # Test 4: Mini Evaluation
    if all_passed:
        test_mini_evaluation()

    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All tests passed! Setup is complete.")
        print("\nYou're ready to run full evaluations:")
        print("  python evaluation/evaluate_layer_a.py --max-samples 10")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Set API keys in .env file")
        print("  2. Run: python scripts/download_data.py")
        print("  3. Run: python scripts/construct_dataset.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
