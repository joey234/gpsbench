#!/usr/bin/env python3
"""
Test script to verify paraphrasing setup before running full job.

This script:
1. Checks if API keys are configured
2. Tests LLM client connection
3. Runs a small paraphrasing test
4. Shows cost estimate for full run
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import directly from llm_client module to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "llm_client",
    Path(__file__).parent.parent / "evaluation" / "llm_client.py"
)
llm_client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_client_module)
LLMClient = llm_client_module.LLMClient

import json


def test_api_keys():
    """Test if API keys are configured."""
    print("="*70)
    print("1. CHECKING API KEYS")
    print("="*70)

    import os
    from dotenv import load_dotenv

    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if openai_key:
        print(f"✅ OPENAI_API_KEY found ({openai_key[:10]}...)")
    else:
        print("❌ OPENAI_API_KEY not found")

    if openrouter_key:
        print(f"✅ OPENROUTER_API_KEY found ({openrouter_key[:10]}...)")
    else:
        print("❌ OPENROUTER_API_KEY not found")

    if not openai_key and not openrouter_key:
        print("\n⚠️  No API keys found!")
        print("Please create a .env file with:")
        print("  OPENAI_API_KEY=your_key_here")
        print("  OR")
        print("  OPENROUTER_API_KEY=your_key_here")
        return False

    return True


def test_llm_connection(model="gpt-4.1-mini"):
    """Test LLM client connection."""
    print("\n" + "="*70)
    print(f"2. TESTING LLM CONNECTION ({model})")
    print("="*70)

    try:
        client = LLMClient(provider="auto", model=model, temperature=0.0)
        print(f"✅ Client initialized: {client}")

        # Test simple query
        response = client.generate("Say 'API connection successful' and nothing else.")

        if response.error:
            print(f"❌ Error: {response.error}")
            return False

        print(f"✅ Response received: '{response.text[:50]}...'")
        print(f"✅ Tokens: {response.total_tokens}, Latency: {response.latency_ms:.0f}ms")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_paraphrasing_sample():
    """Test paraphrasing with a sample reasoning."""
    print("\n" + "="*70)
    print("3. TESTING PARAPHRASING")
    print("="*70)

    # Load a sample reasoning
    data_path = Path(__file__).parent.parent / "data" / "track_pure_gps" / "task5_distance_calculation.json"

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return False

    with open(data_path) as f:
        data = json.load(f)

    # Find example with reasoning
    example = None
    for ex in data:
        if 'reasoning_steps' in ex and ex['reasoning_steps']:
            example = ex
            break

    if not example:
        print("❌ No examples with reasoning found")
        return False

    original_steps = example['reasoning_steps']

    print(f"Original reasoning ({len(original_steps)} steps):")
    for step in original_steps[:3]:
        print(f"  {step}")
    print("  ...")

    # Paraphrase
    from scripts.paraphrase_reasoning_steps import (
        create_paraphrase_prompt,
        paraphrase_reasoning,
        PARAPHRASING_STYLES
    )

    try:
        client = LLMClient(provider="auto", model="gpt-4.1-mini", temperature=0.7)

        print("\nParaphrasing with 'tutorial' style...")
        paraphrased = paraphrase_reasoning(original_steps, client, style="tutorial")

        if paraphrased != original_steps:
            print("\n✅ Paraphrasing successful!")
            print(f"\nParaphrased reasoning ({len(paraphrased)} steps):")
            for step in paraphrased[:3]:
                print(f"  {step}")
            print("  ...")
            return True
        else:
            print("\n⚠️  Paraphrasing returned original (validation may have failed)")
            return False

    except Exception as e:
        print(f"\n❌ Paraphrasing failed: {e}")
        return False


def estimate_costs():
    """Estimate costs for full paraphrasing run."""
    print("\n" + "="*70)
    print("4. COST ESTIMATION")
    print("="*70)

    # Count examples with reasoning
    data_dir = Path(__file__).parent.parent / "data"

    total_with_reasoning = 0

    for task_file in data_dir.rglob("task*.json"):
        try:
            with open(task_file) as f:
                data = json.load(f)
                count = sum(1 for ex in data if 'reasoning_steps' in ex and ex['reasoning_steps'])
                if count > 0:
                    total_with_reasoning += count
        except:
            pass

    print(f"Total examples with reasoning: {total_with_reasoning:,}")

    # Estimate tokens per example
    avg_input_tokens = 300  # Prompt + original reasoning
    avg_output_tokens = 200  # Paraphrased reasoning

    # Model costs (per 1M tokens)
    costs = {
        "gpt-4.1-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }

    print("\nEstimated costs for FULL paraphrasing:")
    print(f"  (all {total_with_reasoning:,} examples)\n")

    for model, pricing in costs.items():
        input_cost = (total_with_reasoning * avg_input_tokens * pricing["input"]) / 1_000_000
        output_cost = (total_with_reasoning * avg_output_tokens * pricing["output"]) / 1_000_000
        total_cost = input_cost + output_cost

        print(f"  {model:20} ${total_cost:.2f}")

    print("\nEstimated costs for 100 examples per task (1,900 total):")
    for model, pricing in costs.items():
        input_cost = (1900 * avg_input_tokens * pricing["input"]) / 1_000_000
        output_cost = (1900 * avg_output_tokens * pricing["output"]) / 1_000_000
        total_cost = input_cost + output_cost

        print(f"  {model:20} ${total_cost:.2f}")

    print("\n💡 Recommendation: Start with --limit 100 using gpt-4.1-mini (~$0.20)")


def main():
    print("="*70)
    print("PARAPHRASING SETUP TEST")
    print("="*70)

    # Run tests
    if not test_api_keys():
        print("\n❌ Setup incomplete. Please configure API keys.")
        return

    if not test_llm_connection():
        print("\n❌ LLM connection failed. Check your API key.")
        return

    if not test_paraphrasing_sample():
        print("\n⚠️  Paraphrasing test had issues. Review the errors above.")
    else:
        print("\n✅ All tests passed!")

    estimate_costs()

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Run a small test:
   python scripts/paraphrase_reasoning_steps.py --task distance --limit 10 --dry-run

2. Run production job:
   python scripts/paraphrase_reasoning_steps.py --task all --limit 100 --model gpt-4.1-mini

3. Check results:
   python << 'EOF'
   import json
   with open('data/track_pure_gps/task5_distance_calculation.json') as f:
       data = json.load(f)
       paraphrased = [ex for ex in data if 'reasoning_style' in ex]
       print(f"Paraphrased: {len(paraphrased)}/{len(data)}")
   EOF

See scripts/PARAPHRASING_README.md for full documentation.
""")


if __name__ == "__main__":
    main()
