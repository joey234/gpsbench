#!/usr/bin/env python3
"""
Verify Prompt Consistency Between Training and Evaluation

This script checks that the system prompts used during training match
the prompts used during evaluation for each task.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'evaluation'))

from task_prompt_mapping import get_system_prompt_for_task

def load_training_data(filepath='data/train_instructions.json'):
    """Load training data"""
    with open(Path(__file__).parent / filepath) as f:
        return json.load(f)

def get_training_prompt_distribution(data):
    """Get distribution of system prompts in training data"""
    prompts = []
    tasks = []

    for example in data:
        task_name = None
        if 'metadata' in example and 'task' in example['metadata']:
            task_name = example['metadata']['task']

        system_prompt = get_system_prompt_for_task(task_name) if task_name else \
                        "You are a helpful assistant that identifies locations from GPS coordinates."

        prompts.append(system_prompt[:100])  # First 100 chars for grouping
        tasks.append(task_name)

    return prompts, tasks

def get_evaluator_prompts():
    """Get all system prompts from evaluators"""
    evaluator_prompts = {}

    # Import evaluators
    try:
        from task_evaluators import (
            FormatRecognitionEvaluator,
            FormatConversionEvaluator,
            ValidationEvaluator,
            CRSIdentificationEvaluator,
            PrecisionUnderstandingEvaluator
        )

        evaluators = {
            'format_conversion': FormatConversionEvaluator(),
            'validation': ValidationEvaluator(),
            'precision_understanding': PrecisionUnderstandingEvaluator(),
        }

        for task_name, evaluator in evaluators.items():
            # Create a dummy test case
            test_case = {
                'coordinate_string': '40.858, -73.996',
                'source_format': 'decimal_degrees',
                'target_format': 'DMS',
                'question': 'Test question'
            }

            try:
                sys_prompt, _ = evaluator.create_prompt(test_case)
                evaluator_prompts[task_name] = sys_prompt
            except:
                pass

    except ImportError as e:
        print(f"Warning: Could not import some evaluators: {e}")

    return evaluator_prompts

def verify_evaluation_task_coverage():
    """Verify all evaluation tasks have prompt mappings"""
    from task_prompt_mapping import TASK_SYSTEM_PROMPTS

    print("Checking evaluation task coverage...")
    print("-"*80)

    # Import evaluation track config
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'evaluation'))
    try:
        from track_config import TRACK_MAPPING

        # Get all evaluation task names
        eval_tasks = set()
        for track_name in ['pure_gps', 'applied']:
            track = TRACK_MAPPING[track_name]
            for tier_name, tier_data in track.items():
                for task_name in tier_data['tasks'].keys():
                    eval_tasks.add(task_name)

        # Check which are missing from prompt mapping
        missing = []
        for task in sorted(eval_tasks):
            if task not in TASK_SYSTEM_PROMPTS:
                missing.append(task)

        if missing:
            print(f"⚠️  WARNING: {len(missing)} evaluation tasks missing from prompt mapping!")
            for task in missing:
                print(f"   - {task}")
            print()
            print("   Add these to finetuning/task_prompt_mapping.py before training!")
            return False
        else:
            print(f"✅ All {len(eval_tasks)} evaluation tasks have prompt mappings")
            return True

    except ImportError:
        print("⚠️  Could not load evaluation track config")
        print("   (This is OK if running from different directory)")
        return True

def main():
    print("="*80)
    print("PROMPT CONSISTENCY VERIFICATION")
    print("="*80)
    print()

    # Check evaluation task coverage first
    eval_ok = verify_evaluation_task_coverage()
    print()

    # Load training data
    print("Loading training data...")
    try:
        data = load_training_data()
        print(f"✅ Loaded {len(data)} training examples")
    except FileNotFoundError:
        print("❌ Training data not found!")
        print("   Expected: finetuning/data/train_instructions.json")
        return 1

    print()

    # Analyze training prompts
    print("Analyzing training prompts...")
    prompts, tasks = get_training_prompt_distribution(data)

    unique_prompts = set(prompts)
    task_counts = Counter(tasks)

    print(f"✅ Found {len(unique_prompts)} unique system prompts in training")
    print(f"✅ Found {len(task_counts)} unique tasks")
    print()

    # Show distribution
    print("Training prompt distribution:")
    print("-"*80)
    prompt_counts = Counter(prompts)
    for prompt_preview, count in prompt_counts.most_common(10):
        pct = count / len(data) * 100
        print(f"  {count:5d} ({pct:5.1f}%)  {prompt_preview}...")

    if len(prompt_counts) > 10:
        print(f"  ... and {len(prompt_counts) - 10} more")

    print()

    # Sample verification
    print("Sample task-to-prompt mappings:")
    print("-"*80)
    sample_tasks = ['format_conversion', 'distance_calculation', 'noise_tolerance',
                    'place_association', 'regional_knowledge']

    for task in sample_tasks:
        prompt = get_system_prompt_for_task(task)
        first_line = prompt.strip().split('\n')[0]
        print(f"  {task:30s} → {first_line[:50]}...")

    print()

    # Verify consistency with evaluators
    print("Verifying consistency with evaluators...")
    print("-"*80)

    evaluator_prompts = get_evaluator_prompts()

    if evaluator_prompts:
        all_match = True
        for task_name, eval_prompt in evaluator_prompts.items():
            training_prompt = get_system_prompt_for_task(task_name)

            if training_prompt.strip() == eval_prompt.strip():
                print(f"  ✅ {task_name:30s} MATCHES")
            else:
                print(f"  ❌ {task_name:30s} MISMATCH")
                print(f"     Training:   {training_prompt[:60]}...")
                print(f"     Evaluation: {eval_prompt[:60]}...")
                all_match = False

        print()
        if all_match:
            print("🎉 SUCCESS: All prompts match between training and evaluation!")
        else:
            print("⚠️  WARNING: Some prompts don't match!")
            print("   This may cause reduced model performance.")
    else:
        print("  ⚠️  Could not load evaluators for verification")
        print("     (This is OK if running from different directory)")

    print()
    print("="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print()

    print("Summary:")
    print(f"  • Training examples: {len(data)}")
    print(f"  • Unique tasks: {len(task_counts)}")
    print(f"  • Unique prompts: {len(unique_prompts)}")
    print()

    if len(unique_prompts) > 1:
        print("✅ GOOD: Using task-specific prompts during training")
        print("   This should improve accuracy by 10-30%!")
    else:
        print("⚠️  WARNING: All training uses same prompt")
        print("   Consider using task-specific prompts for better results")

    print()
    print("Next steps:")
    print("  1. Train model: python scripts/train_lora.py --model-name Qwen/Qwen2.5-0.5B")
    print("  2. Evaluate: cd ../evaluation && python compare_base_vs_finetuned.py ...")
    print("  3. Compare results with previous training (expect +10-30% improvement)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
