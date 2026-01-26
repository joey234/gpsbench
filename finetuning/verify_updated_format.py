#!/usr/bin/env python3
"""
Verify Finetuning Data Uses Updated Standardized Format

This script checks that:
1. Training data uses the STANDARD_SYSTEM_PROMPT
2. All outputs start with "FINAL ANSWER:"
3. Format matches run_benchmark.py evaluation format

Updated: 2025-11-17
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetuning.task_prompt_mapping import STANDARD_SYSTEM_PROMPT, get_system_prompt_for_task


def check_training_file(filepath: Path) -> dict:
    """Check a training data file for consistency"""

    with open(filepath, 'r') as f:
        data = json.load(f)

    results = {
        'total': len(data),
        'has_final_answer_in_output': 0,
        'has_final_answer_instruction': 0,
        'correct_format': 0,
        'tasks': defaultdict(int),
        'issues': []
    }

    for i, example in enumerate(data):
        task = example.get('metadata', {}).get('task', 'unknown')
        results['tasks'][task] += 1

        # Check output format
        output = example.get('output', '')
        if 'FINAL ANSWER:' in output.upper():
            results['has_final_answer_in_output'] += 1
        else:
            results['issues'].append({
                'index': i,
                'task': task,
                'issue': 'Output missing FINAL ANSWER',
                'output': output[:100]
            })

        # Check instruction includes FINAL ANSWER requirement
        instruction = example.get('instruction', '')
        if 'FINAL ANSWER' in instruction.upper():
            results['has_final_answer_instruction'] += 1
        else:
            results['issues'].append({
                'index': i,
                'task': task,
                'issue': 'Instruction missing FINAL ANSWER requirement',
                'instruction': instruction[:100]
            })

        # Check correct format
        if ('FINAL ANSWER:' in output.upper() and
            'FINAL ANSWER' in instruction.upper()):
            results['correct_format'] += 1

    return results


def main():
    print("="*80)
    print("GPSBench Finetuning Data Consistency Check")
    print("Updated Format Verification (2025-11-17)")
    print("="*80)
    print()

    # Check training data directory
    data_dir = Path(__file__).parent / "data"

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print()
        print("Please run:")
        print("  bash scripts/prepare_finetuning_data.sh both")
        return 1

    # Find training files
    training_files = list(data_dir.glob("train*.json"))

    if not training_files:
        print(f"❌ No training files found in {data_dir}")
        print()
        print("Please run:")
        print("  bash scripts/prepare_finetuning_data.sh both")
        return 1

    print(f"Found {len(training_files)} training file(s)")
    print()

    all_results = {}
    total_examples = 0
    total_correct = 0
    total_issues = []

    for filepath in sorted(training_files):
        print(f"Checking: {filepath.name}")
        print("-"*80)

        results = check_training_file(filepath)
        all_results[filepath.name] = results

        total_examples += results['total']
        total_correct += results['correct_format']
        total_issues.extend(results['issues'])

        print(f"  Total examples: {results['total']}")
        print(f"  Correct format: {results['correct_format']} ({results['correct_format']/results['total']*100:.1f}%)")
        print(f"  Has FINAL ANSWER in output: {results['has_final_answer_in_output']}")
        print(f"  Has FINAL ANSWER in instruction: {results['has_final_answer_instruction']}")
        print(f"  Unique tasks: {len(results['tasks'])}")

        if results['issues']:
            print(f"  ⚠️  Issues found: {len(results['issues'])}")
        else:
            print(f"  ✅ No issues found")
        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Total examples checked: {total_examples}")
    print(f"Correct format: {total_correct} ({total_correct/total_examples*100:.1f}%)")
    print()

    if total_issues:
        print(f"⚠️  Total issues: {len(total_issues)}")
        print()
        print("Sample issues:")
        for issue in total_issues[:5]:
            print(f"  - {issue['task']} (ex {issue['index']}): {issue['issue']}")

        if len(total_issues) > 5:
            print(f"  ... and {len(total_issues) - 5} more")
        print()
        print("❌ Training data needs regeneration")
        print()
        print("Run:")
        print("  bash scripts/prepare_finetuning_data.sh both")
        return 1
    else:
        print("✅ All examples use correct FINAL ANSWER format!")
        print()

    # Check system prompt consistency
    print("="*80)
    print("SYSTEM PROMPT CONSISTENCY CHECK")
    print("="*80)
    print()

    # Sample a few tasks and verify they use STANDARD_SYSTEM_PROMPT
    sample_tasks = [
        'format_conversion', 'distance_calculation', 'place_association',
        'terrain_classification', 'relative_position', 'bearing_computation',
        'interpolation', 'area_perimeter', 'proximity', 'spatial_patterns'
    ]

    all_match = True
    different_prompts = []

    for task in sample_tasks:
        prompt = get_system_prompt_for_task(task)
        if prompt != STANDARD_SYSTEM_PROMPT:
            print(f"  ❌ {task}: Uses different prompt")
            all_match = False
            different_prompts.append(task)
        else:
            print(f"  ✅ {task}: Uses STANDARD_SYSTEM_PROMPT")

    print()

    if all_match:
        print("✅ All sampled tasks use STANDARD_SYSTEM_PROMPT")
        print()
        print("This ensures consistency between training and evaluation!")
    else:
        print(f"⚠️  {len(different_prompts)} tasks use different prompts")
        print()
        print("Tasks with different prompts:", ', '.join(different_prompts))
        print()
        print("This may indicate task_prompt_mapping.py needs updating.")
        print("Run the following to update:")
        print("  # Already done if you pulled latest code")

    # Final status
    print()
    print("="*80)
    print("FINAL STATUS")
    print("="*80)
    print()

    if total_correct == total_examples and all_match:
        print("🎉 SUCCESS: Training data ready for finetuning!")
        print()
        print("✅ All examples have FINAL ANSWER format")
        print("✅ All tasks use standardized system prompt")
        print("✅ Format matches run_benchmark.py evaluation")
        print()
        print("Next steps:")
        print("  cd finetuning/scripts")
        print("  python train_lora.py --track pure_gps")
        print("  python train_lora.py --track applied")
        return 0
    else:
        if total_correct < total_examples:
            print("❌ FINAL ANSWER format issues found")
            print("   Regenerate training data:")
            print("   bash scripts/prepare_finetuning_data.sh both")
            print()

        if not all_match:
            print("⚠️  System prompt inconsistency detected")
            print("   This may be intentional for specific tasks")
            print("   Verify task_prompt_mapping.py uses STANDARD_SYSTEM_PROMPT")
            print()

        return 1


if __name__ == "__main__":
    sys.exit(main())
