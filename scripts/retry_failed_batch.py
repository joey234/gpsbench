#!/usr/bin/env python3
"""
Retry Failed Batch Samples
===========================

Helper script to retry failed samples from a completed batch job.

Usage:
    # Retry all failed samples from a batch
    python scripts/retry_failed_batch.py --batch-id batch_ABC123

    # Retry with custom max attempts
    python scripts/retry_failed_batch.py --batch-id batch_ABC123 --max-retries 3

    # Specify custom batch directory
    python scripts/retry_failed_batch.py --batch-id batch_ABC123 --batch-dir my_batches
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.llm_client import LLMClient
from run_benchmark import GPSBenchRunner


def main():
    parser = argparse.ArgumentParser(
        description="Retry failed samples from a completed batch job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--batch-id',
        type=str,
        required=True,
        help='Batch job ID to retry failed samples from'
    )
    parser.add_argument(
        '--batch-dir',
        type=str,
        default='batch_files',
        help='Directory containing batch files (default: batch_files)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum number of retry attempts (default: 2)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model to use for retries (default: same as original batch)'
    )
    parser.add_argument(
        '--reasoning-effort',
        type=str,
        choices=['minimal', 'none', 'low', 'medium', 'high'],
        default='medium',
        help='Reasoning effort for reasoning models (default: medium)'
    )

    args = parser.parse_args()

    # Load metadata
    metadata_file = os.path.join(args.batch_dir, f"{args.batch_id}_metadata.json")
    if not os.path.exists(metadata_file):
        print(f"❌ Error: Metadata file not found: {metadata_file}")
        print(f"Make sure batch ID is correct and batch directory is set properly.")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Determine model
    if args.model:
        model = args.model
    else:
        # Try to infer from batch file
        batch_file = metadata.get("batch_file")
        if batch_file and os.path.exists(batch_file):
            with open(batch_file, 'r') as f:
                first_line = f.readline()
                try:
                    first_request = json.loads(first_line)
                    model = first_request.get("body", {}).get("model", "gpt-4")
                except:
                    model = "gpt-4"
        else:
            model = "gpt-4"

    print(f"\n{'='*80}")
    print(f"Retry Failed Batch Samples")
    print(f"{'='*80}")
    print(f"Batch ID: {args.batch_id}")
    print(f"Model: {model}")
    print(f"Max Retries: {args.max_retries}")
    print(f"{'='*80}\n")

    # Initialize LLM client
    llm_client = LLMClient(
        provider="auto",
        model=model,
        temperature=0.0,
        max_tokens=7000,
        timeout=120,
        reasoning_effort=args.reasoning_effort
    )

    # Initialize runner
    runner = GPSBenchRunner(
        llm_client=llm_client,
        data_dir="data",
        results_dir="results"
    )

    # First, process the batch to identify failures
    print("Loading batch results to identify failed samples...")
    task_data = metadata["task_data"]
    task_info = metadata["task_info"]

    # Retrieve batch results
    results_file = os.path.join(args.batch_dir, f"{args.batch_id}_results.jsonl")
    llm_responses = llm_client.retrieve_batch_results(
        args.batch_id,
        results_file,
        expected_count=len(task_data)
    )

    # Find failed samples
    failed_indices = []
    for idx, llm_response in enumerate(llm_responses):
        if llm_response.error:
            failed_indices.append(idx)

    if not failed_indices:
        print("✅ No failed samples found! All samples completed successfully.")
        return

    print(f"\n⚠️  Found {len(failed_indices)} failed samples:")
    print(f"   Failed indices: {failed_indices[:20]}{'...' if len(failed_indices) > 20 else ''}")
    print(f"\nStarting retry process...")

    # Retry failed samples
    retry_result = runner.retry_failed_batch(
        batch_id=args.batch_id,
        failed_indices=failed_indices,
        batch_dir=args.batch_dir,
        max_retries=args.max_retries
    )

    # Summary
    print(f"\n{'='*80}")
    print("Retry Summary")
    print(f"{'='*80}")
    print(f"Original failures: {len(failed_indices)}")
    print(f"Successfully retried: {len(retry_result)}")
    print(f"Still failing: {len(failed_indices) - len(retry_result)}")

    if retry_result:
        # Save updated results
        # Re-process the entire batch with merged retry results
        print(f"\nRe-processing batch with retry results...")
        final_result = runner.process_batch_results(
            batch_id=args.batch_id,
            batch_dir=args.batch_dir,
            auto_retry=False  # Don't retry again
        )

        # Manually merge retry results
        for idx, retry_data in retry_result.items():
            final_result["results"][idx] = retry_data

        # Recalculate metrics
        correct = sum(1 for r in final_result["results"] if r.get("correct", False))
        final_result["correct"] = correct
        final_result["accuracy"] = (correct / len(task_data) * 100) if task_data else 0.0

        # Save updated results
        updated_results_file = os.path.join(args.batch_dir, f"{args.batch_id}_with_retries.json")
        with open(updated_results_file, 'w') as f:
            json.dump(final_result, f, indent=2)

        print(f"✅ Updated results saved to: {updated_results_file}")
        print(f"   Final accuracy: {final_result['accuracy']:.2f}%")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
