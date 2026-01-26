#!/usr/bin/env python3
"""
Re-run specific tasks for models with inconsistent data.

This script re-runs only the tasks that differ from the canonical data (GPT-4.1)
and saves results to the existing model directories.

- Uses BATCH MODE for OpenAI and Gemini (cost savings, handles rate limits)
- Uses CONCURRENT MODE (10 workers) for OpenRouter

Usage:
    python rerun_inconsistent_tasks.py --dry-run          # Show what would be run
    python rerun_inconsistent_tasks.py                    # Run all needed re-runs
    python rerun_inconsistent_tasks.py --model gpt-5.1    # Run only for specific model
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Define what needs to be re-run
# Updated: Re-run cardinal direction tasks (converted from Yes/No to MCQ format)
CARDINAL_DIRECTION_TASKS = [
    ("pure_gps", "task10_relative_position"),
    ("applied", "task5_relative_position"),
]

RERUNS_NEEDED = {
    # Re-run GPT-5-mini/nano with correct reasoning_effort='low' setting
    "gpt-5-mini-2025-08-07": {
        "dir": "gpsbench_gpt-5-mini-2025-08-07_20251205_142538",
        "provider": "openai",
        "mode": "concurrent",
        "tasks": CARDINAL_DIRECTION_TASKS
    },
    "gpt-5-nano-2025-08-07": {
        "dir": "gpsbench_gpt-5-nano-2025-08-07_20251205_162633",
        "provider": "openai",
        "mode": "concurrent",
        "tasks": CARDINAL_DIRECTION_TASKS
    },
    # Re-run Gemini
    "gemini-2.5-pro": {
        "dir": "gpsbench_gemini-2.5-pro_20251210_182109",
        "provider": "gemini",
        "mode": "concurrent",
        "tasks": CARDINAL_DIRECTION_TASKS
    },
}

# Canonical data source (GPT-4.1)
CANONICAL_DIR = "gpsbench_gpt-4.1_20251201_235326"

TASK_NAMES = {
    "pure_gps_task1_format_conversion": "Format Conversion",
    "pure_gps_task3_coordinate_system_transformation": "Coordinate System Transformation",
    "pure_gps_task4_distance_calculation": "Distance Calculation",
    "pure_gps_task5_bearing_computation": "Bearing Computation",
    "pure_gps_task6_interpolation": "Coordinate Interpolation",
    "pure_gps_task7_area_perimeter": "Polygon Area",
    "pure_gps_task8_bounding_box": "Bounding Box",
    "pure_gps_task9_route_geometry": "Route Geometry",
    "pure_gps_task10_relative_position": "Relative Position",
    "applied_task1_place_association": "Place Association",
    "applied_task3_name_disambiguation": "Name Disambiguation",
    "applied_task5_relative_position": "Relative Position",
    "applied_task6_proximity": "Proximity & Nearest Neighbor",
    "applied_task7_route_analysis": "Route Analysis",
    "applied_task8_boundary_analysis": "Boundary Analysis",
    "applied_task9_spatial_patterns": "Spatial Patterns",
    "applied_task11_missing_data": "Missing Data Inference",
    "applied_task13_terrain_classification": "Terrain Classification",
}


def get_canonical_data(results_dir: Path, track: str, task: str) -> list:
    """Load canonical data from GPT-4.1 results."""
    canonical_path = results_dir / CANONICAL_DIR / "task_results" / f"{track}_{task}.json"
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical data not found: {canonical_path}")

    with open(canonical_path) as f:
        data = json.load(f)

    return data.get("results", [])


def get_test_data(track: str, task: str) -> list:
    """Load test data from data directory."""
    test_path = Path("data") / f"track_{track}" / "splits" / f"{task}_test.json"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    with open(test_path) as f:
        return json.load(f)


def run_task_concurrent(
    model: str,
    provider: str,
    track: str,
    task: str,
    output_dir: Path,
    results_dir: Path,
    max_workers: int = 10
) -> bool:
    """Run a single task using concurrent mode (for OpenRouter)."""

    task_output_file = output_dir / "task_results" / f"{track}_{task}.json"

    # Load test data (has full example structure needed for evaluation)
    test_data = get_test_data(track, task)
    print(f"  Loaded {len(test_data)} samples from test data")
    print(f"  Mode: CONCURRENT (max {max_workers} workers)")

    # Import dependencies
    sys.path.insert(0, str(Path(__file__).parent))
    from run_benchmark import GPSBenchRunner
    from evaluation.llm_client import LLMClient
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Create client and runner (match run_benchmark.py settings)
    client = LLMClient(provider=provider, model=model, max_tokens=8192, reasoning_effort='low')
    runner = GPSBenchRunner(
        llm_client=client,
        data_dir="data",
        results_dir=str(results_dir),
        max_workers=max_workers
    )

    results = []
    correct = 0

    def evaluate_single(idx_example):
        idx, example = idx_example
        ground_truth = example.get("ground_truth", {})

        # Construct prompt from example (returns system_prompt, user_prompt tuple)
        system_prompt, user_prompt = runner.create_prompt(example, task)

        try:
            llm_response = client.generate(user_prompt, system_prompt=system_prompt)
            response_text = llm_response.text
            is_correct = runner.simple_evaluate(response_text, example)

            return {
                "index": idx,
                "prompt": user_prompt,
                "ground_truth": ground_truth,
                "model_response": response_text,
                "is_correct": is_correct
            }
        except Exception as e:
            return {
                "index": idx,
                "prompt": user_prompt,
                "ground_truth": ground_truth,
                "model_response": f"ERROR: {str(e)}",
                "is_correct": False,
                "error": str(e)
            }

    # Run with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single, (i, ex)): i
            for i, ex in enumerate(test_data)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {task}"):
            result = future.result()
            results.append(result)
            if result.get("is_correct"):
                correct += 1

    # Sort by index
    results.sort(key=lambda x: x["index"])

    # Calculate accuracy
    accuracy = (correct / len(results) * 100) if results else 0

    # Save results
    save_task_result(task_output_file, task, track, model, provider, results, correct, accuracy)

    print(f"  ✓ Completed: {correct}/{len(results)} = {accuracy:.1f}%")
    return True


def run_task_batch(
    model: str,
    provider: str,
    track: str,
    task: str,
    output_dir: Path,
    results_dir: Path,
    batch_dir: Path
) -> bool:
    """Run a single task using batch mode (for OpenAI and Gemini)."""

    task_output_file = output_dir / "task_results" / f"{track}_{task}.json"

    # Load test data (has full example structure needed for evaluation)
    test_data = get_test_data(track, task)
    print(f"  Loaded {len(test_data)} samples from test data")
    print(f"  Mode: BATCH")

    # Import dependencies
    sys.path.insert(0, str(Path(__file__).parent))
    from run_benchmark import GPSBenchRunner
    from evaluation.llm_client import LLMClient

    # Create client and runner (match run_benchmark.py settings)
    client = LLMClient(provider=provider, model=model, max_tokens=8192, reasoning_effort='low')
    runner = GPSBenchRunner(
        llm_client=client,
        data_dir="data",
        results_dir=str(results_dir)
    )

    # Create prompts from test data (returns list of (system_prompt, user_prompt) tuples)
    prompt_pairs = [runner.create_prompt(ex, task) for ex in test_data]
    system_prompt = prompt_pairs[0][0] if prompt_pairs else ""  # System prompt is same for all
    user_prompts = [p[1] for p in prompt_pairs]

    # Prepare batch requests
    batch_dir.mkdir(exist_ok=True, parents=True)
    batch_filename = f"rerun_{track}_{task}_{model.replace('/', '_')}.jsonl"
    batch_filepath = batch_dir / batch_filename

    # Create batch file
    client.create_batch_file(
        prompts=user_prompts,
        system_prompt=system_prompt,
        output_file=str(batch_filepath)
    )
    print(f"  Created batch file: {batch_filepath}")

    # Submit batch
    print(f"  Submitting batch job...")
    batch_id = client.submit_batch(str(batch_filepath))
    print(f"  Batch ID: {batch_id}")

    # Monitor batch progress
    print(f"  Waiting for batch to complete...")
    poll_interval = 30  # seconds
    max_wait = 3600  # 1 hour max
    elapsed = 0

    while elapsed < max_wait:
        status = client.check_batch_status(batch_id)
        state = status.get("status", "unknown")

        if state in ["completed", "SUCCEEDED", "JOB_STATE_SUCCEEDED"]:
            print(f"  Batch completed!")
            break
        elif state in ["failed", "FAILED", "JOB_STATE_FAILED", "expired", "cancelled"]:
            print(f"  Batch failed with state: {state}")
            return False
        else:
            completed = status.get("completed", 0)
            total = status.get("total", len(test_data))
            print(f"  Status: {state} ({completed}/{total} completed)")
            time.sleep(poll_interval)
            elapsed += poll_interval

    if elapsed >= max_wait:
        print(f"  Timeout waiting for batch")
        return False

    # Get batch results
    print(f"  Retrieving batch results...")
    batch_results = client.retrieve_batch_results(batch_id)

    # Process results
    results = []
    correct = 0

    for idx, (example, batch_result) in enumerate(zip(test_data, batch_results)):
        prompt = user_prompts[idx]
        ground_truth = example.get("ground_truth", {})
        response = batch_result.get("response", batch_result.get("content", ""))

        if isinstance(response, dict):
            response = response.get("content", str(response))

        is_correct = runner.simple_evaluate(response, example)

        if is_correct:
            correct += 1

        results.append({
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "is_correct": is_correct
        })

    # Calculate accuracy
    accuracy = (correct / len(results) * 100) if results else 0

    # Save results
    save_task_result(task_output_file, task, track, model, provider, results, correct, accuracy)

    print(f"  ✓ Completed: {correct}/{len(results)} = {accuracy:.1f}%")
    return True


def save_task_result(
    output_file: Path,
    task: str,
    track: str,
    model: str,
    provider: str,
    results: list,
    correct: int,
    accuracy: float
):
    """Save task results to file."""
    task_result = {
        "task": task,
        "track": track,
        "model": model,
        "provider": provider,
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, 'w') as f:
        json.dump(task_result, f, indent=2)

    print(f"  Saved to: {output_file}")


def run_all_batches_parallel(reruns: dict, results_dir: Path, batch_dir: Path) -> dict:
    """Submit all batches in parallel, monitor together, then process results."""
    from run_benchmark import GPSBenchRunner
    from evaluation.llm_client import LLMClient

    batch_dir.mkdir(exist_ok=True, parents=True)

    # Phase 1: Prepare all batches and submit them
    print("\n" + "="*70)
    print("PHASE 1: Submitting all batches")
    print("="*70)

    batch_jobs = []  # List of (batch_id, client, task_info, test_data, runner, user_prompts)

    for model, info in reruns.items():
        output_dir = results_dir / info["dir"]
        if not output_dir.exists():
            print(f"⚠️ Output directory not found: {output_dir}")
            continue

        provider = info["provider"]

        for track, task in info["tasks"]:
            print(f"\n  Preparing: {model} / {track}_{task}")

            # Load test data
            test_data = get_test_data(track, task)
            print(f"    Loaded {len(test_data)} samples")

            # Create client and runner
            client = LLMClient(provider=provider, model=model, max_tokens=8192, reasoning_effort='low')
            runner = GPSBenchRunner(
                llm_client=client,
                data_dir="data",
                results_dir=str(results_dir)
            )

            # Create prompts
            prompt_pairs = [runner.create_prompt(ex, task) for ex in test_data]
            system_prompt = prompt_pairs[0][0] if prompt_pairs else ""
            user_prompts = [p[1] for p in prompt_pairs]

            # Create batch file
            batch_filename = f"rerun_{track}_{task}_{model.replace('/', '_')}.jsonl"
            batch_filepath = batch_dir / batch_filename
            client.create_batch_file(
                prompts=user_prompts,
                system_prompt=system_prompt,
                output_file=str(batch_filepath)
            )

            # Submit batch
            batch_id = client.submit_batch(str(batch_filepath))
            print(f"    Submitted batch: {batch_id}")

            batch_jobs.append({
                "batch_id": batch_id,
                "client": client,
                "model": model,
                "provider": provider,
                "track": track,
                "task": task,
                "test_data": test_data,
                "runner": runner,
                "user_prompts": user_prompts,
                "output_dir": output_dir,
                "status": "submitted"
            })

    print(f"\n  Total batches submitted: {len(batch_jobs)}")

    # Phase 2: Monitor all batches together
    print("\n" + "="*70)
    print("PHASE 2: Monitoring batches")
    print("="*70)

    poll_interval = 15  # seconds
    max_wait = 3600  # 1 hour
    elapsed = 0

    while elapsed < max_wait:
        pending = [j for j in batch_jobs if j["status"] == "submitted"]
        if not pending:
            break

        print(f"\n  [{elapsed}s] Checking {len(pending)} pending batches...")

        for job in pending:
            status = job["client"].check_batch_status(job["batch_id"])
            state = status.get("status", "unknown")
            completed = status.get("request_counts", {}).get("completed", 0)
            total = status.get("request_counts", {}).get("total", len(job["test_data"]))

            if state in ["completed", "SUCCEEDED", "JOB_STATE_SUCCEEDED"]:
                job["status"] = "completed"
                print(f"    ✓ {job['model']}/{job['task']}: COMPLETED")
            elif state in ["failed", "FAILED", "JOB_STATE_FAILED", "expired", "cancelled"]:
                job["status"] = "failed"
                print(f"    ✗ {job['model']}/{job['task']}: FAILED ({state})")
            else:
                print(f"    ○ {job['model']}/{job['task']}: {state} ({completed}/{total})")

        pending = [j for j in batch_jobs if j["status"] == "submitted"]
        if pending:
            time.sleep(poll_interval)
            elapsed += poll_interval

    # Phase 3: Retrieve and process results
    print("\n" + "="*70)
    print("PHASE 3: Processing results")
    print("="*70)

    results_summary = {}

    for job in batch_jobs:
        if job["status"] != "completed":
            print(f"\n  Skipping {job['model']}/{job['task']}: {job['status']}")
            continue

        print(f"\n  Processing: {job['model']}/{job['task']}")

        # Retrieve results
        batch_results = job["client"].retrieve_batch_results(job["batch_id"])

        # Process results
        results = []
        correct = 0

        for idx, (example, batch_result) in enumerate(zip(job["test_data"], batch_results)):
            prompt = job["user_prompts"][idx]
            ground_truth = example.get("ground_truth", {})

            # Extract response text from LLMResponse
            if hasattr(batch_result, 'text'):
                response = batch_result.text
            elif isinstance(batch_result, dict):
                response = batch_result.get("response", batch_result.get("content", ""))
            else:
                response = str(batch_result)

            is_correct = job["runner"].simple_evaluate(response, example)
            if is_correct:
                correct += 1

            results.append({
                "index": idx,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "model_response": response,
                "is_correct": is_correct
            })

        accuracy = (correct / len(results) * 100) if results else 0

        # Save results
        task_output_file = job["output_dir"] / "task_results" / f"{job['track']}_{job['task']}.json"
        save_task_result(task_output_file, job["task"], job["track"], job["model"], job["provider"], results, correct, accuracy)

        print(f"    ✓ {correct}/{len(results)} = {accuracy:.1f}%")

        # Track for summary rebuild
        model_key = job["model"]
        if model_key not in results_summary:
            results_summary[model_key] = {"output_dir": job["output_dir"], "completed": 0}
        results_summary[model_key]["completed"] += 1

    # Rebuild summaries
    print("\n" + "="*70)
    print("PHASE 4: Rebuilding summaries")
    print("="*70)

    for model, info in results_summary.items():
        print(f"\n  Rebuilding summary for {model}...")
        rebuild_summary(info["output_dir"])

    return results_summary


def rebuild_summary(output_dir: Path):
    """Rebuild summary.json from task results."""
    summary_path = output_dir / "summary.json"

    if not summary_path.exists():
        print(f"  Warning: No summary.json found at {output_dir}")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    task_results_dir = output_dir / "task_results"

    tracks = {"pure_gps": {"tasks_summary": []}, "applied": {"tasks_summary": []}}

    for task_file in sorted(task_results_dir.glob("*.json")):
        task_stem = task_file.stem
        track = "pure_gps" if task_stem.startswith("pure_gps") else "applied"

        with open(task_file) as f:
            task_data = json.load(f)

        task_name = TASK_NAMES.get(task_stem, task_stem)
        task_filename = task_stem.replace(f"{track}_", "") + ".json"

        tracks[track]["tasks_summary"].append({
            "task_name": task_name,
            "task_file": task_filename,
            "total": task_data.get("total", 0),
            "correct": task_data.get("correct", 0),
            "accuracy": task_data.get("accuracy", 0)
        })

    for track_name, track_data in tracks.items():
        tasks = track_data["tasks_summary"]
        total_correct = sum(t["correct"] for t in tasks)
        total_cases = sum(t["total"] for t in tasks)

        track_data["track_name"] = track_name
        track_data["track_display_name"] = "Pure GPS Track" if track_name == "pure_gps" else "Applied Track"
        track_data["total_tasks"] = len(tasks)
        track_data["total_correct"] = total_correct
        track_data["total_cases"] = total_cases
        track_data["overall_accuracy"] = (total_correct / total_cases * 100) if total_cases > 0 else 0

    overall_correct = tracks["pure_gps"]["total_correct"] + tracks["applied"]["total_correct"]
    overall_total = tracks["pure_gps"]["total_cases"] + tracks["applied"]["total_cases"]

    summary["tracks"] = tracks
    summary["overall"] = {
        "total_cases": overall_total,
        "total_correct": overall_correct,
        "overall_accuracy": (overall_correct / overall_total * 100) if overall_total > 0 else 0
    }
    summary["_rerun_rebuild"] = datetime.now().isoformat()

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Rebuilt summary: {summary['overall']['overall_accuracy']:.1f}% overall")


def run_task(
    model: str,
    provider: str,
    mode: str,
    track: str,
    task: str,
    output_dir: Path,
    results_dir: Path,
    batch_dir: Path,
    dry_run: bool = False,
    max_workers: int = 10
) -> bool:
    """Run a single task for a model."""

    task_output_file = output_dir / "task_results" / f"{track}_{task}.json"

    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Provider: {provider}")
    print(f"Task: {track}/{task}")
    print(f"Mode: {mode.upper()}")
    print(f"Output: {task_output_file}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Would run this task")
        return True

    if mode == "batch":
        return run_task_batch(
            model=model,
            provider=provider,
            track=track,
            task=task,
            output_dir=output_dir,
            results_dir=results_dir,
            batch_dir=batch_dir
        )
    else:  # concurrent
        return run_task_concurrent(
            model=model,
            provider=provider,
            track=track,
            task=task,
            output_dir=output_dir,
            results_dir=results_dir,
            max_workers=max_workers
        )


def main():
    parser = argparse.ArgumentParser(description="Re-run inconsistent tasks")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument("--model", type=str, help="Run only for specific model")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--batch-dir", type=str, default="batch_files", help="Batch files directory")
    parser.add_argument("--max-workers", type=int, default=10, help="Max concurrent workers (for OpenRouter)")
    parser.add_argument("--parallel-batch", action="store_true", help="Submit all batches in parallel, then monitor together")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    batch_dir = Path(args.batch_dir)

    print("="*70)
    print("GPSBench Inconsistent Task Re-runner")
    print("="*70)
    print(f"\nCanonical data: {CANONICAL_DIR}")
    print(f"Results directory: {results_dir}")
    print(f"Batch directory: {batch_dir}")
    print(f"\nModes:")
    print(f"  - OpenAI/Gemini: BATCH (auto submit, monitor, retrieve)")
    print(f"  - OpenRouter: CONCURRENT (max {args.max_workers} workers)")
    if args.parallel_batch:
        print(f"  - PARALLEL BATCH: Submit all at once, monitor together")

    # Filter by model if specified
    reruns = RERUNS_NEEDED.copy()
    if args.model:
        if args.model in reruns:
            reruns = {args.model: reruns[args.model]}
        else:
            print(f"\nError: Model '{args.model}' not in re-run list")
            print(f"Available models: {list(RERUNS_NEEDED.keys())}")
            return 1

    # Count total tasks
    total_tasks = sum(len(info["tasks"]) for info in reruns.values())
    print(f"\nTotal re-runs needed: {total_tasks}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]")

    # Use parallel batch mode if requested
    if args.parallel_batch and not args.dry_run:
        run_all_batches_parallel(reruns, results_dir, batch_dir)
        return 0

    # Run each task
    completed = 0
    failed = 0

    for model, info in reruns.items():
        output_dir = results_dir / info["dir"]

        if not output_dir.exists():
            print(f"\n⚠️ Output directory not found: {output_dir}")
            continue

        for track, task in info["tasks"]:
            try:
                success = run_task(
                    model=model,
                    provider=info["provider"],
                    mode=info["mode"],
                    track=track,
                    task=task,
                    output_dir=output_dir,
                    results_dir=results_dir,
                    batch_dir=batch_dir,
                    dry_run=args.dry_run,
                    max_workers=args.max_workers
                )
                if success:
                    completed += 1
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

        # Rebuild summary after all tasks for this model
        if not args.dry_run:
            print(f"\n  Rebuilding summary for {model}...")
            rebuild_summary(output_dir)

    print(f"\n{'='*70}")
    print(f"COMPLETED: {completed}/{total_tasks} tasks")
    if failed:
        print(f"FAILED: {failed} tasks")
    print(f"{'='*70}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
