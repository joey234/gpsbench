#!/usr/bin/env python3
"""
Transform cardinal direction yes/no tasks to multiple choice format.

Before:
  Question: "Is X west of Y?"
  Answer: true/false

After:
  Question: "What is the cardinal direction from Y to X?"
  Options: ["North", "South", "East", "West"]
  Answer: "West"
"""

import json
import re
import argparse
from pathlib import Path


def extract_direction_from_question(question: str) -> str:
    """Extract the cardinal direction from a yes/no question."""
    # Pattern: "Is X north/south/east/west of Y?"
    match = re.search(r'\b(north|south|east|west)\b', question.lower())
    if match:
        return match.group(1).capitalize()
    raise ValueError(f"Could not extract direction from: {question}")


def compute_actual_direction(point_a: dict, point_b: dict) -> str:
    """
    Compute the primary cardinal direction from point_b to point_a.

    For questions like "Is A [direction] of B?", we want the direction
    from B's perspective looking at A.
    """
    lat_a, lon_a = point_a["lat"], point_a["lon"]
    lat_b, lon_b = point_b["lat"], point_b["lon"]

    lat_diff = abs(lat_a - lat_b)
    lon_diff = abs(lon_a - lon_b)

    # Handle longitude wrapping (e.g., -170 to 170 is actually 20 degrees apart)
    if lon_diff > 180:
        lon_diff = 360 - lon_diff

    # Determine primary direction based on larger difference
    if lat_diff >= lon_diff:
        # Primarily north/south
        return "North" if lat_a > lat_b else "South"
    else:
        # Primarily east/west
        # Need to handle international date line
        raw_lon_diff = lon_a - lon_b
        if raw_lon_diff > 180:
            raw_lon_diff -= 360
        elif raw_lon_diff < -180:
            raw_lon_diff += 360
        return "East" if raw_lon_diff > 0 else "West"


def get_direction_for_item(item: dict) -> str:
    """
    Get the actual cardinal direction for an item.

    If the original answer was True, the asked direction is correct.
    If False, we need to compute the actual direction.
    """
    asked_direction = extract_direction_from_question(item["question"])
    original_answer = item["ground_truth"]["answer"]

    if original_answer:
        return asked_direction
    else:
        # Compute the actual direction
        return compute_actual_direction(item["point_a"], item["point_b"])


def transform_item(item: dict) -> dict:
    """Transform a single item from yes/no to multiple choice format."""
    point_a = item["point_a"]
    point_b = item["point_b"]

    # Get the actual direction
    actual_direction = get_direction_for_item(item)

    # Create new question format
    new_question = f"What is the cardinal direction from {point_b['name']} to {point_a['name']}?"

    # Build new item
    new_item = {
        "task": item["task"],
        "subtask": "cardinal_direction_mcq",  # New subtask name
        "point_a": point_a,
        "point_b": point_b,
        "question": new_question,
        "options": ["North", "South", "East", "West"],
        "ground_truth": {
            "answer": actual_direction,
            "explanation": f"{point_a['name']} is {actual_direction.lower()} of {point_b['name']}"
        }
    }

    # Preserve metadata if present
    if "metadata" in item:
        new_item["metadata"] = item["metadata"]

    # Preserve distance_km if present in ground_truth
    if "distance_km" in item["ground_truth"]:
        new_item["ground_truth"]["distance_km"] = item["ground_truth"]["distance_km"]

    return new_item


def is_already_transformed(item: dict) -> bool:
    """Check if an item is already in MCQ format."""
    return "options" in item and item["question"].startswith("What is the cardinal direction")


def transform_file(input_path: Path, output_path: Path = None):
    """Transform an entire task file."""
    print(f"Reading: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"  Found {len(data)} items")

    # Check if already transformed
    if data and is_already_transformed(data[0]):
        print(f"  Already transformed, skipping")
        return len(data), {}

    # Transform all items
    transformed = []
    direction_counts = {"North": 0, "South": 0, "East": 0, "West": 0}

    for item in data:
        new_item = transform_item(item)
        transformed.append(new_item)
        direction_counts[new_item["ground_truth"]["answer"]] += 1

    # Use input path if no output specified (overwrite)
    if output_path is None:
        output_path = input_path

    print(f"  Direction distribution: {direction_counts}")
    print(f"Writing: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(transformed, f, indent=2)

    return len(transformed), direction_counts


def main():
    parser = argparse.ArgumentParser(description="Transform cardinal direction tasks to MCQ format")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--task", choices=["pure_gps", "applied", "both"], default="both",
                        help="Which task(s) to transform")
    parser.add_argument("--include-splits", action="store_true", help="Also transform split files")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "data"

    tasks = []
    if args.task in ["pure_gps", "both"]:
        tasks.append(("track_pure_gps", "task10_relative_position"))
    if args.task in ["applied", "both"]:
        tasks.append(("track_applied", "task5_relative_position"))

    for track, task_name in tasks:
        # Main task file
        main_file = base_dir / track / f"{task_name}.json"
        files_to_transform = [main_file]

        # Add split files if requested
        if args.include_splits:
            splits_dir = base_dir / track / "splits"
            for suffix in ["_train.json", "_dev.json", "_test.json"]:
                split_file = splits_dir / f"{task_name}{suffix}"
                if split_file.exists():
                    files_to_transform.append(split_file)

        for input_path in files_to_transform:
            if args.dry_run:
                print(f"\n[DRY RUN] Would transform: {input_path}")
                with open(input_path, 'r') as f:
                    data = json.load(f)

                # Show sample transformation
                sample = data[0]
                transformed = transform_item(sample)
                print(f"  Original question: {sample['question']}")
                print(f"  Original answer: {sample['ground_truth']['answer']}")
                print(f"  New question: {transformed['question']}")
                print(f"  New answer: {transformed['ground_truth']['answer']}")
                print(f"  Options: {transformed['options']}")
            else:
                count, dist = transform_file(input_path)
                print(f"  Transformed {count} items\n")


if __name__ == "__main__":
    main()
