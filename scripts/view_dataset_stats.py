#!/usr/bin/env python3
"""
View GPSBench Layer A Dataset Statistics
Provides a comprehensive overview of the generated test cases
"""

import json
from pathlib import Path
from collections import Counter


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_dataset():
    """Analyze and display dataset statistics"""
    data_dir = Path("data/layer_a")

    print("=" * 80)
    print("GPSBench - Layer A Dataset Statistics")
    print("=" * 80)

    # Load summary
    summary = load_json(data_dir / "dataset_summary.json")

    print(f"\n📊 OVERALL STATISTICS")
    print(f"{'─' * 80}")
    print(f"  Total Locations:        {summary['total_locations']}")
    print(f"  Total Test Cases:       {summary['total_test_cases']:,}")
    print(f"  Number of Tasks:        {len(summary['tasks'])}")

    # Geographic distribution
    print(f"\n🌍 GEOGRAPHIC DISTRIBUTION")
    print(f"{'─' * 80}")
    geo_dist = summary['geographic_coverage']

    # Sort by count
    sorted_regions = sorted(geo_dist.items(), key=lambda x: x[1], reverse=True)

    for region, count in sorted_regions:
        percentage = (count / summary['total_locations']) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {region:25s} {count:3d} ({percentage:5.1f}%) {bar}")

    # Task breakdown
    print(f"\n📋 TASK BREAKDOWN")
    print(f"{'─' * 80}")

    task_files = {
        'task1_format_recognition.json': 'Format Recognition & Parsing',
        'task2_format_conversion.json': 'Format Conversion',
        'task3_validation.json': 'Validation & Error Detection',
        'task4_coordinate_systems.json': 'Coordinate System Recognition',
        'task5_precision.json': 'Precision Understanding'
    }

    total_tests = 0
    for filename, task_name in task_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            data = load_json(filepath)
            count = len(data)
            total_tests += count
            percentage = (count / summary['total_test_cases']) * 100
            print(f"  {task_name:40s} {count:5d} tests ({percentage:5.1f}%)")

    # Detailed task analysis
    print(f"\n🔍 DETAILED TASK ANALYSIS")
    print(f"{'─' * 80}")

    # Task 1: Format Recognition
    print(f"\n  Task 1: Format Recognition & Parsing")
    task1 = load_json(data_dir / "task1_format_recognition.json")
    format_counts = Counter(t['ground_truth']['format_type'] for t in task1)
    for fmt, count in format_counts.most_common():
        print(f"    • {fmt:20s}: {count:3d} tests")

    # Task 2: Format Conversion
    print(f"\n  Task 2: Format Conversion")
    task2 = load_json(data_dir / "task2_format_conversion.json")
    conversion_pairs = Counter(
        f"{t['source_format']} → {t['target_format']}" for t in task2
    )
    for pair, count in conversion_pairs.most_common():
        print(f"    • {pair:40s}: {count:3d} tests")

    # Task 3: Validation
    print(f"\n  Task 3: Validation & Error Detection")
    task3 = load_json(data_dir / "task3_validation.json")
    error_counts = Counter(t['error_type'] for t in task3)
    for error_type, count in error_counts.most_common():
        print(f"    • {error_type:35s}: {count:3d} tests")

    # Task 4: Coordinate Systems
    print(f"\n  Task 4: Coordinate System Recognition")
    task4 = load_json(data_dir / "task4_coordinate_systems.json")
    system_counts = Counter(t['ground_truth']['system_name'] for t in task4)
    for system, count in system_counts.most_common():
        print(f"    • {system:20s}: {count:3d} tests")

    # Task 5: Precision
    print(f"\n  Task 5: Precision Understanding")
    task5 = load_json(data_dir / "task5_precision.json")
    precision_counts = Counter(t['ground_truth']['decimal_places'] for t in task5)
    for decimals in sorted(precision_counts.keys()):
        count = precision_counts[decimals]
        accuracy = [p for p in task5 if p['ground_truth']['decimal_places'] == decimals][0]['ground_truth']['accuracy_meters']
        use_case = [p for p in task5 if p['ground_truth']['decimal_places'] == decimals][0]['ground_truth']['appropriate_use_case']
        print(f"    • {decimals} decimals (~{accuracy:,}m - {use_case}): {count:3d} tests")

    # Regional coverage per task
    print(f"\n🗺️  REGIONAL COVERAGE BY TASK")
    print(f"{'─' * 80}")

    for filename, task_name in task_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            data = load_json(filepath)
            # Count unique regions (if region field exists)
            regions = set()
            for test in data:
                if 'region' in test:
                    regions.add(test['region'])

            if regions:
                print(f"  {task_name:40s}: {len(regions)} regions covered")

    # Sample test cases
    print(f"\n📝 SAMPLE TEST CASES")
    print(f"{'─' * 80}")

    # Show one example from each task
    examples = [
        ("Format Recognition", task1[0]),
        ("Format Conversion", task2[0]),
        ("Validation", task3[0]),
        ("Coordinate Systems", task4[0]),
        ("Precision", task5[0])
    ]

    for task_name, example in examples:
        print(f"\n  {task_name}:")
        if 'location_name' in example:
            print(f"    Location: {example['location_name']}")
        if 'coordinate_string' in example:
            print(f"    Input:    {example['coordinate_string']}")
        if 'ground_truth' in example:
            gt = example['ground_truth']
            if 'format_type' in gt:
                print(f"    Expected: {gt['format_type']}")
            elif 'target_string' in gt:
                print(f"    Expected: {gt['target_string']}")
            elif 'is_valid' in gt:
                print(f"    Valid:    {gt['is_valid']}")
            elif 'system_name' in gt:
                print(f"    System:   {gt['system_name']}")
            elif 'appropriate_use_case' in gt:
                print(f"    Use Case: {gt['appropriate_use_case']}")

    # Quality metrics
    print(f"\n✅ QUALITY METRICS")
    print(f"{'─' * 80}")

    # Calculate unique locations per task
    all_locations = set()
    for task_file in ['task1_format_recognition.json', 'task2_format_conversion.json',
                      'task3_validation.json', 'task4_coordinate_systems.json',
                      'task5_precision.json']:
        filepath = data_dir / task_file
        if filepath.exists():
            data = load_json(filepath)
            for test in data:
                if 'location_name' in test:
                    all_locations.add(test['location_name'])

    print(f"  Unique locations used:           {len(all_locations)}")
    print(f"  Average tests per location:      {summary['total_test_cases'] / len(all_locations):.1f}")
    print(f"  Regions with representation:     {len(geo_dist)}")
    print(f"  Format types covered:            {len(format_counts)}")
    print(f"  Conversion pairs tested:         {len(conversion_pairs)}")
    print(f"  Error types tested:              {len(error_counts)}")

    print(f"\n{'=' * 80}")
    print(f"✨ Dataset generation complete!")
    print(f"{'=' * 80}")
    print(f"\n📁 Data location: {data_dir.absolute()}")
    print(f"📖 Documentation: {data_dir / 'README.md'}")
    print(f"\n🚀 Ready for LLM evaluation!")


if __name__ == "__main__":
    analyze_dataset()
