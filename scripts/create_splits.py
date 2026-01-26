"""
Create train/dev/test splits for GPSBench tasks.

This script implements three splitting strategies:
1. Random Stratified: Random split maintaining distributions
2. Geographic: Split by geographic regions
3. Difficulty-Based: Split by difficulty levels

Usage:
    python create_splits.py --strategy random_stratified --seed 42
    python create_splits.py --strategy geographic --task layer_b/task1_place_association.json
    python create_splits.py --all  # Split all tasks with default strategy
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import numpy as np


# Geographic region mappings
COUNTRY_TO_REGION = {
    # North America
    'US': 'North America', 'CA': 'North America', 'MX': 'North America',

    # South America
    'BR': 'South America', 'AR': 'South America', 'CL': 'South America',
    'CO': 'South America', 'PE': 'South America',

    # Europe
    'GB': 'Europe', 'FR': 'Europe', 'DE': 'Europe', 'IT': 'Europe',
    'ES': 'Europe', 'PL': 'Europe', 'NL': 'Europe', 'SE': 'Europe',
    'NO': 'Europe', 'FI': 'Europe', 'DK': 'Europe', 'CH': 'Europe',
    'AT': 'Europe', 'BE': 'Europe', 'GR': 'Europe', 'PT': 'Europe',
    'CZ': 'Europe', 'RO': 'Europe', 'HU': 'Europe',

    # Asia
    'CN': 'Asia', 'JP': 'Asia', 'IN': 'Asia', 'KR': 'Asia',
    'TH': 'Asia', 'VN': 'Asia', 'ID': 'Asia', 'MY': 'Asia',
    'PH': 'Asia', 'SG': 'Asia', 'TW': 'Asia', 'KZ': 'Asia',
    'UZ': 'Asia', 'PK': 'Asia', 'BD': 'Asia', 'MM': 'Asia',

    # Africa
    'ZA': 'Africa', 'EG': 'Africa', 'NG': 'Africa', 'KE': 'Africa',
    'ET': 'Africa', 'TZ': 'Africa', 'UG': 'Africa', 'GH': 'Africa',
    'MA': 'Africa', 'DZ': 'Africa',

    # Oceania
    'AU': 'Oceania', 'NZ': 'Oceania', 'FJ': 'Oceania', 'PG': 'Oceania',

    # Middle East
    'SA': 'Middle East', 'AE': 'Middle East', 'IL': 'Middle East',
    'TR': 'Middle East', 'IR': 'Middle East', 'IQ': 'Middle East',
    'JO': 'Middle East', 'LB': 'Middle East', 'KW': 'Middle East',

    # Arctic/Special
    'RU': 'Asia',  # Russia - considered Asia for simplicity
    'IS': 'Europe',  # Iceland
    'GL': 'Arctic',  # Greenland
    'SJ': 'Arctic',  # Svalbard
}

GEOGRAPHIC_SPLITS = {
    'train': ['North America', 'South America', 'Asia'],
    'dev': ['Europe', 'Africa'],
    'test': ['Oceania', 'Middle East', 'Arctic']
}


def get_region_from_example(example: Dict) -> str:
    """Extract geographic region from example."""
    # Try different keys where country/region might be
    if 'ground_truth' in example:
        gt = example['ground_truth']

        # Check region first (more reliable)
        if 'region' in gt and gt['region']:
            return gt['region']

        # Then check country
        if 'country' in gt:
            country = gt['country']
            # Skip if country is None or empty
            if country is None or country == '':
                pass
            # Handle country codes
            elif len(country) == 2:
                return COUNTRY_TO_REGION.get(country, 'Unknown')
            # Handle country names - try to match
            else:
                for code, region in COUNTRY_TO_REGION.items():
                    if country.lower() in code.lower():
                        return region

    # Check point_a and point_b for distance tasks
    for key in ['point_a', 'point_b']:
        if key in example and 'name' in example[key]:
            name = example[key]['name']
            # Extract country code from name if present
            for code in COUNTRY_TO_REGION:
                if code in name:
                    return COUNTRY_TO_REGION[code]

    return 'Unknown'


def get_stratification_keys(example: Dict, task_type: str) -> Dict[str, str]:
    """Extract stratification keys from example."""
    keys = {}

    # Difficulty
    if 'difficulty' in example:
        keys['difficulty'] = example['difficulty']

    # Category
    if 'category' in example:
        keys['category'] = example['category']

    # Subtask
    if 'subtask' in example:
        keys['subtask'] = example['subtask']

    # Task-specific categories
    if task_type == 'distance_calculation' and 'category' in example:
        keys['distance_category'] = example['category']

    if task_type == 'noise_tolerance' and 'metadata' in example:
        if 'noise_std' in example['metadata']:
            keys['noise_level'] = example['metadata']['noise_std']

    # Region
    keys['region'] = get_region_from_example(example)

    return keys


def stratified_split(
    data: List[Dict],
    ratios: Tuple[float, float, float] = (0.6, 0.1, 0.3),
    stratify_keys: List[str] = None,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Perform stratified split maintaining distributions.

    Args:
        data: List of examples
        ratios: (train, dev, test) ratios
        stratify_keys: Keys to stratify by
        seed: Random seed

    Returns:
        (train, dev, test) splits
    """
    random.seed(seed)
    np.random.seed(seed)

    if not stratify_keys:
        # Simple random split
        shuffled = data.copy()
        random.shuffle(shuffled)

        n = len(data)
        train_end = int(n * ratios[0])
        dev_end = train_end + int(n * ratios[1])

        return shuffled[:train_end], shuffled[train_end:dev_end], shuffled[dev_end:]

    # Group by stratification keys
    groups = defaultdict(list)
    for example in data:
        # Create composite key from all stratification attributes
        key_values = tuple(example.get(k, 'unknown') for k in stratify_keys)
        groups[key_values].append(example)

    # Split each group proportionally
    train, dev, test = [], [], []

    for group_examples in groups.values():
        random.shuffle(group_examples)
        n = len(group_examples)

        train_end = max(1, int(n * ratios[0]))  # At least 1 in train
        dev_end = train_end + max(1, int(n * ratios[1]))  # At least 1 in dev

        train.extend(group_examples[:train_end])
        dev.extend(group_examples[train_end:dev_end])
        test.extend(group_examples[dev_end:])

    # Shuffle final splits
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    return train, dev, test


def geographic_split(
    data: List[Dict],
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split by geographic regions.

    Args:
        data: List of examples
        seed: Random seed

    Returns:
        (train, dev, test) splits
    """
    random.seed(seed)

    # Group by region
    region_groups = defaultdict(list)
    unknown = []

    for example in data:
        region = get_region_from_example(example)
        if region == 'Unknown':
            unknown.append(example)
        else:
            region_groups[region].append(example)

    # Assign to splits based on region
    train, dev, test = [], [], []

    for region, examples in region_groups.items():
        if region in GEOGRAPHIC_SPLITS['train']:
            train.extend(examples)
        elif region in GEOGRAPHIC_SPLITS['dev']:
            dev.extend(examples)
        elif region in GEOGRAPHIC_SPLITS['test']:
            test.extend(examples)
        else:
            # Unknown regions - distribute randomly
            unknown.extend(examples)

    # Distribute unknown examples proportionally
    if unknown:
        random.shuffle(unknown)
        n = len(unknown)
        train_end = int(n * 0.6)
        dev_end = train_end + int(n * 0.1)

        train.extend(unknown[:train_end])
        dev.extend(unknown[train_end:dev_end])
        test.extend(unknown[dev_end:])

    # Shuffle
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    return train, dev, test


def difficulty_split(
    data: List[Dict],
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split by difficulty levels (train=easy+medium, dev=medium+hard, test=hard+edge).

    Args:
        data: List of examples
        seed: Random seed

    Returns:
        (train, dev, test) splits
    """
    random.seed(seed)

    # Group by difficulty
    by_difficulty = defaultdict(list)

    for example in data:
        difficulty = example.get('difficulty', 'medium')
        by_difficulty[difficulty].append(example)

    # Assign to splits
    train = []
    dev = []
    test = []

    # Easy -> train
    if 'easy' in by_difficulty:
        train.extend(by_difficulty['easy'])

    # Medium -> split between train and dev
    if 'medium' in by_difficulty:
        medium = by_difficulty['medium']
        random.shuffle(medium)
        split_point = len(medium) // 2
        train.extend(medium[:split_point])
        dev.extend(medium[split_point:])

    # Hard -> split between dev and test
    if 'hard' in by_difficulty:
        hard = by_difficulty['hard']
        random.shuffle(hard)
        split_point = len(hard) // 2
        dev.extend(hard[:split_point])
        test.extend(hard[split_point:])

    # Edge cases -> test
    if 'edge_case' in by_difficulty:
        test.extend(by_difficulty['edge_case'])

    # Shuffle
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    return train, dev, test


def create_split(
    task_file: Path,
    strategy: str = 'random_stratified',
    ratios: Tuple[float, float, float] = (0.6, 0.1, 0.3),
    seed: int = 42,
    output_dir: Path = None
) -> Dict:
    """
    Create train/dev/test split for a task.

    Args:
        task_file: Path to task JSON file
        strategy: Splitting strategy ('random_stratified', 'geographic', 'difficulty')
        ratios: (train, dev, test) ratios (for random_stratified)
        seed: Random seed
        output_dir: Output directory (default: task_file.parent / 'splits')

    Returns:
        Split metadata
    """
    # Load data
    with open(task_file, 'r') as f:
        data = json.load(f)

    print(f"Splitting {task_file.name} ({len(data)} examples) with {strategy} strategy...")

    # Get task type
    task_name = task_file.stem
    task_type = data[0].get('task', 'unknown') if data else 'unknown'

    # Apply splitting strategy
    if strategy == 'random_stratified':
        # Determine stratification keys
        if data:
            sample = data[0]
            stratify_keys = []
            if 'difficulty' in sample:
                stratify_keys.append('difficulty')
            if 'category' in sample:
                stratify_keys.append('category')
            if 'subtask' in sample:
                stratify_keys.append('subtask')
        else:
            stratify_keys = []

        train, dev, test = stratified_split(data, ratios, stratify_keys, seed)

    elif strategy == 'geographic':
        train, dev, test = geographic_split(data, seed)

    elif strategy == 'difficulty':
        train, dev, test = difficulty_split(data, seed)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create output directory
    if output_dir is None:
        output_dir = task_file.parent / 'splits'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save splits
    train_file = output_dir / f"{task_name}_train.json"
    dev_file = output_dir / f"{task_name}_dev.json"
    test_file = output_dir / f"{task_name}_test.json"

    with open(train_file, 'w') as f:
        json.dump(train, f, indent=2)

    with open(dev_file, 'w') as f:
        json.dump(dev, f, indent=2)

    with open(test_file, 'w') as f:
        json.dump(test, f, indent=2)

    # Create metadata
    metadata = {
        'task': task_name,
        'task_type': task_type,
        'split_strategy': strategy,
        'split_ratios': {'train': ratios[0], 'dev': ratios[1], 'test': ratios[2]} if strategy == 'random_stratified' else None,
        'seed': seed,
        'total_examples': len(data),
        'splits': {
            'train': len(train),
            'dev': len(dev),
            'test': len(test)
        },
        'files': {
            'train': str(train_file.name),
            'dev': str(dev_file.name),
            'test': str(test_file.name)
        }
    }

    # Save metadata
    metadata_file = output_dir / f"{task_name}_split_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Train: {len(train)} examples")
    print(f"  Dev: {len(dev)} examples")
    print(f"  Test: {len(test)} examples")
    print(f"  Saved to {output_dir}")

    return metadata


def get_recommended_strategy(task_file: Path) -> str:
    """Get recommended splitting strategy for a task."""
    task_name = task_file.stem

    # Geographic split for location-based tasks
    geographic_tasks = [
        'place_association',
        'regional_knowledge',
        'context_grounding',
        'name_disambiguation',
        'proximity',
        'boundary_analysis'
    ]

    if any(t in task_name for t in geographic_tasks):
        return 'geographic'

    # Difficulty split for robustness tasks
    difficulty_tasks = [
        'hallucination_detection',
        'noise_tolerance'
    ]

    if any(t in task_name for t in difficulty_tasks):
        return 'difficulty'

    # Default: random stratified
    return 'random_stratified'


def main():
    parser = argparse.ArgumentParser(
        description="Create train/dev/test splits for GPSBench tasks"
    )

    parser.add_argument(
        '--task-file',
        type=str,
        help='Specific task file to split'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Split all tasks in data directory'
    )
    parser.add_argument(
        '--track',
        type=str,
        choices=['pure_gps', 'applied', 'both'],
        help='Split tasks from specific track (pure_gps, applied, or both)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['random_stratified', 'geographic', 'difficulty', 'auto'],
        default='auto',
        help='Splitting strategy (auto=recommended per task)'
    )
    parser.add_argument(
        '--ratios',
        type=float,
        nargs=3,
        default=[0.6, 0.1, 0.3],
        help='Train/dev/test ratios (for random_stratified)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: <layer>/splits/)'
    )

    args = parser.parse_args()

    ratios = tuple(args.ratios)

    if args.task_file:
        # Split single task
        task_file = Path(args.task_file)
        strategy = get_recommended_strategy(task_file) if args.strategy == 'auto' else args.strategy

        create_split(
            task_file,
            strategy=strategy,
            ratios=ratios,
            seed=args.seed,
            output_dir=Path(args.output_dir) if args.output_dir else None
        )

    elif args.all or args.track:
        # Split all tasks or tasks from specific track
        data_dir = Path('data')
        all_metadata = []

        # Determine which directories to process
        if args.track:
            if args.track == 'both':
                track_patterns = ['track_pure_gps', 'track_applied']
            else:
                track_patterns = [f'track_{args.track}']
        else:
            # Look for both track_* directories (new structure) and layer_* (legacy)
            track_patterns = ['track_*', 'layer_*', 'legacy_layers/layer_*']

        found_dirs = []
        for pattern in track_patterns:
            found_dirs.extend(sorted(data_dir.glob(pattern)))

        # Remove duplicates and sort
        found_dirs = sorted(set(found_dirs))

        if not found_dirs:
            print("No track or layer directories found!")
            print("Looking for: track_pure_gps, track_applied, or layer_* directories")
            return

        for track_dir in found_dirs:
            print(f"\n{'='*60}")
            print(f"Processing {track_dir.name}")
            print('='*60)

            for task_file in sorted(track_dir.glob('task*.json')):
                # Skip backup files and MC duplicates for now
                if 'backup' in task_file.name or '_mc' in task_file.name:
                    continue

                try:
                    strategy = get_recommended_strategy(task_file) if args.strategy == 'auto' else args.strategy

                    metadata = create_split(
                        task_file,
                        strategy=strategy,
                        ratios=ratios,
                        seed=args.seed
                    )

                    all_metadata.append(metadata)
                    print()

                except Exception as e:
                    print(f"  Error: {e}")
                    print()

        # Save summary
        summary_file = data_dir / 'splits_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'total_tasks': len(all_metadata),
                'seed': args.seed,
                'ratios': {'train': ratios[0], 'dev': ratios[1], 'test': ratios[2]},
                'tasks': all_metadata
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Summary saved to {summary_file}")
        print(f"Total tasks split: {len(all_metadata)}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
