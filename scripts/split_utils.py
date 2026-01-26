"""
Utility functions for loading and using GPSBench splits.

Usage:
    from scripts.split_utils import load_split, load_all_splits

    # Load specific split
    train_data = load_split('layer_b/task1_place_association', split='train')

    # Load all splits for a task
    train, dev, test = load_all_splits('layer_b/task1_place_association')

    # Load all tasks for fine-tuning
    all_train_data = load_all_tasks(split='train')
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def load_split(
    task_path: str,
    split: str = 'test',
    data_dir: str = 'data'
) -> List[Dict]:
    """
    Load a specific split for a task.

    Args:
        task_path: Task path (e.g., 'layer_b/task1_place_association')
        split: Split name ('train', 'dev', or 'test')
        data_dir: Base data directory

    Returns:
        List of examples for the split

    Examples:
        >>> train_data = load_split('layer_b/task1_place_association', split='train')
        >>> test_data = load_split('layer_c/task1_distance_calculation', split='test')
    """
    if split not in ['train', 'dev', 'test']:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'dev', or 'test'")

    # Parse task path
    parts = Path(task_path).parts
    if len(parts) == 1:
        # Just task name, need to find it
        task_name = parts[0]
        # Search for it
        base_path = Path(data_dir)
        for layer_dir in base_path.glob('layer_*'):
            splits_dir = layer_dir / 'splits'
            split_file = splits_dir / f"{task_name}_{split}.json"
            if split_file.exists():
                with open(split_file) as f:
                    return json.load(f)
        raise FileNotFoundError(f"Could not find {task_name} in any layer")

    else:
        # Full path provided
        layer = parts[0]
        task_name = parts[1] if len(parts) > 1 else parts[0]

        split_file = Path(data_dir) / layer / 'splits' / f"{task_name}_{split}.json"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file) as f:
            return json.load(f)


def load_all_splits(
    task_path: str,
    data_dir: str = 'data'
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load all splits (train, dev, test) for a task.

    Args:
        task_path: Task path (e.g., 'layer_b/task1_place_association')
        data_dir: Base data directory

    Returns:
        Tuple of (train, dev, test) data

    Example:
        >>> train, dev, test = load_all_splits('layer_b/task1_place_association')
        >>> print(f"Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")
    """
    train = load_split(task_path, split='train', data_dir=data_dir)
    dev = load_split(task_path, split='dev', data_dir=data_dir)
    test = load_split(task_path, split='test', data_dir=data_dir)

    return train, dev, test


def load_split_metadata(
    task_path: str,
    data_dir: str = 'data'
) -> Dict:
    """
    Load split metadata for a task.

    Args:
        task_path: Task path
        data_dir: Base data directory

    Returns:
        Split metadata dictionary

    Example:
        >>> meta = load_split_metadata('layer_b/task1_place_association')
        >>> print(f"Strategy: {meta['split_strategy']}")
    """
    parts = Path(task_path).parts
    layer = parts[0]
    task_name = parts[1] if len(parts) > 1 else parts[0]

    info_file = Path(data_dir) / layer / 'splits' / f"{task_name}_split_info.json"

    if not info_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {info_file}")

    with open(info_file) as f:
        return json.load(f)


def load_all_tasks(
    split: str = 'train',
    layers: Optional[List[str]] = None,
    tracks: Optional[List[str]] = None,
    data_dir: str = 'data'
) -> Dict[str, List[Dict]]:
    """
    Load a specific split for all tasks.

    Args:
        split: Split name ('train', 'dev', or 'test')
        layers: List of layers to include (None = use tracks or all)
        tracks: List of tracks to include (e.g., ['pure_gps', 'applied'])
        data_dir: Base data directory

    Returns:
        Dictionary mapping task names to data

    Example:
        >>> # Load from tracks (new structure)
        >>> all_train = load_all_tasks(split='train', tracks=['pure_gps', 'applied'])
        >>>
        >>> # Load from layers (legacy)
        >>> all_train = load_all_tasks(split='train', layers=['layer_c'])
        >>>
        >>> # Load all (tracks + legacy)
        >>> all_train = load_all_tasks(split='train')
    """
    base_path = Path(data_dir)
    all_data = {}

    # Determine which directories to scan
    if tracks is not None:
        # Use track-based structure
        track_dirs = [base_path / f'track_{track}' for track in tracks]
        source_dirs = track_dirs
    elif layers is not None:
        # Use layer-based structure (legacy)
        layer_dirs = [base_path / layer for layer in layers]
        source_dirs = layer_dirs
    else:
        # Auto-detect: prefer tracks if they exist, otherwise use layers
        track_dirs = sorted(base_path.glob('track_*'))
        layer_dirs = sorted(base_path.glob('layer_*'))
        legacy_dirs = sorted(base_path.glob('legacy_layers/layer_*'))

        if track_dirs:
            # Use track structure (new)
            source_dirs = track_dirs
        else:
            # Fall back to layers (old structure)
            source_dirs = layer_dirs + legacy_dirs

    # Load data from each directory
    for source_dir in source_dirs:
        splits_dir = source_dir / 'splits'
        if not splits_dir.exists():
            continue

        for split_file in sorted(splits_dir.glob(f'*_{split}.json')):
            task_name = split_file.stem.replace(f'_{split}', '')
            full_task_name = f"{source_dir.name}/{task_name}"

            with open(split_file) as f:
                data = json.load(f)
                all_data[full_task_name] = data

    return all_data


def load_multitask_dataset(
    split: str = 'train',
    layers: Optional[List[str]] = None,
    task_filter: Optional[List[str]] = None,
    data_dir: str = 'data'
) -> List[Dict]:
    """
    Load combined multi-task dataset from splits.

    Args:
        split: Split name ('train', 'dev', or 'test')
        layers: List of layers to include
        task_filter: List of task names to include (None = all)
        data_dir: Base data directory

    Returns:
        Combined list of examples from all tasks

    Example:
        >>> # Load all training data
        >>> train_data = load_multitask_dataset(split='train')
        >>> print(f"Total training examples: {len(train_data)}")

        >>> # Load only Layer C tasks (geometric operations)
        >>> geom_train = load_multitask_dataset(split='train', layers=['layer_c'])

        >>> # Load specific tasks
        >>> tasks = ['task1_distance_calculation', 'task2_bearing_computation']
        >>> specific_train = load_multitask_dataset(split='train', task_filter=tasks)
    """
    all_data = load_all_tasks(split=split, layers=layers, data_dir=data_dir)

    combined = []

    for task_name, data in all_data.items():
        # Apply task filter if provided
        if task_filter is not None:
            task_basename = Path(task_name).name
            if task_basename not in task_filter:
                continue

        # Add task name to each example for reference
        for example in data:
            if 'source_task' not in example:
                example['source_task'] = task_name

        combined.extend(data)

    return combined


def get_split_statistics(data_dir: str = 'data') -> Dict:
    """
    Get statistics for all splits.

    Args:
        data_dir: Base data directory

    Returns:
        Dictionary with split statistics

    Example:
        >>> stats = get_split_statistics()
        >>> print(json.dumps(stats, indent=2))
    """
    stats = {
        'total_tasks': 0,
        'total_examples': 0,
        'by_split': {
            'train': {'tasks': 0, 'examples': 0},
            'dev': {'tasks': 0, 'examples': 0},
            'test': {'tasks': 0, 'examples': 0}
        },
        'by_layer': {},
        'tasks': []
    }

    base_path = Path(data_dir)

    for layer_dir in sorted(base_path.glob('layer_*')):
        splits_dir = layer_dir / 'splits'
        if not splits_dir.exists():
            continue

        layer_stats = {
            'train': 0,
            'dev': 0,
            'test': 0
        }

        # Find all tasks
        info_files = list(splits_dir.glob('*_split_info.json'))

        for info_file in info_files:
            with open(info_file) as f:
                metadata = json.load(f)

            stats['total_tasks'] += 1

            task_stats = {
                'task': metadata['task'],
                'layer': layer_dir.name,
                'strategy': metadata['split_strategy'],
                'total': metadata['total_examples'],
                'train': metadata['splits']['train'],
                'dev': metadata['splits']['dev'],
                'test': metadata['splits']['test']
            }

            stats['tasks'].append(task_stats)

            # Update totals
            stats['total_examples'] += metadata['total_examples']

            for split_name in ['train', 'dev', 'test']:
                count = metadata['splits'][split_name]
                stats['by_split'][split_name]['examples'] += count
                layer_stats[split_name] += count

            stats['by_split']['train']['tasks'] = stats['total_tasks']
            stats['by_split']['dev']['tasks'] = stats['total_tasks']
            stats['by_split']['test']['tasks'] = stats['total_tasks']

        stats['by_layer'][layer_dir.name] = layer_stats

    return stats


# Example usage
if __name__ == '__main__':
    import sys

    # Print usage examples
    print("GPSBench Split Utilities")
    print("=" * 60)
    print()

    # Get statistics
    try:
        stats = get_split_statistics()

        print(f"Total Tasks: {stats['total_tasks']}")
        print(f"Total Examples: {stats['total_examples']}")
        print()

        print("Split Distribution:")
        for split_name, split_stats in stats['by_split'].items():
            print(f"  {split_name.capitalize()}: {split_stats['examples']} examples "
                  f"({split_stats['examples']/stats['total_examples']*100:.1f}%)")
        print()

        print("By Layer:")
        for layer, layer_stats in stats['by_layer'].items():
            total = sum(layer_stats.values())
            print(f"  {layer}: {total} total "
                  f"(Train: {layer_stats['train']}, "
                  f"Dev: {layer_stats['dev']}, "
                  f"Test: {layer_stats['test']})")

    except Exception as e:
        print(f"Error: {e}")
        print("\nSplits may not be created yet. Run:")
        print("  python scripts/create_splits.py --all")
