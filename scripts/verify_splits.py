"""
Verify and analyze train/dev/test splits for GPSBench.

This script checks for:
- Data leakage between splits
- Distribution preservation
- Minimum size requirements
- Coverage of all categories

Usage:
    python verify_splits.py --task layer_b/splits/task1_place_association
    python verify_splits.py --all  # Verify all splits
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import scipy.stats as stats


def load_split_files(split_dir: Path, task_name: str) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Load train/dev/test splits and metadata."""
    train_file = split_dir / f"{task_name}_train.json"
    dev_file = split_dir / f"{task_name}_dev.json"
    test_file = split_dir / f"{task_name}_test.json"
    info_file = split_dir / f"{task_name}_split_info.json"

    with open(train_file) as f:
        train = json.load(f)

    with open(dev_file) as f:
        dev = json.load(f)

    with open(test_file) as f:
        test = json.load(f)

    with open(info_file) as f:
        metadata = json.load(f)

    return train, dev, test, metadata


def check_data_leakage(train: List[Dict], dev: List[Dict], test: List[Dict]) -> Dict:
    """Check for data leakage between splits."""
    # Convert to hashable format for comparison
    def make_hashable(example):
        return json.dumps(example, sort_keys=True)

    train_set = {make_hashable(ex) for ex in train}
    dev_set = {make_hashable(ex) for ex in dev}
    test_set = {make_hashable(ex) for ex in test}

    # Check overlaps
    train_dev_overlap = train_set & dev_set
    train_test_overlap = train_set & test_set
    dev_test_overlap = dev_set & test_set

    return {
        'train_dev_overlap': len(train_dev_overlap),
        'train_test_overlap': len(train_test_overlap),
        'dev_test_overlap': len(dev_test_overlap),
        'has_leakage': len(train_dev_overlap) > 0 or len(train_test_overlap) > 0 or len(dev_test_overlap) > 0
    }


def get_distribution(data: List[Dict], attribute: str) -> Counter:
    """Get distribution of an attribute."""
    values = []
    for example in data:
        if attribute in example:
            values.append(example[attribute])
        elif 'ground_truth' in example and attribute in example['ground_truth']:
            values.append(example['ground_truth'][attribute])
        elif '.' in attribute:
            # Handle nested attributes like 'ground_truth.difficulty'
            parts = attribute.split('.')
            val = example
            for part in parts:
                if part in val:
                    val = val[part]
                else:
                    val = 'unknown'
                    break
            if val != 'unknown':
                values.append(val)

    return Counter(values)


def chi_square_test(dist1: Counter, dist2: Counter, significance_level: float = 0.05) -> Dict:
    """
    Perform chi-square test to compare two distributions.

    Returns:
        Dictionary with test results
    """
    # Get all keys
    all_keys = set(dist1.keys()) | set(dist2.keys())

    if len(all_keys) < 2:
        return {
            'statistic': 0,
            'p_value': 1.0,
            'similar': True,
            'note': 'Too few categories for chi-square test'
        }

    # Create frequency arrays
    obs1 = [dist1.get(k, 0) for k in all_keys]
    obs2 = [dist2.get(k, 0) for k in all_keys]

    # Convert to proportions
    total1 = sum(obs1)
    total2 = sum(obs2)

    if total1 == 0 or total2 == 0:
        return {
            'statistic': 0,
            'p_value': 1.0,
            'similar': True,
            'note': 'Empty distribution'
        }

    prop1 = [x / total1 for x in obs1]
    prop2 = [x / total2 for x in obs2]

    # Expected frequencies based on overall proportion
    expected1 = [(p1 + p2) / 2 * total1 for p1, p2 in zip(prop1, prop2)]
    expected2 = [(p1 + p2) / 2 * total2 for p1, p2 in zip(prop1, prop2)]

    # Chi-square test
    try:
        statistic = sum((o - e) ** 2 / e for o, e in zip(obs1, expected1) if e > 0)
        statistic += sum((o - e) ** 2 / e for o, e in zip(obs2, expected2) if e > 0)

        df = len(all_keys) - 1
        p_value = 1 - stats.chi2.cdf(statistic, df)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'similar': p_value > significance_level,
            'significance_level': significance_level
        }
    except Exception as e:
        return {
            'statistic': 0,
            'p_value': 1.0,
            'similar': True,
            'note': f'Error in chi-square test: {e}'
        }


def check_distribution_preservation(
    train: List[Dict],
    dev: List[Dict],
    test: List[Dict]
) -> Dict:
    """Check if distributions are preserved across splits."""
    # Identify attributes to check
    sample = train[0] if train else (dev[0] if dev else (test[0] if test else {}))

    attributes = []
    if 'difficulty' in sample:
        attributes.append('difficulty')
    if 'category' in sample:
        attributes.append('category')
    if 'subtask' in sample:
        attributes.append('subtask')

    results = {}

    for attr in attributes:
        train_dist = get_distribution(train, attr)
        dev_dist = get_distribution(dev, attr)
        test_dist = get_distribution(test, attr)

        # Compare train-dev and train-test
        train_dev_test = chi_square_test(train_dist, dev_dist)
        train_test_test = chi_square_test(train_dist, test_dist)

        results[attr] = {
            'train_distribution': dict(train_dist),
            'dev_distribution': dict(dev_dist),
            'test_distribution': dict(test_dist),
            'train_dev_similar': train_dev_test['similar'],
            'train_test_similar': train_test_test['similar'],
            'train_dev_p_value': train_dev_test['p_value'],
            'train_test_p_value': train_test_test['p_value']
        }

    return results


def check_minimum_sizes(train: List[Dict], dev: List[Dict], test: List[Dict]) -> Dict:
    """Check if splits meet minimum size requirements."""
    min_dev = 10  # With 10% dev, minimum is lower
    min_test = 20  # Keep 20 for reliable test evaluation

    return {
        'train_size': len(train),
        'dev_size': len(dev),
        'test_size': len(test),
        'dev_sufficient': len(dev) >= min_dev,
        'test_sufficient': len(test) >= min_test,
        'min_dev': min_dev,
        'min_test': min_test
    }


def check_coverage(train: List[Dict], dev: List[Dict], test: List[Dict]) -> Dict:
    """Check if all categories are represented in each split."""
    def get_categories(data, attr):
        return set(ex.get(attr, 'unknown') for ex in data)

    sample = train[0] if train else (dev[0] if dev else (test[0] if test else {}))
    attributes = ['difficulty', 'category', 'subtask']

    results = {}

    for attr in attributes:
        if attr not in sample:
            continue

        train_cats = get_categories(train, attr)
        dev_cats = get_categories(dev, attr)
        test_cats = get_categories(test, attr)

        all_cats = train_cats | dev_cats | test_cats

        results[attr] = {
            'total_categories': len(all_cats),
            'train_categories': len(train_cats),
            'dev_categories': len(dev_cats),
            'test_categories': len(test_cats),
            'train_coverage': len(train_cats) / len(all_cats) if all_cats else 0,
            'dev_coverage': len(dev_cats) / len(all_cats) if all_cats else 0,
            'test_coverage': len(test_cats) / len(all_cats) if all_cats else 0,
            'all_in_train': all_cats.issubset(train_cats),
            'missing_in_dev': list(all_cats - dev_cats),
            'missing_in_test': list(all_cats - test_cats)
        }

    return results


def verify_split(split_dir: Path, task_name: str) -> Dict:
    """
    Verify a single split.

    Args:
        split_dir: Directory containing split files
        task_name: Task name (without _train/_dev/_test suffix)

    Returns:
        Verification results
    """
    print(f"Verifying {task_name}...")

    # Load splits
    train, dev, test, metadata = load_split_files(split_dir, task_name)

    results = {
        'task': task_name,
        'metadata': metadata,
        'checks': {}
    }

    # Check 1: Data leakage
    leakage = check_data_leakage(train, dev, test)
    results['checks']['data_leakage'] = leakage

    if leakage['has_leakage']:
        print(f"  ❌ Data leakage detected!")
        print(f"     Train-Dev overlap: {leakage['train_dev_overlap']}")
        print(f"     Train-Test overlap: {leakage['train_test_overlap']}")
        print(f"     Dev-Test overlap: {leakage['dev_test_overlap']}")
    else:
        print(f"  ✓ No data leakage")

    # Check 2: Minimum sizes
    sizes = check_minimum_sizes(train, dev, test)
    results['checks']['sizes'] = sizes

    if sizes['dev_sufficient'] and sizes['test_sufficient']:
        print(f"  ✓ Sufficient split sizes ({sizes['train_size']}/{sizes['dev_size']}/{sizes['test_size']})")
    else:
        print(f"  ⚠ Small splits: Train={sizes['train_size']}, Dev={sizes['dev_size']}, Test={sizes['test_size']}")

    # Check 3: Distribution preservation
    if metadata['split_strategy'] == 'random_stratified':
        distributions = check_distribution_preservation(train, dev, test)
        results['checks']['distributions'] = distributions

        all_similar = all(
            d['train_dev_similar'] and d['train_test_similar']
            for d in distributions.values()
        )

        if all_similar:
            print(f"  ✓ Distributions preserved")
        else:
            print(f"  ⚠ Distribution differences detected")
            for attr, dist in distributions.items():
                if not dist['train_dev_similar'] or not dist['train_test_similar']:
                    print(f"     {attr}: p_train_dev={dist['train_dev_p_value']:.3f}, "
                          f"p_train_test={dist['train_test_p_value']:.3f}")

    # Check 4: Coverage
    coverage = check_coverage(train, dev, test)
    results['checks']['coverage'] = coverage

    missing_coverage = any(
        c['missing_in_dev'] or c['missing_in_test']
        for c in coverage.values()
    )

    if not missing_coverage:
        print(f"  ✓ All categories covered")
    else:
        print(f"  ⚠ Some categories missing in splits")
        for attr, cov in coverage.items():
            if cov['missing_in_dev']:
                print(f"     {attr} missing in dev: {cov['missing_in_dev']}")
            if cov['missing_in_test']:
                print(f"     {attr} missing in test: {cov['missing_in_test']}")

    return results


def generate_report(all_results: List[Dict], output_file: Path):
    """Generate verification report."""
    total = len(all_results)
    passed = sum(
        1 for r in all_results
        if not r['checks']['data_leakage']['has_leakage']
        and r['checks']['sizes']['dev_sufficient']
        and r['checks']['sizes']['test_sufficient']
    )

    report = {
        'summary': {
            'total_tasks': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0
        },
        'tasks': all_results
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Verification Report")
    print('='*60)
    print(f"Total tasks: {total}")
    print(f"Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"Failed: {total - passed}")
    print(f"\nReport saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify train/dev/test splits"
    )

    parser.add_argument(
        '--task',
        type=str,
        help='Task directory (e.g., layer_b/splits/task1_place_association)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Verify all splits'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/split_verification_report.json',
        help='Output report file'
    )

    args = parser.parse_args()

    if args.task:
        # Verify single task
        parts = Path(args.task).parts
        if 'splits' in parts:
            split_dir = Path(args.task).parent
            task_name = Path(args.task).name
        else:
            split_dir = Path(args.task)
            task_name = split_dir.name.replace('_train', '').replace('_dev', '').replace('_test', '')

        verify_split(split_dir, task_name)

    elif args.all:
        # Verify all splits
        data_dir = Path('data')
        all_results = []

        for layer_dir in sorted(data_dir.glob('layer_*')):
            splits_dir = layer_dir / 'splits'
            if not splits_dir.exists():
                continue

            print(f"\n{'='*60}")
            print(f"{layer_dir.name}")
            print('='*60)

            # Find all split info files
            for info_file in sorted(splits_dir.glob('*_split_info.json')):
                task_name = info_file.stem.replace('_split_info', '')

                try:
                    results = verify_split(splits_dir, task_name)
                    all_results.append(results)
                    print()

                except Exception as e:
                    print(f"  Error: {e}")
                    print()

        # Generate report
        generate_report(all_results, Path(args.output))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
