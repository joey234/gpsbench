#!/usr/bin/env python3
"""
Analyze MapEval results filtered by:
1. Only samples with successful geocoding (num_geocoded > 0)
2. Exclude unanswerable classification
"""

import json
from pathlib import Path

def main():
    results_dir = Path(__file__).parent / "results"

    # Find latest v3 results (fall back to v2)
    base_files = sorted(results_dir.glob("mapeval_v2_base_gpt-4.1_*.json"))
    aug_files = sorted(results_dir.glob("mapeval_v2_augmented_gpt-4.1_*.json"))

    if not base_files or not aug_files:
        print("No results found")
        return

    base_file = base_files[-1]
    aug_file = aug_files[-1]

    print(f"Loading base results: {base_file.name}")
    print(f"Loading augmented results: {aug_file.name}")

    with open(base_file) as f:
        base_data = json.load(f)
    with open(aug_file) as f:
        aug_data = json.load(f)

    # Load original dataset for geocoding info (prefer v4)
    data_file = Path(__file__).parent / "mapeval_gps_augmented_v4.json"
    if not data_file.exists():
        data_file = Path(__file__).parent / "mapeval_gps_augmented_v3.json"
    if not data_file.exists():
        data_file = Path(__file__).parent / "mapeval_gps_augmented_v2.json"
    with open(data_file) as f:
        samples = json.load(f)

    # Create lookup by id
    sample_info = {s['id']: s for s in samples}

    base_results = base_data['results']
    aug_results = aug_data['results']

    # Filter criteria
    def should_include(result):
        sample = sample_info.get(result['id'])
        if not sample:
            return False
        # Exclude unanswerable
        if result.get('classification') == 'unanswerable':
            return False
        # Must have at least one geocoded place
        if sample.get('num_geocoded', 0) == 0:
            return False
        return True

    # Apply filter
    base_filtered = [r for r in base_results if should_include(r)]
    aug_filtered = [r for r in aug_results if should_include(r)]

    print(f"\n{'='*60}")
    print("FILTERED ANALYSIS")
    print(f"{'='*60}")
    print(f"Original samples: {len(base_results)}")
    print(f"After filtering (geocoded + non-unanswerable): {len(base_filtered)}")

    # Calculate metrics
    def calc_metrics(results):
        total = len(results)
        correct = sum(1 for r in results if r.get('is_correct', False))

        by_class = {}
        for r in results:
            cls = r.get('classification', 'unknown')
            if cls not in by_class:
                by_class[cls] = {'total': 0, 'correct': 0}
            by_class[cls]['total'] += 1
            if r.get('is_correct', False):
                by_class[cls]['correct'] += 1

        return {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0,
            'by_class': by_class
        }

    base_metrics = calc_metrics(base_filtered)
    aug_metrics = calc_metrics(aug_filtered)

    print(f"\n{'='*60}")
    print("RESULTS (Filtered: geocoded samples only, excluding unanswerable)")
    print(f"{'='*60}")

    base_acc = base_metrics['accuracy'] * 100
    aug_acc = aug_metrics['accuracy'] * 100
    diff = aug_acc - base_acc

    print(f"\nOverall Accuracy:")
    print(f"  Base:      {base_acc:.1f}% ({base_metrics['correct']}/{base_metrics['total']})")
    print(f"  Augmented: {aug_acc:.1f}% ({aug_metrics['correct']}/{aug_metrics['total']})")
    print(f"  Difference: {diff:+.1f}%")

    print(f"\nBy Classification:")
    print(f"  {'Class':<15} {'Base':>10} {'Augmented':>10} {'Diff':>10}")
    print(f"  {'-'*45}")

    all_classes = set(base_metrics['by_class'].keys()) | set(aug_metrics['by_class'].keys())

    for cls in sorted(all_classes):
        base_cls = base_metrics['by_class'].get(cls, {'total': 0, 'correct': 0})
        aug_cls = aug_metrics['by_class'].get(cls, {'total': 0, 'correct': 0})

        base_cls_acc = (base_cls['correct'] / base_cls['total'] * 100) if base_cls['total'] > 0 else 0
        aug_cls_acc = (aug_cls['correct'] / aug_cls['total'] * 100) if aug_cls['total'] > 0 else 0
        cls_diff = aug_cls_acc - base_cls_acc

        print(f"  {cls:<15} {base_cls_acc:>9.1f}% {aug_cls_acc:>9.1f}% {cls_diff:>+9.1f}%")
        print(f"  {'':<15} ({base_cls['correct']}/{base_cls['total']}) ({aug_cls['correct']}/{aug_cls['total']})")

    # Additional analysis: by geocoding quality
    print(f"\n{'='*60}")
    print("ANALYSIS BY GEOCODING QUALITY")
    print(f"{'='*60}")

    # Group by percentage of places geocoded
    def get_geocode_ratio(result):
        sample = sample_info.get(result['id'])
        if not sample or sample.get('num_places', 0) == 0:
            return 0
        return sample.get('num_geocoded', 0) / sample['num_places']

    # High quality: >80% geocoded
    high_quality_base = [r for r in base_filtered if get_geocode_ratio(r) >= 0.8]
    high_quality_aug = [r for r in aug_filtered if get_geocode_ratio(r) >= 0.8]

    hq_base_metrics = calc_metrics(high_quality_base)
    hq_aug_metrics = calc_metrics(high_quality_aug)

    hq_base_acc = hq_base_metrics['accuracy'] * 100
    hq_aug_acc = hq_aug_metrics['accuracy'] * 100
    hq_diff = hq_aug_acc - hq_base_acc

    print(f"\nHigh-quality geocoding (>=80% places geocoded): {len(high_quality_base)} samples")
    print(f"  Base:      {hq_base_acc:.1f}% ({hq_base_metrics['correct']}/{hq_base_metrics['total']})")
    print(f"  Augmented: {hq_aug_acc:.1f}% ({hq_aug_metrics['correct']}/{hq_aug_metrics['total']})")
    print(f"  Difference: {hq_diff:+.1f}%")

    # 100% geocoded
    full_geocode_base = [r for r in base_filtered if get_geocode_ratio(r) == 1.0]
    full_geocode_aug = [r for r in aug_filtered if get_geocode_ratio(r) == 1.0]

    if full_geocode_base:
        fg_base_metrics = calc_metrics(full_geocode_base)
        fg_aug_metrics = calc_metrics(full_geocode_aug)

        fg_base_acc = fg_base_metrics['accuracy'] * 100
        fg_aug_acc = fg_aug_metrics['accuracy'] * 100
        fg_diff = fg_aug_acc - fg_base_acc

        print(f"\nFull geocoding (100% places geocoded): {len(full_geocode_base)} samples")
        print(f"  Base:      {fg_base_acc:.1f}% ({fg_base_metrics['correct']}/{fg_base_metrics['total']})")
        print(f"  Augmented: {fg_aug_acc:.1f}% ({fg_aug_metrics['correct']}/{fg_aug_metrics['total']})")
        print(f"  Difference: {fg_diff:+.1f}%")


if __name__ == "__main__":
    main()
