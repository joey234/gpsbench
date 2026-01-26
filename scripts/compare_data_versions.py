#!/usr/bin/env python3
"""
Compare GPSBench v1 (data/) vs v2 (data_v2/)
Analyzes differences in sample counts, geographic coverage, and data quality.
"""

import json
import sys
from pathlib import Path
from collections import Counter
import argparse

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def count_samples(data_dir, split='test'):
    """Count samples in each task"""
    counts = {}
    split_dir = Path(data_dir)

    for track in ['track_pure_gps', 'track_applied']:
        track_path = split_dir / track / 'splits'
        if not track_path.exists():
            continue

        for split_file in track_path.glob(f'*_{split}.json'):
            task_name = split_file.name.replace(f'_{split}.json', '')
            try:
                data = load_json(split_file)
                counts[f"{track}/{task_name}"] = len(data)
            except Exception as e:
                counts[f"{track}/{task_name}"] = 0

    return counts

def analyze_geographic_coverage(task_file):
    """Simple geographic analysis"""
    try:
        data = load_json(task_file)
        # Count unique coordinates
        coords = set()
        for item in data[:100]:  # Sample first 100
            if 'point_a' in item:
                lat = item['point_a'].get('lat')
                lon = item['point_a'].get('lon')
                if lat and lon:
                    coords.add((round(lat, 2), round(lon, 2)))
            if 'point_b' in item:
                lat = item['point_b'].get('lat')
                lon = item['point_b'].get('lon')
                if lat and lon:
                    coords.add((round(lat, 2), round(lon, 2)))

        return len(coords)
    except:
        return 0

def compare_versions(v1_dir, v2_dir):
    """Compare two versions of GPSBench data"""

    print("="*80)
    print("GPSBench Version Comparison")
    print("="*80)
    print(f"v1: {v1_dir}")
    print(f"v2: {v2_dir}")
    print("="*80)

    # Count test samples
    print("\n📊 Test Sample Counts:")
    print("-"*80)

    v1_counts = count_samples(v1_dir, 'test')
    v2_counts = count_samples(v2_dir, 'test')

    all_tasks = sorted(set(list(v1_counts.keys()) + list(v2_counts.keys())))

    total_v1 = 0
    total_v2 = 0

    for task in all_tasks:
        v1_count = v1_counts.get(task, 0)
        v2_count = v2_counts.get(task, 0)
        total_v1 += v1_count
        total_v2 += v2_count

        diff = v2_count - v1_count
        if diff > 0:
            status = "📈"
            pct_change = f"+{(diff/v1_count*100):.0f}%" if v1_count > 0 else "NEW"
        elif diff < 0:
            status = "📉"
            pct_change = f"{(diff/v1_count*100):.0f}%"
        else:
            status = "="
            pct_change = "0%"

        task_short = task.split('/')[-1][:45]
        print(f"{status} {task_short:45} v1:{v1_count:5} → v2:{v2_count:5}  ({pct_change})")

    print("-"*80)
    print(f"{'TOTAL':45} v1:{total_v1:5} → v2:{total_v2:5}  ({total_v2-total_v1:+5})")

    increase_pct = ((total_v2 / total_v1) - 1) * 100 if total_v1 > 0 else 0
    print(f"\nOverall increase: {increase_pct:.1f}%")

    # Check for completeness
    print("\n✅ Completeness Check:")
    print("-"*80)

    tasks_at_target = sum(1 for count in v2_counts.values() if count >= 1000)
    tasks_total = len(v2_counts)
    print(f"Tasks with ≥1000 test samples: {tasks_at_target}/{tasks_total}")

    if tasks_at_target == tasks_total:
        print("✅ All tasks meet the 1000 sample target!")
    else:
        print(f"⚠️  {tasks_total - tasks_at_target} tasks below target")
        for task, count in v2_counts.items():
            if count < 1000:
                task_short = task.split('/')[-1]
                print(f"   - {task_short}: {count} (need {1000-count} more)")

    # File existence check
    print("\n📁 File Check:")
    print("-"*80)

    v1_files = set(v1_counts.keys())
    v2_files = set(v2_counts.keys())

    missing_in_v2 = v1_files - v2_files
    new_in_v2 = v2_files - v1_files

    if missing_in_v2:
        print(f"⚠️  Files in v1 but missing in v2: {len(missing_in_v2)}")
        for f in missing_in_v2:
            print(f"   - {f}")
    else:
        print("✅ All v1 files present in v2")

    if new_in_v2:
        print(f"📝 New files in v2: {len(new_in_v2)}")
        for f in new_in_v2:
            print(f"   + {f}")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if total_v2 > total_v1 * 2 and tasks_at_target >= tasks_total * 0.9:
        print("✅ Version 2 looks GOOD!")
        print(f"   - {increase_pct:.0f}% more test samples")
        print(f"   - {tasks_at_target}/{tasks_total} tasks meet 1000 sample target")
        print("\nSuggested action:")
        print("  1. Run: python scripts/verify_global_coverage.py")
        print("  2. If coverage is balanced, proceed with replacement:")
        print("     mv data data_v1_backup_$(date +%Y%m%d)")
        print("     mv data_v2 data")
    elif total_v2 > total_v1:
        print("⚠️  Version 2 is better, but some tasks need attention")
        print(f"   - {increase_pct:.0f}% more test samples")
        print(f"   - Only {tasks_at_target}/{tasks_total} tasks meet target")
        print("\nSuggested action:")
        print("  1. Review tasks below 1000 samples")
        print("  2. Re-generate specific tasks if needed")
        print("  3. Then proceed with replacement")
    else:
        print("❌ Version 2 did not increase sample count")
        print("\nSuggested action:")
        print("  1. Check generation logs for errors")
        print("  2. Verify generation scripts ran successfully")
        print("  3. Do NOT replace v1 with v2")

    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Compare GPSBench data versions')
    parser.add_argument('--v1-dir', default='data', help='Path to v1 data directory')
    parser.add_argument('--v2-dir', default='data_v2', help='Path to v2 data directory')
    parser.add_argument('--summary-only', action='store_true', help='Show only summary')
    args = parser.parse_args()

    v1_path = Path(args.v1_dir)
    v2_path = Path(args.v2_dir)

    if not v1_path.exists():
        print(f"❌ Error: {v1_path} does not exist")
        sys.exit(1)

    if not v2_path.exists():
        print(f"❌ Error: {v2_path} does not exist")
        print(f"   Run ./generate_v2_safe.sh first to generate v2 data")
        sys.exit(1)

    compare_versions(v1_path, v2_path)

if __name__ == "__main__":
    main()
