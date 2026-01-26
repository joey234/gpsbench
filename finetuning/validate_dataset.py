#!/usr/bin/env python3
"""
Validate generated dataset for correctness and noise distribution.

Checks:
1. Coordinate validity (lat in [-90, 90], lon in [-180, 180])
2. Noise is actually applied (noisy coords != original)
3. Noise distribution matches expected (Gaussian)
4. Location names are valid (non-empty, proper format)
5. Coverage statistics (cities per country, coordinate distribution)
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict
import matplotlib.pyplot as plt


def load_dataset(file_path: Path) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def validate_coordinates(data: List[Dict]) -> Dict:
    """Validate coordinate ranges."""
    print("\n" + "="*80)
    print("1. COORDINATE VALIDITY CHECK")
    print("="*80)

    invalid_lat = []
    invalid_lon = []

    for i, ex in enumerate(data):
        meta = ex['metadata']
        lat = meta.get('original_lat') or meta.get('noisy_lat')
        lon = meta.get('original_lon') or meta.get('noisy_lon')

        if lat is None or lon is None:
            continue

        if not (-90 <= lat <= 90):
            invalid_lat.append((i, lat))
        if not (-180 <= lon <= 180):
            invalid_lon.append((i, lon))

    print(f"✓ Total examples: {len(data)}")
    print(f"✓ Valid latitudes: {len(data) - len(invalid_lat)}/{len(data)}")
    print(f"✓ Valid longitudes: {len(data) - len(invalid_lon)}/{len(data)}")

    if invalid_lat:
        print(f"\n⚠️  Found {len(invalid_lat)} invalid latitudes:")
        for i, lat in invalid_lat[:5]:
            print(f"   Example {i}: lat={lat}")

    if invalid_lon:
        print(f"\n⚠️  Found {len(invalid_lon)} invalid longitudes:")
        for i, lon in invalid_lon[:5]:
            print(f"   Example {i}: lon={lon}")

    return {
        'total': len(data),
        'invalid_lat': len(invalid_lat),
        'invalid_lon': len(invalid_lon),
        'valid': len(data) - len(invalid_lat) - len(invalid_lon)
    }


def validate_noise_application(data: List[Dict]) -> Dict:
    """Validate that noise is actually applied."""
    print("\n" + "="*80)
    print("2. NOISE APPLICATION CHECK")
    print("="*80)

    noise_levels = defaultdict(int)
    noise_distances = defaultdict(list)

    clean_count = 0
    noisy_count = 0
    noise_not_applied = 0

    for ex in data:
        meta = ex['metadata']
        noise_std = meta.get('noise_std', 0.0)

        noise_levels[noise_std] += 1

        if noise_std == 0.0:
            clean_count += 1
        else:
            noisy_count += 1

            # Check if noise was actually applied
            orig_lat = meta.get('original_lat')
            orig_lon = meta.get('original_lon')
            noisy_lat = meta.get('noisy_lat')
            noisy_lon = meta.get('noisy_lon')

            if orig_lat and noisy_lat and orig_lon and noisy_lon:
                lat_diff = abs(noisy_lat - orig_lat)
                lon_diff = abs(noisy_lon - orig_lon)

                # Calculate actual noise distance
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                noise_distances[noise_std].append(distance)

                # If noise std > 0 but coords are identical, flag it
                if lat_diff == 0 and lon_diff == 0:
                    noise_not_applied += 1

    print(f"\n✓ Clean examples (no noise): {clean_count} ({clean_count/len(data)*100:.1f}%)")
    print(f"✓ Noisy examples: {noisy_count} ({noisy_count/len(data)*100:.1f}%)")

    if noise_not_applied > 0:
        print(f"\n⚠️  WARNING: {noise_not_applied} examples have noise_std > 0 but identical coordinates!")

    print(f"\nNoise distribution:")
    for noise_std in sorted(noise_levels.keys()):
        count = noise_levels[noise_std]
        pct = count / len(data) * 100
        meters = noise_std * 111000
        print(f"  {noise_std:.4f}° (~{meters:>6.0f}m): {count:>8,} examples ({pct:>5.1f}%)")

        # Show actual noise statistics
        if noise_std > 0 and noise_distances[noise_std]:
            distances = noise_distances[noise_std]
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            mean_meters = mean_dist * 111000
            std_meters = std_dist * 111000
            print(f"    → Actual noise: mean={mean_dist:.4f}° (~{mean_meters:.0f}m), "
                  f"std={std_dist:.4f}° (~{std_meters:.0f}m)")

    return {
        'clean_count': clean_count,
        'noisy_count': noisy_count,
        'noise_not_applied': noise_not_applied,
        'noise_distances': dict(noise_distances)
    }


def validate_gaussian_distribution(noise_distances: Dict) -> None:
    """Validate that noise follows Gaussian distribution."""
    print("\n" + "="*80)
    print("3. GAUSSIAN DISTRIBUTION CHECK")
    print("="*80)

    for noise_std, distances in noise_distances.items():
        if noise_std == 0:
            continue

        distances = np.array(distances)

        # Expected: mean ≈ 0 (for individual lat/lon), but distance is always positive
        # For 2D Gaussian, distance follows Rayleigh distribution
        # Mean distance ≈ 1.253 * noise_std
        expected_mean = 1.253 * noise_std
        actual_mean = np.mean(distances)

        # Check if within reasonable range (±30%)
        lower_bound = expected_mean * 0.7
        upper_bound = expected_mean * 1.3

        within_range = lower_bound <= actual_mean <= upper_bound
        status = "✓" if within_range else "⚠️"

        print(f"\n{status} Noise level: {noise_std:.4f}° (~{noise_std*111000:.0f}m)")
        print(f"   Expected mean distance: {expected_mean:.4f}° (~{expected_mean*111000:.0f}m)")
        print(f"   Actual mean distance:   {actual_mean:.4f}° (~{actual_mean*111000:.0f}m)")
        print(f"   Deviation: {(actual_mean/expected_mean - 1)*100:+.1f}%")

        if not within_range:
            print(f"   ⚠️  WARNING: Mean distance outside expected range!")


def validate_location_names(data: List[Dict]) -> Dict:
    """Validate location names."""
    print("\n" + "="*80)
    print("4. LOCATION NAME VALIDATION")
    print("="*80)

    empty_names = 0
    missing_country = 0
    format_issues = 0

    country_counts = Counter()

    for i, ex in enumerate(data):
        output = ex.get('output', '')

        if not output or output.strip() == '':
            empty_names += 1
            continue

        # Check format: should be "City, CC" or just "City"
        parts = output.split(',')

        if len(parts) == 2:
            city, country = [p.strip() for p in parts]
            country_counts[country] += 1
        elif len(parts) == 1:
            missing_country += 1
        else:
            format_issues += 1

    print(f"✓ Total examples: {len(data)}")
    print(f"✓ Valid location names: {len(data) - empty_names - format_issues}")
    print(f"  - With country code: {len(data) - empty_names - missing_country - format_issues}")
    print(f"  - City only: {missing_country}")

    if empty_names > 0:
        print(f"\n⚠️  Found {empty_names} empty location names")
    if format_issues > 0:
        print(f"⚠️  Found {format_issues} format issues (unexpected commas)")

    print(f"\nTop 10 countries by examples:")
    for country, count in country_counts.most_common(10):
        pct = count / len(data) * 100
        print(f"  {country}: {count:>6,} examples ({pct:>5.1f}%)")

    return {
        'total': len(data),
        'empty_names': empty_names,
        'missing_country': missing_country,
        'format_issues': format_issues,
        'country_counts': dict(country_counts)
    }


def validate_coverage(data: List[Dict]) -> Dict:
    """Validate geographic coverage."""
    print("\n" + "="*80)
    print("5. GEOGRAPHIC COVERAGE CHECK")
    print("="*80)

    lats = []
    lons = []
    populations = []

    for ex in data:
        meta = ex['metadata']
        lat = meta.get('original_lat') or meta.get('noisy_lat')
        lon = meta.get('original_lon') or meta.get('noisy_lon')
        pop = meta.get('population', 0)

        if lat is not None:
            lats.append(lat)
        if lon is not None:
            lons.append(lon)
        if pop > 0:
            populations.append(pop)

    lats = np.array(lats)
    lons = np.array(lons)

    print(f"Latitude coverage:")
    print(f"  Min: {np.min(lats):.2f}°")
    print(f"  Max: {np.max(lats):.2f}°")
    print(f"  Mean: {np.mean(lats):.2f}°")
    print(f"  Std: {np.std(lats):.2f}°")

    print(f"\nLongitude coverage:")
    print(f"  Min: {np.min(lons):.2f}°")
    print(f"  Max: {np.max(lons):.2f}°")
    print(f"  Mean: {np.mean(lons):.2f}°")
    print(f"  Std: {np.std(lons):.2f}°")

    if populations:
        populations = np.array(populations)
        print(f"\nPopulation distribution:")
        print(f"  Min: {np.min(populations):,}")
        print(f"  Max: {np.max(populations):,}")
        print(f"  Median: {np.median(populations):,.0f}")
        print(f"  Mean: {np.mean(populations):,.0f}")

    # Check hemisphere coverage
    northern = np.sum(lats >= 0)
    southern = np.sum(lats < 0)
    eastern = np.sum(lons >= 0)
    western = np.sum(lons < 0)

    print(f"\nHemisphere distribution:")
    print(f"  Northern: {northern:>6,} ({northern/len(lats)*100:.1f}%)")
    print(f"  Southern: {southern:>6,} ({southern/len(lats)*100:.1f}%)")
    print(f"  Eastern:  {eastern:>6,} ({eastern/len(lons)*100:.1f}%)")
    print(f"  Western:  {western:>6,} ({western/len(lons)*100:.1f}%)")

    return {
        'lat_range': (float(np.min(lats)), float(np.max(lats))),
        'lon_range': (float(np.min(lons)), float(np.max(lons))),
        'northern': int(northern),
        'southern': int(southern),
        'eastern': int(eastern),
        'western': int(western)
    }


def validate_prompt_diversity(data: List[Dict]) -> Dict:
    """Validate prompt variations."""
    print("\n" + "="*80)
    print("6. PROMPT DIVERSITY CHECK")
    print("="*80)

    prompts = [ex['input'] for ex in data]

    # Extract prompt templates (remove numbers)
    import re
    templates = []
    for prompt in prompts:
        # Replace coordinate numbers with placeholder
        template = re.sub(r'-?\d+\.\d+', 'X', prompt)
        templates.append(template)

    template_counts = Counter(templates)

    print(f"✓ Total prompts: {len(prompts)}")
    print(f"✓ Unique templates: {len(template_counts)}")

    print(f"\nPrompt template distribution:")
    for i, (template, count) in enumerate(template_counts.most_common(), 1):
        pct = count / len(prompts) * 100
        # Show template (truncated)
        template_short = template[:60] + '...' if len(template) > 60 else template
        print(f"  {i}. [{count:>8,} uses, {pct:>5.1f}%] {template_short}")

    # Check if distribution is balanced
    expected_per_template = len(prompts) / len(template_counts)
    max_deviation = max(abs(count - expected_per_template) / expected_per_template
                       for _, count in template_counts.items())

    if max_deviation > 0.5:  # More than 50% deviation
        print(f"\n⚠️  WARNING: Unbalanced template distribution (max deviation: {max_deviation*100:.0f}%)")
    else:
        print(f"\n✓ Templates are well-balanced (max deviation: {max_deviation*100:.0f}%)")

    return {
        'total_prompts': len(prompts),
        'unique_templates': len(template_counts),
        'template_distribution': dict(template_counts.most_common(5))
    }


def show_sample_examples(data: List[Dict], n: int = 5) -> None:
    """Show sample examples for manual inspection."""
    print("\n" + "="*80)
    print("7. SAMPLE EXAMPLES")
    print("="*80)

    # Show examples with different noise levels
    noise_levels = sorted(set(ex['metadata'].get('noise_std', 0) for ex in data))

    for noise_std in noise_levels[:3]:  # Show first 3 noise levels
        examples = [ex for ex in data if ex['metadata'].get('noise_std') == noise_std]
        if not examples:
            continue

        meters = noise_std * 111000
        print(f"\nNoise level: {noise_std:.4f}° (~{meters:.0f}m)")
        print("-" * 80)

        for i, ex in enumerate(examples[:2], 1):  # 2 examples per noise level
            meta = ex['metadata']
            print(f"\nExample {i}:")
            print(f"  Input:  {ex['input']}")
            print(f"  Output: {ex['output']}")

            if noise_std > 0:
                orig_lat = meta.get('original_lat', 0)
                orig_lon = meta.get('original_lon', 0)
                noisy_lat = meta.get('noisy_lat', 0)
                noisy_lon = meta.get('noisy_lon', 0)

                lat_diff = abs(noisy_lat - orig_lat) * 111000  # meters
                lon_diff = abs(noisy_lon - orig_lon) * 111000  # meters

                print(f"  Original: ({orig_lat:.6f}, {orig_lon:.6f})")
                print(f"  Noisy:    ({noisy_lat:.6f}, {noisy_lon:.6f})")
                print(f"  Offset:   ({lat_diff:.0f}m lat, {lon_diff:.0f}m lon)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate dataset quality')
    parser.add_argument('--file', type=str, required=True,
                       help='Dataset file to validate')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of sample examples to show')

    args = parser.parse_args()

    file_path = Path(args.file)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    print("="*80)
    print(f"DATASET VALIDATION: {file_path.name}")
    print("="*80)
    print(f"\nFile: {file_path}")
    print(f"Size: {file_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Load data
    print("\nLoading dataset...")
    data = load_dataset(file_path)
    print(f"✓ Loaded {len(data):,} examples")

    # Run validations
    results = {}

    results['coordinates'] = validate_coordinates(data)
    results['noise'] = validate_noise_application(data)

    # Validate Gaussian distribution if we have noise
    if results['noise']['noise_distances']:
        validate_gaussian_distribution(results['noise']['noise_distances'])

    results['location_names'] = validate_location_names(data)
    results['coverage'] = validate_coverage(data)
    results['prompts'] = validate_prompt_diversity(data)

    # Show samples
    show_sample_examples(data, n=args.samples)

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    all_good = True

    if results['coordinates']['invalid_lat'] == 0 and results['coordinates']['invalid_lon'] == 0:
        print("✓ All coordinates are valid")
    else:
        print(f"⚠️  Invalid coordinates found")
        all_good = False

    if results['noise']['noise_not_applied'] == 0:
        print("✓ Noise is properly applied")
    else:
        print(f"⚠️  {results['noise']['noise_not_applied']} examples missing noise")
        all_good = False

    if results['location_names']['empty_names'] == 0:
        print("✓ All location names are valid")
    else:
        print(f"⚠️  {results['location_names']['empty_names']} empty names found")
        all_good = False

    if results['prompts']['unique_templates'] >= 5:
        print(f"✓ Good prompt diversity ({results['prompts']['unique_templates']} templates)")
    else:
        print(f"⚠️  Low prompt diversity ({results['prompts']['unique_templates']} templates)")
        all_good = False

    print()
    if all_good:
        print("✅ DATASET VALIDATION PASSED - Ready for training!")
    else:
        print("⚠️  DATASET VALIDATION FOUND ISSUES - Review warnings above")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
