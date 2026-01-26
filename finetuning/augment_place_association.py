#!/usr/bin/env python3
"""
Augment place_association training data with Gaussian noise.

This script generates additional training samples by:
1. Adding Gaussian noise to coordinates at different levels
2. Varying coordinate precision/formatting
3. Creating multiple samples per original location

Usage:
    python augment_place_association.py --samples-per-location 5 --output-samples 10000
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np

# Country code to full name mapping
COUNTRY_NAMES = {
    'US': 'United States', 'GB': 'United Kingdom', 'CN': 'China', 'IN': 'India',
    'JP': 'Japan', 'DE': 'Germany', 'FR': 'France', 'BR': 'Brazil', 'IT': 'Italy',
    'CA': 'Canada', 'RU': 'Russia', 'KR': 'South Korea', 'ES': 'Spain', 'MX': 'Mexico',
    'AU': 'Australia', 'ID': 'Indonesia', 'NL': 'Netherlands', 'SA': 'Saudi Arabia',
    'TR': 'Turkey', 'CH': 'Switzerland', 'PL': 'Poland', 'BE': 'Belgium', 'SE': 'Sweden',
    'NG': 'Nigeria', 'AR': 'Argentina', 'NO': 'Norway', 'AT': 'Austria', 'AE': 'UAE',
    'IL': 'Israel', 'HK': 'Hong Kong', 'IE': 'Ireland', 'SG': 'Singapore', 'DK': 'Denmark',
    'MY': 'Malaysia', 'CO': 'Colombia', 'PH': 'Philippines', 'PK': 'Pakistan', 'CL': 'Chile',
    'FI': 'Finland', 'BD': 'Bangladesh', 'EG': 'Egypt', 'VN': 'Vietnam', 'CZ': 'Czech Republic',
    'PT': 'Portugal', 'RO': 'Romania', 'NZ': 'New Zealand', 'IR': 'Iran', 'TH': 'Thailand',
    'ZA': 'South Africa', 'GR': 'Greece', 'HU': 'Hungary', 'UA': 'Ukraine', 'DZ': 'Algeria',
    'GT': 'Guatemala', 'PE': 'Peru', 'EC': 'Ecuador', 'VE': 'Venezuela', 'CU': 'Cuba',
    'BO': 'Bolivia', 'DO': 'Dominican Republic', 'HN': 'Honduras', 'PY': 'Paraguay',
    'SV': 'El Salvador', 'NI': 'Nicaragua', 'CR': 'Costa Rica', 'PA': 'Panama', 'UY': 'Uruguay',
    'KE': 'Kenya', 'ET': 'Ethiopia', 'TZ': 'Tanzania', 'UG': 'Uganda', 'GH': 'Ghana',
    'CI': 'Ivory Coast', 'CM': 'Cameroon', 'MG': 'Madagascar', 'AO': 'Angola', 'MZ': 'Mozambique',
    'NE': 'Niger', 'ML': 'Mali', 'BF': 'Burkina Faso', 'MW': 'Malawi', 'ZM': 'Zambia',
    'ZW': 'Zimbabwe', 'SN': 'Senegal', 'TD': 'Chad', 'SO': 'Somalia', 'GN': 'Guinea',
    'RW': 'Rwanda', 'BJ': 'Benin', 'TN': 'Tunisia', 'SS': 'South Sudan', 'TG': 'Togo',
    'SL': 'Sierra Leone', 'LY': 'Libya', 'CG': 'Congo', 'LR': 'Liberia', 'CF': 'Central African Republic',
    'MR': 'Mauritania', 'ER': 'Eritrea', 'GM': 'Gambia', 'BW': 'Botswana', 'NA': 'Namibia',
    'GA': 'Gabon', 'LS': 'Lesotho', 'GW': 'Guinea-Bissau', 'MU': 'Mauritius', 'SZ': 'Eswatini',
    'DJ': 'Djibouti', 'RE': 'Reunion', 'KM': 'Comoros', 'CV': 'Cape Verde', 'ST': 'Sao Tome and Principe',
    'SC': 'Seychelles', 'MM': 'Myanmar', 'AF': 'Afghanistan', 'NP': 'Nepal', 'KH': 'Cambodia',
    'LA': 'Laos', 'LK': 'Sri Lanka', 'KZ': 'Kazakhstan', 'UZ': 'Uzbekistan', 'TM': 'Turkmenistan',
    'KG': 'Kyrgyzstan', 'TJ': 'Tajikistan', 'AZ': 'Azerbaijan', 'GE': 'Georgia', 'AM': 'Armenia',
    'MN': 'Mongolia', 'BN': 'Brunei', 'TL': 'Timor-Leste', 'BT': 'Bhutan', 'MV': 'Maldives',
    'IQ': 'Iraq', 'SY': 'Syria', 'JO': 'Jordan', 'LB': 'Lebanon', 'PS': 'Palestine',
    'YE': 'Yemen', 'OM': 'Oman', 'KW': 'Kuwait', 'BH': 'Bahrain', 'QA': 'Qatar',
    'CY': 'Cyprus', 'AL': 'Albania', 'MK': 'North Macedonia', 'RS': 'Serbia', 'BA': 'Bosnia and Herzegovina',
    'HR': 'Croatia', 'SI': 'Slovenia', 'SK': 'Slovakia', 'LT': 'Lithuania', 'LV': 'Latvia',
    'EE': 'Estonia', 'BY': 'Belarus', 'MD': 'Moldova', 'BG': 'Bulgaria', 'ME': 'Montenegro',
    'XK': 'Kosovo', 'IS': 'Iceland', 'LU': 'Luxembourg', 'MT': 'Malta', 'AD': 'Andorra',
    'MC': 'Monaco', 'SM': 'San Marino', 'LI': 'Liechtenstein', 'VA': 'Vatican City',
    'FJ': 'Fiji', 'PG': 'Papua New Guinea', 'SB': 'Solomon Islands', 'VU': 'Vanuatu',
    'NC': 'New Caledonia', 'PF': 'French Polynesia', 'WS': 'Samoa', 'GU': 'Guam',
    'KI': 'Kiribati', 'FM': 'Micronesia', 'TO': 'Tonga', 'PW': 'Palau', 'MH': 'Marshall Islands',
    'TV': 'Tuvalu', 'NR': 'Nauru', 'JM': 'Jamaica', 'TT': 'Trinidad and Tobago', 'BS': 'Bahamas',
    'BB': 'Barbados', 'LC': 'Saint Lucia', 'GD': 'Grenada', 'VC': 'Saint Vincent', 'AG': 'Antigua and Barbuda',
    'DM': 'Dominica', 'KN': 'Saint Kitts and Nevis', 'HT': 'Haiti', 'BZ': 'Belize', 'SR': 'Suriname',
    'GY': 'Guyana', 'TW': 'Taiwan', 'PR': 'Puerto Rico', 'KP': 'North Korea', 'CD': 'DR Congo',
}

# Noise levels in degrees (approximately in meters at equator)
# 0.0001° ≈ 11m, 0.001° ≈ 111m, 0.01° ≈ 1.1km, 0.1° ≈ 11km
NOISE_LEVELS = {
    'precise': 0.0001,      # ~11m - very precise GPS
    'standard': 0.001,      # ~111m - standard GPS
    'coarse': 0.005,        # ~555m - coarse GPS
    'city_level': 0.01,     # ~1.1km - city level
    'region_level': 0.05,   # ~5.5km - region level
}


def add_gaussian_noise(lat: float, lon: float, noise_std: float) -> tuple:
    """Add Gaussian noise to coordinates."""
    noisy_lat = lat + np.random.normal(0, noise_std)
    noisy_lon = lon + np.random.normal(0, noise_std)

    # Clamp to valid ranges
    noisy_lat = max(-90, min(90, noisy_lat))
    noisy_lon = max(-180, min(180, noisy_lon))

    return noisy_lat, noisy_lon


def format_coordinate(lat: float, lon: float, precision: int = 6) -> str:
    """Format coordinates with variable precision."""
    return f"{lat:.{precision}f}, {lon:.{precision}f}"


def create_augmented_sample(
    original: Dict,
    noise_level: str,
    coord_precision: int = 6
) -> Dict:
    """Create an augmented training sample with noise."""

    coord = original.get('coordinate', {})
    gt = original.get('ground_truth', {})

    # Get original coordinates
    lat = coord.get('lat', coord.get('true_lat', 0))
    lon = coord.get('lon', coord.get('true_lon', 0))

    # Add noise
    noise_std = NOISE_LEVELS.get(noise_level, 0.001)
    noisy_lat, noisy_lon = add_gaussian_noise(lat, lon, noise_std)

    # Format coordinate string
    coord_str = format_coordinate(noisy_lat, noisy_lon, coord_precision)

    # Get location info
    location_name = gt.get('location_name', gt.get('name', 'Unknown'))
    country_code = gt.get('country_code', gt.get('country', ''))
    country_name = COUNTRY_NAMES.get(country_code, country_code)

    # Create instruction (matching benchmark format)
    instruction = (
        f"What location is at coordinates {coord_str}?\n\n"
        f"Provide your answer in the format: City, Region/State, Country "
        f"(e.g., 'Paris, Île-de-France, France' or 'Tokyo, Tokyo, Japan').\n\n"
        f"FINAL ANSWER: [your answer]"
    )

    # Create output
    if country_name:
        output = f"FINAL ANSWER: {location_name}, {country_name}"
    else:
        output = f"FINAL ANSWER: {location_name}"

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "metadata": {
            "task": "place_association",
            "source_task": "augmented/place_association",
            "noise_level": noise_level,
            "noise_std_deg": noise_std,
            "original_lat": lat,
            "original_lon": lon,
            "noisy_lat": noisy_lat,
            "noisy_lon": noisy_lon,
            "coord_precision": coord_precision,
            "location_name": location_name,
            "country_code": country_code,
        }
    }


def load_original_data(data_path: Path) -> List[Dict]:
    """Load original place_association data from split file."""
    split_file = data_path / "track_applied" / "splits" / "task1_place_association_train.json"

    with open(split_file) as f:
        return json.load(f)


def augment_dataset(
    original_data: List[Dict],
    samples_per_location: int = 5,
    target_samples: int = None,
    include_original: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Augment the dataset with noisy samples.

    Args:
        original_data: Original training examples
        samples_per_location: Number of augmented samples per original location
        target_samples: Target total samples (if set, overrides samples_per_location)
        include_original: Whether to include original (non-noisy) samples
        seed: Random seed for reproducibility

    Returns:
        Augmented dataset
    """
    random.seed(seed)
    np.random.seed(seed)

    augmented = []
    noise_levels = list(NOISE_LEVELS.keys())
    precisions = [4, 5, 6]  # Variable precision

    # Calculate samples per location if target is set
    if target_samples:
        samples_per_location = max(1, target_samples // len(original_data))

    for original in original_data:
        # Optionally include the original (precise) sample
        if include_original:
            orig_sample = create_augmented_sample(original, 'precise', coord_precision=6)
            orig_sample['metadata']['is_original'] = True
            augmented.append(orig_sample)

        # Create augmented samples with different noise levels
        for i in range(samples_per_location):
            noise_level = random.choice(noise_levels)
            precision = random.choice(precisions)

            aug_sample = create_augmented_sample(original, noise_level, precision)
            aug_sample['metadata']['is_original'] = False
            augmented.append(aug_sample)

    # Shuffle
    random.shuffle(augmented)

    # Trim to target if specified
    if target_samples and len(augmented) > target_samples:
        augmented = augmented[:target_samples]

    return augmented


def main():
    parser = argparse.ArgumentParser(
        description="Augment place_association data with Gaussian noise"
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='../data',
        help='Path to GPSBench data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for augmented data'
    )
    parser.add_argument(
        '--samples-per-location',
        type=int,
        default=5,
        help='Number of augmented samples per original location'
    )
    parser.add_argument(
        '--target-samples',
        type=int,
        default=None,
        help='Target total number of samples (overrides samples-per-location)'
    )
    parser.add_argument(
        '--include-original',
        action='store_true',
        default=True,
        help='Include original (non-noisy) samples'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Place Association Data Augmentation")
    print("=" * 70)
    print()

    # Load original data
    data_path = Path(args.data_path)
    print(f"Loading original data from {data_path}...")
    original_data = load_original_data(data_path)
    print(f"Loaded {len(original_data)} original samples")
    print()

    # Augment
    print("Augmenting data...")
    print(f"  Samples per location: {args.samples_per_location}")
    print(f"  Target samples: {args.target_samples or 'auto'}")
    print(f"  Include original: {args.include_original}")
    print(f"  Noise levels: {list(NOISE_LEVELS.keys())}")
    print()

    augmented = augment_dataset(
        original_data,
        samples_per_location=args.samples_per_location,
        target_samples=args.target_samples,
        include_original=args.include_original,
        seed=args.seed
    )

    print(f"Generated {len(augmented)} augmented samples")

    # Split into train/dev (90/10)
    split_idx = int(len(augmented) * 0.9)
    train_data = augmented[:split_idx]
    dev_data = augmented[split_idx:]

    print(f"  Train: {len(train_data)}")
    print(f"  Dev: {len(dev_data)}")
    print()

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    train_file = output_dir / "train_place_association_augmented.json"
    dev_file = output_dir / "dev_place_association_augmented.json"

    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved: {train_file}")

    with open(dev_file, 'w') as f:
        json.dump(dev_data, f, indent=2)
    print(f"Saved: {dev_file}")

    # Show sample
    print()
    print("=" * 70)
    print("Sample Augmented Examples")
    print("=" * 70)

    for i, sample in enumerate(train_data[:3]):
        meta = sample['metadata']
        print(f"\nExample {i+1} ({meta['noise_level']}, precision={meta['coord_precision']}):")
        print(f"  Instruction: {sample['instruction'][:80]}...")
        print(f"  Output: {sample['output']}")
        print(f"  Noise: {meta['noise_std_deg']*111:.0f}m")

    # Stats by noise level
    print()
    print("=" * 70)
    print("Distribution by Noise Level")
    print("=" * 70)

    from collections import Counter
    noise_dist = Counter(s['metadata']['noise_level'] for s in train_data)
    for level, count in sorted(noise_dist.items()):
        pct = count / len(train_data) * 100
        print(f"  {level}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
