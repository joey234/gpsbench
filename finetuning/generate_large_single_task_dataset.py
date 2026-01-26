#!/usr/bin/env python3
"""
Generate large single-task dataset: Coordinate → Location
Uses all 32K+ cities from GeoNames with noise augmentation and prompt variations

Expected output: 100K+ training examples
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


def load_cities_data(cities_file: str, min_population: int = 0) -> List[Dict]:
    """
    Load city data from GeoNames cities15000.txt file.

    Format: geonameid, name, asciiname, alternatenames, latitude, longitude,
            feature class, feature code, country code, cc2, admin1 code,
            admin2 code, admin3 code, admin4 code, population, elevation,
            dem, timezone, modification date
    """
    cities = []

    print(f"Loading cities from {cities_file}...")
    with open(cities_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 19:
                continue

            try:
                population = int(parts[14]) if parts[14] else 0

                # Skip small cities if filter applied
                if population < min_population:
                    continue

                city_data = {
                    'geonameid': parts[0],
                    'name': parts[1],
                    'ascii_name': parts[2],
                    'latitude': float(parts[4]),
                    'longitude': float(parts[5]),
                    'country_code': parts[8],
                    'admin1': parts[10],  # State/province
                    'population': population,
                }
                cities.append(city_data)
            except (ValueError, IndexError) as e:
                continue

    print(f"Loaded {len(cities)} cities (min population: {min_population})")
    return cities


def add_gaussian_noise(coord: float, noise_std: float) -> float:
    """Add Gaussian noise to a coordinate value."""
    noise = np.random.normal(0, noise_std)
    return coord + noise


def get_prompt_variations() -> List[str]:
    """Get various prompt templates for diversity."""
    return [
        "What location is at coordinates {lat:.6f}, {lon:.6f}?",
        "Identify the location at {lat:.6f}, {lon:.6f}.",
        "Which place is located at coordinates {lat:.6f}, {lon:.6f}?",
        "What city or place can be found at {lat:.6f}, {lon:.6f}?",
        "Name the location at GPS coordinates {lat:.6f}, {lon:.6f}.",
        "What is the name of the place at coordinates ({lat:.6f}, {lon:.6f})?",
        "Coordinates: {lat:.6f}, {lon:.6f}. What location is this?",
        "Where are coordinates {lat:.6f}, {lon:.6f} located?",
        "What place corresponds to coordinates {lat:.6f}, {lon:.6f}?",
        "Identify this location: {lat:.6f}, {lon:.6f}",
    ]


def format_location_name(city: Dict, style: str = 'standard') -> str:
    """Format location name in different styles."""
    name = city['name']
    country = city['country_code']

    if style == 'name_only':
        return name
    elif style == 'name_country':
        return f"{name}, {country}"
    elif style == 'full':
        # For very popular cities, just name + country
        # For others, might include more detail
        if city['population'] > 1000000:
            return f"{name}, {country}"
        else:
            return f"{name}, {country}"
    else:  # standard
        return f"{name}, {country}"


def create_training_example(
    city: Dict,
    noise_std: float = 0.0,
    prompt_template: str = None,
    name_style: str = 'standard'
) -> Dict:
    """
    Create a training example with optional noise and variations.

    Args:
        city: City data dictionary
        noise_std: Standard deviation of Gaussian noise in degrees
                   0.001° ≈ 111m, 0.01° ≈ 1.1km
        prompt_template: Template for prompt (uses random if None)
        name_style: Style for formatting location name
    """
    lat = city['latitude']
    lon = city['longitude']

    # Add noise if specified
    if noise_std > 0:
        lat_noisy = add_gaussian_noise(lat, noise_std)
        lon_noisy = add_gaussian_noise(lon, noise_std)
    else:
        lat_noisy = lat
        lon_noisy = lon

    # Get prompt template
    if prompt_template is None:
        prompt_template = random.choice(get_prompt_variations())

    # Create prompt
    prompt = prompt_template.format(lat=lat_noisy, lon=lon_noisy)

    # Format answer
    answer = format_location_name(city, style=name_style)

    # System prompt
    system_prompt = """You are a geographic AI assistant. Given coordinates, identify the location name.
Provide clear, concise answers with the place name and country if relevant."""

    return {
        "input": prompt,
        "output": answer,
        "system_prompt": system_prompt,
        "metadata": {
            "task": "place_association",
            "geonameid": city['geonameid'],
            "original_lat": lat,
            "original_lon": lon,
            "noisy_lat": lat_noisy,
            "noisy_lon": lon_noisy,
            "noise_std": noise_std,
            "population": city['population'],
            "country": city['country_code']
        }
    }


def generate_dataset(
    cities: List[Dict],
    train_ratio: float = 0.85,
    dev_ratio: float = 0.10,
    # test_ratio = 0.05 (implicit)
    noise_levels: List[float] = [0.0, 0.001, 0.005, 0.01, 0.02],
    samples_per_noise: int = 2,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate training, dev, and test datasets with multiple noise levels.

    Strategy:
    - For each city, create:
      * 1 clean example (no noise)
      * N noisy examples per noise level (with different prompts)
    - This creates diversity and robustness

    Args:
        cities: List of city data
        train_ratio: Ratio of cities for training
        dev_ratio: Ratio of cities for dev
        noise_levels: List of noise standard deviations (in degrees)
        samples_per_noise: Number of samples per noise level
        seed: Random seed

    Returns:
        Tuple of (train_data, dev_data, test_data)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Shuffle cities
    cities = cities.copy()
    random.shuffle(cities)

    # Split cities
    n_cities = len(cities)
    train_end = int(n_cities * train_ratio)
    dev_end = train_end + int(n_cities * dev_ratio)

    train_cities = cities[:train_end]
    dev_cities = cities[train_end:dev_end]
    test_cities = cities[dev_end:]

    print(f"\nCity splits:")
    print(f"  Train: {len(train_cities)} cities")
    print(f"  Dev:   {len(dev_cities)} cities")
    print(f"  Test:  {len(test_cities)} cities")

    def generate_examples_for_cities(city_list: List[Dict], is_train: bool = True) -> List[Dict]:
        """Generate examples for a list of cities."""
        examples = []

        for city in city_list:
            # Clean example (no noise)
            examples.append(create_training_example(city, noise_std=0.0))

            # Noisy examples with different levels
            for noise_std in noise_levels:
                if noise_std == 0.0:
                    continue  # Already added clean example

                n_samples = samples_per_noise if is_train else 1
                for _ in range(n_samples):
                    examples.append(create_training_example(
                        city,
                        noise_std=noise_std
                    ))

        random.shuffle(examples)
        return examples

    print(f"\nGenerating training examples...")
    train_data = generate_examples_for_cities(train_cities, is_train=True)

    print(f"Generating dev examples...")
    dev_data = generate_examples_for_cities(dev_cities, is_train=False)

    print(f"Generating test examples...")
    test_data = generate_examples_for_cities(test_cities, is_train=False)

    return train_data, dev_data, test_data


def save_dataset(data: List[Dict], output_file: Path):
    """Save dataset to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {len(data)} examples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate large single-task dataset: coordinate → location"
    )
    parser.add_argument(
        '--cities-file',
        type=str,
        default='../data/raw/cities15000.txt',
        help='Path to cities15000.txt file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--min-population',
        type=int,
        default=0,
        help='Minimum city population (0 = all cities, 50000 = major cities only)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.85,
        help='Ratio of cities for training'
    )
    parser.add_argument(
        '--dev-ratio',
        type=float,
        default=0.10,
        help='Ratio of cities for dev'
    )
    parser.add_argument(
        '--noise-levels',
        type=float,
        nargs='+',
        default=[0.0, 0.001, 0.005, 0.01, 0.02],
        help='Noise standard deviations (0.001° ≈ 111m, 0.01° ≈ 1.1km)'
    )
    parser.add_argument(
        '--samples-per-noise',
        type=int,
        default=2,
        help='Number of samples per noise level (higher = more data)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Large Single-Task Dataset Generation")
    print("Task: Coordinate → Location (Place Association)")
    print("=" * 80)

    # Load cities
    cities = load_cities_data(args.cities_file, min_population=args.min_population)

    if len(cities) == 0:
        print("ERROR: No cities loaded!")
        return

    # Calculate expected dataset size
    noise_with_samples = [n for n in args.noise_levels if n > 0]
    examples_per_city = 1 + (len(noise_with_samples) * args.samples_per_noise)
    expected_train = int(len(cities) * args.train_ratio) * examples_per_city

    print(f"\nDataset parameters:")
    print(f"  Cities: {len(cities)}")
    print(f"  Noise levels: {args.noise_levels}")
    print(f"  Samples per noise: {args.samples_per_noise}")
    print(f"  Examples per city: {examples_per_city}")
    print(f"  Expected train size: ~{expected_train:,} examples")

    # Generate dataset
    print(f"\nGenerating dataset...")
    train_data, dev_data, test_data = generate_dataset(
        cities=cities,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        noise_levels=args.noise_levels,
        samples_per_noise=args.samples_per_noise,
        seed=args.seed
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets
    print(f"\nSaving datasets to {output_dir}/...")
    save_dataset(train_data, output_dir / 'train_single_task_large.json')
    save_dataset(dev_data, output_dir / 'dev_single_task_large.json')
    save_dataset(test_data, output_dir / 'test_single_task_large.json')

    # Print statistics
    print(f"\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    print(f"Training examples:   {len(train_data):>8,}")
    print(f"Dev examples:        {len(dev_data):>8,}")
    print(f"Test examples:       {len(test_data):>8,}")
    print(f"Total examples:      {len(train_data) + len(dev_data) + len(test_data):>8,}")

    # Analyze noise distribution
    noise_counts = {}
    for ex in train_data:
        noise = ex['metadata']['noise_std']
        noise_counts[noise] = noise_counts.get(noise, 0) + 1

    print(f"\nNoise distribution in training:")
    for noise in sorted(noise_counts.keys()):
        count = noise_counts[noise]
        pct = count / len(train_data) * 100
        meters = noise * 111000  # Convert degrees to meters (approx)
        print(f"  {noise:.4f}° (~{meters:>6.0f}m): {count:>8,} examples ({pct:>5.1f}%)")

    # Sample examples
    print(f"\n" + "=" * 80)
    print("Sample Training Examples")
    print("=" * 80)

    for i, example in enumerate(train_data[:5]):
        noise = example['metadata']['noise_std']
        meters = noise * 111000
        print(f"\nExample {i+1} (noise: {noise:.4f}° ≈ {meters:.0f}m):")
        print(f"  Input:  {example['input']}")
        print(f"  Output: {example['output']}")

    print(f"\n" + "=" * 80)
    print("Dataset generation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Review: head -100 {output_dir}/train_single_task_large.json")
    print(f"2. Train:  python scripts/train_single_task.py \\")
    print(f"             --train-file {output_dir}/train_single_task_large.json \\")
    print(f"             --dev-file {output_dir}/dev_single_task_large.json")


if __name__ == '__main__':
    main()
