"""
Generate coordinate-to-location dataset with Gaussian noise for fine-tuning.

This script:
1. Loads city data from GPSBench's cities15000.txt
2. Extracts coordinates and location names
3. Adds Gaussian noise to coordinates for robustness
4. Creates training and validation splits
5. Saves datasets in JSON format
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


def load_cities_data(cities_file: str, max_cities: int = 10000) -> List[Dict]:
    """
    Load city data from GeoNames cities15000.txt file.

    Format: geonameid, name, asciiname, alternatenames, latitude, longitude,
            feature class, feature code, country code, cc2, admin1 code,
            admin2 code, admin3 code, admin4 code, population, elevation,
            dem, timezone, modification date
    """
    cities = []

    with open(cities_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_cities:
                break

            parts = line.strip().split('\t')
            if len(parts) < 19:
                continue

            try:
                city_data = {
                    'geonameid': parts[0],
                    'name': parts[1],
                    'ascii_name': parts[2],
                    'latitude': float(parts[4]),
                    'longitude': float(parts[5]),
                    'country_code': parts[8],
                    'population': int(parts[14]) if parts[14] else 0,
                }
                cities.append(city_data)
            except (ValueError, IndexError) as e:
                print(f"Skipping line {i}: {e}")
                continue

    return cities


def add_gaussian_noise(coord: float, noise_std: float) -> float:
    """
    Add Gaussian noise to a coordinate value.

    Args:
        coord: Original coordinate value
        noise_std: Standard deviation of noise (in degrees)

    Returns:
        Noisy coordinate value
    """
    noise = np.random.normal(0, noise_std)
    return coord + noise


def create_training_example(
    city: Dict,
    noise_std: float = 0.001,
    include_noise: bool = True
) -> Dict:
    """
    Create a training example with optional noise.

    Args:
        city: City data dictionary
        noise_std: Standard deviation of Gaussian noise in degrees
                   (0.001 degrees ≈ 111 meters at equator)
        include_noise: Whether to add noise to coordinates

    Returns:
        Training example dictionary
    """
    lat = city['latitude']
    lon = city['longitude']

    if include_noise:
        lat_noisy = add_gaussian_noise(lat, noise_std)
        lon_noisy = add_gaussian_noise(lon, noise_std)
    else:
        lat_noisy = lat
        lon_noisy = lon

    # Format location name (include country for clarity)
    location = f"{city['name']}, {city['country_code']}"

    # Create instruction-following format
    instruction = f"What location is at coordinates ({lat_noisy:.6f}, {lon_noisy:.6f})?"
    response = location

    return {
        'instruction': instruction,
        'input': '',  # Empty for single-turn tasks
        'output': response,
        'metadata': {
            'original_lat': lat,
            'original_lon': lon,
            'noisy_lat': lat_noisy,
            'noisy_lon': lon_noisy,
            'noise_std': noise_std if include_noise else 0.0,
            'population': city['population'],
            'geonameid': city['geonameid']
        }
    }


def generate_dataset(
    cities: List[Dict],
    train_ratio: float = 0.8,
    noise_levels: List[float] = [0.001, 0.005, 0.01],
    samples_per_city: int = 3,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training and validation datasets with multiple noise levels.

    Args:
        cities: List of city data
        train_ratio: Ratio of cities to use for training
        noise_levels: List of noise standard deviations to use
        samples_per_city: Number of noisy samples per city per noise level
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Shuffle and split cities
    cities = cities.copy()
    random.shuffle(cities)

    split_idx = int(len(cities) * train_ratio)
    train_cities = cities[:split_idx]
    val_cities = cities[split_idx:]

    print(f"Train cities: {len(train_cities)}, Val cities: {len(val_cities)}")

    # Generate training data
    train_data = []
    for city in train_cities:
        # Add one clean example per city
        train_data.append(create_training_example(city, include_noise=False))

        # Add noisy examples with different noise levels
        for noise_std in noise_levels:
            for _ in range(samples_per_city):
                train_data.append(create_training_example(
                    city,
                    noise_std=noise_std,
                    include_noise=True
                ))

    # Generate validation data (fewer samples, mixed noise)
    val_data = []
    for city in val_cities:
        # Clean example
        val_data.append(create_training_example(city, include_noise=False))

        # One noisy example per noise level
        for noise_std in noise_levels:
            val_data.append(create_training_example(
                city,
                noise_std=noise_std,
                include_noise=True
            ))

    # Shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)

    return train_data, val_data


def save_dataset(data: List[Dict], output_file: str):
    """Save dataset to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} examples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate coordinate-to-location dataset with Gaussian noise"
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
        default='../data',
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--max-cities',
        type=int,
        default=10000,
        help='Maximum number of cities to use'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of cities for training (0-1)'
    )
    parser.add_argument(
        '--noise-levels',
        type=float,
        nargs='+',
        default=[0.001, 0.005, 0.01],
        help='Noise standard deviations in degrees (0.001° ≈ 111m)'
    )
    parser.add_argument(
        '--samples-per-city',
        type=int,
        default=3,
        help='Number of noisy samples per city per noise level'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cities from {args.cities_file}...")
    cities = load_cities_data(args.cities_file, max_cities=args.max_cities)
    print(f"Loaded {len(cities)} cities")

    print(f"\nGenerating dataset with noise levels: {args.noise_levels}")
    train_data, val_data = generate_dataset(
        cities=cities,
        train_ratio=args.train_ratio,
        noise_levels=args.noise_levels,
        samples_per_city=args.samples_per_city,
        seed=args.seed
    )

    # Save datasets
    train_file = output_dir / 'train_dataset.json'
    val_file = output_dir / 'val_dataset.json'

    save_dataset(train_data, str(train_file))
    save_dataset(val_data, str(val_file))

    # Print statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Total examples: {len(train_data) + len(val_data)}")

    # Sample examples
    print(f"\n=== Sample Training Examples ===")
    for i, example in enumerate(train_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Output: {example['output']}")
        print(f"Noise std: {example['metadata']['noise_std']:.6f}°")


if __name__ == '__main__':
    main()
