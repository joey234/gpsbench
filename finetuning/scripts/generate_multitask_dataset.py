"""
Multi-task dataset generation for comprehensive geospatial fine-tuning.

This script generates training data for 10 different geospatial tasks:
1. Format conversion (DMS ↔ Decimal)
2. Distance calculation (Haversine)
3. Bearing computation
4. Interpolation (midpoints)
5. Area calculation (polygons)
6. Reverse geocoding (coord → location)
7. Forward geocoding (location → coord)
8. Regional knowledge (country/continent)
9. Relative position (north/south/east/west)
10. Proximity ranking

All tasks include Gaussian noise for robustness.
"""

import json
import random
import numpy as np
import math
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from dataclasses import dataclass


@dataclass
class CityData:
    """City information from GeoNames."""
    name: str
    latitude: float
    longitude: float
    country_code: str
    country_name: str
    continent: str
    population: int


# Country code to country name mapping
COUNTRY_NAMES = {
    'US': 'United States', 'GB': 'United Kingdom', 'FR': 'France',
    'DE': 'Germany', 'IT': 'Italy', 'ES': 'Spain', 'CN': 'China',
    'JP': 'Japan', 'IN': 'India', 'BR': 'Brazil', 'RU': 'Russia',
    'CA': 'Canada', 'AU': 'Australia', 'MX': 'Mexico', 'KR': 'South Korea',
    # Add more as needed
}

# Country to continent mapping
COUNTRY_TO_CONTINENT = {
    'US': 'North America', 'CA': 'North America', 'MX': 'North America',
    'GB': 'Europe', 'FR': 'Europe', 'DE': 'Europe', 'IT': 'Europe',
    'ES': 'Europe', 'RU': 'Europe',
    'CN': 'Asia', 'JP': 'Asia', 'IN': 'Asia', 'KR': 'Asia',
    'BR': 'South America',
    'AU': 'Oceania',
    # Add more as needed
}


def load_cities(cities_file: str, max_cities: int = 10000) -> List[CityData]:
    """Load city data from GeoNames file."""
    cities = []

    with open(cities_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_cities:
                break

            parts = line.strip().split('\t')
            if len(parts) < 19:
                continue

            try:
                country_code = parts[8]
                city = CityData(
                    name=parts[1],
                    latitude=float(parts[4]),
                    longitude=float(parts[5]),
                    country_code=country_code,
                    country_name=COUNTRY_NAMES.get(country_code, country_code),
                    continent=COUNTRY_TO_CONTINENT.get(country_code, 'Unknown'),
                    population=int(parts[14]) if parts[14] else 0,
                )
                cities.append(city)
            except (ValueError, IndexError):
                continue

    return cities


def add_noise(coord: float, noise_std: float) -> float:
    """Add Gaussian noise to a coordinate."""
    if noise_std == 0:
        return coord
    return coord + np.random.normal(0, noise_std)


def decimal_to_dms(decimal: float, is_latitude: bool) -> str:
    """Convert decimal degrees to DMS format."""
    is_positive = decimal >= 0
    decimal = abs(decimal)

    degrees = int(decimal)
    minutes_decimal = (decimal - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = (minutes_decimal - minutes) * 60

    if is_latitude:
        direction = 'N' if is_positive else 'S'
    else:
        direction = 'E' if is_positive else 'W'

    return f"{degrees}°{minutes}'{seconds:.1f}\"{direction}"


def dms_to_decimal(dms: str) -> float:
    """Convert DMS format to decimal degrees (for reference/testing)."""
    # Simple parser for generated DMS strings
    import re
    match = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", dms)
    if not match:
        return 0.0

    degrees, minutes, seconds, direction = match.groups()
    decimal = int(degrees) + int(minutes)/60 + float(seconds)/3600

    if direction in ['S', 'W']:
        decimal = -decimal

    return decimal


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in kilometers."""
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing in degrees (0-360)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def bearing_to_direction(bearing: float) -> str:
    """Convert bearing to compass direction."""
    directions = ['North', 'Northeast', 'East', 'Southeast',
                  'South', 'Southwest', 'West', 'Northwest']
    index = int((bearing + 22.5) / 45) % 8
    return directions[index]


def interpolate_point(lat1: float, lon1: float, lat2: float, lon2: float, fraction: float = 0.5) -> Tuple[float, float]:
    """Interpolate point along great circle path."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    d = 2 * math.asin(math.sqrt(math.sin((lat2-lat1)/2)**2 +
                                 math.cos(lat1) * math.cos(lat2) * math.sin((lon2-lon1)/2)**2))

    if d == 0:
        return math.degrees(lat1), math.degrees(lon1)

    a = math.sin((1-fraction)*d) / math.sin(d)
    b = math.sin(fraction*d) / math.sin(d)

    x = a * math.cos(lat1) * math.cos(lon1) + b * math.cos(lat2) * math.cos(lon2)
    y = a * math.cos(lat1) * math.sin(lon1) + b * math.cos(lat2) * math.sin(lon2)
    z = a * math.sin(lat1) + b * math.sin(lat2)

    lat_result = math.atan2(z, math.sqrt(x**2 + y**2))
    lon_result = math.atan2(y, x)

    return math.degrees(lat_result), math.degrees(lon_result)


def polygon_area(coords: List[Tuple[float, float]]) -> float:
    """Calculate polygon area in km² (simple spherical approximation)."""
    if len(coords) < 3:
        return 0.0

    # Simple approximation using planar geometry with correction factor
    # For more accuracy, should use proper spherical polygon area
    R = 6371  # Earth radius in km

    # Convert to radians
    coords_rad = [(math.radians(lat), math.radians(lon)) for lat, lon in coords]

    # Spherical excess method (simplified)
    area = 0
    n = len(coords_rad)

    for i in range(n):
        lat1, lon1 = coords_rad[i]
        lat2, lon2 = coords_rad[(i+1) % n]

        area += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))

    area = abs(area * R * R / 2.0)

    return area


# Task generators

def generate_format_conversion_task(cities: List[CityData], noise_std: float = 0) -> Dict:
    """Generate format conversion task (DMS ↔ Decimal)."""
    city = random.choice(cities)

    lat = add_noise(city.latitude, noise_std)
    lon = add_noise(city.longitude, noise_std)

    # Randomly choose direction
    if random.random() < 0.5:
        # DMS to Decimal
        lat_dms = decimal_to_dms(lat, True)
        lon_dms = decimal_to_dms(lon, False)

        instruction = f"Convert this coordinate from DMS to decimal degrees: {lat_dms}, {lon_dms}"
        output = f"{lat:.6f}, {lon:.6f}"
    else:
        # Decimal to DMS
        instruction = f"Convert this coordinate from decimal degrees to DMS: {lat:.6f}, {lon:.6f}"
        output = f"{decimal_to_dms(lat, True)}, {decimal_to_dms(lon, False)}"

    return {
        'task_type': 'format_conversion',
        'instruction': instruction,
        'input': '',
        'output': output,
        'metadata': {'noise_std': noise_std, 'city': city.name}
    }


def generate_distance_task(cities: List[CityData], noise_std: float = 0) -> Dict:
    """Generate distance calculation task."""
    city1, city2 = random.sample(cities, 2)

    lat1 = add_noise(city1.latitude, noise_std)
    lon1 = add_noise(city1.longitude, noise_std)
    lat2 = add_noise(city2.latitude, noise_std)
    lon2 = add_noise(city2.longitude, noise_std)

    distance = haversine_distance(lat1, lon1, lat2, lon2)

    instruction = (
        f"Calculate the distance between these two points:\n"
        f"Point A: ({lat1:.6f}, {lon1:.6f})\n"
        f"Point B: ({lat2:.6f}, {lon2:.6f})\n"
        f"Provide the distance in kilometers."
    )

    output = f"{distance:.1f} km"

    return {
        'task_type': 'distance_calculation',
        'instruction': instruction,
        'input': '',
        'output': output,
        'metadata': {
            'noise_std': noise_std,
            'city1': city1.name,
            'city2': city2.name,
            'actual_distance': distance
        }
    }


def generate_bearing_task(cities: List[CityData], noise_std: float = 0) -> Dict:
    """Generate bearing computation task."""
    city1, city2 = random.sample(cities, 2)

    lat1 = add_noise(city1.latitude, noise_std)
    lon1 = add_noise(city1.longitude, noise_std)
    lat2 = add_noise(city2.latitude, noise_std)
    lon2 = add_noise(city2.longitude, noise_std)

    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
    direction = bearing_to_direction(bearing)

    instruction = (
        f"Calculate the initial bearing from:\n"
        f"Point A: ({lat1:.6f}, {lon1:.6f})\n"
        f"To Point B: ({lat2:.6f}, {lon2:.6f})\n"
        f"Provide the bearing in degrees (0-360)."
    )

    output = f"{bearing:.1f} degrees ({direction})"

    return {
        'task_type': 'bearing_computation',
        'instruction': instruction,
        'input': '',
        'output': output,
        'metadata': {
            'noise_std': noise_std,
            'city1': city1.name,
            'city2': city2.name
        }
    }


def generate_interpolation_task(cities: List[CityData], noise_std: float = 0) -> Dict:
    """Generate interpolation task."""
    city1, city2 = random.sample(cities, 2)

    lat1 = add_noise(city1.latitude, noise_std)
    lon1 = add_noise(city1.longitude, noise_std)
    lat2 = add_noise(city2.latitude, noise_std)
    lon2 = add_noise(city2.longitude, noise_std)

    fraction = random.choice([0.25, 0.5, 0.75])
    mid_lat, mid_lon = interpolate_point(lat1, lon1, lat2, lon2, fraction)

    fraction_text = {0.25: "quarter-point", 0.5: "midpoint", 0.75: "three-quarter point"}[fraction]

    instruction = (
        f"Find the {fraction_text} along the great circle path between:\n"
        f"Point A: ({lat1:.6f}, {lon1:.6f})\n"
        f"Point B: ({lat2:.6f}, {lon2:.6f})"
    )

    output = f"({mid_lat:.4f}, {mid_lon:.4f})"

    return {
        'task_type': 'interpolation',
        'instruction': instruction,
        'input': '',
        'output': output,
        'metadata': {
            'noise_std': noise_std,
            'fraction': fraction,
            'city1': city1.name,
            'city2': city2.name
        }
    }


def generate_area_task(cities: List[CityData], noise_std: float = 0) -> Dict:
    """Generate polygon area calculation task."""
    # Generate a triangle or quadrilateral
    n_vertices = random.choice([3, 4])
    city = random.choice(cities)

    # Create vertices around a central point
    vertices = []
    for i in range(n_vertices):
        angle = (2 * math.pi * i) / n_vertices
        offset_lat = math.cos(angle) * 0.1  # About 11km offset
        offset_lon = math.sin(angle) * 0.1

        lat = add_noise(city.latitude + offset_lat, noise_std)
        lon = add_noise(city.longitude + offset_lon, noise_std)
        vertices.append((lat, lon))

    area = polygon_area(vertices)

    vertices_str = "\n".join([f"{i+1}. ({lat:.6f}, {lon:.6f})" for i, (lat, lon) in enumerate(vertices)])

    instruction = (
        f"Calculate the area of a polygon with these vertices:\n"
        f"{vertices_str}\n"
        f"Provide the area in square kilometers."
    )

    output = f"{area:.2f} km²"

    return {
        'task_type': 'area_calculation',
        'instruction': instruction,
        'input': '',
        'output': output,
        'metadata': {
            'noise_std': noise_std,
            'n_vertices': n_vertices,
            'area': area
        }
    }


def generate_reverse_geocoding_task(cities: List[CityData], noise_std: float = 0) -> Dict:
    """Generate reverse geocoding task (coord → location)."""
    city = random.choice(cities)

    lat = add_noise(city.latitude, noise_std)
    lon = add_noise(city.longitude, noise_std)

    instruction = f"What location is at coordinates ({lat:.6f}, {lon:.6f})?"
    output = f"{city.name}, {city.country_code}"

    return {
        'task_type': 'reverse_geocoding',
        'instruction': instruction,
        'input': '',
        'output': output,
        'metadata': {
            'noise_std': noise_std,
            'city': city.name,
            'country': city.country_code
        }
    }


def generate_forward_geocoding_task(cities: List[CityData], noise_std: float = 0) -> Dict:
    """Generate forward geocoding task (location → coord)."""
    city = random.choice(cities)

    # Note: Forward geocoding doesn't use coordinate noise
    # but we can add slight variations to the answer for robustness

    instruction = f"What are the coordinates of {city.name}, {city.country_name}?"
    output = f"({city.latitude:.4f}, {city.longitude:.4f})"

    return {
        'task_type': 'forward_geocoding',
        'instruction': instruction,
        'input': '',
        'output': output,
        'metadata': {
            'noise_std': 0,  # Not applicable for forward geocoding
            'city': city.name,
            'country': city.country_code
        }
    }


def generate_regional_knowledge_task(cities: List[CityData], noise_std: float = 0) -> Dict:
    """Generate regional knowledge task (country/continent)."""
    city = random.choice(cities)

    lat = add_noise(city.latitude, noise_std)
    lon = add_noise(city.longitude, noise_std)

    # Randomly choose country or continent
    if random.random() < 0.5:
        instruction = f"Which country is this coordinate in: ({lat:.6f}, {lon:.6f})?"
        output = city.country_name
    else:
        instruction = f"What continent is this coordinate on: ({lat:.6f}, {lon:.6f})?"
        output = city.continent

    return {
        'task_type': 'regional_knowledge',
        'instruction': instruction,
        'input': '',
        'output': output,
        'metadata': {
            'noise_std': noise_std,
            'city': city.name,
            'country': city.country_code,
            'continent': city.continent
        }
    }


def generate_relative_position_task(cities: List[CityData], noise_std: float = 0) -> Dict:
    """Generate relative position task (north/south/east/west)."""
    city1, city2 = random.sample(cities, 2)

    lat1 = add_noise(city1.latitude, noise_std)
    lon1 = add_noise(city1.longitude, noise_std)
    lat2 = add_noise(city2.latitude, noise_std)
    lon2 = add_noise(city2.longitude, noise_std)

    # Choose direction to ask about
    directions = []
    if lat1 > lat2:
        directions.append(('north', 'Yes'))
    else:
        directions.append(('north', 'No'))

    if lat1 < lat2:
        directions.append(('south', 'Yes'))
    else:
        directions.append(('south', 'No'))

    if lon1 > lon2:
        directions.append(('east', 'Yes'))
    else:
        directions.append(('east', 'No'))

    if lon1 < lon2:
        directions.append(('west', 'Yes'))
    else:
        directions.append(('west', 'No'))

    direction, answer = random.choice(directions)

    instruction = (
        f"Is the location at ({lat1:.6f}, {lon1:.6f}) {direction} of "
        f"the location at ({lat2:.6f}, {lon2:.6f})?"
    )

    output = answer

    return {
        'task_type': 'relative_position',
        'instruction': instruction,
        'input': '',
        'output': output,
        'metadata': {
            'noise_std': noise_std,
            'city1': city1.name,
            'city2': city2.name,
            'direction': direction
        }
    }


def generate_proximity_task(cities: List[CityData], noise_std: float = 0) -> Dict:
    """Generate proximity ranking task."""
    # Pick one reference city and 3 candidates
    ref_city, *candidates = random.sample(cities, 4)

    ref_lat = add_noise(ref_city.latitude, noise_std)
    ref_lon = add_noise(ref_city.longitude, noise_std)

    # Calculate distances
    distances = []
    for city in candidates:
        lat = add_noise(city.latitude, noise_std)
        lon = add_noise(city.longitude, noise_std)
        dist = haversine_distance(ref_lat, ref_lon, lat, lon)
        distances.append((city, dist, lat, lon))

    # Sort by distance
    distances.sort(key=lambda x: x[1])
    closest_city, closest_dist, _, _ = distances[0]

    candidates_str = ", ".join([
        f"{city.name} ({lat:.4f}, {lon:.4f})"
        for city, _, lat, lon in distances
    ])

    instruction = (
        f"Which city is closest to ({ref_lat:.6f}, {ref_lon:.6f})?\n"
        f"Options: {candidates_str}"
    )

    output = f"{closest_city.name} ({closest_dist:.1f} km)"

    return {
        'task_type': 'proximity',
        'instruction': instruction,
        'input': '',
        'output': output,
        'metadata': {
            'noise_std': noise_std,
            'reference': ref_city.name,
            'closest': closest_city.name
        }
    }


# Main dataset generation

TASK_GENERATORS = {
    'format_conversion': generate_format_conversion_task,
    'distance_calculation': generate_distance_task,
    'bearing_computation': generate_bearing_task,
    'interpolation': generate_interpolation_task,
    'area_calculation': generate_area_task,
    'reverse_geocoding': generate_reverse_geocoding_task,
    'forward_geocoding': generate_forward_geocoding_task,
    'regional_knowledge': generate_regional_knowledge_task,
    'relative_position': generate_relative_position_task,
    'proximity': generate_proximity_task,
}


def generate_multitask_dataset(
    cities: List[CityData],
    samples_per_task: int = 1000,
    noise_levels: List[float] = [0, 0.001, 0.005, 0.01],
    seed: int = 42
) -> List[Dict]:
    """
    Generate multi-task dataset.

    Args:
        cities: List of city data
        samples_per_task: Number of samples per task type
        noise_levels: List of noise standard deviations
        seed: Random seed

    Returns:
        List of training examples
    """
    random.seed(seed)
    np.random.seed(seed)

    dataset = []

    for task_name, generator in TASK_GENERATORS.items():
        print(f"Generating {task_name}...")

        for _ in range(samples_per_task):
            # Sample noise level
            noise_std = random.choice(noise_levels)

            # Generate example
            example = generator(cities, noise_std)
            dataset.append(example)

    # Shuffle
    random.shuffle(dataset)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-task geospatial dataset"
    )

    parser.add_argument(
        '--cities-file',
        type=str,
        default='../../data/raw/cities15000.txt',
        help='Path to cities15000.txt'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data',
        help='Output directory'
    )
    parser.add_argument(
        '--max-cities',
        type=int,
        default=10000,
        help='Maximum cities to load'
    )
    parser.add_argument(
        '--samples-per-task',
        type=int,
        default=1000,
        help='Samples per task type'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training split ratio'
    )
    parser.add_argument(
        '--noise-levels',
        type=float,
        nargs='+',
        default=[0, 0.001, 0.005, 0.01],
        help='Noise standard deviations'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Load cities
    print(f"Loading cities from {args.cities_file}...")
    cities = load_cities(args.cities_file, args.max_cities)
    print(f"Loaded {len(cities)} cities")

    # Generate dataset
    print(f"\nGenerating multi-task dataset...")
    print(f"Tasks: {list(TASK_GENERATORS.keys())}")
    print(f"Samples per task: {args.samples_per_task}")
    print(f"Noise levels: {args.noise_levels}")

    dataset = generate_multitask_dataset(
        cities,
        samples_per_task=args.samples_per_task,
        noise_levels=args.noise_levels,
        seed=args.seed
    )

    print(f"\nGenerated {len(dataset)} total examples")

    # Split train/val
    split_idx = int(len(dataset) * args.train_ratio)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / 'train_multitask_dataset.json'
    val_file = output_dir / 'val_multitask_dataset.json'

    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved {len(train_data)} training examples to {train_file}")

    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Saved {len(val_data)} validation examples to {val_file}")

    # Print statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Total examples: {len(dataset)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    print(f"\n=== Task Distribution ===")
    task_counts = {}
    for example in train_data:
        task_type = example['task_type']
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    for task, count in sorted(task_counts.items()):
        print(f"{task}: {count}")

    print(f"\n=== Sample Examples ===")
    for task_type in list(TASK_GENERATORS.keys())[:3]:
        examples = [ex for ex in train_data if ex['task_type'] == task_type]
        if examples:
            ex = examples[0]
            print(f"\n{task_type}:")
            print(f"Instruction: {ex['instruction']}")
            print(f"Output: {ex['output']}")


if __name__ == '__main__':
    main()
