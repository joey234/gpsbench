#!/usr/bin/env python3
"""
Augment hierarchical_spatial.json with GPS coordinates for each city pair.
"""

import json
import time
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


def geocode_city(geolocator, city_name, cache, retries=2):
    """Geocode a city name to lat/lon."""
    cache_key = city_name.lower().strip()

    if cache_key in cache:
        return cache[cache_key]

    for attempt in range(retries):
        try:
            time.sleep(1.1)  # Rate limit
            location = geolocator.geocode(city_name, timeout=10)
            if location:
                result = {'lat': location.latitude, 'lon': location.longitude}
                cache[cache_key] = result
                return result
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"  Attempt {attempt+1} failed for '{city_name}': {e}")
            if attempt < retries - 1:
                time.sleep(2)

    cache[cache_key] = None
    return None


def main():
    data_dir = Path(__file__).parent

    # Load dataset
    with open(data_dir / "hierarchical_spatial.json") as f:
        data = json.load(f)

    print(f"Loaded {len(data['questions'])} questions")

    # Initialize geocoder
    geolocator = Nominatim(user_agent="hierarchical_spatial_gps_augmentation")
    cache = {}

    # Geocode all cities
    augmented_questions = []

    for q in data['questions']:
        print(f"Processing Q{q['id']}: {q['city_a']} -> {q['city_b']}")

        # Geocode both cities
        coords_a = geocode_city(geolocator, q['city_a'], cache)
        coords_b = geocode_city(geolocator, q['city_b'], cache)

        # Create augmented question
        aug_q = {**q}
        aug_q['city_a_coords'] = coords_a
        aug_q['city_b_coords'] = coords_b

        if coords_a and coords_b:
            aug_q['gps_augmentation'] = (
                f"{q['city_a']} is located at coordinates ({coords_a['lat']:.4f}°N, {coords_a['lon']:.4f}°W). "
                f"{q['city_b']} is located at coordinates ({coords_b['lat']:.4f}°N, {coords_b['lon']:.4f}°W)."
            )
            print(f"  ✓ Both cities geocoded")
        else:
            aug_q['gps_augmentation'] = None
            print(f"  ✗ Failed to geocode one or both cities")

        augmented_questions.append(aug_q)

    # Save augmented dataset
    augmented_data = {
        'metadata': data['metadata'],
        'benchmark_info': data['benchmark_info'],
        'questions': augmented_questions
    }

    output_file = data_dir / "hierarchical_spatial_gps.json"
    with open(output_file, 'w') as f:
        json.dump(augmented_data, f, indent=2)

    print(f"\nSaved augmented dataset to {output_file}")

    # Stats
    geocoded = sum(1 for q in augmented_questions if q['gps_augmentation'])
    print(f"Successfully augmented: {geocoded}/{len(augmented_questions)} questions")


if __name__ == "__main__":
    main()
