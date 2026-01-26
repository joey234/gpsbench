#!/usr/bin/env python3
"""
Improved MapEval GPS augmentation with better geocoding.

Key improvements:
1. Extract city/country context from question to improve geocoding
2. Skip street names and route numbers (not useful for geocoding)
3. Validate geocoded coordinates are in expected region
4. Better GPS augmentation format
"""

import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datasets import load_dataset
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.distance import geodesic
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "geocode_cache_v2"
CACHE_DIR.mkdir(exist_ok=True)

# Patterns to skip (street names, route numbers, directions)
SKIP_PATTERNS = [
    r'^(I-\d+|US-\d+|SR-\d+|Route \d+)',  # Highway/route numbers
    r'^(N|S|E|W|North|South|East|West)\s+\w+\s+(St|Ave|Rd|Blvd|Dr|Way|Ln)',  # Directional streets
    r'^(Main|High|First|Second|Third|1st|2nd|3rd)\s+(St|Street|Ave|Avenue)',  # Generic streets
    r'^(left|right|straight|turn)',  # Directions
    r'^\d+\s*(mins?|minutes?|hours?|km|miles?)',  # Time/distance
]

# Common location suffixes to help identify cities
CITY_INDICATORS = ['city', 'town', 'village', 'district', 'province', 'state', 'country']


def load_geocode_cache() -> Dict[str, dict]:
    cache_file = CACHE_DIR / "geocode_cache_v2.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return {}


def save_geocode_cache(cache: Dict[str, dict]):
    cache_file = CACHE_DIR / "geocode_cache_v2.json"
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)


def should_skip_place(name: str) -> bool:
    """Check if place name should be skipped (streets, routes, etc.)."""
    name_lower = name.lower().strip()

    for pattern in SKIP_PATTERNS:
        if re.match(pattern, name_lower, re.IGNORECASE):
            return True

    # Skip very short names or pure numbers
    if len(name_lower) < 3 or name_lower.isdigit():
        return True

    return False


def extract_location_context(question: str, context: str) -> Optional[str]:
    """Extract city/country context from question or context."""
    # Common patterns: "in [City], [Country]" or "in [City]"
    patterns = [
        r'in\s+([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)',  # "in City, Country"
        r'in\s+([A-Z][a-zA-Z\s]+,\s*[A-Z]{2})',  # "in City, ST"
        r'in\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)',  # "in City" or "in City Name"
        r'from\s+([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)',  # "from City, Country"
        r'at\s+([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)',  # "at City, Country"
    ]

    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip()

    # Try to extract from context (look for addresses)
    address_match = re.search(r',\s*([A-Z][a-zA-Z\s]+)\s+\d{5}', context)  # "City 12345"
    if address_match:
        return address_match.group(1).strip()

    # Look for country names
    country_match = re.search(r',\s*(Peru|Mexico|USA|Canada|UK|France|Italy|Spain|Germany|Greece|Japan|China|India|Australia)', context, re.IGNORECASE)
    if country_match:
        return country_match.group(1)

    return None


def geocode_with_context(
    geolocator: Nominatim,
    name: str,
    address: Optional[str],
    location_context: Optional[str],
    cache: Dict[str, dict],
    retries: int = 2
) -> Tuple[Optional[float], Optional[float], str]:
    """Geocode with location context for better accuracy."""

    # Build query with context
    queries_to_try = []

    if address:
        queries_to_try.append(address)

    if location_context:
        queries_to_try.append(f"{name}, {location_context}")

    queries_to_try.append(name)

    for query in queries_to_try:
        cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()

        if cache_key in cache:
            cached = cache[cache_key]
            if cached.get('lat') is not None:
                return cached['lat'], cached['lon'], 'cached'
            continue  # Skip failed cache entries

        for attempt in range(retries):
            try:
                time.sleep(1.1)
                location = geolocator.geocode(query, timeout=10)
                if location:
                    cache[cache_key] = {'lat': location.latitude, 'lon': location.longitude, 'query': query}
                    return location.latitude, location.longitude, query
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                if attempt < retries - 1:
                    time.sleep(2)

        # Cache the failure
        cache[cache_key] = {'lat': None, 'lon': None, 'query': query}

    return None, None, 'failed'


def validate_coordinates(
    places: List[dict],
    location_context: Optional[str]
) -> List[dict]:
    """Filter out coordinates that are likely wrong (outliers)."""
    if len(places) < 3:
        return places

    # Get all valid coordinates
    valid_coords = [(p['lat'], p['lon']) for p in places if p['lat'] is not None]
    if len(valid_coords) < 3:
        return places

    # Calculate centroid
    avg_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
    avg_lon = sum(c[1] for c in valid_coords) / len(valid_coords)

    # Calculate median distance to centroid
    distances = []
    for lat, lon in valid_coords:
        try:
            dist = geodesic((lat, lon), (avg_lat, avg_lon)).km
            distances.append(dist)
        except:
            distances.append(0)

    if not distances:
        return places

    median_dist = sorted(distances)[len(distances) // 2]
    threshold = max(100, median_dist * 5)  # At least 100km, or 5x median

    # Filter out outliers
    validated = []
    for p in places:
        if p['lat'] is None:
            validated.append(p)
            continue

        try:
            dist = geodesic((p['lat'], p['lon']), (avg_lat, avg_lon)).km
            if dist <= threshold:
                validated.append(p)
            else:
                # Mark as invalid
                validated.append({**p, 'lat': None, 'lon': None, 'filtered': 'outlier'})
                logger.debug(f"Filtered outlier: {p['name']} ({dist:.0f}km from centroid)")
        except:
            validated.append(p)

    return validated


def extract_places_with_addresses(context: str) -> Dict[str, Optional[str]]:
    """Extract place names and addresses from context."""
    places = {}

    # Pattern 1: "Information of <b>Place</b>:\n- Location: Address"
    pattern1 = r'Information of <b>([^<]+)</b>:\s*\n- Location:\s*([^\n]+)'
    for match in re.finditer(pattern1, context):
        name = match.group(1).strip()
        address = match.group(2).strip().rstrip('.')
        if not should_skip_place(name):
            places[name] = address

    # Pattern 2: Numbered list with addresses
    pattern2 = r'\d+\.\s*<b>([^<]+)</b>\s*\n\s*-\s*Address:\s*([^\n]+)'
    for match in re.finditer(pattern2, context):
        name = match.group(1).strip()
        address = match.group(2).strip().rstrip('.')
        if name not in places and not should_skip_place(name):
            places[name] = address

    # Pattern 3: Remaining <b> tags (POIs without addresses)
    all_bold = re.findall(r'<b>([^<]+)</b>', context)
    for name in all_bold:
        name = name.strip()
        if name and name not in places and not should_skip_place(name):
            places[name] = None

    return places


def format_gps_augmentation(places: List[dict], location_context: Optional[str]) -> str:
    """Format GPS coordinates in a more useful way."""
    geocoded = [p for p in places if p.get('lat') is not None]

    if not geocoded:
        return ""

    lines = ["\n\n[GPS Reference Coordinates]"]
    if location_context:
        lines.append(f"Location: {location_context}")

    for p in geocoded:
        lat, lon = p['lat'], p['lon']
        # Format with N/S and E/W
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        lines.append(f"• {p['name']}: {abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}")

    return "\n".join(lines)


def main():
    output_dir = Path(__file__).parent

    logger.info("Loading MapEval-Textual dataset...")
    ds = load_dataset("MapEval/MapEval-Textual", "benchmark", split="test")
    logger.info(f"Loaded {len(ds)} samples")

    geolocator = Nominatim(user_agent="mapeval_gps_v2")
    cache = load_geocode_cache()
    logger.info(f"Loaded {len(cache)} cached results")

    # Phase 1: Extract all places with context
    logger.info("Phase 1: Extracting places with location context...")
    sample_data = []

    for sample in ds:
        places_dict = extract_places_with_addresses(sample['context'])
        location_context = extract_location_context(sample['question'], sample['context'])

        sample_data.append({
            'sample': sample,
            'places_dict': places_dict,
            'location_context': location_context
        })

    # Phase 2: Geocode unique places
    logger.info("Phase 2: Geocoding places...")

    # Collect unique (place, context) pairs
    unique_queries = {}  # (name, context) -> (address, location_context)
    for sd in sample_data:
        ctx = sd['location_context']
        for name, address in sd['places_dict'].items():
            key = (name, ctx)
            if key not in unique_queries:
                unique_queries[key] = (address, ctx)

    logger.info(f"Found {len(unique_queries)} unique place-context pairs")

    # Geocode
    place_coords = {}  # (name, context) -> (lat, lon)
    for i, ((name, ctx), (address, location_context)) in enumerate(unique_queries.items()):
        lat, lon, source = geocode_with_context(geolocator, name, address, location_context, cache)
        place_coords[(name, ctx)] = (lat, lon)

        if (i + 1) % 50 == 0:
            save_geocode_cache(cache)
            geocoded = sum(1 for c in place_coords.values() if c[0] is not None)
            logger.info(f"Progress: {i+1}/{len(unique_queries)}, {geocoded} geocoded")

    save_geocode_cache(cache)

    # Phase 3: Build augmented dataset
    logger.info("Phase 3: Building augmented dataset...")

    augmented_samples = []
    total_places = 0
    total_geocoded = 0

    for sd in sample_data:
        sample = sd['sample']
        places_dict = sd['places_dict']
        location_context = sd['location_context']

        # Build places list with coordinates
        places = []
        for name, address in places_dict.items():
            lat, lon = place_coords.get((name, location_context), (None, None))
            places.append({
                'name': name,
                'address': address,
                'lat': lat,
                'lon': lon
            })

        # Validate coordinates (filter outliers)
        places = validate_coordinates(places, location_context)

        # Create augmented context
        gps_text = format_gps_augmentation(places, location_context)
        augmented_context = sample['context'] + gps_text

        num_places = len(places)
        num_geocoded = sum(1 for p in places if p.get('lat') is not None)

        augmented = {
            'id': sample['id'],
            'context': sample['context'],
            'context_augmented': augmented_context,
            'question': sample['question'],
            'options': sample['options'],
            'answer': sample['answer'],
            'classification': sample['classification'],
            'location_context': location_context,
            'places': places,
            'num_places': num_places,
            'num_geocoded': num_geocoded
        }
        augmented_samples.append(augmented)

        total_places += num_places
        total_geocoded += num_geocoded

    # Save
    output_file = output_dir / "mapeval_gps_augmented_v2.json"
    with open(output_file, 'w') as f:
        json.dump(augmented_samples, f, indent=2)
    logger.info(f"Saved to {output_file}")

    # Stats
    logger.info(f"\n{'='*50}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total samples: {len(augmented_samples)}")
    logger.info(f"Total places: {total_places}")
    logger.info(f"Geocoded: {total_geocoded} ({100*total_geocoded/total_places:.1f}%)")

    # Save simple format
    simple = []
    for s in augmented_samples:
        simple.append({
            'id': s['id'],
            'context': s['context_augmented'],
            'question': s['question'],
            'options': s['options'],
            'answer': s['answer'],
            'classification': s['classification']
        })

    simple_file = output_dir / "mapeval_test_gps_v2.json"
    with open(simple_file, 'w') as f:
        json.dump(simple, f, indent=2)
    logger.info(f"Saved simple format to {simple_file}")


if __name__ == "__main__":
    main()
