#!/usr/bin/env python3
"""
Augment MapEval-Textual dataset with GPS coordinates (Optimized version).

This script:
1. Loads the MapEval-Textual test set from HuggingFace
2. First pass: Extract all unique places from all samples
3. Geocode each unique place once (with caching)
4. Second pass: Apply coordinates to all samples
"""

import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datasets import load_dataset
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache directory for geocoding results
CACHE_DIR = Path(__file__).parent / "geocode_cache"
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_key(query: str) -> str:
    """Generate a cache key for a geocoding query."""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def load_geocode_cache() -> Dict[str, dict]:
    """Load the geocoding cache from disk."""
    cache_file = CACHE_DIR / "geocode_cache.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return {}


def save_geocode_cache(cache: Dict[str, dict]):
    """Save the geocoding cache to disk."""
    cache_file = CACHE_DIR / "geocode_cache.json"
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)


def extract_places_with_addresses(context: str) -> Dict[str, Optional[str]]:
    """
    Extract place names and their addresses from MapEval context.

    Returns dict of {place_name: address_or_None}
    """
    places = {}

    # Pattern 1: "Information of <b>Place</b>:\n- Location: Address"
    pattern1 = r'Information of <b>([^<]+)</b>:\s*\n- Location:\s*([^\n]+)'
    for match in re.finditer(pattern1, context):
        name = match.group(1).strip()
        address = match.group(2).strip().rstrip('.')
        places[name] = address

    # Pattern 2: Numbered list items with addresses
    pattern2 = r'\d+\.\s*<b>([^<]+)</b>\s*\n\s*-\s*Address:\s*([^\n]+)'
    for match in re.finditer(pattern2, context):
        name = match.group(1).strip()
        address = match.group(2).strip().rstrip('.')
        if name not in places:
            places[name] = address

    # Pattern 3: Any remaining <b> tagged names (skip non-places)
    skip_patterns = {'left', 'right', '1st', '2nd', '3rd', '4th', 'north', 'south', 'east', 'west'}
    all_bold = re.findall(r'<b>([^<]+)</b>', context)
    for name in all_bold:
        name = name.strip()
        if name and name not in places and name.lower() not in skip_patterns:
            # Skip if it looks like a direction or HTML artifact
            if len(name) > 2 and not name[0].isdigit():
                places[name] = None

    return places


def geocode_query(
    geolocator: Nominatim,
    query: str,
    cache: Dict[str, dict],
    retries: int = 2
) -> Tuple[Optional[float], Optional[float]]:
    """Geocode a single query with caching and rate limiting."""
    cache_key = get_cache_key(query)

    if cache_key in cache:
        cached = cache[cache_key]
        return cached.get('lat'), cached.get('lon')

    for attempt in range(retries):
        try:
            time.sleep(1.1)  # Nominatim rate limit
            location = geolocator.geocode(query, timeout=10)
            if location:
                cache[cache_key] = {'lat': location.latitude, 'lon': location.longitude, 'query': query}
                return location.latitude, location.longitude
            else:
                cache[cache_key] = {'lat': None, 'lon': None, 'query': query}
                return None, None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.warning(f"Attempt {attempt+1} failed for '{query}': {e}")
            if attempt < retries - 1:
                time.sleep(2)

    cache[cache_key] = {'lat': None, 'lon': None, 'query': query}
    return None, None


def format_gps_augmentation(places_with_coords: List[Tuple[str, float, float]]) -> str:
    """Format GPS coordinates as text to append to context."""
    if not places_with_coords:
        return "\n\n--- GPS Coordinates ---\nNo coordinates available."

    lines = ["\n\n--- GPS Coordinates ---"]
    for name, lat, lon in places_with_coords:
        lines.append(f"• {name}: ({lat:.6f}, {lon:.6f})")

    return "\n".join(lines)


def main():
    output_dir = Path(__file__).parent

    logger.info("Loading MapEval-Textual dataset...")
    ds = load_dataset("MapEval/MapEval-Textual", "benchmark", split="test")
    logger.info(f"Loaded {len(ds)} samples")

    # Initialize geocoder
    geolocator = Nominatim(user_agent="mapeval_gps_augmentation_research_v2")

    # Load cache
    cache = load_geocode_cache()
    logger.info(f"Loaded {len(cache)} cached geocoding results")

    # ============================================
    # PHASE 1: Extract all unique places
    # ============================================
    logger.info("Phase 1: Extracting all unique places...")

    # Store places per sample for Phase 2
    sample_places = []  # List of dicts {place_name: address}

    # Collect all unique (place, address) pairs
    unique_places = {}  # place_name -> best_address (prefer non-None)

    for sample in ds:
        places = extract_places_with_addresses(sample['context'])
        sample_places.append(places)

        for name, address in places.items():
            if name not in unique_places or (address and not unique_places[name]):
                unique_places[name] = address

    logger.info(f"Found {len(unique_places)} unique places across all samples")

    # ============================================
    # PHASE 2: Geocode all unique places
    # ============================================
    logger.info("Phase 2: Geocoding unique places...")

    # Count how many need geocoding (not in cache)
    to_geocode = []
    for name, address in unique_places.items():
        query = address if address else name
        if get_cache_key(query) not in cache:
            to_geocode.append((name, address))

    logger.info(f"Need to geocode {len(to_geocode)} places ({len(unique_places) - len(to_geocode)} cached)")

    # Geocode with progress
    place_coords = {}  # place_name -> (lat, lon)

    for i, (name, address) in enumerate(unique_places.items()):
        # Try address first, then name
        query = address if address else name
        lat, lon = geocode_query(geolocator, query, cache)

        # If address failed, try name only
        if lat is None and address:
            lat, lon = geocode_query(geolocator, name, cache)

        place_coords[name] = (lat, lon)

        if (i + 1) % 50 == 0:
            save_geocode_cache(cache)
            geocoded_count = sum(1 for c in place_coords.values() if c[0] is not None)
            logger.info(f"Progress: {i+1}/{len(unique_places)} places, {geocoded_count} geocoded, cache saved")

    # Final cache save
    save_geocode_cache(cache)
    geocoded_count = sum(1 for c in place_coords.values() if c[0] is not None)
    logger.info(f"Geocoding complete: {geocoded_count}/{len(unique_places)} places resolved")

    # ============================================
    # PHASE 3: Create augmented dataset
    # ============================================
    logger.info("Phase 3: Creating augmented dataset...")

    augmented_samples = []

    for i, sample in enumerate(ds):
        places = sample_places[i]

        # Get coordinates for this sample's places
        places_with_coords = []
        for name in places.keys():
            lat, lon = place_coords.get(name, (None, None))
            if lat is not None:
                places_with_coords.append((name, lat, lon))

        # Create augmented context
        gps_text = format_gps_augmentation(places_with_coords)
        augmented_context = sample['context'] + gps_text

        augmented = {
            'id': sample['id'],
            'context': sample['context'],
            'context_augmented': augmented_context,
            'question': sample['question'],
            'options': sample['options'],
            'answer': sample['answer'],
            'classification': sample['classification'],
            'places': [
                {'name': name, 'address': addr, 'lat': place_coords[name][0], 'lon': place_coords[name][1]}
                for name, addr in places.items()
            ],
            'num_places': len(places),
            'num_geocoded': len(places_with_coords)
        }
        augmented_samples.append(augmented)

    # Save full augmented dataset
    output_file = output_dir / "mapeval_gps_augmented.json"
    with open(output_file, 'w') as f:
        json.dump(augmented_samples, f, indent=2)
    logger.info(f"Saved augmented dataset to {output_file}")

    # Save simple format (ready for evaluation)
    simple_format = []
    for s in augmented_samples:
        simple_format.append({
            'id': s['id'],
            'context': s['context_augmented'],
            'question': s['question'],
            'options': s['options'],
            'answer': s['answer'],
            'classification': s['classification']
        })

    simple_file = output_dir / "mapeval_test_gps.json"
    with open(simple_file, 'w') as f:
        json.dump(simple_format, f, indent=2)
    logger.info(f"Saved evaluation format to {simple_file}")

    # Save statistics
    total_places = sum(s['num_places'] for s in augmented_samples)
    total_geocoded = sum(s['num_geocoded'] for s in augmented_samples)

    stats = {
        'total_samples': len(augmented_samples),
        'unique_places': len(unique_places),
        'unique_places_geocoded': geocoded_count,
        'total_place_mentions': total_places,
        'total_geocoded_mentions': total_geocoded,
        'geocode_success_rate': geocoded_count / len(unique_places) if unique_places else 0,
        'cache_size': len(cache),
        'by_classification': {}
    }

    # Stats by classification
    for classification in ['trip', 'routing', 'nearby', 'poi', 'unanswerable']:
        class_samples = [s for s in augmented_samples if s['classification'] == classification]
        if class_samples:
            stats['by_classification'][classification] = {
                'count': len(class_samples),
                'avg_places': sum(s['num_places'] for s in class_samples) / len(class_samples),
                'avg_geocoded': sum(s['num_geocoded'] for s in class_samples) / len(class_samples)
            }

    stats_file = output_dir / "augmentation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total samples: {len(augmented_samples)}")
    logger.info(f"Unique places: {len(unique_places)}")
    logger.info(f"Successfully geocoded: {geocoded_count} ({100*geocoded_count/len(unique_places):.1f}%)")
    logger.info(f"Output files:")
    logger.info(f"  - {output_file}")
    logger.info(f"  - {simple_file}")
    logger.info(f"  - {stats_file}")


if __name__ == "__main__":
    main()
