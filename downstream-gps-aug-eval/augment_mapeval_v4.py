#!/usr/bin/env python3
"""
MapEval GPS augmentation v4 - FAST version.

Optimizations:
1. Extract coordinates directly from address when available (no API call needed)
2. Reuse v2 cache for successful geocodes
3. Single query attempt with country context
4. Parallel-friendly with async (future improvement)
"""

import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.distance import geodesic
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "geocode_cache_v4"
CACHE_DIR.mkdir(exist_ok=True)

KNOWN_COUNTRIES = [
    'Peru', 'USA', 'Australia', 'Saudi Arabia', 'Greece',
    'UK', 'United Kingdom', 'France', 'Italy', 'Spain',
    'Germany', 'Japan', 'Mexico', 'Canada', 'Brazil',
    'India', 'China', 'Thailand', 'Turkey', 'Egypt',
    'Portugal', 'Netherlands', 'Belgium', 'Austria',
    'Switzerland', 'Sweden', 'Norway', 'Denmark', 'Finland',
    'Poland', 'Czech Republic', 'Hungary', 'Russia',
    'South Korea', 'Singapore', 'Malaysia', 'Indonesia',
    'Philippines', 'Vietnam', 'New Zealand', 'Ireland',
    'Argentina', 'Chile', 'Colombia', 'Ecuador', 'Bolivia',
    'Morocco', 'South Africa', 'Kenya', 'Nigeria',
    'United Arab Emirates', 'Qatar', 'Kuwait', 'Bahrain',
    'Israel', 'Jordan', 'Lebanon', 'Cyprus', 'Malta',
    'Iceland', 'Luxembourg', 'Monaco', 'Croatia', 'Slovenia',
    'Slovakia', 'Romania', 'Bulgaria', 'Serbia', 'Ukraine',
    'Taiwan', 'Hong Kong', 'Macau', 'Sri Lanka', 'Nepal',
    'Bangladesh', 'Pakistan', 'Myanmar', 'Cambodia', 'Laos',
]

SKIP_PATTERNS = [
    r'^(I-\d+|US-\d+|SR-\d+|Route \d+)',
    r'^(N|S|E|W|North|South|East|West)\s+\w+\s+(St|Ave|Rd|Blvd|Dr|Way|Ln)',
    r'^(Main|High|First|Second|Third|1st|2nd|3rd)\s+(St|Street|Ave|Avenue)',
    r'^(left|right|straight|turn)',
    r'^\d+\s*(mins?|minutes?|hours?|km|miles?)',
]


def load_all_caches() -> Dict[str, dict]:
    """Load v4 cache + import successful results from v2 cache."""
    cache = {}

    # Load v4 cache
    cache_file = CACHE_DIR / "geocode_cache_v4.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        logger.info(f"Loaded {len(cache)} entries from v4 cache")

    # Import successful geocodes from v2 cache (but verify country)
    v2_cache_file = Path(__file__).parent / "geocode_cache" / "geocode_cache.json"
    if v2_cache_file.exists():
        with open(v2_cache_file) as f:
            v2_cache = json.load(f)

        imported = 0
        for key, value in v2_cache.items():
            if key not in cache and value.get('lat') is not None:
                # Only import if query included country context
                query = value.get('query', '').lower()
                has_country = any(c.lower() in query for c in KNOWN_COUNTRIES)
                if has_country:
                    cache[key] = value
                    imported += 1

        if imported > 0:
            logger.info(f"Imported {imported} entries from v2 cache")

    return cache


def save_cache(cache: Dict[str, dict]):
    cache_file = CACHE_DIR / "geocode_cache_v4.json"
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)


def should_skip_place(name: str) -> bool:
    name_lower = name.lower().strip()
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, name_lower, re.IGNORECASE):
            return True
    if len(name_lower) < 3 or name_lower.isdigit():
        return True
    return False


def extract_coords_from_address(address: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract coordinates directly from address if present (e.g., 'Place (lat, lon)')."""
    if not address:
        return None, None

    # Pattern: (lat, lon) or (lat,lon)
    match = re.search(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', address)
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            # Validate reasonable coordinates
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass

    return None, None


def extract_country_from_address(address: str) -> Optional[str]:
    if not address:
        return None
    address_lower = address.lower()
    for country in KNOWN_COUNTRIES:
        if country.lower() in address_lower:
            return country
    return None


def extract_country_from_context(context: str) -> Optional[str]:
    """Extract country from entire sample context."""
    if not context:
        return None
    context_lower = context.lower()
    for country in KNOWN_COUNTRIES:
        if country.lower() in context_lower:
            return country
    return None


def geocode_place(
    geolocator: Nominatim,
    name: str,
    address: Optional[str],
    sample_country: Optional[str],  # NEW: country from sample context
    cache: Dict[str, dict]
) -> Tuple[Optional[float], Optional[float], str]:
    """Geocode with minimal API calls."""

    # 1. Try to extract coordinates directly from address (FREE - no API call!)
    lat, lon = extract_coords_from_address(address)
    if lat is not None:
        return lat, lon, 'extracted_from_address'

    # 2. Get country context - prefer from address, fallback to sample context
    country = extract_country_from_address(address)
    if not country:
        country = sample_country
    if not country:
        return None, None, 'no_country_context'

    # 3. Build single best query: name + country
    query = f"{name}, {country}"
    cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()

    # 4. Check cache
    if cache_key in cache:
        cached = cache[cache_key]
        if cached.get('lat') is not None:
            return cached['lat'], cached['lon'], 'cached'
        return None, None, 'cached_failure'

    # 5. Make API call (only if not cached)
    try:
        time.sleep(1.1)  # Nominatim rate limit
        location = geolocator.geocode(query, timeout=10)
        if location:
            cache[cache_key] = {'lat': location.latitude, 'lon': location.longitude, 'query': query}
            return location.latitude, location.longitude, query
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.warning(f"Geocoding failed for '{query}': {e}")

    cache[cache_key] = {'lat': None, 'lon': None, 'query': query}
    return None, None, 'api_failed'


def validate_coordinates_median(places: List[dict]) -> List[dict]:
    """Filter outliers using MEDIAN centroid."""
    if len(places) < 3:
        return places

    valid_coords = [(p['lat'], p['lon']) for p in places if p['lat'] is not None]
    if len(valid_coords) < 3:
        return places

    sorted_lats = sorted(c[0] for c in valid_coords)
    sorted_lons = sorted(c[1] for c in valid_coords)
    median_lat = sorted_lats[len(sorted_lats) // 2]
    median_lon = sorted_lons[len(sorted_lons) // 2]

    distances = []
    for lat, lon in valid_coords:
        try:
            dist = geodesic((lat, lon), (median_lat, median_lon)).km
            distances.append(dist)
        except:
            distances.append(0)

    if not distances:
        return places

    sorted_dists = sorted(distances)
    median_dist = sorted_dists[len(sorted_dists) // 2]
    threshold = max(50, median_dist * 3)

    validated = []
    for p in places:
        if p['lat'] is None:
            validated.append(p)
            continue

        try:
            dist = geodesic((p['lat'], p['lon']), (median_lat, median_lon)).km
            if dist <= threshold:
                validated.append(p)
            else:
                validated.append({**p, 'lat': None, 'lon': None, 'filtered': f'outlier ({dist:.0f}km)'})
                logger.info(f"Filtered outlier: {p['name']} ({dist:.0f}km)")
        except:
            validated.append(p)

    return validated


def extract_places_with_addresses(context: str) -> Dict[str, Optional[str]]:
    """Extract place names and addresses from context."""
    places = {}

    pattern1 = r'Information of <b>([^<]+)</b>:\s*\n- Location:\s*([^\n]+)'
    for match in re.finditer(pattern1, context):
        name = match.group(1).strip()
        address = match.group(2).strip().rstrip('.')
        if not should_skip_place(name):
            places[name] = address

    pattern2 = r'\d+\.\s*<b>([^<]+)</b>\s*\n\s*-\s*Address:\s*([^\n]+)'
    for match in re.finditer(pattern2, context):
        name = match.group(1).strip()
        address = match.group(2).strip().rstrip('.')
        if name not in places and not should_skip_place(name):
            places[name] = address

    pattern3 = r'<b>([^<]+)</b>:\s*([^<\n]+)'
    for match in re.finditer(pattern3, context):
        name = match.group(1).strip()
        address = match.group(2).strip().rstrip('.')
        if name not in places and not should_skip_place(name):
            places[name] = address

    all_bold = re.findall(r'<b>([^<]+)</b>', context)
    for name in all_bold:
        name = name.strip()
        if name and name not in places and not should_skip_place(name):
            places[name] = None

    return places


def format_gps_augmentation(places: List[dict]) -> str:
    geocoded = [p for p in places if p.get('lat') is not None]
    if not geocoded:
        return ""

    lines = ["\n\n[GPS Reference Coordinates]"]
    for p in geocoded:
        lat, lon = p['lat'], p['lon']
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        lines.append(f"* {p['name']}: {abs(lat):.4f} deg {lat_dir}, {abs(lon):.4f} deg {lon_dir}")

    return "\n".join(lines)


def main():
    output_dir = Path(__file__).parent

    logger.info("Loading MapEval-Textual dataset...")
    ds = load_dataset("MapEval/MapEval-Textual", "benchmark", split="test")
    logger.info(f"Loaded {len(ds)} samples")

    geolocator = Nominatim(user_agent="mapeval_gps_v4")
    cache = load_all_caches()

    # Phase 1: Extract all places + sample country
    logger.info("Phase 1: Extracting places...")
    all_places = []  # (sample_idx, name, address, sample_country)
    sample_places = []  # List of dicts per sample

    for idx, sample in enumerate(ds):
        places_dict = extract_places_with_addresses(sample['context'])
        sample_country = extract_country_from_context(sample['context'])
        sample_places.append({'sample': sample, 'places_dict': places_dict, 'country': sample_country})
        for name, address in places_dict.items():
            all_places.append((idx, name, address, sample_country))

    logger.info(f"Found {len(all_places)} total places across {len(ds)} samples")

    # Phase 2: Geocode (optimized)
    logger.info("Phase 2: Geocoding...")

    place_coords = {}  # (name, address, country) -> (lat, lon)
    stats = {'extracted': 0, 'cached': 0, 'api_calls': 0, 'no_country': 0, 'failed': 0}

    for i, (sample_idx, name, address, sample_country) in enumerate(all_places):
        key = (name, address, sample_country)
        if key in place_coords:
            continue  # Already processed

        lat, lon, source = geocode_place(geolocator, name, address, sample_country, cache)
        place_coords[key] = (lat, lon)

        # Track stats
        if source == 'extracted_from_address':
            stats['extracted'] += 1
        elif source == 'cached':
            stats['cached'] += 1
        elif source == 'no_country_context':
            stats['no_country'] += 1
        elif source == 'api_failed' or source == 'cached_failure':
            stats['failed'] += 1
        else:
            stats['api_calls'] += 1

        # Progress + save cache periodically
        if (i + 1) % 100 == 0:
            save_cache(cache)
            total_success = stats['extracted'] + stats['cached'] + stats['api_calls']
            logger.info(f"Progress: {i+1}/{len(all_places)}, success: {total_success}")

    save_cache(cache)

    logger.info(f"Geocoding stats: extracted={stats['extracted']}, cached={stats['cached']}, "
                f"api_calls={stats['api_calls']}, no_country={stats['no_country']}, failed={stats['failed']}")

    # Phase 3: Build augmented dataset
    logger.info("Phase 3: Building augmented dataset...")

    augmented_samples = []
    total_places = 0
    total_geocoded = 0
    total_filtered = 0

    for sd in sample_places:
        sample = sd['sample']
        places_dict = sd['places_dict']
        sample_country = sd['country']

        places = []
        for name, address in places_dict.items():
            lat, lon = place_coords.get((name, address, sample_country), (None, None))
            places.append({'name': name, 'address': address, 'lat': lat, 'lon': lon})

        places_before = sum(1 for p in places if p['lat'] is not None)
        places = validate_coordinates_median(places)
        places_after = sum(1 for p in places if p['lat'] is not None)
        total_filtered += (places_before - places_after)

        gps_text = format_gps_augmentation(places)
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
            'places': places,
            'num_places': num_places,
            'num_geocoded': num_geocoded
        }
        augmented_samples.append(augmented)

        total_places += num_places
        total_geocoded += num_geocoded

    # Save
    output_file = output_dir / "mapeval_gps_augmented_v4.json"
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
    logger.info(f"Filtered as outliers: {total_filtered}")


if __name__ == "__main__":
    main()
