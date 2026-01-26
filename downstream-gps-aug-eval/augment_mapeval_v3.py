#!/usr/bin/env python3
"""
MapEval GPS augmentation v3 - Fixed geocoding.

Key fixes:
1. Extract country from address - NEVER geocode without country context
2. Use median for outlier detection (robust to outliers)
3. Better query ordering
"""

import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datasets import load_dataset
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.distance import geodesic
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "geocode_cache_v3"
CACHE_DIR.mkdir(exist_ok=True)

# Known countries in MapEval dataset
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

# Patterns to skip
SKIP_PATTERNS = [
    r'^(I-\d+|US-\d+|SR-\d+|Route \d+)',
    r'^(N|S|E|W|North|South|East|West)\s+\w+\s+(St|Ave|Rd|Blvd|Dr|Way|Ln)',
    r'^(Main|High|First|Second|Third|1st|2nd|3rd)\s+(St|Street|Ave|Avenue)',
    r'^(left|right|straight|turn)',
    r'^\d+\s*(mins?|minutes?|hours?|km|miles?)',
]


def load_geocode_cache() -> Dict[str, dict]:
    cache_file = CACHE_DIR / "geocode_cache_v3.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return {}


def save_geocode_cache(cache: Dict[str, dict]):
    cache_file = CACHE_DIR / "geocode_cache_v3.json"
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


def extract_country_from_address(address: str) -> Optional[str]:
    """Extract country from address string."""
    if not address:
        return None

    address_lower = address.lower()
    for country in KNOWN_COUNTRIES:
        if country.lower() in address_lower:
            return country

    return None


def extract_city_from_address(address: str) -> Optional[str]:
    """Try to extract city from address."""
    if not address:
        return None

    # Common pattern: "Street, City PostalCode, Country"
    parts = [p.strip() for p in address.split(',')]

    if len(parts) >= 2:
        # Look for city (usually before postal code, after street)
        for part in parts[1:-1]:  # Skip first (street) and last (country)
            # Remove postal codes
            cleaned = re.sub(r'\d{4,}', '', part).strip()
            cleaned = re.sub(r'[A-Z]{2}\s*\d+', '', cleaned).strip()  # Remove "NSW 2000" etc
            if cleaned and len(cleaned) > 2:
                return cleaned

    return None


def extract_location_context(question: str, context: str) -> Optional[str]:
    """Extract city/country context from question or context."""
    patterns = [
        r'in\s+([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)',
        r'in\s+([A-Z][a-zA-Z\s]+,\s*[A-Z]{2})',
        r'in\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)',
        r'from\s+([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)',
        r'at\s+([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip()

    return None


def geocode_with_country(
    geolocator: Nominatim,
    name: str,
    address: Optional[str],
    location_context: Optional[str],
    cache: Dict[str, dict],
    retries: int = 2
) -> Tuple[Optional[float], Optional[float], str]:
    """Geocode with country context - NEVER without country."""

    # Extract country from address
    country = extract_country_from_address(address)
    city = extract_city_from_address(address)

    # Build queries - always include country!
    queries_to_try = []

    # 1. Try full address first
    if address:
        # Clean address (remove coordinates if present)
        clean_addr = re.sub(r'\([0-9., -]+\)', '', address).strip()
        queries_to_try.append(clean_addr)

    # 2. Try name + city + country
    if city and country:
        queries_to_try.append(f"{name}, {city}, {country}")

    # 3. Try name + country (from address)
    if country:
        queries_to_try.append(f"{name}, {country}")

    # 4. Try name + location_context (from question)
    if location_context and location_context not in [q for q in queries_to_try]:
        queries_to_try.append(f"{name}, {location_context}")

    # 5. NEVER try just the name without country context!
    # This is the key fix - no fallback to name-only

    for query in queries_to_try:
        cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()

        if cache_key in cache:
            cached = cache[cache_key]
            if cached.get('lat') is not None:
                return cached['lat'], cached['lon'], 'cached'
            continue

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

        cache[cache_key] = {'lat': None, 'lon': None, 'query': query}

    return None, None, 'failed (no country-less fallback)'


def validate_coordinates_median(places: List[dict]) -> List[dict]:
    """Filter outliers using MEDIAN centroid (robust to outliers)."""
    if len(places) < 3:
        return places

    valid_coords = [(p['lat'], p['lon']) for p in places if p['lat'] is not None]
    if len(valid_coords) < 3:
        return places

    # Use MEDIAN for centroid (robust to outliers)
    sorted_lats = sorted(c[0] for c in valid_coords)
    sorted_lons = sorted(c[1] for c in valid_coords)
    median_lat = sorted_lats[len(sorted_lats) // 2]
    median_lon = sorted_lons[len(sorted_lons) // 2]

    # Calculate distances to median centroid
    distances = []
    for lat, lon in valid_coords:
        try:
            dist = geodesic((lat, lon), (median_lat, median_lon)).km
            distances.append(dist)
        except:
            distances.append(0)

    if not distances:
        return places

    # Use stricter threshold: max(50km, 3x median distance)
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
                logger.info(f"Filtered outlier: {p['name']} ({dist:.0f}km from median centroid, threshold={threshold:.0f}km)")
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

    # POI format: "Place: City, Country(lat, lon)"
    pattern3 = r'<b>([^<]+)</b>:\s*([^<\n]+)'
    for match in re.finditer(pattern3, context):
        name = match.group(1).strip()
        address = match.group(2).strip().rstrip('.')
        if name not in places and not should_skip_place(name):
            places[name] = address

    # Remaining <b> tags
    all_bold = re.findall(r'<b>([^<]+)</b>', context)
    for name in all_bold:
        name = name.strip()
        if name and name not in places and not should_skip_place(name):
            places[name] = None

    return places


def format_gps_augmentation(places: List[dict], location_context: Optional[str]) -> str:
    """Format GPS coordinates."""
    geocoded = [p for p in places if p.get('lat') is not None]

    if not geocoded:
        return ""

    lines = ["\n\n[GPS Reference Coordinates]"]
    if location_context:
        lines.append(f"Location: {location_context}")

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

    geolocator = Nominatim(user_agent="mapeval_gps_v3")
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
    logger.info("Phase 2: Geocoding places (with country context)...")

    unique_queries = {}
    for sd in sample_data:
        ctx = sd['location_context']
        for name, address in sd['places_dict'].items():
            key = (name, address, ctx)
            if key not in unique_queries:
                unique_queries[key] = (address, ctx)

    logger.info(f"Found {len(unique_queries)} unique place-address pairs")

    place_coords = {}
    for i, ((name, address, ctx), (addr, location_context)) in enumerate(unique_queries.items()):
        lat, lon, source = geocode_with_country(geolocator, name, addr, location_context, cache)
        place_coords[(name, address, ctx)] = (lat, lon)

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
    total_filtered = 0

    for sd in sample_data:
        sample = sd['sample']
        places_dict = sd['places_dict']
        location_context = sd['location_context']

        places = []
        for name, address in places_dict.items():
            lat, lon = place_coords.get((name, address, location_context), (None, None))
            places.append({
                'name': name,
                'address': address,
                'lat': lat,
                'lon': lon
            })

        # Validate coordinates (filter outliers with median)
        places_before = sum(1 for p in places if p['lat'] is not None)
        places = validate_coordinates_median(places)
        places_after = sum(1 for p in places if p['lat'] is not None)
        total_filtered += (places_before - places_after)

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
    output_file = output_dir / "mapeval_gps_augmented_v3.json"
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
