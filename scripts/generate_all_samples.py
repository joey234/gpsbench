#!/usr/bin/env python3
"""
Master script to generate samples for all GPSBench tasks to reach 1000 per task.

This script generates high-quality samples using:
- GeoNames cities15000.txt for location data
- Vincenty/Haversine formulas for geodetic calculations
- Standard coordinate conversion libraries
"""

import json
import math
import random
from pathlib import Path
from collections import Counter
import string

# Set seed for reproducibility
random.seed(42)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km using Haversine formula."""
    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def initial_bearing(lat1, lon1, lat2, lon2):
    """Calculate initial bearing in degrees."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)

    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360

def bearing_to_cardinal(bearing):
    """Convert bearing to cardinal direction."""
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = round(bearing / 45) % 8
    return directions[index]

def interpolate_point(lat1, lon1, lat2, lon2, fraction):
    """Interpolate a point along great circle path."""
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    d = 2 * math.asin(math.sqrt(
        math.sin((lat2_rad - lat1_rad)/2)**2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin((lon2_rad - lon1_rad)/2)**2
    ))

    if d == 0:
        return lat1, lon1

    a = math.sin((1-fraction)*d) / math.sin(d)
    b = math.sin(fraction*d) / math.sin(d)

    x = a * math.cos(lat1_rad) * math.cos(lon1_rad) + b * math.cos(lat2_rad) * math.cos(lon2_rad)
    y = a * math.cos(lat1_rad) * math.sin(lon1_rad) + b * math.cos(lat2_rad) * math.sin(lon2_rad)
    z = a * math.sin(lat1_rad) + b * math.sin(lat2_rad)

    lat = math.atan2(z, math.sqrt(x**2 + y**2))
    lon = math.atan2(y, x)

    return math.degrees(lat), math.degrees(lon)

def spherical_polygon_area(vertices):
    """
    Calculate spherical polygon area in km² using the spherical excess method.

    Uses the standard approach:
    1. Convert lat/lon to 3D unit vectors on a sphere
    2. Triangulate polygon from first vertex
    3. Calculate spherical excess of each triangle using L'Huilier's formula
    4. Sum excesses and multiply by R²

    This matches standard geodesy formulas that LLMs would learn.
    """
    R = 6371.0  # Earth radius in km
    n = len(vertices)
    if n < 3:
        return 0.0

    def to_cartesian(lat, lon):
        """Convert lat/lon (degrees) to 3D unit vector."""
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x = math.cos(lat_rad) * math.cos(lon_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad)
        z = math.sin(lat_rad)
        return (x, y, z)

    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def cross(a, b):
        return (
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        )

    def norm(v):
        return math.sqrt(dot(v, v))

    def angle_between(a, b):
        """Angle between two unit vectors (central angle on sphere)."""
        d = dot(a, b)
        d = max(-1.0, min(1.0, d))  # Clamp for numerical stability
        return math.acos(d)

    def spherical_triangle_area(p1, p2, p3):
        """
        Calculate area of spherical triangle using L'Huilier's formula.
        p1, p2, p3 are 3D unit vectors.
        Returns area on unit sphere (multiply by R² for actual area).
        """
        # Side lengths (central angles)
        a = angle_between(p2, p3)
        b = angle_between(p1, p3)
        c = angle_between(p1, p2)

        # Semi-perimeter
        s = (a + b + c) / 2.0

        # L'Huilier's formula for spherical excess
        # tan(E/4) = sqrt(tan(s/2) * tan((s-a)/2) * tan((s-b)/2) * tan((s-c)/2))
        try:
            t1 = math.tan(s / 2.0)
            t2 = math.tan((s - a) / 2.0)
            t3 = math.tan((s - b) / 2.0)
            t4 = math.tan((s - c) / 2.0)

            product = t1 * t2 * t3 * t4
            if product < 0:
                product = 0  # Handle numerical issues

            tan_E_4 = math.sqrt(product)
            E = 4.0 * math.atan(tan_E_4)
            return E  # Spherical excess = area on unit sphere
        except (ValueError, ZeroDivisionError):
            return 0.0

    # Convert all vertices to 3D unit vectors
    points = [to_cartesian(v[0], v[1]) for v in vertices]

    # Triangulate from first vertex and sum areas
    total_excess = 0.0
    for i in range(1, n - 1):
        excess = spherical_triangle_area(points[0], points[i], points[i + 1])
        total_excess += excess

    # Area = R² × spherical_excess
    area = R * R * total_excess
    return area

def polygon_perimeter(vertices):
    """Calculate polygon perimeter in km."""
    perimeter = 0
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        perimeter += haversine_distance(
            vertices[i][0], vertices[i][1],
            vertices[j][0], vertices[j][1]
        )
    return perimeter

def add_gaussian_noise(lat, lon, noise_meters):
    """
    Add Gaussian noise to coordinates.

    Args:
        lat: Original latitude
        lon: Original longitude
        noise_meters: Standard deviation of noise in meters

    Returns:
        Tuple of (noisy_lat, noisy_lon, actual_distance_m)
    """
    # Convert noise from meters to degrees
    # 1 degree latitude ≈ 111,000 meters
    # 1 degree longitude varies by latitude
    noise_lat_deg = noise_meters / 111000
    noise_lon_deg = noise_meters / (111000 * max(math.cos(math.radians(lat)), 0.01))

    # Sample from 2D Gaussian
    delta_lat = random.gauss(0, noise_lat_deg)
    delta_lon = random.gauss(0, noise_lon_deg)

    noisy_lat = lat + delta_lat
    noisy_lon = lon + delta_lon

    # Clamp to valid ranges
    noisy_lat = max(-90, min(90, noisy_lat))
    noisy_lon = ((noisy_lon + 180) % 360) - 180

    # Calculate actual distance
    actual_distance = haversine_distance(lat, lon, noisy_lat, noisy_lon) * 1000  # km to m

    return noisy_lat, noisy_lon, actual_distance


def decimal_to_dms(decimal, is_lat):
    """Convert decimal degrees to DMS string."""
    direction = ''
    if is_lat:
        direction = 'N' if decimal >= 0 else 'S'
    else:
        direction = 'E' if decimal >= 0 else 'W'

    decimal = abs(decimal)
    degrees = int(decimal)
    minutes = int((decimal - degrees) * 60)
    seconds = round((decimal - degrees - minutes/60) * 3600, 1)

    return f"{degrees}°{minutes}'{seconds}\"{direction}"

def dms_to_decimal(dms_str):
    """Parse DMS string to decimal."""
    # Simple parser for format like "40°42'46.0"N"
    import re
    match = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", dms_str)
    if match:
        d, m, s, direction = match.groups()
        decimal = float(d) + float(m)/60 + float(s)/3600
        if direction in ['S', 'W']:
            decimal = -decimal
        return decimal
    return None

class SampleGenerator:
    def __init__(self, geonames_path):
        self.cities = self.load_cities(geonames_path)
        print(f"Loaded {len(self.cities)} cities")

        # Continent rotation counter for balanced sampling across calls
        self.continent_rotation_index = 0

        # Define continent mapping (ISO country codes)
        self.continent_map = {
            'Africa': ['DZ', 'AO', 'BJ', 'BW', 'BF', 'BI', 'CM', 'CV', 'CF', 'TD', 'KM', 'CG', 'CD', 'CI', 'DJ', 'EG', 'GQ', 'ER', 'ET', 'GA', 'GM', 'GH', 'GN', 'GW', 'KE', 'LS', 'LR', 'LY', 'MG', 'MW', 'ML', 'MR', 'MU', 'YT', 'MA', 'MZ', 'NA', 'NE', 'NG', 'RE', 'RW', 'SH', 'ST', 'SN', 'SC', 'SL', 'SO', 'ZA', 'SS', 'SD', 'SZ', 'TZ', 'TG', 'TN', 'UG', 'ZM', 'ZW'],
            'Asia': ['AF', 'AM', 'AZ', 'BH', 'BD', 'BT', 'BN', 'KH', 'CN', 'CX', 'CC', 'IO', 'GE', 'HK', 'IN', 'ID', 'IR', 'IQ', 'IL', 'JP', 'JO', 'KZ', 'KW', 'KG', 'LA', 'LB', 'MO', 'MY', 'MV', 'MN', 'MM', 'NP', 'KP', 'OM', 'PK', 'PS', 'PH', 'QA', 'SA', 'SG', 'KR', 'LK', 'SY', 'TW', 'TJ', 'TH', 'TL', 'TR', 'TM', 'AE', 'UZ', 'VN', 'YE'],
            'Europe': ['AX', 'AL', 'AD', 'AT', 'BY', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FO', 'FI', 'FR', 'DE', 'GI', 'GR', 'GG', 'VA', 'HU', 'IS', 'IE', 'IM', 'IT', 'JE', 'XK', 'LV', 'LI', 'LT', 'LU', 'MK', 'MT', 'MD', 'MC', 'ME', 'NL', 'NO', 'PL', 'PT', 'RO', 'RU', 'SM', 'RS', 'SK', 'SI', 'ES', 'SJ', 'SE', 'CH', 'UA', 'GB'],
            'North America': ['AI', 'AG', 'AW', 'BS', 'BB', 'BZ', 'BM', 'BQ', 'CA', 'KY', 'CR', 'CU', 'CW', 'DM', 'DO', 'SV', 'GL', 'GD', 'GP', 'GT', 'HT', 'HN', 'JM', 'MQ', 'MX', 'MS', 'NI', 'PA', 'PM', 'PR', 'BL', 'KN', 'LC', 'MF', 'VC', 'SX', 'TT', 'TC', 'US', 'VG', 'VI'],
            'South America': ['AR', 'BO', 'BV', 'BR', 'CL', 'CO', 'EC', 'FK', 'GF', 'GY', 'PY', 'PE', 'GS', 'SR', 'UY', 'VE'],
            'Oceania': ['AS', 'AU', 'CK', 'FJ', 'PF', 'GU', 'KI', 'MH', 'FM', 'NR', 'NC', 'NZ', 'NU', 'NF', 'MP', 'PW', 'PG', 'PN', 'WS', 'SB', 'TK', 'TO', 'TV', 'UM', 'VU', 'WF'],
            'Antarctica': ['AQ']
        }

        # Build reverse lookup: country code -> continent
        self.country_to_continent = {}
        for continent, countries in self.continent_map.items():
            for country in countries:
                self.country_to_continent[country] = continent

        # Group cities by country for diversity
        self.cities_by_country = {}
        for city in self.cities:
            country = city['country']
            if country not in self.cities_by_country:
                self.cities_by_country[country] = []
            self.cities_by_country[country].append(city)

        # Group cities by continent for global coverage
        self.cities_by_continent = {cont: [] for cont in self.continent_map}
        for city in self.cities:
            for continent, countries in self.continent_map.items():
                if city['country'] in countries:
                    self.cities_by_continent[continent].append(city)
                    break

        print(f"Cities from {len(self.cities_by_country)} countries")
        print("Cities per continent:")
        for cont in sorted(self.cities_by_continent.keys()):
            print(f"  {cont:20} {len(self.cities_by_continent[cont]):6}")

    def load_cities(self, geonames_path):
        """Load cities from GeoNames."""
        cities = []
        with open(geonames_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 15:
                    try:
                        # Parse alternate names
                        alt_names = parts[3].split(',') if parts[3] else []
                        # Limit to first 5 alternate names
                        alt_names = [name.strip() for name in alt_names[:5]]

                        city = {
                            'name': parts[1],
                            'lat': float(parts[4]),
                            'lon': float(parts[5]),
                            'country': parts[8],
                            'population': int(parts[14]) if parts[14] else 0,
                            'alternate_names': alt_names
                        }
                        if -90 <= city['lat'] <= 90 and -180 <= city['lon'] <= 180:
                            cities.append(city)
                    except (ValueError, IndexError):
                        continue
        return cities

    def get_random_cities(self, n, min_distance_km=100):
        """Get n random cities with minimum distance between them."""
        selected = []
        attempts = 0
        max_attempts = n * 20

        while len(selected) < n and attempts < max_attempts:
            city = random.choice(self.cities)
            attempts += 1

            # Check distance to all selected cities
            too_close = False
            for s in selected:
                dist = haversine_distance(city['lat'], city['lon'], s['lat'], s['lon'])
                if dist < min_distance_km:
                    too_close = True
                    break

            if not too_close:
                selected.append(city)

        return selected

    def get_regional_cities(self, n, min_distance_km=50, max_distance_km=2000):
        """
        Get n random cities from the SAME continent/region.

        This creates realistic polygons (like countries/states) rather than
        globe-spanning shapes. Cities are selected from a single continent
        with distances constrained to create sensible geographic regions.

        Args:
            n: Number of cities to select
            min_distance_km: Minimum distance between cities (default 50km)
            max_distance_km: Maximum distance from first city (default 2000km)

        Returns:
            List of city dicts from the same region
        """
        # Pick a random continent with enough cities
        continents_with_cities = [
            cont for cont, cities in self.cities_by_continent.items()
            if len(cities) >= n * 2  # Need enough for selection
        ]

        if not continents_with_cities:
            return self.get_random_cities(n, min_distance_km)

        continent = random.choice(continents_with_cities)
        continent_cities = self.cities_by_continent[continent]

        # Start with a random anchor city
        anchor = random.choice(continent_cities)
        selected = [anchor]

        # Find nearby cities within max_distance_km of anchor
        nearby_cities = []
        for city in continent_cities:
            if city == anchor:
                continue
            dist = haversine_distance(anchor['lat'], anchor['lon'], city['lat'], city['lon'])
            if min_distance_km <= dist <= max_distance_km:
                nearby_cities.append((city, dist))

        # Sort by distance and try to select diverse cities
        random.shuffle(nearby_cities)

        attempts = 0
        max_attempts = len(nearby_cities) * 3

        while len(selected) < n and attempts < max_attempts and nearby_cities:
            attempts += 1

            # Pick a candidate
            idx = attempts % len(nearby_cities)
            city, _ = nearby_cities[idx]

            # Check minimum distance to all selected cities
            too_close = False
            for s in selected:
                dist = haversine_distance(city['lat'], city['lon'], s['lat'], s['lon'])
                if dist < min_distance_km:
                    too_close = True
                    break

            if not too_close:
                selected.append(city)
                # Remove from candidates
                nearby_cities = [(c, d) for c, d in nearby_cities if c != city]

        return selected

    def get_globally_balanced_cities(self, n, min_distance_km=100):
        """
        Get n random cities with global geographic balance.

        Ensures representation from all continents based on target distribution:
        - Africa: 20%
        - Asia: 25%
        - Europe: 20%
        - North America: 15%
        - South America: 15%
        - Oceania: 5%

        This balances against the natural data skew (Asia=37%, Europe=24%, Oceania=1%).
        """
        target_distribution = {
            'Africa': 0.20,
            'Asia': 0.25,
            'Europe': 0.20,
            'North America': 0.15,
            'South America': 0.15,
            'Oceania': 0.05
        }

        # For small n, use weighted random sampling to ensure all continents get represented over time
        if n < 6:
            # Randomly select n continents with probability based on target distribution
            continents = list(target_distribution.keys())
            weights = [target_distribution[c] for c in continents]

            # Randomly select n continents (with replacement, so same continent can be picked multiple times)
            selected_continents = random.choices(continents, weights=weights, k=n)

            # Count how many cities from each continent
            continent_quotas = {}
            for cont in selected_continents:
                continent_quotas[cont] = continent_quotas.get(cont, 0) + 1
        else:
            # For larger n, use proportional allocation
            continent_quotas = {}
            for continent, ratio in target_distribution.items():
                quota = int(n * ratio)
                continent_quotas[continent] = quota

            # Ensure at least 1 per continent for larger samples
            for continent in continent_quotas:
                if continent_quotas[continent] == 0:
                    continent_quotas[continent] = 1

            # Adjust to match exact n
            total_allocated = sum(continent_quotas.values())

            if total_allocated < n:
                # Add extras to largest continents
                continents_by_priority = ['Asia', 'Africa', 'Europe', 'North America', 'South America', 'Oceania']
                for continent in continents_by_priority:
                    if total_allocated >= n:
                        break
                    continent_quotas[continent] += 1
                    total_allocated += 1
            elif total_allocated > n:
                # Remove from smallest allocations first (preserve diversity where possible)
                continents_sorted = sorted(continent_quotas.items(), key=lambda x: (x[1], x[0]), reverse=True)
                for continent, quota in continents_sorted:
                    if total_allocated <= n:
                        break
                    if quota > 0:
                        reduction = min(quota, total_allocated - n)
                        continent_quotas[continent] -= reduction
                        total_allocated -= reduction

        selected = []
        attempts_per_continent = {cont: 0 for cont in target_distribution}
        max_attempts_per_continent = 200

        for continent, quota in continent_quotas.items():
            continent_cities = self.cities_by_continent.get(continent, [])
            if not continent_cities:
                continue

            continent_selected = []
            while len(continent_selected) < quota and attempts_per_continent[continent] < max_attempts_per_continent:
                city = random.choice(continent_cities)
                attempts_per_continent[continent] += 1

                # Check distance to all previously selected cities (including from other continents)
                too_close = False
                for s in selected + continent_selected:
                    dist = haversine_distance(city['lat'], city['lon'], s['lat'], s['lon'])
                    if dist < min_distance_km:
                        too_close = True
                        break

                if not too_close:
                    continent_selected.append(city)

            selected.extend(continent_selected)

        # If we didn't get enough, fill from any continent
        attempts = 0
        max_attempts = n * 20
        while len(selected) < n and attempts < max_attempts:
            city = random.choice(self.cities)
            attempts += 1

            too_close = False
            for s in selected:
                dist = haversine_distance(city['lat'], city['lon'], s['lat'], s['lon'])
                if dist < min_distance_km:
                    too_close = True
                    break

            if not too_close:
                selected.append(city)

        return selected

    def get_same_continent_cities(self, n, continent=None, min_distance_km=100, max_distance_km=3000):
        """
        Get n cities from the same continent within a specified distance range.

        This is used for Applied Relative Position to make questions harder
        by selecting cities that are geographically closer together.

        Args:
            n: Number of cities to select
            continent: Specific continent (if None, randomly chosen)
            min_distance_km: Minimum distance between selected cities
            max_distance_km: Maximum distance between first two cities

        Returns:
            List of city dictionaries
        """
        if continent is None:
            # Randomly select a continent weighted by city count
            continents = list(self.cities_by_continent.keys())
            weights = [len(self.cities_by_continent[c]) for c in continents]
            continent = random.choices(continents, weights=weights, k=1)[0]

        continent_cities = self.cities_by_continent.get(continent, [])
        if len(continent_cities) < n:
            # Fall back to global selection if continent too small
            return self.get_globally_balanced_cities(n, min_distance_km)

        selected = []
        attempts = 0
        max_attempts = n * 100

        # Select first city randomly
        first_city = random.choice(continent_cities)
        selected.append(first_city)

        # Select remaining cities within distance constraints
        while len(selected) < n and attempts < max_attempts:
            city = random.choice(continent_cities)
            attempts += 1

            # Check distance constraints
            dist_to_first = haversine_distance(
                city['lat'], city['lon'],
                first_city['lat'], first_city['lon']
            )

            # Must be within max distance of first city
            if dist_to_first > max_distance_km:
                continue

            # Must not be too close to any selected city
            too_close = False
            for s in selected:
                dist = haversine_distance(city['lat'], city['lon'], s['lat'], s['lon'])
                if dist < min_distance_km:
                    too_close = True
                    break

            if not too_close:
                selected.append(city)

        return selected

    # =========================================================================
    # DISTANCE CALCULATION
    # =========================================================================
    def generate_distance_samples(self, current_data, target=1000):
        """Generate distance calculation samples."""
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} distance samples...")
        new_samples = []

        for _ in range(needed):
            cities = self.get_globally_balanced_cities(2, min_distance_km=50)
            if len(cities) < 2:
                continue

            c1, c2 = cities
            distance = haversine_distance(c1['lat'], c1['lon'], c2['lat'], c2['lon'])

            sample = {
                'task': 'distance_calculation',
                'point_a': {
                    'lat': c1['lat'],
                    'lon': c1['lon'],
                    'name': c1['name']
                },
                'point_b': {
                    'lat': c2['lat'],
                    'lon': c2['lon'],
                    'name': c2['name']
                },
                'ground_truth': {
                    'distance_km': round(distance, 2),
                    'distance_miles': round(distance * 0.621371, 2),
                    'method': 'haversine',
                    'tolerance_km': max(5.0, distance * 0.05)
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # BEARING COMPUTATION
    # =========================================================================
    def generate_bearing_samples(self, current_data, target=1000):
        """Generate bearing computation samples."""
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} bearing samples...")
        new_samples = []

        for _ in range(needed):
            cities = self.get_globally_balanced_cities(2, min_distance_km=100)
            if len(cities) < 2:
                continue

            c1, c2 = cities
            bearing = initial_bearing(c1['lat'], c1['lon'], c2['lat'], c2['lon'])
            cardinal = bearing_to_cardinal(bearing)

            sample = {
                'task': 'bearing_computation',
                'point_a': {
                    'lat': c1['lat'],
                    'lon': c1['lon'],
                    'name': c1['name']
                },
                'point_b': {
                    'lat': c2['lat'],
                    'lon': c2['lon'],
                    'name': c2['name']
                },
                'ground_truth': {
                    'initial_bearing_deg': round(bearing, 2),
                    'cardinal_direction': cardinal,
                    'tolerance_deg': 5.0
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # INTERPOLATION
    # =========================================================================
    def generate_interpolation_samples(self, current_data, target=1000):
        """Generate interpolation samples."""
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} interpolation samples...")
        new_samples = []

        fractions = [0.25, 0.5, 0.75]

        for _ in range(needed):
            cities = self.get_globally_balanced_cities(2, min_distance_km=500)
            if len(cities) < 2:
                continue

            c1, c2 = cities
            fraction = random.choice(fractions)
            interp_lat, interp_lon = interpolate_point(
                c1['lat'], c1['lon'], c2['lat'], c2['lon'], fraction
            )

            sample = {
                'task': 'interpolation',
                'point_a': {
                    'lat': c1['lat'],
                    'lon': c1['lon'],
                    'name': c1['name']
                },
                'point_b': {
                    'lat': c2['lat'],
                    'lon': c2['lon'],
                    'name': c2['name']
                },
                'fraction': fraction,
                'ground_truth': {
                    'interpolated_point': {
                        'lat': round(interp_lat, 6),
                        'lon': round(interp_lon, 6)
                    },
                    'tolerance_deg': 0.01
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # AREA & PERIMETER
    # =========================================================================
    def generate_area_perimeter_samples(self, current_data, target=1000):
        """Generate area & perimeter samples with regional polygons.

        Uses get_regional_cities to create realistic polygon sizes
        (country/state-sized regions) rather than globe-spanning shapes.
        """
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} area/perimeter samples (regional polygons)...")
        new_samples = []

        for _ in range(needed):
            # Generate 3-6 sided polygons
            num_vertices = random.randint(3, 6)
            # Use regional cities for realistic polygon sizes (50-2000km range)
            cities = self.get_regional_cities(num_vertices, min_distance_km=50, max_distance_km=2000)
            if len(cities) < num_vertices:
                continue

            vertices = [(c['lat'], c['lon']) for c in cities]

            # Sort vertices by angle from centroid for proper polygon
            centroid_lat = sum(v[0] for v in vertices) / len(vertices)
            centroid_lon = sum(v[1] for v in vertices) / len(vertices)

            def angle_from_centroid(v):
                return math.atan2(v[1] - centroid_lon, v[0] - centroid_lat)

            # Get sorted indices to maintain city name mapping
            indexed_vertices = list(enumerate(vertices))
            sorted_indexed = sorted(indexed_vertices, key=lambda x: angle_from_centroid(x[1]))
            sorted_indices = [idx for idx, _ in sorted_indexed]
            sorted_vertices = [v for _, v in sorted_indexed]

            area = spherical_polygon_area(sorted_vertices)
            perimeter = polygon_perimeter(sorted_vertices)

            # Use sorted vertices with correct city names
            points = [
                {'lat': sorted_vertices[i][0], 'lon': sorted_vertices[i][1], 'name': cities[sorted_indices[i]]['name']}
                for i in range(len(sorted_vertices))
            ]

            sample = {
                'task': 'area_perimeter',
                'polygon_type': f'{num_vertices}-sided',
                'vertices': points,
                'ground_truth': {
                    'area_km2': round(area, 2),
                    'perimeter_km': round(perimeter, 2),
                    'tolerance_percent': 5.0
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # BOUNDING BOX
    # =========================================================================
    def generate_bounding_box_samples(self, current_data, target=1000):
        """Generate bounding box samples."""
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} bounding box samples...")
        new_samples = []

        for _ in range(needed):
            num_points = random.randint(3, 8)
            cities = self.get_globally_balanced_cities(num_points, min_distance_km=50)
            if len(cities) < 3:
                continue

            lats = [c['lat'] for c in cities]
            lons = [c['lon'] for c in cities]

            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)

            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            centroid_lat = sum(lats) / len(lats)
            centroid_lon = sum(lons) / len(lons)

            diagonal = haversine_distance(min_lat, min_lon, max_lat, max_lon)

            points = [
                {'lat': c['lat'], 'lon': c['lon'], 'name': c['name']}
                for c in cities
            ]

            sample = {
                'task': 'bounding_box',
                'points': points,
                'ground_truth': {
                    'min_lat': round(min_lat, 6),
                    'max_lat': round(max_lat, 6),
                    'min_lon': round(min_lon, 6),
                    'max_lon': round(max_lon, 6),
                    'center': {
                        'lat': round(center_lat, 6),
                        'lon': round(center_lon, 6)
                    },
                    'centroid': {
                        'lat': round(centroid_lat, 6),
                        'lon': round(centroid_lon, 6)
                    },
                    'diagonal_km': round(diagonal, 2),
                    'width_deg': round(max_lon - min_lon, 6),
                    'height_deg': round(max_lat - min_lat, 6)
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # FORMAT CONVERSION
    # =========================================================================
    def generate_format_conversion_samples(self, current_data, target=1000):
        """Generate format conversion samples."""
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} format conversion samples...")
        new_samples = []

        conversions = [
            ('decimal_degrees', 'DMS'),
            ('DMS', 'decimal_degrees'),
        ]

        for _ in range(needed):
            city = random.choice(self.cities)
            lat, lon = city['lat'], city['lon']
            conversion = random.choice(conversions)

            if conversion[0] == 'decimal_degrees':
                source_str = f"{lat:.6f}, {lon:.6f}"
                target_str = f"{decimal_to_dms(lat, True)}, {decimal_to_dms(lon, False)}"
            else:
                source_str = f"{decimal_to_dms(lat, True)}, {decimal_to_dms(lon, False)}"
                target_str = f"{lat:.6f}, {lon:.6f}"

            sample = {
                'task': 'format_conversion',
                'location_name': city['name'],
                'source_format': conversion[0],
                'source_string': source_str,
                'target_format': conversion[1],
                'ground_truth': {
                    'target_string': target_str,
                    'latitude': lat,
                    'longitude': lon
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # PRECISION UNDERSTANDING
    # =========================================================================
    def generate_precision_samples(self, current_data, target=1000):
        """Generate precision understanding samples."""
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} precision samples...")
        new_samples = []

        precision_levels = [
            (0, 111000, 'country-level'),
            (1, 11100, 'large-city'),
            (2, 1110, 'neighborhood'),
            (3, 111, 'street'),
            (4, 11.1, 'building'),
            (5, 1.1, 'precise'),
            (6, 0.11, 'survey-grade'),
        ]

        for _ in range(needed):
            city = random.choice(self.cities)
            decimals, accuracy, use_case = random.choice(precision_levels)

            # Round coordinates to specified precision
            lat_str = f"{city['lat']:.{decimals}f}" if decimals > 0 else str(int(round(city['lat'])))
            lon_str = f"{city['lon']:.{decimals}f}" if decimals > 0 else str(int(round(city['lon'])))

            sample = {
                'task': 'precision_understanding',
                'location_name': city['name'],
                'coordinate_string': f"{lat_str}, {lon_str}",
                'ground_truth': {
                    'decimal_places': decimals,
                    'accuracy_meters': accuracy,
                    'appropriate_use_case': use_case,
                    'latitude': city['lat'],
                    'longitude': city['lon']
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # RELATIVE POSITION
    # =========================================================================
    def generate_relative_position_samples(self, current_data, target=1000):
        """Generate relative position samples."""
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} relative position samples...")
        new_samples = []

        questions = [
            ('north', lambda c1, c2: c1['lat'] > c2['lat']),
            ('south', lambda c1, c2: c1['lat'] < c2['lat']),
            ('east', lambda c1, c2: c1['lon'] > c2['lon']),
            ('west', lambda c1, c2: c1['lon'] < c2['lon']),
        ]

        for _ in range(needed):
            cities = self.get_globally_balanced_cities(2, min_distance_km=100)
            if len(cities) < 2:
                continue

            c1, c2 = cities
            direction, check_fn = random.choice(questions)
            answer = check_fn(c1, c2)

            sample = {
                'task': 'relative_position',
                'subtask': 'cardinal_direction',
                'point_a': {
                    'lat': c1['lat'],
                    'lon': c1['lon'],
                    'name': c1['name']
                },
                'point_b': {
                    'lat': c2['lat'],
                    'lon': c2['lon'],
                    'name': c2['name']
                },
                'question': f"Is {c1['name']} {direction} of {c2['name']}?",
                'ground_truth': {
                    'answer': answer,
                    'explanation': f"{c1['name']} is {'not ' if not answer else ''}{direction} of {c2['name']}"
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    def generate_relative_position_applied_samples(self, current_data, target=1000):
        """
        Generate relative position samples for Applied track with CLOSER city pairs.

        Unlike the Pure GPS version which uses globally distributed cities (making
        the task trivially easy), this version selects cities from the SAME CONTINENT
        within 500-2000km of each other. This requires actual geographic knowledge
        since the direction isn't obvious from just knowing the hemisphere.

        Example:
        - Pure GPS: "Is Tokyo east of Paris?" → trivially yes (Asia vs Europe)
        - Applied: "Is Lyon east of Paris?" → requires knowing France geography
        """
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} relative position samples with closer city pairs...")
        new_samples = []

        questions = [
            ('north', lambda c1, c2: c1['lat'] > c2['lat']),
            ('south', lambda c1, c2: c1['lat'] < c2['lat']),
            ('east', lambda c1, c2: c1['lon'] > c2['lon']),
            ('west', lambda c1, c2: c1['lon'] < c2['lon']),
        ]

        # Distance ranges to cycle through (in km)
        # These create progressively harder questions
        distance_ranges = [
            (200, 500),    # Very close - hardest
            (500, 1000),   # Close
            (1000, 2000),  # Moderate
            (2000, 3000),  # Further but still same region
        ]

        for i in range(needed):
            # Cycle through distance ranges for variety
            min_dist, max_dist = distance_ranges[i % len(distance_ranges)]

            # Get cities from same continent within distance range
            cities = self.get_same_continent_cities(
                2,
                min_distance_km=min_dist,
                max_distance_km=max_dist
            )

            if len(cities) < 2:
                # Fall back to global selection if same-continent fails
                cities = self.get_globally_balanced_cities(2, min_distance_km=100)
                if len(cities) < 2:
                    continue

            c1, c2 = cities
            direction, check_fn = random.choice(questions)
            answer = check_fn(c1, c2)

            # Calculate actual distance for metadata
            actual_distance = haversine_distance(c1['lat'], c1['lon'], c2['lat'], c2['lon'])

            sample = {
                'task': 'relative_position',
                'subtask': 'cardinal_direction',
                'point_a': {
                    'lat': c1['lat'],
                    'lon': c1['lon'],
                    'name': c1['name']
                },
                'point_b': {
                    'lat': c2['lat'],
                    'lon': c2['lon'],
                    'name': c2['name']
                },
                'question': f"Is {c1['name']} {direction} of {c2['name']}?",
                'ground_truth': {
                    'answer': answer,
                    'explanation': f"{c1['name']} is {'not ' if not answer else ''}{direction} of {c2['name']}",
                    'distance_km': round(actual_distance, 1)
                },
                'metadata': {
                    'track': 'applied',
                    'distance_range_km': [min_dist, max_dist]
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # PROXIMITY
    # =========================================================================
    def generate_proximity_samples(self, current_data, target=1000):
        """Generate proximity (nearest neighbor) samples."""
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} proximity samples...")
        new_samples = []

        for _ in range(needed):
            cities = self.get_globally_balanced_cities(5, min_distance_km=100)
            if len(cities) < 5:
                continue

            reference = cities[0]
            candidates = cities[1:]

            # Calculate distances
            distances = {}
            for c in candidates:
                dist = haversine_distance(reference['lat'], reference['lon'], c['lat'], c['lon'])
                distances[c['name']] = dist

            # Find nearest
            nearest = min(distances, key=distances.get)

            candidate_points = [
                {'lat': c['lat'], 'lon': c['lon'], 'name': c['name']}
                for c in candidates
            ]

            names = ', '.join([c['name'] for c in candidates])

            sample = {
                'task': 'proximity',
                'subtask': 'nearest_point',
                'reference_point': {
                    'lat': reference['lat'],
                    'lon': reference['lon'],
                    'name': reference['name']
                },
                'candidate_points': candidate_points,
                'question': f"Which city is closest to {reference['name']}: {names}?",
                'ground_truth': {
                    'answer': nearest,
                    'distance_km': round(distances[nearest], 1),
                    'all_distances_km': {k: round(v, 1) for k, v in distances.items()}
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # ROUTE ANALYSIS
    # =========================================================================
    def generate_route_analysis_samples(self, current_data, target=1000):
        """Generate route analysis samples."""
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} route analysis samples...")
        new_samples = []

        for _ in range(needed):
            cities = self.get_globally_balanced_cities(3, min_distance_km=200)
            if len(cities) < 3:
                continue

            start, end, waypoint = cities

            direct = haversine_distance(start['lat'], start['lon'], end['lat'], end['lon'])
            via_waypoint = (
                haversine_distance(start['lat'], start['lon'], waypoint['lat'], waypoint['lon']) +
                haversine_distance(waypoint['lat'], waypoint['lon'], end['lat'], end['lon'])
            )

            detour = via_waypoint - direct
            detour_percent = (detour / direct * 100) if direct > 0 else 0

            # On route if detour < 10%
            on_route = detour_percent < 10

            sample = {
                'task': 'route_analysis',
                'subtask': 'on_route_detection',
                'start': {
                    'lat': start['lat'],
                    'lon': start['lon'],
                    'name': start['name']
                },
                'end': {
                    'lat': end['lat'],
                    'lon': end['lon'],
                    'name': end['name']
                },
                'waypoint': {
                    'lat': waypoint['lat'],
                    'lon': waypoint['lon'],
                    'name': waypoint['name']
                },
                'question': f"Is {waypoint['name']} approximately on the direct path from {start['name']} to {end['name']}?",
                'ground_truth': {
                    'answer': on_route,
                    'direct_distance_km': round(direct, 1),
                    'via_waypoint_km': round(via_waypoint, 1),
                    'detour_km': round(detour, 1),
                    'detour_percent': round(detour_percent, 1)
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # SPATIAL PATTERNS (Outlier Detection)
    # =========================================================================
    def generate_spatial_patterns_samples(self, current_data, target=1000):
        """Generate spatial patterns samples with clear outliers."""
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} spatial patterns samples...")
        new_samples = []

        # Group cities by region
        regions = {
            'Europe': ['GB', 'DE', 'FR', 'IT', 'ES', 'PL', 'NL', 'RU'],
            'Asia': ['CN', 'JP', 'IN', 'KR', 'ID', 'TH', 'PH', 'VN', 'MY'],
            'Africa': ['NG', 'EG', 'ZA', 'KE', 'ET', 'GH'],
            'North America': ['US', 'CA', 'MX'],
            'South America': ['BR', 'AR', 'CL', 'CO', 'PE'],
            'Oceania': ['AU', 'NZ']
        }

        region_cities = {r: [] for r in regions}
        for city in self.cities:
            for region, countries in regions.items():
                if city['country'] in countries:
                    region_cities[region].append(city)
                    break

        valid_regions = [r for r, cities in region_cities.items() if len(cities) >= 10]

        attempts = 0
        max_attempts = needed * 50

        while len(new_samples) < needed and attempts < max_attempts:
            attempts += 1

            if len(valid_regions) < 2:
                break

            # Select main region and outlier region
            main_region = random.choice(valid_regions)
            outlier_region = random.choice([r for r in valid_regions if r != main_region])

            # Get cities
            main_cities = random.sample(region_cities[main_region], min(4, len(region_cities[main_region])))
            outlier_city = random.choice(region_cities[outlier_region])

            points = []
            for c in main_cities + [outlier_city]:
                points.append({'lat': c['lat'], 'lon': c['lon'], 'name': c['name']})

            # Validate outlier clarity
            lats = [p['lat'] for p in points]
            lons = [p['lon'] for p in points]
            centroid_lat = sum(lats) / len(lats)
            centroid_lon = sum(lons) / len(lons)

            distances = {}
            for p in points:
                dist = haversine_distance(p['lat'], p['lon'], centroid_lat, centroid_lon)
                distances[p['name']] = dist

            # Check 2x median and clarity ratio
            dist_values = sorted(distances.values())
            n = len(dist_values)
            median = dist_values[n//2] if n % 2 else (dist_values[n//2-1] + dist_values[n//2])/2

            sorted_dists = sorted(distances.items(), key=lambda x: -x[1])
            furthest_name, furthest_dist = sorted_dists[0]
            second_dist = sorted_dists[1][1]

            # Must have exactly one outlier > 2x median and clarity > 1.3
            outliers = [name for name, dist in distances.items() if dist > 2 * median]
            if len(outliers) != 1 or furthest_dist / second_dist < 1.3:
                continue

            names = ', '.join([p['name'] for p in points])

            sample = {
                'task': 'spatial_patterns',
                'subtask': 'outlier_detection',
                'points': points,
                'question': f"Which location is the geographic outlier among: {names}?",
                'ground_truth': {
                    'answer': furthest_name,
                    'outlier_distance_km': round(furthest_dist, 1),
                    'second_furthest_km': round(second_dist, 1),
                    'clarity_ratio': round(furthest_dist / second_dist, 2),
                    'median_distance_km': round(median, 1),
                    'distances_to_centroid': {k: round(v, 1) for k, v in distances.items()}
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # REGIONAL KNOWLEDGE
    # =========================================================================
    def generate_regional_knowledge_samples(self, current_data, target=1000):
        """Generate regional knowledge (country identification) samples."""
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} regional knowledge samples...")
        new_samples = []

        # Country code to name mapping
        country_names = {
            'US': 'United States', 'CN': 'China', 'JP': 'Japan', 'DE': 'Germany',
            'GB': 'United Kingdom', 'FR': 'France', 'IN': 'India', 'BR': 'Brazil',
            'RU': 'Russia', 'AU': 'Australia', 'CA': 'Canada', 'IT': 'Italy',
            'ES': 'Spain', 'MX': 'Mexico', 'KR': 'South Korea', 'ID': 'Indonesia',
            'NL': 'Netherlands', 'SA': 'Saudi Arabia', 'TR': 'Turkey', 'CH': 'Switzerland',
            'PL': 'Poland', 'SE': 'Sweden', 'BE': 'Belgium', 'TH': 'Thailand',
            'AT': 'Austria', 'NO': 'Norway', 'AE': 'United Arab Emirates', 'IL': 'Israel',
            'ZA': 'South Africa', 'EG': 'Egypt', 'PH': 'Philippines', 'MY': 'Malaysia',
            'SG': 'Singapore', 'HK': 'Hong Kong', 'DK': 'Denmark', 'FI': 'Finland',
            'CL': 'Chile', 'CO': 'Colombia', 'AR': 'Argentina', 'VN': 'Vietnam',
            'NG': 'Nigeria', 'PK': 'Pakistan', 'BD': 'Bangladesh', 'UA': 'Ukraine',
            'CZ': 'Czech Republic', 'RO': 'Romania', 'NZ': 'New Zealand', 'PT': 'Portugal',
            'GR': 'Greece', 'HU': 'Hungary', 'IE': 'Ireland', 'PE': 'Peru', 'KE': 'Kenya'
        }

        for _ in range(needed):
            city = random.choice(self.cities)
            country_code = city['country']
            country_name = country_names.get(country_code, country_code)

            sample = {
                'task': 'regional_knowledge',
                'subtask': 'country_identification',
                'coordinate_string': f"{city['lat']}, {city['lon']}",
                'ground_truth': {
                    'country': country_name,
                    'country_code': country_code,
                    'city': city['name']
                }
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # PLACE ASSOCIATION
    # =========================================================================
    def generate_place_association_samples(self, current_data, target=1000):
        """Generate place association (city identification) samples with noise levels.

        Noise levels (Gaussian, σ in meters):
        - clean: No noise (original coordinates)
        - 10m: High-quality GPS (smartphone outdoors)
        - 50m: Standard GPS (typical smartphone)
        - 100m: Degraded GPS (urban canyon, indoors)
        - 500m: Poor GPS (heavy interference)
        - 1000m: Very poor GPS (extreme conditions)

        Distribution: ~17% each noise level (6 levels)
        """
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} place association samples with noise levels...")
        new_samples = []

        # Noise levels in meters (0 = clean)
        noise_levels = [0, 10, 50, 100, 500, 1000]
        noise_labels = {
            0: 'clean',
            10: 'gps_10m',
            50: 'gps_50m',
            100: 'gps_100m',
            500: 'gps_500m',
            1000: 'gps_1000m'
        }

        for i in range(needed):
            city = random.choice(self.cities)

            # Rotate through noise levels for balanced distribution
            noise_sigma = noise_levels[i % len(noise_levels)]
            noise_label = noise_labels[noise_sigma]

            # Apply noise
            if noise_sigma == 0:
                # Clean: just add tiny offset for realism (±10m)
                coord_lat = city['lat'] + random.uniform(-0.0001, 0.0001)
                coord_lon = city['lon'] + random.uniform(-0.0001, 0.0001)
                actual_noise_m = 0
            else:
                # Add Gaussian noise
                coord_lat, coord_lon, actual_noise_m = add_gaussian_noise(
                    city['lat'], city['lon'], noise_sigma
                )

            coord_string = f"{coord_lat:.6f}, {coord_lon:.6f}"

            # Determine difficulty based on population
            if city['population'] >= 500000:
                difficulty = "easy"
                precision_level = "high"
            elif city['population'] >= 100000:
                difficulty = "medium"
                precision_level = "medium"
            else:
                difficulty = "hard"
                precision_level = "medium"

            sample = {
                'task': 'place_association',
                'subtask': 'city_identification',
                'coordinate': {
                    'lat': coord_lat,
                    'lon': coord_lon,
                    'precision_level': precision_level,
                    'true_lat': city['lat'],
                    'true_lon': city['lon']
                },
                'coordinate_string': coord_string,
                'ground_truth': {
                    'type': 'city',
                    'country_code': city['country'],
                    'population': city['population'],
                    'alternate_names': city.get('alternate_names', []),
                    'granularity_level': 'city',
                    'location_name': city['name'],
                    'has_noise': noise_sigma > 0,
                    'noise_type': noise_label,
                    'noise_sigma_m': noise_sigma,
                    'actual_noise_m': round(actual_noise_m, 1)
                },
                'difficulty': difficulty,
                'metadata': {
                    'track': 'applied',
                    'track_task_id': 'task1',
                    'track_description': 'Identify places/landmarks from coordinates (reverse geocoding)',
                    'original_task': 'place_association'
                },
                'reasoning_steps': [
                    'Step 1: Parse coordinates',
                    f'  Input: {coord_string}',
                    'Step 2: Search geographic database (e.g., GeoNames)',
                    f'  Query for locations near ({coord_string})',
                    'Step 3: Find closest match',
                    f'  Closest location: {city["name"]}',
                    f'Answer: {city["name"]}'
                ],
                'question': f"What location is at coordinates {coord_string}?"
            }
            new_samples.append(sample)

        return current_data + new_samples

    # =========================================================================
    # NAME DISAMBIGUATION
    # =========================================================================
    def generate_name_disambiguation_samples(self, current_data, target=1000):
        """
        Generate name disambiguation samples.

        IMPORTANT: Adds Gaussian noise (100-500m) to query coordinates to prevent
        exact coordinate matching as a shortcut. The model must use geographic
        knowledge to identify the correct city, not just match coordinates.
        """
        needed = target - len(current_data)
        if needed <= 0:
            return current_data

        print(f"  Generating {needed} name disambiguation samples with noisy coordinates...")

        # Find cities with duplicate names
        name_groups = {}
        for city in self.cities:
            name = city['name']
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(city)

        # Filter to names with 4+ cities
        duplicate_names = {k: v for k, v in name_groups.items() if len(v) >= 4}

        if not duplicate_names:
            print("    Warning: No duplicate city names found")
            return current_data

        new_samples = []

        # Noise levels to cycle through (in meters)
        # Range: 100m to 500m - enough to prevent exact matching but still
        # close enough that the correct city is identifiable by region
        noise_levels = [100, 200, 300, 400, 500]

        for i in range(needed):
            name = random.choice(list(duplicate_names.keys()))
            cities = duplicate_names[name]

            # Select 4 candidates
            candidates = random.sample(cities, min(4, len(cities)))

            # Pick one as the answer
            answer = random.choice(candidates)

            # Add Gaussian noise to the query coordinates
            # This prevents models from just matching the exact coordinates
            noise_meters = noise_levels[i % len(noise_levels)]
            noisy_lat, noisy_lon, actual_noise = add_gaussian_noise(
                answer['lat'], answer['lon'], noise_meters
            )

            # Round to 5 decimal places (about 1m precision)
            noisy_lat = round(noisy_lat, 5)
            noisy_lon = round(noisy_lon, 5)

            candidate_info = []
            for c in candidates:
                candidate_info.append({
                    'name': c['name'],
                    'country': c['country'],
                    'lat': c['lat'],
                    'lon': c['lon'],
                    'population': c['population']
                })

            sample = {
                'task': 'name_disambiguation',
                'subtask': 'city_name_disambiguation',
                'coordinate_string': f"{noisy_lat}, {noisy_lon}",
                'duplicate_name': name,
                'question': f"There are {len(candidates)} cities named '{name}'. Which one is at coordinates {noisy_lat}, {noisy_lon}?",
                'candidates': candidate_info,
                'ground_truth': {
                    'name': answer['name'],
                    'country': answer['country'],
                    'population': answer['population'],
                    'original_coords': {'lat': answer['lat'], 'lon': answer['lon']},
                    'noise_meters': noise_meters,
                    'actual_noise_meters': round(actual_noise, 1)
                }
            }
            new_samples.append(sample)

        return current_data + new_samples


def main():
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data"
    geonames_path = data_path / "raw" / "cities15000.txt"

    print("="*70)
    print("GPSBENCH SAMPLE GENERATION")
    print("="*70)
    print(f"\nTarget: 3400 samples per task (→ ~1000 test samples with 30% split)\n")

    # Initialize generator
    generator = SampleGenerator(geonames_path)

    # Define tasks and their generation functions
    # Only generate the 18 tasks that are actually used in benchmarks
    tasks = [
        # Pure GPS Track (7 tasks)
        ('track_pure_gps/task1_format_conversion.json', generator.generate_format_conversion_samples),
        ('track_pure_gps/task4_distance_calculation.json', generator.generate_distance_samples),
        ('track_pure_gps/task5_bearing_computation.json', generator.generate_bearing_samples),
        ('track_pure_gps/task6_interpolation.json', generator.generate_interpolation_samples),
        ('track_pure_gps/task7_area_perimeter.json', generator.generate_area_perimeter_samples),
        ('track_pure_gps/task8_bounding_box.json', generator.generate_bounding_box_samples),
        ('track_pure_gps/task10_relative_position.json', generator.generate_relative_position_samples),

        # Applied Track (6 tasks - excluding boundary_analysis which has no generator)
        ('track_applied/task1_place_association.json', generator.generate_place_association_samples),
        ('track_applied/task3_name_disambiguation.json', generator.generate_name_disambiguation_samples),
        # Use the Applied-specific generator with closer city pairs (same continent, 200-3000km apart)
        ('track_applied/task5_relative_position.json', generator.generate_relative_position_applied_samples),
        ('track_applied/task6_proximity.json', generator.generate_proximity_samples),
        ('track_applied/task7_route_analysis.json', generator.generate_route_analysis_samples),
        ('track_applied/task9_spatial_patterns.json', generator.generate_spatial_patterns_samples),
    ]

    results = []

    for task_file, generate_fn in tasks:
        filepath = data_path / task_file
        task_name = task_file.split('/')[-1].replace('.json', '')

        print(f"\n{task_name}:")

        if filepath.exists():
            current_data = load_json(filepath)
            print(f"  Current: {len(current_data)} samples")
        else:
            current_data = []
            print(f"  Current: 0 samples (file not found)")

        new_data = generate_fn(current_data, target=3400)

        print(f"  After generation: {len(new_data)} samples")

        save_json(filepath, new_data)
        results.append((task_name, len(current_data), len(new_data)))

    # Summary
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)

    for task_name, before, after in results:
        added = after - before
        status = "✅" if after >= 3400 else f"⚠️ {3400 - after} short"
        print(f"{task_name:45} {before:5} → {after:5} (+{added:4})  {status}")

    total_added = sum(after - before for _, before, after in results)
    print(f"\nTotal samples added: {total_added}")

if __name__ == "__main__":
    main()
