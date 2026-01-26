#!/usr/bin/env python3
"""
Geographic utilities for Layer B dataset generation
Provides land/water classification and country boundary checking
"""

import json
import shapefile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from shapely.geometry import Point, shape
from shapely.prepared import prep

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
NATURAL_EARTH_DIR = DATA_DIR / 'raw' / 'natural_earth'
LAYER_B_DIR = DATA_DIR / 'layer_b'


class GeoUtils:
    """Utility class for geographic operations needed in Layer B"""

    def __init__(self):
        """Initialize and load geographic data"""
        self.countries = None
        self.land_shapes = None
        self.ocean_shapes = None
        self.geonames_places = None
        self.country_metadata = None
        self.curated_landmarks = None

        # Lazy loading - only load when needed
        self._countries_loaded = False
        self._land_loaded = False
        self._ocean_loaded = False
        self._places_loaded = False

    def _load_countries(self):
        """Load country boundaries from Natural Earth shapefile"""
        if self._countries_loaded:
            return

        print("Loading country boundaries...")
        shapefile_path = str(NATURAL_EARTH_DIR / 'ne_10m_admin_0_countries.shp')

        self.countries = []
        with shapefile.Reader(shapefile_path) as sf:
            for shape_record in sf.shapeRecords():
                geom = shape(shape_record.shape.__geo_interface__)
                # Get country attributes from .dbf
                attrs = shape_record.record.as_dict()

                self.countries.append({
                    'geometry': prep(geom),  # Prepared geometry for faster lookups
                    'geometry_raw': geom,    # Keep raw for other operations
                    'name': attrs.get('NAME', ''),
                    'name_long': attrs.get('NAME_LONG', ''),
                    'code': attrs.get('ISO_A2', ''),
                    'code3': attrs.get('ISO_A3', ''),
                    'continent': attrs.get('CONTINENT', ''),
                    'region': attrs.get('REGION_WB', ''),
                    'subregion': attrs.get('SUBREGION', '')
                })

        print(f"✓ Loaded {len(self.countries)} countries")
        self._countries_loaded = True

    def _load_land(self):
        """Load land polygons from Natural Earth"""
        if self._land_loaded:
            return

        print("Loading land polygons...")
        shapefile_path = str(NATURAL_EARTH_DIR / 'ne_10m_land.shp')

        self.land_shapes = []
        with shapefile.Reader(shapefile_path) as sf:
            for shape_record in sf.shapeRecords():
                geom = shape(shape_record.shape.__geo_interface__)
                self.land_shapes.append(prep(geom))

        print(f"✓ Loaded {len(self.land_shapes)} land polygons")
        self._land_loaded = True

    def _load_ocean(self):
        """Load ocean polygons from Natural Earth"""
        if self._ocean_loaded:
            return

        print("Loading ocean polygons...")
        shapefile_path = str(NATURAL_EARTH_DIR / 'ne_10m_ocean.shp')

        self.ocean_shapes = []
        with shapefile.Reader(shapefile_path) as sf:
            for shape_record in sf.shapeRecords():
                geom = shape(shape_record.shape.__geo_interface__)
                self.ocean_shapes.append(prep(geom))

        print(f"✓ Loaded {len(self.ocean_shapes)} ocean polygons")
        self._ocean_loaded = True

    def _load_places(self):
        """Load processed GeoNames places data"""
        if self._places_loaded:
            return

        print("Loading GeoNames places...")
        places_file = LAYER_B_DIR / 'geonames_places.json'
        with open(places_file, 'r', encoding='utf-8') as f:
            self.geonames_places = json.load(f)

        # Load curated landmarks
        landmarks_file = LAYER_B_DIR / 'curated_landmarks.json'
        with open(landmarks_file, 'r', encoding='utf-8') as f:
            self.curated_landmarks = json.load(f)

        # Load country metadata
        country_file = LAYER_B_DIR / 'country_metadata.json'
        with open(country_file, 'r', encoding='utf-8') as f:
            self.country_metadata = json.load(f)

        print(f"✓ Loaded {len(self.geonames_places)} GeoNames places")
        print(f"✓ Loaded {len(self.curated_landmarks)} curated landmarks")
        print(f"✓ Loaded {len(self.country_metadata)} country metadata")
        self._places_loaded = True

    # =========================================================================
    # Task 1: Place Association utilities
    # =========================================================================

    def find_nearest_place(self, lat: float, lon: float, max_distance_km: float = 50) -> Optional[Dict]:
        """
        Find the nearest place to given coordinates

        Args:
            lat: Latitude
            lon: Longitude
            max_distance_km: Maximum search distance in kilometers

        Returns:
            Dictionary with place info, or None if nothing within range
        """
        self._load_places()

        from geopy.distance import geodesic

        point = (lat, lon)
        nearest = None
        min_dist = float('inf')

        # Search through all places
        for place in self.geonames_places:
            place_point = (place['lat'], place['lon'])
            dist = geodesic(point, place_point).kilometers

            if dist < min_dist and dist <= max_distance_km:
                min_dist = dist
                nearest = {
                    **place,
                    'distance_km': dist
                }

        return nearest

    def find_landmark_at_location(self, lat: float, lon: float, tolerance_km: float = 1.0) -> Optional[Dict]:
        """
        Check if there's a famous landmark at or very near these coordinates

        Args:
            lat: Latitude
            lon: Longitude
            tolerance_km: How close coordinates must be (in km)

        Returns:
            Landmark dict if found, None otherwise
        """
        self._load_places()

        from geopy.distance import geodesic

        point = (lat, lon)

        for landmark in self.curated_landmarks:
            landmark_point = (landmark['lat'], landmark['lon'])
            dist = geodesic(point, landmark_point).kilometers

            if dist <= tolerance_km:
                return {
                    **landmark,
                    'distance_km': dist
                }

        return None

    # =========================================================================
    # Task 2: Semantic Validation utilities
    # =========================================================================

    def is_on_land(self, lat: float, lon: float) -> bool:
        """
        Check if coordinates are on land

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            True if on land, False if in ocean
        """
        self._load_land()

        point = Point(lon, lat)  # Note: shapely uses (lon, lat) order

        for land_geom in self.land_shapes:
            if land_geom.contains(point):
                return True

        return False

    def is_in_ocean(self, lat: float, lon: float) -> bool:
        """
        Check if coordinates are in ocean

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            True if in ocean, False if on land
        """
        return not self.is_on_land(lat, lon)

    # =========================================================================
    # Task 3: Regional Knowledge utilities
    # =========================================================================

    def find_country_at_location(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Find which country contains the given coordinates

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Country dict if found, None if in ocean or disputed territory
        """
        self._load_countries()

        point = Point(lon, lat)  # Note: shapely uses (lon, lat) order

        for country in self.countries:
            if country['geometry'].contains(point):
                return {
                    'name': country['name'],
                    'name_long': country['name_long'],
                    'code': country['code'],
                    'code3': country['code3'],
                    'continent': country['continent'],
                    'region': country['region'],
                    'subregion': country['subregion']
                }

        return None

    def is_near_border(self, lat: float, lon: float, tolerance_km: float = 5.0) -> bool:
        """
        Check if coordinates are near a country border

        Args:
            lat: Latitude
            lon: Longitude
            tolerance_km: Distance from border to consider "near" (in km)

        Returns:
            True if near a border, False otherwise
        """
        self._load_countries()

        point = Point(lon, lat)

        # Convert km to degrees (rough approximation: 1 degree ≈ 111 km at equator)
        buffer_deg = tolerance_km / 111.0

        # Check if point is within tolerance of any country border
        for country in self.countries:
            # Get the boundary of the country
            boundary = country['geometry_raw'].boundary

            # Check if point is close to boundary
            if boundary.distance(point) <= buffer_deg:
                return True

        return False

    def get_country_by_code(self, code: str) -> Optional[Dict]:
        """
        Get country metadata by ISO code

        Args:
            code: 2-letter or 3-letter ISO code

        Returns:
            Country metadata dict
        """
        self._load_places()

        code_upper = code.upper()

        for country in self.country_metadata:
            if country['code'] == code_upper or country['code3'] == code_upper:
                return country

        return None

    # =========================================================================
    # Task 4: Precision-based grounding utilities
    # =========================================================================

    def get_appropriate_description_level(self, lat: float, lon: float,
                                          decimal_places: int) -> Dict:
        """
        Determine appropriate level of description based on coordinate precision

        Args:
            lat: Latitude
            lon: Longitude
            decimal_places: Number of decimal places in the coordinates

        Returns:
            Dict with appropriate description levels and entities
        """
        self._load_places()
        self._load_countries()

        # Precision guidelines (at equator):
        # 0 decimals: ~111 km -> country
        # 1 decimal: ~11 km -> region/state
        # 2 decimals: ~1.1 km -> city
        # 3 decimals: ~111 m -> neighborhood/district
        # 4 decimals: ~11 m -> street/building
        # 5+ decimals: ~1 m -> specific landmark

        result = {
            'decimal_places': decimal_places,
            'precision_meters': self._decimal_to_meters(decimal_places),
            'appropriate_levels': [],
            'too_specific': [],
            'too_vague': []
        }

        # Find what's at this location
        country = self.find_country_at_location(lat, lon)
        nearest_place = self.find_nearest_place(lat, lon, max_distance_km=100)
        landmark = self.find_landmark_at_location(lat, lon, tolerance_km=0.5)

        if decimal_places <= 1:
            # Country/region level
            result['appropriate_levels'] = ['country', 'continent', 'large_region']
            if country:
                result['suggestions'] = [country['name'], country['continent']]
            result['too_specific'] = ['city', 'neighborhood', 'landmark']

        elif decimal_places == 2:
            # City level
            result['appropriate_levels'] = ['city', 'region']
            if nearest_place and nearest_place['distance_km'] < 50:
                result['suggestions'] = [nearest_place['name']]
            result['too_specific'] = ['street', 'building', 'landmark']
            result['too_vague'] = ['planet', 'hemisphere']

        elif decimal_places == 3:
            # Neighborhood level
            result['appropriate_levels'] = ['neighborhood', 'district', 'city_area']
            if nearest_place:
                result['suggestions'] = [f"Near {nearest_place['name']}"]
            result['too_specific'] = ['building', 'address']
            result['too_vague'] = ['country', 'continent']

        elif decimal_places >= 4:
            # Specific landmark/building level
            result['appropriate_levels'] = ['landmark', 'building', 'street']
            if landmark:
                result['suggestions'] = [landmark['name']]
            elif nearest_place:
                result['suggestions'] = [nearest_place['name']]
            result['too_vague'] = ['city', 'region', 'country']

        return result

    @staticmethod
    def _decimal_to_meters(decimal_places: int) -> float:
        """Convert decimal places to approximate meters precision (at equator)"""
        # Each decimal place is roughly 1/10 of the previous
        base = 111000  # ~111 km in meters at equator
        return base / (10 ** decimal_places)


# =============================================================================
# Convenience functions for quick access
# =============================================================================

_geo_utils = None


def get_geo_utils() -> GeoUtils:
    """Get singleton instance of GeoUtils"""
    global _geo_utils
    if _geo_utils is None:
        _geo_utils = GeoUtils()
    return _geo_utils


# Quick access functions
def is_on_land(lat: float, lon: float) -> bool:
    """Check if coordinates are on land"""
    return get_geo_utils().is_on_land(lat, lon)


def find_country(lat: float, lon: float) -> Optional[Dict]:
    """Find country at coordinates"""
    return get_geo_utils().find_country_at_location(lat, lon)


def find_nearest_place(lat: float, lon: float, max_distance_km: float = 50) -> Optional[Dict]:
    """Find nearest place to coordinates"""
    return get_geo_utils().find_nearest_place(lat, lon, max_distance_km)


# =============================================================================
# Test/demo code
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Testing GeoUtils")
    print("="*60)

    geo = GeoUtils()

    # Test locations
    test_locations = [
        ("Eiffel Tower", 48.8584, 2.2945),
        ("Middle of Pacific Ocean", 0.0, -160.0),
        ("New York City", 40.7128, -74.0060),
        ("Mount Everest", 27.9881, 86.9250),
    ]

    for name, lat, lon in test_locations:
        print(f"\n{name} ({lat}, {lon}):")
        print(f"  On land: {geo.is_on_land(lat, lon)}")

        country = geo.find_country_at_location(lat, lon)
        if country:
            print(f"  Country: {country['name']} ({country['code']})")
            print(f"  Continent: {country['continent']}")

        place = geo.find_nearest_place(lat, lon, max_distance_km=50)
        if place:
            print(f"  Nearest place: {place['name']} ({place['distance_km']:.1f} km away)")

        landmark = geo.find_landmark_at_location(lat, lon, tolerance_km=1.0)
        if landmark:
            print(f"  Landmark: {landmark['name']}")

    print("\n" + "="*60)
    print("GeoUtils test complete!")
    print("="*60)
