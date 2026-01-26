#!/usr/bin/env python3
"""
Geodetic calculation utilities for Layer C (Spatial Computation & Geometry)
Provides distance, bearing, interpolation, and area calculations
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any
from geopy.distance import geodesic, great_circle


class GeodesicCalculator:
    """
    Utility class for geodetic calculations
    All calculations account for Earth's curvature using geodesic methods
    """

    # Earth constants
    EARTH_RADIUS_KM = 6371.0  # Mean radius in kilometers

    # =========================================================================
    # Task 1: Distance Calculations
    # =========================================================================

    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float,
                          method: str = 'vincenty') -> Dict[str, float]:
        """
        Calculate distance between two points

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            method: 'vincenty' (accurate) or 'haversine' (fast)

        Returns:
            Dictionary with distance in km and miles
        """
        point1 = (lat1, lon1)
        point2 = (lat2, lon2)

        if method == 'vincenty':
            # Vincenty formula (more accurate, accounts for ellipsoid)
            dist = geodesic(point1, point2)
        else:
            # Haversine formula (simpler, assumes sphere)
            dist = great_circle(point1, point2)

        return {
            'distance_km': round(dist.kilometers, 2),
            'distance_miles': round(dist.miles, 2),
            'distance_m': round(dist.meters, 2),
            'method': method
        }

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance using Haversine formula

        Args:
            lat1, lon1: First point
            lat2, lon2: Second point

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return GeodesicCalculator.EARTH_RADIUS_KM * c

    # =========================================================================
    # Task 2: Bearing Calculations
    # =========================================================================

    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> Dict[str, Any]:
        """
        Calculate initial and final bearing between two points

        Args:
            lat1, lon1: Start point
            lat2, lon2: End point

        Returns:
            Dictionary with initial bearing, final bearing, and cardinal direction
        """
        # Convert to radians
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

        # Calculate initial bearing
        dlon = lon2_rad - lon1_rad
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)

        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        initial_bearing = (initial_bearing + 360) % 360  # Normalize to 0-360

        # Calculate final bearing (reverse bearing from end point)
        dlon_rev = lon1_rad - lon2_rad
        x_rev = math.sin(dlon_rev) * math.cos(lat1_rad)
        y_rev = math.cos(lat2_rad) * math.sin(lat1_rad) - \
                math.sin(lat2_rad) * math.cos(lat1_rad) * math.cos(dlon_rev)

        final_bearing = math.atan2(x_rev, y_rev)
        final_bearing = math.degrees(final_bearing)
        final_bearing = (final_bearing + 180) % 360  # Reverse and normalize

        return {
            'initial_bearing_deg': round(initial_bearing, 2),
            'final_bearing_deg': round(final_bearing, 2),
            'cardinal_direction': GeodesicCalculator.bearing_to_cardinal(initial_bearing),
            'intercardinal_direction': GeodesicCalculator.bearing_to_intercardinal(initial_bearing)
        }

    @staticmethod
    def bearing_to_cardinal(bearing: float) -> str:
        """Convert bearing to cardinal direction (N, S, E, W)"""
        # Normalize to 0-360
        bearing = bearing % 360

        if 337.5 <= bearing or bearing < 22.5:
            return 'N'
        elif 22.5 <= bearing < 67.5:
            return 'NE'
        elif 67.5 <= bearing < 112.5:
            return 'E'
        elif 112.5 <= bearing < 157.5:
            return 'SE'
        elif 157.5 <= bearing < 202.5:
            return 'S'
        elif 202.5 <= bearing < 247.5:
            return 'SW'
        elif 247.5 <= bearing < 292.5:
            return 'W'
        else:  # 292.5 <= bearing < 337.5
            return 'NW'

    @staticmethod
    def bearing_to_intercardinal(bearing: float) -> str:
        """Convert bearing to 16-point compass direction"""
        bearing = bearing % 360

        directions = [
            'N', 'NNE', 'NE', 'ENE',
            'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW',
            'W', 'WNW', 'NW', 'NNW'
        ]

        index = round(bearing / 22.5) % 16
        return directions[index]

    # =========================================================================
    # Task 3: Coordinate Interpolation
    # =========================================================================

    @staticmethod
    def interpolate_great_circle(lat1: float, lon1: float, lat2: float, lon2: float,
                                 fraction: float) -> Dict[str, float]:
        """
        Interpolate point along great circle path

        Args:
            lat1, lon1: Start point
            lat2, lon2: End point
            fraction: Fraction along path (0.0 to 1.0)

        Returns:
            Dictionary with interpolated lat/lon
        """
        # Convert to radians
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

        # Calculate angular distance
        d = 2 * math.asin(math.sqrt(
            math.sin((lat2_rad - lat1_rad) / 2)**2 +
            math.cos(lat1_rad) * math.cos(lat2_rad) *
            math.sin((lon2_rad - lon1_rad) / 2)**2
        ))

        if d == 0:
            # Same point
            return {'lat': lat1, 'lon': lon1}

        # Interpolation
        a = math.sin((1 - fraction) * d) / math.sin(d)
        b = math.sin(fraction * d) / math.sin(d)

        x = a * math.cos(lat1_rad) * math.cos(lon1_rad) + \
            b * math.cos(lat2_rad) * math.cos(lon2_rad)
        y = a * math.cos(lat1_rad) * math.sin(lon1_rad) + \
            b * math.cos(lat2_rad) * math.sin(lon2_rad)
        z = a * math.sin(lat1_rad) + b * math.sin(lat2_rad)

        lat_interp = math.atan2(z, math.sqrt(x**2 + y**2))
        lon_interp = math.atan2(y, x)

        return {
            'lat': round(math.degrees(lat_interp), 6),
            'lon': round(math.degrees(lon_interp), 6)
        }

    @staticmethod
    def calculate_midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> Dict[str, float]:
        """
        Calculate midpoint between two coordinates

        Args:
            lat1, lon1: First point
            lat2, lon2: Second point

        Returns:
            Dictionary with midpoint lat/lon
        """
        return GeodesicCalculator.interpolate_great_circle(lat1, lon1, lat2, lon2, 0.5)

    @staticmethod
    def divide_path(lat1: float, lon1: float, lat2: float, lon2: float,
                    num_segments: int) -> List[Dict[str, float]]:
        """
        Divide path into equal segments

        Args:
            lat1, lon1: Start point
            lat2, lon2: End point
            num_segments: Number of segments

        Returns:
            List of intermediate points (including start and end)
        """
        points = []
        for i in range(num_segments + 1):
            fraction = i / num_segments
            point = GeodesicCalculator.interpolate_great_circle(
                lat1, lon1, lat2, lon2, fraction
            )
            points.append(point)

        return points

    # =========================================================================
    # Task 4: Area & Perimeter Calculations
    # =========================================================================

    @staticmethod
    def calculate_polygon_area(vertices: List[Dict[str, float]]) -> float:
        """
        Calculate area of polygon on sphere (spherical excess method)

        Args:
            vertices: List of dicts with 'lat' and 'lon' keys

        Returns:
            Area in square kilometers
        """
        if len(vertices) < 3:
            return 0.0

        # Convert to radians
        coords_rad = [(math.radians(v['lat']), math.radians(v['lon']))
                     for v in vertices]

        # Close the polygon
        if coords_rad[0] != coords_rad[-1]:
            coords_rad.append(coords_rad[0])

        # Calculate area using spherical excess formula
        total = 0.0
        for i in range(len(coords_rad) - 1):
            lat1, lon1 = coords_rad[i]
            lat2, lon2 = coords_rad[i + 1]

            total += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))

        area_rad = abs(total) / 2.0

        # Convert to km²
        earth_radius_km = GeodesicCalculator.EARTH_RADIUS_KM
        area_km2 = area_rad * earth_radius_km**2

        return round(area_km2, 2)

    @staticmethod
    def calculate_polygon_perimeter(vertices: List[Dict[str, float]]) -> float:
        """
        Calculate perimeter of polygon

        Args:
            vertices: List of dicts with 'lat' and 'lon' keys

        Returns:
            Perimeter in kilometers
        """
        if len(vertices) < 2:
            return 0.0

        perimeter = 0.0

        # Calculate distance between consecutive vertices
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]  # Wrap around to first vertex

            dist = GeodesicCalculator.haversine_distance(
                v1['lat'], v1['lon'],
                v2['lat'], v2['lon']
            )
            perimeter += dist

        return round(perimeter, 2)

    # =========================================================================
    # Task 5: Bounding Box & Spatial Extent
    # =========================================================================

    @staticmethod
    def calculate_bounding_box(points: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate bounding box for set of points

        Args:
            points: List of dicts with 'lat' and 'lon' keys

        Returns:
            Dictionary with min/max lat/lon, center, and diagonal distance
        """
        if not points:
            return {}

        lats = [p['lat'] for p in points]
        lons = [p['lon'] for p in points]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Calculate center (simple average)
        center_lat = round(sum(lats) / len(lats), 6)
        center_lon = round(sum(lons) / len(lons), 6)

        # Calculate diagonal distance (from SW corner to NE corner)
        diagonal_dist = GeodesicCalculator.haversine_distance(
            min_lat, min_lon,
            max_lat, max_lon
        )

        return {
            'min_lat': round(min_lat, 6),
            'max_lat': round(max_lat, 6),
            'min_lon': round(min_lon, 6),
            'max_lon': round(max_lon, 6),
            'center': {'lat': center_lat, 'lon': center_lon},
            'diagonal_km': round(diagonal_dist, 2),
            'width_deg': round(max_lon - min_lon, 6),
            'height_deg': round(max_lat - min_lat, 6)
        }

    @staticmethod
    def calculate_centroid(points: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate centroid (geometric center) of point set

        Args:
            points: List of dicts with 'lat' and 'lon' keys

        Returns:
            Dictionary with centroid lat/lon
        """
        if not points:
            return {'lat': 0.0, 'lon': 0.0}

        # Convert to Cartesian coordinates
        x_sum, y_sum, z_sum = 0.0, 0.0, 0.0

        for p in points:
            lat_rad = math.radians(p['lat'])
            lon_rad = math.radians(p['lon'])

            x_sum += math.cos(lat_rad) * math.cos(lon_rad)
            y_sum += math.cos(lat_rad) * math.sin(lon_rad)
            z_sum += math.sin(lat_rad)

        n = len(points)
        x_avg = x_sum / n
        y_avg = y_sum / n
        z_avg = z_sum / n

        # Convert back to lat/lon
        lon_avg = math.atan2(y_avg, x_avg)
        hyp = math.sqrt(x_avg**2 + y_avg**2)
        lat_avg = math.atan2(z_avg, hyp)

        return {
            'lat': round(math.degrees(lat_avg), 6),
            'lon': round(math.degrees(lon_avg), 6)
        }


# =============================================================================
# Convenience functions
# =============================================================================

def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Quick distance calculation in km"""
    return GeodesicCalculator.calculate_distance(lat1, lon1, lat2, lon2)['distance_km']


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Quick bearing calculation in degrees"""
    return GeodesicCalculator.calculate_bearing(lat1, lon1, lat2, lon2)['initial_bearing_deg']


def midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple:
    """Quick midpoint calculation"""
    mp = GeodesicCalculator.calculate_midpoint(lat1, lon1, lat2, lon2)
    return (mp['lat'], mp['lon'])


# =============================================================================
# Test/demo code
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Testing GeodesicCalculator")
    print("="*60)

    calc = GeodesicCalculator()

    # Test distance
    print("\n1. Distance Calculation")
    print("-"*60)
    nyc = (40.7128, -74.0060)
    london = (51.5074, -0.1278)

    dist = calc.calculate_distance(*nyc, *london)
    print(f"NYC to London: {dist['distance_km']} km ({dist['distance_miles']} miles)")

    # Test bearing
    print("\n2. Bearing Calculation")
    print("-"*60)
    bear = calc.calculate_bearing(*nyc, *london)
    print(f"Initial bearing: {bear['initial_bearing_deg']}° ({bear['cardinal_direction']})")
    print(f"Final bearing: {bear['final_bearing_deg']}°")

    # Test interpolation
    print("\n3. Interpolation")
    print("-"*60)
    mid = calc.calculate_midpoint(*nyc, *london)
    print(f"Midpoint: {mid['lat']}, {mid['lon']}")

    quarter = calc.interpolate_great_circle(*nyc, *london, 0.25)
    print(f"Quarter point: {quarter['lat']}, {quarter['lon']}")

    # Test area
    print("\n4. Polygon Area")
    print("-"*60)
    triangle = [
        {'lat': 40.7128, 'lon': -74.0060},  # NYC
        {'lat': 40.7589, 'lon': -73.9851},  # Times Square
        {'lat': 40.7484, 'lon': -73.9857}   # Empire State
    ]
    area = calc.calculate_polygon_area(triangle)
    perimeter = calc.calculate_polygon_perimeter(triangle)
    print(f"Triangle area: {area} km²")
    print(f"Triangle perimeter: {perimeter} km")

    # Test bounding box
    print("\n5. Bounding Box")
    print("-"*60)
    cities = [
        {'lat': 40.7128, 'lon': -74.0060, 'name': 'NYC'},
        {'lat': 34.0522, 'lon': -118.2437, 'name': 'LA'},
        {'lat': 41.8781, 'lon': -87.6298, 'name': 'Chicago'}
    ]
    bbox = calc.calculate_bounding_box(cities)
    print(f"Min lat/lon: {bbox['min_lat']}, {bbox['min_lon']}")
    print(f"Max lat/lon: {bbox['max_lat']}, {bbox['max_lon']}")
    print(f"Center: {bbox['center']}")
    print(f"Diagonal: {bbox['diagonal_km']} km")

    print("\n" + "="*60)
    print("GeodesicCalculator test complete!")
    print("="*60)
