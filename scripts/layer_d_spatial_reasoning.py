"""
Spatial Reasoning Utilities for GPSBench Layer D

Provides functions for:
- Relative position analysis (north/south/east/west)
- Proximity calculations (nearest, farthest, within radius)
- Route analysis (waypoints, detours, path optimization)
- Boundary crossing detection
- Spatial pattern recognition (clustering, outliers)
"""

import math
from typing import List, Dict, Tuple, Optional, Any
from geopy.distance import geodesic, great_circle
import numpy as np


class SpatialReasoning:
    """Utilities for spatial reasoning and relationship analysis"""

    @staticmethod
    def is_north_of(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
        """Check if point1 is north of point2"""
        return lat1 > lat2

    @staticmethod
    def is_south_of(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
        """Check if point1 is south of point2"""
        return lat1 < lat2

    @staticmethod
    def is_east_of(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
        """
        Check if point1 is east of point2
        Note: Handles date line crossing
        """
        # Normalize longitudes to -180 to 180
        lon1_norm = ((lon1 + 180) % 360) - 180
        lon2_norm = ((lon2 + 180) % 360) - 180
        
        # Calculate difference accounting for date line
        diff = lon1_norm - lon2_norm
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
            
        return diff > 0

    @staticmethod
    def is_west_of(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
        """Check if point1 is west of point2"""
        return not SpatialReasoning.is_east_of(lat1, lon1, lat2, lon2)

    @staticmethod
    def get_cardinal_direction(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
        """
        Get primary cardinal direction from point1 to point2
        Returns: N, NE, E, SE, S, SW, W, NW
        """
        # Calculate bearing
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        x = math.sin(dlon_rad) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
        
        bearing = math.degrees(math.atan2(x, y))
        bearing = (bearing + 360) % 360
        
        # Convert to cardinal direction
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = int((bearing + 22.5) / 45) % 8
        return directions[index]

    @staticmethod
    def find_nearest_point(
        reference: Dict[str, float],
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Find the nearest point to reference from candidates
        
        Args:
            reference: Dict with 'lat', 'lon'
            candidates: List of dicts with 'lat', 'lon', 'name'
        
        Returns:
            Nearest candidate with distance added
        """
        ref_point = (reference['lat'], reference['lon'])
        min_distance = float('inf')
        nearest = None
        
        for candidate in candidates:
            cand_point = (candidate['lat'], candidate['lon'])
            distance = geodesic(ref_point, cand_point).kilometers
            
            if distance < min_distance:
                min_distance = distance
                nearest = candidate.copy()
                nearest['distance_km'] = distance
        
        return nearest

    @staticmethod
    def find_farthest_point(
        reference: Dict[str, float],
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find the farthest point from reference"""
        ref_point = (reference['lat'], reference['lon'])
        max_distance = 0.0
        farthest = None
        
        for candidate in candidates:
            cand_point = (candidate['lat'], candidate['lon'])
            distance = geodesic(ref_point, cand_point).kilometers
            
            if distance > max_distance:
                max_distance = distance
                farthest = candidate.copy()
                farthest['distance_km'] = distance
        
        return farthest

    @staticmethod
    def points_within_radius(
        reference: Dict[str, float],
        candidates: List[Dict[str, Any]],
        radius_km: float
    ) -> List[Dict[str, Any]]:
        """Find all points within radius of reference"""
        ref_point = (reference['lat'], reference['lon'])
        within_radius = []
        
        for candidate in candidates:
            cand_point = (candidate['lat'], candidate['lon'])
            distance = geodesic(ref_point, cand_point).kilometers
            
            if distance <= radius_km:
                result = candidate.copy()
                result['distance_km'] = distance
                within_radius.append(result)
        
        return within_radius

    @staticmethod
    def rank_by_distance(
        reference: Dict[str, float],
        candidates: List[Dict[str, Any]],
        ascending: bool = True
    ) -> List[Dict[str, Any]]:
        """Rank candidates by distance from reference"""
        ref_point = (reference['lat'], reference['lon'])
        ranked = []
        
        for candidate in candidates:
            cand_point = (candidate['lat'], candidate['lon'])
            distance = geodesic(ref_point, cand_point).kilometers
            
            result = candidate.copy()
            result['distance_km'] = distance
            ranked.append(result)
        
        ranked.sort(key=lambda x: x['distance_km'], reverse=not ascending)
        return ranked

    @staticmethod
    def is_on_route(
        waypoint: Dict[str, float],
        start: Dict[str, float],
        end: Dict[str, float],
        tolerance_km: float = 200.0
    ) -> Dict[str, Any]:
        """
        Check if waypoint is on the great circle path from start to end
        
        Returns:
            Dict with is_on_route, perpendicular_distance_km, detour_km, detour_percent
        """
        # Calculate direct distance and distance via waypoint
        start_point = (start['lat'], start['lon'])
        end_point = (end['lat'], end['lon'])
        waypoint_point = (waypoint['lat'], waypoint['lon'])
        
        direct_distance = geodesic(start_point, end_point).kilometers
        via_waypoint = (
            geodesic(start_point, waypoint_point).kilometers +
            geodesic(waypoint_point, end_point).kilometers
        )
        
        # Calculate perpendicular distance to path (approximation)
        detour_km = via_waypoint - direct_distance
        detour_percent = (detour_km / direct_distance * 100) if direct_distance > 0 else 0
        
        # Simple heuristic: if detour is small, waypoint is on route
        is_on_route = detour_km <= tolerance_km
        
        return {
            'is_on_route': is_on_route,
            'direct_distance_km': direct_distance,
            'via_waypoint_km': via_waypoint,
            'detour_km': detour_km,
            'detour_percent': detour_percent,
            'tolerance_km': tolerance_km
        }

    @staticmethod
    def order_points_by_latitude(
        points: List[Dict[str, Any]],
        ascending: bool = True
    ) -> List[Dict[str, Any]]:
        """Order points from south to north (or reverse)"""
        sorted_points = sorted(points, key=lambda p: p['lat'], reverse=not ascending)
        return sorted_points

    @staticmethod
    def order_points_by_longitude(
        points: List[Dict[str, Any]],
        ascending: bool = True
    ) -> List[Dict[str, Any]]:
        """Order points from west to east (or reverse)"""
        sorted_points = sorted(points, key=lambda p: p['lon'], reverse=not ascending)
        return sorted_points

    @staticmethod
    def is_between(
        point: Dict[str, float],
        start: Dict[str, float],
        end: Dict[str, float],
        tolerance_km: float = 100.0
    ) -> Dict[str, Any]:
        """
        Check if point is between start and end
        
        Returns:
            Dict with is_between, distance_from_path_km
        """
        # Check if point is roughly on the line segment
        result = SpatialReasoning.is_on_route(point, start, end, tolerance_km)
        
        # Additional check: is point actually between start and end?
        start_point = (start['lat'], start['lon'])
        end_point = (end['lat'], end['lon'])
        point_coords = (point['lat'], point['lon'])
        
        dist_start_to_point = geodesic(start_point, point_coords).kilometers
        dist_point_to_end = geodesic(point_coords, end_point).kilometers
        dist_start_to_end = geodesic(start_point, end_point).kilometers
        
        # If point is between, sum of distances should be close to total
        is_between = (
            abs((dist_start_to_point + dist_point_to_end) - dist_start_to_end) <= tolerance_km
        )
        
        return {
            'is_between': is_between,
            'distance_start_to_point_km': dist_start_to_point,
            'distance_point_to_end_km': dist_point_to_end,
            'distance_start_to_end_km': dist_start_to_end,
            'on_route_check': result['is_on_route'],
            'detour_km': result['detour_km']
        }

    @staticmethod
    def detect_outlier(points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect the geographic outlier in a set of points
        Uses distance to centroid as metric
        
        Returns:
            Dict with outlier name and distances
        """
        if len(points) < 3:
            return {'outlier': None, 'reason': 'Need at least 3 points'}
        
        # Calculate centroid
        mean_lat = np.mean([p['lat'] for p in points])
        mean_lon = np.mean([p['lon'] for p in points])
        centroid = (mean_lat, mean_lon)
        
        # Calculate distances to centroid
        distances = {}
        for point in points:
            point_coords = (point['lat'], point['lon'])
            dist = geodesic(centroid, point_coords).kilometers
            distances[point['name']] = dist
        
        # Find outlier (farthest from centroid)
        outlier_name = max(distances, key=distances.get)
        outlier_distance = distances[outlier_name]
        
        # Calculate median distance for comparison
        median_distance = np.median(list(distances.values()))
        
        return {
            'outlier': outlier_name,
            'outlier_distance_km': outlier_distance,
            'median_distance_km': median_distance,
            'distances_to_centroid': distances,
            'centroid': {'lat': mean_lat, 'lon': mean_lon}
        }

    @staticmethod
    def identify_clusters(
        points: List[Dict[str, Any]],
        threshold_km: float = 100.0
    ) -> List[List[str]]:
        """
        Simple clustering: group points within threshold distance
        Uses greedy single-linkage clustering
        
        Returns:
            List of clusters (each cluster is list of point names)
        """
        n = len(points)
        clusters = [[i] for i in range(n)]  # Start with each point in own cluster
        
        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                point_i = (points[i]['lat'], points[i]['lon'])
                point_j = (points[j]['lat'], points[j]['lon'])
                dist = geodesic(point_i, point_j).kilometers
                distances[i][j] = dist
                distances[j][i] = dist
        
        # Merge clusters within threshold
        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Check if any pair of points from different clusters is close
                    min_dist = float('inf')
                    for idx_i in clusters[i]:
                        for idx_j in clusters[j]:
                            if distances[idx_i][idx_j] < min_dist:
                                min_dist = distances[idx_i][idx_j]
                    
                    if min_dist <= threshold_km:
                        # Merge clusters
                        clusters[i].extend(clusters[j])
                        clusters.pop(j)
                        merged = True
                        break
                if merged:
                    break
        
        # Convert indices to names
        result_clusters = []
        for cluster in clusters:
            result_clusters.append([points[idx]['name'] for idx in cluster])
        
        return result_clusters

    @staticmethod
    def calculate_multi_leg_distance(points: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate total distance for a multi-leg route
        
        Args:
            points: List of waypoints in order
        
        Returns:
            Dict with total_distance_km and leg_distances
        """
        if len(points) < 2:
            return {'total_distance_km': 0.0, 'leg_distances': []}
        
        leg_distances = []
        total_distance = 0.0
        
        for i in range(len(points) - 1):
            point_a = (points[i]['lat'], points[i]['lon'])
            point_b = (points[i + 1]['lat'], points[i + 1]['lon'])
            distance = geodesic(point_a, point_b).kilometers
            leg_distances.append({
                'from': points[i].get('name', f'Point {i}'),
                'to': points[i + 1].get('name', f'Point {i+1}'),
                'distance_km': distance
            })
            total_distance += distance
        
        return {
            'total_distance_km': total_distance,
            'num_legs': len(leg_distances),
            'leg_distances': leg_distances
        }

    @staticmethod
    def greedy_tsp_order(
        points: List[Dict[str, Any]],
        start_index: int = 0
    ) -> Dict[str, Any]:
        """
        Greedy nearest neighbor TSP approximation
        
        Returns:
            Dict with optimized order and total distance
        """
        if len(points) <= 1:
            return {'order': [p['name'] for p in points], 'total_distance_km': 0.0}
        
        unvisited = list(range(len(points)))
        current = start_index
        unvisited.remove(current)
        
        route = [current]
        total_distance = 0.0
        
        while unvisited:
            # Find nearest unvisited point
            current_point = (points[current]['lat'], points[current]['lon'])
            min_dist = float('inf')
            nearest = None
            
            for idx in unvisited:
                next_point = (points[idx]['lat'], points[idx]['lon'])
                dist = geodesic(current_point, next_point).kilometers
                if dist < min_dist:
                    min_dist = dist
                    nearest = idx
            
            route.append(nearest)
            total_distance += min_dist
            current = nearest
            unvisited.remove(nearest)
        
        ordered_names = [points[idx]['name'] for idx in route]
        
        return {
            'order': ordered_names,
            'total_distance_km': total_distance,
            'route_indices': route
        }


# Convenience functions
def is_north_of(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
    return SpatialReasoning.is_north_of(lat1, lon1, lat2, lon2)

def is_east_of(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
    return SpatialReasoning.is_east_of(lat1, lon1, lat2, lon2)

def get_cardinal_direction(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    return SpatialReasoning.get_cardinal_direction(lat1, lon1, lat2, lon2)
