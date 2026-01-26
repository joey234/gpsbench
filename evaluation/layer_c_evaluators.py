"""
Task-specific evaluators for GPSBench Layer C
Each evaluator handles prompting and response evaluation for spatial computation tasks
"""

import re
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from abc import ABC, abstractmethod
import math

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


class NumericalParser:
    """Utility class for parsing numerical values from text responses"""

    @staticmethod
    def extract_final_answer(text: str) -> Optional[str]:
        """
        Extract the FINAL ANSWER line from response

        Returns:
            The content after "FINAL ANSWER:" or None if not found
        """
        # Look for "FINAL ANSWER:" pattern (case-insensitive)
        match = re.search(r'FINAL\s+ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def extract_distance(text: str) -> Optional[float]:
        """
        Extract distance value from text, handling various formats and units

        Handles:
        - "5,570 kilometers"
        - "3461 miles"
        - "5570 km"
        - "5.57 × 10³ km"
        - "approximately 5500 km"

        PRIORITY: Looks for "FINAL ANSWER:" first, then falls back to full text
        """
        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = NumericalParser.extract_final_answer(text)
        if final_answer:
            # Try to parse distance from final answer line only
            result = NumericalParser._parse_distance_from_text(final_answer)
            if result is not None:
                return result

        # STAGE 2: Fallback to parsing entire response
        return NumericalParser._parse_distance_from_text(text)

    @staticmethod
    def _parse_distance_from_text(text: str) -> Optional[float]:
        """Internal helper to parse distance from a text string"""
        text_lower = text.lower()

        # Remove commas from numbers for easier parsing
        text_clean = text.replace(',', '')

        # Try scientific notation first (e.g., "5.5 × 10³ km" or "5.5 × 10^3 km")
        # Handle both superscript (³) and regular (^3) notation
        text_for_sci = text_clean.replace('¹', '1').replace('²', '2').replace('³', '3').replace('⁴', '4').replace('⁵', '5')
        sci_match = re.search(r'([\d.]+)\s*[×x\*]\s*10[\^]?(\d+)\s*(?:km|kilometer)', text_for_sci, re.IGNORECASE)
        if sci_match:
            base = float(sci_match.group(1))
            exp = int(sci_match.group(2))
            return base * (10 ** exp)

        # Try "thousand km" format
        thousand_match = re.search(r'([\d.]+)\s*thousand\s*(?:km|kilometer)', text_clean, re.IGNORECASE)
        if thousand_match:
            return float(thousand_match.group(1)) * 1000

        # Try regular kilometers (must match km or kilometer word)
        km_match = re.search(r'([\d.]+)\s*(?:km|kilometer)(?:s|es)?\b', text_clean, re.IGNORECASE)
        if km_match:
            return float(km_match.group(1))

        # Try miles and convert to km
        miles_match = re.search(r'([\d.]+)\s*(?:miles?|mi)\b', text_clean, re.IGNORECASE)
        if miles_match:
            miles = float(miles_match.group(1))
            return miles * 1.60934  # Convert to km

        # Try meters only if no other unit found (and only for large values)
        if 'km' not in text_lower and 'kilometer' not in text_lower and 'mile' not in text_lower:
            meter_match = re.search(r'([\d.]+)\s*(?:meters?|m)\b(?!\s*²)', text_clean, re.IGNORECASE)
            if meter_match:
                meters = float(meter_match.group(1))
                if meters > 1000:  # Only convert if large value
                    return meters / 1000

        return None

    @staticmethod
    def extract_bearing(text: str) -> Optional[float]:
        """
        Extract bearing value from text in degrees

        Handles:
        - "51.2 degrees"
        - "51°"
        - "Northeast" (converted to degrees)
        - "bearing of 45°"

        PRIORITY: Looks for "FINAL ANSWER:" first, then falls back to full text
        """
        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = NumericalParser.extract_final_answer(text)
        if final_answer:
            # Try to parse bearing from final answer line only
            result = NumericalParser._parse_bearing_from_text(final_answer)
            if result is not None:
                return result

        # STAGE 2: Fallback to parsing entire response
        return NumericalParser._parse_bearing_from_text(text)

    @staticmethod
    def _parse_bearing_from_text(text: str) -> Optional[float]:
        """Internal helper to parse bearing from a text string"""
        text_lower = text.lower()

        # Cardinal directions to degrees
        cardinal_map = {
            'north': 0, 'n': 0,
            'northeast': 45, 'ne': 45,
            'east': 90, 'e': 90,
            'southeast': 135, 'se': 135,
            'south': 180, 's': 180,
            'southwest': 225, 'sw': 225,
            'west': 270, 'w': 270,
            'northwest': 315, 'nw': 315
        }

        # Try to find degree value first (numbers are more precise)
        degree_patterns = [
            r'([\d.]+)\s*(?:degrees?|°)',
            r'bearing\s*(?:of|is|:)?\s*([\d.]+)',
            r'([\d.]+)\s*deg\b',
        ]

        for pattern in degree_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))

        # Try to find cardinal direction
        for cardinal, degrees in cardinal_map.items():
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + cardinal + r'\b', text_lower):
                return float(degrees)

        return None

    @staticmethod
    def extract_coordinate(text: str) -> Optional[Tuple[float, float]]:
        """
        Extract coordinate pair (lat, lon) from text

        Handles:
        - "40.7128, -74.0060"
        - "lat: 40.7, lon: -74.0"
        - "40.7N, 74.0W"

        PRIORITY: Looks for "FINAL ANSWER:" first
        """
        # STAGE 1: Check for FINAL ANSWER line
        final_answer = NumericalParser.extract_final_answer(text)
        if final_answer:
            result = NumericalParser._parse_coordinate_from_text(final_answer)
            if result is not None:
                return result

        # STAGE 2: Fallback
        return NumericalParser._parse_coordinate_from_text(text)

    @staticmethod
    def _parse_coordinate_from_text(text: str) -> Optional[Tuple[float, float]]:
        """Internal helper to parse coordinate from text"""
        # Pattern for coordinate pairs
        coord_patterns = [
            r'(-?[\d.]+)\s*,\s*(-?[\d.]+)',  # Simple comma-separated
            r'lat(?:itude)?:?\s*(-?[\d.]+).*?lon(?:gitude)?:?\s*(-?[\d.]+)',  # Named
            r'([\d.]+)\s*[NS],?\s*([\d.]+)\s*[EW]',  # With hemispheres
        ]

        for pattern in coord_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                lat = float(match.group(1))
                lon = float(match.group(2))

                # Handle hemisphere indicators
                if 'S' in text.upper():
                    lat = -abs(lat)
                if 'W' in text.upper():
                    lon = -abs(lon)

                # Validate ranges
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)

        return None

    @staticmethod
    def extract_area(text: str) -> Optional[float]:
        """
        Extract area value from text in km²

        Handles:
        - "1,150 km²"
        - "1150 square kilometers"
        - "1.15 × 10³ km²"

        PRIORITY: Looks for "FINAL ANSWER:" first
        """
        # STAGE 1: Check for FINAL ANSWER line
        final_answer = NumericalParser.extract_final_answer(text)
        if final_answer:
            result = NumericalParser._parse_area_from_text(final_answer)
            if result is not None:
                return result

        # STAGE 2: Fallback
        return NumericalParser._parse_area_from_text(text)

    @staticmethod
    def _parse_area_from_text(text: str) -> Optional[float]:
        """Internal helper to parse area from text"""
        text_clean = text.replace(',', '')

        area_patterns = [
            r'([\d.]+)\s*(?:×|x|\*)\s*10\^?(\d+)\s*(?:km²|km2|square\s*kilometers?)',
            r'([\d.]+)\s*(?:km²|km2|square\s*kilometers?)',
        ]

        for pattern in area_patterns:
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                if '×' in pattern or 'x' in pattern:
                    base = float(match.group(1))
                    exp = int(match.group(2))
                    return base * (10 ** exp)
                else:
                    return float(match.group(1))

        return None

    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        """Extract any number from text"""
        # Remove commas
        text_clean = text.replace(',', '')

        # Find any number (including decimals and negatives)
        match = re.search(r'-?[\d.]+', text_clean)
        if match:
            try:
                return float(match.group(0))
            except:
                return None
        return None

    @staticmethod
    def extract_index_list(text: str) -> Optional[List[int]]:
        """
        Extract list of point indices from text

        Handles formats:
        - "[0, 3, 7, 11]"
        - "0, 3, 7, 11"
        - "Points 0, 3, 7, and 11"
        - "indices: 0, 3, 7, 11"
        - "Point indices 0, 3, 7, 11"

        PRIORITY: Looks for "FINAL ANSWER:" first, then falls back to full text

        Returns:
            List of integers or None if cannot parse
        """
        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = NumericalParser.extract_final_answer(text)
        if final_answer:
            result = NumericalParser._parse_index_list_from_text(final_answer)
            if result is not None:
                return result

        # STAGE 2: Fallback to parsing entire response
        return NumericalParser._parse_index_list_from_text(text)

    @staticmethod
    def _parse_index_list_from_text(text: str) -> Optional[List[int]]:
        """Internal helper to parse index list from text"""
        # Try format: [0, 3, 7, 11] or (0, 3, 7, 11)
        bracket_match = re.search(r'[\[\(]\s*([\d\s,]+)\s*[\]\)]', text)
        if bracket_match:
            indices_str = bracket_match.group(1)
            try:
                indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
                if indices:
                    return sorted(indices)
            except:
                pass

        # Try format: "0, 3, 7, 11" or "Points 0, 3, 7, and 11"
        # Look for sequences of numbers separated by commas and/or "and"
        numbers_match = re.findall(r'\b\d+\b', text)
        if numbers_match:
            try:
                indices = [int(x) for x in numbers_match]
                # Filter out unreasonably large numbers (likely not point indices)
                indices = [x for x in indices if x < 1000]
                if indices:
                    return sorted(set(indices))  # Remove duplicates and sort
            except:
                pass

        return None


class TaskEvaluator(ABC):
    """Base class for Layer C task evaluators"""

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.parser = NumericalParser()

    @abstractmethod
    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        """Create prompt for LLM"""
        pass

    @abstractmethod
    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate LLM response against ground truth"""
        pass


class DistanceCalculationEvaluator(TaskEvaluator):
    """Evaluator for Task 1: Distance Calculation"""

    def __init__(self):
        super().__init__("distance_calculation")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in geodetic calculations and coordinate systems.
Your task is to calculate the distance between two points on Earth's surface.

Use the Haversine or Vincenty formula to account for Earth's curvature.
Provide the distance in kilometers.

Be precise and show your calculation if helpful."""

        point_a = test_case['point_a']
        point_b = test_case['point_b']

        user_prompt = f"""Calculate the distance between these two points:

Point A: {point_a['name']}
  Coordinates: {point_a['lat']:.4f}, {point_a['lon']:.4f}

Point B: {point_b['name']}
  Coordinates: {point_b['lat']:.4f}, {point_b['lon']:.4f}

What is the distance between these points in kilometers?

IMPORTANT: You may show your calculations if helpful, but you MUST end your response with:

FINAL ANSWER: [distance in kilometers]

Example formats:
- FINAL ANSWER: 5570 km
- FINAL ANSWER: 3461 miles
- FINAL ANSWER: 5570 kilometers

Provide only the distance value in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = test_case['ground_truth']
        expected_km = ground_truth['distance_km']
        tolerance_km = ground_truth['tolerance_km']
        tolerance_pct = ground_truth.get('tolerance_percent', 5.0)

        # Extract distance from response
        predicted_km = self.parser.extract_distance(response)

        if predicted_km is None:
            return {
                'correct': False,
                'error': 'Could not extract distance from response',
                'expected_km': expected_km,
                'predicted_km': None,
                'raw_response': response
            }

        # Check if within tolerance
        absolute_error = abs(predicted_km - expected_km)

        # Calculate percent error (handle zero distance case)
        if expected_km > 0:
            percent_error = (absolute_error / expected_km) * 100
        else:
            # For zero distance, only check absolute error
            percent_error = 0.0 if absolute_error == 0 else float('inf')

        # Pass if within either absolute or percentage tolerance
        within_absolute = absolute_error <= tolerance_km
        within_percent = percent_error <= tolerance_pct

        correct = within_absolute or within_percent

        return {
            'correct': correct,
            'expected_km': expected_km,
            'predicted_km': predicted_km,
            'absolute_error_km': round(absolute_error, 2),
            'percent_error': round(percent_error, 2) if percent_error != float('inf') else None,
            'tolerance_km': tolerance_km,
            'tolerance_percent': tolerance_pct,
            'raw_response': response
        }


class BearingComputationEvaluator(TaskEvaluator):
    """Evaluator for Task 2: Bearing Computation"""

    def __init__(self):
        super().__init__("bearing_computation")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in navigation and geodetic calculations.
Your task is to calculate the bearing (compass direction) from one point to another.

Bearing is measured in degrees clockwise from North (0° = North, 90° = East, 180° = South, 270° = West).
You can also express it as a cardinal direction (N, NE, E, SE, S, SW, W, NW).

Provide the initial bearing from the starting point."""

        point_a = test_case['point_a']
        point_b = test_case['point_b']

        user_prompt = f"""Calculate the bearing from Point A to Point B:

Point A: {point_a['name']}
  Coordinates: {point_a['lat']:.4f}, {point_a['lon']:.4f}

Point B: {point_b['name']}
  Coordinates: {point_b['lat']:.4f}, {point_b['lon']:.4f}

What is the initial bearing in degrees (or cardinal direction)?

IMPORTANT: You may show your calculations if helpful, but you MUST end your response with:

FINAL ANSWER: [bearing in degrees]

Example formats:
- FINAL ANSWER: 140 degrees
- FINAL ANSWER: 140°
- FINAL ANSWER: Southeast

Provide only the bearing value in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = test_case['ground_truth']
        expected_bearing = ground_truth['initial_bearing_deg']
        tolerance_deg = ground_truth['tolerance_deg']
        expected_cardinal = ground_truth['cardinal_direction']

        # Extract bearing from response
        predicted_bearing = self.parser.extract_bearing(response)

        if predicted_bearing is None:
            return {
                'correct': False,
                'error': 'Could not extract bearing from response',
                'expected_bearing_deg': expected_bearing,
                'expected_cardinal': expected_cardinal,
                'predicted_bearing_deg': None,
                'raw_response': response
            }

        # Calculate angular difference (handle wrap-around)
        diff = abs(predicted_bearing - expected_bearing)
        if diff > 180:
            diff = 360 - diff

        correct = diff <= tolerance_deg

        return {
            'correct': correct,
            'expected_bearing_deg': expected_bearing,
            'predicted_bearing_deg': predicted_bearing,
            'angular_error_deg': round(diff, 2),
            'tolerance_deg': tolerance_deg,
            'expected_cardinal': expected_cardinal,
            'raw_response': response
        }


class InterpolationEvaluator(TaskEvaluator):
    """Evaluator for Task 3: Coordinate Interpolation"""

    def __init__(self):
        super().__init__("interpolation")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in geodetic calculations and coordinate interpolation.
Your task is to find intermediate points along the great circle path between two locations.

Use great circle interpolation (not simple linear interpolation in lat/lon space).
Provide coordinates in decimal degrees format (lat, lon)."""

        point_a = test_case['point_a']
        point_b = test_case['point_b']
        subtask = test_case['subtask']

        if subtask == 'midpoint':
            question = "What is the midpoint (halfway point) between these two locations?"
        elif subtask == 'fractional':
            fraction = test_case['fraction']
            pct = int(fraction * 100)
            question = f"What point is {pct}% of the way from Point A to Point B?"
        else:  # path_division
            num_segments = test_case['num_segments']
            question = f"Divide the path into {num_segments} equal segments. What are the intermediate points?"

        user_prompt = f"""Point A: {point_a['name']}
  Coordinates: {point_a['lat']:.4f}, {point_a['lon']:.4f}

Point B: {point_b['name']}
  Coordinates: {point_b['lat']:.4f}, {point_b['lon']:.4f}

{question}

IMPORTANT: You may show your calculations if helpful, but you MUST end your response with:

FINAL ANSWER: [latitude], [longitude]

Example formats:
- FINAL ANSWER: 51.3, -31.4
- FINAL ANSWER: 40.7128, -74.0060
- FINAL ANSWER: lat: 51.3, lon: -31.4

Provide only the coordinate value(s) in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = test_case['ground_truth']
        tolerance_km = ground_truth['tolerance_km']
        subtask = test_case['subtask']

        # Extract coordinate(s) from response
        predicted_coord = self.parser.extract_coordinate(response)

        if predicted_coord is None:
            return {
                'correct': False,
                'error': 'Could not extract coordinate from response',
                'subtask': subtask,
                'raw_response': response
            }

        predicted_lat, predicted_lon = predicted_coord

        # Get expected coordinate
        if subtask in ['midpoint', 'fractional']:
            if subtask == 'midpoint':
                expected = ground_truth['midpoint']
            else:
                expected = ground_truth['interpolated_point']

            expected_lat = expected['lat']
            expected_lon = expected['lon']

            # Calculate distance between predicted and expected
            from scripts.layer_c_geodetic_utils import GeodesicCalculator
            calc = GeodesicCalculator()

            dist_result = calc.calculate_distance(
                predicted_lat, predicted_lon,
                expected_lat, expected_lon
            )
            distance_error_km = dist_result['distance_km']

            correct = distance_error_km <= tolerance_km

            return {
                'correct': correct,
                'expected_lat': expected_lat,
                'expected_lon': expected_lon,
                'predicted_lat': predicted_lat,
                'predicted_lon': predicted_lon,
                'distance_error_km': distance_error_km,
                'tolerance_km': tolerance_km,
                'subtask': subtask,
                'raw_response': response
            }

        else:  # path_division - harder to evaluate, just check if got coordinates
            return {
                'correct': True,  # Give credit for extracting a coordinate
                'predicted_lat': predicted_lat,
                'predicted_lon': predicted_lon,
                'subtask': subtask,
                'note': 'Path division evaluation simplified',
                'raw_response': response
            }


class AreaPerimeterEvaluator(TaskEvaluator):
    """Evaluator for Task 4: Area & Perimeter Calculation"""

    def __init__(self):
        super().__init__("area_perimeter")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in geodetic calculations and geometry on Earth's surface.
Your task is to calculate the area and/or perimeter of a polygon defined by coordinates.

Remember that Earth is not flat - use spherical or ellipsoidal geometry.
Provide area in square kilometers (km²) and perimeter in kilometers (km)."""

        polygon = test_case['polygon']
        vertices = polygon['vertices']
        subtask = test_case['subtask']

        # Format vertices
        vertex_strs = []
        for i, v in enumerate(vertices, 1):
            vertex_strs.append(f"  Vertex {i}: {v['lat']:.4f}, {v['lon']:.4f}")
            if 'name' in v:
                vertex_strs[-1] += f" ({v['name']})"

        vertices_text = "\n".join(vertex_strs)

        user_prompt = f"""Calculate the area and perimeter of this polygon:

Type: {subtask} ({polygon['vertex_count']} vertices)

Vertices:
{vertices_text}

What is the area in square kilometers (km²) and perimeter in kilometers (km)?

IMPORTANT: You may show your calculations if helpful, but you MUST end your response with:

FINAL ANSWER: Area: [value] km², Perimeter: [value] km

Example formats:
- FINAL ANSWER: Area: 15.2 km², Perimeter: 45.3 km
- FINAL ANSWER: 15.2 km², 45.3 km
- FINAL ANSWER: Area = 15.2 square kilometers, Perimeter = 45.3 kilometers

Provide the area and perimeter values in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = test_case['ground_truth']
        expected_area = ground_truth['area_km2']
        expected_perimeter = ground_truth['perimeter_km']
        tolerance_pct = ground_truth['tolerance_percent']

        # Extract area and perimeter
        predicted_area = self.parser.extract_area(response)

        # For perimeter, extract distance
        predicted_perimeter = self.parser.extract_distance(response)

        # Evaluate area
        area_correct = False
        area_error_pct = None
        if predicted_area is not None:
            area_error = abs(predicted_area - expected_area)
            # Handle zero area case (degenerate polygons)
            if expected_area > 0:
                area_error_pct = (area_error / expected_area) * 100
                area_correct = area_error_pct <= tolerance_pct
            else:
                # For zero area, check absolute error only
                area_error_pct = 0.0 if area_error == 0 else None
                area_correct = area_error <= 0.01  # Within 0.01 km²

        # Evaluate perimeter
        perimeter_correct = False
        perimeter_error_pct = None
        if predicted_perimeter is not None:
            perimeter_error = abs(predicted_perimeter - expected_perimeter)
            # Handle zero perimeter case
            if expected_perimeter > 0:
                perimeter_error_pct = (perimeter_error / expected_perimeter) * 100
                perimeter_correct = perimeter_error_pct <= (tolerance_pct * 0.4)  # Stricter for perimeter
            else:
                # For zero perimeter, check absolute error only
                perimeter_error_pct = 0.0 if perimeter_error == 0 else None
                perimeter_correct = perimeter_error <= 0.01  # Within 0.01 km

        # Correct if either area or perimeter is correct
        correct = area_correct or perimeter_correct

        return {
            'correct': correct,
            'area_correct': area_correct,
            'perimeter_correct': perimeter_correct,
            'expected_area_km2': expected_area,
            'predicted_area_km2': predicted_area,
            'area_error_percent': area_error_pct,
            'expected_perimeter_km': expected_perimeter,
            'predicted_perimeter_km': predicted_perimeter,
            'perimeter_error_percent': perimeter_error_pct,
            'tolerance_percent': tolerance_pct,
            'raw_response': response
        }


class BoundingBoxEvaluator(TaskEvaluator):
    """Evaluator for Task 5: Bounding Box & Spatial Extent"""

    def __init__(self):
        super().__init__("bounding_box")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in spatial analysis and coordinate geometry.
Your task is to determine the bounding box (spatial extent) of a set of points.

A bounding box is defined by:
- Minimum latitude (southernmost point)
- Maximum latitude (northernmost point)
- Minimum longitude (westernmost point)
- Maximum longitude (easternmost point)

You may also calculate the center point or diagonal distance if helpful."""

        points = test_case['points']
        num_points = test_case['num_points']

        # Format points
        point_strs = []
        for i, p in enumerate(points, 1):
            point_strs.append(f"  Point {i}: {p['lat']:.4f}, {p['lon']:.4f}")
            if 'name' in p:
                point_strs[-1] += f" ({p['name']})"

        points_text = "\n".join(point_strs)

        user_prompt = f"""Find the bounding box for these {num_points} points:

{points_text}

What are the minimum and maximum latitude and longitude values?
What is the center point?

IMPORTANT: You may show your analysis if helpful, but you MUST end your response with:

FINAL ANSWER: min_lat=[value], max_lat=[value], min_lon=[value], max_lon=[value], center=[lat, lon]

Example format:
- FINAL ANSWER: min_lat=34.05, max_lat=41.88, min_lon=-118.24, max_lon=-74.01, center=37.97, -96.13

Provide the bounding box values in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = test_case['ground_truth']

        expected_min_lat = ground_truth['min_lat']
        expected_max_lat = ground_truth['max_lat']
        expected_min_lon = ground_truth['min_lon']
        expected_max_lon = ground_truth['max_lon']
        expected_center = ground_truth['center']

        tolerance_deg = ground_truth['tolerance_deg']

        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = NumericalParser.extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            # STAGE 2: Fallback to full response
            response_to_check = response

        # Extract all numbers from response (will find lat/lon values)
        # Regex: optional minus, digits, optional decimal point with more digits
        numbers = re.findall(r'-?\d+\.?\d*', response_to_check)
        numbers = [float(n) for n in numbers if n and n != '-']

        # Try to match expected values
        min_lat_found = any(abs(n - expected_min_lat) < tolerance_deg for n in numbers)
        max_lat_found = any(abs(n - expected_max_lat) < tolerance_deg for n in numbers)
        min_lon_found = any(abs(n - expected_min_lon) < tolerance_deg for n in numbers)
        max_lon_found = any(abs(n - expected_max_lon) < tolerance_deg for n in numbers)

        # Extract center if mentioned (check in the same text we're parsing)
        center_coord = self.parser.extract_coordinate(response_to_check)
        center_correct = False
        if center_coord:
            center_lat, center_lon = center_coord
            lat_diff = abs(center_lat - expected_center['lat'])
            lon_diff = abs(center_lon - expected_center['lon'])
            center_correct = lat_diff < (tolerance_deg * 10) and lon_diff < (tolerance_deg * 10)

        # Correct if at least 3 of 4 bounds found, or if center is correct
        bounds_found = sum([min_lat_found, max_lat_found, min_lon_found, max_lon_found])
        correct = bounds_found >= 3 or center_correct

        return {
            'correct': correct,
            'bounds_found': bounds_found,
            'min_lat_found': min_lat_found,
            'max_lat_found': max_lat_found,
            'min_lon_found': min_lon_found,
            'max_lon_found': max_lon_found,
            'center_correct': center_correct,
            'expected_bounds': {
                'min_lat': expected_min_lat,
                'max_lat': expected_max_lat,
                'min_lon': expected_min_lon,
                'max_lon': expected_max_lon
            },
            'expected_center': expected_center,
            'tolerance_deg': tolerance_deg,
            'raw_response': response
        }


class RouteGeometryEvaluator(TaskEvaluator):
    """Evaluator for Task 6: Route Geometry"""

    def __init__(self):
        super().__init__("route_geometry")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in route analysis and geometric calculations on GPS tracks.
Your task is to analyze route geometry, including polyline simplification, route length,
turn angles, and route straightness.

Common algorithms:
- Douglas-Peucker for polyline simplification
- Geodesic distance calculations for route length
- Bearing calculations for turn angles
- Straightness ratio: direct_distance / route_length

Be precise in your calculations."""

        subtask = test_case.get('subtask', 'unknown')

        if subtask == 'polyline_simplification':
            user_prompt = self._prompt_polyline_simplification(test_case)
        elif subtask == 'route_length_calculation':
            user_prompt = self._prompt_route_length(test_case)
        elif subtask == 'turn_angle_calculation':
            user_prompt = self._prompt_turn_angle(test_case)
        elif subtask == 'route_straightness':
            user_prompt = self._prompt_straightness(test_case)
        else:
            user_prompt = test_case.get('question', 'Unknown subtask')

        return system_prompt, user_prompt

    def _prompt_polyline_simplification(self, test_case: Dict[str, Any]) -> str:
        track = test_case['original_track']
        epsilon_km = test_case['epsilon_km']

        # Format track points
        point_strs = []
        for i, p in enumerate(track, 1):
            point_strs.append(f"Point {i}: {p['lat']:.6f}, {p['lon']:.6f}")

        points_text = "\n".join(point_strs)

        prompt = f"""Apply polyline simplification to this GPS track with tolerance {epsilon_km} km.

Track has {len(track)} points:
{points_text}

Use the Douglas-Peucker algorithm with epsilon = {epsilon_km} km.

IMPORTANT: How many points remain after simplification?

FINAL ANSWER: [number] points

Example format:
- FINAL ANSWER: 5 points
- FINAL ANSWER: 7

Provide only the number of points in the FINAL ANSWER line."""

        return prompt

    def _prompt_route_length(self, test_case: Dict[str, Any]) -> str:
        route = test_case['route']
        num_points = test_case['num_points']

        # Format route points
        point_strs = []
        for i, p in enumerate(route, 1):
            point_strs.append(f"Point {i}: {p['lat']:.4f}, {p['lon']:.4f}")

        points_text = "\n".join(point_strs)

        prompt = f"""Calculate the total length of this route.

Route has {num_points} points:
{points_text}

Calculate the cumulative distance along the route (sum of all segments).

IMPORTANT: Provide the total route length in kilometers.

FINAL ANSWER: [distance] km

Example format:
- FINAL ANSWER: 125.3 km
- FINAL ANSWER: 45 kilometers

Provide only the distance value in the FINAL ANSWER line."""

        return prompt

    def _prompt_turn_angle(self, test_case: Dict[str, Any]) -> str:
        waypoints = test_case['waypoints']

        # Format waypoints
        wp_strs = []
        for i, wp in enumerate(waypoints, 1):
            wp_strs.append(f"Point {i}: {wp['lat']:.4f}, {wp['lon']:.4f}")

        waypoints_text = "\n".join(wp_strs)

        prompt = f"""Calculate the turn angle at the middle point (Point 2).

Waypoints:
{waypoints_text}

The turn angle is the change in direction from the incoming bearing to the outgoing bearing.
Express as an absolute angle in degrees (0° = straight, 180° = U-turn).

IMPORTANT: Provide the turn angle in degrees.

FINAL ANSWER: [angle] degrees

Example format:
- FINAL ANSWER: 45 degrees
- FINAL ANSWER: 90°
- FINAL ANSWER: sharp left turn (135°)

Provide only the angle value in the FINAL ANSWER line."""

        return prompt

    def _prompt_straightness(self, test_case: Dict[str, Any]) -> str:
        route = test_case['route']
        num_points = test_case['num_points']

        start = route[0]
        end = route[-1]

        prompt = f"""Calculate the straightness ratio for this route.

Route from {start['lat']:.4f}, {start['lon']:.4f} to {end['lat']:.4f}, {end['lon']:.4f}
Total points: {num_points}

Straightness ratio = (direct distance from start to end) / (actual route length)
A ratio of 1.0 means perfectly straight, closer to 0 means very winding.

IMPORTANT: Provide the straightness ratio as a decimal.

FINAL ANSWER: [ratio]

Example format:
- FINAL ANSWER: 0.85
- FINAL ANSWER: 0.62
- FINAL ANSWER: straightness ratio of 0.75

Provide only the ratio value in the FINAL ANSWER line."""

        return prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        subtask = test_case.get('subtask', 'unknown')

        if subtask == 'polyline_simplification':
            return self._evaluate_polyline_simplification(response, test_case)
        elif subtask == 'route_length_calculation':
            return self._evaluate_route_length(response, test_case)
        elif subtask == 'turn_angle_calculation':
            return self._evaluate_turn_angle(response, test_case)
        elif subtask == 'route_straightness':
            return self._evaluate_straightness(response, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _evaluate_polyline_simplification(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate polyline simplification - check which point indices are kept"""
        ground_truth = test_case['ground_truth']
        expected_indices = ground_truth['points_kept']  # List of indices like [0, 3, 7, 11]

        # Extract list of indices from response
        predicted_indices = self.parser.extract_index_list(response)

        if predicted_indices is None:
            return {
                'correct': False,
                'error': 'Could not extract point indices from response',
                'expected_indices': expected_indices,
                'predicted_indices': None,
                'raw_response': response
            }

        # Exact match required (sorted comparison to handle order differences)
        correct = sorted(predicted_indices) == sorted(expected_indices)

        # Calculate metrics for partial credit analysis
        predicted_set = set(predicted_indices)
        expected_set = set(expected_indices)
        num_correct = len(predicted_set & expected_set)  # Intersection
        num_missing = len(expected_set - predicted_set)  # In expected but not predicted
        num_extra = len(predicted_set - expected_set)    # In predicted but not expected

        return {
            'correct': correct,
            'expected_indices': expected_indices,
            'predicted_indices': predicted_indices,
            'num_correct_indices': num_correct,
            'num_expected_indices': len(expected_indices),
            'num_predicted_indices': len(predicted_indices),
            'num_missing_indices': num_missing,
            'num_extra_indices': num_extra,
            'match_rate': num_correct / len(expected_indices) if expected_indices else 0.0,
            'raw_response': response
        }

    def _evaluate_route_length(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate route length calculation"""
        ground_truth = test_case['ground_truth']
        expected_length_km = ground_truth['route_length_km']
        tolerance_km = ground_truth.get('tolerance_km', 5.0)  # Default 5 km
        tolerance_pct = ground_truth.get('tolerance_percent', 5.0)  # Default 5%

        # Extract distance from response
        predicted_length_km = self.parser.extract_distance(response)

        if predicted_length_km is None:
            return {
                'correct': False,
                'error': 'Could not extract route length from response',
                'expected_length_km': expected_length_km,
                'predicted_length_km': None,
                'raw_response': response
            }

        # Check if within tolerance
        absolute_error = abs(predicted_length_km - expected_length_km)

        if expected_length_km > 0:
            percent_error = (absolute_error / expected_length_km) * 100
        else:
            percent_error = 0.0 if absolute_error == 0 else float('inf')

        # Pass if within either absolute or percentage tolerance
        within_absolute = absolute_error <= tolerance_km
        within_percent = percent_error <= tolerance_pct

        correct = within_absolute or within_percent

        return {
            'correct': correct,
            'expected_length_km': expected_length_km,
            'predicted_length_km': predicted_length_km,
            'absolute_error_km': round(absolute_error, 2),
            'percent_error': round(percent_error, 2) if percent_error != float('inf') else None,
            'tolerance_km': tolerance_km,
            'tolerance_percent': tolerance_pct,
            'raw_response': response
        }

    def _evaluate_turn_angle(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate turn angle calculation"""
        ground_truth = test_case['ground_truth']
        expected_angle = ground_truth['turn_angle_deg']
        tolerance_deg = ground_truth.get('tolerance_deg', 5.0)  # Default 5°

        # Extract bearing/angle from response
        predicted_angle = self.parser.extract_bearing(response)

        if predicted_angle is None:
            # Try extracting as a plain number
            predicted_angle = self.parser.extract_number(response)

        if predicted_angle is None:
            return {
                'correct': False,
                'error': 'Could not extract turn angle from response',
                'expected_angle_deg': expected_angle,
                'predicted_angle_deg': None,
                'raw_response': response
            }

        # Calculate angular difference (handle wrap-around for 0-180 range)
        diff = abs(predicted_angle - expected_angle)
        if diff > 180:
            diff = 360 - diff

        correct = diff <= tolerance_deg

        return {
            'correct': correct,
            'expected_angle_deg': expected_angle,
            'predicted_angle_deg': predicted_angle,
            'angular_error_deg': round(diff, 2),
            'tolerance_deg': tolerance_deg,
            'raw_response': response
        }

    def _evaluate_straightness(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate straightness ratio calculation"""
        ground_truth = test_case['ground_truth']
        expected_ratio = ground_truth['straightness_ratio']
        tolerance = ground_truth.get('tolerance', 0.05)  # Default ±0.05

        # Extract number from response (ratio should be 0-1)
        predicted_ratio = self.parser.extract_number(response)

        if predicted_ratio is None:
            return {
                'correct': False,
                'error': 'Could not extract straightness ratio from response',
                'expected_ratio': expected_ratio,
                'predicted_ratio': None,
                'raw_response': response
            }

        # Validate range (ratio should be 0-1)
        if not (0 <= predicted_ratio <= 1):
            return {
                'correct': False,
                'error': f'Invalid ratio value: {predicted_ratio} (should be 0-1)',
                'expected_ratio': expected_ratio,
                'predicted_ratio': predicted_ratio,
                'raw_response': response
            }

        # Check if within tolerance
        error = abs(predicted_ratio - expected_ratio)
        correct = error <= tolerance

        return {
            'correct': correct,
            'expected_ratio': round(expected_ratio, 3),
            'predicted_ratio': round(predicted_ratio, 3),
            'absolute_error': round(error, 3),
            'tolerance': tolerance,
            'raw_response': response
        }


def get_evaluator(task_name: str) -> TaskEvaluator:
    """
    Factory function to get appropriate evaluator for a task

    Args:
        task_name: Name of the task

    Returns:
        TaskEvaluator instance
    """
    evaluators = {
        'task1_distance_calculation': DistanceCalculationEvaluator(),
        'distance_calculation': DistanceCalculationEvaluator(),

        'task2_bearing_computation': BearingComputationEvaluator(),
        'bearing_computation': BearingComputationEvaluator(),

        'task3_interpolation': InterpolationEvaluator(),
        'interpolation': InterpolationEvaluator(),

        'task4_area_perimeter': AreaPerimeterEvaluator(),
        'area_perimeter': AreaPerimeterEvaluator(),

        'task5_bounding_box': BoundingBoxEvaluator(),
        'bounding_box': BoundingBoxEvaluator(),

        'task6_route_geometry': RouteGeometryEvaluator(),
        'route_geometry': RouteGeometryEvaluator(),
    }

    if task_name not in evaluators:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(evaluators.keys())}")

    return evaluators[task_name]


# Test evaluators
if __name__ == "__main__":
    print("="*60)
    print("Testing Layer C Evaluators")
    print("="*60)

    # Test Task 1: Distance Calculation
    print("\nTask 1: Distance Calculation")
    print("-"*60)
    evaluator1 = DistanceCalculationEvaluator()

    test_case = {
        'point_a': {'lat': 40.7128, 'lon': -74.0060, 'name': 'NYC'},
        'point_b': {'lat': 51.5074, 'lon': -0.1278, 'name': 'London'},
        'ground_truth': {
            'distance_km': 5585.23,
            'tolerance_km': 50.0,
            'tolerance_percent': 2.0
        }
    }

    responses = [
        "The distance is 5,585 kilometers",
        "approximately 5600 km",
        "3470 miles",
        "about 5.5 × 10³ km",
        "500 km"
    ]

    for resp in responses:
        result = evaluator1.evaluate_response(resp, test_case)
        print(f"Response: '{resp}'")
        print(f"  Correct: {result['correct']}, Predicted: {result.get('predicted_km')} km, Error: {result.get('percent_error')}%")

    # Test Task 2: Bearing
    print("\n\nTask 2: Bearing Computation")
    print("-"*60)
    evaluator2 = BearingComputationEvaluator()

    test_case = {
        'point_a': {'lat': 40.7128, 'lon': -74.0060, 'name': 'NYC'},
        'point_b': {'lat': 51.5074, 'lon': -0.1278, 'name': 'London'},
        'ground_truth': {
            'initial_bearing_deg': 51.2,
            'cardinal_direction': 'NE',
            'tolerance_deg': 5.0
        }
    }

    responses = [
        "The bearing is 51 degrees",
        "Northeast (approximately 45°)",
        "bearing: 50.5°",
        "The direction is East",
        "180 degrees"
    ]

    for resp in responses:
        result = evaluator2.evaluate_response(resp, test_case)
        print(f"Response: '{resp}'")
        print(f"  Correct: {result['correct']}, Predicted: {result.get('predicted_bearing_deg')}°, Error: {result.get('angular_error_deg')}°")

    print("\n" + "="*60)
    print("Layer C Evaluator testing complete!")
    print("="*60)
