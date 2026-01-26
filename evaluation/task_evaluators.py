"""
Task-specific evaluators for GPSBench Layer A
Each evaluator handles prompting and response evaluation for one task
"""

import re
import json
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
from .mc_utils import is_mc_test, evaluate_mc_response, format_mc_question


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


class TaskEvaluator(ABC):
    """Base class for task evaluators"""

    def __init__(self, task_name: str):
        self.task_name = task_name

    @abstractmethod
    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        """
        Create prompt for LLM

        Args:
            test_case: Test case dictionary

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        pass

    @abstractmethod
    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate LLM response against ground truth

        Args:
            response: LLM response text
            test_case: Test case with ground truth

        Returns:
            Dictionary with evaluation results
        """
        pass


class FormatRecognitionEvaluator(TaskEvaluator):
    """Evaluator for Task 1: Format Recognition & Parsing"""

    def __init__(self):
        super().__init__("format_recognition")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in GPS coordinate systems and formats.
Your task is to identify the format of coordinate strings.

Possible formats:
- DMS: Degrees Minutes Seconds (e.g., 40°51'30"N, 73°59'45"W)
- decimal_degrees: Decimal degrees (e.g., 40.858, -73.996)
- ISO_6709: ISO 6709 format (e.g., +40.858-073.996/)
- plus_code: Plus Codes/Open Location Code (e.g., 87G8Q25M+)
- MGRS: Military Grid Reference System (e.g., 18T WL 80819 51585)
- UTM: Universal Transverse Mercator (e.g., Zone 18N 585628E 4515585N)"""

        coord_str = test_case['coordinate_string']
        user_prompt = f"""Identify the format of this coordinate: {coord_str}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [format name]

Example: FINAL ANSWER: DMS

Provide only the format name (DMS, decimal_degrees, ISO_6709, plus_code, MGRS, or UTM) in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_format = test_case['ground_truth']['format_type'].lower()

        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            # STAGE 2: Fallback to full response
            response_to_check = response

        response_lower = response_to_check.lower().strip()

        # Check if expected format is mentioned in response
        correct = expected_format in response_lower

        # Extract what format the model predicted
        predicted_format = None
        for fmt in ['dms', 'decimal_degrees', 'iso_6709', 'plus_code', 'mgrs', 'utm']:
            if fmt in response_lower or fmt.replace('_', ' ') in response_lower:
                predicted_format = fmt
                break

        return {
            'correct': correct,
            'expected': expected_format,
            'predicted': predicted_format,
            'raw_response': response
        }


class FormatConversionEvaluator(TaskEvaluator):
    """Evaluator for Task 2: Format Conversion"""

    def __init__(self):
        super().__init__("format_conversion")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in GPS coordinate conversions.
Convert coordinates between different formats accurately.

Be precise with your conversions. For decimal degrees, use 6 decimal places.
For DMS, format as: DD°MM'SS.S"[N/S], DDD°MM'SS.S"[E/W]"""

        source_format = test_case['source_format']
        target_format = test_case['target_format']
        source_string = test_case['source_string']

        user_prompt = f"""Convert this coordinate from {source_format} to {target_format}:

Input ({source_format}): {source_string}

IMPORTANT: You may show your conversion process, but you MUST end your response with:

FINAL ANSWER: [converted coordinate]

Example: FINAL ANSWER: 40.858333, -73.996250

Provide only the converted coordinate in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected = test_case['ground_truth']['target_string']

        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            # STAGE 2: Fallback to full response
            response_to_check = response

        response_clean = response_to_check.strip()

        # Extract numbers from both expected and response for comparison
        expected_numbers = self._extract_numbers(expected)
        response_numbers = self._extract_numbers(response_clean)

        # Check if numbers match (with tolerance for floating point and DMS rounding)
        correct = self._numbers_match(expected_numbers, response_numbers, tolerance=0.01)

        return {
            'correct': correct,
            'expected': expected,
            'predicted': response_clean,
            'raw_response': response
        }

    def _extract_numbers(self, text: str) -> list:
        """Extract all numbers from text"""
        # Match integers and floats, including negative
        return [float(x) for x in re.findall(r'-?\d+\.?\d*', text) if x]

    def _numbers_match(self, nums1: list, nums2: list, tolerance: float = 0.01) -> bool:
        """Check if two lists of numbers match within tolerance"""
        if len(nums1) != len(nums2):
            return False
        return all(abs(a - b) < tolerance for a, b in zip(nums1, nums2))


class ValidationEvaluator(TaskEvaluator):
    """Evaluator for Task 3: Validation & Error Detection

    Supports both free-form and multiple choice formats.
    """

    def __init__(self):
        super().__init__("validation")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return format_mc_question(test_case)

        # Free-form format
        system_prompt = """You are an expert in GPS coordinate validation.
Determine if coordinates are valid and identify any errors.

Valid coordinate ranges:
- Latitude: -90 to +90 degrees
- Longitude: -180 to +180 degrees

Common errors to check for:
- Out of range values
- Transposed coordinates (lat/lon swapped)
- Low precision for intended use
- Malformed format"""

        coord_str = test_case['coordinate_string']
        user_prompt = f"""Validate this coordinate: {coord_str}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: VALID
(or FINAL ANSWER: INVALID)

Provide only VALID or INVALID in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return evaluate_mc_response(response, test_case)

        # Free-form evaluation
        expected_valid = test_case['ground_truth']['is_valid']

        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            # STAGE 2: Fallback to full response
            response_to_check = response

        response_lower = response_to_check.lower()

        # Determine if model said valid or invalid
        # CRITICAL FIX: Check for 'invalid' FIRST (since 'invalid' contains 'valid')
        if 'invalid' in response_lower:
            model_says_valid = False
        elif 'valid' in response_lower:
            model_says_valid = True
        else:
            # No clear valid/invalid keyword found - try to infer
            model_says_valid = not any(word in response_lower for word in ['error', 'incorrect', 'wrong', 'exceed', 'out of range'])

        correct = model_says_valid == expected_valid

        return {
            'correct': correct,
            'expected_valid': expected_valid,
            'predicted_valid': model_says_valid,
            'error_type': test_case.get('error_type', 'none'),
            'raw_response': response
        }


class CoordinateSystemEvaluator(TaskEvaluator):
    """Evaluator for Task 4: Coordinate System Recognition"""

    def __init__(self):
        super().__init__("coordinate_system_recognition")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in coordinate reference systems.
Identify which coordinate system is being used.

Common systems:
- WGS84: World Geodetic System 1984 (standard lat/lon in decimal degrees, used by GPS)
- WGS84_DMS: World Geodetic System 1984 in DMS (Degrees Minutes Seconds) format
- UTM: Universal Transverse Mercator (zone-based projected coordinates)
- MGRS: Military Grid Reference System (alphanumeric grid)"""

        coord_str = test_case['coordinate_string']
        user_prompt = f"""Identify the coordinate system: {coord_str}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [system name]

Example: FINAL ANSWER: WGS84

Provide only the system name (WGS84, WGS84_DMS, UTM, or MGRS) in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_system = test_case['ground_truth']['system_name'].upper()

        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            # STAGE 2: Fallback to full response
            response_to_check = response

        response_upper = response_to_check.upper().strip()

        # Extract predicted system (check longer strings first to avoid WGS84 matching WGS84_DMS)
        predicted_system = None
        for sys in ['WGS84_DMS', 'WGS84', 'MGRS', 'UTM']:  # Note: WGS84_DMS before WGS84!
            if sys in response_upper:
                predicted_system = sys
                break

        # Check if predicted matches expected (not just substring match)
        correct = predicted_system == expected_system

        return {
            'correct': correct,
            'expected': expected_system,
            'predicted': predicted_system,
            'raw_response': response
        }


class PrecisionEvaluator(TaskEvaluator):
    """Evaluator for Task 5: Precision Understanding"""

    def __init__(self):
        super().__init__("precision_understanding")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in GPS coordinate precision.
Analyze the precision of coordinates and determine their accuracy.

Precision guide (at equator):
- 0 decimals: ~111 km (country-level)
- 1 decimal: ~11 km (large city)
- 2 decimals: ~1.1 km (neighborhood)
- 3 decimals: ~111 m (street)
- 4 decimals: ~11 m (individual building)
- 5 decimals: ~1.1 m (tree, survey marker)
- 6 decimals: ~0.11 m (precise surveying)"""

        coord_str = test_case['coordinate_string']
        user_prompt = f"""Analyze the precision of this coordinate: {coord_str}

IMPORTANT: You may explain your analysis, but you MUST end your response with:

FINAL ANSWER: [decimal_places] decimal places, [accuracy] meters, [use_case]

Example: FINAL ANSWER: 4 decimal places, 11 meters, street-level

Provide the decimal places, accuracy, and use case in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_decimals = test_case['ground_truth']['decimal_places']
        expected_accuracy = test_case['ground_truth']['accuracy_meters']
        expected_use_case = test_case['ground_truth']['appropriate_use_case']

        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            # STAGE 2: Fallback to full response
            response_to_check = response

        response_lower = response_to_check.lower()

        # Extract decimal places from response (handle "decimals" or "decimal places")
        decimal_match = re.search(r'(\d+)\s*decimal', response_lower)
        predicted_decimals = int(decimal_match.group(1)) if decimal_match else None

        # Extract accuracy - handle both meters and kilometers, with or without commas
        # Try km first (convert to meters)
        km_match = re.search(r'([\d,]+(?:\.\d+)?)\s*(?:km|kilometer)', response_lower)
        if km_match:
            # Remove commas and convert km to m
            num_str = km_match.group(1).replace(',', '')
            predicted_accuracy = float(num_str) * 1000
        else:
            # Try meters
            m_match = re.search(r'([\d,]+(?:\.\d+)?)\s*(?:m\b|meter)', response_lower)
            if m_match:
                # Remove commas
                num_str = m_match.group(1).replace(',', '')
                predicted_accuracy = float(num_str)
            else:
                predicted_accuracy = None

        # Check if decimals match
        decimals_correct = predicted_decimals == expected_decimals if predicted_decimals is not None else False

        # Check if accuracy is approximately correct (within order of magnitude)
        accuracy_correct = False
        if predicted_accuracy is not None:
            ratio = predicted_accuracy / expected_accuracy if expected_accuracy > 0 else 0
            accuracy_correct = 0.1 <= ratio <= 10  # Within one order of magnitude

        # Check if use case is mentioned
        use_case_correct = expected_use_case.lower() in response_lower

        # Overall correct if at least 2 out of 3 are correct
        correct = sum([decimals_correct, accuracy_correct, use_case_correct]) >= 2

        return {
            'correct': correct,
            'decimals_correct': decimals_correct,
            'accuracy_correct': accuracy_correct,
            'use_case_correct': use_case_correct,
            'expected_decimals': expected_decimals,
            'predicted_decimals': predicted_decimals,
            'expected_accuracy': expected_accuracy,
            'predicted_accuracy': predicted_accuracy,
            'raw_response': response
        }


class CoordinateSystemTransformationEvaluator(TaskEvaluator):
    """Evaluator for Task 4: Coordinate System Transformation"""

    def __init__(self):
        super().__init__("coordinate_system_transformation")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in coordinate reference systems (CRS) and transformations.
Your task is to accurately transform coordinates between different coordinate systems.

Common transformations:
- WGS84 (EPSG:4326) to UTM zones
- WGS84 to Web Mercator (EPSG:3857)
- UTM to WGS84
- Between different projections

Important:
- Be precise with transformations
- Consider the appropriate UTM zone for locations
- Web Mercator uses meters from origin
- UTM provides easting/northing in meters"""

        subtask = test_case.get('subtask', 'unknown')
        question = test_case['question']

        user_prompt = question

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        subtask = test_case.get('subtask', 'unknown')

        # Extract FINAL ANSWER
        final_answer = extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            response_to_check = response

        if subtask == "wgs84_to_utm":
            return self._evaluate_utm_transformation(response_to_check, test_case)
        elif subtask == "wgs84_to_web_mercator":
            return self._evaluate_web_mercator(response_to_check, test_case)
        elif subtask == "utm_to_wgs84":
            return self._evaluate_utm_to_wgs84(response_to_check, test_case)
        elif subtask == "crs_identification":
            return self._evaluate_crs_identification(response_to_check, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _extract_numbers(self, text: str) -> list:
        """Extract all numbers from text"""
        # Match integers and decimals, including negative
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        return [float(m) for m in matches if m]

    def _evaluate_utm_transformation(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate WGS84 to UTM transformation"""
        gt = test_case['ground_truth']
        expected_easting = gt['easting']
        expected_northing = gt['northing']

        # Extract easting and northing from response
        numbers = self._extract_numbers(response)

        if len(numbers) < 2:
            return {
                'correct': False,
                'expected': f"Easting: {expected_easting}, Northing: {expected_northing}",
                'predicted': 'Could not extract coordinates',
                'raw_response': response
            }

        # Typically easting comes first, northing second
        predicted_easting = numbers[0]
        predicted_northing = numbers[1]

        # Allow 10 meter tolerance (reasonable for coordinate transformations)
        tolerance = 10
        easting_diff = abs(predicted_easting - expected_easting)
        northing_diff = abs(predicted_northing - expected_northing)

        correct = easting_diff <= tolerance and northing_diff <= tolerance

        return {
            'correct': correct,
            'expected': f"Easting: {expected_easting:.2f}, Northing: {expected_northing:.2f}",
            'predicted': f"Easting: {predicted_easting:.2f}, Northing: {predicted_northing:.2f}",
            'easting_diff': easting_diff,
            'northing_diff': northing_diff,
            'tolerance': tolerance,
            'raw_response': response
        }

    def _evaluate_web_mercator(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate WGS84 to Web Mercator transformation"""
        gt = test_case['ground_truth']
        expected_x = gt['x']
        expected_y = gt['y']

        # Extract x and y from response
        numbers = self._extract_numbers(response)

        if len(numbers) < 2:
            return {
                'correct': False,
                'expected': f"X: {expected_x}, Y: {expected_y}",
                'predicted': 'Could not extract coordinates',
                'raw_response': response
            }

        predicted_x = numbers[0]
        predicted_y = numbers[1]

        # Allow 100 meter tolerance
        tolerance = 100
        x_diff = abs(predicted_x - expected_x)
        y_diff = abs(predicted_y - expected_y)

        correct = x_diff <= tolerance and y_diff <= tolerance

        return {
            'correct': correct,
            'expected': f"X: {expected_x:.2f}, Y: {expected_y:.2f}",
            'predicted': f"X: {predicted_x:.2f}, Y: {predicted_y:.2f}",
            'x_diff': x_diff,
            'y_diff': y_diff,
            'tolerance': tolerance,
            'raw_response': response
        }

    def _evaluate_utm_to_wgs84(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate UTM to WGS84 transformation"""
        gt = test_case['ground_truth']
        expected_lat = gt['lat']
        expected_lon = gt['lon']

        # Extract lat and lon from response
        numbers = self._extract_numbers(response)

        if len(numbers) < 2:
            return {
                'correct': False,
                'expected': f"Lat: {expected_lat}, Lon: {expected_lon}",
                'predicted': 'Could not extract coordinates',
                'raw_response': response
            }

        # Try to identify which is lat and which is lon
        # Latitude is typically -90 to 90, longitude -180 to 180
        predicted_lat = None
        predicted_lon = None

        for num in numbers[:2]:
            if -90 <= num <= 90 and predicted_lat is None:
                predicted_lat = num
            elif -180 <= num <= 180 and predicted_lon is None:
                predicted_lon = num

        if predicted_lat is None or predicted_lon is None:
            # Fallback: assume first is lat, second is lon
            predicted_lat = numbers[0]
            predicted_lon = numbers[1]

        # Allow 0.001 degree tolerance (~100m)
        tolerance = 0.001
        lat_diff = abs(predicted_lat - expected_lat)
        lon_diff = abs(predicted_lon - expected_lon)

        correct = lat_diff <= tolerance and lon_diff <= tolerance

        return {
            'correct': correct,
            'expected': f"Lat: {expected_lat:.6f}, Lon: {expected_lon:.6f}",
            'predicted': f"Lat: {predicted_lat:.6f}, Lon: {predicted_lon:.6f}",
            'lat_diff': lat_diff,
            'lon_diff': lon_diff,
            'tolerance': tolerance,
            'raw_response': response
        }

    def _evaluate_crs_identification(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate CRS identification for scenarios"""
        gt = test_case['ground_truth']
        expected_crs = gt['best_crs'].lower()

        response_lower = response.lower()

        # Check if the expected CRS is mentioned
        crs_keywords = {
            "utm": ["utm", "universal transverse mercator"],
            "web mercator": ["web mercator", "epsg:3857", "google mercator"],
            "wgs84": ["wgs84", "wgs 84", "epsg:4326", "geographic"]
        }

        # Find which CRS type is expected
        expected_type = None
        for crs_type, keywords in crs_keywords.items():
            if any(keyword in expected_crs for keyword in keywords):
                expected_type = crs_type
                break

        # Check if response mentions the expected CRS type
        correct = False
        if expected_type:
            correct = any(keyword in response_lower for keyword in crs_keywords[expected_type])

        return {
            'correct': correct,
            'expected': gt['best_crs'],
            'predicted': response,
            'reason': gt.get('reason', ''),
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
        # Removed: 'format_recognition' (redundant with format_conversion)
        'format_conversion': FormatConversionEvaluator(),
        'validation': ValidationEvaluator(),
        # Removed: 'coordinate_system_recognition' (redundant with format_conversion)
        'precision_understanding': PrecisionEvaluator(),
        'coordinate_system_transformation': CoordinateSystemTransformationEvaluator()
    }

    if task_name not in evaluators:
        raise ValueError(f"Unknown task: {task_name}")

    return evaluators[task_name]


# Test evaluators
if __name__ == "__main__":
    # Test Format Recognition
    test_case = {
        'coordinate_string': '40°51\'30"N, 73°59\'45"W',
        'ground_truth': {
            'format_type': 'DMS',
            'latitude': 40.858,
            'longitude': -73.996
        }
    }

    evaluator = FormatRecognitionEvaluator()
    system_prompt, user_prompt = evaluator.create_prompt(test_case)
    print("Format Recognition Prompt:")
    print(user_prompt)
    print()

    # Test response evaluation
    response = "This is in DMS format (Degrees Minutes Seconds)."
    result = evaluator.evaluate_response(response, test_case)
    print(f"Evaluation Result: {result}")
