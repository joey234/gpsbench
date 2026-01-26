"""
Task-specific evaluators for GPSBench Layer E: Robustness & Error Handling

Each evaluator handles prompting and response evaluation for one task.
All tasks are designed as MEMORIZATION ADVERSARIAL TESTS.
"""

import re
import json
import math
from typing import Dict, Any, Tuple, Optional, List
from abc import ABC, abstractmethod
from hierarchical_location_matcher import HierarchicalLocationMatcher
from .mc_utils import is_mc_test, evaluate_mc_response, format_mc_question


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the FINAL ANSWER line from response

    Returns:
        The content after "FINAL ANSWER:" or None if not found
    """
    match = re.search(r'FINAL\s+ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text"""
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', text) if x]


def parse_yes_no(text: str) -> Optional[bool]:
    """Parse yes/no response"""
    text_lower = text.lower().strip()
    if 'yes' in text_lower and 'no' not in text_lower:
        return True
    if 'no' in text_lower and 'yes' not in text_lower:
        return False
    if text_lower in ['true', 't', '1']:
        return True
    if text_lower in ['false', 'f', '0']:
        return False
    return None


class LayerEEvaluator(ABC):
    """Base class for Layer E task evaluators"""

    def __init__(self, task_name: str):
        self.task_name = task_name

    @abstractmethod
    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        """Create prompt for LLM"""
        pass

    @abstractmethod
    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate LLM response against ground truth"""
        pass


# =============================================================================
# TASK 1: NOISE TOLERANCE
# =============================================================================

class NoiseToleranceEvaluator(LayerEEvaluator):
    """Evaluator for Task 1: Noise Tolerance"""

    def __init__(self):
        super().__init__("noise_tolerance")
        self.hierarchical_matcher = HierarchicalLocationMatcher()

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        subtask = test_case['subtask']

        if subtask == 'gps_noise_recognition':
            return self._prompt_gps_noise_recognition(test_case)
        elif subtask == 'noise_sensitivity':
            # Noise sensitivity tests - same as GPS noise recognition
            return self._prompt_gps_noise_recognition(test_case)
        elif subtask == 'coordinate_matching_with_tolerance':
            return self._prompt_coordinate_matching(test_case)
        elif subtask == 'low_precision_location_identification':
            return self._prompt_low_precision_identification(test_case)
        elif subtask == 'systematic_offset_recognition':
            return self._prompt_systematic_offset(test_case)
        elif subtask == 'multipath_error_handling':
            return self._prompt_multipath_error(test_case)
        elif subtask == 'gps_track_outlier_detection':
            return self._prompt_outlier_detection(test_case)
        else:
            raise ValueError(f"Unknown subtask: {subtask}")

    def _prompt_gps_noise_recognition(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are a geospatial AI assistant. Answer questions about locations based on coordinates."""

        user_prompt = f"""{test_case['question']}

Note: GPS readings may have some noise (typically ±10-100 meters).

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location name]

Example: FINAL ANSWER: New York City"""

        return system_prompt, user_prompt

    def _prompt_coordinate_matching(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are a geospatial AI assistant expert in GPS coordinate tolerance and matching.
Determine if coordinates represent the same location considering GPS accuracy (typically ±10-50 meters)."""

        user_prompt = f"""{test_case['question']}

Note: GPS readings have inherent uncertainty. Consider whether the difference falls within typical GPS tolerance.

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: Yes OR No

Example: FINAL ANSWER: Yes"""

        return system_prompt, user_prompt

    def _prompt_low_precision_identification(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are a geospatial AI assistant. Answer questions about locations based on coordinates."""

        user_prompt = f"""{test_case['question']}

Note: These coordinates have low precision (rounded).

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location name]

Example: FINAL ANSWER: London"""

        return system_prompt, user_prompt

    def _prompt_systematic_offset(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are a geospatial AI assistant. Answer questions about locations based on coordinates."""

        user_prompt = f"""{test_case['question']}

Note: These coordinates may have a systematic offset (datum error or coordinate system mismatch).

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location name]

Example: FINAL ANSWER: Tokyo"""

        return system_prompt, user_prompt

    def _prompt_multipath_error(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are a geospatial AI assistant. Answer questions about locations based on coordinates."""

        user_prompt = f"""{test_case['question']}

Note: In urban environments, GPS signals can reflect off buildings (multipath errors), causing 10-50m position errors.

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location name]

Example: FINAL ANSWER: Shanghai"""

        return system_prompt, user_prompt

    def _prompt_outlier_detection(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are a geospatial AI assistant expert in GPS data quality analysis."""

        # Format the GPS track
        track_lines = []
        for reading in test_case['track']:
            track_lines.append(f"  Reading {reading['reading_id']}: {reading['lat']:.6f}, {reading['lon']:.6f}")

        track_str = "\n".join(track_lines)

        user_prompt = f"""A GPS device recorded the following track from a stationary position:

{track_str}

The device should be stationary, but one reading appears to be an outlier (caused by poor satellite geometry or signal interference).

Which reading is the outlier?

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: Reading [number]

Example: FINAL ANSWER: Reading 7"""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        subtask = test_case['subtask']

        if subtask == 'gps_noise_recognition':
            return self._evaluate_location_identification(response, test_case)
        elif subtask == 'noise_sensitivity':
            # Noise sensitivity tests - same evaluation as GPS noise recognition
            return self._evaluate_location_identification(response, test_case)
        elif subtask == 'coordinate_matching_with_tolerance':
            return self._evaluate_coordinate_matching(response, test_case)
        elif subtask == 'low_precision_location_identification':
            return self._evaluate_location_identification(response, test_case)
        elif subtask == 'systematic_offset_recognition':
            return self._evaluate_location_identification(response, test_case)
        elif subtask == 'multipath_error_handling':
            return self._evaluate_location_identification(response, test_case)
        elif subtask == 'gps_track_outlier_detection':
            return self._evaluate_outlier_detection(response, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _evaluate_location_identification(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate location identification response with hierarchical matching
        All Task 1 subtasks now ask "What location?" and expect a location name
        """
        ground_truth = test_case['ground_truth']
        expected_location = ground_truth['location']

        # Extract FINAL ANSWER or use full response
        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        # First, check for exact match using normalized names
        expected_normalized = self._normalize_location_name(expected_location)
        predicted_normalized = self._normalize_location_name(response_to_check)

        if expected_normalized in predicted_normalized or predicted_normalized in expected_normalized:
            # Exact match found - also set hierarchical flags
            # Use hierarchical matcher to properly set country/continent flags
            # (now that Layer E ground truth has been enriched with this metadata)
            hierarchical_gt = {
                'name': expected_location,
                'type': 'city',
                'country_code': ground_truth.get('country_code'),
                'country': ground_truth.get('country'),
                'continent': ground_truth.get('continent')
            }

            # Use hierarchical matcher to get proper flags
            hierarchical_result = self.hierarchical_matcher.match_with_hierarchy(
                response_to_check, hierarchical_gt
            )

            return {
                'correct': True,
                'match_type': 'exact',
                'expected': expected_location,
                'predicted': response_to_check,
                # Use hierarchical matcher results for all flags
                'exact_location_match': True,  # We already confirmed exact match
                'city_match': True,  # If exact location matches, city also matches
                'region_match': hierarchical_result.get('region_match', False),
                'country_match': hierarchical_result.get('country_match', False),
                'continent_match': hierarchical_result.get('continent_match', False),
                'location_name': expected_location,
                'city_name': expected_location,
                'region_name': hierarchical_result.get('region_name'),
                'country_name': hierarchical_result.get('country_name'),
                'continent_name': hierarchical_result.get('continent_name'),
                'match_notes': hierarchical_result.get('match_notes', f'Exact location: {expected_location}'),
                'raw_response': response
            }

        # If no exact match, try hierarchical matching
        # Create ground_truth structure for hierarchical matcher using enriched metadata
        hierarchical_gt = {
            'name': expected_location,
            'type': 'city',  # Most Layer E locations are cities from GeoNames
            'country_code': ground_truth.get('country_code'),
            'country': ground_truth.get('country'),
            'continent': ground_truth.get('continent')
        }

        # Use hierarchical matcher - returns separate scores for each level
        hierarchical_result = self.hierarchical_matcher.match_with_hierarchy(
            response_to_check, hierarchical_gt
        )

        if hierarchical_result['any_match']:
            return {
                'correct': hierarchical_result['exact_location_match'],
                'match_type': 'hierarchical',
                'expected': expected_location,
                'predicted': response_to_check,
                # Separate scores for each level
                'exact_location_match': hierarchical_result['exact_location_match'],
                'city_match': hierarchical_result['city_match'],
                'region_match': hierarchical_result['region_match'],
                'country_match': hierarchical_result['country_match'],
                'continent_match': hierarchical_result['continent_match'],
                # What was matched at each level
                'location_name': hierarchical_result['location_name'],
                'city_name': hierarchical_result['city_name'],
                'region_name': hierarchical_result['region_name'],
                'country_name': hierarchical_result['country_name'],
                'continent_name': hierarchical_result['continent_name'],
                'match_notes': hierarchical_result['match_notes'],
                'raw_response': response
            }

        # No match at all
        return {
            'correct': False,
            'match_type': 'none',
            'expected': expected_location,
            'predicted': response_to_check,
            'exact_location_match': False,
            'city_match': False,
            'region_match': False,
            'country_match': False,
            'continent_match': False,
            'raw_response': response
        }

    def _normalize_location_name(self, name: str) -> str:
        """Normalize location name for comparison (lowercase, remove punctuation)"""
        import unicodedata
        # Normalize unicode characters
        name = unicodedata.normalize('NFKD', name)
        # Convert to lowercase and remove non-alphanumeric
        name = re.sub(r'[^a-z0-9\s]', '', name.lower())
        # Remove extra whitespace
        name = ' '.join(name.split())
        return name

    def _evaluate_coordinate_matching(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate coordinate matching with tolerance (yes/no question)
        Tests if model understands that coordinates within GPS tolerance are the same location
        """
        ground_truth = test_case['ground_truth']

        # The key field is 'should_accept_as_same' - this is the correct answer
        # for whether the coordinates should be considered the same location
        expected_answer = ground_truth.get('should_accept_as_same', ground_truth.get('answer'))

        # Extract FINAL ANSWER or use full response
        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        # Parse yes/no from response
        predicted_answer = parse_yes_no(response_to_check)

        if predicted_answer is None:
            return {
                'correct': False,
                'error': 'Could not extract yes/no answer from response',
                'response': response[:200]
            }

        # Check if answer matches
        correct = (predicted_answer == expected_answer)

        return {
            'correct': correct,
            'expected': expected_answer,
            'predicted': predicted_answer,
            'distance_meters': ground_truth.get('distance_meters'),
            'response': response[:200]
        }

    def _evaluate_outlier_detection(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate GPS track outlier detection"""
        # Find the ground truth outlier reading
        outlier_reading_id = test_case['ground_truth'].get('outlier_reading_id')

        if outlier_reading_id is None:
            return {'correct': False, 'error': 'No outlier_reading_id in ground truth'}

        # Extract FINAL ANSWER or use full response
        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        # Extract reading number from response
        reading_match = re.search(r'reading\s+(\d+)', response_to_check, re.IGNORECASE)
        predicted_reading = None
        if reading_match:
            predicted_reading = int(reading_match.group(1))
        else:
            # Try to find just a number
            numbers = extract_numbers(response_to_check)
            if numbers and 1 <= numbers[0] <= 10:
                predicted_reading = int(numbers[0])

        correct = (predicted_reading == outlier_reading_id)

        return {
            'correct': correct,
            'expected': f'Reading {outlier_reading_id}',
            'predicted': f'Reading {predicted_reading}' if predicted_reading else 'Unknown',
            'expected_reading': outlier_reading_id,
            'predicted_reading': predicted_reading,
            'subtask': 'gps_track_outlier_detection',
            'raw_response': response
        }


# =============================================================================
# TASK 2: MISSING DATA HANDLING
# =============================================================================

class MissingDataEvaluator(LayerEEvaluator):
    """Evaluator for Task 2: Missing Data Handling"""

    def __init__(self):
        super().__init__("missing_data")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        subtask = test_case['subtask']

        if subtask == 'partial_coordinate_inference':
            return self._prompt_partial_inference(test_case)
        elif subtask == 'coordinate_component_recovery':
            return self._prompt_component_recovery(test_case)
        elif subtask == 'interpolation_from_sparse_data':
            return self._prompt_interpolation(test_case)
        elif subtask == 'damaged_format_recovery':
            return self._prompt_damaged_format(test_case)
        else:
            raise ValueError(f"Unknown subtask: {subtask}")

    def _prompt_partial_inference(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in geographic coordinates and location knowledge.
Infer missing coordinate components based on location names and partial information."""

        user_prompt = f"""{test_case['question']}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [coordinate value]

Example: FINAL ANSWER: 51.5074"""

        return system_prompt, user_prompt

    def _prompt_component_recovery(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in GPS coordinate validation and error correction.
Recover missing signs/hemispheres for coordinates based on location knowledge."""

        user_prompt = f"""{test_case['question']}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [latitude, longitude]

Example: FINAL ANSWER: 51.5074, -0.1278"""

        return system_prompt, user_prompt

    def _prompt_interpolation(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in spatial interpolation and GPS tracking.
Estimate positions along routes using linear interpolation based on location knowledge."""

        # The data structure uses flat fields, not nested journey
        user_prompt = f"""{test_case['question']}

Note: You need to recall the coordinates for both locations, then interpolate.

IMPORTANT: You may show your calculation, but you MUST end your response with:

FINAL ANSWER: [latitude, longitude]

Example: FINAL ANSWER: 41.5, -72.5"""

        return system_prompt, user_prompt

    def _prompt_damaged_format(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in data recovery and coordinate parsing.
Attempt to recover or recognize when coordinates are unrecoverable."""

        user_prompt = f"""{test_case['question']}

If recovery is possible, provide the recovered coordinates.
If recovery is not possible, state that the data is unrecoverable.

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [recovered coordinates OR "UNRECOVERABLE"]

Example: FINAL ANSWER: 40.7489, -73.9680
Example: FINAL ANSWER: UNRECOVERABLE"""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        subtask = test_case['subtask']

        if subtask == 'partial_coordinate_inference':
            return self._evaluate_partial_inference(response, test_case)
        elif subtask == 'coordinate_component_recovery':
            return self._evaluate_component_recovery(response, test_case)
        elif subtask == 'interpolation_from_sparse_data':
            return self._evaluate_interpolation(response, test_case)
        elif subtask == 'damaged_format_recovery':
            return self._evaluate_damaged_format(response, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _evaluate_partial_inference(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected = test_case['ground_truth']['answer']
        tolerance = test_case['ground_truth'].get('tolerance', 0.1)

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        numbers = extract_numbers(response_to_check)

        if numbers:
            predicted = numbers[0]
            correct = abs(predicted - expected) <= tolerance
        else:
            predicted = None
            correct = False

        return {
            'correct': correct,
            'expected': expected,
            'predicted': predicted,
            'tolerance': tolerance,
            'raw_response': response
        }

    def _evaluate_component_recovery(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_lat = test_case['ground_truth']['lat']
        expected_lon = test_case['ground_truth']['lon']
        tolerance = test_case['ground_truth'].get('tolerance', 0.001)

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        numbers = extract_numbers(response_to_check)

        if len(numbers) >= 2:
            predicted_lat = numbers[0]
            predicted_lon = numbers[1]
            correct = (abs(predicted_lat - expected_lat) <= tolerance and
                      abs(predicted_lon - expected_lon) <= tolerance)
        else:
            predicted_lat = None
            predicted_lon = None
            correct = False

        return {
            'correct': correct,
            'expected': {'lat': expected_lat, 'lon': expected_lon},
            'predicted': {'lat': predicted_lat, 'lon': predicted_lon} if predicted_lat else None,
            'tolerance': tolerance,
            'raw_response': response
        }

    def _evaluate_interpolation(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_lat = test_case['ground_truth']['lat']
        expected_lon = test_case['ground_truth']['lon']
        tolerance = test_case['ground_truth'].get('tolerance', 0.5)

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        numbers = extract_numbers(response_to_check)

        if len(numbers) >= 2:
            predicted_lat = numbers[0]
            predicted_lon = numbers[1]
            correct = (abs(predicted_lat - expected_lat) <= tolerance and
                      abs(predicted_lon - expected_lon) <= tolerance)
        else:
            predicted_lat = None
            predicted_lon = None
            correct = False

        return {
            'correct': correct,
            'expected': {'lat': expected_lat, 'lon': expected_lon},
            'predicted': {'lat': predicted_lat, 'lon': predicted_lon} if predicted_lat else None,
            'tolerance': tolerance,
            'raw_response': response
        }

    def _evaluate_damaged_format(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        recoverable = test_case['ground_truth']['recoverable']

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        # Check if model recognizes unrecoverable data
        if recoverable == False:
            contains_unrecoverable = 'unrecoverable' in response_to_check.lower() or 'cannot' in response_to_check.lower() or 'missing' in response_to_check.lower()
            return {
                'correct': contains_unrecoverable,
                'expected': 'unrecoverable',
                'predicted': 'unrecoverable' if contains_unrecoverable else 'attempted_recovery',
                'raw_response': response
            }
        else:
            # Try to extract recovered coordinates
            expected_lat = test_case['ground_truth']['original_lat']
            expected_lon = test_case['ground_truth']['original_lon']
            tolerance = test_case['ground_truth'].get('tolerance', 0.01)

            numbers = extract_numbers(response_to_check)

            if len(numbers) >= 2:
                predicted_lat = numbers[0]
                predicted_lon = numbers[1]
                correct = (abs(predicted_lat - expected_lat) <= tolerance and
                          abs(predicted_lon - expected_lon) <= tolerance)
            else:
                predicted_lat = None
                predicted_lon = None
                correct = False

            return {
                'correct': correct,
                'expected': {'lat': expected_lat, 'lon': expected_lon},
                'predicted': {'lat': predicted_lat, 'lon': predicted_lon} if predicted_lat else None,
                'tolerance': tolerance,
                'raw_response': response
            }


# =============================================================================
# TASK 3: CONSISTENCY CHECKING
# =============================================================================

class ConsistencyEvaluator(LayerEEvaluator):
    """
    Evaluator for Task 3: Consistency Checking

    NOTE: This evaluator requires running the same query multiple times.
    The evaluate_response method checks consistency across multiple runs.
    """

    def __init__(self):
        super().__init__("consistency")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        subtask = test_case['subtask']

        if subtask == 'repeated_query_stability':
            return self._prompt_repeated_query(test_case)
        elif subtask == 'paraphrase_robustness':
            return self._prompt_paraphrase(test_case)
        elif subtask == 'order_independence':
            return self._prompt_order_independence(test_case)
        else:
            raise ValueError(f"Unknown subtask: {subtask}")

    def _prompt_repeated_query(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are a geospatial AI assistant. Answer questions about locations and coordinates."""

        user_prompt = f"""{test_case['question']}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [coordinates]

Example: FINAL ANSWER: 40.7128, -74.0060"""

        return system_prompt, user_prompt

    def _prompt_paraphrase(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in geographic distance calculations.
Calculate distances between locations accurately."""

        # Use first phrasing as the question
        user_prompt = f"""{test_case['phrasings'][0]}

IMPORTANT: You may explain your calculation, but you MUST end your response with:

FINAL ANSWER: [distance in km]

Example: FINAL ANSWER: 342"""

        return system_prompt, user_prompt

    def _prompt_order_independence(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in geographic distance comparisons.
Determine which location is closer to a reference point."""

        user_prompt = f"""{test_case['question_v1']}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location name]

Example: FINAL ANSWER: London"""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        NOTE: For consistency testing, this method evaluates a SINGLE response.
        To test consistency across multiple runs, the evaluation script should:
        1. Run the query multiple times
        2. Collect all responses
        3. Check if they match
        """
        subtask = test_case['subtask']

        if subtask == 'repeated_query_stability':
            return self._evaluate_repeated_query(response, test_case)
        elif subtask == 'paraphrase_robustness':
            return self._evaluate_paraphrase(response, test_case)
        elif subtask == 'order_independence':
            return self._evaluate_order_independence(response, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _evaluate_repeated_query(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected = test_case['ground_truth']['correct_answer']

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        # For now, just check if response contains expected coordinates
        # In actual evaluation, we'd run this multiple times and check consistency
        correct = expected.lower() in response_to_check.lower()

        return {
            'correct': correct,
            'expected': expected,
            'predicted': response_to_check,
            'note': 'Run this query multiple times to check consistency',
            'raw_response': response
        }

    def _evaluate_paraphrase(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_distance = test_case['ground_truth']['correct_distance_km']
        tolerance = test_case['ground_truth']['tolerance_km']

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        numbers = extract_numbers(response_to_check)

        if numbers:
            predicted_distance = numbers[0]
            correct = abs(predicted_distance - expected_distance) <= tolerance
        else:
            predicted_distance = None
            correct = False

        return {
            'correct': correct,
            'expected': expected_distance,
            'predicted': predicted_distance,
            'tolerance': tolerance,
            'note': 'Test all phrasings and check for consistency',
            'raw_response': response
        }

    def _evaluate_order_independence(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_answer = test_case['ground_truth']['correct_answer']

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        # Check if expected location name is in response
        correct = expected_answer.lower() in response_to_check.lower()

        return {
            'correct': correct,
            'expected': expected_answer,
            'predicted': response_to_check,
            'note': 'Test both question orderings and check for consistency',
            'raw_response': response
        }


# =============================================================================
# TASK 4: HALLUCINATION DETECTION
# =============================================================================

class HallucinationDetectionEvaluator(LayerEEvaluator):
    """Evaluator for Task 4: Hallucination Detection"""

    def __init__(self):
        super().__init__("hallucination_detection")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return format_mc_question(test_case)

        # Free-form format
        subtask = test_case['subtask']

        if subtask == 'cross_location_attribution':
            return self._prompt_cross_location(test_case)
        elif subtask == 'coordinate_swapping_misattribution':
            return self._prompt_coordinate_swapping(test_case)
        elif subtask == 'non_existent_location_detection':
            return self._prompt_non_existent(test_case)
        elif subtask == 'plausibility_checking':
            return self._prompt_plausibility(test_case)
        elif subtask == 'fabrication_detection':
            return self._prompt_fabrication(test_case)
        else:
            raise ValueError(f"Unknown subtask: {subtask}")

    def _prompt_cross_location(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in geographic coordinates and location knowledge.
Verify whether coordinates match claimed locations."""

        user_prompt = f"""{test_case['question']}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [YES or NO]

Example: FINAL ANSWER: NO"""

        return system_prompt, user_prompt

    def _prompt_coordinate_swapping(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in GPS coordinate validation and location verification.
Validate coordinates and identify errors."""

        user_prompt = f"""{test_case['question']}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [YES or NO]

Example: FINAL ANSWER: NO"""

        return system_prompt, user_prompt

    def _prompt_non_existent(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in geographic knowledge.
Recognize fictional locations and ambiguous place names."""

        user_prompt = f"""{test_case['question']}

If the location doesn't exist or is fictional, say so.
If the name is ambiguous (multiple places with same name), ask for clarification.

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [coordinates OR "DOES NOT EXIST" OR "AMBIGUOUS"]

Example: FINAL ANSWER: DOES NOT EXIST
Example: FINAL ANSWER: AMBIGUOUS"""

        return system_prompt, user_prompt

    def _prompt_plausibility(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in geographic plausibility checking.
Determine if coordinates are plausible for claimed locations."""

        user_prompt = f"""{test_case['question']}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [YES or NO]

Example: FINAL ANSWER: NO"""

        return system_prompt, user_prompt

    def _prompt_fabrication(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in geographic knowledge.
Provide coordinates only when you have reliable data. Express uncertainty when appropriate."""

        user_prompt = f"""{test_case['question']}

If you don't have precise coordinate data for this location, say so clearly.

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [coordinates OR "UNCERTAIN/NO DATA"]

Example: FINAL ANSWER: 40.7128, -74.0060
Example: FINAL ANSWER: UNCERTAIN"""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return evaluate_mc_response(response, test_case)

        # Free-form evaluation
        subtask = test_case['subtask']

        if subtask == 'cross_location_attribution':
            return self._evaluate_cross_location(response, test_case)
        elif subtask == 'coordinate_swapping_misattribution':
            return self._evaluate_coordinate_swapping(response, test_case)
        elif subtask == 'non_existent_location_detection':
            return self._evaluate_non_existent(response, test_case)
        elif subtask == 'plausibility_checking':
            return self._evaluate_plausibility(response, test_case)
        elif subtask == 'fabrication_detection':
            return self._evaluate_fabrication(response, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _evaluate_cross_location(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_answer = test_case['ground_truth']['answer']  # Should be False

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        predicted_answer = parse_yes_no(response_to_check)

        if predicted_answer is not None:
            correct = predicted_answer == expected_answer
        else:
            correct = False

        return {
            'correct': correct,
            'expected': expected_answer,
            'predicted': predicted_answer,
            'explanation': test_case['ground_truth']['explanation'],
            'raw_response': response
        }

    def _evaluate_coordinate_swapping(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_answer = test_case['ground_truth']['answer']  # Should be False

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        predicted_answer = parse_yes_no(response_to_check)

        if predicted_answer is not None:
            correct = predicted_answer == expected_answer
        else:
            correct = False

        return {
            'correct': correct,
            'expected': expected_answer,
            'predicted': predicted_answer,
            'reason': test_case['ground_truth']['reason'],
            'raw_response': response
        }

    def _evaluate_non_existent(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_answer = test_case['ground_truth']['answer']  # 'does_not_exist' or 'ambiguous'

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        response_lower = response_to_check.lower()

        # Check if model correctly identifies the issue
        if expected_answer == 'does_not_exist':
            correct = any(word in response_lower for word in ['does not exist', 'fictional', 'not exist', 'imaginary', 'not real'])
        elif expected_answer == 'ambiguous':
            correct = any(word in response_lower for word in ['ambiguous', 'multiple', 'clarification', 'which', 'several'])
        else:
            correct = False

        return {
            'correct': correct,
            'expected': expected_answer,
            'predicted': 'detected_issue' if correct else 'provided_coordinates',
            'raw_response': response
        }

    def _evaluate_plausibility(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_answer = test_case['ground_truth']['answer']  # Should be False

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        predicted_answer = parse_yes_no(response_to_check)

        if predicted_answer is not None:
            correct = predicted_answer == expected_answer
        else:
            correct = False

        return {
            'correct': correct,
            'expected': expected_answer,
            'predicted': predicted_answer,
            'explanation': test_case['ground_truth']['explanation'],
            'raw_response': response
        }

    def _evaluate_fabrication(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        should_express_uncertainty = test_case['ground_truth']['should_express_uncertainty']

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        response_lower = response_to_check.lower()

        # Check if model expresses uncertainty
        expresses_uncertainty = any(word in response_lower for word in [
            'uncertain', 'don\'t have', 'cannot provide', 'not have precise',
            'no data', 'unknown', 'unclear', 'cannot determine'
        ])

        correct = expresses_uncertainty == should_express_uncertainty

        return {
            'correct': correct,
            'expected': 'should_express_uncertainty' if should_express_uncertainty else 'should_provide_coordinates',
            'predicted': 'expressed_uncertainty' if expresses_uncertainty else 'provided_coordinates',
            'raw_response': response
        }


# =============================================================================
# TASK 5: UNIT CONFUSION HANDLING
# =============================================================================

class UnitConfusionEvaluator(LayerEEvaluator):
    """Evaluator for Task 5: Unit Confusion Handling"""

    def __init__(self):
        super().__init__("unit_confusion")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return format_mc_question(test_case)

        # Free-form format
        subtask = test_case['subtask']

        if subtask == 'distance_unit_detection':
            return self._prompt_distance_unit(test_case)
        elif subtask == 'angular_unit_confusion':
            return self._prompt_angular_unit(test_case)
        elif subtask == 'precision_unit_mismatch':
            return self._prompt_precision_unit(test_case)
        elif subtask == 'coordinate_unit_conversion':
            return self._prompt_coordinate_conversion(test_case)
        else:
            raise ValueError(f"Unknown subtask: {subtask}")

    def _prompt_distance_unit(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in distance measurements and unit interpretation.
Determine the most likely unit for distance measurements."""

        user_prompt = f"""{test_case['question']}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [unit name]

Example: FINAL ANSWER: kilometers"""

        return system_prompt, user_prompt

    def _prompt_angular_unit(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in angular measurements and coordinate systems.
Convert coordinates from different angular units (radians, gradians) to decimal degrees."""

        user_prompt = f"""{test_case['question']}

Recall:
- 180 degrees = π radians
- 360 degrees = 400 gradians

IMPORTANT: You may show your conversion, but you MUST end your response with:

FINAL ANSWER: [location name], [latitude in degrees], [longitude in degrees]

Example: FINAL ANSWER: Paris, 48.8566, 2.3522"""

        return system_prompt, user_prompt

    def _prompt_precision_unit(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in GPS precision and measurement units.
Determine the most likely unit for precision measurements."""

        user_prompt = f"""{test_case['question']}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [unit], [uncertainty in meters]

Example: FINAL ANSWER: degrees, 5500"""

        return system_prompt, user_prompt

    def _prompt_coordinate_conversion(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are a geospatial AI assistant. Convert coordinates between different units and identify locations."""

        user_prompt = f"""{test_case['question']}

Recall:
- 180 degrees = π radians
- 360 degrees = 400 gradians

IMPORTANT: You may show your conversion process, but you MUST end your response with:

FINAL ANSWER: [location name], [latitude in degrees], [longitude in degrees]

Example: FINAL ANSWER: Paris, 48.8566, 2.3522"""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return evaluate_mc_response(response, test_case)

        # Free-form evaluation
        subtask = test_case['subtask']

        if subtask == 'distance_unit_detection':
            return self._evaluate_distance_unit(response, test_case)
        elif subtask == 'angular_unit_confusion':
            return self._evaluate_angular_unit(response, test_case)
        elif subtask == 'precision_unit_mismatch':
            return self._evaluate_precision_unit(response, test_case)
        elif subtask == 'coordinate_unit_conversion':
            return self._evaluate_coordinate_conversion(response, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _evaluate_distance_unit(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_unit = test_case['ground_truth']['most_likely_unit']

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        response_lower = response_to_check.lower()

        # Check if expected unit is mentioned
        correct = expected_unit.lower() in response_lower

        return {
            'correct': correct,
            'expected': expected_unit,
            'predicted': response_to_check,
            'raw_response': response
        }

    def _evaluate_angular_unit(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_lat = test_case['ground_truth']['lat_degrees']
        expected_lon = test_case['ground_truth']['lon_degrees']
        tolerance = test_case['ground_truth'].get('tolerance', 0.01)

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        numbers = extract_numbers(response_to_check)

        if len(numbers) >= 2:
            # Take last two numbers as lat, lon
            predicted_lat = numbers[-2]
            predicted_lon = numbers[-1]
            correct = (abs(predicted_lat - expected_lat) <= tolerance and
                      abs(predicted_lon - expected_lon) <= tolerance)
        else:
            predicted_lat = None
            predicted_lon = None
            correct = False

        return {
            'correct': correct,
            'expected': {'lat': expected_lat, 'lon': expected_lon},
            'predicted': {'lat': predicted_lat, 'lon': predicted_lon} if predicted_lat else None,
            'tolerance': tolerance,
            'raw_response': response
        }

    def _evaluate_precision_unit(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_unit = test_case['ground_truth']['likely_unit']
        expected_meters = test_case['ground_truth']['precision_in_meters']

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        response_lower = response_to_check.lower()

        # Check if expected unit is mentioned
        unit_correct = expected_unit.lower() in response_lower

        # Check if precision in meters is approximately correct
        numbers = extract_numbers(response_to_check)
        if numbers:
            # Allow 50% tolerance on precision estimate
            meters_correct = any(abs(n - expected_meters) <= expected_meters * 0.5 for n in numbers)
        else:
            meters_correct = False

        correct = unit_correct and meters_correct

        return {
            'correct': correct,
            'expected_unit': expected_unit,
            'expected_meters': expected_meters,
            'unit_correct': unit_correct,
            'meters_correct': meters_correct,
            'raw_response': response
        }

    def _evaluate_coordinate_conversion(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        expected_lat = test_case['ground_truth']['lat']
        expected_lon = test_case['ground_truth']['lon']
        expected_location = test_case['ground_truth']['location']
        tolerance = test_case['ground_truth'].get('tolerance', 0.01)

        final_answer = extract_final_answer(response)
        response_to_check = final_answer if final_answer else response

        # Check coordinates
        numbers = extract_numbers(response_to_check)
        if len(numbers) >= 2:
            # Take last two numbers as lat, lon
            predicted_lat = numbers[-2]
            predicted_lon = numbers[-1]
            coords_correct = (abs(predicted_lat - expected_lat) <= tolerance and
                             abs(predicted_lon - expected_lon) <= tolerance)
        else:
            predicted_lat = None
            predicted_lon = None
            coords_correct = False

        # Check location identification (normalize for comparison)
        expected_normalized = self._normalize_location_name(expected_location)
        predicted_normalized = self._normalize_location_name(response_to_check)
        location_correct = expected_normalized in predicted_normalized

        # Both must be correct
        correct = coords_correct and location_correct

        return {
            'correct': correct,
            'coords_correct': coords_correct,
            'location_correct': location_correct,
            'expected': {'lat': expected_lat, 'lon': expected_lon, 'location': expected_location},
            'predicted': {
                'lat': predicted_lat,
                'lon': predicted_lon,
                'location_match': location_correct
            } if predicted_lat else None,
            'tolerance': tolerance,
            'raw_response': response
        }

    def _normalize_location_name(self, name: str) -> str:
        """Normalize location name for comparison (lowercase, remove punctuation)"""
        import unicodedata
        # Normalize unicode characters
        name = unicodedata.normalize('NFKD', name)
        # Convert to lowercase and remove non-alphanumeric
        name = re.sub(r'[^a-z0-9\s]', '', name.lower())
        # Remove extra whitespace
        name = ' '.join(name.split())
        return name


# =============================================================================
# EVALUATOR FACTORY
# =============================================================================

def get_evaluator(task_name: str) -> LayerEEvaluator:
    """
    Factory function to get evaluator for a specific Layer E task

    Args:
        task_name: Name of the task

    Returns:
        TaskEvaluator instance

    Raises:
        ValueError: If task_name is not recognized
    """
    evaluators = {
        'noise_tolerance': NoiseToleranceEvaluator(),
        'missing_data': MissingDataEvaluator(),
        'consistency': ConsistencyEvaluator(),
        'hallucination_detection': HallucinationDetectionEvaluator(),
        'unit_confusion': UnitConfusionEvaluator()
    }

    if task_name not in evaluators:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(evaluators.keys())}")

    return evaluators[task_name]
