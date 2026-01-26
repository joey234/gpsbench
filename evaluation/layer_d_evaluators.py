"""
Evaluators for GPSBench Layer D: Spatial Reasoning & Relations

Provides evaluation functions for:
1. Relative Position & Direction
2. Proximity & Nearest Neighbor
3. Route & Path Analysis
4. Boundary & Region Analysis
5. Spatial Patterns & Clustering
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from difflib import SequenceMatcher

# Multiple choice utilities
from .mc_utils import is_mc_test, evaluate_mc_response, format_mc_question


class TextParser:
    """Utilities for parsing responses"""

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
    def extract_yes_no(text: str) -> Optional[bool]:
        """
        Extract yes/no answer from text
        Returns: True (yes), False (no), or None (unclear)

        PRIORITY: Looks for "FINAL ANSWER:" first
        """
        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = TextParser.extract_final_answer(text)
        if final_answer:
            result = TextParser._parse_yes_no_from_text(final_answer)
            if result is not None:
                return result

        # STAGE 2: Fallback to parsing entire response
        return TextParser._parse_yes_no_from_text(text)

    @staticmethod
    def _parse_yes_no_from_text(text: str) -> Optional[bool]:
        """Internal helper to parse yes/no from text"""
        text_lower = text.lower()

        # Check for explicit answers
        yes_keywords = ['yes', 'correct', 'true', 'affirmative']
        no_keywords = ['no', 'incorrect', 'false', 'negative']

        # Check for negation patterns
        if any(neg in text_lower for neg in ['is not', 'not', "isn't", "aren't", "doesn't"]):
            # Look at what comes after negation
            if any(kw in text_lower for kw in yes_keywords):
                return False  # "not yes" = no
            if any(kw in text_lower for kw in no_keywords):
                return True  # "not no" = yes (double negative)
            return False  # General negation = no

        # Positive keywords
        if any(kw in text_lower for kw in yes_keywords):
            return True
        if any(kw in text_lower for kw in no_keywords):
            return False

        return None

    @staticmethod
    def extract_location_name(text: str, candidates: List[str]) -> Optional[str]:
        """
        Extract location name from text using fuzzy matching

        Args:
            text: Response text
            candidates: List of possible location names

        Returns:
            Best matching candidate or None

        PRIORITY: Looks for "FINAL ANSWER:" first
        """
        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = TextParser.extract_final_answer(text)
        if final_answer:
            result = TextParser._parse_location_name_from_text(final_answer, candidates)
            if result is not None:
                return result

        # STAGE 2: Fallback to parsing entire response
        return TextParser._parse_location_name_from_text(text, candidates)

    @staticmethod
    def _parse_location_name_from_text(text: str, candidates: List[str]) -> Optional[str]:
        """Internal helper to parse location name from text"""
        text_lower = text.lower()

        # Try exact match first
        for candidate in candidates:
            if candidate.lower() in text_lower:
                return candidate

        # Try fuzzy matching
        best_match = None
        best_ratio = 0.0

        for candidate in candidates:
            # Calculate similarity
            ratio = SequenceMatcher(None, candidate.lower(), text_lower).ratio()

            # Also check if candidate words appear in text
            candidate_words = set(candidate.lower().split())
            text_words = set(text_lower.split())
            word_overlap = len(candidate_words & text_words) / len(candidate_words) if candidate_words else 0

            combined_score = max(ratio, word_overlap)

            if combined_score > best_ratio:
                best_ratio = combined_score
                best_match = candidate

        # Return if similarity is high enough
        if best_ratio > 0.6:
            return best_match

        return None

    @staticmethod
    def extract_location_list(text: str, candidates: List[str]) -> List[str]:
        """
        Extract list of locations from text
        
        Returns:
            List of matched candidate names
        """
        matches = []
        text_lower = text.lower()

        for candidate in candidates:
            if candidate.lower() in text_lower:
                matches.append(candidate)

        return matches

    @staticmethod
    def extract_ordered_list(text: str, candidates: List[str]) -> Optional[List[str]]:
        """
        Extract ordered list from text

        Handles formats like:
        - "1. A, 2. B, 3. C"
        - "A, B, C"
        - "A then B then C"
        - "A → B → C"

        Returns:
            Ordered list of candidate names or None

        PRIORITY: Looks for "FINAL ANSWER:" first
        """
        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = TextParser.extract_final_answer(text)
        if final_answer:
            result = TextParser._parse_ordered_list_from_text(final_answer, candidates)
            if result is not None:
                return result

        # STAGE 2: Fallback to parsing entire response
        return TextParser._parse_ordered_list_from_text(text, candidates)

    @staticmethod
    def _parse_ordered_list_from_text(text: str, candidates: List[str]) -> Optional[List[str]]:
        """Internal helper to parse ordered list from text"""
        text_lower = text.lower()

        # Try to find candidates in order they appear
        positions = []
        for candidate in candidates:
            idx = text_lower.find(candidate.lower())
            if idx != -1:
                positions.append((idx, candidate))

        # Sort by position in text
        positions.sort(key=lambda x: x[0])
        ordered = [cand for _, cand in positions]

        # Only return if we found all candidates
        if len(ordered) == len(candidates):
            return ordered

        return None

    @staticmethod
    def extract_number(text: str) -> Optional[int]:
        """
        Extract a number from text

        PRIORITY: Looks for "FINAL ANSWER:" first
        """
        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = TextParser.extract_final_answer(text)
        if final_answer:
            result = TextParser._parse_number_from_text(final_answer)
            if result is not None:
                return result

        # STAGE 2: Fallback to parsing entire response
        return TextParser._parse_number_from_text(text)

    @staticmethod
    def _parse_number_from_text(text: str) -> Optional[int]:
        """Internal helper to parse number from text"""
        # Look for numbers
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            return int(numbers[0])

        # Look for word numbers
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }

        for word, num in word_to_num.items():
            if word in text.lower():
                return num

        return None


# ===== Task-Specific Evaluators =====

class RelativePositionEvaluator:
    """Evaluator for Task 1: Relative Position & Direction"""

    def __init__(self):
        self.parser = TextParser()

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        """Create system and user prompts"""
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return format_mc_question(test_case)

        # Free-form format
        system_prompt = """You are an expert in geography and spatial relationships.
Answer questions about relative positions of locations clearly and concisely.
When answering yes/no questions, state your answer clearly.
When ordering locations, provide a clear ordered list."""

        user_prompt = test_case['question']

        # Add coordinate context if helpful
        if 'point_a' in test_case:
            user_prompt += f"\n\n{test_case['point_a']['name']}: ({test_case['point_a']['lat']}, {test_case['point_a']['lon']})"
        if 'point_b' in test_case:
            user_prompt += f"\n{test_case['point_b']['name']}: ({test_case['point_b']['lat']}, {test_case['point_b']['lon']})"
        if 'points' in test_case:
            user_prompt += "\n\nLocations:"
            for point in test_case['points']:
                user_prompt += f"\n- {point['name']}: ({point['lat']}, {point['lon']})"

        # Add FINAL ANSWER format enforcement based on subtask
        subtask = test_case.get('subtask', '')
        if subtask == 'ordering':
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location1, location2, location3, ...]

Provide only the ordered list in the FINAL ANSWER line."""
        elif subtask in ['cardinal_direction', 'between_relationship']:
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: YES
(or FINAL ANSWER: NO)

Provide only YES or NO in the FINAL ANSWER line."""
        # Removed: comparative_distance (redundant with Task 2 nearest_point)

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate LLM response"""
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return evaluate_mc_response(response, test_case)

        # Free-form evaluation
        subtask = test_case['subtask']

        if subtask == 'cardinal_direction':
            return self._evaluate_cardinal_direction(response, test_case)
        elif subtask == 'between_relationship':
            return self._evaluate_between(response, test_case)
        elif subtask == 'ordering':
            return self._evaluate_ordering(response, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _evaluate_cardinal_direction(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate yes/no cardinal direction question"""
        expected_answer = test_case['ground_truth']['answer']
        model_answer = self.parser.extract_yes_no(response)

        correct = model_answer == expected_answer if model_answer is not None else False

        return {
            'correct': correct,
            'expected_answer': expected_answer,
            'model_answer': model_answer,
            'explanation': test_case['ground_truth'].get('explanation', '')
        }

    # Removed: _evaluate_comparative_distance (redundant with Task 2 nearest_point)

    def _evaluate_between(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate between relationship"""
        expected_answer = test_case['ground_truth']['answer']
        model_answer = self.parser.extract_yes_no(response)

        correct = model_answer == expected_answer if model_answer is not None else False

        return {
            'correct': correct,
            'expected_answer': expected_answer,
            'model_answer': model_answer
        }

    def _evaluate_ordering(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate ordering question"""
        expected_order = test_case['ground_truth']['correct_order']
        candidates = [p['name'] for p in test_case['points']]

        model_order = self.parser.extract_ordered_list(response, candidates)

        # Check if order matches
        correct = model_order == expected_order if model_order is not None else False

        # Partial credit for getting some positions correct
        partial_score = 0.0
        if model_order and len(model_order) == len(expected_order):
            matches = sum(1 for i in range(len(expected_order)) if model_order[i] == expected_order[i])
            partial_score = matches / len(expected_order)

        return {
            'correct': correct,
            'expected_order': expected_order,
            'model_order': model_order,
            'partial_score': partial_score
        }


class ProximityEvaluator:
    """Evaluator for Task 2: Proximity & Nearest Neighbor"""

    def __init__(self):
        self.parser = TextParser()

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        """Create system and user prompts"""
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return format_mc_question(test_case)

        # Free-form format
        system_prompt = """You are an expert in geography and spatial analysis.
Answer questions about distances and proximity between locations.
Provide clear, direct answers."""

        user_prompt = test_case['question']

        # Add coordinate information
        if 'reference_point' in test_case:
            user_prompt += f"\n\nReference: {test_case['reference_point']['name']} ({test_case['reference_point']['lat']}, {test_case['reference_point']['lon']})"

        if 'candidate_points' in test_case:
            user_prompt += "\n\nCandidates:"
            for point in test_case['candidate_points']:
                user_prompt += f"\n- {point['name']}: ({point['lat']}, {point['lon']})"

        # Add FINAL ANSWER format enforcement based on subtask
        subtask = test_case.get('subtask', '')
        if subtask in ['nearest_point', 'farthest_point']:
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location name]

Provide only the location name in the FINAL ANSWER line."""
        elif subtask == 'within_radius':
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location1, location2, location3, ...]

Provide only the comma-separated list of locations in the FINAL ANSWER line."""
        elif subtask == 'proximity_ranking':
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [closest, second, third, ..., farthest]

Provide only the ordered list from closest to farthest in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate LLM response"""
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return evaluate_mc_response(response, test_case)

        # Free-form evaluation
        subtask = test_case['subtask']

        if subtask == 'nearest_point':
            return self._evaluate_nearest(response, test_case)
        elif subtask == 'farthest_point':
            return self._evaluate_farthest(response, test_case)
        elif subtask == 'within_radius':
            return self._evaluate_within_radius(response, test_case)
        elif subtask == 'proximity_ranking':
            return self._evaluate_ranking(response, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _evaluate_nearest(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate nearest point identification"""
        expected_answer = test_case['ground_truth']['answer']
        candidates = [p['name'] for p in test_case['candidate_points']]

        model_answer = self.parser.extract_location_name(response, candidates)

        correct = model_answer == expected_answer if model_answer is not None else False

        return {
            'correct': correct,
            'expected_answer': expected_answer,
            'model_answer': model_answer,
            'distances': test_case['ground_truth']['all_distances_km']
        }

    def _evaluate_farthest(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate farthest point identification"""
        expected_answer = test_case['ground_truth']['answer']
        candidates = [p['name'] for p in test_case['candidate_points']]

        model_answer = self.parser.extract_location_name(response, candidates)

        correct = model_answer == expected_answer if model_answer is not None else False

        return {
            'correct': correct,
            'expected_answer': expected_answer,
            'model_answer': model_answer
        }

    def _evaluate_within_radius(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate within radius selection"""
        expected_answer = set(test_case['ground_truth']['answer'])
        candidates = [p['name'] for p in test_case['candidate_points']]

        model_answer = set(self.parser.extract_location_list(response, candidates))

        # Check if sets match
        correct = model_answer == expected_answer

        # Calculate precision and recall
        if model_answer:
            precision = len(model_answer & expected_answer) / len(model_answer)
            recall = len(model_answer & expected_answer) / len(expected_answer) if expected_answer else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1_score = 0.0

        return {
            'correct': correct,
            'expected_answer': list(expected_answer),
            'model_answer': list(model_answer),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def _evaluate_ranking(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate proximity ranking"""
        expected_order = test_case['ground_truth']['correct_order']
        candidates = [p['name'] for p in test_case['candidate_points']]

        model_order = self.parser.extract_ordered_list(response, candidates)

        correct = model_order == expected_order if model_order is not None else False

        # Calculate Kendall tau or position accuracy
        position_accuracy = 0.0
        if model_order and len(model_order) == len(expected_order):
            matches = sum(1 for i in range(len(expected_order)) if model_order[i] == expected_order[i])
            position_accuracy = matches / len(expected_order)

        return {
            'correct': correct,
            'expected_order': expected_order,
            'model_order': model_order,
            'position_accuracy': position_accuracy
        }


class RouteAnalysisEvaluator:
    """Evaluator for Task 3: Route & Path Analysis"""

    def __init__(self):
        self.parser = TextParser()

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        """Create system and user prompts"""
        system_prompt = """You are an expert in route planning and geographic pathfinding.
Answer questions about routes, waypoints, and travel paths clearly."""

        user_prompt = test_case['question']

        # Add location information
        if 'start' in test_case:
            user_prompt += f"\n\nStart: {test_case['start']['name']} ({test_case['start']['lat']}, {test_case['start']['lon']})"
        if 'end' in test_case:
            user_prompt += f"\nEnd: {test_case['end']['name']} ({test_case['end']['lat']}, {test_case['end']['lon']})"
        if 'waypoint' in test_case:
            user_prompt += f"\nWaypoint: {test_case['waypoint']['name']} ({test_case['waypoint']['lat']}, {test_case['waypoint']['lon']})"
        if 'points' in test_case:
            user_prompt += "\n\nRoute points:"
            for i, point in enumerate(test_case['points']):
                user_prompt += f"\n{i+1}. {point['name']}: ({point['lat']}, {point['lon']})"

        # Add FINAL ANSWER format enforcement based on subtask
        subtask = test_case.get('subtask', '')
        if subtask in ['on_route_detection', 'detour_detection']:
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: YES
(or FINAL ANSWER: NO)

Provide only YES or NO in the FINAL ANSWER line."""
        elif subtask == 'multi_leg_distance':
            user_prompt += """

IMPORTANT: You may show your calculations, but you MUST end your response with:

FINAL ANSWER: [distance in kilometers]

Example: FINAL ANSWER: 1520 km

Provide only the distance value in the FINAL ANSWER line."""
        elif subtask == 'waypoint_ordering':
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location1, location2, location3, ...]

Provide only the ordered list in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate LLM response"""
        subtask = test_case['subtask']

        if subtask == 'on_route_detection':
            return self._evaluate_on_route(response, test_case)
        elif subtask == 'detour_detection':
            return self._evaluate_detour(response, test_case)
        elif subtask == 'waypoint_ordering':
            return self._evaluate_waypoint_ordering(response, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _evaluate_on_route(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate on-route detection"""
        expected_answer = test_case['ground_truth']['answer']
        model_answer = self.parser.extract_yes_no(response)

        correct = model_answer == expected_answer if model_answer is not None else False

        return {
            'correct': correct,
            'expected_answer': expected_answer,
            'model_answer': model_answer,
            'detour_km': test_case['ground_truth']['detour_km']
        }

    def _evaluate_detour(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate detour detection"""
        expected_answer = test_case['ground_truth']['answer']
        model_answer = self.parser.extract_yes_no(response)

        correct = model_answer == expected_answer if model_answer is not None else False

        return {
            'correct': correct,
            'expected_answer': expected_answer,
            'model_answer': model_answer,
            'detour_percent': test_case['ground_truth']['detour_percent']
        }

    def _evaluate_multi_leg(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate multi-leg distance calculation"""
        from layer_c_evaluators import NumericalParser

        expected_distance = test_case['ground_truth']['total_distance_km']
        model_distance = NumericalParser.extract_distance(response)

        # Check if within tolerance
        tolerance_km = test_case['ground_truth']['tolerance_km']
        correct = False
        error_km = None

        if model_distance is not None:
            error_km = abs(model_distance - expected_distance)
            correct = error_km <= tolerance_km

        return {
            'correct': correct,
            'expected_distance_km': expected_distance,
            'model_distance_km': model_distance,
            'error_km': error_km,
            'tolerance_km': tolerance_km
        }

    def _evaluate_waypoint_ordering(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate waypoint ordering"""
        suggested_order = test_case['ground_truth']['suggested_order']
        candidates = [p['name'] for p in test_case['points']]

        model_order = self.parser.extract_ordered_list(response, candidates)

        # For TSP, we accept any reasonable order (not just the greedy one)
        # So we check if the model's order is "reasonable" rather than exact
        exact_match = model_order == suggested_order if model_order is not None else False

        return {
            'correct': exact_match,  # Strict for now, could be relaxed
            'suggested_order': suggested_order,
            'model_order': model_order,
            'note': 'Multiple valid orderings may exist'
        }


class BoundaryAnalysisEvaluator:
    """Evaluator for Task 4: Boundary & Region Analysis"""

    def __init__(self):
        self.parser = TextParser()

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        """Create system and user prompts"""
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return format_mc_question(test_case)

        # Free-form format
        system_prompt = """You are an expert in political geography and regional boundaries.
Answer questions about countries, regions, and geographic groupings.

Valid continents (use only these, not sub-regions):
- Africa
- Asia
- Europe
- North America
- South America
- Oceania"""

        user_prompt = test_case['question']

        # Add location information
        if 'points' in test_case:
            user_prompt += "\n\nLocations:"
            for point in test_case['points']:
                user_prompt += f"\n- {point['name']}: ({point['lat']}, {point['lon']})"

        # Add FINAL ANSWER format enforcement
        subtask = test_case.get('subtask', '')
        if subtask == 'regional_grouping':
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [number of distinct continents]

Example: FINAL ANSWER: 3

Use only the 6 valid continents listed above (not sub-regions like "East Asia" or "Western Europe").
Provide only the number of distinct continents in the FINAL ANSWER line."""
        else:
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [number]

Example: FINAL ANSWER: 3

Provide only the number in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate LLM response"""
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return evaluate_mc_response(response, test_case)

        # Free-form evaluation
        subtask = test_case['subtask']

        if subtask == 'regional_grouping':
            return self._evaluate_regional_grouping(response, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _evaluate_regional_grouping(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate regional grouping"""
        expected_num_continents = test_case['ground_truth']['num_continents']

        # Extract number of groups mentioned
        model_num = self.parser.extract_number(response)

        correct = model_num == expected_num_continents if model_num is not None else False

        return {
            'correct': correct,
            'expected_num_continents': expected_num_continents,
            'model_num_continents': model_num,
            'expected_grouping': test_case['ground_truth']['grouping']
        }


class SpatialPatternsEvaluator:
    """Evaluator for Task 5: Spatial Patterns & Clustering"""

    def __init__(self):
        self.parser = TextParser()

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        """Create system and user prompts"""
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return format_mc_question(test_case)

        # Free-form format
        system_prompt = """You are an expert in spatial analysis and pattern recognition.
Identify geographic patterns, clusters, and outliers in sets of locations."""

        user_prompt = test_case['question']

        # Add location information
        if 'points' in test_case:
            user_prompt += "\n\nLocations:"
            for point in test_case['points']:
                user_prompt += f"\n- {point['name']}: ({point['lat']}, {point['lon']})"

        # Add FINAL ANSWER format enforcement based on subtask
        subtask = test_case.get('subtask', '')
        if subtask == 'outlier_detection':
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [location name]

Example: FINAL ANSWER: Miami

Provide only the outlier location name in the FINAL ANSWER line."""
        elif subtask == 'cluster_identification':
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [number]

Example: FINAL ANSWER: 3

Provide only the number of clusters in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate LLM response"""
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return evaluate_mc_response(response, test_case)

        # Free-form evaluation
        subtask = test_case['subtask']

        if subtask == 'outlier_detection':
            return self._evaluate_outlier(response, test_case)
        elif subtask == 'cluster_identification':
            return self._evaluate_clusters(response, test_case)
        else:
            return {'correct': False, 'error': f'Unknown subtask: {subtask}'}

    def _evaluate_outlier(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate outlier detection"""
        expected_outlier = test_case['ground_truth']['answer']
        candidates = [p['name'] for p in test_case['points']]

        model_outlier = self.parser.extract_location_name(response, candidates)

        correct = model_outlier == expected_outlier if model_outlier is not None else False

        return {
            'correct': correct,
            'expected_outlier': expected_outlier,
            'model_outlier': model_outlier,
            'outlier_distance_km': test_case['ground_truth']['outlier_distance_km']
        }

    def _evaluate_clusters(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate cluster identification"""
        expected_num = test_case['ground_truth']['num_clusters']

        # Extract number of clusters mentioned
        model_num = self.parser.extract_number(response)

        correct = model_num == expected_num if model_num is not None else False

        return {
            'correct': correct,
            'expected_num_clusters': expected_num,
            'model_num_clusters': model_num,
            'actual_clusters': test_case['ground_truth']['clusters']
        }


# ===== Get Evaluator Function =====

def get_evaluator(task_name: str):
    """
    Get the appropriate evaluator for a task
    
    Args:
        task_name: Name of the task
    
    Returns:
        Evaluator instance
    """
    evaluators = {
        'relative_position': RelativePositionEvaluator,
        'proximity': ProximityEvaluator,
        'route_analysis': RouteAnalysisEvaluator,
        'boundary_analysis': BoundaryAnalysisEvaluator,
        'spatial_patterns': SpatialPatternsEvaluator
    }

    if task_name not in evaluators:
        raise ValueError(f"Unknown task: {task_name}")

    return evaluators[task_name]()
