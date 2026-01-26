"""
Task-specific evaluators for GPSBench Layer B
Each evaluator handles prompting and response evaluation for one task
"""

import re
import json
from typing import Dict, Any, Tuple, List, Optional
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from hierarchical_location_matcher import HierarchicalLocationMatcher
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
    """Base class for Layer B task evaluators"""

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


class PlaceAssociationEvaluator(TaskEvaluator):
    """Evaluator for Task 1: Place Association (Reverse Geocoding)"""

    def __init__(self):
        super().__init__("place_association")
        self.hierarchical_matcher = HierarchicalLocationMatcher()

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in geography and coordinate systems.
Your task is to identify what place or location corresponds to given coordinates.

Consider the precision level:
- High precision (6 decimals): Specific landmarks or buildings
- Medium precision (3-4 decimals): Neighborhoods, streets, or districts
- Low precision (1-2 decimals): Cities or regions
- Very low precision (0 decimals): Countries or large regions

Provide the most specific location that the coordinate precision supports."""

        coord_str = test_case['coordinate_string']
        subtask = test_case.get('subtask', 'identification')

        if subtask == 'landmark_identification':
            user_prompt = f"""What famous landmark is located at these coordinates: {coord_str}?

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [landmark name]

Example: FINAL ANSWER: Eiffel Tower, Paris

Provide only the place name in the FINAL ANSWER line."""
        elif subtask == 'city_identification':
            user_prompt = f"""What city is located at these coordinates: {coord_str}?

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [city name]

Example: FINAL ANSWER: Paris

Provide only the city name in the FINAL ANSWER line."""
        else:
            user_prompt = f"""What place is located at these coordinates: {coord_str}?

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [place name]

Example: FINAL ANSWER: London

Provide only the place name in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = test_case['ground_truth']
        # Support both 'name' (old) and 'location_name' (new) fields
        expected_name = ground_truth.get('location_name') or ground_truth.get('name', '')
        granularity_level = ground_truth.get('granularity_level', 'unknown')

        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            # STAGE 2: Fallback to full response
            response_to_check = response.strip()

        response_clean = response_to_check.lower().strip()
        expected_lower = expected_name.lower().strip()

        # Check for exact match (strict - requires substantial overlap)
        # For compound names like "Eiffel Tower, Paris", require the main part to match
        # Split on comma and check first part (the most specific location)
        expected_parts = [p.strip() for p in expected_lower.split(',')]
        response_parts = [p.strip() for p in response_clean.split(',')]

        # Exact match if:
        # 1. Full name match, OR
        # 2. First part (most specific) matches exactly
        exact_match = (expected_lower == response_clean or
                      expected_lower in response_clean or
                      (len(expected_parts) > 0 and expected_parts[0] in response_clean and
                       len(expected_parts[0]) > 5))  # Avoid matching short words

        if exact_match:
            return {
                'correct': True,
                'match_type': 'exact',
                'expected': expected_name,
                'predicted': response_to_check,
                'granularity_level': granularity_level,
                'raw_response': response
            }

        # Check alternate names
        alternate_names = ground_truth.get('alternate_names', [])
        for alt_name in alternate_names:
            if alt_name.lower() in response_clean:
                return {
                    'correct': True,
                    'match_type': 'alternate',
                    'expected': expected_name,
                    'predicted': response_to_check,
                    'granularity_level': granularity_level,
                    'raw_response': response
                }

        # Use hierarchical location matcher - returns separate scores for each level
        hierarchical_result = self.hierarchical_matcher.match_with_hierarchy(
            response_to_check, ground_truth
        )

        # If any hierarchical match found (but not exact, since we checked that already)
        if hierarchical_result['any_match']:
            return {
                'correct': hierarchical_result['exact_location_match'],
                'match_type': 'hierarchical',
                'expected': expected_name,
                'predicted': response_to_check,
                'granularity_level': granularity_level,
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

        # Fuzzy matching for typos/variations
        similarity = SequenceMatcher(None, expected_lower, response_clean).ratio()
        if similarity > 0.8:
            return {
                'correct': True,
                'match_type': 'fuzzy',
                'similarity': similarity,
                'expected': expected_name,
                'predicted': response_to_check,
                'granularity_level': granularity_level,
                'raw_response': response
            }

        # No match
        return {
            'correct': False,
            'match_type': 'none',
            'expected': expected_name,
            'predicted': response_to_check,
            'granularity_level': granularity_level,
            'raw_response': response
        }


class SemanticValidationEvaluator(TaskEvaluator):
    """Evaluator for Task 2: Semantic Validation (Plausibility Detection)"""

    def __init__(self):
        super().__init__("semantic_validation")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return format_mc_question(test_case)

        # Free-form format
        system_prompt = """You are an expert in geography and spatial reasoning.
Your task is to determine if a given location and context combination is plausible.

Consider:
- Is the location on land or water?
- Does the context make sense for the terrain?
- Are political boundaries respected?
- Is the context realistic for the geographic region?"""

        coord_str = test_case['coordinate_string']
        context = test_case['context']

        user_prompt = f"""Is it plausible that a {context} is located at these coordinates: {coord_str}?

Determine if this combination makes geographic sense.

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: PLAUSIBLE
(or FINAL ANSWER: IMPLAUSIBLE)

Provide only PLAUSIBLE or IMPLAUSIBLE in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return evaluate_mc_response(response, test_case)

        # Free-form evaluation
        expected_plausible = test_case['ground_truth']['is_plausible']

        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            # STAGE 2: Fallback to full response
            response_to_check = response

        response_lower = response_to_check.lower()

        # Determine if model said plausible or implausible
        # Check 'implausible' FIRST (since 'plausible' is substring of 'implausible')
        if 'implausible' in response_lower or 'not plausible' in response_lower:
            model_says_plausible = False
        elif 'plausible' in response_lower:
            model_says_plausible = True
        else:
            # Fallback: check for other keywords
            # Check negative keywords FIRST (before checking positive) to avoid substring issues
            # e.g., "impossible" contains "possible"
            negative_keywords = ['impossible', 'cannot', 'unrealistic', 'unlikely', 'invalid', 'incorrect']
            positive_keywords = ['reasonable', 'realistic', 'likely', 'valid', 'correct']  # removed 'possible' to avoid confusion

            # Use word boundaries to avoid substring matches
            has_negative = any(kw in response_lower for kw in negative_keywords)
            has_positive = any(kw in response_lower for kw in positive_keywords) and not has_negative

            if has_negative and not has_positive:
                model_says_plausible = False
            elif has_positive and not has_negative:
                model_says_plausible = True
            else:
                # Ambiguous, default to opposite of expected (so it's marked wrong)
                model_says_plausible = not expected_plausible

        correct = model_says_plausible == expected_plausible

        return {
            'correct': correct,
            'expected_plausible': expected_plausible,
            'predicted_plausible': model_says_plausible,
            'validation_type': test_case['ground_truth']['validation_type'],
            'context': test_case['context'],
            'raw_response': response
        }


class RegionalKnowledgeEvaluator(TaskEvaluator):
    """Evaluator for Task 3: Regional Knowledge"""

    def __init__(self):
        super().__init__("regional_knowledge")

        # Common alternate country names
        self.country_alternates = {
            'united states': ['usa', 'us', 'america', 'united states of america'],
            'united kingdom': ['uk', 'britain', 'great britain', 'england'],
            'south korea': ['korea', 'republic of korea', 'rok'],
            'north korea': ['dprk', 'democratic people\'s republic of korea'],
            'china': ['prc', 'people\'s republic of china'],
            'netherlands': ['holland'],
            'myanmar': ['burma'],
            'czechia': ['czech republic'],
            # Add more as needed
        }

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return format_mc_question(test_case)

        # Free-form format
        system_prompt = """You are an expert in political geography and coordinate systems.
Answer questions about which countries, continents, or regions contain given coordinates.

Be precise and consider political boundaries carefully."""

        coord_str = test_case['coordinate_string']
        question = test_case['question']
        subtask = test_case.get('subtask', 'country_membership')

        user_prompt = f"""Coordinates: {coord_str}

Question: {question}"""

        # Add FINAL ANSWER format enforcement based on subtask
        if subtask == 'country_membership':
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: YES
(or FINAL ANSWER: NO)

Provide only YES or NO in the FINAL ANSWER line."""
        elif subtask == 'continent_identification':
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [continent name]

Example: FINAL ANSWER: Asia

Provide only the continent name in the FINAL ANSWER line."""
        elif subtask == 'border_proximity':
            user_prompt += """

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [country name]

Example: FINAL ANSWER: France

Provide only the country name in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return evaluate_mc_response(response, test_case)

        # Free-form evaluation
        ground_truth = test_case['ground_truth']
        subtask = test_case.get('subtask', 'country_membership')

        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            # STAGE 2: Fallback to full response
            response_to_check = response

        response_lower = response_to_check.lower().strip()

        if subtask == 'country_membership':
            # Yes/No question
            expected_answer = ground_truth['answer']

            # Check for yes/no in response
            has_yes = 'yes' in response_lower
            has_no = 'no' in response_lower

            if has_yes and not has_no:
                model_answer = True
            elif has_no and not has_yes:
                model_answer = False
            else:
                # Ambiguous - try to infer
                # If expected country name appears, probably yes
                if 'country' in ground_truth:
                    country_lower = ground_truth['country'].lower()
                    model_answer = country_lower in response_lower
                else:
                    model_answer = not expected_answer  # Mark as wrong

            correct = model_answer == expected_answer

            return {
                'correct': correct,
                'expected_answer': expected_answer,
                'predicted_answer': model_answer,
                'subtask': subtask,
                'raw_response': response
            }

        elif subtask == 'continent_identification':
            expected_continent = ground_truth['continent']

            # Continent variations
            continent_map = {
                'africa': ['africa', 'african'],
                'asia': ['asia', 'asian'],
                'europe': ['europe', 'european'],
                'north america': ['north america', 'northern america'],
                'south america': ['south america', 'southern america'],
                'oceania': ['oceania', 'australia', 'australasia'],
                'antarctica': ['antarctica', 'antarctic']
            }

            # Check if expected continent appears in response
            expected_lower = expected_continent.lower()
            correct = False

            if expected_lower in continent_map:
                for variant in continent_map[expected_lower]:
                    if variant in response_lower:
                        correct = True
                        break
            else:
                correct = expected_lower in response_lower

            return {
                'correct': correct,
                'expected_continent': expected_continent,
                'predicted': response_to_check.strip(),
                'subtask': subtask,
                'raw_response': response
            }

        elif subtask == 'border_proximity':
            # Country identification question
            expected_country = ground_truth['country']
            expected_lower = expected_country.lower()

            # Check for exact match or alternates
            correct = expected_lower in response_lower

            # Check country alternates
            for country_key, alternates in self.country_alternates.items():
                if country_key in expected_lower or expected_lower in country_key:
                    if any(alt in response_lower for alt in alternates):
                        correct = True
                        break

            # If near border, be more lenient
            if ground_truth.get('is_near_border', False) and not correct:
                # Could be neighboring country - partial credit
                return {
                    'correct': False,
                    'partial_credit': 0.5,
                    'expected_country': expected_country,
                    'predicted': response_to_check.strip(),
                    'note': 'Near border - neighboring country possible',
                    'subtask': subtask,
                    'raw_response': response
                }

            return {
                'correct': correct,
                'expected_country': expected_country,
                'predicted': response_to_check.strip(),
                'subtask': subtask,
                'raw_response': response
            }

        else:
            # Unknown subtask
            return {
                'correct': False,
                'error': f'Unknown subtask: {subtask}',
                'raw_response': response
            }


class ContextGroundingEvaluator(TaskEvaluator):
    """Evaluator for Task 4: Context-Appropriate Grounding"""

    def __init__(self):
        super().__init__("context_grounding")

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = """You are an expert in GPS precision and geographic description.
Your task is to describe a location with appropriate specificity based on coordinate precision.

Precision guidelines:
- 0 decimals (~111 km): Country or large region
- 1 decimal (~11 km): State or province
- 2 decimals (~1.1 km): City
- 3 decimals (~111 m): Neighborhood or district
- 4 decimals (~11 m): Street or building
- 5+ decimals (~1 m): Specific landmark

Match your description specificity to the coordinate precision.
Do NOT over-specify (claiming to identify a building when precision only supports city-level).
Do NOT under-specify (saying "Earth" for high-precision coordinates)."""

        coord_str = test_case['coordinate_string']
        decimal_places = test_case['decimal_places']

        user_prompt = f"""Describe the location at these coordinates: {coord_str}

The coordinates have {decimal_places} decimal place(s).
Provide an appropriate description based on this precision level.

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [precision-appropriate description]

Example: FINAL ANSWER: Manhattan, New York City

Provide only the location description in the FINAL ANSWER line."""

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = test_case['ground_truth']
        decimal_places = test_case['decimal_places']

        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            # STAGE 2: Fallback to full response
            response_to_check = response

        response_lower = response_to_check.lower()

        # Expected precision level
        expected_level = ground_truth['precision_level']
        appropriate_levels = ground_truth['appropriate_levels']
        too_specific = ground_truth.get('too_specific', [])
        too_vague = ground_truth.get('too_vague', [])

        # Check if response mentions appropriate descriptions
        mentioned_appropriate = any(level in response_lower for level in appropriate_levels)

        # Check if example appropriate description is mentioned
        example_appropriate = ground_truth.get('example_appropriate', '')
        if example_appropriate:
            # Extract key terms (e.g., "Paris" from "Paris, France")
            key_terms = [term.strip().lower() for term in example_appropriate.split(',')]
            mentions_example = any(term in response_lower for term in key_terms if len(term) > 3)
        else:
            mentions_example = False

        # Check for over-specification
        over_specified = any(level in response_lower for level in too_specific)

        # Check for under-specification
        under_specified = any(level in response_lower for level in too_vague)

        # Determine correctness
        # Correct if: mentions appropriate level OR mentions example, AND doesn't over/under specify
        appropriate = mentioned_appropriate or mentions_example
        incorrect_specificity = over_specified or under_specified

        correct = appropriate and not incorrect_specificity

        # Additional analysis: extract what the model described
        predicted_level = self._infer_description_level(response_lower, decimal_places)

        return {
            'correct': correct,
            'expected_level': expected_level,
            'predicted_level': predicted_level,
            'decimal_places': decimal_places,
            'appropriate': appropriate,
            'over_specified': over_specified,
            'under_specified': under_specified,
            'mentions_example': mentions_example,
            'raw_response': response
        }

    def _infer_description_level(self, response_lower: str, decimal_places: int) -> str:
        """Infer what level of description the model provided"""

        # Keywords for different levels
        landmark_keywords = ['landmark', 'building', 'monument', 'tower', 'statue', 'specific']
        street_keywords = ['street', 'road', 'avenue', 'address', 'block']
        neighborhood_keywords = ['neighborhood', 'district', 'area', 'quarter']
        city_keywords = ['city', 'town', 'municipality']
        region_keywords = ['region', 'state', 'province', 'county']
        country_keywords = ['country', 'nation']
        vague_keywords = ['continent', 'hemisphere', 'planet', 'earth', 'world']

        if any(kw in response_lower for kw in landmark_keywords):
            return 'landmark'
        elif any(kw in response_lower for kw in street_keywords):
            return 'street'
        elif any(kw in response_lower for kw in neighborhood_keywords):
            return 'neighborhood'
        elif any(kw in response_lower for kw in city_keywords):
            return 'city'
        elif any(kw in response_lower for kw in region_keywords):
            return 'region'
        elif any(kw in response_lower for kw in country_keywords):
            return 'country'
        elif any(kw in response_lower for kw in vague_keywords):
            return 'vague'
        else:
            return 'unknown'


class NameDisambiguationEvaluator(TaskEvaluator):
    """Evaluator for Task 5: Name Disambiguation & Context"""

    def __init__(self):
        super().__init__("name_disambiguation")
        self.hierarchical_matcher = HierarchicalLocationMatcher()

    def create_prompt(self, test_case: Dict[str, Any]) -> Tuple[str, str]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return format_mc_question(test_case)

        # Free-form format
        subtask = test_case.get('subtask', 'city_name_disambiguation')

        if subtask == 'city_name_disambiguation':
            system_prompt = """You are an expert in geography and location disambiguation.
When multiple places share the same name, use coordinates and context to identify the correct one.

Consider:
- Precise coordinates to determine exact location
- Country and state/province information
- Population size and geographic context"""

            coord_str = test_case['coordinate_string']
            duplicate_name = test_case['duplicate_name']
            num_candidates = test_case['num_candidates']

            # Provide candidate information
            candidates_info = "\n".join([
                f"  - {c['name']}, {c['state']}, {c['country']} (pop: {c['population']:,})"
                for c in test_case['candidates'][:5]  # Show top 5 by population
            ])

            user_prompt = f"""There are {num_candidates} cities named "{duplicate_name}" in the world.

Top candidates (by population):
{candidates_info}

Which "{duplicate_name}" is at coordinates {coord_str}?

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: [City Name], [State/Province], [Country]

Example: FINAL ANSWER: Springfield, Illinois, US

Provide the complete location in the FINAL ANSWER line."""

        elif subtask == 'descriptive_context_query':
            system_prompt = """You are an expert in geography and spatial context.
Answer questions about location characteristics based on coordinates.

Consider:
- Urban vs rural characteristics (population density)
- Coastal vs inland geography
- Climate zones based on latitude
- Metropolitan status and city size"""

            coord_str = test_case['coordinate_string']
            question = test_case['question']
            context_type = test_case.get('context_type', 'unknown')

            user_prompt = f"""{question}

IMPORTANT: You may explain your reasoning, but you MUST end your response with:

FINAL ANSWER: YES
(or FINAL ANSWER: NO for yes/no questions)
(or FINAL ANSWER: [climate zone] for climate questions)

Provide only the answer in the FINAL ANSWER line."""

        else:
            # Fallback for unknown subtask
            system_prompt = "You are an expert in geography."
            user_prompt = test_case.get('question', 'Unknown question type')

        return system_prompt, user_prompt

    def evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        # Check if this is a multiple choice test
        if is_mc_test(test_case):
            return evaluate_mc_response(response, test_case)

        # Free-form evaluation
        subtask = test_case.get('subtask', 'city_name_disambiguation')
        ground_truth = test_case['ground_truth']

        # STAGE 1: Check for FINAL ANSWER line (PRIORITY)
        final_answer = extract_final_answer(response)
        if final_answer:
            response_to_check = final_answer
        else:
            # STAGE 2: Fallback to full response
            response_to_check = response

        if subtask == 'city_name_disambiguation':
            return self._evaluate_disambiguation(response_to_check, response, test_case, ground_truth)
        elif subtask == 'descriptive_context_query':
            return self._evaluate_context_query(response_to_check, response, test_case, ground_truth)
        else:
            return {
                'correct': False,
                'error': f'Unknown subtask: {subtask}',
                'raw_response': response
            }

    def _evaluate_disambiguation(self, response_to_check: str, full_response: str,
                                 test_case: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate city name disambiguation"""
        expected_name = ground_truth['name']
        expected_country = ground_truth['country_code']
        expected_state = ground_truth.get('state', 'N/A')

        response_lower = response_to_check.lower().strip()

        # Check if response contains the correct city name
        city_name_match = expected_name.lower() in response_lower

        # Check if response contains the correct country
        country_match = expected_country.lower() in response_lower

        # Check if response contains the correct state/province (if available)
        state_match = False
        if expected_state and expected_state != 'N/A':
            state_match = expected_state.lower() in response_lower

        # For disambiguation, we need BOTH location name AND country/state to be correct
        # (since multiple cities share the same name)
        correct = city_name_match and (country_match or state_match)

        # Use hierarchical matcher as fallback
        if not correct:
            hierarchy = {
                'name': expected_name,
                'country': expected_country,
                'country_code': expected_country,
                'region': expected_state
            }
            match_result = self.hierarchical_matcher.match_with_hierarchy(
                response_to_check, hierarchy
            )

            # For disambiguation, require at least city + country match
            if match_result['city_match'] and match_result['country_match']:
                correct = True

        return {
            'correct': correct,
            'expected_name': expected_name,
            'expected_country': expected_country,
            'expected_state': expected_state,
            'predicted': response_to_check,
            'city_name_match': city_name_match,
            'country_match': country_match,
            'state_match': state_match,
            'num_candidates': test_case['num_candidates'],
            'subtask': 'city_name_disambiguation',
            'raw_response': full_response
        }

    def _evaluate_context_query(self, response_to_check: str, full_response: str,
                                test_case: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate descriptive context queries (YES/NO or climate zone)"""
        context_type = test_case.get('context_type', 'unknown')
        expected_answer = ground_truth['answer']
        response_lower = response_to_check.lower().strip()

        # Determine expected answer type
        if expected_answer in ['YES', 'NO']:
            # YES/NO question
            # Check 'NO' FIRST (since 'NO' can appear in 'NOT')
            if 'no' in response_lower and 'not' in response_lower:
                predicted_answer = 'NO'
            elif 'no' in response_lower:
                predicted_answer = 'NO'
            elif 'yes' in response_lower:
                predicted_answer = 'YES'
            else:
                # Try to infer from context
                positive_keywords = ['is', 'has', 'does', 'can', 'should', 'would', 'true', 'correct', 'valid']
                negative_keywords = ['not', 'isn\'t', 'hasn\'t', 'doesn\'t', 'cannot', 'shouldn\'t', 'wouldn\'t',
                                    'false', 'incorrect', 'invalid', 'unlikely', 'impossible']

                has_negative = any(kw in response_lower for kw in negative_keywords)
                has_positive = any(kw in response_lower for kw in positive_keywords) and not has_negative

                if has_negative:
                    predicted_answer = 'NO'
                elif has_positive:
                    predicted_answer = 'YES'
                else:
                    predicted_answer = 'UNKNOWN'

            correct = predicted_answer == expected_answer

            return {
                'correct': correct,
                'expected_answer': expected_answer,
                'predicted_answer': predicted_answer,
                'context_type': context_type,
                'subtask': 'descriptive_context_query',
                'raw_response': full_response
            }

        else:
            # Climate zone or other categorical answer
            expected_lower = expected_answer.lower().strip()

            # Check for exact or partial match
            correct = expected_lower in response_lower

            # Climate zone variations
            climate_variations = {
                'tropical': ['tropical', 'equatorial'],
                'subtropical': ['subtropical', 'sub-tropical'],
                'temperate': ['temperate', 'moderate'],
                'polar': ['polar', 'arctic', 'antarctic'],
                'subpolar': ['subpolar', 'sub-polar']
            }

            if not correct and expected_lower in climate_variations:
                for variant in climate_variations[expected_lower]:
                    if variant in response_lower:
                        correct = True
                        break

            return {
                'correct': correct,
                'expected_answer': expected_answer,
                'predicted': response_to_check,
                'context_type': context_type,
                'subtask': 'descriptive_context_query',
                'raw_response': full_response
            }


def get_evaluator(task_name: str) -> TaskEvaluator:
    """
    Factory function to get appropriate evaluator for a task

    Args:
        task_name: Name of the task

    Returns:
        TaskEvaluator instance
    """
    # Map task file names to evaluator classes
    evaluators = {
        'task1_place_association': PlaceAssociationEvaluator(),
        'place_association': PlaceAssociationEvaluator(),

        'task2_semantic_validation': SemanticValidationEvaluator(),
        'semantic_validation': SemanticValidationEvaluator(),

        'task3_regional_knowledge': RegionalKnowledgeEvaluator(),
        'regional_knowledge': RegionalKnowledgeEvaluator(),

        'task4_context_grounding': ContextGroundingEvaluator(),
        'context_grounding': ContextGroundingEvaluator(),

        'task5_name_disambiguation': NameDisambiguationEvaluator(),
        'name_disambiguation': NameDisambiguationEvaluator(),
    }

    if task_name not in evaluators:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(evaluators.keys())}")

    return evaluators[task_name]


# Test evaluators
if __name__ == "__main__":
    print("="*60)
    print("Testing Layer B Evaluators")
    print("="*60)

    # Test Task 1: Place Association
    print("\nTask 1: Place Association")
    print("-"*60)
    evaluator1 = PlaceAssociationEvaluator()

    test_case = {
        'coordinate_string': '48.8584, 2.2945',
        'subtask': 'landmark_identification',
        'ground_truth': {
            'name': 'Eiffel Tower, Paris',
            'country': 'FR',
            'alternate_names': []
        }
    }

    responses = [
        "Eiffel Tower",
        "The Eiffel Tower in Paris",
        "Paris, France",
        "London"
    ]

    for resp in responses:
        result = evaluator1.evaluate_response(resp, test_case)
        print(f"Response: '{resp}'")
        print(f"  Correct: {result['correct']}, Match: {result['match_type']}")

    # Test Task 2: Semantic Validation
    print("\n\nTask 2: Semantic Validation")
    print("-"*60)
    evaluator2 = SemanticValidationEvaluator()

    test_case = {
        'coordinate_string': '0.0, -160.0',
        'context': 'restaurant',
        'ground_truth': {
            'is_plausible': False,
            'validation_type': 'land_water'
        }
    }

    responses = [
        "IMPLAUSIBLE - This is in the middle of the ocean",
        "Not plausible, there cannot be a restaurant in the ocean",
        "PLAUSIBLE",
        "This location is impossible for a restaurant"
    ]

    for resp in responses:
        result = evaluator2.evaluate_response(resp, test_case)
        print(f"Response: '{resp}'")
        print(f"  Correct: {result['correct']}, Predicted: {result['predicted_plausible']}")

    # Test Task 3: Regional Knowledge
    print("\n\nTask 3: Regional Knowledge")
    print("-"*60)
    evaluator3 = RegionalKnowledgeEvaluator()

    test_case = {
        'coordinate_string': '35.6762, 139.6503',
        'question': 'Is this coordinate in Japan?',
        'subtask': 'country_membership',
        'ground_truth': {
            'answer': True,
            'country': 'Japan'
        }
    }

    responses = [
        "Yes, these coordinates are in Japan",
        "No",
        "Yes - Tokyo, Japan",
        "These coordinates are in Japan"
    ]

    for resp in responses:
        result = evaluator3.evaluate_response(resp, test_case)
        print(f"Response: '{resp}'")
        print(f"  Correct: {result['correct']}")

    # Test Task 4: Context Grounding
    print("\n\nTask 4: Context-Appropriate Grounding")
    print("-"*60)
    evaluator4 = ContextGroundingEvaluator()

    test_case = {
        'coordinate_string': '40.7128, -74.0060',
        'decimal_places': 4,
        'ground_truth': {
            'precision_level': 'street',
            'appropriate_levels': ['street', 'building', 'landmark'],
            'too_specific': [],
            'too_vague': ['country', 'continent'],
            'example_appropriate': 'Manhattan, New York City'
        }
    }

    responses = [
        "This is a specific street in Manhattan, New York City",
        "This is in the United States",
        "The exact building at this address in NYC",
        "Manhattan area"
    ]

    for resp in responses:
        result = evaluator4.evaluate_response(resp, test_case)
        print(f"Response: '{resp}'")
        print(f"  Correct: {result['correct']}, Level: {result['predicted_level']}")
        print(f"  Over-specified: {result['over_specified']}, Under-specified: {result['under_specified']}")

    print("\n" + "="*60)
    print("Layer B Evaluator testing complete!")
    print("="*60)
