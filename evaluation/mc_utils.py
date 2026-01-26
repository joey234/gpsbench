"""
Multiple Choice Evaluation Utilities

Shared utilities for evaluating multiple choice test responses across all GPSBench layers.
"""

import re
from typing import Dict, Optional, Tuple


def extract_option_letter(response: str, valid_options: list = None) -> Optional[str]:
    """
    Extract the selected option letter (A, B, C, D, etc.) from model response.

    Handles various response formats:
    - "A" or "A)" or "(A)"
    - "The answer is A"
    - "FINAL ANSWER: A"
    - "A) Option text"
    - Full option text matching

    Args:
        response: Model's response string
        valid_options: List of valid option letters (e.g., ['A', 'B', 'C', 'D'])
                      If None, accepts any uppercase letter

    Returns:
        Single uppercase letter (A, B, C, D...) or None if cannot extract
    """
    if not response:
        return None

    # Default to A-H if not specified
    if valid_options is None:
        valid_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # Convert to uppercase for matching
    response_upper = response.upper()

    # Strategy 1: Look for "FINAL ANSWER: X" pattern (most reliable)
    final_answer_match = re.search(r'FINAL\s+ANSWER\s*:\s*([A-H])', response_upper)
    if final_answer_match:
        letter = final_answer_match.group(1)
        if letter in valid_options:
            return letter

    # Strategy 2: Look for standalone option letter with common delimiters
    # Patterns: "A)", "(A)", "A.", "A:", "Option A", etc.
    for option in valid_options:
        patterns = [
            rf'\b{option}\)',  # A)
            rf'\({option}\)',  # (A)
            rf'\b{option}\.',  # A.
            rf'\b{option}:',   # A:
            rf'\b{option}\b',  # Just A (word boundary)
            rf'OPTION\s+{option}',  # Option A
            rf'ANSWER\s+{option}',  # Answer A
            rf'CHOICE\s+{option}',  # Choice A
        ]

        for pattern in patterns:
            if re.search(pattern, response_upper):
                return option

    # Strategy 3: Look for first occurrence of any valid option letter
    # This is a fallback for simple responses like "A" or "The answer is B"
    for char in response_upper:
        if char in valid_options:
            return char

    return None


def evaluate_mc_response(response: str, test: dict) -> dict:
    """
    Evaluate a multiple choice response.

    Args:
        response: Model's response string
        test: Test case dictionary with 'format': 'multiple_choice'

    Returns:
        {
            'correct': bool,
            'predicted_option': str or None,
            'correct_option': str,
            'expected': str,  # The correct option letter
            'predicted': str,  # The predicted option letter or 'NONE'
            'evaluation_error': bool,  # True if couldn't extract option
        }
    """
    # Verify this is a MC test
    if test.get('format') != 'multiple_choice':
        raise ValueError(f"Test is not multiple choice format: {test.get('format')}")

    # Get valid options from test
    options = test.get('options', {})
    valid_options = list(options.keys())

    # Get correct option
    correct_option = test.get('ground_truth', {}).get('correct_option')

    if not correct_option:
        raise ValueError("Test missing 'correct_option' in ground_truth")

    # Extract predicted option from response
    predicted_option = extract_option_letter(response, valid_options)

    # Check if extraction failed
    evaluation_error = predicted_option is None

    # Evaluate correctness
    correct = predicted_option == correct_option if predicted_option else False

    return {
        'correct': correct,
        'predicted_option': predicted_option,
        'correct_option': correct_option,
        'expected': correct_option,
        'predicted': predicted_option if predicted_option else 'NONE',
        'evaluation_error': evaluation_error,
    }


def format_mc_question(test: dict) -> Tuple[str, str]:
    """
    Format a multiple choice question for model prompting.

    Args:
        test: Test case dictionary with 'format': 'multiple_choice'

    Returns:
        Tuple of (system_prompt, user_prompt) with formatted MC question
    """
    if test.get('format') != 'multiple_choice':
        raise ValueError(f"Test is not multiple choice format: {test.get('format')}")

    question = test.get('question', '')
    options = test.get('options', {})

    # Build options string
    options_str = ""
    for letter in sorted(options.keys()):
        options_str += f"{letter}) {options[letter]}\n"

    # System prompt for MC format
    system_prompt = """You are an expert in GPS coordinates and geospatial analysis.
For multiple choice questions, select the best answer from the given options.

IMPORTANT: You MUST end your response with:

FINAL ANSWER: [letter]

where [letter] is A, B, C, D, etc.

Example:
FINAL ANSWER: B"""

    # User prompt with question and options
    user_prompt = f"""{question}

{options_str.strip()}

Provide your reasoning if helpful, but you MUST end with:

FINAL ANSWER: [letter]"""

    return system_prompt, user_prompt


def is_mc_test(test: dict) -> bool:
    """
    Check if a test case is in multiple choice format.

    Args:
        test: Test case dictionary

    Returns:
        True if test has 'format': 'multiple_choice'
    """
    return test.get('format') == 'multiple_choice'


def get_mc_statistics(results: list) -> dict:
    """
    Calculate statistics for multiple choice evaluation results.

    Args:
        results: List of evaluation result dictionaries

    Returns:
        Dictionary with MC-specific statistics:
        {
            'total_mc_tests': int,
            'extraction_success_rate': float,
            'accuracy': float,
            'option_distribution': dict,  # How often each option was chosen
            'confusion_matrix': dict,  # Predicted vs correct options
        }
    """
    if not results:
        return {}

    total = len(results)
    extraction_failures = sum(1 for r in results if r.get('evaluation_error', False))
    correct = sum(1 for r in results if r.get('correct', False))

    # Count option distribution (which options models chose)
    option_distribution = {}
    for result in results:
        pred = result.get('predicted_option')
        if pred:
            option_distribution[pred] = option_distribution.get(pred, 0) + 1

    # Build confusion matrix (predicted vs correct)
    confusion_matrix = {}
    for result in results:
        pred = result.get('predicted_option', 'NONE')
        correct_opt = result.get('correct_option', 'UNKNOWN')
        key = f"{pred}_vs_{correct_opt}"
        confusion_matrix[key] = confusion_matrix.get(key, 0) + 1

    return {
        'total_mc_tests': total,
        'extraction_success_rate': (total - extraction_failures) / total if total > 0 else 0,
        'extraction_failures': extraction_failures,
        'accuracy': correct / total if total > 0 else 0,
        'correct_count': correct,
        'option_distribution': option_distribution,
        'confusion_matrix': confusion_matrix,
    }


def print_mc_statistics(stats: dict):
    """
    Print formatted MC evaluation statistics.

    Args:
        stats: Statistics dictionary from get_mc_statistics()
    """
    print("\n" + "="*80)
    print("MULTIPLE CHOICE EVALUATION STATISTICS")
    print("="*80)

    total = stats.get('total_mc_tests', 0)
    if total == 0:
        print("No multiple choice tests evaluated.")
        return

    print(f"\nTotal MC Tests: {total}")
    print(f"Extraction Success Rate: {stats.get('extraction_success_rate', 0)*100:.1f}%")
    print(f"Extraction Failures: {stats.get('extraction_failures', 0)}")
    print(f"Accuracy: {stats.get('accuracy', 0)*100:.1f}%")
    print(f"Correct: {stats.get('correct_count', 0)}/{total}")

    # Print option distribution
    option_dist = stats.get('option_distribution', {})
    if option_dist:
        print("\nOption Distribution (Model Choices):")
        for option in sorted(option_dist.keys()):
            count = option_dist[option]
            pct = (count / total) * 100
            print(f"  {option}: {count:4d} ({pct:5.1f}%)")

    print("="*80)
