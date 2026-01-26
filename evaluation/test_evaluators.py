#!/usr/bin/env python3
"""
Test script to verify all task evaluators work correctly with various response formats.
Tests the fixes made to parsing logic and regex patterns.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.task_evaluators import (
    FormatRecognitionEvaluator,
    FormatConversionEvaluator,
    ValidationEvaluator,
    CoordinateSystemEvaluator,
    PrecisionEvaluator
)


def test_format_recognition():
    """Test Task 1: Format Recognition Evaluator"""
    print("=" * 60)
    print("Testing Task 1: Format Recognition Evaluator")
    print("=" * 60)

    evaluator = FormatRecognitionEvaluator()

    # Test case
    test_case = {
        'coordinate_string': '40°51\'30"N, 73°59\'45"W',
        'ground_truth': {
            'format_type': 'DMS',
            'latitude': 40.858,
            'longitude': -73.996
        }
    }

    # Test various response formats
    test_responses = [
        ("This is in DMS format (Degrees Minutes Seconds).", True),
        ("The format is DMS", True),
        ("dms", True),
        ("decimal_degrees", False),
        ("This is a decimal degrees format", False),
    ]

    for response, expected_correct in test_responses:
        result = evaluator.evaluate_response(response, test_case)
        status = "✓" if result['correct'] == expected_correct else "✗"
        print(f"{status} Response: '{response}'")
        print(f"   Expected: {expected_correct}, Got: {result['correct']}")
        print(f"   Predicted: {result['predicted']}")

    print()


def test_format_conversion():
    """Test Task 2: Format Conversion Evaluator (with new tolerance)"""
    print("=" * 60)
    print("Testing Task 2: Format Conversion Evaluator (tolerance=0.01)")
    print("=" * 60)

    evaluator = FormatConversionEvaluator()

    # Test case with DMS conversion
    test_case = {
        'source_format': 'decimal_degrees',
        'target_format': 'DMS',
        'source_string': '40.858333, -73.996111',
        'ground_truth': {
            'target_string': '40°51\'30"N, 73°59\'46"W'
        }
    }

    # Test responses with slight rounding differences
    test_responses = [
        # Exact match
        ('40°51\'30"N, 73°59\'46"W', True),
        # Within 0.01 tolerance (0.005 difference)
        ('40°51\'30"N, 73°59\'46.5"W', True),
        # Within 0.01 tolerance (seconds vary slightly)
        ('40°51\'29.5"N, 73°59\'46"W', True),
        # Outside tolerance
        ('40°52\'30"N, 73°59\'46"W', False),
    ]

    for response, expected_correct in test_responses:
        result = evaluator.evaluate_response(response, test_case)
        status = "✓" if result['correct'] == expected_correct else "✗"
        print(f"{status} Response: '{response}'")
        print(f"   Expected: {expected_correct}, Got: {result['correct']}")

    print()


def test_validation():
    """Test Task 3: Validation Evaluator (CRITICAL FIX TEST)"""
    print("=" * 60)
    print("Testing Task 3: Validation Evaluator (CRITICAL FIX)")
    print("=" * 60)

    evaluator = ValidationEvaluator()

    # Test case for valid coordinate
    valid_test = {
        'coordinate_string': '40.858, -73.996',
        'ground_truth': {
            'is_valid': True
        }
    }

    # Test case for invalid coordinate
    invalid_test = {
        'coordinate_string': '95.000, -73.996',
        'ground_truth': {
            'is_valid': False
        },
        'error_type': 'out_of_range'
    }

    print("\nValid Coordinate Tests:")
    valid_responses = [
        ("VALID", True),
        ("This coordinate is valid", True),
        ("The coordinates are valid and within range", True),
        ("INVALID - latitude exceeds 90 degrees", False),  # Should be wrong
        ("invalid", False),  # Should be wrong
    ]

    for response, expected_correct in valid_responses:
        result = evaluator.evaluate_response(response, valid_test)
        status = "✓" if result['correct'] == expected_correct else "✗"
        print(f"{status} Response: '{response}'")
        print(f"   Expected valid={valid_test['ground_truth']['is_valid']}, Got valid={result['predicted_valid']}")
        print(f"   Correct? Expected: {expected_correct}, Got: {result['correct']}")

    print("\nInvalid Coordinate Tests (CRITICAL - 'invalid' contains 'valid'):")
    invalid_responses = [
        ("INVALID - latitude exceeds 90 degrees", True),
        ("This coordinate is invalid", True),
        ("invalid", True),
        ("The latitude is out of range", True),
        ("VALID", False),  # Should be wrong
        ("This coordinate is valid", False),  # Should be wrong
    ]

    for response, expected_correct in invalid_responses:
        result = evaluator.evaluate_response(response, invalid_test)
        status = "✓" if result['correct'] == expected_correct else "✗"
        print(f"{status} Response: '{response}'")
        print(f"   Expected valid={invalid_test['ground_truth']['is_valid']}, Got valid={result['predicted_valid']}")
        print(f"   Correct? Expected: {expected_correct}, Got: {result['correct']}")

    print()


def test_coordinate_system():
    """Test Task 4: Coordinate System Evaluator (WGS84_DMS support)"""
    print("=" * 60)
    print("Testing Task 4: Coordinate System Evaluator (WGS84_DMS)")
    print("=" * 60)

    evaluator = CoordinateSystemEvaluator()

    # Test WGS84 vs WGS84_DMS differentiation
    wgs84_test = {
        'coordinate_string': '40.858, -73.996',
        'ground_truth': {
            'system_name': 'WGS84'
        }
    }

    wgs84_dms_test = {
        'coordinate_string': '40°51\'30"N, 73°59\'46"W',
        'ground_truth': {
            'system_name': 'WGS84_DMS'
        }
    }

    print("\nWGS84 (decimal) Tests:")
    wgs84_responses = [
        ("WGS84", True),
        ("This is WGS84 in decimal degrees", True),
        ("WGS84_DMS", False),  # Should be wrong - checking WGS84_DMS before WGS84
    ]

    for response, expected_correct in wgs84_responses:
        result = evaluator.evaluate_response(response, wgs84_test)
        status = "✓" if result['correct'] == expected_correct else "✗"
        print(f"{status} Response: '{response}'")
        print(f"   Expected: {wgs84_test['ground_truth']['system_name']}, Predicted: {result['predicted']}")
        print(f"   Correct? Expected: {expected_correct}, Got: {result['correct']}")

    print("\nWGS84_DMS Tests:")
    wgs84_dms_responses = [
        ("WGS84_DMS", True),
        ("This is WGS84_DMS format", True),
        ("WGS84 in DMS format (WGS84_DMS)", True),
        ("WGS84", False),  # Should be wrong - need DMS suffix
    ]

    for response, expected_correct in wgs84_dms_responses:
        result = evaluator.evaluate_response(response, wgs84_dms_test)
        status = "✓" if result['correct'] == expected_correct else "✗"
        print(f"{status} Response: '{response}'")
        print(f"   Expected: {wgs84_dms_test['ground_truth']['system_name']}, Predicted: {result['predicted']}")
        print(f"   Correct? Expected: {expected_correct}, Got: {result['correct']}")

    print()


def test_precision():
    """Test Task 5: Precision Evaluator (improved regex)"""
    print("=" * 60)
    print("Testing Task 5: Precision Evaluator (improved regex)")
    print("=" * 60)

    evaluator = PrecisionEvaluator()

    # Test case
    test_case = {
        'coordinate_string': '40.86, -73.99',
        'ground_truth': {
            'decimal_places': 2,
            'accuracy_meters': 1100,
            'appropriate_use_case': 'neighborhood'
        }
    }

    print("\nTesting various response formats:")
    test_responses = [
        # Standard format
        "2 decimal places, approximately 1100 m accuracy, suitable for neighborhood-level mapping",
        # With commas in numbers
        "2 decimals, accuracy: 1,100 m, use case: neighborhood navigation",
        # Using km instead of m
        "2 decimal places, about 1.1 km precision, good for neighborhood",
        # Different wording
        "This has 2 decimal places. The accuracy is roughly 1100 meters. Appropriate for neighborhood-scale applications.",
    ]

    for response in test_responses:
        result = evaluator.evaluate_response(response, test_case)
        status = "✓" if result['correct'] else "✗"
        print(f"\n{status} Response: '{response}'")
        print(f"   Decimals: expected={test_case['ground_truth']['decimal_places']}, "
              f"predicted={result['predicted_decimals']}, correct={result['decimals_correct']}")
        print(f"   Accuracy: expected={test_case['ground_truth']['accuracy_meters']}m, "
              f"predicted={result['predicted_accuracy']}m, correct={result['accuracy_correct']}")
        print(f"   Use case: expected='{test_case['ground_truth']['appropriate_use_case']}', "
              f"correct={result['use_case_correct']}")
        print(f"   Overall: {result['correct']} (2/3 must be correct)")

    print()


def main():
    """Run all evaluator tests"""
    print("\n" + "=" * 60)
    print("GPSBench Layer A Evaluator Testing Suite")
    print("Testing fixes to parsing logic and regex patterns")
    print("=" * 60 + "\n")

    try:
        test_format_recognition()
        test_format_conversion()
        test_validation()
        test_coordinate_system()
        test_precision()

        print("=" * 60)
        print("All evaluator tests completed!")
        print("=" * 60)
        print("\nReview the results above:")
        print("✓ = Test passed as expected")
        print("✗ = Test failed (unexpected result)")
        print("\nKey fixes tested:")
        print("1. Task 2: Increased tolerance from 0.001 to 0.01")
        print("2. Task 3: Fixed 'invalid' vs 'valid' substring bug (CRITICAL)")
        print("3. Task 4: Added WGS84_DMS support and proper ordering")
        print("4. Task 5: Improved regex to handle commas and km units")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
