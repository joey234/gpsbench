"""
GPSBench Evaluation Framework
"""

from .llm_client import LLMClient, LLMResponse, get_available_models
from .task_evaluators import (
    TaskEvaluator,
    FormatRecognitionEvaluator,
    FormatConversionEvaluator,
    ValidationEvaluator,
    CoordinateSystemEvaluator,
    PrecisionEvaluator,
    get_evaluator
)

__all__ = [
    'LLMClient',
    'LLMResponse',
    'get_available_models',
    'TaskEvaluator',
    'FormatRecognitionEvaluator',
    'FormatConversionEvaluator',
    'ValidationEvaluator',
    'CoordinateSystemEvaluator',
    'PrecisionEvaluator',
    'get_evaluator',
]

__version__ = '1.0.0'
