"""
Task-to-System-Prompt Mapping for GPSBench Fine-tuning

This maps training task types to their corresponding system prompts
used during evaluation, ensuring consistency between training and inference.

UPDATED: 2025-12-26 to load prompts from prompts/ directory for consistency with run_benchmark.py
"""

from pathlib import Path


def _load_system_prompt() -> str:
    """Load system prompt from prompts/system_prompt.txt for consistency with run_benchmark.py"""
    # Try multiple paths to find the prompts directory
    possible_paths = [
        Path(__file__).parent.parent / "prompts" / "system_prompt.txt",  # From finetuning/
        Path(__file__).parent / "prompts" / "system_prompt.txt",  # Direct
        Path("prompts") / "system_prompt.txt",  # From repo root
    ]

    for path in possible_paths:
        if path.exists():
            return path.read_text().strip()

    # Fallback if file not found (should not happen in normal usage)
    print("WARNING: Could not find prompts/system_prompt.txt, using fallback prompt")
    return """You are an expert in GPS coordinates, geographic information systems, and spatial reasoning.

INSTRUCTIONS:
1. Be CONCISE - use brief step-by-step reasoning (2-4 steps max)
2. Skip lengthy explanations - show only key calculations
3. ALWAYS end with: FINAL ANSWER: [your answer]

FORMAT RULES:
- Multiple choice: FINAL ANSWER: A (just the letter)
- Numeric: FINAL ANSWER: 100.5 km (number with units)
- Location: FINAL ANSWER: City, Region, Country

CRITICAL: Keep your response under 500 words. Always include FINAL ANSWER at the end."""


# Load system prompt from file to ensure consistency with run_benchmark.py
STANDARD_SYSTEM_PROMPT = _load_system_prompt()

# System prompts for different task categories
# NOTE: For finetuning, we now use the STANDARD_SYSTEM_PROMPT for consistency
# Task-specific hints are included in the user prompt/instruction instead
TASK_SYSTEM_PROMPTS = {
    # ===== ALL TASKS USE STANDARD PROMPT =====
    # This ensures consistency between training and evaluation
    "format_conversion": STANDARD_SYSTEM_PROMPT,
    "precision_understanding": STANDARD_SYSTEM_PROMPT,

    "validation": STANDARD_SYSTEM_PROMPT,
    "crs_identification": STANDARD_SYSTEM_PROMPT,

    # ===== TIER 2: GEOMETRIC OPERATIONS =====
    "distance_calculation": STANDARD_SYSTEM_PROMPT,
    "bearing_computation": STANDARD_SYSTEM_PROMPT,
    "interpolation": STANDARD_SYSTEM_PROMPT,
    "area_perimeter": STANDARD_SYSTEM_PROMPT,
    "route_geometry": STANDARD_SYSTEM_PROMPT,
    "turn_angle_calculation": STANDARD_SYSTEM_PROMPT,
    "route_length_calculation": STANDARD_SYSTEM_PROMPT,
    "route_straightness": STANDARD_SYSTEM_PROMPT,
    "polyline_simplification": STANDARD_SYSTEM_PROMPT,
    "bounding_box": STANDARD_SYSTEM_PROMPT,

    # ===== TIER 3: ROBUSTNESS (PURE GPS) =====
    "noise_tolerance": STANDARD_SYSTEM_PROMPT,
    "coordinate_matching_with_tolerance": STANDARD_SYSTEM_PROMPT,
    "coordinate_matching": STANDARD_SYSTEM_PROMPT,
    "outlier_detection": STANDARD_SYSTEM_PROMPT,
    "missing_data": STANDARD_SYSTEM_PROMPT,
    "interpolation_from_sparse_data": STANDARD_SYSTEM_PROMPT,

    # ===== APPLIED TRACK: LOCATION GROUNDING =====
    "place_association": STANDARD_SYSTEM_PROMPT,
    "regional_knowledge": STANDARD_SYSTEM_PROMPT,
    "context_grounding": STANDARD_SYSTEM_PROMPT,
    "semantic_validation": STANDARD_SYSTEM_PROMPT,
    "city_name_disambiguation": STANDARD_SYSTEM_PROMPT,
    "terrain_classification": STANDARD_SYSTEM_PROMPT,

    # ===== APPLIED TRACK: SPATIAL RELATIONS =====
    "relative_position": STANDARD_SYSTEM_PROMPT,
    "proximity": STANDARD_SYSTEM_PROMPT,
    "route_analysis": STANDARD_SYSTEM_PROMPT,
    "boundary_analysis": STANDARD_SYSTEM_PROMPT,
    "spatial_patterns": STANDARD_SYSTEM_PROMPT,

    # ===== APPLIED TRACK: ROBUSTNESS =====
    "descriptive_context_query": STANDARD_SYSTEM_PROMPT,
    "order_independence": STANDARD_SYSTEM_PROMPT,
    "paraphrase_robustness": STANDARD_SYSTEM_PROMPT,
    "consistency": STANDARD_SYSTEM_PROMPT,
    "plausibility_checking": STANDARD_SYSTEM_PROMPT,
    "cross_location_attribution": STANDARD_SYSTEM_PROMPT,
    "hallucination_detection": STANDARD_SYSTEM_PROMPT,
    "coordinate_swapping_misattribution": STANDARD_SYSTEM_PROMPT,
    "distance_unit_detection": STANDARD_SYSTEM_PROMPT,
    "precision_unit_mismatch": STANDARD_SYSTEM_PROMPT,
    "unit_confusion": STANDARD_SYSTEM_PROMPT,
    "wgs84_to_web_mercator": STANDARD_SYSTEM_PROMPT,

    # ===== DEFAULT =====
    "general": STANDARD_SYSTEM_PROMPT
}


def get_system_prompt_for_task(task_name: str) -> str:
    """
    Get the appropriate system prompt for a given task.

    Args:
        task_name: Name of the task from metadata

    Returns:
        System prompt string
    """
    # Direct match
    if task_name in TASK_SYSTEM_PROMPTS:
        return TASK_SYSTEM_PROMPTS[task_name]

    # Fallback to general prompt
    return TASK_SYSTEM_PROMPTS["general"]


def get_task_category(task_name: str) -> str:
    """
    Get the category of a task (for grouping).

    Returns:
        Category name: 'syntax_formats', 'geometric_ops', 'pure_robustness',
                      'location_grounding', 'spatial_relations', 'applied_robustness'
    """
    tier1_tasks = {
        'format_conversion', 'precision_understanding', 'validation',
        'crs_identification'
    }

    tier2_tasks = {
        'distance_calculation', 'bearing_computation', 'interpolation',
        'area_perimeter', 'route_geometry', 'turn_angle_calculation',
        'route_length_calculation', 'route_straightness', 'polyline_simplification',
        'bounding_box'
    }

    tier3_pure = {
        'noise_tolerance', 'coordinate_matching_with_tolerance', 'coordinate_matching',
        'outlier_detection', 'missing_data', 'interpolation_from_sparse_data'
    }

    applied_grounding = {
        'place_association', 'regional_knowledge', 'context_grounding',
        'semantic_validation', 'city_name_disambiguation'
    }

    applied_spatial = {
        'relative_position', 'proximity', 'route_analysis',
        'boundary_analysis', 'spatial_patterns'
    }

    applied_robustness = {
        'descriptive_context_query', 'order_independence', 'paraphrase_robustness',
        'consistency', 'plausibility_checking', 'cross_location_attribution',
        'hallucination_detection', 'coordinate_swapping_misattribution',
        'distance_unit_detection', 'precision_unit_mismatch', 'unit_confusion',
        'wgs84_to_web_mercator'
    }

    if task_name in tier1_tasks:
        return 'syntax_formats'
    elif task_name in tier2_tasks:
        return 'geometric_ops'
    elif task_name in tier3_pure:
        return 'pure_robustness'
    elif task_name in applied_grounding:
        return 'location_grounding'
    elif task_name in applied_spatial:
        return 'spatial_relations'
    elif task_name in applied_robustness:
        return 'applied_robustness'
    else:
        return 'general'


if __name__ == "__main__":
    # Test the mapping
    print("Task-to-Prompt Mapping Test")
    print("="*80)

    test_tasks = [
        'format_conversion',
        'distance_calculation',
        'noise_tolerance',
        'place_association',
        'unknown_task'
    ]

    for task in test_tasks:
        prompt = get_system_prompt_for_task(task)
        category = get_task_category(task)
        print(f"\nTask: {task}")
        print(f"Category: {category}")
        print(f"Prompt: {prompt[:80]}...")
