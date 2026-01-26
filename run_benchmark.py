#!/usr/bin/env python3
"""
GPSBench Evaluation Runner
==========================

Run the complete GPSBench benchmark on any LLM via OpenAI API, OpenRouter, or Google Gemini.

IMPORTANT: This script ONLY evaluates on TEST SPLITS to prevent data leakage.
           Test splits are located in data/track_*/splits/*_test.json

Usage:
    # Normal mode - OpenAI GPT-4 (real-time)
    python run_benchmark.py --provider openai --model gpt-4

    # Google Gemini (auto-detected from model name)
    python run_benchmark.py --model gemini-2.5-flash
    python run_benchmark.py --model gemini-2.5-pro --max-samples 10

    # Batch API mode - Simple (50% cost savings, auto submit→monitor→process)
    python run_benchmark.py --model gpt-4 --use-batch

    # Batch API - With options
    python run_benchmark.py --model gpt-4 --use-batch --track pure_gps --max-samples 10

    # OpenRouter Claude
    python run_benchmark.py --provider openrouter --model anthropic/claude-3-5-sonnet

    # Quick test (10 samples per task)
    python run_benchmark.py --model gpt-3.5-turbo --max-samples 10

    # Advanced: Manual batch control (for interrupted workflows)
    # Step 1: Submit batch jobs
    python run_benchmark.py --model gpt-4 --batch-mode submit --track both

    # Step 2: Monitor batch jobs
    python run_benchmark.py --batch-mode monitor --batch-ids batch_ABC123 batch_XYZ789

    # Step 3: Process completed results
    python run_benchmark.py --batch-mode process --batch-ids batch_ABC123 batch_XYZ789

Prerequisites:
    Test splits must be generated first:
    python scripts/create_splits.py --track both --seed 42

Environment Variables:
    OPENAI_API_KEY      - Required for OpenAI provider
    OPENROUTER_API_KEY  - Required for OpenRouter provider
    GEMINI_API_KEY      - Required for Gemini provider (or GOOGLE_API_KEY)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from functools import lru_cache

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent / "evaluation"))

from llm_client import LLMClient

# Prompt template loading
@lru_cache(maxsize=32)
def load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    template_path = Path(__file__).parent / "prompts" / f"{template_name}.txt"
    if template_path.exists():
        return template_path.read_text()
    return None
from tqdm import tqdm


class GPSBenchRunner:
    """Runner for GPSBench evaluation"""

    def __init__(
        self,
        llm_client: LLMClient,
        data_dir: str = "data",
        results_dir: str = "results",
        batch_size: int = 50,
        max_workers: int = 10
    ):
        self.llm_client = llm_client
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.max_workers = max_workers

        # Track structure - UPDATED to match current task files
        self.tracks = {
            "pure_gps": {
                "name": "Pure GPS Track",
                "tasks": [
                    # All Pure GPS tasks (9 tasks total)
                    {"file": "task1_format_conversion.json", "name": "Format Conversion"},
                    {"file": "task3_coordinate_system_transformation.json", "name": "Coordinate System Transformation"},
                    {"file": "task4_distance_calculation.json", "name": "Distance Calculation"},
                    {"file": "task5_bearing_computation.json", "name": "Bearing Computation"},
                    {"file": "task6_interpolation.json", "name": "Coordinate Interpolation"},
                    {"file": "task7_area_perimeter.json", "name": "Polygon Area"},
                    {"file": "task8_bounding_box.json", "name": "Bounding Box"},
                    {"file": "task9_route_geometry.json", "name": "Route Geometry"},
                    {"file": "task10_relative_position.json", "name": "Relative Position"},
                ]
            },
            "applied": {
                "name": "Applied Track",
                "tasks": [
                    # All Applied tasks (9 tasks total)
                    # Note: task1 now includes noise tolerance/sensitivity items
                    {"file": "task1_place_association.json", "name": "Place Association"},
                    {"file": "task3_name_disambiguation.json", "name": "Name Disambiguation"},
                    {"file": "task5_relative_position.json", "name": "Relative Position"},
                    {"file": "task6_proximity.json", "name": "Proximity & Nearest Neighbor"},
                    {"file": "task7_route_analysis.json", "name": "Route Analysis"},
                    {"file": "task8_boundary_analysis.json", "name": "Boundary Analysis"},
                    {"file": "task9_spatial_patterns.json", "name": "Spatial Patterns"},
                    {"file": "task11_missing_data.json", "name": "Missing Data Inference"},
                    {"file": "task13_terrain_classification.json", "name": "Terrain Classification"},
                ]
            }
        }

        # For incremental saving
        self._current_output_dir = None

    def init_output_dir(self, output_name: Optional[str] = None) -> Path:
        """Initialize the output directory for saving results incrementally.

        Returns:
            Path to the output directory
        """
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_safe = self.llm_client.model.replace("/", "_").replace(":", "_")
            folder_name = f"gpsbench_{model_safe}_{timestamp}"
        else:
            folder_name = output_name.replace('.json', '')

        output_dir = self.results_dir / folder_name
        output_dir.mkdir(exist_ok=True, parents=True)

        # Create task_results subfolder
        task_results_dir = output_dir / "task_results"
        task_results_dir.mkdir(exist_ok=True)

        self._current_output_dir = output_dir
        print(f"\n📁 Results will be saved to: {output_dir}/")

        return output_dir

    def save_task_result_incremental(
        self,
        output_dir: Path,
        track_name: str,
        task_result: Dict[str, Any]
    ) -> Path:
        """Save a single task result immediately after evaluation.

        Args:
            output_dir: The output directory
            track_name: Name of the track (e.g., 'pure_gps', 'applied')
            task_result: The task result dictionary

        Returns:
            Path to the saved task result file
        """
        task_results_dir = output_dir / "task_results"
        task_results_dir.mkdir(exist_ok=True)

        task_file_name = task_result.get("task_file", "unknown.json").replace(".json", "")
        task_result_path = task_results_dir / f"{track_name}_{task_file_name}.json"

        with open(task_result_path, 'w') as f:
            json.dump(task_result, f, indent=2)

        print(f"   💾 Saved: {task_result_path.name}")

        return task_result_path

    def add_format_instruction(self, user_prompt: str, ground_truth: Dict[str, Any], example: Dict[str, Any] = None) -> str:
        """
        Add format instruction to prompt based on ground truth structure

        Returns:
            user_prompt with format instruction appended
        """
        # Skip if prompt already has format instruction
        if "FINAL ANSWER:" in user_prompt and "[" in user_prompt.split("FINAL ANSWER:")[-1][:50]:
            return user_prompt

        format_instruction = None

        # Detect format from ground truth
        # Check for MCQ format first (options field in example)
        if example and "options" in example:
            options = example["options"]
            options_str = "/".join(options)
            format_instruction = f"Answer with one of: {options_str}"

        # IMPORTANT: Check boolean answers FIRST, as some tasks have both 'answer' and 'distance_km'
        elif "answer" in ground_truth and isinstance(ground_truth["answer"], bool):
            format_instruction = "Answer with Yes or No"

        elif "initial_bearing_deg" in ground_truth or "final_bearing_deg" in ground_truth:
            format_instruction = "Provide your bearing in degrees (e.g., 45.5° or 45.5 degrees)"

        elif "distance_km" in ground_truth or "distance_miles" in ground_truth:
            format_instruction = "Provide distance with units (e.g., 100.5 km)"

        elif "midpoint" in ground_truth or "interpolated_point" in ground_truth:
            format_instruction = "Provide coordinates as latitude, longitude (e.g., 12.34, 56.78)"

        elif "area_km2" in ground_truth:
            format_instruction = "Provide area in km² (e.g., 1234.5 km² or 1234.5)"

        elif all(k in ground_truth for k in ["min_lat", "max_lat", "min_lon", "max_lon"]):
            format_instruction = "Provide bounding box as: min_lat, max_lat, min_lon, max_lon"

        elif ("x" in ground_truth and "y" in ground_truth) or ("easting" in ground_truth and "northing" in ground_truth):
            if "easting" in ground_truth:
                format_instruction = "Provide as Easting: [value] meters, Northing: [value] meters"
            else:
                format_instruction = "Provide as X: [value], Y: [value]"

        elif "latitude" in ground_truth and "longitude" in ground_truth:
            format_instruction = "Provide as latitude, longitude (e.g., -18.1416, 178.4419)"

        elif "answer" in ground_truth:
            answer_val = ground_truth["answer"]
            # Boolean answers already handled above
            if isinstance(answer_val, (int, float)):
                if "tolerance" in ground_truth:
                    # Missing data - coordinate inference
                    format_instruction = "Provide the coordinate value (e.g., 106.6)"
                else:
                    format_instruction = "Provide the answer as a number"

        elif "name" in ground_truth or "location_name" in ground_truth:
            # For place association tasks, request structured format for granularity evaluation
            if "granularity_level" in ground_truth or "country_code" in ground_truth:
                format_instruction = "Provide your answer in the format: City, Region/State, Country (e.g., 'Paris, Île-de-France, France' or 'Tokyo, Tokyo, Japan')"
            else:
                format_instruction = "Provide the location name"

        elif "terrain_type" in ground_truth:
            format_instruction = "Provide the terrain type or corresponding letter"

        # Append format instruction if detected
        if format_instruction:
            # Check if prompt already ends with "FINAL ANSWER:" line
            if not user_prompt.rstrip().endswith("FINAL ANSWER: [your answer]") and "FINAL ANSWER:" not in user_prompt.split('\n')[-3:]:
                user_prompt += f"\n\n{format_instruction}.\n\nBe brief. Show key steps only, then provide FINAL ANSWER: [your answer]"

        return user_prompt

    def create_prompt(self, task_data: Dict[str, Any], task_name: str) -> tuple[str, str]:
        """
        Create system and user prompts for a task

        Returns:
            tuple of (system_prompt, user_prompt)
        """
        # Load system prompt from template
        system_prompt = load_prompt_template("system_prompt") or "You are an expert in GPS coordinates and spatial reasoning."

        # Check if pre-formatted question exists
        if "question" in task_data:
            user_prompt = task_data["question"]

            # If question references coordinates but doesn't include them, prepend coordinate_string
            if ("this coordinate" in user_prompt.lower() or "these coordinates" in user_prompt.lower()):
                if "coordinate_string" in task_data:
                    coord_str = task_data["coordinate_string"]
                    user_prompt = f"Coordinate: {coord_str}\n\n{user_prompt}"
                elif "coordinate" in task_data and isinstance(task_data["coordinate"], dict):
                    coord = task_data["coordinate"]
                    coord_str = f"{coord.get('lat')}, {coord.get('lon')}"
                    user_prompt = f"Coordinate: {coord_str}\n\n{user_prompt}"

            # Handle terrain classification - add standard candidates if missing
            if task_data.get("task") == "terrain_classification" and "candidates" not in task_data:
                # Standard terrain types
                task_data["candidates"] = [
                    "A) Urban",
                    "B) Arctic/Ice",
                    "C) Ocean/Sea",
                    "D) Mountain",
                    "E) River/Lake",
                    "F) Desert",
                    "G) Forest",
                    "H) Grassland"
                ]

            # Handle Pure GPS tasks with point_a/point_b - replace names with coordinates
            # This applies to: distance, bearing, interpolation tasks
            if "point_a" in task_data and "point_b" in task_data:
                point_a = task_data["point_a"]
                point_b = task_data["point_b"]

                # Check if coordinates are already in the question
                coords_in_question = (
                    str(point_a.get('lat', '')) in user_prompt or
                    str(point_b.get('lat', '')) in user_prompt
                )

                # For Pure GPS tasks, replace place names with coordinates
                # Check both metadata.track and task type (for split files without metadata)
                #
                # Pure GPS tasks that have point_a/point_b:
                #   - distance_calculation (task4)
                #   - bearing_computation (task5)
                #   - interpolation (task6)
                #   - relative_position (task10) - in track_pure_gps, uses coordinates
                #
                # Applied tasks that have point_a/point_b (should NOT replace):
                #   - relative_position (task5) - in track_applied, uses place names intentionally
                #
                is_pure_gps = (
                    task_data.get('metadata', {}).get('track') == 'pure_gps' or
                    # For split files without metadata, check task type
                    task_data.get('task') in ['distance_calculation', 'bearing_computation', 'interpolation', 'relative_position']
                )

                # Ensure we don't replace for Applied tasks (even if they have point_a/point_b)
                # For relative_position, check the task_name or file path to determine track
                is_applied = task_data.get('metadata', {}).get('track') == 'applied'
                if is_applied:
                    is_pure_gps = False
                # Special handling for relative_position: if it's from track_applied, don't replace
                if task_data.get('task') == 'relative_position' and 'applied' in task_name.lower():
                    is_pure_gps = False

                if not coords_in_question and is_pure_gps:
                    name_a = point_a.get('name', 'Point A')
                    name_b = point_b.get('name', 'Point B')

                    # Replace place names in question with coordinate notation
                    # Handle various phrasings: "from X to Y", "between X and Y", etc.
                    coord_a_str = f"Point A ({point_a['lat']}, {point_a['lon']})"
                    coord_b_str = f"Point B ({point_b['lat']}, {point_b['lon']})"

                    # Replace the first occurrence of name_a and name_b with coordinates
                    if name_a in user_prompt and name_b in user_prompt:
                        # Replace name_a with coordinate notation
                        user_prompt = user_prompt.replace(name_a, coord_a_str, 1)
                        # Replace name_b with coordinate notation
                        user_prompt = user_prompt.replace(name_b, coord_b_str, 1)

            # Handle route geometry - add waypoints if mentioned but not included
            if "waypoints" in task_data and "waypoints" in user_prompt.lower():
                waypoints = task_data["waypoints"]
                # Check if waypoints are already in the prompt (they should show coordinates)
                if not any(f"{wp['lat']}" in user_prompt for wp in waypoints[:2]):
                    # Waypoints not in prompt, add them
                    waypoint_strs = []
                    for i, wp in enumerate(waypoints, 1):
                        waypoint_strs.append(f"  Waypoint {i}: ({wp['lat']}, {wp['lon']})")
                    # Insert waypoints after the "Route passes through X waypoints" line
                    lines = user_prompt.split('\n')
                    for i, line in enumerate(lines):
                        if 'waypoint' in line.lower() and 'passes through' in line.lower():
                            lines.insert(i+1, '\nWaypoints:\n' + '\n'.join(waypoint_strs) + '\n')
                            break
                    user_prompt = '\n'.join(lines)

            # Handle bounding box / boundary analysis / route geometry - add points if mentioned but not included
            # Detect if points are needed: "these points", "these cities", "points should be kept", "Douglas-Peucker"
            points_needed = (
                "these points" in user_prompt.lower() or
                "these cities" in user_prompt.lower() or
                "these " in user_prompt.lower() or
                "points should be kept" in user_prompt.lower() or
                "douglas-peucker" in user_prompt.lower() or
                (task_data.get("subtask") == "polyline_simplification" and "options" in task_data)
            )

            if "points" in task_data and points_needed:
                points = task_data["points"]
                # Check if points are already in the prompt
                if not any(f"{p['lat']}" in user_prompt for p in points[:2]):
                    # Points not in prompt, add them
                    point_strs = []

                    # For route geometry, use indexed format (Point 0, Point 1, ...)
                    if task_data.get("subtask") == "polyline_simplification" or "simplification" in user_prompt.lower():
                        for p in points:
                            idx = p.get('index', points.index(p))
                            point_strs.append(f"  Point {idx}: {p['lat']}, {p['lon']}")
                        heading = "\n\nGPS Track Points:"
                        user_prompt += heading + "\n" + '\n'.join(point_strs)
                    elif "cities by continent" in user_prompt.lower() or task_data.get('task') == 'boundary_analysis':
                        # For boundary analysis (grouping by continent), do NOT add coordinates
                        # This requires geographic knowledge - knowing which city is in which continent
                        # City names are already in the question, no additional info needed
                        pass
                    else:
                        # For bounding box and other tasks, add coordinates
                        for i, p in enumerate(points, 1):
                            name = p.get('name', f'Point {i}')
                            point_strs.append(f"  {i}. {name} ({p['lat']}, {p['lon']})")
                        heading = "\n\nPoints:"
                        user_prompt += heading + "\n" + '\n'.join(point_strs)

            # Handle polygon area - add vertices if mentioned but not included
            # Check for both polygon.vertices (original format) and vertices (test split format)
            vertices = None
            if "polygon" in task_data and "this polygon" in user_prompt.lower():
                polygon = task_data["polygon"]
                vertices = polygon.get("vertices", [])
            elif "vertices" in task_data and "this polygon" in user_prompt.lower():
                vertices = task_data["vertices"]

            if vertices:
                # Check if vertices are already in the prompt
                if not any(f"{v['lat']}" in user_prompt for v in vertices[:2]):
                    # Vertices not in prompt, add them
                    vertex_strs = []
                    for i, v in enumerate(vertices, 1):
                        name = v.get('name', f'Vertex {i}')
                        vertex_strs.append(f"  Vertex {i}: {name} ({v['lat']}, {v['lon']})")
                    user_prompt += f"\n\nPolygon vertices:\n" + '\n'.join(vertex_strs)

            # Handle multiple choice options (dict format like {A: [...], B: [...], C: [...], D: [...]})
            if "options" in task_data and isinstance(task_data["options"], dict):
                options = task_data["options"]
                option_strs = []
                for letter in sorted(options.keys()):
                    value = options[letter]
                    if isinstance(value, list):
                        # Format list values (e.g., point indices)
                        value_str = str(value)
                    else:
                        value_str = str(value)
                    option_strs.append(f"{letter}) {value_str}")
                user_prompt += "\n\nOptions:\n" + "\n".join(option_strs)
                user_prompt += "\n\nProvide your answer as FINAL ANSWER: [letter]"

            # Handle candidates (could be list of strings or list of dicts)
            elif "candidates" in task_data:
                candidates = task_data["candidates"]
                if candidates and isinstance(candidates[0], dict):
                    # Format candidate dicts (e.g., name disambiguation)
                    candidate_strs = []
                    for i, cand in enumerate(candidates):
                        if "name" in cand and "country" in cand:
                            # Format: City, Country (Population: X) - no coordinates
                            # This makes name disambiguation require geographic knowledge
                            pop = cand.get('population', 0)
                            if pop:
                                candidate_strs.append(
                                    f"{chr(65+i)}) {cand['name']}, {cand['country']} (Pop: {pop:,})"
                                )
                            else:
                                candidate_strs.append(
                                    f"{chr(65+i)}) {cand['name']}, {cand['country']}"
                                )
                        else:
                            candidate_strs.append(f"{chr(65+i)}) {str(cand)}")
                    user_prompt += "\n\nOptions:\n" + "\n".join(candidate_strs)
                else:
                    # Candidates are already strings
                    user_prompt += "\n\nOptions:\n" + "\n".join(candidates)

                user_prompt += "\n\nProvide your answer as FINAL ANSWER: [letter]"

            # Add format instruction based on ground truth
            ground_truth = task_data.get("ground_truth", {})
            user_prompt = self.add_format_instruction(user_prompt, ground_truth, example=task_data)

            return system_prompt, user_prompt

        # Task-specific prompt generation based on task type
        task_type = task_data.get("task", "")

        # Format Conversion
        if task_type == "format_conversion":
            template = load_prompt_template("format_conversion")
            if template:
                user_prompt = template.format(
                    source_format=task_data.get("source_format", ""),
                    target_format=task_data.get("target_format", ""),
                    source_string=task_data.get("source_string", "")
                )
            else:
                user_prompt = "Convert the coordinate.\n\nFINAL ANSWER: [converted coordinate]"

        # Coordinate System Transformation
        elif task_type == "coordinate_system_transformation":
            template = load_prompt_template("coordinate_system_transformation")
            if template and "source_coords" in task_data:
                source_coords = task_data["source_coords"]
                user_prompt = template.format(
                    source_crs=task_data.get("source_crs", "WGS84"),
                    target_crs=task_data.get("target_crs", "UTM"),
                    latitude=source_coords.get("lat", ""),
                    longitude=source_coords.get("lon", ""),
                    location_name=task_data.get("location_name", "")
                )
            elif "source_coords" in task_data:
                # No template, create prompt manually
                source_coords = task_data["source_coords"]
                subtask = task_data.get("subtask", "")
                user_prompt = f"""Convert the following coordinates from {task_data.get("source_crs", "WGS84")} to {task_data.get("target_crs", "UTM")}:

Location: {task_data.get("location_name", "Unknown")}
Latitude: {source_coords.get("lat")}
Longitude: {source_coords.get("lon")}

FINAL ANSWER: [converted coordinates]"""
            else:
                user_prompt = "Transform the coordinate system.\n\nFINAL ANSWER: [converted coordinates]"

        # Precision Understanding
        elif task_type == "precision_understanding":
            template = load_prompt_template("precision_understanding")
            if template:
                user_prompt = template.format(
                    coordinate_string=task_data.get("coordinate_string", ""),
                    location=task_data.get("location_name", "")
                )
            else:
                user_prompt = "Analyze the coordinate precision.\n\nFINAL ANSWER: [answer]"

        # Distance Calculation
        elif task_type == "distance_calculation" or "distance" in task_name.lower():
            template = load_prompt_template("distance_calculation")
            if template and "point_a" in task_data and "point_b" in task_data:
                point_a = task_data["point_a"]
                point_b = task_data["point_b"]
                user_prompt = template.format(
                    point_a_name=point_a.get('name', 'Location A'),
                    point_a_lat=point_a['lat'],
                    point_a_lon=point_a['lon'],
                    point_b_name=point_b.get('name', 'Location B'),
                    point_b_lat=point_b['lat'],
                    point_b_lon=point_b['lon']
                )
            else:
                user_prompt = "Calculate the requested distance.\n\nFINAL ANSWER: [distance] km"

        # Bearing Computation
        elif task_type == "bearing_computation" or "bearing" in task_name.lower():
            template = load_prompt_template("bearing_computation")
            if template and "point_a" in task_data and "point_b" in task_data:
                point_a = task_data["point_a"]
                point_b = task_data["point_b"]
                user_prompt = template.format(
                    point_a_name=point_a.get('name', 'Location A'),
                    point_a_lat=point_a['lat'],
                    point_a_lon=point_a['lon'],
                    point_b_name=point_b.get('name', 'Location B'),
                    point_b_lat=point_b['lat'],
                    point_b_lon=point_b['lon']
                )
            else:
                user_prompt = "Calculate the requested bearing.\n\nFINAL ANSWER: [bearing]°"

        # Interpolation
        elif task_type == "interpolation":
            template = load_prompt_template("interpolation")
            if template and "point_a" in task_data and "point_b" in task_data:
                point_a = task_data["point_a"]
                point_b = task_data["point_b"]
                fraction = task_data.get("fraction", 0.5)
                user_prompt = template.format(
                    point_a_name=point_a.get('name', 'Location A'),
                    point_a_lat=point_a['lat'],
                    point_a_lon=point_a['lon'],
                    point_b_name=point_b.get('name', 'Location B'),
                    point_b_lat=point_b['lat'],
                    point_b_lon=point_b['lon'],
                    fraction=fraction
                )
            else:
                user_prompt = "Calculate the interpolated point.\n\nFINAL ANSWER: [latitude], [longitude]"

        # Polygon Area
        elif task_type == "area_perimeter":
            template = load_prompt_template("area_perimeter")
            # Check for vertices in either polygon.vertices (old format) or vertices (new format)
            vertices = None
            if "polygon" in task_data:
                polygon = task_data["polygon"]
                vertices = polygon.get("vertices", [])
            elif "vertices" in task_data:
                vertices = task_data["vertices"]

            if template and vertices:
                vertex_strs = []
                for i, v in enumerate(vertices, 1):
                    vertex_strs.append(f"  Vertex {i}: {v.get('name', f'Point {i}')} ({v['lat']}, {v['lon']})")
                user_prompt = template.format(vertices=chr(10).join(vertex_strs))
            else:
                user_prompt = "Calculate the area of the polygon.\n\nFINAL ANSWER: [area] km²"

        # Bounding Box
        elif task_type == "bounding_box":
            template = load_prompt_template("bounding_box")
            if template and "points" in task_data:
                points = task_data["points"]
                point_strs = [f"  - {p.get('name', 'Point')} ({p['lat']}, {p['lon']})" for p in points]
                user_prompt = template.format(
                    num_points=len(points),
                    points=chr(10).join(point_strs)
                )
            else:
                user_prompt = "Calculate the bounding box.\n\nFINAL ANSWER: [min_lat], [max_lat], [min_lon], [max_lon]"

        # Route Geometry
        elif task_type == "route_geometry" or ("route" in task_data) or ("waypoints" in task_data):
            # Handle both "route" and "waypoints" fields
            if "waypoints" in task_data:
                # New format with waypoints array
                template = load_prompt_template("route_geometry_waypoints")
                if template:
                    waypoints = task_data["waypoints"]
                    point_strs = [f"  Waypoint {i+1}: ({p['lat']}, {p['lon']})"
                                 for i, p in enumerate(waypoints)]
                    user_prompt = template.format(
                        num_waypoints=len(waypoints),
                        waypoints=chr(10).join(point_strs)
                    )
                else:
                    user_prompt = "Analyze the route geometry.\n\nFINAL ANSWER: [your answer]"
            elif "route" in task_data:
                # Old format with route object
                template = load_prompt_template("route_geometry_route")
                if template:
                    route = task_data["route"]
                    points = route.get("points", [])
                    question = task_data.get("geometry_question", "Analyze the route geometry.")
                    point_strs = [f"  Point {i+1}: {p.get('name', f'P{i+1}')} ({p['lat']}, {p['lon']})"
                                 for i, p in enumerate(points[:5])]
                    if len(points) > 5:
                        point_strs.append(f"  ... and {len(points)-5} more points")
                    user_prompt = template.format(
                        points=chr(10).join(point_strs),
                        question=question
                    )
                else:
                    user_prompt = "Analyze the route geometry.\n\nFINAL ANSWER: [your answer]"
            else:
                user_prompt = "Analyze the route geometry.\n\nFINAL ANSWER: [your answer]"

        # Place Association (coordinate-based)
        elif task_type == "place_association" and "coordinate_string" in task_data:
            template = load_prompt_template("place_association")
            if template:
                user_prompt = template.format(
                    coordinate_string=task_data["coordinate_string"]
                )
            else:
                user_prompt = "What location/place is at the given coordinates?\n\nFINAL ANSWER: [location name]"

        # Fallback for unknown task types
        else:
            # Try to use coordinate_string if available
            if "coordinate_string" in task_data:
                coord_str = task_data["coordinate_string"]
                user_prompt = f"""Analyze the coordinates: {coord_str}

Task: {task_type or task_name}

Provide your analysis.

FINAL ANSWER: [your answer]"""
            else:
                # Last resort: dump relevant data
                user_prompt = f"Task: {task_type or task_name}\n\n"
                user_prompt += str(task_data.get("prompt", "Analyze the given data."))
                user_prompt += "\n\nFINAL ANSWER: [your answer]"

        # Add format instruction based on ground truth
        ground_truth = task_data.get("ground_truth", {})
        user_prompt = self.add_format_instruction(user_prompt, ground_truth, example=task_data)

        return system_prompt, user_prompt

    def _parse_cached_batch_results(self, results_file: str, expected_count: int) -> List:
        """
        Parse cached batch results from local JSONL file.
        Handles both Gemini and OpenAI result formats.

        Args:
            results_file: Path to the cached results JSONL file
            expected_count: Expected number of results

        Returns:
            List of LLMResponse objects in original order
        """
        from evaluation.llm_client import LLMResponse

        results_dict = {}

        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                result = json.loads(line)

                # Handle Gemini format (key: "request-{idx}")
                if 'key' in result:
                    key = result.get('key', '')
                    try:
                        idx = int(key.split('-')[-1])
                    except (ValueError, IndexError):
                        continue

                    if 'response' in result and result['response']:
                        response = result['response']
                        text = ""
                        if 'candidates' in response and response['candidates']:
                            candidate = response['candidates'][0]
                            if 'content' in candidate and 'parts' in candidate['content']:
                                for part in candidate['content']['parts']:
                                    if 'text' in part:
                                        text = part['text']
                                        break
                        usage = response.get('usageMetadata', {})
                        results_dict[idx] = LLMResponse(
                            text=text.strip() if text else "",
                            model=self.llm_client.model,
                            prompt_tokens=usage.get('promptTokenCount', 0) or 0,
                            completion_tokens=usage.get('candidatesTokenCount', 0) or 0,
                            total_tokens=usage.get('totalTokenCount', 0) or 0,
                            error=None
                        )
                    elif 'error' in result:
                        results_dict[idx] = LLMResponse(
                            text="",
                            model=self.llm_client.model,
                            error=str(result['error'])
                        )

                # Handle OpenAI format (custom_id: "request-{idx}")
                elif 'custom_id' in result:
                    custom_id = result['custom_id']
                    idx = int(custom_id.split('-')[-1])

                    if result['response']['status_code'] == 200:
                        body = result['response']['body']
                        if 'output' in body:
                            # Responses API format
                            output = body['output']
                            if isinstance(output, list):
                                text = ""
                                for item in output:
                                    if item.get('type') == 'message' and 'content' in item:
                                        for content_item in item['content']:
                                            if content_item.get('type') == 'output_text':
                                                text = content_item.get('text', '')
                                                break
                                        if text:
                                            break
                            else:
                                text = str(output)
                        elif 'choices' in body:
                            text = body['choices'][0]['message']['content'].strip()
                        else:
                            text = str(body)

                        usage = body.get('usage', {})
                        results_dict[idx] = LLMResponse(
                            text=text.strip() if isinstance(text, str) else str(text),
                            model=self.llm_client.model,
                            prompt_tokens=usage.get('prompt_tokens', 0),
                            completion_tokens=usage.get('completion_tokens', 0),
                            total_tokens=usage.get('total_tokens', 0),
                            error=None
                        )
                    else:
                        error_msg = result['response'].get('body', {}).get('error', {}).get('message', 'Unknown error')
                        results_dict[idx] = LLMResponse(
                            text="",
                            model=self.llm_client.model,
                            error=error_msg
                        )

        # Convert to ordered list
        results = []
        for i in range(expected_count):
            if i in results_dict:
                results.append(results_dict[i])
            else:
                results.append(LLMResponse(
                    text="",
                    model=self.llm_client.model,
                    error="Result not found in cached batch output"
                ))

        return results

    def evaluate_task_batch(
        self,
        track_name: str,
        task_info: Dict[str, str],
        max_samples: Optional[int] = None,
        batch_dir: str = "batch_files"
    ) -> str:
        """
        Prepare and submit a task for batch evaluation using OpenAI Batch API
        Returns batch_id for later retrieval

        Args:
            track_name: Name of the track
            task_info: Task information dictionary
            max_samples: Optional limit on number of samples
            batch_dir: Directory to store batch files

        Returns:
            Batch job ID
        """
        import os

        # Create batch directory
        os.makedirs(batch_dir, exist_ok=True)

        # Load task data
        task_name = task_info["file"].replace('.json', '')
        task_file = self.data_dir / f"track_{track_name}" / "splits" / f"{task_name}_test.json"

        if not task_file.exists():
            raise FileNotFoundError(f"Test split not found: {task_file}")

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        if max_samples:
            task_data = task_data[:max_samples]

        print(f"\n{'='*80}")
        print(f"Preparing Batch: {task_info['name']}")
        print(f"Samples: {len(task_data)}")
        print(f"{'='*80}")

        # Create prompts for all examples
        prompts = []
        system_prompts = []

        for example in task_data:
            system_prompt, user_prompt = self.create_prompt(example, task_info["name"])
            prompts.append(user_prompt)
            system_prompts.append(system_prompt)

        # Check all system prompts are the same
        if len(set(system_prompts)) > 1:
            print("Warning: Multiple different system prompts detected")

        system_prompt = system_prompts[0] if system_prompts else None

        # Create batch file
        batch_filename = f"{track_name}_{task_name}_batch.jsonl"
        batch_filepath = os.path.join(batch_dir, batch_filename)

        self.llm_client.create_batch_file(
            prompts=prompts,
            system_prompt=system_prompt,
            output_file=batch_filepath
        )

        print(f"Created batch file: {batch_filepath}")

        # Submit batch
        batch_id = self.llm_client.submit_batch(batch_filepath)
        print(f"Submitted batch job: {batch_id}")

        # Save metadata for later result processing
        metadata = {
            "batch_id": batch_id,
            "track_name": track_name,
            "task_info": task_info,
            "task_file": str(task_file),
            "num_samples": len(task_data),
            "batch_file": batch_filepath,
            "task_data": task_data  # Save for evaluation
        }

        # Sanitize batch_id for filename (Gemini batch IDs contain '/')
        safe_batch_id = batch_id.replace('/', '_')
        metadata_file = os.path.join(batch_dir, f"{safe_batch_id}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return batch_id

    def evaluate_task(
        self,
        track_name: str,
        task_info: Dict[str, str],
        max_samples: Optional[int] = None,
        delay: float = 0.5,
        use_batch_api: bool = False,
        batch_dir: str = "batch_files",
        use_concurrent: bool = False,
        max_workers: int = 10
    ) -> Dict[str, Any]:
        """Evaluate a single task"""

        # USE TEST SPLIT ONLY - NO FALLBACK
        task_name = task_info["file"].replace('.json', '')
        task_file = self.data_dir / f"track_{track_name}" / "splits" / f"{task_name}_test.json"

        if not task_file.exists():
            print(f"\n{'='*80}")
            print(f"❌ ERROR: Test split not found!")
            print(f"File: {task_file}")
            print(f"\nThe benchmark MUST run on test splits only to prevent data leakage.")
            print(f"Please generate test splits using: python scripts/create_splits.py --track both")
            print(f"{'='*80}\n")
            return {
                "task_name": task_info["name"],
                "status": "test_split_not_found",
                "error": f"Test split not found: {task_file}",
                "accuracy": 0.0,
                "total": 0,
                "correct": 0
            }

        # Load task data from test split
        with open(task_file, 'r') as f:
            task_data = json.load(f)

        # Limit samples if requested
        if max_samples:
            task_data = task_data[:max_samples]

        print(f"\n{'='*80}")
        print(f"Task: {task_info['name']}")
        print(f"File: {task_info['file']}")
        print(f"Samples: {len(task_data)}")
        if use_concurrent:
            print(f"Mode: Concurrent ({max_workers} workers)")
        print(f"{'='*80}")

        results = []
        correct = 0

        # Concurrent processing mode
        if use_concurrent:
            # Prepare all prompts
            prompts = []
            system_prompts = []
            for example in task_data:
                system_prompt, user_prompt = self.create_prompt(example, task_info["name"])
                prompts.append(user_prompt)
                system_prompts.append(system_prompt)

            # Check all system prompts are the same
            if len(set(system_prompts)) > 1:
                print("Warning: Multiple different system prompts detected, using first one")
            system_prompt = system_prompts[0] if system_prompts else None

            # Batch generate with concurrent requests
            print(f"Processing {len(prompts)} requests concurrently (max {max_workers} parallel workers)...")
            responses = self.llm_client.batch_generate(
                prompts=prompts,
                system_prompt=system_prompt,
                delay=0,  # No delay needed for concurrent requests
                max_workers=max_workers
            )

            # Evaluate responses
            for idx, (example, response) in enumerate(tqdm(
                zip(task_data, responses),
                total=len(task_data),
                desc=f"Evaluating {task_info['name']}"
            )):
                if response.error:
                    print(f"❌ Error on example {idx}: {response.error}")
                    results.append({
                        "example_id": idx,
                        "prompt": prompts[idx],
                        "system_prompt": system_prompt,
                        "response": "",
                        "ground_truth": example.get("ground_truth", {}),
                        "error": response.error,
                        "correct": False
                    })
                    continue

                # Evaluate answer
                is_correct = self.simple_evaluate(response.text, example)

                if is_correct:
                    correct += 1

                results.append({
                    "example_id": idx,
                    "prompt": prompts[idx],
                    "system_prompt": system_prompt,
                    "response": response.text,
                    "ground_truth": example.get("ground_truth", {}),
                    "correct": is_correct,
                    "latency_ms": response.latency_ms,
                    "tokens": response.total_tokens
                })

        # Sequential processing mode (original)
        else:
            # Process each example
            for idx, example in enumerate(tqdm(task_data, desc=f"Evaluating {task_info['name']}")):
                try:
                    # Create prompt
                    system_prompt, user_prompt = self.create_prompt(example, task_info["name"])

                    # Get LLM response
                    response = self.llm_client.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt
                    )

                    if response.error:
                        print(f"❌ Error on example {idx}: {response.error}")
                        results.append({
                            "example_id": idx,
                            "prompt": user_prompt,
                            "system_prompt": system_prompt,
                            "response": "",
                            "ground_truth": example.get("ground_truth", {}),
                            "error": response.error,
                            "correct": False
                        })
                        continue

                    # Extract answer
                    answer_text = response.text

                    # Simple evaluation (can be enhanced with task-specific evaluators)
                    is_correct = self.simple_evaluate(answer_text, example)

                    if is_correct:
                        correct += 1

                    results.append({
                        "example_id": idx,
                        "prompt": user_prompt,
                        "system_prompt": system_prompt,
                        "response": answer_text,
                        "ground_truth": example.get("ground_truth", {}),
                        "correct": is_correct,
                        "latency_ms": response.latency_ms,
                        "tokens": response.total_tokens
                    })

                    # Rate limiting
                    if delay > 0:
                        time.sleep(delay)

                except Exception as e:
                    print(f"❌ Exception on example {idx}: {e}")
                    # Try to include prompt/ground_truth if available
                    result = {
                        "example_id": idx,
                        "error": str(e),
                        "correct": False
                    }
                    try:
                        result["prompt"] = user_prompt
                        result["system_prompt"] = system_prompt
                        result["ground_truth"] = example.get("ground_truth", {})
                    except:
                        pass
                    results.append(result)

        # Calculate metrics
        accuracy = (correct / len(task_data) * 100) if task_data else 0.0

        return {
            "task_name": task_info["name"],
            "task_file": task_info["file"],
            "total": len(task_data),
            "correct": correct,
            "accuracy": accuracy,
            "results": results
        }

    def extract_numbers(self, text: str) -> List[float]:
        """
        Extract numbers from text, handling commas in large numbers.

        Examples:
            "7,241,000 km²" -> [7241000.0]
            "15.5 km" -> [15.5]
            "-42.123" -> [-42.123]
        """
        import re

        # Match numbers with optional commas and decimals
        # Pattern: optional minus, digits with optional commas, optional decimal part
        pattern = r'-?\d+(?:,\d+)*(?:\.\d+)?'
        matches = re.findall(pattern, text)

        # Convert to float, removing commas
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match.replace(',', '')))
            except ValueError:
                continue

        return numbers

    def simple_evaluate(self, response: str, example: Dict[str, Any]) -> bool:
        """
        Simple evaluation - extract FINAL ANSWER and compare to ground truth

        This is a basic evaluator. For production, use task-specific evaluators.
        """
        # Extract FINAL ANSWER
        final_answer = None
        for line in response.split('\n'):
            if 'FINAL ANSWER:' in line.upper():
                final_answer = line.split(':', 1)[1].strip()
                break

        if not final_answer:
            # Fallback: try to extract answer from end of response
            lines = response.strip().split('\n')
            # Check last few lines for an answer
            for line in reversed(lines[-10:]):
                line = line.strip()
                # Skip common explanation starters and empty lines
                if line and not line.startswith(('To ', 'The ', 'Step ', 'First', 'Next', 'Calculate', 'We ', 'Using')):
                    # Prefer lines with = (final calculations) or lines with numbers
                    if '=' in line or any(c.isdigit() for c in line):
                        final_answer = line
                        break

        if not final_answer:
            # Last resort: use full response
            final_answer = response.strip()

        # Get ground truth
        ground_truth = example.get("ground_truth", {})
        task_type = example.get("task", "")

        # MCQ answer (options field present with string answer)
        if "options" in example and "answer" in ground_truth and isinstance(ground_truth["answer"], str):
            expected = ground_truth["answer"].lower()
            answer_lower = final_answer.lower().strip()
            options = [opt.lower() for opt in example["options"]]

            # Check if the expected answer word appears in the response
            if expected in answer_lower:
                return True

            # Check if answer is just the option word or starts with it
            for opt in options:
                if answer_lower == opt or answer_lower.startswith(opt + " ") or answer_lower.startswith(opt + "."):
                    return opt == expected

            # Check single letter answer (e.g., "N" for North when options are North/South/East/West)
            if len(answer_lower) == 1 and answer_lower.isalpha():
                for opt in options:
                    if opt.startswith(answer_lower):
                        return opt == expected

            return False

        # Boolean answer (True/False questions)
        if "answer" in ground_truth and isinstance(ground_truth["answer"], bool):
            expected = ground_truth["answer"]
            # Check for Yes/True or No/False in response
            answer_lower = final_answer.lower()
            if expected:
                return any(word in answer_lower for word in ["yes", "true", "correct"])
            else:
                return any(word in answer_lower for word in ["no", "false", "incorrect"])

        # Name Disambiguation special case: ground truth has "name" but LLM answers with letter
        if "name" in ground_truth and "candidates" in example:
            expected_name = ground_truth["name"]
            candidates = example["candidates"]

            # Check if LLM answered with a letter
            stripped_answer = final_answer.strip()
            if stripped_answer and len(stripped_answer) <= 3 and stripped_answer[0].upper().isalpha():
                # LLM gave letter answer - map to candidate name
                letter = stripped_answer[0].upper()
                letter_index = ord(letter) - ord('A')

                if 0 <= letter_index < len(candidates):
                    selected_candidate = candidates[letter_index]
                    # Candidates are dicts with 'name' field
                    if isinstance(selected_candidate, dict):
                        selected_name = selected_candidate.get('name', '')
                        return selected_name.lower() == expected_name.lower()

            # Otherwise check if name is in response
            return expected_name.lower() in final_answer.lower()

        # Multiple choice - check answer letter (only for single-letter answers)
        if "answer" in ground_truth:
            answer_value = ground_truth["answer"]

            # Check if this is a numeric answer with tolerance (e.g., missing data inference)
            if isinstance(answer_value, (int, float)) and "tolerance" in ground_truth:
                # Numeric answer with tolerance - use numeric comparison
                response_nums = self.extract_numbers(final_answer)
                if response_nums:
                    try:
                        predicted = float(response_nums[0])
                        expected = float(answer_value)
                        tolerance = ground_truth["tolerance"]
                        return abs(predicted - expected) <= tolerance
                    except (ValueError, IndexError):
                        pass

            # Convert to string for other checks
            expected = str(answer_value).strip()

            # Check if this is a single-letter answer (A, B, C, D, etc.)
            if len(expected) == 1 and expected.isalpha():
                # Extract just the letter from response
                predicted = final_answer[0].upper() if final_answer else ""
                return predicted == expected.upper()
            else:
                # String answer (e.g., city name) - check if answer is in response
                return expected.lower() in final_answer.lower()

        # Location name matching
        if "name" in ground_truth or "location_name" in ground_truth or "location" in ground_truth:
            expected = (ground_truth.get("name") or ground_truth.get("location_name") or ground_truth.get("location", "")).lower()
            return expected in final_answer.lower()

        # Terrain type - check for letter answer or terrain name
        if "terrain_type" in ground_truth:
            expected_terrain = ground_truth["terrain_type"].lower()
            answer_lower = final_answer.lower()

            # Check if answered with letter (e.g., "G" for Forest)
            if "answer" in ground_truth:
                expected_letter = ground_truth["answer"].strip().upper()
                # Extract first letter from response
                for char in final_answer:
                    if char.isalpha():
                        if char.upper() == expected_letter:
                            return True
                        break  # Only check first letter

            # Fallback: Check if terrain type name is in answer
            return expected_terrain in answer_lower

        # Interpolation - midpoint or interpolated_point coordinates
        interpolation_point = ground_truth.get("midpoint") or ground_truth.get("interpolated_point")
        if interpolation_point and isinstance(interpolation_point, dict):
            expected_lat = interpolation_point.get("lat")
            expected_lon = interpolation_point.get("lon")

            # Extract numbers from response
            numbers = self.extract_numbers(final_answer)
            if len(numbers) >= 2:
                try:
                    predicted_lat = numbers[0]
                    predicted_lon = numbers[1]

                    # Check with tolerance
                    tolerance_km = ground_truth.get("tolerance_km", 5.0)
                    # ~1 degree ≈ 111 km
                    tolerance_deg = tolerance_km / 111.0

                    lat_match = abs(predicted_lat - expected_lat) <= tolerance_deg
                    lon_match = abs(predicted_lon - expected_lon) <= tolerance_deg

                    return lat_match and lon_match
                except (ValueError, IndexError):
                    pass

        # Route Geometry - polyline simplification (points_kept)
        if "points_kept" in ground_truth:
            import re
            expected_points = set(ground_truth["points_kept"])

            # Extract all numbers from response
            numbers = re.findall(r'\d+', final_answer)
            predicted_points = set(int(n) for n in numbers if n.isdigit())

            # Check if sets match
            return predicted_points == expected_points

        # Precision Understanding - multi-field answer
        if task_type == "precision_understanding" and all(k in ground_truth for k in ["decimal_places", "accuracy_meters"]):
            import re
            numbers = re.findall(r'\d+', final_answer.replace(',', ''))

            if len(numbers) >= 2:
                try:
                    # Extract decimal places and accuracy
                    predicted_decimal = int(numbers[0])
                    predicted_accuracy = int(numbers[1])

                    expected_decimal = ground_truth["decimal_places"]
                    expected_accuracy = ground_truth["accuracy_meters"]

                    # Check decimal places (exact match)
                    decimal_match = predicted_decimal == expected_decimal

                    # Check accuracy (within 20% tolerance due to approximation)
                    accuracy_tolerance = expected_accuracy * 0.2
                    accuracy_match = abs(predicted_accuracy - expected_accuracy) <= accuracy_tolerance

                    return decimal_match and accuracy_match
                except (ValueError, IndexError):
                    pass

        # Grouping/boundary analysis (check if key continents/regions are mentioned)
        if "grouping" in ground_truth:
            grouping = ground_truth["grouping"]
            # Check if all groups are mentioned in the answer
            answer_lower = final_answer.lower()
            groups_found = sum(1 for group_name in grouping.keys() if group_name.lower() in answer_lower)
            # Consider correct if at least half the groups are mentioned
            return groups_found >= len(grouping) / 2

        # Polygon Area - parse area value from response
        if "area_km2" in ground_truth:
            try:
                # Extract numbers from response
                numbers = self.extract_numbers(final_answer)
                if numbers:
                    predicted_area = float(numbers[0])
                    expected_area = ground_truth["area_km2"]

                    # Allow 5% tolerance
                    area_tolerance = abs(expected_area * 0.05) if expected_area != 0 else 1.0
                    area_correct = abs(predicted_area - expected_area) <= area_tolerance

                    return area_correct
            except (ValueError, AttributeError):
                pass

        # Coordinate Transformation - special case for (x, y) or (easting, northing) pairs
        # Handles Web Mercator (x, y) and UTM (easting, northing)
        coord_pairs = [
            ("x", "y"),  # Web Mercator
            ("easting", "northing"),  # UTM
        ]

        for coord1_key, coord2_key in coord_pairs:
            if coord1_key in ground_truth and coord2_key in ground_truth:
                numbers = self.extract_numbers(final_answer)

                if len(numbers) >= 2:
                    try:
                        predicted_1 = float(numbers[0])
                        predicted_2 = float(numbers[1])
                        expected_1 = ground_truth[coord1_key]
                        expected_2 = ground_truth[coord2_key]

                        # Use 1% tolerance for coordinate transformations (they should be precise)
                        # Web Mercator, UTM have meter precision
                        tolerance_percent = 0.01  # 1%
                        tolerance_1 = abs(expected_1 * tolerance_percent) if expected_1 != 0 else 100.0  # 100m minimum
                        tolerance_2 = abs(expected_2 * tolerance_percent) if expected_2 != 0 else 100.0

                        coord1_match = abs(predicted_1 - expected_1) <= tolerance_1
                        coord2_match = abs(predicted_2 - expected_2) <= tolerance_2

                        return coord1_match and coord2_match
                    except (ValueError, IndexError):
                        pass

        # Bounding Box - special case for 4 coordinates
        if all(k in ground_truth for k in ["min_lat", "max_lat", "min_lon", "max_lon"]):
            numbers = self.extract_numbers(final_answer)

            if len(numbers) >= 4:
                try:
                    predicted = [float(n) for n in numbers[:4]]
                    expected = [
                        ground_truth["min_lat"],
                        ground_truth["max_lat"],
                        ground_truth["min_lon"],
                        ground_truth["max_lon"]
                    ]

                    # Check each coordinate with tolerance
                    tolerance_deg = ground_truth.get("tolerance_deg", 0.1)
                    matches = all(abs(p - e) <= tolerance_deg for p, e in zip(predicted, expected))

                    return matches
                except (ValueError, IndexError):
                    pass

        # Numeric comparison (distance, bearing, etc.)
        numeric_keys = [
            "distance", "distance_km", "distance_miles",
            "bearing", "initial_bearing_deg", "final_bearing_deg",
            "latitude", "longitude",
            "original_lat", "original_lon",  # For missing data inference task
            "min_lat", "max_lat", "min_lon", "max_lon",
            "straightness_ratio"  # For route geometry task
        ]

        if any(key in ground_truth for key in numeric_keys):
            # Extract numbers from both
            response_nums = self.extract_numbers(final_answer)
            if response_nums:
                predicted = float(response_nums[0])

                # Try each numeric key in priority order
                for key in numeric_keys:
                    if key in ground_truth:
                        expected = ground_truth[key]

                        # Check for task-specific tolerance fields first
                        if "tolerance_deg" in ground_truth:
                            # Absolute degree tolerance (for bearing tasks)
                            tolerance = ground_truth["tolerance_deg"]
                        elif "tolerance_km" in ground_truth:
                            # Absolute km tolerance (for distance tasks)
                            tolerance = ground_truth["tolerance_km"]
                        elif "tolerance_percent" in ground_truth:
                            # Percentage tolerance
                            tolerance = abs(expected * ground_truth["tolerance_percent"] / 100.0)
                        else:
                            # Default: 5% tolerance
                            tolerance = abs(expected * 0.05) if expected != 0 else 0.1

                        return abs(predicted - expected) <= tolerance

        # Default: exact string match
        return final_answer.lower() == str(ground_truth).lower()

    def filter_tasks(self, tasks: List[Dict], task_filter: Optional[List[str]]) -> List[Dict]:
        """Filter tasks based on partial name matching."""
        if not task_filter:
            return tasks

        filtered = []
        for task_info in tasks:
            task_file = task_info['file'].lower()
            task_name = task_info['name'].lower()

            for pattern in task_filter:
                pattern_lower = pattern.lower().replace('_', ' ').replace('-', ' ')
                # Match against file name or task name
                if (pattern_lower in task_file.replace('_', ' ') or
                    pattern_lower in task_name or
                    pattern.lower() in task_file):
                    filtered.append(task_info)
                    break

        return filtered

    def evaluate_track(
        self,
        track_name: str,
        max_samples: Optional[int] = None,
        delay: float = 0.5,
        use_concurrent: bool = False,
        max_workers: int = 10,
        task_filter: Optional[List[str]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Evaluate all tasks in a track

        Args:
            track_name: Name of the track to evaluate
            max_samples: Maximum samples per task
            delay: Delay between API calls
            use_concurrent: Use concurrent processing
            max_workers: Number of concurrent workers
            task_filter: Filter for specific tasks
            output_dir: Output directory for incremental saving (if None, uses self._current_output_dir)
        """

        if track_name not in self.tracks:
            raise ValueError(f"Unknown track: {track_name}. Choose from: {list(self.tracks.keys())}")

        track_info = self.tracks[track_name]
        tasks_to_run = self.filter_tasks(track_info["tasks"], task_filter)

        if not tasks_to_run:
            print(f"\nNo tasks matched filter: {task_filter}")
            return None

        # Use provided output_dir or fall back to instance variable
        save_dir = output_dir or self._current_output_dir

        print(f"\n{'='*80}")
        print(f"Evaluating {track_info['name']}")
        print(f"Tasks to run: {len(tasks_to_run)}/{len(track_info['tasks'])}")
        if task_filter:
            print(f"Task filter: {task_filter}")
        print(f"{'='*80}")

        task_results = []
        total_correct = 0
        total_cases = 0

        for task_info in tasks_to_run:
            task_result = self.evaluate_task(
                track_name=track_name,
                task_info=task_info,
                max_samples=max_samples,
                delay=delay,
                use_concurrent=use_concurrent,
                max_workers=max_workers
            )
            task_results.append(task_result)
            total_correct += task_result.get("correct", 0)
            total_cases += task_result.get("total", 0)

            # Save task result incrementally
            if save_dir:
                self.save_task_result_incremental(save_dir, track_name, task_result)

        overall_accuracy = (total_correct / total_cases * 100) if total_cases > 0 else 0.0

        return {
            "track_name": track_name,
            "track_display_name": track_info["name"],
            "total_tasks": len(track_info["tasks"]),
            "total_cases": total_cases,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy,
            "tasks": task_results
        }

    def evaluate_all(
        self,
        max_samples: Optional[int] = None,
        delay: float = 0.5,
        use_concurrent: bool = False,
        max_workers: int = 10,
        task_filter: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        output_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate both tracks

        Args:
            max_samples: Maximum samples per task
            delay: Delay between API calls
            use_concurrent: Use concurrent processing
            max_workers: Number of concurrent workers
            task_filter: Filter for specific tasks
            output_dir: Output directory for incremental saving (creates one if None)
            output_name: Name for the output folder (used if output_dir is None)
        """

        # Initialize output directory for incremental saving
        if output_dir is None:
            output_dir = self.init_output_dir(output_name)

        results = {
            "model": self.llm_client.model,
            "provider": self.llm_client.provider,
            "timestamp": datetime.now().isoformat(),
            "max_samples_per_task": max_samples,
            "tracks": {}
        }

        for track_name in self.tracks.keys():
            track_result = self.evaluate_track(
                track_name=track_name,
                max_samples=max_samples,
                delay=delay,
                use_concurrent=use_concurrent,
                max_workers=max_workers,
                task_filter=task_filter,
                output_dir=output_dir
            )
            if track_result:  # Skip if no tasks matched filter
                results["tracks"][track_name] = track_result

        if not results["tracks"]:
            print(f"\nNo tasks matched filter: {task_filter}")
            return None

        # Overall metrics
        total_correct = sum(t["total_correct"] for t in results["tracks"].values())
        total_cases = sum(t["total_cases"] for t in results["tracks"].values())
        overall_accuracy = (total_correct / total_cases * 100) if total_cases > 0 else 0.0

        results["overall"] = {
            "total_cases": total_cases,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy
        }

        return results

    def process_batch_results(
        self,
        batch_id: str,
        batch_dir: str = "batch_files",
        auto_retry: bool = False,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Retrieve and evaluate results from a completed batch job

        Args:
            batch_id: Batch job ID
            batch_dir: Directory containing batch metadata
            auto_retry: Automatically retry failed samples
            max_retries: Maximum number of retry attempts

        Returns:
            Task results dictionary
        """
        # Load metadata (sanitize batch_id for filename - Gemini batch IDs contain '/')
        safe_batch_id = batch_id.replace('/', '_')
        metadata_file = os.path.join(batch_dir, f"{safe_batch_id}_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        task_data = metadata["task_data"]
        task_info = metadata["task_info"]

        # Retrieve batch results (use cached file if available)
        results_file = os.path.join(batch_dir, f"{safe_batch_id}_results.jsonl")

        # Check if results are already cached locally
        if os.path.exists(results_file):
            # Use cached results - parse from local file
            llm_responses = self._parse_cached_batch_results(results_file, len(task_data))
        else:
            # Download from API
            llm_responses = self.llm_client.retrieve_batch_results(batch_id, results_file, expected_count=len(task_data))

        print(f"\n{'='*80}")
        print(f"Processing Batch Results: {task_info['name']}")
        print(f"Samples: {len(task_data)}")
        print(f"{'='*80}")

        # Evaluate each response and collect failed indices
        results = []
        correct = 0
        failed_indices = []

        for idx, (example, llm_response) in enumerate(zip(task_data, llm_responses)):
            if llm_response.error:
                print(f"❌ Error on example {idx}: {llm_response.error}")
                failed_indices.append(idx)
                results.append({
                    "example_id": idx,
                    "ground_truth": example.get("ground_truth", {}),
                    "response": "",
                    "error": llm_response.error,
                    "correct": False
                })
                continue

            # Evaluate answer
            is_correct = self.simple_evaluate(llm_response.text, example)

            if is_correct:
                correct += 1

            # Create prompt for logging (reconstruct it)
            system_prompt, user_prompt = self.create_prompt(example, task_info["name"])

            results.append({
                "example_id": idx,
                "prompt": user_prompt,
                "system_prompt": system_prompt,
                "response": llm_response.text,
                "ground_truth": example.get("ground_truth", {}),
                "correct": is_correct,
                "tokens": llm_response.total_tokens
            })

        # Handle retries if enabled and there are failures
        if auto_retry and failed_indices:
            print(f"\n⚠️  Found {len(failed_indices)} failed samples. Auto-retry enabled...")
            retry_result = self.retry_failed_batch(
                batch_id=batch_id,
                failed_indices=failed_indices,
                batch_dir=batch_dir,
                max_retries=max_retries
            )

            # Merge retry results back
            if retry_result:
                for idx, retry_data in retry_result.items():
                    results[idx] = retry_data
                    if retry_data.get("correct", False):
                        correct += 1

                print(f"✅ Retry complete. Updated {len(retry_result)} samples.")

        # Calculate metrics
        accuracy = (correct / len(task_data) * 100) if task_data else 0.0

        # Add failure info to results
        final_failed = [i for i in range(len(results)) if results[i].get("error") is not None]

        return {
            "task_name": task_info["name"],
            "task_file": task_info["file"],
            "total": len(task_data),
            "correct": correct,
            "accuracy": accuracy,
            "results": results,
            "batch_id": batch_id,
            "failed_count": len(final_failed),
            "failed_indices": final_failed
        }

    def retry_failed_batch(
        self,
        batch_id: str,
        failed_indices: List[int],
        batch_dir: str = "batch_files",
        max_retries: int = 2
    ) -> Dict[int, Dict[str, Any]]:
        """
        Retry failed samples from a batch job

        Args:
            batch_id: Original batch job ID
            failed_indices: List of indices that failed
            batch_dir: Directory containing batch metadata
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary mapping index to updated result data
        """
        if not failed_indices:
            return {}

        # Load metadata from original batch (sanitize batch_id for filename)
        safe_batch_id = batch_id.replace('/', '_')
        metadata_file = os.path.join(batch_dir, f"{safe_batch_id}_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        task_data = metadata["task_data"]
        task_info = metadata["task_info"]

        # Create prompts for all samples (needed for retry)
        all_prompts = []
        system_prompts = []
        for example in task_data:
            system_prompt, user_prompt = self.create_prompt(example, task_info["name"])
            all_prompts.append(user_prompt)
            system_prompts.append(system_prompt)

        system_prompt = system_prompts[0] if system_prompts else None

        retry_count = 0
        updated_results = {}
        remaining_failures = list(failed_indices)

        while retry_count < max_retries and remaining_failures:
            retry_count += 1
            print(f"\n🔄 Retry attempt {retry_count}/{max_retries} for {len(remaining_failures)} samples...")

            # Submit retry batch (use safe_batch_id for filename)
            retry_file = os.path.join(batch_dir, f"{safe_batch_id}_retry_{retry_count}.jsonl")
            retry_batch_id = self.llm_client.retry_failed_batch_samples(
                failed_indices=remaining_failures,
                original_prompts=all_prompts,
                system_prompt=system_prompt,
                output_file=retry_file
            )

            print(f"  Submitted retry batch: {retry_batch_id}")
            print(f"  Monitoring retry batch...")

            # Monitor retry batch
            retry_completed = False
            poll_count = 0
            max_polls = 1440  # 24 hours at 60 second intervals

            while not retry_completed and poll_count < max_polls:
                time.sleep(60)  # Wait 60 seconds
                poll_count += 1

                status_info = self.llm_client.check_batch_status(retry_batch_id)

                if status_info['status'] == 'completed':
                    retry_completed = True
                    break
                elif status_info['status'] in ['failed', 'expired', 'cancelled']:
                    print(f"  ❌ Retry batch {retry_batch_id[:12]}... {status_info['status']}")
                    break

                if poll_count % 5 == 0:  # Print update every 5 minutes
                    retry_counts = status_info['request_counts']
                    retry_progress = f"({retry_counts['completed']}/{retry_counts['total']})" if retry_counts['total'] > 0 else "(processing)"
                    print(f"  Retry batch status: {status_info['status']} {retry_progress}")

            if not retry_completed:
                print(f"  ⚠️  Retry batch did not complete successfully")
                break

            # Process retry results (sanitize retry_batch_id for filename)
            safe_retry_batch_id = retry_batch_id.replace('/', '_')
            retry_results_file = os.path.join(batch_dir, f"{safe_retry_batch_id}_results.jsonl")
            retry_responses = self.llm_client.retrieve_batch_results(retry_batch_id, retry_results_file, expected_count=len(task_data))

            # Update results for successful retries
            new_failures = []
            for idx in remaining_failures:
                llm_response = retry_responses[idx]

                if llm_response.error:
                    new_failures.append(idx)
                    continue

                # Evaluate answer
                example = task_data[idx]
                is_correct = self.simple_evaluate(llm_response.text, example)

                # Create prompt for logging
                system_prompt, user_prompt = self.create_prompt(example, task_info["name"])

                updated_results[idx] = {
                    "example_id": idx,
                    "prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "response": llm_response.text,
                    "ground_truth": example.get("ground_truth", {}),
                    "correct": is_correct,
                    "tokens": llm_response.total_tokens,
                    "retried": True,
                    "retry_attempt": retry_count
                }

            remaining_failures = new_failures

            if not remaining_failures:
                print(f"  ✅ All samples successfully retried!")
                break

        if remaining_failures:
            print(f"  ⚠️  {len(remaining_failures)} samples still failed after {retry_count} retries")
            print(f"  Failed indices: {remaining_failures}")

        return updated_results

    def monitor_batches(
        self,
        batch_ids: List[str],
        poll_interval: int = 60,
        batch_dir: str = "batch_files"
    ) -> Dict[str, Any]:
        """
        Monitor multiple batch jobs and process results when complete

        Args:
            batch_ids: List of batch job IDs
            poll_interval: Seconds between status checks
            batch_dir: Directory containing batch files

        Returns:
            Combined results from all batches
        """
        import time

        pending = set(batch_ids)
        completed_results = {}

        print(f"\n{'='*80}")
        print(f"Monitoring {len(batch_ids)} batch jobs")
        print(f"{'='*80}")

        # Track how long each batch has been in progress
        batch_start_times = {batch_id: time.time() for batch_id in pending}
        stuck_threshold = 600  # 10 minutes with no progress

        while pending:
            print(f"\nChecking status... ({len(pending)} pending)")

            for batch_id in list(pending):
                status_info = self.llm_client.check_batch_status(batch_id)
                elapsed = time.time() - batch_start_times[batch_id]
                elapsed_str = f"{int(elapsed/60)}m {int(elapsed%60)}s"

                # Format progress - show "processing" if stats not available
                req_counts = status_info['request_counts']
                if req_counts['total'] > 0:
                    progress_str = f"({req_counts['completed']}/{req_counts['total']})"
                else:
                    progress_str = "(processing)"
                print(f"  {batch_id[:12]}... - {status_info['status']} {progress_str} [{elapsed_str}]")

                if status_info['status'] == 'completed':
                    print(f"  ✅ Batch {batch_id[:12]}... completed! Processing results...")
                    result = self.process_batch_results(batch_id, batch_dir)
                    completed_results[batch_id] = result
                    pending.remove(batch_id)

                elif status_info['status'] == 'failed':
                    print(f"  ❌ Batch {batch_id[:12]}... failed!")
                    pending.remove(batch_id)

                elif status_info['status'] == 'expired':
                    print(f"  ⏰ Batch {batch_id[:12]}... expired!")
                    pending.remove(batch_id)

                elif status_info['status'] == 'cancelled':
                    print(f"  🚫 Batch {batch_id[:12]}... cancelled!")
                    pending.remove(batch_id)

                # Check for stuck batches (in progress for too long with no completed requests)
                # Only check if stats are available (total > 0)
                elif (status_info['status'] in ['in_progress', 'validating'] and
                      elapsed > stuck_threshold and
                      req_counts['total'] > 0 and
                      req_counts['completed'] == 0):
                    print(f"  ⚠️  Batch appears stuck (no progress in {int(stuck_threshold/60)} minutes)")
                    print(f"      You can cancel it manually: python scripts/cancel_batch.py {batch_id}")

            if pending:
                print(f"\nWaiting {poll_interval} seconds before next check...")
                time.sleep(poll_interval)

        return completed_results

    def save_results(self, results: Dict[str, Any], output_file: Optional[str] = None):
        """Save results to structured folder with multiple files

        Note: If _current_output_dir is set (from incremental saving), this will
        reuse that directory instead of creating a new one. Task results have
        already been saved incrementally during evaluation.
        """

        # Use existing output dir if available (from incremental saving)
        if self._current_output_dir is not None:
            output_dir = self._current_output_dir
            # Task results already saved incrementally
        else:
            # Create new output directory
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_safe = results["model"].replace("/", "_").replace(":", "_")
                folder_name = f"gpsbench_{model_safe}_{timestamp}"
            else:
                # Use provided name as folder name (strip .json if present)
                folder_name = output_file.replace('.json', '')

            output_dir = self.results_dir / folder_name
            output_dir.mkdir(exist_ok=True, parents=True)

        # Create task_results subfolder (may already exist from incremental saving)
        task_results_dir = output_dir / "task_results"
        task_results_dir.mkdir(exist_ok=True)

        # 1. Save summary with overall metrics
        summary = {
            "model": results["model"],
            "provider": results["provider"],
            "timestamp": results["timestamp"],
            "max_samples_per_task": results.get("max_samples_per_task"),
            "batch_mode": results.get("batch_mode", False),
            "overall": results.get("overall", {})
        }

        # Add track summaries (without full results)
        summary["tracks"] = {}
        for track_name, track_data in results.get("tracks", {}).items():
            summary["tracks"][track_name] = {
                "track_name": track_data.get("track_name", track_name),
                "track_display_name": track_data.get("track_display_name", track_name),
                "total_tasks": track_data.get("total_tasks", len(track_data.get("tasks", []))),
                "total_cases": track_data.get("total_cases", 0),
                "total_correct": track_data.get("total_correct", 0),
                "overall_accuracy": track_data.get("overall_accuracy", 0.0),
                "tasks_summary": []
            }

            # Add per-task summary
            for task in track_data.get("tasks", []):
                summary["tracks"][track_name]["tasks_summary"].append({
                    "task_name": task.get("task_name"),
                    "task_file": task.get("task_file"),
                    "total": task.get("total", 0),
                    "correct": task.get("correct", 0),
                    "accuracy": task.get("accuracy", 0.0)
                })

        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # 2. Save each track's results
        for track_name, track_data in results.get("tracks", {}).items():
            track_path = output_dir / f"track_{track_name}.json"

            # Create track file with metadata but without individual results
            track_info = {
                "track_name": track_data.get("track_name", track_name),
                "track_display_name": track_data.get("track_display_name", track_name),
                "total_tasks": track_data.get("total_tasks", len(track_data.get("tasks", []))),
                "total_cases": track_data.get("total_cases", 0),
                "total_correct": track_data.get("total_correct", 0),
                "overall_accuracy": track_data.get("overall_accuracy", 0.0),
                "tasks": []
            }

            # 3. Save individual task results
            for task in track_data.get("tasks", []):
                task_name = task.get("task_name", "unknown_task")
                task_file_name = task.get("task_file", "unknown.json").replace(".json", "")

                # Save full task results to separate file
                task_result_path = task_results_dir / f"{track_name}_{task_file_name}.json"
                with open(task_result_path, 'w') as f:
                    json.dump(task, f, indent=2)

                # Add reference to track file
                track_info["tasks"].append({
                    "task_name": task.get("task_name"),
                    "task_file": task.get("task_file"),
                    "total": task.get("total", 0),
                    "correct": task.get("correct", 0),
                    "accuracy": task.get("accuracy", 0.0),
                    "results_file": f"task_results/{track_name}_{task_file_name}.json"
                })

            with open(track_path, 'w') as f:
                json.dump(track_info, f, indent=2)

        # 4. Create README with summary
        readme_content = f"""# GPSBench Evaluation Results

**Model:** {results['model']}
**Provider:** {results['provider']}
**Timestamp:** {results['timestamp']}
**Batch Mode:** {results.get('batch_mode', False)}

## Overall Results

- **Total Cases:** {results.get('overall', {}).get('total_cases', 0)}
- **Total Correct:** {results.get('overall', {}).get('total_correct', 0)}
- **Overall Accuracy:** {results.get('overall', {}).get('overall_accuracy', 0.0):.2f}%

## Track Results

"""
        for track_name, track_data in results.get("tracks", {}).items():
            readme_content += f"""### {track_data.get('track_display_name', track_name)}

- **Accuracy:** {track_data.get('overall_accuracy', 0.0):.2f}%
- **Correct:** {track_data.get('total_correct', 0)}/{track_data.get('total_cases', 0)}
- **Tasks:** {track_data.get('total_tasks', len(track_data.get('tasks', [])))}

"""
            # Add task breakdown
            for task in track_data.get("tasks", []):
                status = "✅" if task.get("accuracy", 0) > 50 else "❌"
                readme_content += f"  {status} **{task.get('task_name')}:** {task.get('accuracy', 0):.1f}% ({task.get('correct', 0)}/{task.get('total', 0)})\n"
            readme_content += "\n"

        readme_content += """## File Structure

```
.
├── README.md              # This file
├── summary.json           # Overall metrics and configuration
├── track_pure_gps.json    # Pure GPS track summary
├── track_applied.json     # Applied track summary
└── task_results/          # Individual task results
    ├── pure_gps_task1_format_conversion.json
    ├── pure_gps_task3_coordinate_system_transformation.json
    └── ...
```

## Files Description

- **summary.json**: High-level overview with overall metrics
- **track_*.json**: Track-level summaries with task references
- **task_results/*.json**: Complete results for each task including:
  - All prompts and responses
  - Ground truth comparisons
  - Individual sample evaluations
  - Token usage and latency metrics
"""

        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

        print(f"\n✅ Results saved to: {output_dir}/")
        print(f"   - summary.json: Overall metrics")
        print(f"   - track_*.json: Track summaries")
        print(f"   - task_results/: Individual task results ({len(list(task_results_dir.glob('*.json')))} files)")
        print(f"   - README.md: Human-readable summary")

        return output_dir

    def update_existing_results(self, results: Dict[str, Any], existing_folder: str) -> Path:
        """Update an existing results folder with new task results (overwrite mode).

        This saves only the task results to an existing folder, overwriting any
        existing task files. Useful for re-running specific tasks.
        """
        # Find the existing folder
        output_dir = self.results_dir / existing_folder
        if not output_dir.exists():
            # Try to find it by pattern
            matches = list(self.results_dir.glob(f"*{existing_folder}*"))
            if matches:
                output_dir = matches[0]
            else:
                raise ValueError(f"Existing folder not found: {existing_folder}")

        task_results_dir = output_dir / "task_results"
        task_results_dir.mkdir(exist_ok=True)

        updated_files = []

        # Save only the new task results (overwriting existing)
        for track_name, track_data in results.get("tracks", {}).items():
            for task in track_data.get("tasks", []):
                task_file_name = task.get("task_file", "unknown.json").replace(".json", "")

                # Save full task results (overwrite existing)
                task_result_path = task_results_dir / f"{track_name}_{task_file_name}.json"
                with open(task_result_path, 'w') as f:
                    json.dump(task, f, indent=2)

                updated_files.append(task_result_path.name)

        print(f"\n✅ Updated existing results: {output_dir}/")
        print(f"   - Updated {len(updated_files)} task files:")
        for f in updated_files:
            print(f"     - task_results/{f}")

        return output_dir

    def find_existing_folder(self, model_name: str) -> Optional[str]:
        """Find the most recent results folder for a model."""
        model_safe = model_name.replace("/", "_").replace(":", "_")
        matches = list(self.results_dir.glob(f"gpsbench_{model_safe}_*"))

        if not matches:
            return None

        # Return most recent (by folder name timestamp)
        return sorted(matches, reverse=True)[0].name


def main():
    parser = argparse.ArgumentParser(
        description="Run GPSBench evaluation on any LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model configuration
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'openrouter', 'gemini', 'auto'],
        default='auto',
        help='API provider (default: auto-detect from model name)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-3.5-turbo',
        help='Model name (default: gpt-3.5-turbo)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (default: 0.0 for deterministic)'
    )
    parser.add_argument(
        '--reasoning-effort',
        type=str,
        choices=['low', 'medium', 'high'],
        default='low',
        help='Reasoning effort for reasoning models like gpt-5.1 (default: low for minimal reasoning)'
    )

    # Evaluation configuration
    parser.add_argument(
        '--track',
        type=str,
        choices=['pure_gps', 'applied', 'both'],
        default='both',
        help='Which track to evaluate (default: both)'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=None,
        help='Specific tasks to run (e.g., name_disambiguation boundary_analysis). Matches partial names.'
    )
    parser.add_argument(
        '--update-existing',
        type=str,
        default=None,
        metavar='FOLDER',
        help='Update existing results folder instead of creating new one (e.g., gpsbench_gpt-5.1_20251208)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per task (default: all)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )

    # Batch processing configuration
    parser.add_argument(
        '--use-batch',
        action='store_true',
        help='Use Batch API (50%% cost savings). Equivalent to --batch-mode auto. Works with OpenAI and Gemini providers.'
    )
    parser.add_argument(
        '--batch-mode',
        choices=['auto', 'submit', 'monitor', 'process', 'none'],
        default='none',
        help='Batch processing mode: auto (submit+monitor+process in one command), submit (create batches), monitor (check status), process (get results), none (normal mode)'
    )
    parser.add_argument(
        '--auto-retry',
        action='store_true',
        help='Automatically retry failed samples when using batch processing'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum number of retry attempts for failed samples (default: 2)'
    )

    # Concurrent processing configuration (alternative to Batch API, works with any provider)
    parser.add_argument(
        '--concurrent',
        action='store_true',
        help='Use concurrent processing with threading (works with any provider including OpenRouter)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Number of concurrent workers for --concurrent mode (default: 10)'
    )
    parser.add_argument(
        '--batch-ids',
        type=str,
        nargs='+',
        help='Batch IDs to monitor or process (space-separated)'
    )
    parser.add_argument(
        '--batch-dir',
        type=str,
        default='batch_files',
        help='Directory for batch files and metadata (default: batch_files)'
    )
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=60,
        help='Seconds between batch status checks (default: 60)'
    )

    # Output configuration
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory (default: data)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Results directory (default: results)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output filename (default: auto-generated)'
    )

    args = parser.parse_args()

    # Handle --use-batch flag (shortcut for --batch-mode auto)
    if args.use_batch:
        args.batch_mode = 'auto'

    # Initialize LLM client
    print(f"\n{'='*80}")
    print(f"GPSBench Evaluation")
    print(f"{'='*80}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    if args.batch_mode != 'none':
        print(f"Batch Mode: {args.batch_mode}")
    if args.concurrent:
        print(f"Concurrent Mode: {args.max_workers} workers")
    print(f"{'='*80}")

    llm_client = LLMClient(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=8192,  # Increased for complex tasks - consistent across OpenAI/Gemini
        timeout=120,  # Increased timeout for complex calculations (Area/Route tasks)
        reasoning_effort=args.reasoning_effort  # For reasoning models (gpt-5.x)
    )

    # Validate batch mode compatibility
    if args.batch_mode != 'none' and llm_client.provider not in ("openai", "gemini"):
        print(f"\n{'='*80}")
        print("❌ ERROR: Batch API Mode Not Supported")
        print(f"{'='*80}")
        print(f"The --use-batch and --batch-mode flags only work with OpenAI and Gemini providers.")
        print(f"Current provider: {llm_client.provider}")
        print(f"\n💡 Use --concurrent instead for {llm_client.provider}:")
        print(f"   python run_benchmark.py --provider {llm_client.provider} --model {args.model} --concurrent --max-workers 20")
        print(f"{'='*80}\n")
        return

    # Initialize runner
    runner = GPSBenchRunner(
        llm_client=llm_client,
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )

    # Handle batch modes
    if args.batch_mode == 'auto':
        # Automatic mode: submit → monitor → process in one command
        print(f"\n{'='*80}")
        print("BATCH MODE: Auto (Submit → Monitor → Process)")
        print(f"{'='*80}")

        if llm_client.provider not in ("openai", "gemini"):
            print("❌ Error: Batch API is only supported for OpenAI and Gemini providers")
            return

        # Step 1: Submit batch jobs
        print("\n[Step 1/3] Submitting batch jobs...")
        batch_ids = []

        if args.track == 'both':
            for track_name in runner.tracks.keys():
                tasks_to_run = runner.filter_tasks(runner.tracks[track_name]["tasks"], args.tasks)
                for task_info in tasks_to_run:
                    batch_id = runner.evaluate_task_batch(
                        track_name=track_name,
                        task_info=task_info,
                        max_samples=args.max_samples,
                        batch_dir=args.batch_dir
                    )
                    batch_ids.append(batch_id)
        else:
            tasks_to_run = runner.filter_tasks(runner.tracks[args.track]["tasks"], args.tasks)
            for task_info in tasks_to_run:
                batch_id = runner.evaluate_task_batch(
                    track_name=args.track,
                    task_info=task_info,
                    max_samples=args.max_samples,
                    batch_dir=args.batch_dir
                )
                batch_ids.append(batch_id)

        # Save batch IDs
        batch_ids_file = os.path.join(args.batch_dir, "batch_ids.txt")
        with open(batch_ids_file, 'w') as f:
            f.write('\n'.join(batch_ids))

        print(f"\n✅ Submitted {len(batch_ids)} batch jobs")
        print(f"Batch IDs saved to: {batch_ids_file}")

        # Step 2: Monitor until completion
        print(f"\n[Step 2/3] Monitoring batch jobs until completion...")
        print(f"This may take up to 24 hours. Polling every {args.poll_interval} seconds...")

        # monitor_batches now returns cached results (processed during monitoring)
        cached_results = runner.monitor_batches(
            batch_ids=batch_ids,
            poll_interval=args.poll_interval,
            batch_dir=args.batch_dir
        )

        # Step 3: Use cached results or re-process if retries needed
        print(f"\n[Step 3/3] Processing batch results...")

        task_results = []
        for batch_id in batch_ids:
            # Use cached result if available and no retry needed
            if batch_id in cached_results and not args.auto_retry:
                result = cached_results[batch_id]
                print(f"\n{'='*80}")
                print(f"Using cached result: {result.get('task_name', 'Unknown')}")
                print(f"{'='*80}")
            else:
                # Re-process (needed for retries or if result wasn't cached)
                result = runner.process_batch_results(
                    batch_id=batch_id,
                    batch_dir=args.batch_dir,
                    auto_retry=args.auto_retry,
                    max_retries=args.max_retries
                )
            task_results.append(result)

        # Organize results by track
        tracks = {}
        for result in task_results:
            # Load metadata to get track name (sanitize batch_id for filename)
            safe_batch_id = result['batch_id'].replace('/', '_')
            metadata_file = os.path.join(args.batch_dir, f"{safe_batch_id}_metadata.json")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            track_name = metadata['track_name']

            if track_name not in tracks:
                tracks[track_name] = {
                    "track_name": track_name,
                    "track_display_name": runner.tracks[track_name]["name"],
                    "total_tasks": 0,
                    "tasks": [],
                    "total_correct": 0,
                    "total_cases": 0
                }

            tracks[track_name]["tasks"].append(result)
            tracks[track_name]["total_tasks"] += 1
            tracks[track_name]["total_correct"] += result.get("correct", 0)
            tracks[track_name]["total_cases"] += result.get("total", 0)

        # Calculate accuracies
        for track_name, track_data in tracks.items():
            if track_data["total_cases"] > 0:
                track_data["overall_accuracy"] = (track_data["total_correct"] / track_data["total_cases"] * 100)
            else:
                track_data["overall_accuracy"] = 0.0

        # Create final results structure
        results = {
            "model": args.model,
            "provider": llm_client.provider,
            "timestamp": datetime.now().isoformat(),
            "max_samples_per_task": args.max_samples,
            "batch_mode": True,
            "tracks": tracks
        }

        # Calculate overall metrics
        total_correct = sum(t["total_correct"] for t in tracks.values())
        total_cases = sum(t["total_cases"] for t in tracks.values())
        overall_accuracy = (total_correct / total_cases * 100) if total_cases > 0 else 0.0

        results["overall"] = {
            "total_cases": total_cases,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy
        }

    elif args.batch_mode == 'submit':
        # Submit batch jobs
        print(f"\n{'='*80}")
        print("BATCH MODE: Submitting batch jobs")
        print(f"{'='*80}")

        if llm_client.provider not in ("openai", "gemini"):
            print("❌ Error: Batch API is only supported for OpenAI and Gemini providers")
            return

        batch_ids = []

        if args.track == 'both':
            for track_name in runner.tracks.keys():
                tasks_to_run = runner.filter_tasks(runner.tracks[track_name]["tasks"], args.tasks)
                for task_info in tasks_to_run:
                    batch_id = runner.evaluate_task_batch(
                        track_name=track_name,
                        task_info=task_info,
                        max_samples=args.max_samples,
                        batch_dir=args.batch_dir
                    )
                    batch_ids.append(batch_id)
        else:
            tasks_to_run = runner.filter_tasks(runner.tracks[args.track]["tasks"], args.tasks)
            for task_info in tasks_to_run:
                batch_id = runner.evaluate_task_batch(
                    track_name=args.track,
                    task_info=task_info,
                    max_samples=args.max_samples,
                    batch_dir=args.batch_dir
                )
                batch_ids.append(batch_id)

        # Save batch IDs for later reference
        batch_ids_file = os.path.join(args.batch_dir, "batch_ids.txt")
        with open(batch_ids_file, 'w') as f:
            f.write('\n'.join(batch_ids))

        print(f"\n{'='*80}")
        print(f"✅ Submitted {len(batch_ids)} batch jobs")
        print(f"Batch IDs saved to: {batch_ids_file}")
        print(f"\nTo monitor progress, run:")
        print(f"  python run_benchmark.py --batch-mode monitor --batch-ids {' '.join(batch_ids[:2])} ...")
        print(f"{'='*80}")
        return

    elif args.batch_mode == 'monitor':
        # Monitor batch jobs
        if not args.batch_ids:
            print("❌ Error: --batch-ids required for monitor mode")
            return

        print(f"\n{'='*80}")
        print("BATCH MODE: Monitoring batch jobs")
        print(f"{'='*80}")

        runner.monitor_batches(
            batch_ids=args.batch_ids,
            poll_interval=args.poll_interval,
            batch_dir=args.batch_dir
        )

        print(f"\n{'='*80}")
        print("All batches completed!")
        print(f"\nTo process results, run:")
        print(f"  python run_benchmark.py --batch-mode process --batch-ids {' '.join(args.batch_ids[:2])} ...")
        print(f"{'='*80}")
        return

    elif args.batch_mode == 'process':
        # Process completed batch results
        if not args.batch_ids:
            print("❌ Error: --batch-ids required for process mode")
            return

        print(f"\n{'='*80}")
        print("BATCH MODE: Processing batch results")
        print(f"{'='*80}")

        # Process each batch and collect results
        task_results = []
        for batch_id in args.batch_ids:
            result = runner.process_batch_results(
                batch_id=batch_id,
                batch_dir=args.batch_dir,
                auto_retry=args.auto_retry,
                max_retries=args.max_retries
            )
            task_results.append(result)

        # Organize results by track
        tracks = {}
        for result in task_results:
            # Load metadata to get track name (sanitize batch_id for filename)
            safe_batch_id = result['batch_id'].replace('/', '_')
            metadata_file = os.path.join(args.batch_dir, f"{safe_batch_id}_metadata.json")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            track_name = metadata['track_name']

            if track_name not in tracks:
                tracks[track_name] = {
                    "track_name": track_name,
                    "tasks": [],
                    "total_correct": 0,
                    "total_cases": 0
                }

            tracks[track_name]["tasks"].append(result)
            tracks[track_name]["total_correct"] += result.get("correct", 0)
            tracks[track_name]["total_cases"] += result.get("total", 0)

        # Calculate accuracies
        for track_name, track_data in tracks.items():
            if track_data["total_cases"] > 0:
                track_data["overall_accuracy"] = (track_data["total_correct"] / track_data["total_cases"] * 100)
            else:
                track_data["overall_accuracy"] = 0.0

        # Create final results structure
        results = {
            "model": args.model,
            "provider": llm_client.provider,
            "timestamp": datetime.now().isoformat(),
            "max_samples_per_task": args.max_samples,
            "batch_mode": True,
            "tracks": tracks
        }

        # Calculate overall metrics
        total_correct = sum(t["total_correct"] for t in tracks.values())
        total_cases = sum(t["total_cases"] for t in tracks.values())
        overall_accuracy = (total_correct / total_cases * 100) if total_cases > 0 else 0.0

        results["overall"] = {
            "total_cases": total_cases,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy
        }

    else:
        # Normal evaluation mode (not batch)
        # Initialize output directory upfront for incremental saving
        output_dir = runner.init_output_dir(args.output)

        if args.track == 'both':
            results = runner.evaluate_all(
                max_samples=args.max_samples,
                delay=args.delay,
                use_concurrent=args.concurrent,
                max_workers=args.max_workers,
                task_filter=args.tasks,
                output_dir=output_dir
            )
        else:
            track_results = runner.evaluate_track(
                track_name=args.track,
                max_samples=args.max_samples,
                delay=args.delay,
                use_concurrent=args.concurrent,
                max_workers=args.max_workers,
                task_filter=args.tasks,
                output_dir=output_dir
            )
            if track_results:
                results = {
                    "model": args.model,
                    "provider": llm_client.provider,
                    "timestamp": datetime.now().isoformat(),
                    "max_samples_per_task": args.max_samples,
                    "tracks": {args.track: track_results}
                }
            else:
                results = None

    if results is None:
        print("\nNo results to save (no tasks matched filter)")
        return

    # Save results
    if args.update_existing:
        # Update existing folder (for re-running specific tasks)
        output_path = runner.update_existing_results(results, args.update_existing)
    else:
        output_path = runner.save_results(results, args.output)

    # Print summary
    print(f"\n{'='*80}")
    print("GPSBench Evaluation Complete")
    print(f"{'='*80}")

    if "overall" in results:
        metrics = results["overall"]
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        print(f"Total Correct: {metrics['total_correct']}/{metrics['total_cases']}")
        print()

    for track_name, track_results in results.get("tracks", {}).items():
        track_display_name = track_results.get('track_display_name', track_results.get('track_name', track_name))
        print(f"{track_display_name}:")
        print(f"  Accuracy: {track_results['overall_accuracy']:.2f}%")
        print(f"  Correct: {track_results['total_correct']}/{track_results['total_cases']}")
        print()

        for task in track_results.get("tasks", []):
            status = "✅" if task.get("accuracy", 0) > 50 else "❌"
            print(f"    {status} {task['task_name']}: {task.get('accuracy', 0):.1f}%")
        print()

    print(f"{'='*80}")
    print(f"Results folder: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
