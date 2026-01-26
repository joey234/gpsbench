#!/usr/bin/env python3
"""
GPSBench Evaluation for Finetuned Model
========================================

Run the complete GPSBench benchmark using a Tinker finetuned model.

Usage:
    # Evaluate on all tasks
    python run_gpsbench_finetuned.py --checkpoint final

    # Evaluate on specific track
    python run_gpsbench_finetuned.py --checkpoint final --track pure_gps

    # Quick test
    python run_gpsbench_finetuned.py --checkpoint final --max-samples 5

    # Compare with checkpoint 000800
    python run_gpsbench_finetuned.py --checkpoint 000800 --track applied
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import lru_cache
from tqdm import tqdm

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import tinker
from tinker import types
from transformers import AutoTokenizer

# Add GPSBench root to path
GPSBENCH_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(GPSBENCH_ROOT))


@lru_cache(maxsize=32)
def load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    template_path = GPSBENCH_ROOT / "prompts" / f"{template_name}.txt"
    if template_path.exists():
        return template_path.read_text()
    return None


class TinkerLLMClient:
    """LLM client using Tinker SamplingClient for finetuned models."""

    def __init__(
        self,
        sampler_path: str,
        base_model: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        batch_size: int = 8  # Number of concurrent requests
    ):
        self.sampler_path = sampler_path
        self.base_model = base_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size

        print(f"Loading tokenizer: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        print(f"Creating sampling client with checkpoint: {sampler_path}")
        client = tinker.ServiceClient()
        self.sampling_client = client.create_sampling_client(
            model_path=sampler_path,
            base_model=base_model
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using the finetuned model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Tokenize
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(prompt_text)

        # Create input
        model_input = types.ModelInput(chunks=[
            types.EncodedTextChunk(tokens=tokens, type='encoded_text')
        ])

        # Sample
        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        response = self.sampling_client.sample(
            model_input, num_samples=1, sampling_params=sampling_params
        ).result()

        # Decode
        output_tokens = response.sequences[0].tokens
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return output_text

    def generate_batch(self, prompts: List[tuple]) -> List[str]:
        """Generate responses for multiple prompts in parallel using futures.

        Args:
            prompts: List of (system_prompt, user_prompt) tuples

        Returns:
            List of response strings
        """
        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        # Submit all requests and collect futures
        futures = []
        for system_prompt, user_prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokens = self.tokenizer.encode(prompt_text)
            model_input = types.ModelInput(chunks=[
                types.EncodedTextChunk(tokens=tokens, type='encoded_text')
            ])

            # Submit request (returns Future)
            future = self.sampling_client.sample(
                model_input, num_samples=1, sampling_params=sampling_params
            )
            futures.append(future)

        # Collect all results
        responses = []
        for future in futures:
            try:
                response = future.result()
                output_tokens = response.sequences[0].tokens
                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                responses.append(output_text)
            except Exception as e:
                responses.append(f"Error: {e}")

        return responses


class GPSBenchFinetunedRunner:
    """Runner for GPSBench evaluation with finetuned model."""

    def __init__(
        self,
        llm_client: TinkerLLMClient,
        data_dir: str = None,
        results_dir: str = None,
        batch_size: int = 16
    ):
        self.llm_client = llm_client
        self.data_dir = Path(data_dir) if data_dir else GPSBENCH_ROOT / "data"
        self.results_dir = Path(results_dir) if results_dir else Path("./outputs/gpsbench_eval")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size

        # Track structure
        self.tracks = {
            "pure_gps": {
                "name": "Pure GPS Track",
                "tasks": [
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

    def add_format_instruction(self, user_prompt: str, ground_truth: Dict[str, Any]) -> str:
        """Add format instruction to prompt based on ground truth structure."""
        if "FINAL ANSWER:" in user_prompt and "[" in user_prompt.split("FINAL ANSWER:")[-1][:50]:
            return user_prompt

        format_instruction = None

        if "initial_bearing_deg" in ground_truth or "final_bearing_deg" in ground_truth:
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
            if isinstance(answer_val, bool):
                format_instruction = "Answer with Yes or No"
            elif isinstance(answer_val, (int, float)):
                if "tolerance" in ground_truth:
                    format_instruction = "Provide the coordinate value (e.g., 106.6)"
                else:
                    format_instruction = "Provide the answer as a number"
        elif "name" in ground_truth or "location_name" in ground_truth:
            if "granularity_level" in ground_truth or "country_code" in ground_truth:
                format_instruction = "Provide your answer in the format: City, Region/State, Country (e.g., 'Paris, Île-de-France, France' or 'Tokyo, Tokyo, Japan')"
            else:
                format_instruction = "Provide the location name"
        elif "terrain_type" in ground_truth:
            format_instruction = "Provide the terrain type or corresponding letter"

        if format_instruction:
            if not user_prompt.rstrip().endswith("FINAL ANSWER: [your answer]") and "FINAL ANSWER:" not in user_prompt.split('\n')[-3:]:
                user_prompt += f"\n\n{format_instruction}.\n\nBe brief. Show key steps only, then provide FINAL ANSWER: [your answer]"

        return user_prompt

    def build_task_prompt(self, task_data: Dict[str, Any], task_name: str) -> str:
        """Build a prompt for tasks without a 'question' field based on task type."""
        task = task_data.get('task', task_name)
        gt = task_data.get('ground_truth', {})

        # Format conversion task
        if task == 'format_conversion' or 'format_conversion' in task_name:
            template = load_prompt_template('format_conversion')
            if template:
                return template.format(
                    source_format=task_data.get('source_format', 'source'),
                    target_format=task_data.get('target_format', 'target'),
                    source_string=task_data.get('source_string', '')
                )
            return (
                f"Convert the coordinate from {task_data.get('source_format', 'source')} to {task_data.get('target_format', 'target')}.\n\n"
                f"Source coordinate ({task_data.get('source_format', 'source')}): {task_data.get('source_string', '')}\n\n"
                f"Provide the converted coordinate in {task_data.get('target_format', 'target')} format.\n\n"
                f"FINAL ANSWER: [converted coordinate]"
            )

        # Distance calculation task
        elif task == 'distance_calculation' or 'distance_calculation' in task_name:
            point_a = task_data.get('point_a', {})
            point_b = task_data.get('point_b', {})
            template = load_prompt_template('distance_calculation')
            if template:
                return template.format(
                    point_a_name=point_a.get('name', 'Point A'),
                    point_a_lat=point_a.get('lat', 0),
                    point_a_lon=point_a.get('lon', 0),
                    point_b_name=point_b.get('name', 'Point B'),
                    point_b_lat=point_b.get('lat', 0),
                    point_b_lon=point_b.get('lon', 0)
                )
            return (
                f"Calculate the distance between these two points:\n\n"
                f"Point A: {point_a.get('name', 'Unknown')}\n"
                f"Coordinates: {point_a.get('lat', 0)}, {point_a.get('lon', 0)}\n\n"
                f"Point B: {point_b.get('name', 'Unknown')}\n"
                f"Coordinates: {point_b.get('lat', 0)}, {point_b.get('lon', 0)}\n\n"
                f"Provide the distance in kilometers.\n\n"
                f"FINAL ANSWER: [distance] km"
            )

        # Bearing computation task
        elif task == 'bearing_computation' or 'bearing_computation' in task_name:
            point_a = task_data.get('point_a', {})
            point_b = task_data.get('point_b', {})
            template = load_prompt_template('bearing_computation')
            if template:
                return template.format(
                    point_a_name=point_a.get('name', 'Point A'),
                    point_a_lat=point_a.get('lat', 0),
                    point_a_lon=point_a.get('lon', 0),
                    point_b_name=point_b.get('name', 'Point B'),
                    point_b_lat=point_b.get('lat', 0),
                    point_b_lon=point_b.get('lon', 0)
                )
            return (
                f"Calculate the bearing (initial direction) from Point A to Point B:\n\n"
                f"Point A: {point_a.get('name', 'Unknown')}\n"
                f"Coordinates: {point_a.get('lat', 0)}, {point_a.get('lon', 0)}\n\n"
                f"Point B: {point_b.get('name', 'Unknown')}\n"
                f"Coordinates: {point_b.get('lat', 0)}, {point_b.get('lon', 0)}\n\n"
                f"Provide the bearing in degrees (0-360°, where 0° is North, 90° is East).\n\n"
                f"FINAL ANSWER: [bearing]°"
            )

        # Interpolation task
        elif task == 'interpolation' or 'interpolation' in task_name:
            point_a = task_data.get('point_a', {})
            point_b = task_data.get('point_b', {})
            fraction = task_data.get('fraction', 0.5)
            template = load_prompt_template('interpolation')
            if template:
                return template.format(
                    fraction=fraction,
                    point_a_name=point_a.get('name', 'Point A'),
                    point_a_lat=point_a.get('lat', 0),
                    point_a_lon=point_a.get('lon', 0),
                    point_b_name=point_b.get('name', 'Point B'),
                    point_b_lat=point_b.get('lat', 0),
                    point_b_lon=point_b.get('lon', 0)
                )
            return (
                f"Find the point that is {fraction} of the way along the great circle path from Point A to Point B.\n\n"
                f"Point A: {point_a.get('name', 'Unknown')}\n"
                f"Coordinates: {point_a.get('lat', 0)}, {point_a.get('lon', 0)}\n\n"
                f"Point B: {point_b.get('name', 'Unknown')}\n"
                f"Coordinates: {point_b.get('lat', 0)}, {point_b.get('lon', 0)}\n\n"
                f"Calculate the interpolated point at fraction {fraction} (where 0 = Point A, 1 = Point B).\n\n"
                f"FINAL ANSWER: [latitude], [longitude]"
            )

        # Area and perimeter task
        elif task == 'area_perimeter' or 'area_perimeter' in task_name:
            vertices = task_data.get('vertices', [])
            if not vertices and 'polygon' in task_data:
                vertices = task_data['polygon'].get('vertices', [])
            vertices_str = "\n".join([
                f"  Vertex {i+1}: {v.get('name', 'Unknown')} ({v.get('lat', 0)}, {v.get('lon', 0)})"
                for i, v in enumerate(vertices)
            ])
            template = load_prompt_template('area_perimeter')
            if template:
                return template.format(vertices=vertices_str)
            return (
                f"Calculate the area of the polygon formed by these coordinates:\n\n"
                f"{vertices_str}\n\n"
                f"Use spherical geometry (Earth radius ≈ 6371 km).\n\n"
                f"FINAL ANSWER: [area] km²"
            )

        # Bounding box task
        elif task == 'bounding_box' or 'bounding_box' in task_name:
            points = task_data.get('points', [])
            points_str = "\n".join([
                f"  {i+1}. {p.get('name', 'Unknown')} ({p.get('lat', 0)}, {p.get('lon', 0)})"
                for i, p in enumerate(points)
            ])
            template = load_prompt_template('bounding_box')
            if template:
                return template.format(num_points=len(points), points=points_str)
            return (
                f"Calculate the bounding box (minimum rectangle) containing these {len(points)} points:\n\n"
                f"{points_str}\n\n"
                f"Provide the bounding box as: min_lat, max_lat, min_lon, max_lon\n\n"
                f"FINAL ANSWER: [min_lat], [max_lat], [min_lon], [max_lon]"
            )

        # Coordinate system transformation
        elif task == 'coordinate_system_transformation' or 'coordinate_system_transformation' in task_name:
            src_coords = task_data.get('source_coords', task_data.get('coordinate', {}))
            src_crs = task_data.get('source_crs', 'WGS84')
            tgt_crs = task_data.get('target_crs', 'UTM')
            location = task_data.get('location_name', 'Unknown')

            # Handle different coord formats
            lat = src_coords.get('lat', src_coords.get('latitude', 0))
            lon = src_coords.get('lon', src_coords.get('longitude', 0))

            if 'web_mercator' in tgt_crs.lower() or tgt_crs == 'EPSG:3857':
                return (
                    f"Convert coordinates from {src_crs} to Web Mercator (EPSG:3857).\n\n"
                    f"Location: {location}\n"
                    f"Source coordinates: {lat}, {lon}\n\n"
                    f"Provide the Web Mercator coordinates (X, Y in meters).\n\n"
                    f"FINAL ANSWER: X: [x] m, Y: [y] m"
                )
            else:
                return (
                    f"Convert coordinates from {src_crs} to {tgt_crs}.\n\n"
                    f"Location: {location}\n"
                    f"Source coordinates: {lat}, {lon}\n\n"
                    f"Provide the converted coordinates.\n\n"
                    f"FINAL ANSWER: Easting: [easting] m, Northing: [northing] m, Zone: [zone][hemisphere]"
                )

        # Place association task
        elif task == 'place_association' or 'place_association' in task_name:
            coord_str = task_data.get('coordinate_string', '')
            if not coord_str and 'coordinate' in task_data:
                coord = task_data['coordinate']
                coord_str = f"{coord.get('lat', coord.get('latitude', 0))}, {coord.get('lon', coord.get('longitude', 0))}"
            template = load_prompt_template('place_association')
            if template:
                return template.format(coordinate_string=coord_str)
            return (
                f"What location is at coordinates {coord_str}?\n\n"
                f"Provide your answer in the format: City, Region/State, Country "
                f"(e.g., 'Paris, Île-de-France, France' or 'Tokyo, Tokyo, Japan').\n\n"
                f"FINAL ANSWER: [location name]"
            )

        # Route geometry task
        elif task == 'route_geometry' or 'route_geometry' in task_name:
            subtask = task_data.get('subtask', 'simplification')
            points = task_data.get('points', [])
            tolerance = task_data.get('tolerance_km', 1.0)

            if subtask == 'polyline_simplification' or 'simplification' in str(task_data):
                points_str = "\n".join([
                    f"  {i}: ({p.get('lat', 0):.4f}, {p.get('lon', 0):.4f})"
                    for i, p in enumerate(points)
                ])
                options = task_data.get('options', [])
                options_str = "\n".join([f"  {chr(65+i)}) {opt}" for i, opt in enumerate(options)])

                return (
                    f"Apply Douglas-Peucker simplification with tolerance {tolerance} km. Which points should be kept?\n\n"
                    f"Route points:\n{points_str}\n\n"
                    f"Options:\n{options_str}\n\n"
                    f"FINAL ANSWER: [letter]"
                )
            return f"Route geometry task.\n\nFINAL ANSWER: [letter]"

        # Relative position (Pure GPS)
        elif 'relative_position' in task_name and 'pure_gps' in task_name:
            point_a = task_data.get('point_a', {})
            point_b = task_data.get('point_b', {})
            direction = task_data.get('direction', 'north')

            return (
                f"Is Point A ({point_a.get('lat', 0)}, {point_a.get('lon', 0)}) {direction} of "
                f"Point B ({point_b.get('lat', 0)}, {point_b.get('lon', 0)})?\n\n"
                f"Answer with Yes or No.\n\n"
                f"Be brief. Show key steps only, then provide FINAL ANSWER: [your answer]"
            )

        # Missing data task
        elif task == 'missing_data_handling' or 'missing_data' in task_name:
            location = task_data.get('location_name', task_data.get('name', 'Unknown'))
            lat = task_data.get('latitude')
            lon = task_data.get('longitude')

            if lat is not None:
                return (
                    f"Given latitude {lat}° and location name '{location}', what is the longitude?\n\n"
                    f"FINAL ANSWER: [longitude]°"
                )
            elif lon is not None:
                return (
                    f"Given longitude {lon}° and location name '{location}', what is the latitude?\n\n"
                    f"FINAL ANSWER: [latitude]°"
                )

        # Return None if no specific handler
        return None

    def create_prompt(self, task_data: Dict[str, Any], task_name: str) -> tuple:
        """Create system and user prompts for a task."""
        system_prompt = load_prompt_template("system_prompt") or "You are an expert in GPS coordinates and spatial reasoning."

        # First, try to build task-specific prompt if no question field
        if "question" not in task_data:
            built_prompt = self.build_task_prompt(task_data, task_name)
            if built_prompt:
                return system_prompt, built_prompt

        if "question" in task_data:
            user_prompt = task_data["question"]

            # Handle coordinate references
            if ("this coordinate" in user_prompt.lower() or "these coordinates" in user_prompt.lower()):
                if "coordinate_string" in task_data:
                    coord_str = task_data["coordinate_string"]
                    user_prompt = f"Coordinate: {coord_str}\n\n{user_prompt}"
                elif "coordinate" in task_data and isinstance(task_data["coordinate"], dict):
                    coord = task_data["coordinate"]
                    coord_str = f"{coord.get('lat')}, {coord.get('lon')}"
                    user_prompt = f"Coordinate: {coord_str}\n\n{user_prompt}"

            # Handle point_a/point_b
            if "point_a" in task_data and "point_b" in task_data:
                point_a = task_data["point_a"]
                point_b = task_data["point_b"]

                coords_in_question = (
                    str(point_a.get('lat', '')) in user_prompt or
                    str(point_b.get('lat', '')) in user_prompt
                )

                is_pure_gps = (
                    task_data.get('metadata', {}).get('track') == 'pure_gps' or
                    task_data.get('task') in ['distance_calculation', 'bearing_computation', 'interpolation', 'relative_position']
                )

                is_applied = task_data.get('metadata', {}).get('track') == 'applied'
                if is_applied:
                    is_pure_gps = False
                if task_data.get('task') == 'relative_position' and 'applied' in task_name.lower():
                    is_pure_gps = False

                if not coords_in_question and is_pure_gps:
                    name_a = point_a.get('name', 'Point A')
                    name_b = point_b.get('name', 'Point B')
                    coord_a_str = f"Point A ({point_a['lat']}, {point_a['lon']})"
                    coord_b_str = f"Point B ({point_b['lat']}, {point_b['lon']})"

                    if name_a in user_prompt and name_b in user_prompt:
                        user_prompt = user_prompt.replace(name_a, coord_a_str, 1)
                        user_prompt = user_prompt.replace(name_b, coord_b_str, 1)

            # Handle vertices for polygon tasks
            vertices = None
            if "polygon" in task_data and "this polygon" in user_prompt.lower():
                polygon = task_data["polygon"]
                vertices = polygon.get("vertices", [])
            elif "vertices" in task_data and "this polygon" in user_prompt.lower():
                vertices = task_data["vertices"]

            if vertices:
                if not any(f"{v['lat']}" in user_prompt for v in vertices[:2]):
                    vertex_strs = []
                    for i, v in enumerate(vertices, 1):
                        name = v.get('name', f'Vertex {i}')
                        vertex_strs.append(f"  Vertex {i}: {name} ({v['lat']}, {v['lon']})")
                    user_prompt += f"\n\nPolygon vertices:\n" + '\n'.join(vertex_strs)

            # Handle candidates
            if "candidates" in task_data:
                candidates = task_data["candidates"]
                if candidates and isinstance(candidates[0], dict):
                    candidate_strs = []
                    for i, cand in enumerate(candidates):
                        if "name" in cand and "country" in cand:
                            pop = cand.get('population', 0)
                            if pop:
                                candidate_strs.append(f"{chr(65+i)}) {cand['name']}, {cand['country']} (Pop: {pop:,})")
                            else:
                                candidate_strs.append(f"{chr(65+i)}) {cand['name']}, {cand['country']}")
                        else:
                            candidate_strs.append(f"{chr(65+i)}) {str(cand)}")
                    user_prompt += "\n\nCandidates:\n" + "\n".join(candidate_strs)
                    user_prompt += "\n\nProvide your answer as FINAL ANSWER: [letter]"
                elif candidates and isinstance(candidates[0], str):
                    user_prompt += "\n\nOptions:\n" + "\n".join(candidates)

            # Add format instruction
            ground_truth = task_data.get("ground_truth", {})
            user_prompt = self.add_format_instruction(user_prompt, ground_truth)

            return system_prompt, user_prompt

        return system_prompt, "No question found in task data"

    def evaluate_response(self, response: str, ground_truth: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Evaluate a model response against ground truth."""
        import re
        from difflib import SequenceMatcher

        # Extract final answer
        prediction = response.strip()
        for pattern in [r'FINAL ANSWER:\s*(.+?)(?:\n|$)', r'Final Answer:\s*(.+?)(?:\n|$)']:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                prediction = match.group(1).strip()
                break

        result = {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": False,
            "score": 0.0,
            "details": {}
        }

        # Format conversion - check latitude/longitude in ground truth
        if "target_string" in ground_truth or ("latitude" in ground_truth and "longitude" in ground_truth and "location_name" not in ground_truth):
            try:
                # Extract numbers from prediction
                pred_numbers = re.findall(r'-?[\d.]+', prediction)
                gt_lat = ground_truth.get("latitude", 0)
                gt_lon = ground_truth.get("longitude", 0)

                if len(pred_numbers) >= 2:
                    pred_lat = float(pred_numbers[0])
                    pred_lon = float(pred_numbers[1])
                    lat_error = abs(pred_lat - gt_lat)
                    lon_error = abs(pred_lon - gt_lon)
                    tolerance = 0.01  # 0.01 degree tolerance

                    result["correct"] = lat_error <= tolerance and lon_error <= tolerance
                    result["score"] = 1.0 if result["correct"] else max(0, 1 - (lat_error + lon_error) / 2)
                    result["details"] = {
                        "pred_lat": pred_lat, "pred_lon": pred_lon,
                        "gt_lat": gt_lat, "gt_lon": gt_lon,
                        "lat_error": lat_error, "lon_error": lon_error
                    }
            except Exception as e:
                result["details"]["error"] = str(e)
            return result

        # Interpolated point
        if "interpolated_point" in ground_truth:
            try:
                interp = ground_truth["interpolated_point"]
                gt_lat = interp.get("lat", 0)
                gt_lon = interp.get("lon", 0)
                tolerance = ground_truth.get("tolerance_deg", 0.01)

                pred_numbers = re.findall(r'-?[\d.]+', prediction)
                if len(pred_numbers) >= 2:
                    pred_lat = float(pred_numbers[0])
                    pred_lon = float(pred_numbers[1])
                    lat_error = abs(pred_lat - gt_lat)
                    lon_error = abs(pred_lon - gt_lon)

                    result["correct"] = lat_error <= tolerance and lon_error <= tolerance
                    result["score"] = 1.0 if result["correct"] else max(0, 1 - (lat_error + lon_error) / tolerance)
                    result["details"] = {"pred": (pred_lat, pred_lon), "gt": (gt_lat, gt_lon)}
            except:
                pass
            return result

        # Bounding box
        if all(k in ground_truth for k in ["min_lat", "max_lat", "min_lon", "max_lon"]):
            try:
                pred_numbers = re.findall(r'-?[\d.]+', prediction)
                if len(pred_numbers) >= 4:
                    tolerance = 0.1
                    pred_vals = [float(x) for x in pred_numbers[:4]]
                    gt_vals = [ground_truth["min_lat"], ground_truth["max_lat"],
                              ground_truth["min_lon"], ground_truth["max_lon"]]

                    errors = [abs(p - g) for p, g in zip(pred_vals, gt_vals)]
                    result["correct"] = all(e <= tolerance for e in errors)
                    result["score"] = 1.0 if result["correct"] else max(0, 1 - sum(errors) / 4)
                    result["details"] = {"pred": pred_vals, "gt": gt_vals, "errors": errors}
            except:
                pass
            return result

        # Web Mercator / coordinate transformation (x, y)
        if "x" in ground_truth and "y" in ground_truth:
            try:
                pred_numbers = re.findall(r'-?[\d.]+', prediction)
                if len(pred_numbers) >= 2:
                    pred_x = float(pred_numbers[0])
                    pred_y = float(pred_numbers[1])
                    gt_x = ground_truth["x"]
                    gt_y = ground_truth["y"]

                    # Use relative tolerance for large coordinates
                    rel_tol = 0.01
                    x_ok = abs(pred_x - gt_x) <= abs(gt_x) * rel_tol
                    y_ok = abs(pred_y - gt_y) <= abs(gt_y) * rel_tol

                    result["correct"] = x_ok and y_ok
                    result["score"] = 1.0 if result["correct"] else 0.5 if (x_ok or y_ok) else 0.0
                    result["details"] = {"pred": (pred_x, pred_y), "gt": (gt_x, gt_y)}
            except:
                pass
            return result

        # UTM transformation (easting, northing)
        if "easting" in ground_truth and "northing" in ground_truth:
            try:
                pred_numbers = re.findall(r'-?[\d.]+', prediction)
                if len(pred_numbers) >= 2:
                    pred_e = float(pred_numbers[0])
                    pred_n = float(pred_numbers[1])
                    gt_e = ground_truth["easting"]
                    gt_n = ground_truth["northing"]

                    rel_tol = 0.01
                    e_ok = abs(pred_e - gt_e) <= abs(gt_e) * rel_tol
                    n_ok = abs(pred_n - gt_n) <= abs(gt_n) * rel_tol

                    result["correct"] = e_ok and n_ok
                    result["score"] = 1.0 if result["correct"] else 0.5 if (e_ok or n_ok) else 0.0
                    result["details"] = {"pred": (pred_e, pred_n), "gt": (gt_e, gt_n)}
            except:
                pass
            return result

        # Area calculation
        if "area_km2" in ground_truth:
            try:
                pred_numbers = re.findall(r'[\d.]+', prediction)
                if pred_numbers:
                    pred_val = float(pred_numbers[0])
                    gt_val = ground_truth["area_km2"]
                    tolerance = ground_truth.get("tolerance_percent", 5.0) / 100 * gt_val

                    error = abs(pred_val - gt_val)
                    result["correct"] = error <= tolerance
                    result["score"] = max(0, 1 - error / gt_val) if gt_val else 0
                    result["details"] = {"predicted": pred_val, "expected": gt_val, "error": error}
            except:
                pass
            return result

        # Task-specific evaluation
        if "location_name" in ground_truth or ("name" in ground_truth and "country_code" in ground_truth):
            # Place association / name disambiguation
            gt_name = ground_truth.get("location_name", ground_truth.get("name", ""))
            pred_lower = prediction.lower().strip()
            gt_lower = gt_name.lower().strip()

            if pred_lower == gt_lower:
                result["correct"] = True
                result["score"] = 1.0
            elif gt_lower in pred_lower or pred_lower in gt_lower:
                result["score"] = 0.9
                result["correct"] = True
            else:
                ratio = SequenceMatcher(None, pred_lower, gt_lower).ratio()
                result["score"] = ratio
                result["correct"] = ratio >= 0.8

        elif "distance_km" in ground_truth:
            # Distance calculation
            try:
                numbers = re.findall(r'[\d.]+', prediction)
                if numbers:
                    pred_value = float(numbers[0])
                    gt_value = ground_truth["distance_km"]
                    tolerance = ground_truth.get("tolerance_km", gt_value * 0.05)
                    error = abs(pred_value - gt_value)
                    result["correct"] = error <= tolerance
                    result["score"] = max(0, 1 - error / gt_value) if gt_value else 0
                    result["details"] = {"predicted": pred_value, "expected": gt_value, "error": error}
            except:
                pass

        elif "initial_bearing_deg" in ground_truth or "final_bearing_deg" in ground_truth:
            # Bearing computation
            try:
                numbers = re.findall(r'[\d.]+', prediction)
                if numbers:
                    pred_value = float(numbers[0])
                    gt_key = "initial_bearing_deg" if "initial_bearing_deg" in ground_truth else "final_bearing_deg"
                    gt_value = ground_truth[gt_key]
                    tolerance = ground_truth.get("tolerance_deg", 1.0)
                    error = min(abs(pred_value - gt_value), 360 - abs(pred_value - gt_value))
                    result["correct"] = error <= tolerance
                    result["score"] = max(0, 1 - error / 180)
                    result["details"] = {"predicted": pred_value, "expected": gt_value, "error": error}
            except:
                pass

        elif "answer" in ground_truth:
            # General answer (boolean, number, letter)
            gt_answer = ground_truth["answer"]

            if isinstance(gt_answer, bool):
                pred_bool = prediction.lower() in ["yes", "true", "correct"]
                result["correct"] = pred_bool == gt_answer
                result["score"] = 1.0 if result["correct"] else 0.0
            elif isinstance(gt_answer, str):
                # Letter answer
                pred_letter = prediction.strip().upper()[:1]
                result["correct"] = pred_letter == gt_answer.upper()[:1]
                result["score"] = 1.0 if result["correct"] else 0.0
            else:
                # Numeric answer
                try:
                    numbers = re.findall(r'-?[\d.]+', prediction)
                    if numbers:
                        pred_value = float(numbers[0])
                        tolerance = ground_truth.get("tolerance", abs(gt_answer) * 0.05)
                        error = abs(pred_value - gt_answer)
                        result["correct"] = error <= tolerance
                        result["score"] = max(0, 1 - error / abs(gt_answer)) if gt_answer else (1.0 if error == 0 else 0.0)
                except:
                    pass

        elif "terrain_type" in ground_truth:
            # Terrain classification
            gt_terrain = ground_truth["terrain_type"].lower()
            pred_lower = prediction.lower()
            result["correct"] = gt_terrain in pred_lower or pred_lower.startswith(gt_terrain[:3])
            result["score"] = 1.0 if result["correct"] else 0.0

        else:
            # Default: string matching
            gt_str = str(ground_truth)
            result["correct"] = prediction.lower() == gt_str.lower()
            result["score"] = 1.0 if result["correct"] else 0.0

        return result

    def evaluate_task(
        self,
        track_name: str,
        task_info: Dict[str, str],
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate a single task with batch processing for speed."""
        batch_size = self.batch_size  # Use instance batch_size
        task_name = task_info["file"].replace('.json', '')
        task_file = self.data_dir / f"track_{track_name}" / "splits" / f"{task_name}_test.json"

        if not task_file.exists():
            print(f"  Skipping {task_info['name']}: Test split not found at {task_file}")
            return None

        with open(task_file) as f:
            task_data = json.load(f)

        if max_samples:
            task_data = task_data[:max_samples]

        print(f"\n  Evaluating: {task_info['name']} ({len(task_data)} samples)")

        results = []
        correct = 0
        total_score = 0.0

        # Prepare all prompts first
        all_prompts = []
        for item in task_data:
            system_prompt, user_prompt = self.create_prompt(item, task_name)
            all_prompts.append((system_prompt, user_prompt))

        # Process in batches
        all_responses = []
        num_batches = (len(all_prompts) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc=f"    {task_name}", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_prompts))
            batch_prompts = all_prompts[start_idx:end_idx]

            # Use batch generation
            batch_responses = self.llm_client.generate_batch(batch_prompts)
            all_responses.extend(batch_responses)

        # Evaluate all responses
        for idx, (item, response) in enumerate(zip(task_data, all_responses)):
            user_prompt = all_prompts[idx][1]
            ground_truth = item.get("ground_truth", {})
            task_type = item.get("task", task_name)
            eval_result = self.evaluate_response(response, ground_truth, task_type)

            results.append({
                "question": user_prompt[:200] + "...",
                "response": response,
                **eval_result
            })

            if eval_result["correct"]:
                correct += 1
            total_score += eval_result["score"]

        accuracy = correct / len(task_data) * 100 if task_data else 0
        avg_score = total_score / len(task_data) * 100 if task_data else 0

        print(f"    Accuracy: {accuracy:.1f}% ({correct}/{len(task_data)})")
        print(f"    Avg Score: {avg_score:.1f}%")

        return {
            "task": task_info["name"],
            "file": task_info["file"],
            "samples": len(task_data),
            "correct": correct,
            "accuracy": accuracy,
            "avg_score": avg_score,
            "results": results
        }

    def evaluate_track(
        self,
        track_name: str,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate all tasks in a track."""
        if track_name not in self.tracks:
            raise ValueError(f"Unknown track: {track_name}")

        track_info = self.tracks[track_name]
        print(f"\n{'='*70}")
        print(f"Track: {track_info['name']}")
        print(f"{'='*70}")

        task_results = []
        total_correct = 0
        total_samples = 0
        total_score = 0.0

        for task_info in track_info["tasks"]:
            result = self.evaluate_task(track_name, task_info, max_samples)
            if result:
                task_results.append(result)
                total_correct += result["correct"]
                total_samples += result["samples"]
                total_score += result["avg_score"] * result["samples"]

        track_accuracy = total_correct / total_samples * 100 if total_samples else 0
        track_avg_score = total_score / total_samples if total_samples else 0

        return {
            "track": track_name,
            "name": track_info["name"],
            "tasks": task_results,
            "total_samples": total_samples,
            "total_correct": total_correct,
            "accuracy": track_accuracy,
            "avg_score": track_avg_score
        }

    def evaluate_all(
        self,
        max_samples: Optional[int] = None,
        track_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate all tracks."""
        results = {
            "model": self.llm_client.base_model,
            "checkpoint": self.llm_client.sampler_path,
            "timestamp": datetime.now().isoformat(),
            "max_samples_per_task": max_samples,
            "tracks": {}
        }

        tracks_to_run = [track_filter] if track_filter else list(self.tracks.keys())

        for track_name in tracks_to_run:
            if track_name in self.tracks:
                track_result = self.evaluate_track(track_name, max_samples)
                results["tracks"][track_name] = track_result

        # Calculate overall metrics
        total_correct = sum(t["total_correct"] for t in results["tracks"].values())
        total_samples = sum(t["total_samples"] for t in results["tracks"].values())
        results["overall_accuracy"] = total_correct / total_samples * 100 if total_samples else 0
        results["total_samples"] = total_samples
        results["total_correct"] = total_correct

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned model on GPSBench")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='final',
        help='Checkpoint name (e.g., "final", "000800", "000400")'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/place_association_augmented',
        help='Directory containing checkpoints.jsonl'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen3-30B-A3B-Instruct-2507',
        help='Base model name'
    )
    parser.add_argument(
        '--track',
        type=str,
        choices=['pure_gps', 'applied', 'both'],
        default='both',
        help='Track to evaluate'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per task (for quick testing)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,  # Keep consistent with training
        help='Maximum tokens for generation'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for parallel inference (default: 16)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GPSBench Evaluation - Finetuned Model")
    print("=" * 70)

    # Load checkpoint info
    output_dir = Path(args.output_dir)
    checkpoints_file = output_dir / "checkpoints.jsonl"

    if not checkpoints_file.exists():
        print(f"Error: {checkpoints_file} not found")
        return

    # Find the requested checkpoint
    sampler_path = None
    with open(checkpoints_file) as f:
        for line in f:
            ckpt = json.loads(line)
            if ckpt['name'] == args.checkpoint:
                sampler_path = ckpt['sampler_path']
                print(f"Using checkpoint: {args.checkpoint}")
                print(f"  Epoch: {ckpt['epoch']}, Batch: {ckpt['batch']}")
                print(f"  Sampler path: {sampler_path}")
                break

    if not sampler_path:
        print(f"Error: Checkpoint '{args.checkpoint}' not found")
        print("Available checkpoints:")
        with open(checkpoints_file) as f:
            for line in f:
                ckpt = json.loads(line)
                print(f"  - {ckpt['name']} (epoch {ckpt['epoch']})")
        return

    # Create LLM client
    llm_client = TinkerLLMClient(
        sampler_path=sampler_path,
        base_model=args.model_name,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size
    )

    # Create runner
    runner = GPSBenchFinetunedRunner(llm_client=llm_client, batch_size=args.batch_size)

    # Run evaluation
    track_filter = None if args.track == 'both' else args.track
    results = runner.evaluate_all(
        max_samples=args.max_samples,
        track_filter=track_filter
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Overall accuracy: {results['overall_accuracy']:.1f}%")
    print()

    for track_name, track_result in results["tracks"].items():
        print(f"\n{track_result['name']}:")
        print(f"  Accuracy: {track_result['accuracy']:.1f}%")
        print(f"  Avg Score: {track_result['avg_score']:.1f}%")
        print("  Tasks:")
        for task in track_result["tasks"]:
            print(f"    - {task['task']}: {task['accuracy']:.1f}% ({task['correct']}/{task['samples']})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = runner.results_dir / f"gpsbench_finetuned_{args.checkpoint}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
