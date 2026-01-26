"""
GPSBench Dataset Loaders for Tinker Fine-tuning

Provides flexible data loading from:
1. Existing instruction files (train_instructions.json, dev_instructions.json)
2. Individual task split files (data/track_pure_gps/splits/, data/track_applied/splits/)
3. Task/track filtering for single-task or multi-task training

UPDATED: 2025-12-26 to use prompt templates from prompts/ for consistency with run_benchmark.py
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import chz
import datasets as hf_datasets
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)
from tinker_cookbook.renderers import TrainOnWhat

# Import system prompts
import sys
sys.path.append(str(Path(__file__).parent.parent))
from task_prompt_mapping import get_system_prompt_for_task


def _get_prompts_dir() -> Optional[Path]:
    """Find the prompts directory relative to this file."""
    possible_paths = [
        Path(__file__).parent.parent.parent / "prompts",  # From tinker/
        Path(__file__).parent.parent / "prompts",  # From finetuning/
        Path("prompts"),  # From repo root
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None


def load_prompt_template(task_name: str) -> Optional[str]:
    """
    Load a prompt template from prompts/ directory.

    This ensures consistency with run_benchmark.py which uses the same templates.

    Args:
        task_name: Name of the task (e.g., 'format_conversion', 'distance_calculation')

    Returns:
        Template string if found, None otherwise
    """
    prompts_dir = _get_prompts_dir()
    if prompts_dir is None:
        return None

    template_path = prompts_dir / f"{task_name}.txt"
    if template_path.exists():
        return template_path.read_text().strip()
    return None


@chz.chz
class GPSBenchInstructionDataset(ChatDatasetBuilder):
    """
    Dataset builder that loads from existing instruction files.

    Uses pre-converted instruction files:
    - train_instructions.json (or custom train_file)
    - dev_instructions.json (or custom dev_file)
    """

    data_path: str = "../data"
    train_file: str | None = None  # Optional: custom train file path
    dev_file: str | None = None    # Optional: custom dev file path
    max_samples: int | None = None
    max_eval_samples: int | None = None
    filter_tasks: List[str] | None = None  # Optional: filter by task names

    def load_conversations(self, split: str) -> List[Dict]:
        """Load from existing instruction files."""
        # Determine file path
        if split == "train":
            if self.train_file:
                file_path = Path(self.train_file)
            else:
                file_path = Path(self.data_path) / "train_instructions.json"
            max_samples = self.max_samples
        elif split == "dev":
            if self.dev_file:
                file_path = Path(self.dev_file)
            else:
                file_path = Path(self.data_path) / "dev_instructions.json"
            max_samples = self.max_eval_samples
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Filter by tasks if requested
        if self.filter_tasks:
            data = [
                ex for ex in data
                if ex.get('metadata', {}).get('task') in self.filter_tasks
            ]

        # Limit samples if requested
        if max_samples:
            data = data[:max_samples]

        # Convert to conversation format
        system_prompt = get_system_prompt_for_task("general")

        conversations = []
        for example in data:
            # Extract task name from metadata for evaluation
            metadata = example.get('metadata', {})
            task_name = metadata.get('task', 'unknown')

            conversation = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example['instruction']},
                    {"role": "assistant", "content": example['output']},
                ],
                "task": task_name  # Store task name as simple string for HF Dataset compatibility
            }
            conversations.append(conversation)

        return conversations

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Build train and eval datasets."""
        # Load conversations
        train_conversations = self.load_conversations("train")
        eval_conversations = self.load_conversations("dev")

        # Create HuggingFace datasets
        train_hf = hf_datasets.Dataset.from_list(train_conversations)
        eval_hf = hf_datasets.Dataset.from_list(eval_conversations)

        # Use train_on_what from common_config if provided
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        # Define mapping function
        def map_fn(row: dict):
            return conversation_to_datum(
                row["messages"],
                self.renderer,
                self.common_config.max_length,
                train_on_what
            )

        # Create supervised datasets
        train_dataset = SupervisedDatasetFromHFDataset(
            train_hf,
            batch_size=self.common_config.batch_size,
            map_fn=map_fn
        )

        eval_dataset = SupervisedDatasetFromHFDataset(
            eval_hf,
            batch_size=len(eval_hf),  # Use full eval set as one batch
            map_fn=map_fn
        )

        return train_dataset, eval_dataset


@chz.chz
class GPSBenchSplitDataset(ChatDatasetBuilder):
    """
    Dataset builder that loads directly from GPSBench task split files.

    Supports:
    - Loading from data/track_pure_gps/splits/
    - Loading from data/track_applied/splits/
    - Task-specific training (single or multiple tasks)
    - Optional chain-of-thought reasoning in outputs
    """

    gpsbench_root: str = "../../"  # Path to GPSBench root
    track: str = "both"  # "pure_gps", "applied", or "both"
    tasks: List[str] | None = None  # Optional: specific task names (e.g., ["task1_format_conversion"])
    max_samples: int | None = None
    max_eval_samples: int | None = None
    include_reasoning: bool = False  # Include chain-of-thought reasoning in outputs

    def format_output_with_reasoning(self, example: Dict, final_answer: str) -> str:
        """
        Format output with optional reasoning steps.

        If include_reasoning is True and reasoning_steps exist in the example,
        prepend the reasoning to the final answer.
        """
        if not self.include_reasoning:
            return final_answer

        reasoning_steps = example.get('reasoning_steps', [])
        if not reasoning_steps:
            return final_answer

        # Join reasoning steps with newlines
        if isinstance(reasoning_steps, list):
            reasoning_text = "\n".join(reasoning_steps)
        else:
            reasoning_text = str(reasoning_steps)

        # The reasoning_steps already include "FINAL ANSWER: ..." at the end
        # So we just return the reasoning text directly
        return reasoning_text

    def get_split_files(self, split: str) -> List[Path]:
        """Get list of split files to load."""
        gpsbench_path = Path(self.gpsbench_root)
        split_suffix = f"_{split}.json"

        files = []

        # Determine which tracks to load
        tracks_to_load = []
        if self.track in ["pure_gps", "both"]:
            tracks_to_load.append("track_pure_gps")
        if self.track in ["applied", "both"]:
            tracks_to_load.append("track_applied")

        # Collect split files
        for track_name in tracks_to_load:
            track_path = gpsbench_path / "data" / track_name / "splits"

            if not track_path.exists():
                continue

            # Get all split files for this split type
            pattern = f"*{split_suffix}"
            split_files = list(track_path.glob(pattern))

            # Filter by specific tasks if requested
            if self.tasks:
                split_files = [
                    f for f in split_files
                    if any(task in f.stem for task in self.tasks)
                ]

            files.extend(split_files)

        return files

    def load_raw_examples(self, split: str) -> List[Dict]:
        """Load raw examples from split files."""
        split_files = self.get_split_files(split)

        all_examples = []
        for file_path in split_files:
            with open(file_path, 'r') as f:
                examples = json.load(f)

            # Add task info to each example
            task_name = file_path.stem.replace(f"_{split}", "")
            for ex in examples:
                ex['_task_file'] = task_name
                all_examples.append(ex)

        return all_examples

    def convert_to_instruction(self, example: Dict) -> Dict:
        """
        Convert raw task example to instruction format using templates from prompts/.

        This uses the same prompt templates as run_benchmark.py to ensure consistency
        between training and evaluation.
        """
        task = example.get('task', 'unknown')
        gt = example.get('ground_truth', {})

        # Format conversion task - use template from prompts/format_conversion.txt
        if task == 'format_conversion':
            template = load_prompt_template('format_conversion')
            if template:
                instruction = template.format(
                    source_format=example.get('source_format', 'source'),
                    target_format=example.get('target_format', 'target'),
                    source_string=example.get('source_string', '')
                )
            else:
                instruction = (
                    f"Convert the coordinate from {example.get('source_format', 'source')} to {example.get('target_format', 'target')}.\n\n"
                    f"Source coordinate ({example.get('source_format', 'source')}): {example.get('source_string', '')}\n\n"
                    f"Provide the converted coordinate in {example.get('target_format', 'target')} format.\n\n"
                    f"FINAL ANSWER: [converted coordinate]"
                )
            output = f"FINAL ANSWER: {gt.get('target_string', '')}"

        # Distance calculation task - use template from prompts/distance_calculation.txt
        elif task == 'distance_calculation':
            point_a = example.get('point_a', {})
            point_b = example.get('point_b', {})
            template = load_prompt_template('distance_calculation')
            if template:
                instruction = template.format(
                    point_a_name=point_a.get('name', 'Point A'),
                    point_a_lat=point_a.get('lat', 0),
                    point_a_lon=point_a.get('lon', 0),
                    point_b_name=point_b.get('name', 'Point B'),
                    point_b_lat=point_b.get('lat', 0),
                    point_b_lon=point_b.get('lon', 0)
                )
            else:
                instruction = (
                    f"Calculate the distance between these two points:\n\n"
                    f"Point A: {point_a.get('name', 'Unknown')}\n"
                    f"Coordinates: {point_a.get('lat', 0)}, {point_a.get('lon', 0)}\n\n"
                    f"Point B: {point_b.get('name', 'Unknown')}\n"
                    f"Coordinates: {point_b.get('lat', 0)}, {point_b.get('lon', 0)}\n\n"
                    f"Provide the distance in kilometers.\n\n"
                    f"FINAL ANSWER: [distance] km"
                )
            output = f"FINAL ANSWER: {gt.get('distance_km', 0):.2f} km"

        # Bearing computation task - use template from prompts/bearing_computation.txt
        elif task == 'bearing_computation':
            point_a = example.get('point_a', {})
            point_b = example.get('point_b', {})
            template = load_prompt_template('bearing_computation')
            if template:
                instruction = template.format(
                    point_a_name=point_a.get('name', 'Point A'),
                    point_a_lat=point_a.get('lat', 0),
                    point_a_lon=point_a.get('lon', 0),
                    point_b_name=point_b.get('name', 'Point B'),
                    point_b_lat=point_b.get('lat', 0),
                    point_b_lon=point_b.get('lon', 0)
                )
            else:
                instruction = (
                    f"Calculate the bearing (initial direction) from Point A to Point B:\n\n"
                    f"Point A: {point_a.get('name', 'Unknown')}\n"
                    f"Coordinates: {point_a.get('lat', 0)}, {point_a.get('lon', 0)}\n\n"
                    f"Point B: {point_b.get('name', 'Unknown')}\n"
                    f"Coordinates: {point_b.get('lat', 0)}, {point_b.get('lon', 0)}\n\n"
                    f"Provide the bearing in degrees (0-360°, where 0° is North, 90° is East, etc.).\n\n"
                    f"FINAL ANSWER: [bearing]°"
                )
            # Output format matches benchmark (just bearing, no cardinal direction)
            bearing = gt.get('initial_bearing_deg', 0)
            output = f"FINAL ANSWER: {bearing:.2f}°"

        # Interpolation task - use template from prompts/interpolation.txt
        elif task == 'interpolation':
            point_a = example.get('point_a', {})
            point_b = example.get('point_b', {})
            fraction = example.get('fraction', 0.5)
            template = load_prompt_template('interpolation')
            if template:
                instruction = template.format(
                    fraction=fraction,
                    point_a_name=point_a.get('name', 'Point A'),
                    point_a_lat=point_a.get('lat', 0),
                    point_a_lon=point_a.get('lon', 0),
                    point_b_name=point_b.get('name', 'Point B'),
                    point_b_lat=point_b.get('lat', 0),
                    point_b_lon=point_b.get('lon', 0)
                )
            else:
                instruction = (
                    f"Find the point that is {fraction} of the way along the great circle path from Point A to Point B.\n\n"
                    f"Point A: {point_a.get('name', 'Unknown')}\n"
                    f"Coordinates: {point_a.get('lat', 0)}, {point_a.get('lon', 0)}\n\n"
                    f"Point B: {point_b.get('name', 'Unknown')}\n"
                    f"Coordinates: {point_b.get('lat', 0)}, {point_b.get('lon', 0)}\n\n"
                    f"Calculate the interpolated point at fraction {fraction} (where 0 = Point A, 1 = Point B).\n"
                    f"Use great circle interpolation (not linear).\n\n"
                    f"FINAL ANSWER: [latitude], [longitude]"
                )
            interp = gt.get('interpolated_point', {'lat': 0, 'lon': 0})
            output = f"FINAL ANSWER: {interp.get('lat', 0):.6f}, {interp.get('lon', 0):.6f}"

        # Area and perimeter task - use template from prompts/area_perimeter.txt
        elif task == 'area_perimeter':
            vertices = example.get('vertices', [])
            vertices_str = "\n".join([f"  Vertex {i+1}: {v.get('name', 'Unknown')} ({v.get('lat', 0)}, {v.get('lon', 0)})" for i, v in enumerate(vertices)])
            template = load_prompt_template('area_perimeter')
            if template:
                instruction = template.format(vertices=vertices_str)
            else:
                instruction = (
                    f"Calculate the area of the polygon formed by these coordinates:\n\n"
                    f"{vertices_str}\n\n"
                    f"Use spherical geometry (Earth radius ≈ 6371 km).\n\n"
                    f"FINAL ANSWER: [area] km²"
                )
            output = f"FINAL ANSWER: {gt.get('area_km2', 0):.2f} km²"

        # Bounding box task - use template from prompts/bounding_box.txt
        elif task == 'bounding_box':
            points = example.get('points', [])
            points_str = "\n".join([f"  {i+1}. {p.get('name', 'Unknown')} ({p.get('lat', 0)}, {p.get('lon', 0)})" for i, p in enumerate(points)])
            template = load_prompt_template('bounding_box')
            if template:
                instruction = template.format(
                    num_points=len(points),
                    points=points_str
                )
            else:
                instruction = (
                    f"Calculate the bounding box (minimum rectangle) containing these {len(points)} points:\n\n"
                    f"{points_str}\n\n"
                    f"Provide the bounding box as: min_lat, max_lat, min_lon, max_lon\n\n"
                    f"FINAL ANSWER: [min_lat], [max_lat], [min_lon], [max_lon]"
                )
            output = f"FINAL ANSWER: {gt.get('min_lat', 0):.6f}, {gt.get('max_lat', 0):.6f}, {gt.get('min_lon', 0):.6f}, {gt.get('max_lon', 0):.6f}"

        # Coordinate system transformation
        elif task == 'coordinate_system_transformation':
            src_coords = example.get('source_coords', {})
            instruction = (
                f"Convert coordinates from {example.get('source_crs', 'WGS84')} to {example.get('target_crs', 'UTM')}.\n\n"
                f"Location: {example.get('location_name', 'Unknown')}\n"
                f"Source coordinates: {src_coords.get('lat', 0)}, {src_coords.get('lon', 0)}\n\n"
                f"Provide the UTM coordinates (easting, northing, zone).\n\n"
                f"FINAL ANSWER: Easting: [easting] m, Northing: [northing] m, Zone: [zone][hemisphere]"
            )
            output = f"FINAL ANSWER: Easting: {gt.get('easting', 0):.2f} m, Northing: {gt.get('northing', 0):.2f} m, Zone: {gt.get('zone', '')}{gt.get('hemisphere', '')}"

        # Place association task - use template from prompts/place_association.txt
        elif task == 'place_association':
            template = load_prompt_template('place_association')
            if template:
                instruction = template.format(
                    coordinate_string=example.get('coordinate_string', '')
                )
            else:
                instruction = (
                    f"What location/place is at coordinates {example.get('coordinate_string', '')}?\n\n"
                    f"Provide the specific location name (city, landmark, region, or country depending on context).\n\n"
                    f"FINAL ANSWER: [location name]"
                )
            output = f"FINAL ANSWER: {gt.get('location_name', 'Unknown')}"

        # Proximity (nearest point)
        elif task == 'proximity':
            ref = example.get('reference_point', {})
            candidates = example.get('candidate_points', [])
            cand_str = "\n".join([f"  {chr(65+i)}) {p.get('name', 'Unknown')}: {p.get('lat', 0)}, {p.get('lon', 0)}" for i, p in enumerate(candidates)])
            instruction = (
                f"Which point is nearest to the reference point?\n\n"
                f"Reference: {ref.get('name', 'Unknown')} ({ref.get('lat', 0)}, {ref.get('lon', 0)})\n\n"
                f"Candidates:\n{cand_str}\n\n"
                f"FINAL ANSWER: [letter]"
            )
            output = f"FINAL ANSWER: {gt.get('answer', 'Unknown')}"

        # Relative position
        elif task == 'relative_position':
            instruction = example.get('question', '')
            if not instruction:
                point_a = example.get('point_a', {})
                point_b = example.get('point_b', {})
                instruction = f"Is {point_a.get('name', 'Point A')} north of {point_b.get('name', 'Point B')}?"
            instruction += "\n\nFINAL ANSWER: [Yes/No]"
            output = f"FINAL ANSWER: {gt.get('answer', 'Unknown')}"

        # Route analysis
        elif task == 'route_analysis':
            instruction = example.get('question', 'Analyze the route.')
            instruction += "\n\nFINAL ANSWER: [Yes/No]"
            output = f"FINAL ANSWER: {gt.get('answer', 'Unknown')}"

        # Route geometry (multiple choice)
        elif task == 'route_geometry':
            subtask = example.get('subtask', 'simplification')
            if subtask == 'polyline_simplification':
                points = example.get('points', [])
                instruction = (
                    f"Given a route with {len(points)} points, which points should be kept after simplification?\n\n"
                    f"Select the minimal set of points that preserve the route shape.\n\n"
                    f"FINAL ANSWER: [letter]"
                )
            else:
                instruction = f"Route geometry task.\n\nFINAL ANSWER: [letter]"
            output = f"FINAL ANSWER: {gt.get('answer', 'Unknown')}"

        # Spatial patterns
        elif task == 'spatial_patterns':
            subtask = example.get('subtask', '')
            if subtask == 'outlier_detection':
                points = example.get('points', [])
                points_str = "\n".join([f"  {chr(65+i)}) {p.get('name', 'Unknown')}: {p.get('lat', 0)}, {p.get('lon', 0)}" for i, p in enumerate(points)])
                instruction = (
                    f"Which point is the geographic outlier (furthest from the group)?\n\n"
                    f"{points_str}\n\n"
                    f"FINAL ANSWER: [letter]"
                )
            else:
                instruction = f"Spatial pattern task.\n\nFINAL ANSWER: [letter]"
            output = f"FINAL ANSWER: {gt.get('answer', 'Unknown')}"

        # Name disambiguation
        elif task == 'name_disambiguation':
            instruction = example.get('question', 'Which location matches the coordinates?')
            instruction += "\n\nFINAL ANSWER: [location], [country]"
            output = f"FINAL ANSWER: {gt.get('name', 'Unknown')}, {gt.get('country', 'Unknown')}"

        # Boundary analysis
        elif task == 'boundary_analysis':
            subtask = example.get('subtask', '')
            if subtask == 'regional_grouping':
                points = example.get('points', [])
                points_str = "\n".join([f"  - {p.get('name', 'Unknown')}: {p.get('continent', 'Unknown')}" for p in points])
                instruction = (
                    f"Group these locations by continent:\n\n"
                    f"{points_str}\n\n"
                    f"How many continents are represented?\n\n"
                    f"FINAL ANSWER: [number]"
                )
            else:
                instruction = f"Boundary analysis task.\n\nFINAL ANSWER: [answer]"
            output = f"FINAL ANSWER: {gt.get('num_continents', gt.get('answer', 'Unknown'))}"

        # Terrain classification
        elif task == 'terrain_classification':
            instruction = example.get('question', 'What terrain type is at these coordinates?')
            candidates = example.get('candidates', [])
            if candidates:
                cand_str = "\n".join(candidates)
                instruction += f"\n\nOptions:\n{cand_str}"
            instruction += "\n\nFINAL ANSWER: [letter]"
            output = f"FINAL ANSWER: {gt.get('answer', 'Unknown')}"

        # Missing data handling
        elif task == 'missing_data_handling':
            instruction = example.get('question', 'Infer the missing coordinate.')
            instruction += "\n\nFINAL ANSWER: [longitude]"
            output = f"FINAL ANSWER: {gt.get('longitude', 0):.6f}"

        # Semantic validation - use template from prompts/semantic_validation.txt
        elif task == 'semantic_validation':
            template = load_prompt_template('semantic_validation')
            if template:
                instruction = template.format(
                    context=example.get('context', 'location'),
                    coordinate_string=example.get('coordinate_string', '')
                )
            else:
                instruction = (
                    f"Is it plausible to have a {example.get('context', 'location')} at coordinates {example.get('coordinate_string', '')}?\n\n"
                    f"FINAL ANSWER: [Yes/No with brief reason]"
                )
            output = f"FINAL ANSWER: {gt.get('answer', 'Unknown')}"

        # Fallback for unknown tasks
        else:
            instruction = f"Task: {task}\n\nData: {json.dumps(example, indent=2)}\n\nFINAL ANSWER: [your answer]"
            output = f"FINAL ANSWER: {json.dumps(gt)}"

        # Apply reasoning formatting if enabled
        output = self.format_output_with_reasoning(example, output)

        return {
            "instruction": instruction,
            "output": output,
            "metadata": example
        }

    def load_conversations(self, split: str) -> List[Dict]:
        """Load and convert task data to conversation format."""
        # Load raw examples
        raw_examples = self.load_raw_examples(split)

        # Limit samples
        max_samples = self.max_samples if split == "train" else self.max_eval_samples
        if max_samples:
            raw_examples = raw_examples[:max_samples]

        # Convert to instruction format
        instructions = [self.convert_to_instruction(ex) for ex in raw_examples]

        # Convert to conversation format
        system_prompt = get_system_prompt_for_task("general")

        conversations = []
        for example in instructions:
            # Extract task name from metadata for evaluation
            metadata = example.get('metadata', {})
            task_name = metadata.get('task', 'unknown')

            conversation = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example['instruction']},
                    {"role": "assistant", "content": example['output']},
                ],
                "task": task_name  # Store task name as simple string for HF Dataset compatibility
            }
            conversations.append(conversation)

        return conversations

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Build train and eval datasets."""
        # Load conversations
        train_conversations = self.load_conversations("train")
        eval_conversations = self.load_conversations("dev")

        # Create HuggingFace datasets
        train_hf = hf_datasets.Dataset.from_list(train_conversations)
        eval_hf = hf_datasets.Dataset.from_list(eval_conversations)

        # Use train_on_what from common_config if provided
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        # Define mapping function
        def map_fn(row: dict):
            return conversation_to_datum(
                row["messages"],
                self.renderer,
                self.common_config.max_length,
                train_on_what
            )

        # Create supervised datasets
        train_dataset = SupervisedDatasetFromHFDataset(
            train_hf,
            batch_size=self.common_config.batch_size,
            map_fn=map_fn
        )

        eval_dataset = SupervisedDatasetFromHFDataset(
            eval_hf,
            batch_size=len(eval_hf),  # Use full eval set as one batch
            map_fn=map_fn
        )

        return train_dataset, eval_dataset
