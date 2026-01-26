#!/usr/bin/env python3
"""
Generate LaTeX formatted prompts for each GPSBench task.

Usage:
    python scripts/generate_latex_prompts.py
    python scripts/generate_latex_prompts.py --output paper/prompts.tex
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '&': r'\&',
        '#': r'\#',
        '%': r'\%',
        '_': r'\_',
        '^': r'\^{}',
        '~': r'\textasciitilde{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def get_task_info() -> Dict[str, Dict]:
    """Define task metadata."""
    return {
        # Pure GPS tasks
        'task1_format_conversion': {
            'name': 'Format Conversion',
            'track': 'Pure GPS',
            'description': 'Convert coordinates between DMS and decimal degrees formats.',
        },
        'task3_coordinate_system_transformation': {
            'name': 'Coordinate System Transformation',
            'track': 'Pure GPS',
            'description': 'Transform coordinates between different coordinate systems (WGS84, UTM, Web Mercator).',
        },
        'task4_distance_calculation': {
            'name': 'Distance Calculation',
            'track': 'Pure GPS',
            'description': 'Calculate geodesic distance between two points.',
        },
        'task5_bearing_computation': {
            'name': 'Bearing Computation',
            'track': 'Pure GPS',
            'description': 'Compute initial bearing/azimuth between two points.',
        },
        'task6_interpolation': {
            'name': 'Coordinate Interpolation',
            'track': 'Pure GPS',
            'description': 'Find intermediate point along a geodesic path.',
        },
        'task7_area_perimeter': {
            'name': 'Area & Perimeter',
            'track': 'Pure GPS',
            'description': 'Calculate area and perimeter of geographic polygons.',
        },
        'task8_bounding_box': {
            'name': 'Bounding Box',
            'track': 'Pure GPS',
            'description': 'Determine bounding box for a set of coordinates.',
        },
        'task9_route_geometry': {
            'name': 'Route Geometry',
            'track': 'Pure GPS',
            'description': 'Analyze geometric properties of routes and paths.',
        },
        'task10_relative_position': {
            'name': 'Relative Position',
            'track': 'Pure GPS',
            'description': 'Determine relative position/direction between points.',
        },
        # Applied tasks
        'task1_place_association': {
            'name': 'Place Association',
            'track': 'Applied',
            'description': 'Associate coordinates with place names and locations.',
        },
        'task3_name_disambiguation': {
            'name': 'Name Disambiguation',
            'track': 'Applied',
            'description': 'Disambiguate between places with similar names.',
        },
        'task5_relative_position': {
            'name': 'Relative Position (Applied)',
            'track': 'Applied',
            'description': 'Determine spatial relationships between named places.',
        },
        'task6_proximity': {
            'name': 'Proximity & Nearest Neighbor',
            'track': 'Applied',
            'description': 'Find nearest locations and proximity relationships.',
        },
        'task7_route_analysis': {
            'name': 'Route Analysis',
            'track': 'Applied',
            'description': 'Analyze routes between named locations.',
        },
        'task8_boundary_analysis': {
            'name': 'Boundary Analysis',
            'track': 'Applied',
            'description': 'Group locations by geographic boundaries (continents, countries).',
        },
        'task9_spatial_patterns': {
            'name': 'Spatial Patterns',
            'track': 'Applied',
            'description': 'Identify geographic outliers and spatial patterns.',
        },
        'task11_missing_data': {
            'name': 'Missing Data Inference',
            'track': 'Applied',
            'description': 'Infer missing coordinate components from partial data.',
        },
        'task13_terrain_classification': {
            'name': 'Terrain Classification',
            'track': 'Applied',
            'description': 'Classify terrain type based on coordinates.',
        },
    }


def load_sample_prompt(task_file: Path) -> str:
    """Load a sample prompt from task results."""
    # Try to find from results first (has actual prompts)
    results_dir = Path("results")
    for model_dir in results_dir.glob("gpsbench_gpt-4.1_*"):
        # Check both tracks
        for track in ['pure_gps', 'applied']:
            task_name = task_file.stem
            result_file = model_dir / "task_results" / f"{track}_{task_name}.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                if data.get('results') and len(data['results']) > 0:
                    return data['results'][0].get('prompt', '')
    return ''


def generate_prompt_from_sample(task_file: Path) -> str:
    """Generate a prompt from task data file."""
    with open(task_file) as f:
        samples = json.load(f)

    if not samples:
        return ''

    sample = samples[0]
    task_name = task_file.stem

    # Generate prompt based on task type
    if 'format_conversion' in task_name:
        return f"""Convert the coordinate from {sample.get('source_format', 'DMS')} to {sample.get('target_format', 'decimal_degrees')}.

Source coordinate ({sample.get('source_format', 'DMS')}): {sample.get('source_string', '')}

Provide the converted coordinate in {sample.get('target_format', 'decimal_degrees')} format.

FINAL ANSWER: [converted coordinate]"""

    elif 'distance_calculation' in task_name:
        pa = sample.get('point_a', {})
        pb = sample.get('point_b', {})
        return f"""Calculate the distance between these two points:

Point A: {pa.get('name', '')}
Coordinates: {pa.get('lat', '')}, {pa.get('lon', '')}

Point B: {pb.get('name', '')}
Coordinates: {pb.get('lat', '')}, {pb.get('lon', '')}

Provide the distance in kilometers.

FINAL ANSWER: [distance] km"""

    elif 'bearing' in task_name:
        pa = sample.get('point_a', {})
        pb = sample.get('point_b', {})
        return f"""Calculate the initial bearing from Point A to Point B:

Point A: {pa.get('name', '')} ({pa.get('lat', '')}, {pa.get('lon', '')})
Point B: {pb.get('name', '')} ({pb.get('lat', '')}, {pb.get('lon', '')})

Provide the bearing in degrees (0-360).

FINAL ANSWER: [bearing]°"""

    elif 'boundary_analysis' in task_name:
        points = sample.get('points', [])
        cities = ', '.join(p.get('name', '') for p in points[:4])
        return f"""Group these cities by continent: {cities}

FINAL ANSWER: [continent]: [cities]; [continent]: [cities]"""

    elif 'terrain' in task_name:
        lat = sample.get('latitude', '')
        lon = sample.get('longitude', '')
        candidates = sample.get('candidates', ['Urban', 'Forest', 'Desert', 'Mountain', 'Ocean', 'Grassland'])
        options = '\n'.join(f"{chr(65+i)}) {c}" for i, c in enumerate(candidates[:6]))
        return f"""What type of terrain is at coordinates {lat}, {lon}?

Options:
{options}

Provide your answer as FINAL ANSWER: [letter]"""

    elif 'missing_data' in task_name:
        if 'latitude' in sample:
            return f"""Given latitude {sample.get('latitude')}° and location name '{sample.get('location_name', '')}', what is the longitude?

FINAL ANSWER: [longitude]°"""
        else:
            return f"""Given longitude {sample.get('longitude')}° and location name '{sample.get('location_name', '')}', what is the latitude?

FINAL ANSWER: [latitude]°"""

    elif 'spatial_patterns' in task_name:
        points = sample.get('points', [])
        cities = ', '.join(p.get('name', '') for p in points[:5])
        return f"""Which location is the geographic outlier among: {cities}?

FINAL ANSWER: [city name]"""

    else:
        # Generic prompt
        return sample.get('question', sample.get('prompt', str(sample)[:500]))


def generate_latex_prompts(output_path: str = None):
    """Generate LaTeX file with all task prompts."""

    lines = []

    # LaTeX preamble with custom environment
    lines.append(r"""% GPSBench Task Prompts
% Include this in your LaTeX document preamble:
%
% \usepackage{tcolorbox}
% \usepackage{listings}
%
% \newtcolorbox{llmprompt}[1][]{
%   colback=gray!5,
%   colframe=gray!50,
%   fonttitle=\bfseries,
%   title={LLM Prompt},
%   #1
% }
%
% Alternative simple environment:
% \newenvironment{llmprompt}{\begin{quote}\ttfamily\small}{\end{quote}}

%==============================================================================
% ENVIRONMENT DEFINITION (add to preamble)
%==============================================================================

\usepackage{tcolorbox}
\tcbuselibrary{listings,skins,breakable}

\newtcolorbox{llmprompt}[1][]{
  enhanced,
  breakable,
  colback=gray!5,
  colframe=gray!40,
  boxrule=0.5pt,
  left=8pt,
  right=8pt,
  top=6pt,
  bottom=6pt,
  fontupper=\ttfamily\small,
  #1
}

\newtcolorbox{systemprompt}[1][]{
  enhanced,
  breakable,
  colback=blue!3,
  colframe=blue!30,
  boxrule=0.5pt,
  left=8pt,
  right=8pt,
  top=6pt,
  bottom=6pt,
  fontupper=\ttfamily\small,
  title={\small\sffamily System Prompt},
  #1
}

%==============================================================================
% SYSTEM PROMPT (used for all tasks)
%==============================================================================

\begin{systemprompt}
You are an expert in GPS coordinates, geographic information systems, and spatial reasoning.

Provide accurate, precise answers based on the given coordinates and geographic data.

You may show your reasoning or calculations, but you MUST always end with:
FINAL ANSWER: [your answer]

For multiple choice questions, include only the letter (A, B, C, D, etc.) in FINAL ANSWER.
For numeric answers, provide the number with appropriate units.
For location names, provide the specific location name.
\end{systemprompt}

%==============================================================================
% TASK PROMPTS
%==============================================================================
""")

    task_info = get_task_info()

    # Process Pure GPS tasks first, then Applied
    for track_name in ['Pure GPS', 'Applied']:
        lines.append(f"\n%------------------------------------------------------------------------------")
        lines.append(f"% {track_name.upper()} TRACK")
        lines.append(f"%------------------------------------------------------------------------------\n")

        track_dir = 'track_pure_gps' if track_name == 'Pure GPS' else 'track_applied'
        data_dir = Path(f"data/{track_dir}")

        for task_file in sorted(data_dir.glob("task*.json")):
            if 'splits' in str(task_file):
                continue

            task_key = task_file.stem
            info = task_info.get(task_key, {})

            if not info or info.get('track') != track_name:
                continue

            # Get prompt
            prompt = load_sample_prompt(task_file)
            if not prompt:
                prompt = generate_prompt_from_sample(task_file)

            if not prompt:
                continue

            # Escape and format
            prompt_escaped = escape_latex(prompt)

            lines.append(f"% Task: {info.get('name', task_key)}")
            lines.append(f"% {info.get('description', '')}")
            lines.append(r"\subsection{" + info.get('name', task_key) + "}")
            lines.append(info.get('description', ''))
            lines.append("")
            lines.append(r"\begin{llmprompt}")
            lines.append(prompt_escaped)
            lines.append(r"\end{llmprompt}")
            lines.append("")

    content = '\n'.join(lines)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(content)
        print(f"Saved: {output_file}")
    else:
        print(content)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX prompts for GPSBench tasks")
    parser.add_argument('--output', type=str, default='paper/task_prompts.tex',
                        help='Output file path')

    args = parser.parse_args()
    generate_latex_prompts(args.output)


if __name__ == "__main__":
    main()
