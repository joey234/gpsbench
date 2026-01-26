#!/usr/bin/env python3
"""
Regenerate finetuning instruction files from GPSBench split files.

This script reads ALL training/dev data from:
- data/track_pure_gps/splits/
- data/track_applied/splits/

And generates properly formatted instruction files for Tinker training.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from task_prompt_mapping import get_system_prompt_for_task

# Add tinker directory for dataset conversion logic
sys.path.insert(0, str(Path(__file__).parent / "tinker"))


def load_prompt_template(task_name: str, prompts_dir: Path) -> str | None:
    """Load a prompt template from prompts/ directory."""
    template_path = prompts_dir / f"{task_name}.txt"
    if template_path.exists():
        return template_path.read_text().strip()
    return None


def convert_to_instruction(example: dict, prompts_dir: Path) -> dict:
    """
    Convert raw task example to instruction format.
    Uses same logic as gpsbench_datasets.py for consistency.
    """
    task = example.get('task', 'unknown')
    gt = example.get('ground_truth', {})

    # Format conversion task
    if task == 'format_conversion':
        template = load_prompt_template('format_conversion', prompts_dir)
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

    # Distance calculation task
    elif task == 'distance_calculation':
        point_a = example.get('point_a', {})
        point_b = example.get('point_b', {})
        template = load_prompt_template('distance_calculation', prompts_dir)
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

    # Bearing computation task
    elif task == 'bearing_computation':
        point_a = example.get('point_a', {})
        point_b = example.get('point_b', {})
        template = load_prompt_template('bearing_computation', prompts_dir)
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
                f"Calculate the bearing from Point A to Point B:\n\n"
                f"Point A: {point_a.get('name', 'Unknown')}\n"
                f"Coordinates: {point_a.get('lat', 0)}, {point_a.get('lon', 0)}\n\n"
                f"Point B: {point_b.get('name', 'Unknown')}\n"
                f"Coordinates: {point_b.get('lat', 0)}, {point_b.get('lon', 0)}\n\n"
                f"Provide the bearing in degrees.\n\n"
                f"FINAL ANSWER: [bearing]°"
            )
        bearing = gt.get('initial_bearing_deg', 0)
        output = f"FINAL ANSWER: {bearing:.2f}°"

    # Interpolation task
    elif task == 'interpolation':
        point_a = example.get('point_a', {})
        point_b = example.get('point_b', {})
        fraction = example.get('fraction', 0.5)
        template = load_prompt_template('interpolation', prompts_dir)
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
                f"Find the point at fraction {fraction} along the path from A to B:\n\n"
                f"Point A: {point_a.get('lat', 0)}, {point_a.get('lon', 0)}\n"
                f"Point B: {point_b.get('lat', 0)}, {point_b.get('lon', 0)}\n\n"
                f"FINAL ANSWER: [latitude], [longitude]"
            )
        interp = gt.get('interpolated_point', {'lat': 0, 'lon': 0})
        output = f"FINAL ANSWER: {interp.get('lat', 0):.6f}, {interp.get('lon', 0):.6f}"

    # Area and perimeter task
    elif task in ['area_perimeter', 'polygon_area']:
        vertices = example.get('vertices', [])
        vertices_str = "\n".join([f"  Vertex {i+1}: {v.get('name', 'Unknown')} ({v.get('lat', 0)}, {v.get('lon', 0)})" for i, v in enumerate(vertices)])
        template = load_prompt_template('area_perimeter', prompts_dir)
        if template:
            instruction = template.format(vertices=vertices_str)
        else:
            instruction = (
                f"Calculate the area of the polygon:\n\n{vertices_str}\n\n"
                f"FINAL ANSWER: [area] km²"
            )
        output = f"FINAL ANSWER: {gt.get('area_km2', 0):.2f} km²"

    # Bounding box task
    elif task == 'bounding_box':
        points = example.get('points', [])
        points_str = "\n".join([f"  {i+1}. {p.get('name', 'Unknown')} ({p.get('lat', 0)}, {p.get('lon', 0)})" for i, p in enumerate(points)])
        template = load_prompt_template('bounding_box', prompts_dir)
        if template:
            instruction = template.format(num_points=len(points), points=points_str)
        else:
            instruction = (
                f"Calculate the bounding box for these {len(points)} points:\n\n{points_str}\n\n"
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
            f"FINAL ANSWER: Easting: [easting] m, Northing: [northing] m, Zone: [zone][hemisphere]"
        )
        output = f"FINAL ANSWER: Easting: {gt.get('easting', 0):.2f} m, Northing: {gt.get('northing', 0):.2f} m, Zone: {gt.get('zone', '')}{gt.get('hemisphere', '')}"

    # Place association task - match benchmark format: City, Country
    elif task == 'place_association':
        # Country code to full name mapping
        COUNTRY_NAMES = {
            'US': 'United States', 'GB': 'United Kingdom', 'CN': 'China', 'IN': 'India',
            'JP': 'Japan', 'DE': 'Germany', 'FR': 'France', 'BR': 'Brazil', 'IT': 'Italy',
            'CA': 'Canada', 'RU': 'Russia', 'KR': 'South Korea', 'ES': 'Spain', 'MX': 'Mexico',
            'AU': 'Australia', 'ID': 'Indonesia', 'NL': 'Netherlands', 'SA': 'Saudi Arabia',
            'TR': 'Turkey', 'CH': 'Switzerland', 'PL': 'Poland', 'BE': 'Belgium', 'SE': 'Sweden',
            'NG': 'Nigeria', 'AR': 'Argentina', 'NO': 'Norway', 'AT': 'Austria', 'AE': 'UAE',
            'IL': 'Israel', 'HK': 'Hong Kong', 'IE': 'Ireland', 'SG': 'Singapore', 'DK': 'Denmark',
            'MY': 'Malaysia', 'CO': 'Colombia', 'PH': 'Philippines', 'PK': 'Pakistan', 'CL': 'Chile',
            'FI': 'Finland', 'BD': 'Bangladesh', 'EG': 'Egypt', 'VN': 'Vietnam', 'CZ': 'Czech Republic',
            'PT': 'Portugal', 'RO': 'Romania', 'NZ': 'New Zealand', 'IR': 'Iran', 'TH': 'Thailand',
            'ZA': 'South Africa', 'GR': 'Greece', 'HU': 'Hungary', 'UA': 'Ukraine', 'DZ': 'Algeria',
            'GT': 'Guatemala', 'PE': 'Peru', 'EC': 'Ecuador', 'VE': 'Venezuela', 'CU': 'Cuba',
            'BO': 'Bolivia', 'DO': 'Dominican Republic', 'HN': 'Honduras', 'PY': 'Paraguay',
            'SV': 'El Salvador', 'NI': 'Nicaragua', 'CR': 'Costa Rica', 'PA': 'Panama', 'UY': 'Uruguay',
            'KE': 'Kenya', 'ET': 'Ethiopia', 'TZ': 'Tanzania', 'UG': 'Uganda', 'GH': 'Ghana',
            'CI': 'Ivory Coast', 'CM': 'Cameroon', 'MG': 'Madagascar', 'AO': 'Angola', 'MZ': 'Mozambique',
            'NE': 'Niger', 'ML': 'Mali', 'BF': 'Burkina Faso', 'MW': 'Malawi', 'ZM': 'Zambia',
            'ZW': 'Zimbabwe', 'SN': 'Senegal', 'TD': 'Chad', 'SO': 'Somalia', 'GN': 'Guinea',
            'RW': 'Rwanda', 'BJ': 'Benin', 'TN': 'Tunisia', 'SS': 'South Sudan', 'TG': 'Togo',
            'SL': 'Sierra Leone', 'LY': 'Libya', 'CG': 'Congo', 'LR': 'Liberia', 'CF': 'Central African Republic',
            'MR': 'Mauritania', 'ER': 'Eritrea', 'GM': 'Gambia', 'BW': 'Botswana', 'NA': 'Namibia',
            'GA': 'Gabon', 'LS': 'Lesotho', 'GW': 'Guinea-Bissau', 'MU': 'Mauritius', 'SZ': 'Eswatini',
            'DJ': 'Djibouti', 'RE': 'Reunion', 'KM': 'Comoros', 'CV': 'Cape Verde', 'ST': 'Sao Tome and Principe',
            'SC': 'Seychelles', 'MM': 'Myanmar', 'AF': 'Afghanistan', 'NP': 'Nepal', 'KH': 'Cambodia',
            'LA': 'Laos', 'LK': 'Sri Lanka', 'KZ': 'Kazakhstan', 'UZ': 'Uzbekistan', 'TM': 'Turkmenistan',
            'KG': 'Kyrgyzstan', 'TJ': 'Tajikistan', 'AZ': 'Azerbaijan', 'GE': 'Georgia', 'AM': 'Armenia',
            'MN': 'Mongolia', 'BN': 'Brunei', 'TL': 'Timor-Leste', 'BT': 'Bhutan', 'MV': 'Maldives',
            'IQ': 'Iraq', 'SY': 'Syria', 'JO': 'Jordan', 'LB': 'Lebanon', 'PS': 'Palestine',
            'YE': 'Yemen', 'OM': 'Oman', 'KW': 'Kuwait', 'BH': 'Bahrain', 'QA': 'Qatar',
            'CY': 'Cyprus', 'AL': 'Albania', 'MK': 'North Macedonia', 'RS': 'Serbia', 'BA': 'Bosnia and Herzegovina',
            'HR': 'Croatia', 'SI': 'Slovenia', 'SK': 'Slovakia', 'LT': 'Lithuania', 'LV': 'Latvia',
            'EE': 'Estonia', 'BY': 'Belarus', 'MD': 'Moldova', 'BG': 'Bulgaria', 'ME': 'Montenegro',
            'XK': 'Kosovo', 'IS': 'Iceland', 'LU': 'Luxembourg', 'MT': 'Malta', 'AD': 'Andorra',
            'MC': 'Monaco', 'SM': 'San Marino', 'LI': 'Liechtenstein', 'VA': 'Vatican City',
            'FJ': 'Fiji', 'PG': 'Papua New Guinea', 'SB': 'Solomon Islands', 'VU': 'Vanuatu',
            'NC': 'New Caledonia', 'PF': 'French Polynesia', 'WS': 'Samoa', 'GU': 'Guam',
            'KI': 'Kiribati', 'FM': 'Micronesia', 'TO': 'Tonga', 'PW': 'Palau', 'MH': 'Marshall Islands',
            'TV': 'Tuvalu', 'NR': 'Nauru', 'JM': 'Jamaica', 'TT': 'Trinidad and Tobago', 'BS': 'Bahamas',
            'BB': 'Barbados', 'LC': 'Saint Lucia', 'GD': 'Grenada', 'VC': 'Saint Vincent', 'AG': 'Antigua and Barbuda',
            'DM': 'Dominica', 'KN': 'Saint Kitts and Nevis', 'HT': 'Haiti', 'BZ': 'Belize', 'SR': 'Suriname',
            'GY': 'Guyana', 'TW': 'Taiwan', 'PR': 'Puerto Rico', 'KP': 'North Korea', 'CD': 'DR Congo',
        }

        # Update instruction to match benchmark format
        coord_str = example.get('coordinate_string', '')
        instruction = (
            f"What location is at coordinates {coord_str}?\n\n"
            f"Provide your answer in the format: City, Region/State, Country "
            f"(e.g., 'Paris, Île-de-France, France' or 'Tokyo, Tokyo, Japan').\n\n"
            f"FINAL ANSWER: [your answer]"
        )

        # Format output: City, Country (we don't have region in ground truth)
        location_name = gt.get('location_name', gt.get('name', 'Unknown'))
        country_code = gt.get('country_code', gt.get('country', ''))
        country_name = COUNTRY_NAMES.get(country_code, country_code)

        if country_name:
            output = f"FINAL ANSWER: {location_name}, {country_name}"
        else:
            output = f"FINAL ANSWER: {location_name}"

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

    # Route geometry
    elif task == 'route_geometry':
        instruction = example.get('question', 'Route geometry task.')
        instruction += "\n\nFINAL ANSWER: [letter]"
        output = f"FINAL ANSWER: {gt.get('answer', 'Unknown')}"

    # Spatial patterns
    elif task == 'spatial_patterns':
        instruction = example.get('question', 'Spatial pattern task.')
        instruction += "\n\nFINAL ANSWER: [letter]"
        output = f"FINAL ANSWER: {gt.get('answer', 'Unknown')}"

    # Name disambiguation
    elif task == 'name_disambiguation':
        instruction = example.get('question', 'Which location matches?')
        instruction += "\n\nFINAL ANSWER: [location], [country]"
        output = f"FINAL ANSWER: {gt.get('name', 'Unknown')}, {gt.get('country', 'Unknown')}"

    # Boundary analysis
    elif task == 'boundary_analysis':
        instruction = example.get('question', 'Boundary analysis task.')
        instruction += "\n\nFINAL ANSWER: [answer]"
        output = f"FINAL ANSWER: {gt.get('num_continents', gt.get('answer', 'Unknown'))}"

    # Terrain classification
    elif task == 'terrain_classification':
        instruction = example.get('question', 'What terrain type?')
        candidates = example.get('candidates', [])
        if candidates:
            cand_str = "\n".join(candidates)
            instruction += f"\n\nOptions:\n{cand_str}"
        instruction += "\n\nFINAL ANSWER: [letter]"
        output = f"FINAL ANSWER: {gt.get('answer', 'Unknown')}"

    # Missing data handling
    elif task in ['missing_data_handling', 'missing_data']:
        instruction = example.get('question', 'Infer the missing coordinate.')
        instruction += "\n\nFINAL ANSWER: [longitude]"
        output = f"FINAL ANSWER: {gt.get('longitude', gt.get('answer', 0))}"

    # Semantic validation
    elif task == 'semantic_validation':
        template = load_prompt_template('semantic_validation', prompts_dir)
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
        instruction = f"Task: {task}\n\nData: {json.dumps(example, indent=2, default=str)}\n\nFINAL ANSWER: [your answer]"
        output = f"FINAL ANSWER: {json.dumps(gt, default=str)}"

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "metadata": {
            "task": task,
            "source_task": example.get('_source_task', 'unknown'),
            "original_data": example
        }
    }


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    prompts_dir = base_dir / "prompts"
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Regenerating Finetuning Instructions from Split Files")
    print("=" * 80)
    print()

    # Collect all split files
    tracks = ['track_pure_gps', 'track_applied']
    splits = ['train', 'dev']

    all_data = {split: [] for split in splits}
    stats = defaultdict(lambda: defaultdict(int))

    for track in tracks:
        splits_dir = data_dir / track / "splits"
        if not splits_dir.exists():
            print(f"Warning: {splits_dir} does not exist, skipping...")
            continue

        for split in splits:
            pattern = f"*_{split}.json"
            split_files = list(splits_dir.glob(pattern))

            print(f"\nLoading {track}/{split}: {len(split_files)} task files")

            for split_file in sorted(split_files):
                task_name = split_file.stem.replace(f"_{split}", "")
                source_task = f"{track}/{task_name}"

                with open(split_file) as f:
                    examples = json.load(f)

                # Add source task info
                for ex in examples:
                    ex['_source_task'] = source_task

                # Convert to instruction format
                converted = []
                for ex in examples:
                    try:
                        instr = convert_to_instruction(ex, prompts_dir)
                        converted.append(instr)
                    except Exception as e:
                        print(f"  Warning: Failed to convert example in {source_task}: {e}")

                all_data[split].extend(converted)
                stats[split][source_task] = len(converted)
                print(f"  {source_task}: {len(converted)} examples")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for split in splits:
        print(f"\n{split.upper()}: {len(all_data[split])} total examples")
        print(f"  Tasks: {len(stats[split])}")

    # Save to files
    print("\n" + "=" * 80)
    print("Saving Files")
    print("=" * 80)

    output_files = {
        'train_instructions.json': all_data['train'],
        'dev_instructions.json': all_data['dev'],
    }

    for filename, data in output_files.items():
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved: {output_path} ({len(data)} examples)")

    # Verify consistency
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)

    print(f"\n{'Task':<50} {'Split Train':>12} {'Instr Train':>12} {'Match':>8}")
    print("-" * 85)

    total_split = 0
    total_instr = 0
    all_match = True

    for track in tracks:
        splits_dir = data_dir / track / "splits"
        if not splits_dir.exists():
            continue

        for train_file in sorted(splits_dir.glob("*_train.json")):
            task_name = train_file.stem.replace("_train", "")
            source_task = f"{track}/{task_name}"

            with open(train_file) as f:
                split_count = len(json.load(f))

            instr_count = stats['train'].get(source_task, 0)

            total_split += split_count
            total_instr += instr_count

            match = "✓" if split_count == instr_count else "✗"
            if split_count != instr_count:
                all_match = False

            print(f"{source_task:<50} {split_count:>12} {instr_count:>12} {match:>8}")

    print("-" * 85)
    print(f"{'TOTAL':<50} {total_split:>12} {total_instr:>12}")

    if all_match:
        print("\n✓ All counts match! Regeneration successful.")
    else:
        print("\n⚠ Some counts don't match. Check for conversion errors.")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
