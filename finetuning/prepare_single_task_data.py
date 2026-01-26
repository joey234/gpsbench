#!/usr/bin/env python3
"""
Prepare single-task fine-tuning data: Coordinate → Location
Focus only on place_association (reverse geocoding)
"""

import json
from pathlib import Path
from typing import List, Dict

def load_place_association_eval_data(data_dir: Path) -> List[Dict]:
    """Load place_association evaluation data"""
    eval_file = data_dir / "track_applied" / "task1_place_association.json"

    with open(eval_file) as f:
        return json.load(f)

def create_training_example(example: Dict) -> Dict:
    """Convert evaluation example to training format"""

    coord = example['coordinate']
    lat = coord['lat']
    lon = coord['lon']
    coord_str = example['coordinate_string']

    gt = example['ground_truth']
    location_name = gt['name']
    country = gt.get('country', '')
    region = gt.get('region', '')

    # Create simple, clear prompt
    prompt = f"What location is at coordinates {coord_str}?"

    # Create clear answer
    answer = location_name
    if country and country != location_name:
        answer = f"{location_name}, {country}"

    # System prompt
    system_prompt = """You are a geographic AI assistant. Given coordinates, identify the location name.
Provide clear, concise answers with the place name and country if relevant."""

    return {
        "input": prompt,
        "output": answer,
        "system_prompt": system_prompt,
        "metadata": {
            "task": "place_association",
            "subtask": example.get('subtask', 'landmark_identification'),
            "coordinate": coord,
            "difficulty": example.get('difficulty', 'medium')
        }
    }

def augment_training_data(examples: List[Dict], augment_factor: int = 2) -> List[Dict]:
    """Augment training data with variations"""
    augmented = []

    for ex in examples:
        # Original example
        augmented.append(ex)

        # Variation 1: Different phrasing
        coord_str = f"{ex['metadata']['coordinate']['lat']}, {ex['metadata']['coordinate']['lon']}"

        var1 = {
            **ex,
            "input": f"Identify the location at {coord_str}."
        }
        augmented.append(var1)

        if augment_factor > 2:
            # Variation 2: Another phrasing
            var2 = {
                **ex,
                "input": f"Which place is located at coordinates {coord_str}?"
            }
            augmented.append(var2)

    return augmented

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Creating Single-Task Fine-Tuning Dataset")
    print("Task: Coordinate → Location (Place Association)")
    print("=" * 80)
    print()

    # Load evaluation data
    print("Loading evaluation data...")
    eval_data = load_place_association_eval_data(data_dir)
    print(f"Loaded {len(eval_data)} evaluation examples")

    # Create training examples
    print("\nCreating training examples...")
    training_examples = [create_training_example(ex) for ex in eval_data]

    # Augment data
    print("Augmenting training data...")
    augmented_examples = augment_training_data(training_examples, augment_factor=3)
    print(f"Created {len(augmented_examples)} training examples (3x augmentation)")

    # Split into train/dev/test
    total = len(augmented_examples)
    train_size = int(0.8 * total)
    dev_size = int(0.1 * total)

    train_data = augmented_examples[:train_size]
    dev_data = augmented_examples[train_size:train_size + dev_size]
    test_data = augmented_examples[train_size + dev_size:]

    print(f"\nSplit:")
    print(f"  Train: {len(train_data)} examples ({len(train_data)/total*100:.1f}%)")
    print(f"  Dev:   {len(dev_data)} examples ({len(dev_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data)} examples ({len(test_data)/total*100:.1f}%)")

    # Save datasets
    print("\nSaving datasets...")

    output_files = {
        'train_single_task.json': train_data,
        'dev_single_task.json': dev_data,
        'test_single_task.json': test_data
    }

    for filename, data in output_files.items():
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {output_path} ({len(data)} examples)")

    # Show sample examples
    print("\n" + "=" * 80)
    print("Sample Training Examples:")
    print("=" * 80)

    for i, ex in enumerate(train_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Input:  {ex['input']}")
        print(f"  Output: {ex['output']}")
        print(f"  Coords: {ex['metadata']['coordinate']}")

    print("\n" + "=" * 80)
    print("Dataset creation complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the generated examples in data/train_single_task.json")
    print("2. Run training with:")
    print("   python scripts/train_single_task.py")

if __name__ == "__main__":
    main()
