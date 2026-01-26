#!/usr/bin/env python3
"""
Show example paraphrasing outputs for different styles.

This script demonstrates what paraphrased reasoning looks like
for each of the 5 styles without actually running the LLM.
"""

import json
from pathlib import Path


# Example paraphrased reasoning for demonstration
PARAPHRASING_EXAMPLES = {
    "original": [
        "Step 1: Identify the coordinates",
        "  Point A: (-1.2864°, 36.8172°) - Nairobi, Kenya",
        "  Point B: (-0.9373°, 37.12275°) - Kenol",
        "Step 2: Apply the Haversine formula for great circle distance",
        "  Δlat = -0.9373 - -1.2864 = 0.3491°",
        "  Δlon = 37.12275 - 36.8172 = 0.3056°",
        "  Convert to radians and calculate using: d = 2R × arcsin(√(sin²(Δlat/2) + cos(lat₁) × cos(lat₂) × sin²(Δlon/2)))",
        "Step 3: Calculate distance",
        "  Distance = 51.44 km",
        "  In miles: 31.97 miles",
        "Answer: 51.44 km"
    ],

    "formal": [
        "Step 1: Define coordinate pairs",
        "  Let P₁ = (-1.2864°, 36.8172°) represent Nairobi, Kenya",
        "  Let P₂ = (-0.9373°, 37.12275°) represent Kenol",
        "Step 2: Compute geodesic distance via Haversine formula",
        "  Δφ = φ₂ - φ₁ = 0.3491° (where φ denotes latitude)",
        "  Δλ = λ₂ - λ₁ = 0.3056° (where λ denotes longitude)",
        "  d = 2R·arcsin(√(sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2))), R = 6371 km",
        "Step 3: Evaluate the distance metric",
        "  d = 51.44 km",
        "  Conversion: d = 31.97 miles",
        "Answer: 51.44 km"
    ],

    "tutorial": [
        "Step 1: First, let's identify our two locations and their coordinates",
        "  Our starting point is Nairobi, Kenya at latitude -1.2864° and longitude 36.8172°",
        "  The destination is Kenol at latitude -0.9373° and longitude 37.12275°",
        "Step 2: Now we'll use the Haversine formula, which calculates the shortest distance between two points on a sphere",
        "  The difference in latitude: -0.9373° - (-1.2864°) = 0.3491°",
        "  The difference in longitude: 37.12275° - 36.8172° = 0.3056°",
        "  The Haversine formula is: d = 2R × arcsin(√(sin²(Δlat/2) + cos(lat₁) × cos(lat₂) × sin²(Δlon/2)))",
        "Step 3: Let's calculate the final distance",
        "  The distance works out to be 51.44 km",
        "  If you need it in miles, that's 31.97 miles",
        "Answer: 51.44 km"
    ],

    "concise": [
        "Step 1: Points: A(-1.2864°, 36.8172°), B(-0.9373°, 37.12275°)",
        "  Nairobi, Kenya → Kenol",
        "Step 2: Haversine calculation",
        "  Δlat = 0.3491°, Δlon = 0.3056°",
        "  d = 2R × arcsin(√(sin²(Δlat/2) + cos(lat₁)cos(lat₂)sin²(Δlon/2)))",
        "Step 3: Result",
        "  51.44 km (31.97 mi)",
        "Answer: 51.44 km"
    ],

    "detailed": [
        "Step 1: Begin by extracting the geographic coordinates for both locations",
        "  Point A is situated at Nairobi, Kenya, with coordinates latitude -1.2864° (south of equator) and longitude 36.8172° (east of prime meridian)",
        "  Point B is located at Kenol, with coordinates latitude -0.9373° and longitude 37.12275°",
        "Step 2: Apply the Haversine formula, which is specifically designed for calculating great circle distances on a sphere, accounting for Earth's curvature",
        "  Calculate the change in latitude: -0.9373° minus -1.2864° equals 0.3491°",
        "  Calculate the change in longitude: 37.12275° minus 36.8172° equals 0.3056°",
        "  The complete formula is: distance = 2 × Earth_radius × arcsin(√(sin²(Δlatitude/2) + cos(latitude₁) × cos(latitude₂) × sin²(Δlongitude/2)))",
        "  Note: Earth's mean radius is approximately 6371 km",
        "Step 3: Perform the calculation to determine the final distance",
        "  The computed distance between these two points is 51.44 kilometers",
        "  For reference, this is equivalent to 31.97 miles in imperial units",
        "Answer: 51.44 km"
    ],

    "conversational": [
        "Step 1: Okay, so we're looking at two locations here",
        "  First up is Nairobi, Kenya at coordinates -1.2864° latitude, 36.8172° longitude",
        "  Then we have Kenol at -0.9373° latitude, 37.12275° longitude",
        "Step 2: To find the distance, we need to use the Haversine formula since we're working with a sphere (Earth)",
        "  The latitude changes by 0.3491° (from -0.9373° - -1.2864°)",
        "  The longitude changes by 0.3056° (from 37.12275° - 36.8172°)",
        "  Plug those into the formula: d = 2R × arcsin(√(sin²(Δlat/2) + cos(lat₁) × cos(lat₂) × sin²(Δlon/2)))",
        "Step 3: Crunching the numbers",
        "  We get a distance of 51.44 km between these two spots",
        "  That's about 31.97 miles if you prefer",
        "Answer: 51.44 km"
    ]
}


def show_comparison():
    """Show side-by-side comparison of all styles."""
    print("="*80)
    print("PARAPHRASING STYLE COMPARISON - DISTANCE CALCULATION EXAMPLE")
    print("="*80)

    styles = ["original", "formal", "tutorial", "concise", "detailed", "conversational"]

    for style in styles:
        print(f"\n{'='*80}")
        print(f"STYLE: {style.upper()}")
        print(f"{'='*80}\n")

        for step in PARAPHRASING_EXAMPLES[style]:
            print(step)


def show_style_characteristics():
    """Show characteristics of each style."""
    print("\n\n" + "="*80)
    print("STYLE CHARACTERISTICS")
    print("="*80)

    characteristics = {
        "formal": {
            "description": "Mathematical precision and rigor",
            "features": [
                "Uses mathematical symbols (φ, λ, Δ)",
                "Formal variable definitions",
                "Technical terminology",
                "Precise, academic tone"
            ],
            "use_case": "Scientific papers, technical documentation"
        },
        "tutorial": {
            "description": "Educational and beginner-friendly",
            "features": [
                "Explains reasoning behind each step",
                "Uses 'we' and 'let's'",
                "Provides context and motivation",
                "Friendly, encouraging tone"
            ],
            "use_case": "Teaching materials, student training"
        },
        "concise": {
            "description": "Brief and to-the-point",
            "features": [
                "Minimal explanations",
                "Abbreviations where possible",
                "Compact formatting",
                "Direct, no fluff"
            ],
            "use_case": "Quick reference, expert users"
        },
        "detailed": {
            "description": "Comprehensive and thorough",
            "features": [
                "Extended explanations",
                "Additional context and notes",
                "Full sentences",
                "Multiple representations (km and miles)"
            ],
            "use_case": "Documentation, comprehensive guides"
        },
        "conversational": {
            "description": "Natural and approachable",
            "features": [
                "Casual language ('Okay', 'so')",
                "Second person ('you')",
                "Contractions and informality",
                "Relatable explanations"
            ],
            "use_case": "Chatbots, interactive tutoring"
        }
    }

    for style, info in characteristics.items():
        print(f"\n{style.upper()}: {info['description']}")
        print(f"  Features:")
        for feature in info['features']:
            print(f"    • {feature}")
        print(f"  Best for: {info['use_case']}")


def show_actual_dataset_examples():
    """Show actual examples from the dataset with reasoning."""
    print("\n\n" + "="*80)
    print("ACTUAL DATASET EXAMPLES")
    print("="*80)

    base_path = Path(__file__).parent.parent / "data"

    tasks = [
        ("track_pure_gps/task5_distance_calculation.json", "Distance Calculation"),
        ("track_pure_gps/task6_bearing_computation.json", "Bearing Computation"),
        ("track_applied/task7_proximity.json", "Proximity/Nearest Neighbor"),
    ]

    for task_file, task_name in tasks:
        filepath = base_path / task_file

        if not filepath.exists():
            continue

        with open(filepath) as f:
            data = json.load(f)

        # Find example with reasoning
        for ex in data[:10]:
            if 'reasoning_steps' in ex and ex['reasoning_steps']:
                print(f"\n{task_name}:")
                print("-" * 80)
                for i, step in enumerate(ex['reasoning_steps'][:4]):
                    print(step)
                if len(ex['reasoning_steps']) > 4:
                    print("  ...")
                    print(ex['reasoning_steps'][-1])
                break


def main():
    show_comparison()
    show_style_characteristics()
    show_actual_dataset_examples()

    print("\n\n" + "="*80)
    print("DIVERSITY BENEFITS")
    print("="*80)
    print("""
By paraphrasing reasoning steps with multiple styles, we:

1. ✅ Prevent overfitting to specific phrasings
   - Models learn concepts, not memorized templates

2. ✅ Improve generalization
   - Models handle different explanation styles

3. ✅ Enable style transfer
   - Models can adopt preferred explanation style

4. ✅ Better instruction following
   - Models learn from diverse instruction patterns

5. ✅ More robust reasoning
   - Same logic expressed in multiple ways reinforces understanding

NEXT STEPS:

1. Test the setup:
   python scripts/test_paraphrasing_setup.py

2. Run paraphrasing:
   python scripts/paraphrase_reasoning_steps.py --task all --limit 100

3. Read the guide:
   cat scripts/PARAPHRASING_README.md
""")


if __name__ == "__main__":
    main()
