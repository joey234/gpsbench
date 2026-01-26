"""
Track Configuration for GPSBench Two-Track System
Maps layers and tasks to Pure GPS vs Applied tracks
"""

TRACK_MAPPING = {
    # Track 1: Pure GPS Knowledge (TEST SPLIT COUNTS)
    "pure_gps": {
        "tier_1_syntax_formats": {
            "layer": "track_pure_gps",
            "tasks": {
                "format_conversion": {
                    "file": "task1_format_conversion.json",
                    "count": 158  # TEST SPLIT COUNT
                },
                "precision_understanding": {
                    "file": "task3_precision_understanding.json",
                    "count": 167  # TEST SPLIT COUNT
                },
                "coordinate_validation": {
                    "file": "task2_coordinate_validation.json",
                    "count": 145  # TEST SPLIT COUNT
                },
                "coordinate_system_transformation": {
                    "file": "task4_coordinate_system_transformation.json",
                    "count": 120  # TEST SPLIT COUNT
                }
            },
            "total": 590  # TEST SPLIT TOTAL
        },
        "tier_2_geometric_ops": {
            "layer": "track_pure_gps",
            "tasks": {
                "distance_calculation": {
                    "file": "task5_distance_calculation.json",
                    "count": 122  # TEST SPLIT COUNT
                },
                "bearing_computation": {
                    "file": "task6_bearing_computation.json",
                    "count": 123  # TEST SPLIT COUNT
                },
                "interpolation": {
                    "file": "task7_interpolation.json",
                    "count": 121  # TEST SPLIT COUNT
                },
                "area_perimeter": {
                    "file": "task8_area_perimeter.json",
                    "count": 122  # TEST SPLIT COUNT
                },
                "bounding_box": {
                    "file": "task9_bounding_box.json",
                    "count": 92  # TEST SPLIT COUNT
                },
                "route_geometry": {
                    "file": "task10_route_geometry.json",
                    "count": 120  # TEST SPLIT COUNT
                }
            },
            "total": 700  # TEST SPLIT TOTAL
        },
        "tier_3_pure_robustness": {
            "layer": "track_pure_gps",
            "tasks": {
                "coordinate_matching": {
                    "file": "task13_coordinate_matching.json",
                    "count": 30,  # TEST SPLIT COUNT
                    "note": "Match coordinates within GPS tolerance"
                },
                "noise_sensitivity": {
                    "file": "task11_noise_sensitivity.json",
                    "count": 120,  # TEST SPLIT COUNT
                    "note": "GPS noise sensitivity across different noise levels"
                },
                "noise_tolerance": {
                    "file": "task11_noise_tolerance.json",
                    "count": 300,  # TEST SPLIT COUNT (NEWLY GENERATED)
                    "note": "Handle GPS noise and measurement uncertainty"
                }
            },
            "total": 450,  # TEST SPLIT TOTAL (UPDATED)
            "note": "Pure robustness - coordinate understanding stability"
        }
    },

    # Track 2: Applied Geographic Reasoning (TEST SPLIT COUNTS)
    "applied": {
        "tier_1_location_grounding": {
            "layer": "track_applied",
            "tasks": {
                "place_association": {
                    "file": "task1_place_association.json",
                    "count": 304  # TEST SPLIT COUNT (NEWLY GENERATED)
                },
                "semantic_validation": {
                    "file": "task5_semantic_validation.json",
                    "count": 100,  # TEST SPLIT COUNT
                    "note": "Plausibility checking for location-entity pairs"
                },
                "regional_knowledge": {
                    "file": "task2_regional_knowledge.json",
                    "count": 111  # TEST SPLIT COUNT
                },
                "context_grounding": {
                    "file": "task3_context_grounding.json",
                    "count": 151  # TEST SPLIT COUNT
                },
                "name_disambiguation": {
                    "file": "task4_name_disambiguation.json",
                    "count": 120  # TEST SPLIT COUNT
                },
                "terrain_classification": {
                    "file": "task5_terrain_classification.json",
                    "count": 300  # TEST SPLIT COUNT (NEWLY GENERATED)
                }
            },
            "total": 1086  # TEST SPLIT TOTAL (UPDATED)
        },
        "tier_2_spatial_relations": {
            "layer": "track_applied",
            "tasks": {
                "relative_position": {
                    "file": "task6_relative_position.json",
                    "count": 123  # TEST SPLIT COUNT
                },
                "proximity": {
                    "file": "task7_proximity.json",
                    "count": 120  # TEST SPLIT COUNT
                },
                "route_analysis": {
                    "file": "task8_route_analysis.json",
                    "count": 123  # TEST SPLIT COUNT
                },
                "boundary_analysis": {
                    "file": "task9_boundary_analysis.json",
                    "count": 120  # TEST SPLIT COUNT
                },
                "spatial_patterns": {
                    "file": "task10_spatial_patterns.json",
                    "count": 146,  # TEST SPLIT COUNT
                    "note": "Cluster identification and outlier detection"
                }
            },
            "total": 632  # TEST SPLIT TOTAL
        },
        "tier_3_robustness": {
            "layer": "track_applied",
            "tasks": {
                "consistency": {
                    "file": "task12_consistency.json",
                    "count": 90,  # TEST SPLIT COUNT
                    "note": "Paraphrase robustness and repeated query stability"
                },
                "hallucination_detection": {
                    "file": "task11_hallucination_detection.json",
                    "count": 150,  # TEST SPLIT COUNT (NEWLY GENERATED)
                    "note": "Detect false or misattributed location claims"
                },
                "unit_confusion": {
                    "file": "task13_unit_confusion.json",
                    "count": 120,  # TEST SPLIT COUNT
                    "note": "Handle ambiguous distance and coordinate units"
                },
                "missing_data": {
                    "file": "task12_missing_data.json",
                    "count": 120,  # TEST SPLIT COUNT (MOVED FROM PURE GPS)
                    "note": "Infer missing coordinate components (requires world knowledge)"
                }
            },
            "total": 480,  # TEST SPLIT TOTAL (UPDATED)
            "note": "Applied robustness - location identification despite coordinate noise/errors"
        }
    }
}


def get_track_info(track_name):
    """Get information about a specific track"""
    if track_name not in TRACK_MAPPING:
        raise ValueError(f"Unknown track: {track_name}. Must be 'pure_gps' or 'applied'")
    return TRACK_MAPPING[track_name]


def get_tier_info(track_name, tier_name):
    """Get information about a specific tier"""
    track = get_track_info(track_name)
    if tier_name not in track:
        raise ValueError(f"Unknown tier: {tier_name} for track {track_name}")
    return track[tier_name]


def get_all_tasks_for_track(track_name):
    """Get all tasks for a track across all tiers"""
    track = get_track_info(track_name)
    all_tasks = []

    for tier_name, tier_data in track.items():
        layer = tier_data["layer"]
        for task_name, task_info in tier_data["tasks"].items():
            all_tasks.append({
                "track": track_name,
                "tier": tier_name,
                "layer": layer,
                "task_name": task_name,
                "file": task_info["file"],
                "subtasks": task_info.get("subtasks"),
                "count": task_info["count"]
            })

    return all_tasks


def get_total_tests(track_name=None, tier_name=None):
    """Get total number of tests"""
    if track_name and tier_name:
        tier = get_tier_info(track_name, tier_name)
        return tier["total"]
    elif track_name:
        track = get_track_info(track_name)
        return sum(tier["total"] for tier in track.values())
    else:
        # Total across all tracks
        return sum(
            sum(tier["total"] for tier in track.values())
            for track in TRACK_MAPPING.values()
        )


# Tier display names
TIER_NAMES = {
    "tier_1_syntax_formats": "Syntax & Formats",
    "tier_2_geometric_ops": "Geometric Operations",
    "tier_3_pure_robustness": "Pure Robustness",
    "tier_1_location_grounding": "Location Grounding",
    "tier_2_spatial_relations": "Spatial Relations",
    "tier_3_robustness": "Applied Robustness",
}


def get_tier_display_name(tier_key):
    """Get human-readable tier name"""
    return TIER_NAMES.get(tier_key, tier_key)


if __name__ == "__main__":
    # Print track summary
    print("=" * 80)
    print("GPSBench Two-Track System")
    print("=" * 80)

    for track_name in ["pure_gps", "applied"]:
        track = get_track_info(track_name)
        total = get_total_tests(track_name)

        print(f"\n{track_name.upper().replace('_', ' ')}: {total} tests")
        print("-" * 80)

        for tier_name, tier_data in track.items():
            tier_display = get_tier_display_name(tier_name)
            print(f"  {tier_display}: {tier_data['total']} tests")
            for task_name, task_info in tier_data["tasks"].items():
                print(f"    - {task_name}: {task_info['count']} tests")

    print("\n" + "=" * 80)
    print(f"TOTAL: {get_total_tests()} tests")
    print("=" * 80)
