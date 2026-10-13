"""
Track Configuration for GPSBench Two-Track System
Maps layers and tasks to Pure GPS vs Applied tracks
"""

# Compatibility view for older track utilities. The current runner uses registry.py.
try:
    from .registry import MAIN_TASKS
except ImportError:
    from registry import MAIN_TASKS

TRACK_MAPPING = {"pure_gps": {}, "applied": {}}
for task in MAIN_TASKS:
    tier = ("tier_1_syntax_formats" if task.key in {"Format", "Transform"} else
            "tier_2_geometric_ops" if task.track == "pure_gps" else
            "tier_1_location_grounding" if task.key in {"Place", "Disambiguation", "Terrain"} else
            "tier_2_spatial_relations")
    group = TRACK_MAPPING[task.track].setdefault(tier, {
        "layer": "track_" + task.track, "tasks": {}, "total": 0})
    count = 1022 if task.key == "Transform" else 1020
    group["tasks"][task.stem.split("_", 1)[1]] = {"file": task.stem + ".json", "count": count}
    group["total"] += count


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
