"""Explicit scoring references for the released examples.

Source annotations are historical inputs. Relative Position directions are
recomputed using the paper's axis rule; Interpolation uses its primary threshold.
"""
from copy import deepcopy

from evaluation.neurips_metrics import INTERPOLATION_TOLERANCE
from evaluation.strict_evaluator import axis_direction, candidate_targets


def scoring_example(source):
    """Return current labels and an accepted answer, omitting legacy solutions."""
    result = deepcopy(source)
    result.pop("reasoning_steps", None)
    if source["task"] == "relative_position":
        result["ground_truth"]["answer"] = axis_direction(source)
    elif source["task"] == "interpolation":
        result["ground_truth"]["tolerance_deg"] = INTERPOLATION_TOLERANCE
    result["scoring_reference_answer"] = reference_answer(source)
    return result


def reference_answer(source):
    """Return one accepted answer, not an independent validation of geography."""
    gt, task = source["ground_truth"], source["task"]
    if task == "relative_position":
        return axis_direction(source)
    if task == "format_conversion":
        return gt["target_string"]
    if task == "coordinate_system_transformation":
        keys = ("x", "y") if "x" in gt else ("easting", "northing")
        return ", ".join(str(gt[k]) for k in keys)
    if task == "distance_calculation":
        return f"{gt['distance_km']} km"
    if task == "bearing_computation":
        return f"{gt['initial_bearing_deg']} degrees"
    if task == "area_perimeter":
        return f"{gt['area_km2']} km²"
    if task == "interpolation":
        return ", ".join(str(gt["interpolated_point"][k]) for k in ("lat", "lon"))
    if task == "bounding_box":
        return ", ".join(str(gt[k]) for k in ("min_lat", "max_lat", "min_lon", "max_lon"))
    if task == "name_disambiguation":
        return chr(65 + candidate_targets(source, gt)[0])
    if task == "boundary_analysis":
        return "; ".join(f"{continent}: {', '.join(names)}" for continent, names in gt["grouping"].items())
    if task == "place_association":
        return gt["location_name"]
    if task == "route_analysis":
        return "Yes" if gt["answer"] else "No"
    if task in {"missing_data", "missing_data_handling"}:
        return str(gt["latitude"] if "latitude" in gt else gt["longitude"])
    return str(gt["answer"])
