"""The 17 released benchmark tasks, plus the optional Missing Data probe."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    key: str
    track: str
    stem: str
    name: str
    role: str = "main"

    @property
    def result_stem(self):
        return f"{self.track}_{self.stem}"


TASKS = (
    Task("Format", "pure_gps", "task1_format_conversion", "Format Conversion"),
    Task("Transform", "pure_gps", "task3_coordinate_system_transformation", "Coordinate System Transformation"),
    Task("Distance", "pure_gps", "task4_distance_calculation", "Distance Calculation"),
    Task("Bearing", "pure_gps", "task5_bearing_computation", "Bearing Computation"),
    Task("Interpolation", "pure_gps", "task6_interpolation", "Coordinate Interpolation"),
    Task("Area", "pure_gps", "task7_area_perimeter", "Polygon Area"),
    Task("BBox", "pure_gps", "task8_bounding_box", "Bounding Box"),
    Task("Geometry", "pure_gps", "task9_route_geometry", "Route Geometry"),
    Task("RelPure", "pure_gps", "task10_relative_position", "Relative Position"),
    Task("Place", "applied", "task1_place_association", "Place Association"),
    Task("Disambiguation", "applied", "task3_name_disambiguation", "Name Disambiguation"),
    Task("RelApplied", "applied", "task5_relative_position", "Relative Position"),
    Task("Proximity", "applied", "task6_proximity", "Proximity & Nearest Neighbor"),
    Task("Route", "applied", "task7_route_analysis", "Route Analysis"),
    Task("Boundary", "applied", "task8_boundary_analysis", "Boundary Analysis"),
    Task("Spatial", "applied", "task9_spatial_patterns", "Spatial Patterns"),
    Task("Terrain", "applied", "task13_terrain_classification", "Terrain Classification"),
    Task("Missing", "applied", "task11_missing_data", "Missing Data Inference", "diagnostic"),
)
MAIN_TASKS = tuple(t for t in TASKS if t.role == "main")


def runner_tracks(include_diagnostics=False):
    return {
        track: {"name": name, "tasks": [
            {"file": t.stem + ".json", "name": t.name}
            for t in TASKS if t.track == track and (include_diagnostics or t.role == "main")
        ]}
        for track, name in (("pure_gps", "Pure GPS Track"), ("applied", "Applied Track"))
    }
