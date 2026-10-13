import json
import unittest
from pathlib import Path

from evaluation.current import score_response
from evaluation.references import scoring_example
from evaluation.registry import TASKS
from scripts.export_scoring_references import build

ROOT = Path(__file__).resolve().parents[1]


class ReferenceConsistencyTests(unittest.TestCase):
    def test_published_references_match_source_files_and_scorer(self):
        published = json.loads((ROOT / "data/scoring_references.json").read_text())
        self.assertEqual(published, build())
        counts = {"main": 0, "diagnostic": 0}
        for task in TASKS:
            path = ROOT / "data" / f"track_{task.track}" / "splits" / f"{task.stem}_test.json"
            sources = json.loads(path.read_text())
            references = published["test_reference_answers"][task.result_stem]
            self.assertEqual(len(references), len(sources))
            for i, (source, answer) in enumerate(zip(sources, references)):
                self.assertTrue(score_response("FINAL ANSWER: " + answer, source).correct,
                                (task.key, i, answer))
                counts[task.role] += 1
        self.assertEqual(counts, {"main": 17342, "diagnostic": 1020})

    def test_scoring_view_does_not_mutate_historical_annotations(self):
        source = {"task": "relative_position", "point_a": {"lat": 10, "lon": -170},
                  "point_b": {"lat": 0, "lon": 170}, "ground_truth": {"answer": "West"},
                  "reasoning_steps": ["FINAL ANSWER: West"]}
        current = scoring_example(source)
        self.assertEqual(current["ground_truth"]["answer"], "East")
        self.assertEqual(current["scoring_reference_answer"], "East")
        self.assertNotIn("reasoning_steps", current)
        self.assertEqual(source["ground_truth"]["answer"], "West")
        self.assertIn("reasoning_steps", source)

    def test_paper_precision_boundaries(self):
        cases = [
            ({"task": "coordinate_system_transformation", "ground_truth": {"easting": 1000, "northing": 2000}}, "1010, 1990", "1010.01, 1990"),
            ({"task": "coordinate_system_transformation", "ground_truth": {"x": 1000, "y": 2000}}, "1100, 1900", "1100, 1899.99"),
            ({"task": "bounding_box", "ground_truth": {"min_lat": 0, "max_lat": 10, "min_lon": 0, "max_lon": 20}}, "-0.5, 10.5, -0.5, 20.5", "-0.51, 10.5, -0.5, 20.5"),
            ({"task": "missing_data", "ground_truth": {"latitude": 0, "tolerance": .1}}, "0.1", "0.1001"),
            ({"task": "format_conversion", "target_format": "decimal_degrees", "ground_truth": {"latitude": 0, "longitude": 0}}, "0.00005, 0", "0.000051, 0"),
        ]
        for source, accepted, rejected in cases:
            self.assertTrue(score_response("FINAL ANSWER: " + accepted, source).correct, source)
            self.assertFalse(score_response("FINAL ANSWER: " + rejected, source).correct, source)


if __name__ == "__main__":
    unittest.main()
