"""Integration checks for current scoring and runner output, without API requests."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.current import aggregate, score_response, score_task_result, rescore_directory
from evaluation.registry import MAIN_TASKS, runner_tracks

ROOT = Path(__file__).resolve().parents[1]


class CurrentScoringTests(unittest.TestCase):
    def test_registry_matches_released_splits(self):
        self.assertEqual(len(MAIN_TASKS), 17)
        self.assertEqual(len(runner_tracks()["applied"]["tasks"]), 8)
        self.assertEqual(len(runner_tracks(True)["applied"]["tasks"]), 9)
        self.assertEqual(sum(len(json.loads((ROOT / "data" / f"track_{t.track}" /
                         "splits" / f"{t.stem}_test.json").read_text())) for t in MAIN_TASKS), 17342)

    def test_primary_precision_overrides_stored_annotation(self):
        source = {"task": "interpolation", "ground_truth": {
            "interpolated_point": {"lat": 10, "lon": 20}, "tolerance_deg": .01}}
        self.assertTrue(score_response("FINAL ANSWER: 10.4, 20.4", source).correct)
        self.assertFalse(score_response("FINAL ANSWER: 10.6, 20.4", source).correct)

    def test_macro_average_excludes_probe_and_does_not_weight_sample_counts(self):
        rows = [{"role": "main", "score": 90, "total": 100, "correct": 90},
                {"role": "main", "score": 10, "total": 1, "correct": 0},
                {"role": "diagnostic", "score": 0, "total": 1000, "correct": 0}]
        report = aggregate(rows)
        self.assertEqual(report["score"], 50)
        self.assertEqual(report["total_cases"], 101)
        self.assertNotEqual(report["score"], report["overall_accuracy"])

    def test_archived_ids_and_references_are_validated(self):
        result = {"task_file": "task4_distance_calculation.json",
                  "results": [{"index": 0, "response": "invalid"}]}
        self.assertEqual(score_task_result(result, "pure_gps", ROOT / "data")["score"], 0)
        self.assertFalse(result["complete"])
        result["results"].append(dict(result["results"][0]))
        with self.assertRaises(ValueError):
            score_task_result(result, "pure_gps", ROOT / "data")
        result["results"] = [{"example_id": 0, "ground_truth": {"distance_km": -1}}]
        with self.assertRaises(ValueError):
            score_task_result(result, "pure_gps", ROOT / "data")

    def test_runner_sequential_concurrent_and_saved_batch_outputs(self):
        from run_benchmark import GPSBenchRunner
        source = json.loads((ROOT / "data/track_pure_gps/splits/task4_distance_calculation_test.json").read_text())[0]
        reply = SimpleNamespace(text=f"FINAL ANSWER: {source['ground_truth']['distance_km'] * 1.1} km",
                                error=None, latency_ms=1, total_tokens=2)
        client = SimpleNamespace(model="offline-test", provider="stub",
                                 generate=lambda **kwargs: reply,
                                 batch_generate=lambda **kwargs: [reply] * len(kwargs["prompts"]))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            runner = GPSBenchRunner(client, data_dir=str(ROOT / "data"), results_dir=tmp)
            task_info = {"file": "task4_distance_calculation.json", "name": "Distance Calculation"}
            for concurrent in (False, True):
                task = runner.evaluate_task("pure_gps", task_info, max_samples=1, delay=0,
                                            use_concurrent=concurrent)
                self.assertAlmostEqual(task["score"], 90)
                self.assertEqual(task["accuracy"], 0)
                self.assertEqual(task["included"], 1)
            run = {"model": "offline-test", "provider": "stub", "timestamp": "test",
                   "batch_mode": True, "tracks": {"pure_gps": {"tasks": [task]}}}
            directory = runner.save_results(run, "test-run")
            summary = json.loads((directory / "summary.json").read_text())
            self.assertAlmostEqual(summary["overall"]["score"], 90)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["tracks"]["pure_gps"]["tasks_summary"][0]["metric"], "normalized_error")
            self.assertAlmostEqual(rescore_directory(directory, ROOT / "data")["overall"]["score"], 90)
            # Updating a saved task must also refresh the existing summary.
            task["results"][0]["response"] = "FINAL ANSWER: 0 km"
            runner.update_existing_results(run, "test-run")
            self.assertEqual(json.loads((directory / "summary.json").read_text())["overall"]["score"], 0)


if __name__ == "__main__":
    unittest.main()
