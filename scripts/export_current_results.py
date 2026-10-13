"""Rescore the fixed paper cohort offline and export public scores and provenance."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.current import compact, rescore_directory
from evaluation.neurips_metrics import VERSION
from evaluation.registry import MAIN_TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=ROOT / "evaluation/cohort.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/current/scores.json")
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text())
    models = []
    hashes = {}
    for name, relative in cohort["cohort"].items():
        report = compact(rescore_directory(ROOT / relative, ROOT / "data"))
        if not report["complete"]:
            raise ValueError(f"Incomplete main-task run: {name}")
        scores = {t.key: next(r["score"] for r in report["tracks"][t.track]["tasks"]
                             if r["task_file"] == t.stem + ".json") for t in MAIN_TASKS}
        models.append({"name": name, "pure": report["tracks"]["pure_gps"]["score"],
                       "applied": report["tracks"]["applied"]["score"],
                       "overall": report["overall"]["score"], "scores": scores})
        for t in MAIN_TASKS:
            p = ROOT / relative / "task_results" / (t.result_stem + ".json")
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{name}: {report['overall']['score']:.4f}", flush=True)
    models.sort(key=lambda m: -m["overall"])
    for t in MAIN_TASKS:
        p = ROOT / "data" / f"track_{t.track}" / "splits" / (t.stem + "_test.json")
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for relative in ("evaluation/current.py", "evaluation/registry.py", "evaluation/strict_evaluator.py", "evaluation/neurips_metrics.py"):
        hashes[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    result = {"version": VERSION, "main_tasks": 17, "test_samples_per_model": 17342,
              "models": models, "input_sha256": hashes}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
