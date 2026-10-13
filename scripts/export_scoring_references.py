"""Export current reference answers and explicit corrections to historical labels."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.neurips_metrics import VERSION, INTERPOLATION_TOLERANCE
from evaluation.references import reference_answer
from evaluation.registry import TASKS
from evaluation.strict_evaluator import axis_direction


def build():
    references, corrections, hashes = {}, {}, {}
    for t in TASKS:
        for split in ("train", "dev", "test"):
            path = ROOT / "data" / f"track_{t.track}" / "splits" / f"{t.stem}_{split}.json"
            rows = json.loads(path.read_text())
            relative = str(path.relative_to(ROOT))
            if split == "test":
                references[t.result_stem] = [reference_answer(s) for s in rows]
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            if t.key in {"RelPure", "RelApplied"}:
                corrections[relative] = [
                    {"example_id": i, "stored_answer": s["ground_truth"]["answer"],
                     "scoring_answer": axis_direction(s)}
                    for i, s in enumerate(rows)
                    if s["ground_truth"]["answer"].casefold() != axis_direction(s).casefold()
                ]
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"version": VERSION,
            "example_ids": "Zero-based positions in the named split file.",
            "scope": "17 main tasks and the separate Missing Data probe. Source labels and worked solutions are historical; use these references or evaluation.references.scoring_example for current scoring.",
            "interpolation_primary_tolerance_deg": INTERPOLATION_TOLERANCE,
            "input_sha256": hashes, "direction_corrections": corrections,
            "test_reference_answers": references}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the export without writing it")
    args = parser.parse_args()
    path = ROOT / "data/scoring_references.json"
    result = build()
    if args.check:
        if json.loads(path.read_text()) != result:
            raise ValueError("Scoring reference export differs; regenerate and review it")
    else:
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Verified {sum(map(len, result['test_reference_answers'].values()))} test references and "
          f"{sum(map(len, result['direction_corrections'].values()))} direction corrections across all splits")


if __name__ == "__main__":
    main()
