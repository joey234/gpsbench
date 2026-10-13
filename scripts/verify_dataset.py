"""Verify the released split checksums, counts, and task registry without editing data."""
import hashlib
import json
from pathlib import Path
from collections import Counter
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.registry import TASKS


def verify(root=ROOT):
    manifest = json.loads((root / "data/MANIFEST.json").read_text())
    registered = {(f"track_{t.track}", t.stem, t.role) for t in TASKS}
    recorded = {(t["track"], t["task"], t["role"]) for t in manifest["tasks"]}
    if registered != recorded:
        raise ValueError("Manifest task list differs from the evaluation registry")
    totals = Counter()
    files = 0
    for task in manifest["tasks"]:
        combined = []
        for split, digest in task["checksums"].items():
            path = root / task["files"][split]
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch: {path}")
            files += 1
            if split in ("train", "dev", "test"):
                rows = json.loads(path.read_text())
                if len(rows) != task["counts"][split]:
                    raise ValueError(f"Count mismatch: {path}")
                combined.extend(rows)
                if task["role"] == "main":
                    totals[split] += len(rows)
        if task["missing_splits"] or len(combined) != task["total_on_disk"]:
            raise ValueError(f"Incomplete task: {task['task']}")
        unsplit = root / "data" / task["track"] / (task["task"] + ".json")
        rows = json.loads(unsplit.read_text())
        # Splits add worked solutions without changing the task or its reference.
        canonical = lambda items: Counter(json.dumps(
            {k: v for k, v in r.items() if k != "reasoning_steps"}, sort_keys=True)
            for r in items)
        if canonical(rows) != canonical(combined):
            raise ValueError(f"Unsplit data differs from released splits: {unsplit}")
    if dict(totals) != {"train": 35522, "dev": 5750, "test": 17342}:
        raise ValueError(f"Unexpected main split totals: {totals}")
    print(f"Verified {files} split/metadata checksums and 18 unsplit task files; main counts: {dict(totals)}")


if __name__ == "__main__":
    verify()
