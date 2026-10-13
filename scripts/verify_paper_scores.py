"""Compare the manuscript's three headline result tables with the released scores."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.registry import MAIN_TASKS


def verify(paper_path):
    # Ignore commented-out source when identifying the currently printed tables.
    paper = "\n".join(line for line in paper_path.read_text().splitlines()
                      if not line.lstrip().startswith("%"))
    scores = json.loads((ROOT / "results/current/scores.json").read_text())
    checked = 0
    for label, keys in (
        ("tab:main_results", ["pure", "applied", "overall"]),
        ("tab:per_task_pure_gps", [t.key for t in MAIN_TASKS if t.track == "pure_gps"]),
        ("tab:per_task_applied", [t.key for t in MAIN_TASKS if t.track == "applied"]),
    ):
        end = paper.index("\\label{" + label + "}")
        start = paper.rfind("\\begin{table", 0, end)
        table = paper[start:end]
        for model in scores["models"]:
            rows = [line for line in table.splitlines() if line.startswith(model["name"] + " & ")]
            if len(rows) != 1:
                raise ValueError(f"Missing or repeated model row: {label}, {model['name']}")
            cells = [v.strip() for v in rows[0].split("&")]
            for i, key in enumerate(keys, 1):
                actual = cells[i].removesuffix("\\\\").strip()
                value = model[key] if label == "tab:main_results" else model["scores"][key]
                if actual != f"{value:.1f}":
                    raise ValueError(f"Mismatch: {label}, {model['name']}, {key}: {actual} != {value:.1f}")
                checked += 1
    print(f"Verified {checked} printed scores against the public result release")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path, help="Path to the current ACL .tex source")
    verify(parser.parse_args().paper)
