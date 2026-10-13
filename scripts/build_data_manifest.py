#!/usr/bin/env python3
"""Per-file counts and SHA-256 checksums for the GPSBench data release."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_JSON = DATA / "MANIFEST.json"
OUT_MD = DATA / "MANIFEST.md"
SUMMARY = DATA / "splits_summary.json"

MAIN_PURE = [
    "task1_format_conversion",
    "task3_coordinate_system_transformation",
    "task4_distance_calculation",
    "task5_bearing_computation",
    "task6_interpolation",
    "task7_area_perimeter",
    "task8_bounding_box",
    "task9_route_geometry",
    "task10_relative_position",
]
MAIN_APPLIED = [
    "task1_place_association",
    "task3_name_disambiguation",
    "task5_relative_position",
    "task6_proximity",
    "task7_route_analysis",
    "task8_boundary_analysis",
    "task9_spatial_patterns",
    "task13_terrain_classification",
]
DIAGNOSTIC = ["task11_missing_data"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_len(path: Path) -> int:
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        for key in ("examples", "data", "results", "items"):
            if isinstance(obj.get(key), list):
                return len(obj[key])
        return 1
    return 0


def collect_task(track: str, stem: str, role: str) -> dict:
    split_dir = DATA / track / "splits"
    files = {}
    counts = {}
    checksums = {}
    missing = []
    for split in ("train", "dev", "test"):
        path = split_dir / f"{stem}_{split}.json"
        rel = str(path.relative_to(ROOT))
        if not path.exists():
            missing.append(split)
            continue
        n = json_len(path)
        files[split] = rel
        counts[split] = n
        checksums[split] = sha256_file(path)
    info = split_dir / f"{stem}_split_info.json"
    if info.exists():
        files["split_info"] = str(info.relative_to(ROOT))
        checksums["split_info"] = sha256_file(info)
    return {
        "task": stem,
        "track": track,
        "role": role,
        "files": files,
        "counts": counts,
        "checksums": checksums,
        "missing_splits": missing,
        "total_on_disk": sum(counts.values()),
    }


def main():
    tasks = (
        [collect_task("track_pure_gps", s, "main") for s in MAIN_PURE]
        + [collect_task("track_applied", s, "main") for s in MAIN_APPLIED]
        + [collect_task("track_applied", s, "diagnostic") for s in DIAGNOSTIC]
    )

    main_tasks = [t for t in tasks if t["role"] == "main"]
    diag = [t for t in tasks if t["role"] == "diagnostic"]
    main_total = sum(t["total_on_disk"] for t in main_tasks)
    test_total = sum(t["counts"].get("test", 0) for t in main_tasks)
    diag_total = sum(t["total_on_disk"] for t in diag)

    summary = {
        "generated": str(date.today()),
        "spec": "evaluation/SCORING_SPEC.md",
        "n_main_tasks": len(main_tasks),
        "n_diagnostic_tasks": len(diag),
        "main_rows_on_disk": main_total,
        "main_test_rows": test_total,
        "diagnostic_rows_on_disk": diag_total,
        "notes": [
            "Missing Data is the Section 4.2 diagnostic, not an 18th main task.",
            "Name Disambiguation is 4,214 (2,883/311/1,020). Test is 1,020 rows / 470 unique questions.",
            "Coordinate Transformation splits 2,039/339/1,022 (still 3,400).",
            "The previously advertised 57,800 assumed 3,400 × 17.",
        ],
        "tasks": tasks,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# GPSBench data manifest (v1.0)",
        "",
        f"Generated: {summary['generated']}",
        "",
        f"- Main tasks: **{len(main_tasks)}**",
        f"- Main rows on disk: **{main_total:,}**",
        f"- Main test rows (reported tables): **{test_total:,}**",
        f"- Diagnostic (Missing Data): **{diag_total:,}**",
        "",
        "Missing Data is not an 18th main task. Name Disambiguation test is 1,020 rows (470 unique questions). Coordinate Transformation is 2,039/339/1,022.",
        "",
        "Source dump: `data/raw/cities15000.txt` has 32,709 rows and 244 country codes. "
        "Unsplit task JSONs live beside `splits/`; Name Disambiguation unsplit is the 4,214-row concat of its splits. "
        "`*.backup` files under `data/` are not part of the release.",
        "",
        "| Task | Role | Train | Dev | Test | Total | Missing |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for t in tasks:
        c = t["counts"]
        lines.append(
            f"| {t['task']} | {t['role']} | {c.get('train', '—')} | {c.get('dev', '—')} | "
            f"{c.get('test', '—')} | {t['total_on_disk']} | "
            f"{', '.join(t['missing_splits']) or '—'} |"
        )
    lines += ["", "## Checksums (SHA-256)", ""]
    for t in tasks:
        for split, digest in t["checksums"].items():
            rel = t["files"].get(split, "")
            lines.append(f"- `{rel}` `{digest}`")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))

    # Refresh splits_summary.json from on-disk counts
    old = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    refreshed = []
    for t in tasks:
        entry = {
            "task": t["task"],
            "role": t["role"],
            "track": t["track"],
            "total_examples": t["total_on_disk"],
            "splits": t["counts"],
            "files": {k: Path(v).name for k, v in t["files"].items() if k in ("train", "dev", "test")},
            "missing_splits": t["missing_splits"],
        }
        if t["task"] == "task3_name_disambiguation":
            entry["notes"] = (
                "Test is 1,020 rows / 470 unique questions (used in all reported tables). "
                "Older splits_summary.json listed test=206 and total=3400."
            )
        if t["role"] == "diagnostic":
            entry["notes"] = "Missing Data probe; not a main benchmark task."
        refreshed.append(entry)
    SUMMARY.write_text(json.dumps({
        "version": "v1.0",
        "seed": old.get("seed", 42),
        "n_main_tasks": len(main_tasks),
        "n_diagnostic_tasks": len(diag),
        "main_rows_on_disk": main_total,
        "main_test_rows": test_total,
        "ratios": old.get("ratios", {"train": 0.6, "dev": 0.1, "test": 0.3}),
        "tasks": refreshed,
    }, indent=2) + "\n")

    print(f"main tasks={len(main_tasks)} rows={main_total} test={test_total}")
    print(f"diagnostic rows={diag_total}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"updated {SUMMARY}")
    for t in tasks:
        if t["missing_splits"]:
            print(f"  missing {t['task']}: {t['missing_splits']}")


if __name__ == "__main__":
    main()
