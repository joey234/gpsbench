"""Current paper scoring for new runs and archived responses. No API calls."""
import argparse
import json
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from statistics import mean

from evaluation.neurips_metrics import VERSION, ERROR_TASKS, included_error, task_score
from evaluation.registry import TASKS, MAIN_TASKS
from evaluation.strict_evaluator import score_row


def score_response(response, source):
    outcome = score_row(response, source["ground_truth"], source["task"], source)
    if source["task"] == "interpolation":
        p = outcome.predicted
        g = source["ground_truth"]["interpolated_point"]
        outcome = replace(outcome, correct=p is not None and
                          abs(p[0] - g["lat"]) <= .5 and abs(p[1] - g["lon"]) <= .5)
    return outcome


@lru_cache(maxsize=64)
def load_sources(path):
    return json.loads(Path(path).read_text())


def compatible(saved, source):
    """Older records may store only a subset of the current reference fields."""
    if isinstance(saved, dict):
        return isinstance(source, dict) and all(k in source and compatible(v, source[k]) for k, v in saved.items())
    return saved == source


def score_task_result(result, track, data_dir):
    stem = Path(result["task_file"]).stem
    task = next(t for t in TASKS if t.track == track and t.stem == stem)
    sources = load_sources(str(Path(data_dir) / f"track_{track}" / "splits" / f"{stem}_test.json"))
    records = result.get("results", [])
    seen = set()
    rows = []
    for record in records:
        i = record.get("example_id", record.get("index"))
        if type(i) is not int or i < 0 or i >= len(sources) or i in seen:
            raise ValueError(f"Invalid or duplicate example_id in {task.result_stem}: {i}")
        seen.add(i)
        source = sources[i]
        if "ground_truth" in record and not compatible(record["ground_truth"], source["ground_truth"]):
            raise ValueError(f"Reference mismatch: {task.result_stem} example {i}")
        response = record.get("response") or record.get("model_response") or ""
        outcome = asdict(score_response(response, source))
        record.update(correct=outcome["correct"], evaluation=outcome)
        rows.append(outcome)
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    result.update(scoring_version=VERSION, role=task.role, total=n, correct=correct,
                  accuracy=100 * correct / n if n else 0.,
                  score=task_score(rows, task.key),
                  metric="normalized_error" if task.key in ERROR_TASKS else "accuracy",
                  parsed=sum(r["parsed"] for r in rows), expected_total=len(sources),
                  complete=n == len(sources))
    if task.key in ERROR_TASKS:
        result["included"] = sum(included_error(r) for r in rows)
    return result


def aggregate(tasks):
    main = [t for t in tasks if t.get("role") == "main"]
    total = sum(t["total"] for t in main)
    correct = sum(t["correct"] for t in main)
    return {"score": mean(t["score"] for t in main) if main else None,
            "scored_tasks": len(main), "total_cases": total, "total_correct": correct,
            "overall_accuracy": 100 * correct / total if total else 0.}


def score_run(results, data_dir):
    all_tasks = []
    for track, data in results.get("tracks", {}).items():
        for task in data.get("tasks", []):
            score_task_result(task, track, data_dir)
        data.update(aggregate(data["tasks"]))
        data["scoring_version"] = VERSION
        all_tasks.extend(data["tasks"])
    results["overall"] = aggregate(all_tasks)
    results["scoring_version"] = VERSION
    results["complete"] = (results["overall"]["scored_tasks"] == len(MAIN_TASKS)
                           and all(t["complete"] for t in all_tasks if t["role"] == "main"))
    return results


def rescore_directory(directory, data_dir):
    directory = Path(directory)
    results = {"tracks": {}}
    for task in TASKS:
        path = directory / "task_results" / f"{task.result_stem}.json"
        if not path.exists():
            continue
        saved = json.loads(path.read_text())
        result = saved if isinstance(saved, dict) else {"results": saved}
        result.update(task_file=task.stem + ".json", task_name=task.name)
        results["tracks"].setdefault(task.track, {"tasks": []})["tasks"].append(result)
    if not results["tracks"]:
        raise ValueError(f"No recognized task_results files in {directory}")
    return score_run(results, data_dir)


def compact(results):
    """Keep task metrics, excluding full prompts and predictions."""
    return {**results, "tracks": {track: {**data, "tasks": [
        {k: v for k, v in t.items() if k != "results"} for t in data["tasks"]
    ]} for track, data in results["tracks"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="One run directory containing task_results/")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True, help="New summary file; must not exist")
    args = parser.parse_args()
    report = compact(rescore_directory(args.results, args.data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    score = report["overall"]["score"]
    display = "n/a (diagnostics only)" if score is None else f"{score:.4f}"
    print(f"Scored {report['overall']['scored_tasks']}/17 main tasks; score={display}; complete={report['complete']}")


if __name__ == "__main__":
    main()
