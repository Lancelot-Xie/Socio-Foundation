"""Paired, group-level task retention reports; invalid judges are never successes."""

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .data import read_rows


def compare(student_file, teacher_files, tolerance=.02, bootstrap=1000, seed=42):
    students = {r["id"]: r for r in read_rows(student_file)}
    grouped, counts = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(int))
    axis_expected, axis_paired = defaultdict(int), defaultdict(int)
    for task, filename in teacher_files.items():
        for teacher in read_rows(filename):
            if teacher["task_id"] != task:
                continue
            student = students.get(teacher["id"])
            counts[task]["expected"] += 1
            if student is None:
                counts[task]["missing_student"] += 1
                continue
            t, s = teacher["evaluation"], student["evaluation"]
            for dim in set(t.get("scores", {})) | set(s.get("scores", {})):
                axis_expected[(task, dim)] += 1
            if not t["valid"] or not s["valid"] or not t.get("reliable", True) or not s.get("reliable", True):
                counts[task]["invalid_pairs"] += 1
                continue
            common = set(t["scores"]) & set(s["scores"])
            if not common:
                counts[task]["invalid_pairs"] += 1
                continue
            counts[task]["paired"] += 1
            for dim in sorted(common):
                if dim in ("U", "T") and t.get("applicability") and s.get("applicability"):
                    key = "temporal_scope" if dim == "T" else "outcome_scope"
                    if t["applicability"].get(key) != s["applicability"].get(key):
                        continue
                axis_paired[(task, dim)] += 1
                ts = t["scores"][dim] if t["constraint_pass"] else 0.0
                ss = s["scores"][dim] if s["constraint_pass"] else 0.0
                grouped[(task, dim)][teacher.get("group_id", teacher["id"])].append((ts, ss))
    rng, report = random.Random(seed), {}
    for (task, dim), groups in grouped.items():
        teacher_means = [statistics.mean(p[0] for p in pairs) for pairs in groups.values()]
        student_means = [statistics.mean(p[1] for p in pairs) for pairs in groups.values()]
        differences = [s-t for t, s in zip(teacher_means, student_means)]
        interval = None
        if len(differences) > 1 and bootstrap:
            draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(bootstrap))
            interval = [draws[int(.025*(bootstrap-1))], draws[int(.975*(bootstrap-1))]]
        complete = (counts[task]["paired"] == counts[task]["expected"] and
                    axis_paired[(task, dim)] == axis_expected[(task, dim)])
        # A numerical point estimate alone is not proof of non-inferiority.
        status = "insufficient_evidence"
        if complete and interval:
            status = "within_tolerance" if interval[0] >= -tolerance else (
                "regression" if interval[1] < -tolerance else "uncertain")
        report[f"{task}/{dim}"] = {"groups": len(groups), "teacher": statistics.mean(teacher_means),
                                   "axis_expected": axis_expected[(task, dim)], "axis_paired": axis_paired[(task, dim)],
                                   "student": statistics.mean(student_means), "delta": statistics.mean(differences),
                                   "bootstrap_95_interval": interval, "status": status}
    deltas = [r["delta"] for r in report.values()]
    return {"tasks": dict(counts), "comparisons": report, "tolerance": tolerance,
            "axis_coverage": {f"{task}/{dim}": {"expected": n, "paired": axis_paired[(task, dim)]}
                              for (task, dim), n in axis_expected.items()},
            "worst_delta": min(deltas) if deltas else None,
            "macro_delta": statistics.mean(deltas) if deltas else None,
            "note": "Group bootstrap is approximate. Static/rubric proxies do not establish full simulator retention."}


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
