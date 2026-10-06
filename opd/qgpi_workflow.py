"""Main v2 pipeline: existing task experts -> quality calibration -> unified QGPI.

No quality-dimension generative LoRAs are trained. The old matrix is an ablation.
"""

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from accelerate.utils import set_seed

from .checkpoints import atomic_json
from .config import load_config
from .data import load_data, write_rows
from .experiments import Runner, base_config, fingerprint, tokenized_data


def calibrate_quality(cfg, output, limit=64):
    from .inference import inference_workers
    with inference_workers() as workers:
        return _calibrate_quality(cfg, output, limit, workers)


def _calibrate_quality(cfg, output, limit, workers):
    from .models import TeacherBank, load_student, load_tokenizer
    from .qgpi import Shortlist, audit_record, collect_decision_batch
    from .routing import Router
    from .workflows import inference_device
    cfg = copy.deepcopy(cfg)
    cfg["qgpi"]["registry_file"] = None
    cfg["qgpi"]["shortlist"] = len(cfg["teachers"])
    cfg["rollout"]["generation_batch_size"] = cfg["inference"]["batch_size"]
    rows = load_data(cfg["data"]["calibration_file"])
    random.Random(cfg["seed"]).shuffle(rows)
    counts = defaultdict(int)
    selected = []
    for row in rows:
        if not limit or counts[row["task_id"]] < limit:
            counts[row["task_id"]] += 1
            selected.append(row)
    # Apply per-task limits BEFORE sharding; limit is global, not per rank.
    local_rows = workers.shard(selected)
    set_seed(cfg["seed"] + workers.rank)
    device, tokenizer = inference_device(), load_tokenizer(cfg)
    student = load_student(cfg).to(device).requires_grad_(False).eval()
    bank = TeacherBank(cfg, tokenizer, device)
    shortlist = Shortlist(cfg, Router(cfg))
    print(f"[calibration] rank={workers.rank}/{workers.world_size} device={device} "
          f"rows={len(local_rows)} batch={cfg['inference']['batch_size']}", flush=True)
    details = []
    size = cfg["inference"]["batch_size"]
    for start in range(0, len(local_rows), size):
        decisions = collect_decision_batch(student, bank, shortlist, local_rows[start:start + size], tokenizer, cfg, device)
        details.extend(audit_record(d) for d in decisions)
    details = workers.gather(details)
    if not workers.main:
        return {}
    return summarize_calibration(cfg, output, selected, details, workers.world_size)


def summarize_calibration(cfg, output, rows, details, world_size=1):
    grouped, invalid, counts = defaultdict(lambda: defaultdict(list)), defaultdict(int), defaultdict(int)
    order = {row["id"]: i for i, row in enumerate(rows)}
    by_row = defaultdict(list)
    for decision in details:
        by_row[decision["row_id"]].append(decision)
    if set(by_row) != set(order) or any(len({d["turn"] for d in ds}) != len(ds) for ds in by_row.values()):
        raise ValueError("Distributed calibration has missing or duplicate decisions")
    details.sort(key=lambda d: (order[d["row_id"]], d["turn"]))
    for row in rows:
        task = row["task_id"]
        counts[task] += 1
        for decision in by_row[row["id"]]:
            base = decision["evaluations"][0]
            for candidate, result in zip(decision["candidates"][1:], decision["evaluations"][1:], strict=True):
                teacher = candidate["teacher"]
                q = cfg["qgpi"]
                if any(not r["valid"] or r.get("confidence", 0) < q["min_confidence"] or
                       r.get("disagreement", 1) > q["max_disagreement"] for r in (base, result)):
                    invalid[(task, teacher)] += 1
                    continue
                group = row.get("group_id", row["id"])
                for dim in [*decision["spec"]["dimensions"], "utility"]:
                    gain = result[dim] - base[dim] if dim == "utility" else result["scores"][dim] - base["scores"][dim]
                    grouped[(task, teacher, dim)][group].append(gain)
    tasks = {}
    for task, route in cfg["routing"]["tasks"].items():
        tasks[task] = {}
        for teacher in route["teachers"]:
            dimensions = {}
            for dim in ("F", "S", "U", "T", "N", "utility"):
                values = [statistics.mean(v) for v in grouped[(task, teacher, dim)].values()]
                n = len(values)
                stderr = statistics.stdev(values)/math.sqrt(n) if n > 1 else None
                gain = statistics.mean(values) if n else None
                dimensions[dim] = {"groups": n, "gain": gain, "stderr": stderr,
                                   "lower_gain": gain - 1.96*stderr if n >= 3 else None}
            utility = dimensions.pop("utility")
            tasks[task][teacher] = {"dimensions": dimensions, "utility": utility,
                                    "lower_gain": utility["lower_gain"] if utility["lower_gain"] is not None else 0.0,
                                    "invalid": invalid[(task, teacher)],
                                    "status": "estimated" if utility["groups"] >= 3 else "insufficient_evidence"}
    report = {"schema": "qgpi-quality-v1", "base_model": cfg["model"]["base_model"], "teachers": cfg["teachers"],
              "tasks": tasks, "rows_per_task": dict(counts), "config": cfg["qgpi"],
              "world_size": world_size, "batch_size_per_rank": cfg["inference"]["batch_size"],
              "note": "Group-level approximate lower bounds rank compatible teachers only; per-state gates still decide imitation. Sparse/invalid evidence is not proof of competence."}
    write_rows(str(output) + ".details.jsonl", details)
    atomic_json(Path(output), report)
    return report


def run_qgpi(plan, resume=False, dry_run=False):
    if plan.get("baseline_scheduler", "distributed") not in ("distributed", "task_pool"):
        raise ValueError("baseline_scheduler must be distributed or task_pool")
    runner = Runner(plan, resume, dry_run)
    data = Path(plan.get("prepared_data") or runner.root / "data")
    cfg = base_config(plan, data)
    if not cfg["qgpi"]["enabled"]:
        raise ValueError("workflow=qgpi requires qgpi.enabled=true")
    provenance = {"plan": plan, "data": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(data.glob("*.jsonl"))}}
    signature = fingerprint(provenance)
    manifest = runner.root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["signature"] != signature:
        raise ValueError("QGPI plan/data changed; use a new output directory")
    atomic_json(manifest, {"signature": signature, **provenance})
    if plan.get("recipe_alignment"):
        atomic_json(runner.root / "recipe_alignment.json", plan["recipe_alignment"])
    if plan.get("expert_inventory"):
        atomic_json(runner.root / "expert_inventory.json", plan["expert_inventory"])
    if not dry_run:
        data = tokenized_data(cfg, data, runner)
        cfg = base_config(plan, data)
    train_rows = load_data(cfg["data"]["train_file"])
    from .quality_basis import quality_spec
    coverage = {}
    for split in ("train", "validation", "calibration", "eval"):
        split_file = data / (split + ".jsonl")
        if not split_file.exists():
            continue
        coverage[split] = {}
        for row in load_data(split_file):
            counter = coverage[split].setdefault(row["task_id"], dict.fromkeys(["rows", "F", "S", "U", "T", "N"], 0))
            counter["rows"] += 1
            for dim in quality_spec(cfg, row, row["messages"], [])["dimensions"]:
                counter[dim] += 1
    atomic_json(runner.root / "quality_coverage.json", {"counts": coverage, "scope": "Static visible evidence applicability, not judge-validity or future interaction"})
    eval_rows = load_data(cfg["data"]["eval_file"]) if plan.get("evaluate", True) else []
    tasks = set(cfg["routing"]["tasks"])
    required = [("train", train_rows)] + ([("evaluation", eval_rows)] if plan.get("evaluate", True) else [])
    if plan.get("calibrate", True):
        required.append(("calibration", load_data(cfg["data"]["calibration_file"])))
    for split, rows in required:
        if tasks - {r["task_id"] for r in rows}:
            raise ValueError(f"QGPI {split} missing a routed task after length filtering")
    results = {}
    for seed in plan.get("seeds", [42]):
        base = copy.deepcopy(cfg)
        base["seed"] = seed
        stem = f"seed_{seed}"
        baselines, version_baselines = {}, {}
        if plan.get("evaluate", True) and plan.get("val_before_train", True):
            requests = []

            def baseline_evaluate(name, eval_cfg, model=None, teacher=None):
                if plan.get("baseline_scheduler", "distributed") == "distributed":
                    return runner.evaluate(name, eval_cfg, model=model, teacher=teacher)
                output = str(runner.root / "evaluation" / (name + ".jsonl"))
                requests.append({"name": name, "eval_file": eval_cfg["data"]["eval_file"],
                                 "output": output, "model": model, "teacher": teacher})
                return output

            baseline = baseline_evaluate(stem + "/base", base, model=base["model"]["base_model"])
            for task, route in base["routing"]["tasks"].items():
                teacher_cfg = copy.deepcopy(base)
                subset = runner.root / "eval_data" / f"{task}.jsonl"
                write_rows(subset, [r for r in eval_rows if r["task_id"] == task])
                teacher_cfg["data"]["eval_file"] = str(subset)
                version_baselines[task] = {}
                for teacher in route["teachers"]:
                    name = task if len(route["teachers"]) == 1 else task + "/" + teacher
                    version_baselines[task][teacher] = baseline_evaluate(stem + "/teacher/" + name, teacher_cfg, teacher=teacher)
                if len(route["teachers"]) == 1:
                    baselines[task] = next(iter(version_baselines[task].values()))
            if requests:
                runner.baseline_pool(stem, base, requests)
        if plan.get("calibrate", True):
            path = runner.config(stem + "/calibration", base)
            registry = runner.root / "calibration" / stem / "registry.json"
            runner.command("calibration/" + stem, runner.inference_command("opd.qgpi_workflow", "calibrate",
                           "--config", path, "--output", registry, "--limit", plan.get("calibration_limit_per_task", 64)),
                           [registry, str(registry) + ".details.jsonl"], {"config": base, "limit": plan.get("calibration_limit_per_task", 64)})
            base["qgpi"]["registry_file"] = str(registry)
            if not dry_run and version_baselines:
                calibration = json.loads(registry.read_text())
                for task, versions in version_baselines.items():
                    estimated = {t: r for t, r in calibration["tasks"][task].items() if r["status"] == "estimated"}
                    if len(versions) > 1 and estimated:
                        selected = max(estimated, key=lambda t: estimated[t]["lower_gain"])
                        baselines[task] = versions[selected]
        base["train"]["max_steps"] = plan["steps"]["final"]
        model, _ = runner.train(stem + "/unified_qgpi", base)
        active = sorted({d for counts in coverage.get("train", {}).values() for d in ("F", "S", "U", "T", "N") if counts[d]})
        results[stem] = {"model": model, "quality_basis": ["F", "S", "U", "T", "N"], "active_dimensions": active,
                         "C": "excluded", "all_expert_evaluations": version_baselines,
                         "retention_references": baselines,
                         "reference_selection": "Single supplied version, or multiple versions ranked on calibration only; insufficient multi-version evidence leaves no selected reference."}
        if plan.get("evaluate", True):
            evaluation = runner.evaluate(stem + "/unified_qgpi", base, model=model)
            results[stem]["evaluation"] = evaluation
            if not dry_run and plan.get("val_before_train", True):
                from .retention import compare, save_report
                for name, reference in (("task_experts", baselines), ("base", {task: baseline for task in tasks})):
                    report = compare(evaluation, reference, plan.get("retention_tolerance", .02), seed=seed)
                    save_report(runner.root / "retention" / stem / (name + ".json"), report)
    summary = {"workflow": "qgpi", "output_dir": str(runner.root), "dry_run": dry_run,
               "training_order": ["reuse_common_mixed_sft_and_task_experts", "quality_calibration", "unified_qgpi", "five_quality_evaluation"],
               "quality_basis": ["F", "S", "U", "T", "N"], "dimension_generation_experts": False,
               "optimizer_steps_per_seed": plan["steps"]["final"], "commands_this_invocation": len(runner.commands),
               "inference_num_processes": plan.get("inference_num_processes", 1),
               "baseline_scheduler": plan.get("baseline_scheduler", "distributed"),
               "baseline_chunk_size": plan.get("baseline_chunk_size", 32),
               "baseline_workers_per_gpu": plan.get("baseline_workers_per_gpu", 1),
               "tasks": sorted(tasks), "active_checkpoints": len(cfg["teachers"]), "active_dimensions": active}
    summary["validation"] = {"before_train": plan.get("evaluate", True) and plan.get("val_before_train", True),
                              "every_optimizer_steps": cfg["train"]["eval_every"],
                              "limit_per_task": cfg["train"]["eval_limit_per_task"],
                              "after_train": plan.get("evaluate", True)}
    if plan.get("recipe_alignment"):
        summary["recipe_alignment"] = plan["recipe_alignment"]
    if cfg["qgpi"]["evaluator"] == "opd.objective:objective_candidates":
        from .objective import METRICS
        summary.update(requires_llm_judge=False, task_metrics={t: METRICS[t] for t in tasks},
                       uncovered_dimensions=[d for d in ("F", "S", "U", "T", "N") if d not in active],
                       scope="T=termination boundary; N=first-turn lexical proxy; not full sequential/narrative coherence or realism")
    atomic_json(runner.root / "plan_summary.json", summary)
    atomic_json(runner.root / "commands.json", runner.commands)
    if not dry_run:
        atomic_json(runner.root / "results.json", results)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["calibrate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative (0=all)")
    calibrate_quality(load_config(args.config), args.output, args.limit)


if __name__ == "__main__":
    main()
