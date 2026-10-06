"""Objective-only first phase: evaluate all checkpoint versions and build a pool.

There is deliberately no optimizer / dimension-LoRA / unified-student training
in this phase. Existing task adapters remain the experts, not new claims of F/S
disentanglement. Later distillation must be selected explicitly.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

from .checkpoints import atomic_json
from .data import load_data, write_rows
from .experiments import Runner, base_config, fingerprint, tokenized_data
from .objective import TASK_AXES, METRICS, validate_objective_config


def run_expert_pool(plan, resume=False, dry_run=False):
    runner = Runner(plan, resume, dry_run)
    data = Path(plan.get("prepared_data") or runner.root / "data")
    base = base_config(plan, data)
    if base["qgpi"]["evaluator"] != "opd.objective:objective_candidates":
        raise ValueError("The objective expert pool requires the strict no-judge evaluator")
    axes = validate_objective_config(base)
    provenance = {"plan": plan, "data": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(data.glob("*.jsonl"))}}
    manifest = runner.root / "manifest.json"
    signature = fingerprint(provenance)
    if manifest.exists() and json.loads(manifest.read_text())["signature"] != signature:
        raise ValueError("Expert-pool plan/data changed; use a new output directory")
    atomic_json(manifest, {"signature": signature, **provenance})
    if plan.get("expert_inventory"):
        atomic_json(runner.root / "expert_inventory.json", plan["expert_inventory"])
    if not dry_run:
        data = tokenized_data(base, data, runner)
        base = base_config(plan, data)
    coverage = {}
    for split in ("train", "validation", "calibration", "eval"):
        rows = load_data(data / (split + ".jsonl"))
        coverage[split] = {t: sum(r["task_id"] == t for r in rows) for t in base["routing"]["tasks"]}
        if split in ("train", "calibration", plan.get("evaluation_split", "validation")):
            if any(count == 0 for count in coverage[split].values()):
                raise ValueError(f"Objective pool {split} lost an entire task; inspect length/group audit")
    atomic_json(runner.root / "objective_coverage.json", {"rows": coverage, "task_axes": {
        t: TASK_AXES[t] for t in base["routing"]["tasks"]}, "active_dimensions": axes,
        "requires_llm_judge": False, "note": "No synthetic U/T/N coverage is inferred from F/S task metrics."})
    evaluation_rows = load_data(base["data"]["eval_file"])
    pool_files = []
    for seed in plan.get("seeds", [42]):
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        stem = f"seed_{seed}"
        baseline = runner.evaluate(stem + "/base", cfg, model=cfg["model"]["base_model"])
        evaluations = {}
        for task, route in cfg["routing"]["tasks"].items():
            subset = runner.root / "eval_data" / f"{task}.jsonl"
            write_rows(subset, [r for r in evaluation_rows if r["task_id"] == task])
            task_cfg = copy.deepcopy(cfg)
            task_cfg["data"]["eval_file"] = str(subset)
            evaluations[task] = {}
            for teacher in route["teachers"]:
                evaluations[task][teacher] = runner.evaluate(stem + "/teacher/" + teacher, task_cfg, teacher=teacher)
        config_file = runner.config(stem + "/calibration", cfg)
        registry = runner.root / "calibration" / stem / "registry.json"
        limit = plan.get("calibration_limit_per_task", 64)
        runner.command("calibration/" + stem, [sys.executable, "-m", "opd.qgpi_workflow", "calibrate",
                       "--config", config_file, "--output", registry, "--limit", limit],
                       [registry, str(registry) + ".details.jsonl"], {"config": cfg, "limit": limit})
        cfg["qgpi"]["registry_file"] = str(registry)
        cfg["output_dir"] = str(runner.root / "future_training" / stem)
        # Explicit opt-in for the next stage; the pool builder never invokes it.
        next_config = runner.config(stem + "/optional_unified_student", cfg)
        pool_file = runner.root / "pools" / stem / "expert_pool.json"
        pool_files.append(str(pool_file))
        if not dry_run:
            calibrated = json.loads(registry.read_text())
            recommendations = {}
            for task, versions in calibrated["tasks"].items():
                reliable = {teacher: info for teacher, info in versions.items() if info["status"] == "estimated"
                            and info["lower_gain"] > cfg["qgpi"]["min_gain"]}
                recommendations[task] = max(reliable, key=lambda t: reliable[t]["lower_gain"]) if reliable else None
            atomic_json(pool_file, {"schema": "objective-expert-pool-v1", "base_model": cfg["model"]["base_model"],
                        "teachers": cfg["teachers"], "routing": cfg["routing"], "registry_file": str(registry),
                        "active_dimensions": axes, "task_metrics": {t: METRICS[t] for t in evaluations},
                        "evaluation": evaluations, "base_evaluation": baseline,
                        "recommended_by_calibration": recommendations, "optional_student_config": str(next_config),
                        "updated_parameters": False, "requires_llm_judge": False,
                        "note": "No recommendation means insufficient positive calibration evidence, not expert success. All checkpoint versions remain traceable; test split did not choose versions."})
    summary = {"workflow": "expert_pool", "tasks": list(base["routing"]["tasks"]),
               "active_checkpoints": len(base["teachers"]), "active_dimensions": axes,
               "requires_llm_judge": False, "optimizer_steps": 0, "dry_run": dry_run,
               "pending_commands": len(runner.commands), "pools": pool_files,
               "training_order": ["reuse_existing_task_adapters", "objective_baselines_all_versions",
                                  "heldout_calibration", "expert_pool_export"],
               "next_stage": "Explicitly choose dimension-expert distillation or unified-student training."}
    atomic_json(runner.root / "plan_summary.json", summary)
    atomic_json(runner.root / "commands.json", runner.commands)
    return summary
