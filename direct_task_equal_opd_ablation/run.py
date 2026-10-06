#!/usr/bin/env python3
"""Small direct OPD ablation with uniformly weighted task experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from opd.checkpoints import atomic_json
from opd.config import load_config, validate
from opd.data import read_rows, write_rows
from opd.expert_manifest import checkpoint_candidates, load_manifest
from opd.experiments import Runner, fingerprint


def subset_by_task(path: Path, tasks: list[str], limit: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in read_rows(path):
        if row["task_id"] in tasks:
            groups[row["task_id"]].append(row)
    missing = set(tasks) - set(groups)
    if missing:
        raise ValueError(f"No source rows for tasks: {sorted(missing)}")
    rng = random.Random(seed)
    selected = []
    for task in tasks:
        rows = groups[task]
        indices = sorted(rng.sample(range(len(rows)), min(limit, len(rows))))
        selected.extend(rows[i] for i in indices)
    return selected


def prepare(args: argparse.Namespace) -> tuple[dict, dict, Path]:
    data_source, output = args.data_source.resolve(), args.output.resolve()
    manifest_path = args.expert_manifest.resolve()
    if data_source == output or data_source in output.parents or output in data_source.parents:
        raise ValueError("Output must be separate from the source data run")
    for name in ("steps", "train_per_task", "validation_per_task", "num_processes",
                 "micro_batch", "global_batch", "save_every", "max_new_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.global_batch % (args.num_processes * args.micro_batch):
        raise ValueError("global_batch must divide evenly across ranks and micro-batches")

    cfg = load_config(data_source / "configs" / "preflight.yaml")
    source_data = data_source / "hierarchy_data" / f"seed_{args.seed}"
    raw_manifest = yaml.safe_load(manifest_path.read_text())
    tasks = [entry["task"] for entry in raw_manifest["tasks"]]
    experts, manifest_info = load_manifest(manifest_path, tasks)
    selected_ids = []
    selected_teachers = []
    for expert in experts:
        candidates = checkpoint_candidates(expert)
        if len(candidates) != 1 or "adapter" not in candidates[0]:
            raise ValueError(f"Expected one enabled LoRA checkpoint for {expert['task']}")
        teacher = candidates[0]
        teacher_id = teacher["id"]
        adapter = Path(teacher["adapter"])
        if not (adapter / "adapter_config.json").is_file() or not any(
            (adapter / file).is_file() for file in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            raise FileNotFoundError(f"Missing adapter config or weights: {adapter}")
        selected_ids.append(teacher_id)
        selected_teachers.append(teacher)
    if len(set(selected_ids)) != len(tasks):
        raise ValueError("Each task must have a distinct expert for this ablation")

    uniform = {teacher_id: 1.0 / len(selected_ids) for teacher_id in selected_ids}
    cfg["teachers"] = selected_teachers
    cfg["routing"]["tasks"] = {
        task: {"teachers": dict(uniform), "strength": 1.0} for task in tasks
    }
    cfg["data"]["task_weights"] = {task: 1.0 for task in tasks}
    cfg["data"]["dimension"] = None
    cfg["data"]["demo_file"] = None
    cfg["data"].pop("calibration_file", None)
    cfg["qgpi"].update(enabled=False, registry_file=None)
    cfg["quality"].update(evaluator=None, filter_demos=False)
    cfg["model"].update(student_mode="full")
    cfg["model"].pop("student_init", None)
    cfg["rollout"].update(max_new_tokens=args.max_new_tokens)
    cfg["train"].update(
        stage="opd", objective="forward_kl", trajectory_source="student",
        max_steps=args.steps, batch_size=args.micro_batch,
        global_prompt_batch=args.global_batch,
        gradient_accumulation_steps=args.global_batch // (args.num_processes * args.micro_batch),
        learning_rate=args.learning_rate, warmup_steps=min(10, args.steps // 10),
        save_every=args.save_every, eval_every=0, anchor_coef=0.0, sft_coef=0.0,
        resume_from=None,
    )
    cfg["seed"] = args.seed
    cfg["output_dir"] = str(output / "unused")

    subsets = {}
    for split, limit in (("train", args.train_per_task),
                         ("validation", args.validation_per_task)):
        rows = subset_by_task(source_data / f"{split}.jsonl", tasks, limit, args.seed)
        subsets[split] = rows
    identity = {
        "data_source": str(data_source), "expert_manifest": str(manifest_path),
        "expert_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "checkpoint_root": manifest_info["checkpoint_root"],
        "seed": args.seed, "teacher_ids": selected_ids,
        "teacher_adapters": {teacher["id"]: teacher["adapter"] for teacher in selected_teachers},
        "teacher_weights": uniform, "tasks": tasks,
        "source_sha256": {
            split: hashlib.sha256((source_data / f"{split}.jsonl").read_bytes()).hexdigest()
            for split in subsets
        },
        "settings": {name: getattr(args, name) for name in (
            "steps", "train_per_task", "validation_per_task", "num_processes",
            "micro_batch", "global_batch", "learning_rate", "save_every",
            "max_new_tokens", "evaluate")},
        "objective": "forward_kl", "trajectory_source": "student",
    }
    manifest = output / "ablation_manifest.json"
    signature = fingerprint(identity)
    if manifest.exists():
        if json.loads(manifest.read_text())["signature"] != signature:
            raise ValueError("Ablation inputs/settings changed; choose a new output directory")
        if not args.resume:
            raise FileExistsError("Ablation already exists; pass --resume")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is nonempty and has no ablation manifest")
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in subsets.items():
        write_rows(output / "data" / f"{split}.jsonl", rows)
    cfg["data"].update(train_file=str(output / "data" / "train.jsonl"),
                       eval_file=str(output / "data" / "validation.jsonl"))
    validate(cfg)
    atomic_json(manifest, {"signature": signature, **identity,
                           "row_counts": {split: len(rows) for split, rows in subsets.items()}})
    return cfg, identity, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", required=True, type=Path,
                        help="Existing hierarchy run supplying prepared training prompts")
    parser.add_argument("--expert-manifest", type=Path,
                        default=PROJECT / "configs" / "task_experts.yaml",
                        help="Canonical task expert checkpoint list")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-task", type=int, default=64)
    parser.add_argument("--validation-per-task", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--global-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--evaluate", action="store_true", help="Run validation after training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg, identity, output = prepare(args)
    plan = {"output_dir": str(output), "num_processes": args.num_processes,
            "inference_num_processes": args.num_processes, "evaluation_split": "validation"}
    runner = Runner(plan, resume=args.resume, dry_run=args.dry_run)
    if args.num_processes > 1:
        launch = yaml.safe_load((PROJECT / "configs" / "accelerate_zero2.yaml").read_text())
        launch["num_processes"] = args.num_processes
        plan["accelerate_config"] = str(runner.config("accelerate", launch))
    os.environ["OPD_NUM_PROCESSES"] = str(args.num_processes)
    final, trained = runner.train("direct_equal_forward_opd", cfg)
    report = {"checkpoint": final, "objective": "forward_kl", "trajectory_source": "student",
              "teacher_count": len(identity["teacher_ids"]),
              "teacher_weight": 1.0 / len(identity["teacher_ids"]),
              "steps": args.steps, "dry_run": args.dry_run}
    if args.evaluate:
        report["evaluation"] = runner.evaluate("direct_equal_forward_opd", trained, model=final)
    atomic_json(output / ("plan_summary.json" if args.dry_run else "results.json"), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
