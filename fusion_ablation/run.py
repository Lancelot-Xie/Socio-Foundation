"""Four final-fusion ablations, reusable from an existing hierarchy run."""

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import yaml

METHODS = ("dimension_offpolicy", "task_sft", "merge_dimensions", "merge_tasks")
PROJECT = Path(__file__).resolve().parents[1]


def source_config(source, method, seed, runner):
    from opd.config import load_config
    from opd.experiments import final_config

    source = Path(source).resolve()
    manifest = json.loads((source / "hierarchy_manifest.json").read_text())
    plan = manifest["plan"]
    if plan.get("workflow") != "hierarchy_five" or seed not in plan["seeds"]:
        raise ValueError("Expected a hierarchy_five run containing the requested seed")
    cfg = load_config(source / "configs/preflight.yaml")
    data = source / "hierarchy_data" / f"seed_{seed}"
    # Reuse all length-audited training prompts and the exact bounded validation set.
    for split in ("train", "validation"):
        if not (data / f"{split}.jsonl").is_file():
            raise FileNotFoundError(data / f"{split}.jsonl")
    cfg["data"].update(train_file=str(data / "train.jsonl"), eval_file=str(data / "validation.jsonl"))
    cfg["data"].pop("calibration_file", None)
    cfg["train"].update(plan["fusion"])
    if method in ("dimension_offpolicy", "merge_dimensions"):
        models = {}
        stages = json.loads((source / "stages.json").read_text())
        for dim in plan["dimensions"]:
            stage = f"seed_{seed}/dimension_{dim}/offline_forward"
            if stages.get(stage, {}).get("status") != "complete":
                raise ValueError(f"Dimension expert is not complete: {stage}")
            path = source / "runs" / stage / "final"
            metadata = json.loads((path / "opd_metadata.json").read_text())
            if (metadata.get("dimension") != dim or metadata.get("student_mode") != "lora"
                    or metadata.get("base_model") != cfg["model"]["base_model"]):
                raise ValueError(f"Dimension expert metadata mismatch: {path}")
            models[dim] = str(path)
        cfg = final_config(cfg, models, data, runner, f"seed_{seed}", "dimensions")
    return cfg, plan.get("upstream_repo")


def training_config(cfg, method, demos):
    cfg = copy.deepcopy(cfg)
    if method == "dimension_offpolicy":
        cfg["train"].update(stage="opd", trajectory_source="teacher", objective="forward_kl")
        cfg["data"]["demo_file"] = str(demos)
    elif method == "task_sft":
        cfg["train"].update(stage="warmup", trajectory_source="teacher")
        cfg["data"].update(train_file=str(demos), demo_file=None)
    else:
        raise ValueError("Not a training ablation")
    return cfg


def run(args):
    from opd.checkpoints import atomic_json
    from opd.config import load_config, validate
    from opd.experiments import Runner, fingerprint
    from opd.fusion import file_hash
    from .merge import normalized_weights, validate_adapters

    for key in ("steps", "num_processes", "micro_batch", "global_batch", "candidates", "save_every"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.eval_every < 0 or args.global_batch % (args.num_processes * args.micro_batch):
        raise ValueError("Invalid eval frequency or global batch / ranks / microbatch")
    if not math.isfinite(args.merge_scale) or args.merge_scale < 0:
        raise ValueError("merge-scale must be finite and nonnegative")
    if not args.method.startswith("merge_") and (args.weights or args.merge_scale != 1.0):
        raise ValueError("--weights and --merge-scale apply only to model merge methods")
    root = Path(args.output).resolve()
    if args.source:
        source = Path(args.source).resolve()
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("Ablation output must be separate from the original hierarchy run")
    manifest_path = root / "ablation_manifest.json"
    if root.exists() and any(root.iterdir()) and not manifest_path.exists():
        raise FileExistsError("Nonempty output without ablation manifest; use a new directory")
    plan = {"output_dir": str(root), "num_processes": args.num_processes,
            "inference_num_processes": args.num_processes, "evaluation_split": "validation"}
    runner = Runner(plan, args.resume, args.dry_run)
    if args.config:
        cfg, upstream = load_config(args.config), None
    else:
        cfg, upstream = source_config(args.source, args.method, args.seed, runner)
    if upstream:
        os.environ.setdefault("SIMULATION_REPO", upstream)
    if cfg["rollout"]["environment_factory"] or cfg["rollout"]["max_turns"] != 1:
        raise ValueError("These hierarchy ablations require single-turn static prompts")
    cfg["seed"] = args.seed
    cfg["model"]["student_mode"] = "full"
    cfg["model"].pop("student_init", None)
    cfg["data"].update(dimension=None, demo_file=None)
    cfg["qgpi"].update(enabled=False, registry_file=None)
    cfg["train"].update(stage="opd", objective="forward_kl", trajectory_source="student",
        max_steps=args.steps, batch_size=args.micro_batch, global_prompt_batch=args.global_batch,
        gradient_accumulation_steps=args.global_batch // (args.num_processes * args.micro_batch),
        anchor_coef=0.0, sft_coef=0.0, resume_from=None, save_every=args.save_every,
        eval_every=args.eval_every)
    if args.learning_rate is not None:
        cfg["train"]["learning_rate"] = args.learning_rate
    cfg["output_dir"] = str(root / "unused")
    validate(cfg)
    # Check actual adapter weights even for dry-run; no model/GPU/API is loaded.
    paths = validate_adapters(cfg)
    hashes = {str(p / name): file_hash(p / name) for p in paths
              for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
              if (p / name).is_file()}
    for field in ("train_file", "eval_file"):
        if cfg["data"].get(field):
            hashes[cfg["data"][field]] = file_hash(cfg["data"][field])
    weights = json.loads(Path(args.weights).read_text()) if args.weights else None
    effective_weights = normalized_weights(cfg["teachers"], weights)
    identity = {"method": args.method, "config": cfg, "inputs": hashes,
        "weights": effective_weights, "merge_scale": args.merge_scale, "candidates": args.candidates,
        "num_processes": args.num_processes, "evaluate": not args.no_evaluate}
    signature = fingerprint(identity)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["signature"] != signature:
            raise ValueError("Ablation inputs/settings changed; use a new output directory")
        if not args.resume:
            raise FileExistsError("Ablation already planned or started; use --resume")
    atomic_json(manifest_path, {"signature": signature, **identity})
    if args.num_processes > 1:
        launch = yaml.safe_load((PROJECT / "configs/accelerate_zero2.yaml").read_text())
        launch["num_processes"] = args.num_processes
        plan["accelerate_config"] = str(runner.config("accelerate", launch))
    os.environ["OPD_NUM_PROCESSES"] = str(args.num_processes)
    cfg_path = runner.config("teachers", cfg)
    export = root / "full_model"
    if args.method.startswith("merge_"):
        settings = root / "merge_weights.json"
        atomic_json(settings, effective_weights)
        command = [sys.executable, "-m", "fusion_ablation", "merge", "--config", cfg_path,
            "--output", export, "--weights", settings, "--scale", str(args.merge_scale)]
        if args.resume:
            command.append("--resume")
        runner.command("merge", command, [export / "export_manifest.json"], identity)
        trained_cfg = cfg
    else:
        demos = root / "demos/teacher.jsonl"
        runner.command("teacher_demos", [sys.executable, "-m", "fusion_ablation", "build-demos",
            "--config", cfg_path, "--output", demos, "--candidates", str(args.candidates)],
            [demos, str(demos) + ".report.json"], {"config": cfg, "candidates": args.candidates, "inputs": hashes})
        training = training_config(cfg, args.method, demos)
        final, trained_cfg = runner.train("final", training)
        runner.command("export", [sys.executable, "-m", "opd.full_export", "--model", final,
            "--base", cfg["model"]["base_model"], "--output", export, "--dtype", cfg["model"]["dtype"],
            *(["--resume"] if args.resume else [])], [export / "export_manifest.json"], identity)
    report = {"method": args.method, "full_model": str(export), "dry_run": args.dry_run,
              "training_steps": 0 if args.method.startswith("merge_") else args.steps}
    if not args.no_evaluate:
        report["evaluation"] = runner.evaluate("final", trained_cfg, model=str(export))
    atomic_json(root / ("plan_summary.json" if args.dry_run else "results.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("run")
    p.add_argument("--method", choices=METHODS, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Original hierarchy run root; reuse completed experts")
    source.add_argument("--config", help="Explicit standard OPD config identifying teachers/data/base")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--global-batch", type=int, default=32)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--candidates", type=int, default=1)
    p.add_argument("--weights", help="JSON teacher ID -> nonnegative merge weight; default equal checkpoint weights")
    p.add_argument("--merge-scale", type=float, default=1.0)
    p.add_argument("--no-evaluate", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    for name in ("build-demos", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        if name == "build-demos":
            p.add_argument("--candidates", type=int, default=1)
        else:
            p.add_argument("--weights")
            p.add_argument("--scale", type=float, default=1.0)
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        result = run(args)
    else:
        from opd.config import load_config
        cfg = load_config(args.config)
        if args.action == "build-demos":
            from .demos import build
            result = build(cfg, args.output, args.candidates)
        else:
            from .merge import merge
            weights = json.loads(Path(args.weights).read_text()) if args.weights else None
            result = merge(cfg, args.output, weights, args.scale, args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False))
