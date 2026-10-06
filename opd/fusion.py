"""Continue completed F/S/N experts into one on-policy student and a standalone model."""

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

from .checkpoints import atomic_json
from .config import load_config, validate
from .data import read_rows
from .experiments import Runner, fingerprint


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_fusion(source, output, *, seed=42, student_mode="full", steps=200,
                   num_processes=8, micro_batch=1, global_batch=32, learning_rate=None,
                   save_every=50, eval_every=50, inference_batch=4, base_model=None,
                   evaluate=True):
    """Read the completed run, preserving its task split, tokenizer and common base."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Fusion output must be separate from the source dimension run")
    for key, value in {"steps": steps, "num_processes": num_processes, "micro_batch": micro_batch,
                       "global_batch": global_batch, "save_every": save_every,
                       "inference_batch": inference_batch}.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if global_batch % (num_processes * micro_batch):
        raise ValueError("global_batch must be divisible by num_processes * micro_batch")
    manifest = json.loads((source / "manifest.json").read_text())
    source_plan = manifest["plan"]
    if source_plan.get("workflow") != "dimension_offline" or set(source_plan["dimensions"]) != {"F", "S", "N"}:
        raise ValueError("Expected a completed dimension_offline run containing F, S and N")
    results = json.loads((source / "results.json").read_text())[f"seed_{seed}"]
    state = json.loads((source / "stages.json").read_text())
    cfg = load_config(source / "configs/preflight.yaml")
    original_base = cfg["model"]["base_model"]
    source_tasks = set(cfg["routing"]["tasks"])
    cfg["seed"] = seed
    cfg["model"].update(student_mode=student_mode, base_model=str(Path(base_model).resolve()) if base_model else original_base)
    cfg["model"].pop("student_init", None)  # Start the single student from the shared base.
    cfg["qgpi"].update(enabled=False, registry_file=None)
    cfg["routing"] = {"tasks": {}}
    cfg["teachers"] = []
    task_dimensions, references, inputs = {}, {}, {}
    for dim in ("F", "S", "N"):
        stage = f"seed_{seed}/dimension_{dim}/offline_forward"
        if state.get(stage, {}).get("status") != "complete":
            raise ValueError(f"Source teacher training is incomplete: {stage}")
        final = source / "runs" / stage / "final"
        metadata = json.loads((final / "opd_metadata.json").read_text())
        if (metadata.get("dimension") != dim or metadata.get("student_mode") != "lora"
                or metadata.get("base_model") != original_base):
            raise ValueError(f"Teacher {dim} has incompatible dimension/base/adapter metadata")
        for filename in ("adapter_config.json", "adapter_model.safetensors", "opd_metadata.json"):
            path = final / filename
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"Missing completed teacher artifact: {path}")
            inputs[str(path)] = file_hash(path)
        cfg["teachers"].append({"id": "dim_" + dim, "adapter": str(final)})
        reference = source / "evaluation" / (stage + ".jsonl")
        if reference.is_file():
            inputs[str(reference)] = file_hash(reference)
        if results[dim].get("steps", 0) < 1 or not results[dim].get("tasks"):
            raise ValueError(f"Teacher {dim} has no recorded training budget/tasks")
        for task in results[dim]["tasks"]:
            if task in task_dimensions:
                raise ValueError(f"Ambiguous dimension for task {task}")
            task_dimensions[task] = dim
            cfg["routing"]["tasks"][task] = {"teachers": {"dim_" + dim: 1.0}, "strength": 1.0}
            if reference.is_file():
                references[task] = str(reference)
    if set(task_dimensions) != source_tasks:
        raise ValueError("Dimension teachers do not cover exactly the original tasks")
    counts = {}
    # Use original length-audited TRAIN prompts, never selected teacher responses.
    # Validation is the same fixed subset used for the completed dimension reports.
    for split, path in (("train", source / "tokenized_data/train.jsonl"),
                        ("eval", source / "eval_data" / f"seed_{seed}.jsonl")):
        count = Counter()
        for row in read_rows(path):
            task = row["task_id"]
            if task not in task_dimensions or row["dimensions"] != [task_dimensions[task]]:
                raise ValueError(f"Unexpected task/dimension in {path}: {row['id']}")
            count[task] += 1
        if set(count) != set(task_dimensions):
            raise ValueError(f"{split} data is missing a source task")
        counts[split] = dict(count)
        cfg["data"][split + "_file"] = str(path)
        inputs[str(path)] = file_hash(path)
    cfg["data"].update(dimension=None, demo_file=None)
    cfg["data"].pop("calibration_file", None)
    cfg["data"]["task_weights"] = {t: cfg["data"]["task_weights"].get(t, 1.0) for t in task_dimensions}
    cfg["train"].update(stage="opd", objective="forward_kl", trajectory_source="student",
        max_steps=steps, batch_size=micro_batch, global_prompt_batch=global_batch,
        gradient_accumulation_steps=global_batch // (num_processes * micro_batch),
        learning_rate=learning_rate if learning_rate is not None else (2e-6 if student_mode == "full" else 1e-5),
        save_every=save_every, eval_every=eval_every, anchor_coef=0.0, sft_coef=0.0, resume_from=None)
    cfg["inference"]["batch_size"] = inference_batch
    cfg["output_dir"] = str(output / "unused")
    validate(cfg)
    launch = None
    if num_processes > 1:
        launch = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/accelerate_zero2.yaml").read_text())
        launch["num_processes"] = num_processes
    plan = {"workflow": "dimension_fusion", "output_dir": str(output), "source_run": str(source),
            "inference_num_processes": num_processes, "num_processes": num_processes,
            "evaluate": evaluate, "source_seed": seed, "launch": launch}
    return {"plan": plan, "config": cfg, "input_hashes": inputs, "task_dimensions": task_dimensions,
            "data_counts": counts, "references": references, "source_manifest_sha256": file_hash(source / "manifest.json"),
            "scope": "F/S/N from the completed source run; N remains a first-turn lexical proxy. T/U/C are not added."}


def run_fusion(prepared, *, resume=False, dry_run=False):
    from .full_export import complete_export
    plan, cfg = copy.deepcopy(prepared["plan"]), copy.deepcopy(prepared["config"])
    root = Path(plan["output_dir"])
    signature = fingerprint(prepared)
    manifest = root / "fusion_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["signature"] != signature:
            raise ValueError("Fusion settings/source artifacts changed; use a new output directory")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError("Fusion output is nonempty and has no matching manifest")
    runner = Runner(plan, resume=resume, dry_run=dry_run)
    atomic_json(manifest, {"signature": signature, **prepared})
    if plan["launch"]:
        plan["accelerate_config"] = str(runner.config("accelerate", plan["launch"]))
    stage = f"seed_{cfg['seed']}/unified_opd"
    # Runner supports a process-count environment override; keep it consistent
    # with the validated global batch even in a shell used for an older job.
    previous = os.environ.get("OPD_NUM_PROCESSES")
    os.environ["OPD_NUM_PROCESSES"] = str(plan["num_processes"])
    try:
        final, trained_cfg = runner.train(stage, cfg)
    finally:
        if previous is None:
            os.environ.pop("OPD_NUM_PROCESSES", None)
        else:
            os.environ["OPD_NUM_PROCESSES"] = previous
    export = root / "full_model"
    export_record = runner.state.get("export/full_model")
    if resume and export_record and export_record["status"] == "complete" and not complete_export(export):
        export_record["status"] = "failed"
    runner.command("export/full_model", [sys.executable, "-m", "opd.full_export",
        "--model", final, "--base", cfg["model"]["base_model"], "--output", export,
        "--dtype", cfg["model"]["dtype"], *(["--resume"] if resume else [])],
        [export / "export_manifest.json"], {"source": final, "fusion_signature": signature})
    report = {"workflow": "dimension_fusion", "dry_run": dry_run, "student_mode": cfg["model"]["student_mode"],
              "base_model": cfg["model"]["base_model"], "trajectory_source": "student", "objective": "forward_kl",
              "steps": cfg["train"]["max_steps"], "task_dimensions": prepared["task_dimensions"],
              "data_counts": prepared["data_counts"], "training_checkpoint": final,
              "full_model": str(export), "scope": prepared["scope"],
              "batch": {"global": cfg["train"]["global_prompt_batch"], "ranks": plan["num_processes"],
                        "micro": cfg["train"]["batch_size"], "accumulation": cfg["train"]["gradient_accumulation_steps"]}}
    if plan["evaluate"]:
        # Evaluate the exact standalone deployment artifact, including BF16 merge rounding.
        evaluation_cfg = copy.deepcopy(trained_cfg)
        evaluation_cfg["model"]["student_mode"] = "full"
        evaluation = runner.evaluate(stage, evaluation_cfg, model=str(export))
        report["evaluation"] = evaluation
        if not dry_run and prepared["references"]:
            from .retention import compare, save_report
            path = root / "retention/against_dimension_teachers.json"
            save_report(path, compare(evaluation, prepared["references"], seed=cfg["seed"]))
            report["retention"] = str(path)
    atomic_json(root / ("plan_summary.json" if dry_run else "results.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Completed dimension_experts_* run root")
    parser.add_argument("--output", required=True, help="New, separate unified student run root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-mode", choices=["full", "lora"], default="full")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--inference-batch", type=int, default=4)
    parser.add_argument("--base-model", help="Relocated identical common base; otherwise use source metadata")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip final evaluation; periodic checks use --eval-every")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Check source artifacts and write commands without loading models")
    args = vars(parser.parse_args())
    resume, dry_run = args.pop("resume"), args.pop("dry_run")
    args["evaluate"] = not args.pop("no_evaluate")
    prepared = prepare_fusion(**args)
    upstream = json.loads((Path(args["source"]) / "manifest.json").read_text())["plan"].get("upstream_repo")
    if upstream:
        os.environ["SIMULATION_REPO"] = upstream
    if not dry_run:
        import torch
        cpu = os.environ.get("ACCELERATE_USE_CPU", "").lower() == "true"
        if not cpu and (not torch.cuda.is_available() or torch.cuda.device_count() < args["num_processes"]):
            raise RuntimeError("Insufficient visible GPUs for the configured training ranks")
    print(json.dumps(run_fusion(prepared, resume=resume, dry_run=dry_run), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
