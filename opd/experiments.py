"""Sequential, resumable experiment matrix with explicit per-stage configurations."""

import argparse
import copy
import contextlib
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .config import merge, DEFAULTS, validate
from .data import load_data, read_rows, write_rows
from .source_data import EXPORTER_VERSION, canonical_task, prepare_sources
from .checkpoints import archive_incomplete, atomic_json, latest_checkpoint


METHODS = ("sft", "offline_forward", "opd_forward", "opd_reverse", "sft_forward", "sft_reverse",
           "mixed_forward", "iterative_sft")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def expanded(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: expanded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expanded(v) for v in value]
    return value


def load_plan(path):
    path = Path(path).resolve()
    plan = expanded(yaml.safe_load(path.read_text()))
    if not isinstance(plan, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    objective = (plan.get("quality", {}).get("evaluator") == "opd.objective:objective_response" and
                 (plan.get("workflow") == "dimension_offline" or
                  plan.get("qgpi", {}).get("evaluator") == "opd.objective:objective_candidates"))
    if os.environ.get("OPD_REQUIRE_OBJECTIVE") == "1" and not objective:
        raise ValueError("train_objective.sh requires the strict no-judge objective evaluator; check OPD_CONFIG/env file")
    if plan.get("workflow", "legacy_matrix") not in ("qgpi", "legacy_matrix", "expert_pool", "dimension_offline"):
        raise ValueError("workflow must be qgpi, expert_pool, dimension_offline or legacy_matrix")
    if objective:
        plan.setdefault("data", {})["objective_only"] = True
        from .objective import backend_hashes
        plan["objective_backend_hashes"] = backend_hashes()
    if plan.get("expert_manifest"):
        from .expert_manifest import load_manifest
        if plan.get("experts"):
            raise ValueError("Choose expert_manifest+tasks OR an inline experts list")
        manifest = (path.parent / plan["expert_manifest"]).resolve()
        plan["expert_manifest"] = str(manifest)
        plan["experts"], plan["expert_inventory"] = load_manifest(manifest, plan.get("tasks", []))
    for variable, field in (("SIMULATION_REPO", "upstream_repo"), ("OPD_DATA_ROOT", "data_root"),
                            ("OPD_OUTPUT_ROOT", "output_dir"), ("OPD_ACCELERATE_CONFIG", "accelerate_config")):
        if os.environ.get(variable):
            plan[field] = str(Path(os.environ[variable]).expanduser().resolve())
    for field in ("upstream_repo", "data_root", "output_dir", "accelerate_config", "fixture_config", "prepared_data"):
        if plan.get(field):
            plan[field] = str((path.parent / plan[field]).resolve())
    from .expert_recipe import apply_expert_recipe
    apply_expert_recipe(plan)
    if plan.get("baseline_scheduler", "distributed") not in ("distributed", "task_pool"):
        raise ValueError("baseline_scheduler must be distributed or task_pool")
    if type(plan.get("baseline_chunk_size", 32)) is not int or plan.get("baseline_chunk_size", 32) < 1:
        raise ValueError("baseline_chunk_size must be a positive integer")
    if type(plan.get("baseline_workers_per_gpu", 1)) is not int or plan.get("baseline_workers_per_gpu", 1) < 1:
        raise ValueError("baseline_workers_per_gpu must be a positive integer")
    checkpoint_items = [c for e in plan.get("experts", []) for c in e.get("checkpoints", [])]
    for item in [plan.get("model", {})] + plan.get("experts", []) + checkpoint_items:
        for field in ("base_model", "student_init", "adapter", "model", "tokenizer"):
            value = item.get(field)
            if value and "$" not in value and (field == "adapter" or value.startswith((".", "/"))
                                                or (path.parent / value).exists()):
                item[field] = str((path.parent / value).resolve())
    if plan.get("qgpi", {}).get("registry_file"):
        plan["qgpi"]["registry_file"] = str((path.parent / plan["qgpi"]["registry_file"]).resolve())
    if not plan.get("experts") and not plan.get("fixture_config"):
        raise ValueError("An explicit expert manifest is required")
    methods = plan.setdefault("methods", ["offline_forward"] if plan.get("workflow") == "dimension_offline" else list(METHODS))
    if not methods or set(methods) - set(METHODS) or len(set(methods)) != len(methods):
        raise ValueError(f"methods must be unique members of {METHODS}")
    plan.setdefault("seeds", [42])
    plan.setdefault("lora_ranks", [plan["model"]["lora_rank"]] if plan.get("workflow") == "dimension_offline" else [64])
    if not plan["lora_ranks"] or len(set(plan["lora_ranks"])) != len(plan["lora_ranks"]) or any(type(r) is not int or r < 1 for r in plan["lora_ranks"]):
        raise ValueError("lora_ranks must be positive integers")
    if not plan["seeds"] or len(set(plan["seeds"])) != len(plan["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    default_steps = {"distill": 200} if plan.get("workflow") == "dimension_offline" else {
        "sft": 1200, "warmup": 200, "distill": 1000, "final": 2000}
    plan["steps"] = {**default_steps, **plan.get("steps", {})}
    if any(type(v) is not int or v < 1 for v in plan["steps"].values()):
        raise ValueError("All stage step counts must be positive")
    if plan.get("evaluation_split", "validation") not in ("validation", "eval"):
        raise ValueError("evaluation_split must be validation or eval")
    tasks = [canonical_task(e["task"]) for e in plan.get("experts", [])]
    if len(tasks) != len(set(tasks)):
        raise ValueError("Expert tasks must be unique after alias normalization")
    if plan.get("workflow") == "dimension_offline":
        from .dimension_workflow import validate_plan
        validate_plan(plan)
    return plan


class Runner:
    def __init__(self, plan, resume=False, dry_run=False):
        self.plan, self.resume, self.dry_run = plan, resume, dry_run
        self.root = Path(plan["output_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "stages.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.commands = []
        self.visited = set()

    def write(self, path, content):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)

    def config(self, name, cfg):
        path = self.root / "configs" / (name + ".yaml")
        self.write(path, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        return path

    def command(self, name, command, artifacts, signature):
        artifacts = [Path(p) for p in artifacts]
        key = fingerprint(signature)
        record = self.state.get(name)
        if record and record["signature"] != key:
            raise ValueError(f"Stage configuration changed: {name}. Use a new output directory")
        if record and record["status"] == "complete" and all(p.exists() for p in artifacts):
            if not self.resume and name not in self.visited:
                raise FileExistsError(f"Stage exists: {name}; use --resume")
            print(f"[retained] {name}", flush=True)
            return
        self.commands.append({"stage": name, "command": list(map(str, command))})
        self.write(self.root / "commands.json", json.dumps(self.commands, indent=2))
        self.visited.add(name)
        if self.dry_run:
            return
        self.state[name] = {"signature": key, "status": "running", "artifacts": list(map(str, artifacts))}
        self.write(self.state_path, json.dumps(self.state, indent=2))
        if self.resume and name.startswith(("eval/",)):
            # An evaluation can have written JSONL just before interruption. Keep
            # that incomplete artifact for inspection, then rerun the whole eval.
            for artifact in artifacts:
                if artifact.exists():
                    artifact.rename(artifact.with_name(artifact.name + f".incomplete.{time.time_ns()}"))
        if self.resume and ("build-demos" in command or "build-corrections" in command):
            for artifact in artifacts:
                if artifact.exists():
                    artifact.rename(artifact.with_name(artifact.name + f".incomplete.{time.time_ns()}"))
        log = self.root / "logs" / (name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        print(f"[run] {name} -> {log}", flush=True)
        with log.open("a") as stream:
            if self.plan.get("in_process_smoke", False):
                if os.environ.get("ACCELERATE_USE_CPU", "").lower() != "true" or self.plan.get("accelerate_config"):
                    raise ValueError("In-process validation is restricted to single-process CPU smoke fixtures")
                import traceback
                try:
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        self._smoke_command(list(map(str, command)))
                    code = 0
                except Exception:
                    traceback.print_exc(file=stream)
                    code = 1
                finally:
                    gc.collect()
                result = subprocess.CompletedProcess(command, code)
            else:
                result = subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
        elapsed = time.monotonic() - started
        self.state[name].update(status="complete" if result.returncode == 0 else "failed", elapsed_seconds=elapsed,
                                returncode=result.returncode)
        self.write(self.state_path, json.dumps(self.state, indent=2))
        if result.returncode:
            print("\n".join(log.read_text().splitlines()[-20:]), file=sys.stderr)
            raise RuntimeError(f"Stage {name} failed; see {log}. Resume after fixing the cause")
        if not all(p.exists() for p in artifacts):
            self.state[name]["status"] = "failed"
            self.write(self.state_path, json.dumps(self.state, indent=2))
            raise RuntimeError(f"Stage {name} returned without its required artifacts")
        print(f"[done] {name} ({elapsed:.1f}s)", flush=True)

    @staticmethod
    def _smoke_command(command):
        from .config import load_config
        def arg(name, default=None):
            return command[command.index(name)+1] if name in command else default
        cfg = load_config(arg("--config"))
        if "opd.dimension_workflow" in command:
            from .dimension_workflow import select_demos
            return select_demos(cfg, json.loads(Path(arg("--inputs")).read_text()), arg("--output"))
        if "opd.eval_pool" in command:
            from .eval_pool import run_pool
            return run_pool(cfg, json.loads(Path(arg("--jobs")).read_text()), arg("--output"),
                            int(arg("--workers")), int(arg("--chunk-size")), "--resume" in command,
                            in_process=True, gpus=int(arg("--gpus")))
        if "opd.qgpi_workflow" in command:
            from .qgpi_workflow import calibrate_quality
            return calibrate_quality(cfg, arg("--output"), int(arg("--limit", 64)))
        if "opd.preflight" in command:
            from .preflight import run
            return run(cfg, Path(arg("--data")), Path(arg("--output")), arg("--policy"))
        operation = command[command.index("opd")+1]
        if operation == "train":
            from .trainer import train
            if arg("--resume"):
                cfg["train"]["resume_from"] = arg("--resume")
            return train(cfg)
        if operation == "build-corrections":
            from .corrections import build_corrections
            return build_corrections(cfg, arg("--output"), int(arg("--candidates", 1)))
        from .workflows import build_demos, evaluate_model
        if operation == "build-demos":
            return build_demos(cfg, arg("--output"), int(arg("--candidates", 1)))
        return evaluate_model(cfg, arg("--output"), arg("--teacher"), arg("--model"))

    def train(self, name, cfg):
        cfg = copy.deepcopy(cfg)
        cfg["output_dir"] = str(self.root / "runs" / name)
        cfg["train"]["resume_from"] = None
        validate(cfg)
        path = self.config(name, cfg)
        cmd = [sys.executable, "-m", "opd", "train", "--config", path]
        if self.plan.get("accelerate_config"):
            cmd = [sys.executable, "-m", "accelerate.commands.launch", "--config_file",
                   self.plan["accelerate_config"]]
            for variable, flag in (("OPD_NUM_PROCESSES", "--num_processes"), ("OPD_MAIN_PROCESS_PORT", "--main_process_port")):
                if os.environ.get(variable):
                    cmd += [flag, os.environ[variable]]
            cmd += ["-m", "opd", "train", "--config", path]
        record = self.state.get(name)
        if record and record["signature"] != fingerprint(cfg):
            raise ValueError(f"Stage configuration changed: {name}. Use a new output directory")
        if self.resume:
            latest = latest_checkpoint(cfg["output_dir"])
            if latest:
                cmd += ["--resume", str(latest)]
            elif record and record["status"] in ("running", "failed"):
                output = Path(cfg["output_dir"])
                if output.exists() and not self.dry_run:
                    archive_incomplete(output)
        final = Path(cfg["output_dir"]) / "final"
        self.command(name, cmd, [final / "opd_metadata.json"], cfg)
        return str(final), cfg

    def demos(self, name, cfg, corrections=False):
        path = self.config(name, cfg)
        output = self.root / "demos" / (name + ".jsonl")
        self.command(name, [sys.executable, "-m", "opd", "build-corrections" if corrections else "build-demos",
                           "--config", path, "--output", output, "--candidates", self.plan.get("demo_candidates", 2)],
                     [output, str(output) + ".report.json"], cfg)
        return str(output)

    def evaluate(self, name, cfg, model=None, teacher=None):
        path = self.config("evaluation/" + name, cfg)
        output = self.root / "evaluation" / (name + ".jsonl")
        cmd = self.inference_command("opd", "evaluate", "--config", path, "--output", output)
        if model:
            cmd += ["--model", model]
        if teacher:
            cmd += ["--teacher", teacher]
        self.command("eval/" + name, cmd, [output, str(output) + ".summary.json", str(output) + ".model.json"],
                     {"config": cfg, "model": model, "teacher": teacher})
        return str(output)

    def inference_command(self, module, *arguments):
        count = self.plan.get("inference_num_processes", 1)
        if type(count) is not int or count < 1:
            raise ValueError("inference_num_processes must be a positive integer")
        if count == 1:
            return [sys.executable, "-m", module, *arguments]
        if self.plan.get("in_process_smoke"):
            raise ValueError("In-process CPU smoke does not support distributed inference")
        # Inference does not use the training DeepSpeed config. One model per
        # visible GPU, data sharding, independent KV caches; random free port.
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nnodes=1", f"--nproc_per_node={count}", "--module", module, *arguments]

    def baseline_pool(self, name, cfg, requests):
        from .eval_pool import artifact_config, prepare_jobs, valid_report, validate_pool
        gpus = self.plan.get("inference_num_processes", 1)
        per_gpu = self.plan.get("baseline_workers_per_gpu", 1)
        if type(per_gpu) is not int or per_gpu < 1:
            raise ValueError("baseline_workers_per_gpu must be a positive integer")
        workers = gpus * per_gpu
        chunk_size = self.plan.get("baseline_chunk_size", 32)
        validate_pool(cfg, workers, chunk_size, gpus)
        path = self.config("baseline_pool/" + name, cfg)
        jobs_path = path.with_suffix(".jobs.json")
        self.write(jobs_path, json.dumps(requests, indent=2))
        output = self.root / "inference_pool" / name
        # Exactly one coordinator. It spawns GPU-bound persistent workers itself.
        cmd = [sys.executable, "-m", "opd.eval_pool", "--config", path, "--jobs", jobs_path,
               "--output", output, "--workers", workers, "--gpus", gpus, "--chunk-size", chunk_size]
        if self.resume:
            cmd += ["--resume"]
        artifacts = [output / "summary.json"]
        for request in requests:
            artifacts.extend(request["output"] + suffix for suffix in ("", ".summary.json", ".model.json"))
        record = self.state.get("baseline_pool/" + name)
        if self.resume and record and record["status"] == "complete":
            evaluations, _ = prepare_jobs(cfg, requests, chunk_size)
            if not all(valid_report(output, evaluation) for evaluation in evaluations):
                # Existing files alone do not establish a complete baseline report.
                record["status"] = "failed"
        # Not an eval/ stage: interrupted pools retain independently committed chunks/reports.
        self.command("baseline_pool/" + name, cmd, artifacts,
                     {"config": artifact_config(cfg), "requests": requests,
                      "workers": workers, "gpus": gpus, "chunk_size": chunk_size})


def base_config(plan, data):
    if plan.get("fixture_config"):
        from .config import load_config
        cfg = load_config(plan["fixture_config"])
    else:
        cfg = copy.deepcopy(DEFAULTS)
        cfg["model"] = merge(cfg["model"], plan["model"])
        cfg["teacher"] = merge(cfg["teacher"], plan.get("teacher", {}))
        cfg["teachers"] = []
        for expert in plan["experts"]:
            from .expert_manifest import checkpoint_candidates
            task = canonical_task(expert["task"])
            candidates = checkpoint_candidates(expert)
            cfg["teachers"].extend(candidates)
            cfg["routing"]["tasks"][task] = {"teachers": {c["id"]: 1.0 for c in candidates}, "strength": 1.0}
            cfg["data"]["task_weights"][task] = expert.get("weight", 1.0)
    for section in ("train", "rollout", "inference", "quality", "qgpi", "routing"):
        cfg[section] = merge(cfg[section], plan.get(section, {}))
    cfg["data"].update(train_file=str(data / "train.jsonl"), calibration_file=str(data / "calibration.jsonl"),
                        eval_file=str(data / (plan.get("evaluation_split", "validation") + ".jsonl")))
    cfg["output_dir"] = str(Path(plan["output_dir"]) / "unused")
    if "$" in json.dumps(cfg):
        raise ValueError("Unresolved model/adapter environment variables; set SFT_BASE and the expert paths in the manifest")
    validate(cfg)
    return cfg


def dimension_config(base, dimension, rows, rank, seed):
    cfg = copy.deepcopy(base)
    tasks = {r["task_id"] for r in rows if dimension in r["dimensions"]}
    if not tasks:
        raise ValueError(f"Dimension {dimension} has no explicitly labeled samples")
    cfg["seed"] = seed
    cfg["data"]["dimension"] = dimension
    cfg["data"]["task_weights"] = {t: base["data"]["task_weights"].get(t, 1.0) for t in sorted(tasks)}
    cfg["routing"]["tasks"] = {t: base["routing"]["tasks"][t] for t in sorted(tasks)}
    active = {t for route in cfg["routing"]["tasks"].values() for t in route["teachers"]}
    cfg["teachers"] = [t for t in base["teachers"] if t["id"] in active]
    cfg["model"].update(student_mode="lora", lora_rank=rank, lora_alpha=2*rank)
    cfg["model"].pop("student_init", None)
    return cfg


def configure_method(cfg, method, demos, steps, warmup=None):
    cfg = copy.deepcopy(cfg)
    cfg["train"].update(stage="opd", max_steps=steps["distill"], trajectory_source="student",
                        objective="forward_kl", sft_coef=0.0)
    if method in ("sft", "iterative_sft"):
        cfg["train"].update(stage="warmup", max_steps=steps["sft"])
        cfg["data"]["train_file"] = demos
    elif method == "offline_forward":
        cfg["train"]["trajectory_source"] = "teacher"
        cfg["data"]["demo_file"] = demos
    elif method == "mixed_forward":
        cfg["train"]["trajectory_source"] = "mixed"
        cfg["data"]["demo_file"] = demos
    if method in ("opd_reverse", "sft_reverse"):
        cfg["train"]["objective"] = "sampled_reverse_kl"
    if method in ("sft_forward", "sft_reverse", "mixed_forward") and warmup:
        cfg["model"]["student_init"] = warmup
    return cfg


def tokenized_data(base, data, runner):
    destination = runner.root / "tokenized_data"
    policy = runner.plan.get("data", {}).get("overlong_policy", "error")
    path = runner.config("preflight", base)
    inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob("*.jsonl"))}
    model_fields = ("base_model", "tokenizer", "trust_remote_code", "chat_template_kwargs")
    identity = {"model": {k: base["model"].get(k) for k in model_fields},
                "max_prompt_tokens": base["rollout"]["max_prompt_tokens"],
                "context_length": base["rollout"]["context_length"],
                "policy": policy, "inputs": inputs}
    runner.command("preflight", [sys.executable, "-m", "opd.preflight", "--config", path,
                                "--data", data, "--output", destination, "--policy", policy],
                   [destination / "length_audit.json"], identity)
    return destination


def final_config(base, models, data, runner, name, mode):
    cfg = copy.deepcopy(base)
    cfg["model"].update(student_mode="full")
    cfg["model"].pop("student_init", None)
    cfg["data"]["dimension"] = None
    cfg["teachers"] = [{"id": "dim_" + d, "adapter": p} for d, p in models.items()]
    if mode == "dimensions_and_tasks":
        cfg["teachers"] += base["teachers"]
    routes, weights = {}, {}
    for split in ("train", "calibration", "eval", "validation"):
        path = data / (split + ".jsonl")
        if not path.exists() or not path.stat().st_size:
            continue
        original = list(read_rows(path))
        patterns = {}
        for row in original:
            active = tuple(d for d in models if d in row["dimensions"])
            if not active:
                raise ValueError(f"No dimension teacher for {row['id']}")
            patterns.setdefault(row["task_id"], set()).add(active)
        converted = []
        for row in original:
            task = row["task_id"]
            active = [d for d in models if d in row["dimensions"]]
            alias = task + "__" + "".join(active)
            teachers = {"dim_"+d: 1.0/len(active) for d in active}
            if mode == "dimensions_and_tasks":
                teachers = {t: w*.5 for t, w in teachers.items()}
                original_route = base["routing"]["tasks"][task]["teachers"]
                total = sum(original_route.values())
                teachers.update({t: .5*w/total for t, w in original_route.items()})
            routes[alias] = {"teachers": teachers, "strength": 1.0}
            if split == "train":
                weights[alias] = base["data"]["task_weights"].get(task, 1.0)/len(patterns[task])
            converted.append({**row, "original_task_id": task, "task_id": alias})
        destination = runner.root / "fusion_data" / name / (split + ".jsonl")
        write_rows(destination, converted)
        field = "eval_file" if split == runner.plan.get("evaluation_split", "validation") else split + "_file"
        if field in ("train_file", "calibration_file", "eval_file"):
            cfg["data"][field] = str(destination)
    cfg["routing"]["tasks"] = routes
    cfg["data"]["task_weights"] = weights
    return cfg


def run_matrix(plan, resume=False, dry_run=False):
    runner = Runner(plan, resume, dry_run)
    data = Path(plan.get("prepared_data") or runner.root / "data")
    base = base_config(plan, data)
    # Include exact prepared data content in the suite fingerprint, not merely paths.
    provenance = {"plan": plan, "data": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(data.glob("*.jsonl"))}}
    manifest = runner.root / "manifest.json"
    signature = fingerprint(provenance)
    if manifest.exists() and json.loads(manifest.read_text())["signature"] != signature:
        raise ValueError("Experiment/data changed; use a new output directory")
    runner.write(manifest, json.dumps({"signature": signature, **provenance}, indent=2))
    if not dry_run:
        data = tokenized_data(base, data, runner)
        base = base_config(plan, data)
    rows = load_data(base["data"]["train_file"])
    missing = set(base["routing"]["tasks"]) - {r["task_id"] for r in rows}
    if missing:
        raise ValueError(f"Length filtering removed all training rows for {sorted(missing)}; inspect length_audit.json")
    dimensions = plan.get("dimensions") or sorted({d for row in rows for d in row["dimensions"]})
    results = {}
    do_eval = plan.get("evaluate", True)
    if do_eval:
        evaluation_rows = load_data(base["data"]["eval_file"])
        missing = {r["task_id"] for r in rows} - {r["task_id"] for r in evaluation_rows}
        if missing:
            raise ValueError(f"Evaluation split lacks tasks {sorted(missing)}; adjust group split, never reuse training rows")
    for seed in plan["seeds"]:
      base["seed"] = seed
      original_baselines = {}
      if do_eval and (plan.get("direct_methods") or plan.get("final", {}).get("enabled")):
        for task, route in base["routing"]["tasks"].items():
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            subset = runner.root / "eval_data" / "original" / f"{task}.jsonl"
            write_rows(subset, [r for r in evaluation_rows if r["task_id"] == task])
            cfg["data"]["eval_file"] = str(subset)
            teacher = max(route["teachers"], key=route["teachers"].get)
            original_baselines[task] = runner.evaluate(f"seed_{seed}/original_teacher/{task}", cfg, teacher=teacher)
      for rank in plan["lora_ranks"]:
        trained = {method: {} for method in plan["methods"]}
        for dimension in dimensions:
            stem = f"seed_{seed}/rank_{rank}/{dimension}"
            cfg = dimension_config(base, dimension, rows, rank, seed)
            needs_demos = any(m not in ("opd_forward", "opd_reverse") for m in plan["methods"])
            demo_cfg = copy.deepcopy(cfg)
            demo_cfg["model"].update(lora_rank=plan["lora_ranks"][0], lora_alpha=2*plan["lora_ranks"][0])
            demos = runner.demos(f"seed_{seed}/demos_{dimension}", demo_cfg) if needs_demos else None
            warmup = None
            if any(m in ("sft_forward", "sft_reverse", "mixed_forward") for m in plan["methods"]):
                warm = configure_method(cfg, "sft", demos, {**plan["steps"], "sft": plan["steps"]["warmup"]})
                warmup, _ = runner.train(stem + "/warmup", warm)
            baselines = {}
            if do_eval:
                for task, route in cfg["routing"]["tasks"].items():
                    eval_cfg = copy.deepcopy(cfg)
                    subset = runner.root / "eval_data" / dimension / f"{task}.jsonl"
                    write_rows(subset, [r for r in evaluation_rows if r["task_id"] == task and dimension in r["dimensions"]])
                    eval_cfg["data"]["eval_file"] = str(subset)
                    teacher = max(route["teachers"], key=route["teachers"].get)
                    baselines[task] = runner.evaluate(stem + "/teacher_" + task, eval_cfg, teacher=teacher)
            for method in plan["methods"]:
                name = stem + "/" + method
                training = configure_method(cfg, method, demos, plan["steps"], warmup)
                if method == "mixed_forward":
                    training["train"]["sft_coef"] = plan.get("mixed_sft_coef", 0.0)
                model, trained_cfg = runner.train(name, training)
                if method == "iterative_sft":
                    replay_files = [demos]
                    for round_index in range(plan.get("correction_rounds", 1)):
                        correction_cfg = copy.deepcopy(cfg)
                        correction_cfg["model"]["student_init"] = model
                        correction = runner.demos(name + f"/corrections_{round_index}", correction_cfg, corrections=True)
                        replay_files.append(correction)
                        replay = runner.root / "demos" / name / f"replay_{round_index}.jsonl"
                        if not dry_run:
                            combined = []
                            for file_index, path in enumerate(replay_files):
                                combined.extend({**r, "id": f"round_{file_index}:{r['id']}"} for r in read_rows(path))
                            write_rows(replay, combined)
                        corrected = configure_method(cfg, "sft", str(replay),
                                                     {**plan["steps"], "sft": plan["steps"]["distill"]})
                        corrected["model"]["student_init"] = model
                        model, trained_cfg = runner.train(name + f"/round_{round_index}", corrected)
                trained[method][dimension] = model
                results[name] = {"model": model, "kind": "dimension", "tasks": list(cfg["routing"]["tasks"])}
                if do_eval:
                    evaluation = runner.evaluate(name, cfg, model=model)
                    if not dry_run:
                        from .retention import compare, save_report
                        report = compare(evaluation, baselines, plan.get("retention_tolerance", .02), seed=seed)
                        save_report(runner.root / "retention" / (name + ".json"), report)
                        results[name]["retention"] = report
        final = plan.get("final", {})
        if final.get("enabled", False):
            for method, models in trained.items():
                if method not in final.get("methods", plan["methods"]):
                    continue
                for mode in final.get("teacher_sets", ["dimensions", "dimensions_and_tasks"]):
                    if mode not in ("dimensions", "dimensions_and_tasks"):
                        raise ValueError(f"Unknown final teacher set {mode}")
                    name = f"seed_{seed}/rank_{rank}/final/{method}/{mode}"
                    cfg = final_config(base, models, data, runner, name, mode)
                    cfg["seed"] = seed
                    if final.get("warmup", True):
                        demos = runner.demos(name + "/demos", cfg)
                        warm = configure_method(cfg, "sft", demos, {**plan["steps"], "sft": plan["steps"]["warmup"]})
                        cfg["model"]["student_init"], _ = runner.train(name + "/warmup", warm)
                    cfg["train"].update(stage="opd", objective=final.get("objective", "forward_kl"),
                                         max_steps=plan["steps"]["final"], trajectory_source="student", sft_coef=0.)
                    cfg["train"]["learning_rate"] = final.get("learning_rate", cfg["train"]["learning_rate"])
                    # Warmup is a sibling stage, never a child of the next
                    # trainer's output directory (which must initially be empty).
                    model, _ = runner.train(name + "/opd", cfg)
                    # Retain the diagnostic from the old colliding parent stage
                    # while marking that the replacement has now succeeded.
                    if not dry_run and runner.state.get(name, {}).get("status") in ("failed", "running"):
                        runner.state[name].update(status="superseded", superseded_by=name + "/opd")
                        runner.write(runner.state_path, json.dumps(runner.state, indent=2))
                    results[name] = {"model": model, "kind": "full", "tasks": list(base["routing"]["tasks"])}
                    if do_eval:
                        # Evaluate on original task ids so all methods use identical task metrics.
                        results[name]["evaluation"] = runner.evaluate(name, base, model=model)
                        if not dry_run:
                            from .retention import compare, save_report
                            report = compare(results[name]["evaluation"], original_baselines,
                                             plan.get("retention_tolerance", .02), seed=seed)
                            save_report(runner.root / "retention" / (name + ".json"), report)
                            results[name]["retention"] = report
        for method in plan.get("direct_methods", []):
            if method not in ("sft", "opd_forward", "opd_reverse"):
                raise ValueError("direct_methods supports sft, opd_forward, opd_reverse")
            name = f"seed_{seed}/direct/{method}"
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            cfg["model"]["student_mode"] = "full"
            demos = runner.demos(f"seed_{seed}/direct/demos", cfg) if method == "sft" else None
            cfg = configure_method(cfg, method, demos, {**plan["steps"], "sft": plan["steps"]["final"],
                                                       "distill": plan["steps"]["final"]})
            model, _ = runner.train(name, cfg)
            results[name] = {"model": model, "kind": "full", "tasks": list(base["routing"]["tasks"])}
            if do_eval:
                results[name]["evaluation"] = runner.evaluate(name, base, model=model)
                if not dry_run:
                    from .retention import compare, save_report
                    report = compare(results[name]["evaluation"], original_baselines,
                                     plan.get("retention_tolerance", .02), seed=seed)
                    save_report(runner.root / "retention" / (name + ".json"), report)
                    results[name]["retention"] = report
    original = plan.get("original_evaluation", {})
    if original.get("enabled", False):
        candidates = {name: r for name, r in results.items()
                      if r["kind"] == "full" or original.get("include_dimensions", True)}
        if original.get("include_teachers", True):
            for expert in base["teachers"]:
                tasks = [t for t, r in base["routing"]["tasks"].items() if expert["id"] in r["teachers"]]
                candidates["original_teacher/" + expert["id"]] = {
                    "model": expert.get("adapter", expert.get("model")), "tasks": tasks}
        for name, result in candidates.items():
            destination = runner.root / "original_evaluation" / name
            cmd = [sys.executable, "-m", "opd.original_eval", "--repo", plan["upstream_repo"],
                   "--data-root", plan["data_root"], "--model", result["model"],
                   "--base", base["model"]["base_model"], "--output", destination,
                   "--upstream-python", original.get("python", os.environ.get("SIMULATION_PYTHON", sys.executable)),
                   "--gpus", original.get("gpus", 8), "--tensor-parallel", original.get("tensor_parallel", 1),
                   "--tasks", *result["tasks"]]
            runner.command("original_eval/" + name, cmd, [destination / "complete.json"],
                           {"result": result, "settings": original})
    runner.write(runner.root / "commands.json", json.dumps(runner.commands, indent=2))
    training = []
    for item in runner.commands:
        cmd = item["command"]
        if "train" in cmd and "--config" in cmd:
            cfg = yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text())
            training.append({"stage": item["stage"], "steps": cfg["train"]["max_steps"],
                             "student_mode": cfg["model"]["student_mode"],
                             "objective": cfg["train"]["stage"] + "/" + cfg["train"]["objective"],
                             "config": cmd[cmd.index("--config") + 1]})
    runner.write(runner.root / "plan_summary.json", json.dumps({
        "pending_commands": len(runner.commands), "pending_training_stages": len(training),
        "pending_optimizer_steps": sum(s["steps"] for s in training), "training": training,
        "note": "Optimizer steps are not equal compute budgets; demos, generation lengths, teacher calls and GPU count also matter. Completed stages are omitted on resume."
    }, indent=2))
    if not dry_run:
        runner.write(runner.root / "results.json", json.dumps(results, indent=2))
    return {"output_dir": str(runner.root), "planned_commands": len(runner.commands), "dry_run": dry_run}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--sample-jsonl", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.config)
    if plan.get("upstream_repo"):
        os.environ["SIMULATION_REPO"] = plan["upstream_repo"]
    root = Path(plan["output_dir"])
    data = Path(plan.get("prepared_data") or root / "data")
    if not plan.get("prepared_data"):
        prep = {**plan, **plan.get("data", {})}
        source_experts = [{k: v for k, v in e.items() if k not in ("adapter", "model", "weight", "checkpoints")}
                          for e in prep["experts"]]
        prep_key = fingerprint({k: prep.get(k) for k in ("upstream_repo", "data_root", "data", "seed")}
                              | {"experts": source_experts, "limit": args.limit_per_task, "sample": args.sample_jsonl,
                                 "exporter_version": EXPORTER_VERSION})
        stamp = data / "preparation.json"
        if stamp.exists():
            if json.loads(stamp.read_text())["signature"] != prep_key:
                raise ValueError("Data preparation settings changed; use a new output directory")
        else:
            if data.exists() and any(data.iterdir()):
                if not args.resume:
                    raise FileExistsError(f"Incomplete data preparation at {data}; use --resume to archive and rebuild")
                archive_incomplete(data)
            audit = prepare_sources(prep, data, args.limit_per_task, args.sample_jsonl)
            atomic_json(stamp, {"signature": prep_key, "exported": audit["exported"]})
    if args.prepare_only:
        print(json.dumps({"data": str(data)}, indent=2))
    else:
        if plan.get("workflow") == "dimension_offline":
            from .dimension_workflow import run_dimensions
            result = run_dimensions(plan, args.resume, args.dry_run)
        elif plan.get("workflow") == "expert_pool":
            from .expert_pool import run_expert_pool
            result = run_expert_pool(plan, args.resume, args.dry_run)
        elif plan.get("workflow") == "qgpi":
            from .qgpi_workflow import run_qgpi
            result = run_qgpi(plan, args.resume, args.dry_run)
        else:
            result = run_matrix(plan, args.resume, args.dry_run)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
