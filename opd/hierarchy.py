"""Five dimension LoRAs from task experts, then one on-policy full student."""

import argparse
import copy
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .checkpoints import archive_incomplete, atomic_json
from .config import load_config, validate
from .data import load_data, read_rows, write_rows
from .dimension_workflow import bounded_rows
from .experiments import Runner, base_config, configure_method, dimension_config, expanded, final_config, fingerprint, tokenized_data
from .expert_manifest import load_manifest
from .fusion import file_hash
from .hierarchy_judge import DIMENSIONS, TASK_DIMENSIONS, VERSION, row_spec, settings
from .quality_basis import VERIFIABLE_AXIS

PROJECT = Path(__file__).resolve().parents[1]
SCOPE = ("Static policy prefixes. U measures actor-visible intent progress, T visible-history consistency, "
         "N response-level human-likeness. These are rubric proxies, not full interactive benchmark results. C excluded.")
PHASES = ("prepare", "trajectories", "dimensions", "fusion")


def preparation_identity(plan):
    """Only inputs that can change rendered, split or dimension-labelled rows."""
    expert_fields = ("task", "dimensions", "eval_files", "group_fields", "rubrics",
                     "quality_evidence", "hard_constraints")
    data_fields = ("coser_reference_prefixes", "validation_fraction", "calibration_fraction",
                   "eval_group_overlap", "objective_only")
    return {
        "upstream_repo": plan["upstream_repo"], "data_root": plan["data_root"],
        "seed": plan["seeds"][0],
        "experts": [{k: copy.deepcopy(expert[k]) for k in expert_fields if k in expert}
                    for expert in plan["experts"]],
        "data": {k: copy.deepcopy(plan.get("data", {}).get(k)) for k in data_fields},
    }


def trajectory_identity(plan):
    """Semantic inputs for reusable teacher demonstrations.

    Runtime batching changes the particular random draw but not its policy or
    distribution, so an explicitly requested reuse may keep already generated
    and content-validated candidates.  Checkpoint locations are likewise
    normalized to their recorded content hashes.
    """
    model_fields = ("base_model", "tokenizer", "dtype", "trust_remote_code", "chat_template_kwargs")
    input_hashes = plan.get("input_hashes", {})

    def checkpoint_identity(expert, checkpoint):
        location = checkpoint.get("adapter") or checkpoint.get("model")
        content = {}
        if location:
            root = Path(location)
            for filename, digest in input_hashes.items():
                try:
                    relative = Path(filename).relative_to(root)
                except ValueError:
                    continue
                content[str(relative)] = digest
        # Plans inspected before check_inputs have no hashes yet; retaining the
        # path is safer than treating unknown checkpoints as interchangeable.
        return {"id": checkpoint["id"], "task": expert["task"],
                "content": content or {"unverified_location": location}}

    rollout = {k: copy.deepcopy(v) for k, v in plan["rollout"].items()
               if k != "generation_batch_size"}
    judge = {k: copy.deepcopy(v) for k, v in plan["judge"].items()
             if k not in ("timeout", "attempts")}
    return {
        "preparation": preparation_identity(plan),
        "model": {k: copy.deepcopy(plan["model"].get(k)) for k in model_fields},
        "teacher_dtype": plan.get("teacher", {}).get("dtype"),
        "checkpoints": [checkpoint_identity(expert, checkpoint)
                        for expert in plan["experts"] for checkpoint in expert["checkpoints"]],
        "rollout": rollout,
        "quality": copy.deepcopy(plan["quality"]),
        "judge": judge,
        "demo_limit_per_task": plan["demo_limit_per_task"],
        "demo_candidates": plan["demo_candidates"],
        "seeds": copy.deepcopy(plan["seeds"]),
        "dimensions": copy.deepcopy(plan["dimensions"]),
        "version": VERSION,
    }


def identity_differences(saved, current, path=""):
    """Small, secret-free structural diff for actionable reuse failures."""
    differences = []
    if isinstance(saved, dict) and isinstance(current, dict):
        for key in sorted(set(saved) | set(current)):
            child = f"{path}.{key}" if path else key
            if key not in saved:
                differences.append({"field": child, "saved": "<missing>", "current": current[key]})
            elif key not in current:
                differences.append({"field": child, "saved": saved[key], "current": "<missing>"})
            else:
                differences.extend(identity_differences(saved[key], current[key], child))
    elif saved != current:
        differences.append({"field": path, "saved": saved, "current": current})
    return differences


def dimension_identity(plan):
    """Inputs that determine the five trained dimension LoRAs."""
    return {
        "trajectories": trajectory_identity(plan),
        "model": copy.deepcopy(plan["model"]),
        "train": copy.deepcopy(plan["train"]),
        "distill_steps": plan["steps"]["distill"],
        "num_processes": plan["num_processes"],
    }


def hierarchy_manifest(root):
    path = Path(root) / "hierarchy_manifest.json"
    return json.loads(path.read_text()) if path.is_file() else None


def load_hierarchy_plan(path, *, output=None, processes=8, dimension_steps=200, fusion_steps=1000,
                        allow_missing_judge=False):
    path = Path(path).resolve()
    plan = expanded(yaml.safe_load(path.read_text()))
    if plan.get("workflow") != "hierarchy_five" or plan.get("dimensions") != list(DIMENSIONS):
        raise ValueError("Expected hierarchy_five with dimensions F/S/U/T/N")
    if any(type(n) is not int or n < 1 for n in (processes, dimension_steps, fusion_steps)):
        raise ValueError("Process and step counts must be positive integers")
    if "$" in str(os.environ.get("SFT_BASE", plan["model"]["base_model"])):
        raise ValueError("Set SFT_BASE to the common Qwen3-8B directory")
    for field, variable in (("upstream_repo", "SIMULATION_REPO"), ("data_root", "OPD_DATA_ROOT")):
        plan[field] = str((path.parent / os.environ.get(variable, plan[field])).resolve())
    plan["output_dir"] = str(Path(output).resolve()) if output else str((path.parent / plan["output_dir"]).resolve())
    plan["model"]["base_model"] = str((path.parent / os.environ.get("SFT_BASE", plan["model"]["base_model"])).resolve())
    manifest = (path.parent / plan["expert_manifest"]).resolve()
    plan["expert_manifest"] = str(manifest)
    plan["experts"], inventory = load_manifest(manifest, plan["tasks"])
    # The new run uses every supplied version, including the explicitly listed
    # BehaviorChain path. Resolution is checked/reported on the training host.
    for expert in plan["experts"]:
        expert["dimensions"] = TASK_DIMENSIONS[expert["task"]]
        for checkpoint in expert["checkpoints"]:
            checkpoint["enabled"] = True
            checkpoint.pop("note", None)
    if len(plan["experts"]) != 14 or sum(len(e["checkpoints"]) for e in plan["experts"]) != 14:
        raise ValueError("This production recipe requires exactly one checkpoint for each of 14 tasks")
    plan["checkpoint_root"] = inventory["checkpoint_root"]
    plan["steps"] = {"distill": dimension_steps, "final": fusion_steps}
    plan["num_processes"] = plan["inference_num_processes"] = processes
    for section in (plan["train"], plan["fusion"]):
        global_batch, micro = section["global_prompt_batch"], section["batch_size"]
        if global_batch % (processes * micro):
            raise ValueError("Global prompt batch must be divisible by ranks x microbatch")
        section["gradient_accumulation_steps"] = global_batch // (processes * micro)
    plan["judge"] = settings(allow_missing=allow_missing_judge)  # Never persist credentials.
    return plan


def check_inputs(plan):
    """Check the actual remote files and record the spelling resolution explicitly."""
    from .models import adapter_path
    base = Path(plan["model"]["base_model"])
    config = json.loads((base / "config.json").read_text())
    expected = {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36,
                "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 151936}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("This recipe requires the common Qwen3-8B base; config does not match its architecture")
    hashes = {str(base / "config.json"): file_hash(base / "config.json")}
    resolved = []
    for expert in plan["experts"]:
        for checkpoint in expert["checkpoints"]:
            supplied = Path(checkpoint["adapter"])
            actual = supplied
            if supplied.name == "loara_adapter" and not supplied.exists():
                actual = supplied.with_name("lora_adapter")
            actual = Path(adapter_path(actual))
            metadata = json.loads((actual / "adapter_config.json").read_text())
            declared = str(metadata.get("base_model_name_or_path", ""))
            if declared and declared != str(base):
                normalized = "".join(c for c in Path(declared).name.lower() if c.isalnum())
                if not normalized.endswith("qwen38b"):
                    raise ValueError(f"{checkpoint['id']} declares another base model: {declared}")
            if metadata.get("peft_type") != "LORA":
                raise ValueError(f"{checkpoint['id']} is not a LoRA checkpoint")
            checkpoint["adapter"] = str(actual)
            resolved.append({"id": checkpoint["id"], "task": expert["task"],
                             "supplied_path": str(supplied), "resolved_path": str(actual),
                             "declared_base": declared})
            for filename in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
                file = actual / filename
                if file.is_file():
                    hashes[str(file)] = file_hash(file)
            print(f"[checkpoint] {checkpoint['id']} -> {actual}", flush=True)
    plan["input_hashes"] = hashes
    plan["resolved_checkpoints"] = resolved
    return resolved


def prepare_data(plan, resume=False):
    from .source_data import EXPORTER_VERSION, prepare_sources
    root = Path(plan["output_dir"])
    data = root / "data"
    inputs = preparation_identity(plan)
    key = fingerprint({"schema": "hierarchy-preparation-v2", "inputs": inputs,
                       "exporter_version": EXPORTER_VERSION, "judge_version": VERSION})
    stamp = data / "preparation.json"
    if stamp.is_file():
        saved = json.loads(stamp.read_text())
        hashes = {p.name: file_hash(p) for p in data.glob("*.jsonl")}
        compatible = saved.get("signature") == key
        # Migrate a legacy full-plan stamp when its recorded plan proves that
        # all actual data-preparation inputs are unchanged.
        if not compatible:
            previous = hierarchy_manifest(root)
            compatible = bool(previous and preparation_identity(previous["plan"]) == inputs)
        if compatible and saved["hashes"] == hashes:
            if saved.get("signature") != key:
                atomic_json(stamp, {**saved, "schema": "hierarchy-preparation-v2",
                                    "signature": key, "inputs": inputs})
                print("[retained] migrated source preparation to dependency-scoped identity", flush=True)
            print("[retained] source preparation", flush=True)
            return data
        if not compatible:
            raise ValueError("Source preparation inputs changed; use a new output directory")
    if data.exists():
        if not resume:
            raise FileExistsError("Incomplete data; rerun with --resume")
        archive_incomplete(data)
    prep = {**plan, **plan["data"], "seed": plan["seeds"][0], "objective_only": False}
    audit = prepare_sources(prep, data)
    # Label only dimensions justified by each particular policy prefix.
    coverage = {}
    for path in sorted(data.glob("*.jsonl")):
        counts, omissions, labeled = defaultdict(Counter), Counter(), []
        for row in read_rows(path):
            spec = row_spec(row)
            omissions.update(f"{row['task_id']}/{d}: {reason}" for d, reason in spec["omitted"].items())
            if not spec["dimensions"]:
                continue
            row["dimensions"] = spec["dimensions"]
            row["evaluator_context"]["quality_evidence"] = spec["quality_evidence"]
            labeled.append(row)
            counts[row["task_id"]].update(row["dimensions"])
        write_rows(path, labeled)
        coverage[path.stem] = {"tasks": {t: dict(c) for t, c in counts.items()}, "omitted": dict(omissions)}
    atomic_json(data / "quality_coverage.json", {"scope": SCOPE, "splits": coverage})
    atomic_json(stamp, {"schema": "hierarchy-preparation-v2", "signature": key, "inputs": inputs,
                       "exported": audit["exported"],
                       "hashes": {p.name: file_hash(p) for p in data.glob("*.jsonl")}})
    return data


def select_dimension(cfg, inputs, output):
    dimension = cfg["data"]["dimension"]
    if dimension not in DIMENSIONS:
        raise ValueError("Select one explicit dimension")
    groups = defaultdict(list)
    counts = defaultdict(Counter)
    for path in inputs:
        for record in read_rows(path):
            for demo in record["candidates"]:
                if dimension not in demo["dimensions"]:
                    continue
                task, source = record["task_id"], record["id"]
                if (demo["task_id"] != task or demo["demo_input_id"] != source or
                    demo["demo_teacher"] not in cfg["routing"]["tasks"].get(task, {}).get("teachers", {})):
                    raise ValueError("Invalid demonstration task/teacher provenance")
                groups[(task, source)].append(demo)
    accepted = []
    for (task, source), demos in sorted(groups.items()):
        passing = []
        counts[task]["inputs"] += 1
        threshold = cfg["quality"]["task_min_scores"].get(task, cfg["quality"]["min_score"])
        for demo in demos:
            counts[task]["generated"] += 1
            result = demo["demo_quality"]
            if not result["valid"] or dimension not in result.get("scores", {}):
                counts[task]["invalid"] += 1
            elif (not result["constraint_pass"] or result["scores"][dimension] < threshold
                  or not demo["response"].strip()):
                counts[task]["rejected"] += 1
            else:
                passing.append(demo)
        random.Random(int(fingerprint([cfg["seed"], dimension, task, source])[:16], 16)).shuffle(passing)
        passing.sort(key=lambda d: d["demo_quality"]["scores"][dimension], reverse=True)
        seen = set()
        for demo in passing:
            tokens = tuple(demo["response_token_ids"])
            if tokens in seen:
                continue
            seen.add(tokens)
            accepted.append({**demo, "dimensions": [dimension]})
            counts[task]["accepted_turns"] += 1
            if len(seen) >= cfg["quality"]["keep_per_prompt"]:
                break
    missing = sorted(set(cfg["routing"]["tasks"]) - {r["task_id"] for r in accepted})
    report = {"dimension": dimension, "tasks": {t: dict(c) for t, c in counts.items()},
              "accepted_turns": len(accepted), "missing_tasks": missing, "evaluator": cfg["quality"]["evaluator"],
              "input_hashes": {str(p): file_hash(p) for p in inputs},
              "selection": "Score on this dimension, across all supplied versions; exact teacher token provenance"}
    output = Path(output)
    atomic_json(Path(str(output) + ".report.json"), report)
    if missing:
        raise ValueError(f"Dimension {dimension}: no accepted demonstrations for {missing}; inspect {output}.report.json")
    temporary = output.with_suffix(".tmp")
    write_rows(temporary, accepted)
    temporary.replace(output)
    report["sha256"] = file_hash(output)
    atomic_json(Path(str(output) + ".report.json"), report)
    return report


class HierarchyRunner(Runner):
    @staticmethod
    def _smoke_command(command):
        def arg(name):
            return command[command.index(name) + 1]
        if "opd.hierarchy" in command:
            return select_dimension(load_config(arg("--config")), json.loads(Path(arg("--inputs")).read_text()), arg("--output"))
        if "opd.full_export" in command:
            from .full_export import export_full
            return export_full(arg("--model"), arg("--base"), arg("--output"), arg("--dtype"), resume="--resume" in command)
        return Runner._smoke_command(command)


def check_coverage(rows, tasks, label):
    coverage = defaultdict(Counter)
    for row in rows:
        if not row["dimensions"] or set(row["dimensions"]) - set(DIMENSIONS):
            raise ValueError(f"Invalid five-axis labels in {label}")
        coverage[row["task_id"]].update(row["dimensions"])
    if set(coverage) != set(tasks) or set(DIMENSIONS) - {d for c in coverage.values() for d in c}:
        raise ValueError(f"{label} must retain all tasks and all five dimensions; inspect coverage/length audit")
    return {t: dict(c) for t, c in coverage.items()}


def reusable_hierarchy_source(plan, root):
    """Validate a prior run before consuming any of its expensive artifacts."""
    root = Path(root).resolve()
    saved = hierarchy_manifest(root)
    if not saved:
        raise ValueError(f"Reusable hierarchy manifest not found: {root}")
    saved_identity, current_identity = trajectory_identity(saved["plan"]), trajectory_identity(plan)
    if saved_identity != current_identity:
        differences = identity_differences(saved_identity, current_identity)[:20]
        raise ValueError("Reusable trajectory inputs differ (data/experts/generation/judge/seed); "
                         "teacher trajectories must be regenerated. Differences: " +
                         json.dumps(differences, ensure_ascii=False, sort_keys=True))
    # New manifests record the actual source-data location.  Fall back to the
    # historical in-output layout so existing production runs stay reusable.
    data = Path(saved.get("data_path", root / "data"))
    hashes = {p.name: file_hash(p) for p in data.glob("*.jsonl")}
    if hashes != saved.get("data"):
        raise ValueError("Reusable prepared data is missing or changed")
    return root, saved, data


def valid_candidate_requests(requests):
    """Content-level validation independent of the old scheduler/stage metadata."""
    try:
        for request in requests:
            expected = list(read_rows(request["eval_file"]))
            records = list(read_rows(request["output"]))
            if [r["id"] for r in records] != [r["id"] for r in expected]:
                return False
            for record in records:
                candidates = record.get("candidates", [])
                if len(candidates) != request["candidates"]:
                    return False
                if any(c.get("demo_teacher") != request["teacher"] or
                       c.get("demo_input_id") != record["id"] for c in candidates):
                    return False
        return bool(requests)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def valid_selected_demos(path, requests):
    try:
        path = Path(path)
        saved = json.loads(Path(str(path) + ".report.json").read_text())
        return (saved["sha256"] == file_hash(path) and saved["input_hashes"] == {
            r["output"]: file_hash(r["output"]) for r in requests})
    except (OSError, ValueError, KeyError, TypeError):
        return False


def reusable_dimension_models(plan, source, seed):
    saved = hierarchy_manifest(source)
    if not saved or dimension_identity(saved["plan"]) != dimension_identity(plan):
        return None
    models = {}
    for dimension in DIMENSIONS:
        final = Path(source) / "runs" / f"seed_{seed}" / f"dimension_{dimension}" / "offline_forward" / "final"
        try:
            metadata = json.loads((final / "opd_metadata.json").read_text())
            if metadata["dimension"] != dimension or not (final / "adapter_config.json").is_file():
                return None
            if not any((final / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
                return None
        except (OSError, ValueError, KeyError):
            return None
        models[dimension] = str(final)
    return models


def run_hierarchy(plan, resume=False, dry_run=False, through="fusion"):
    from .full_export import complete_export
    plan = copy.deepcopy(plan)
    if through not in PHASES[1:]:
        raise ValueError(f"through must be one of {PHASES[1:]}")
    if plan["dimensions"] != list(DIMENSIONS) or plan["model"]["student_mode"] != "lora":
        raise ValueError("Five LoRA dimension experts are required")
    runner = HierarchyRunner(plan, resume, dry_run)
    reuse_root = plan.get("reuse_from")
    reused = reusable_hierarchy_source(plan, reuse_root) if reuse_root else None
    # A reuse run must consume the exact source data validated with the donor
    # trajectories, even when a fixture/custom plan also names prepared_data.
    data = Path(reused[2] if reused else (plan.get("prepared_data") or runner.root / "data"))
    provenance = {"plan": plan, "data_path": str(data.resolve()),
                  "data": {p.name: file_hash(p) for p in data.glob("*.jsonl")}}
    signature = fingerprint(provenance)
    manifest = runner.root / "hierarchy_manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    if previous and previous["signature"] != signature:
        if previous.get("data") != provenance["data"]:
            raise ValueError("Prepared data content changed; use a new output directory")
        if trajectory_identity(previous["plan"]) != trajectory_identity(plan):
            raise ValueError("Hierarchy trajectory inputs changed; use a new output directory")
        print("[retained] training/fusion settings changed; upstream trajectory identity is unchanged", flush=True)
    atomic_json(manifest, {"signature": signature, **provenance})
    if plan["num_processes"] > 1:
        launch = yaml.safe_load((PROJECT / "configs/accelerate_zero2.yaml").read_text())
        launch["num_processes"] = plan["num_processes"]
        plan["accelerate_config"] = str(runner.config("accelerate", launch))
    os.environ["OPD_NUM_PROCESSES"] = str(plan["num_processes"])
    base = base_config(plan, data)
    if not dry_run:
        data = tokenized_data(base, data, runner)
        base = base_config(plan, data)
    rows, evaluations = load_data(base["data"]["train_file"]), load_data(base["data"]["eval_file"])
    tasks = base["routing"]["tasks"]
    coverage = {"train": check_coverage(rows, tasks, "train"),
                "validation": check_coverage(evaluations, tasks, "validation")}
    results = {"workflow": "hierarchy_five", "base_model": base["model"]["base_model"],
               "dimensions": list(DIMENSIONS), "tasks": sorted(tasks), "active_checkpoints": len(base["teachers"]),
               "steps_per_dimension": plan["steps"]["distill"], "fusion_steps": plan["steps"]["final"],
               "optimizer_steps_per_seed": 5 * plan["steps"]["distill"] + plan["steps"]["final"],
               "scope": SCOPE, "coverage": coverage, "dry_run": dry_run, "through": through,
               "reuse_from": str(reused[0]) if reused else None, "seeds": {}}
    for seed in plan["seeds"]:
        base["seed"] = seed
        stem = f"seed_{seed}"
        train_inputs = bounded_rows(rows, plan["demo_limit_per_task"], seed)
        # Sample by task AND applicable-axis pattern so a rare T prefix cannot
        # disappear from validation or from the bounded candidate pool.
        def ensure_patterns(source, subset):
            represented = {(r["task_id"], tuple(r["dimensions"])) for r in subset}
            for row in source:
                pattern = (row["task_id"], tuple(row["dimensions"]))
                if pattern not in represented:
                    subset.append(row)
                    represented.add(pattern)
            return subset
        train_inputs = ensure_patterns(rows, train_inputs)
        validation = ensure_patterns(evaluations, bounded_rows(evaluations, plan["validation_limit_per_task"], seed))
        validation_tasks = check_coverage(validation, tasks, "bounded validation")
        for task in tasks:
            if set(coverage["train"][task]) - set(validation_tasks[task]):
                raise ValueError(f"Validation lacks trained dimensions for task {task}")
        stage_data = runner.root / "hierarchy_data" / stem
        write_rows(stage_data / "train.jsonl", rows)
        write_rows(stage_data / "validation.jsonl", validation)
        base["data"]["eval_file"] = str(stage_data / "validation.jsonl")
        trajectory_root = reused[0] if reused else runner.root
        requests = []
        for task in sorted(tasks):
            inputs = trajectory_root / "demo_inputs" / stem / (task + ".jsonl")
            if not reused:
                write_rows(inputs, [r for r in train_inputs if r["task_id"] == task])
            for teacher in tasks[task]["teachers"]:
                name = stem + "/" + task + "/" + teacher
                requests.append({"name": name, "teacher": teacher, "eval_file": str(inputs),
                    "mode": "demonstrations", "candidates": plan["demo_candidates"],
                    "output": str(trajectory_root / "demo_candidates" / (name + ".jsonl"))})
        atomic_json(runner.root / "demo_budget.json", {
            "candidate_generations": sum(sum(r["task_id"] == t for r in train_inputs) * len(tasks[t]["teachers"])
                                         * plan["demo_candidates"] for t in tasks),
            "judge_requests_before_cache_or_retry": sum(sum(r["task_id"] == t for r in train_inputs) * len(tasks[t]["teachers"])
                * plan["demo_candidates"] for t in tasks if t not in VERIFIABLE_AXIS),
            "training_loss_calls_external_judge": False})
        if valid_candidate_requests(requests):
            print(f"[retained] validated teacher trajectories from {trajectory_root}", flush=True)
        elif reused:
            raise ValueError(f"Reusable teacher trajectories are incomplete or corrupt: {trajectory_root}")
        else:
            runner.baseline_pool(stem + "/teacher_demonstrations", base, requests)
        input_file = runner.root / "configs" / (stem + "_demo_inputs.json")
        runner.write(input_file, json.dumps([r["output"] for r in requests]))

        demos_by_dimension = {}
        for dimension in DIMENSIONS:
            cfg = dimension_config(base, dimension, rows, base["model"]["lora_rank"], seed)
            cfg["model"]["lora_alpha"] = base["model"]["lora_alpha"]
            demos = runner.root / "demos" / stem / (dimension + ".jsonl")
            demos_by_dimension[dimension] = (demos, cfg)
            selection = "demo_selection/" + stem + "/" + dimension
            if valid_selected_demos(demos, requests):
                print(f"[retained] selected {dimension} trajectories", flush=True)
                continue
            selection_signature = {"trajectory_identity": trajectory_identity(plan),
                                   "dimension": dimension,
                                   "candidate_hashes": {r["output"]: (file_hash(r["output"])
                                       if Path(r["output"]).is_file() else None) for r in requests}}
            record = runner.state.get(selection)
            if record and record["signature"] != fingerprint(selection_signature):
                # Selection is a cheap deterministic derivative of already
                # validated candidates; an obsolete stage record need not force
                # expensive teacher generation again.
                runner.state.pop(selection)
                runner.write(runner.state_path, json.dumps(runner.state, indent=2))
            path = runner.config(selection, cfg)
            runner.command(selection, [sys.executable, "-m", "opd.hierarchy", "select", "--config", path,
                "--inputs", input_file, "--output", demos], [demos, str(demos) + ".report.json"],
                selection_signature)

        if through == "trajectories":
            results["seeds"][stem] = {"trajectories": {d: str(v[0]) for d, v in demos_by_dimension.items()}}
            continue

        trained, dimension_results = {}, {}
        imported = reusable_dimension_models(plan, reused[0], seed) if reused else None
        if imported:
            print(f"[retained] validated five dimension experts from {reused[0]}", flush=True)
        for dimension in DIMENSIONS:
            demos, cfg = demos_by_dimension[dimension]
            training = configure_method(cfg, "offline_forward", str(demos), plan["steps"])
            name = stem + "/dimension_" + dimension + "/offline_forward"
            if imported:
                final = imported[dimension]
            else:
                final, _ = runner.train(name, training)
            trained[dimension] = final
            dimension_results[dimension] = {"model": final, "steps": plan["steps"]["distill"],
                                             "tasks": sorted(cfg["routing"]["tasks"]),
                                             "reused": bool(imported)}
            if plan.get("evaluate", True) and not imported:
                dimension_results[dimension]["evaluation"] = runner.evaluate(name, cfg, model=final)
        if through == "dimensions":
            results["seeds"][stem] = {"dimensions": dimension_results}
            continue

        unified = final_config(base, trained, stage_data, runner, stem, "dimensions")
        unified["data"].pop("calibration_file", None)
        unified["data"]["demo_file"] = None
        # Keep dimension training on fixed teacher trajectories (forward KL),
        # but let the fusion section choose the on-policy objective explicitly.
        # This supports MOPD-style sampled reverse KL without changing the five
        # upstream capability experts.
        unified["train"].update(plan["fusion"])
        unified["train"].update(max_steps=plan["steps"]["final"], trajectory_source="student")
        unified["rollout"]["generation_batch_size"] = unified["train"]["batch_size"]
        validate(unified)
        final, trained_cfg = runner.train(stem + "/unified_opd", unified)
        export = runner.root / ("full_model" if len(plan["seeds"]) == 1 else stem + "/full_model")
        export_stage = "export/" + stem
        record = runner.state.get(export_stage)
        if resume and record and record["status"] == "complete" and not complete_export(export):
            record["status"] = "failed"
        runner.command(export_stage, [sys.executable, "-m", "opd.full_export", "--model", final,
            "--base", base["model"]["base_model"], "--output", export, "--dtype", base["model"]["dtype"],
            *(["--resume"] if resume else [])], [export / "export_manifest.json"],
            {"source": final, "signature": signature})
        result = {"dimensions": dimension_results, "training_checkpoint": final, "full_model": str(export)}
        if plan.get("evaluate", True):
            result["evaluation"] = runner.evaluate(stem + "/unified_full_model", trained_cfg, model=str(export))
        results["seeds"][stem] = result
    results["commands_this_invocation"] = len(runner.commands)
    atomic_json(runner.root / ("plan_summary.json" if dry_run else "results.json"), results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["train", "plan", "select", *PHASES])
    parser.add_argument("--config", default=str(PROJECT / "configs/hierarchy_five.yaml"))
    parser.add_argument("--output")
    parser.add_argument("--inputs")
    parser.add_argument("--reuse-from", help="Validated prior hierarchy output supplying prepared data, trajectories, and compatible dimension experts")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--dimension-steps", type=int, default=200)
    parser.add_argument("--fusion-steps", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.operation == "select":
        print(json.dumps(select_dimension(load_config(args.config), json.loads(Path(args.inputs).read_text()), args.output)))
        return
    plan = load_hierarchy_plan(args.config, output=args.output, processes=args.num_processes,
                               dimension_steps=args.dimension_steps, fusion_steps=args.fusion_steps,
                               allow_missing_judge=args.operation == "plan")
    if args.reuse_from:
        plan["reuse_from"] = str(Path(args.reuse_from).resolve())
    os.environ["SIMULATION_REPO"] = plan["upstream_repo"]
    if args.operation == "plan":
        # No files/models/data/API required to inspect budgets, assignments and paths.
        print(json.dumps({"plan": plan, "scope": SCOPE, "path_status": "not checked"}, indent=2, ensure_ascii=False))
        return
    root = Path(plan["output_dir"])
    if root.exists() and any(root.iterdir()) and not (root / "hierarchy_manifest.json").exists():
        # Source preparation/probe can be interrupted before the main manifest.
        allowed = {".hierarchy.lock", "data", "judge_cache", "judge_probe.json"}
        if any(p.name not in allowed and not p.name.startswith("data.incomplete.") for p in root.iterdir()):
            raise FileExistsError("Output belongs to another run; choose a new hierarchy output directory")
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / ".hierarchy.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another hierarchy coordinator is already using this output directory") from None
    check_inputs(plan)
    if args.operation == "prepare":
        if args.reuse_from:
            raise ValueError("prepare creates source data; --reuse-from is for later phases")
        print("[prepare] Rendering original actor prompts and auditing visible evidence", flush=True)
        data = prepare_data(plan, args.resume)
        print(json.dumps({"phase": "prepare", "data": str(data)}, indent=2))
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.num_processes:
        raise RuntimeError("Not enough visible GPUs for the configured ranks")
    os.environ["OPD_JUDGE_CACHE_DIR"] = str(root / "judge_cache")
    if args.reuse_from:
        print(f"[reuse] Validating reusable artifacts from {args.reuse_from}", flush=True)
    else:
        print("[prepare] Rendering original actor prompts and auditing visible evidence", flush=True)
        prepare_data(plan, args.resume)
    from .hierarchy_judge import probe
    atomic_json(root / "judge_probe.json", probe())
    print("[judge] Ordinary chat completions + JSON score parsing passed", flush=True)
    through = "fusion" if args.operation == "train" else args.operation
    print(json.dumps(run_hierarchy(plan, args.resume, through=through), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
