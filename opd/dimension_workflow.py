"""Task teachers -> fixed objective-filtered demonstrations -> independent LoRAs.

One offline forward-KL method, no student rollouts, QGPI selection or fusion.
The existing persistent GPU pool supplies resumable, batched teacher sampling.
"""

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from .checkpoints import archive_incomplete, atomic_json
from .data import load_data, read_rows, tokenizer_signature, write_rows
from .evaluation import evaluate_response, scalar_score
from .experiments import Runner, base_config, configure_method, dimension_config, fingerprint, tokenized_data


def validate_plan(plan):
    dims = plan.get("dimensions", [])
    if not dims or len(set(dims)) != len(dims) or set(dims) - {"F", "S", "N"}:
        raise ValueError("dimension_offline currently supports explicitly selected F/S/N; T/C excluded, U pending")
    if plan.get("methods", ["offline_forward"]) != ["offline_forward"] or plan.get("qgpi", {}).get("enabled"):
        raise ValueError("dimension_offline requires only offline_forward and qgpi.enabled=false")
    if plan.get("model", {}).get("student_mode") != "lora":
        raise ValueError("dimension_offline exports independent LoRA experts; student_mode must be lora")
    if plan.get("baseline_scheduler", "task_pool") != "task_pool":
        raise ValueError("dimension_offline uses baseline_scheduler=task_pool for parallel teacher demonstrations")
    if not plan.get("quality", {}).get("filter_demos", True):
        raise ValueError("Dimension teacher demonstrations must be quality filtered")
    for key in ("demo_candidates", "demo_limit_per_task", "validation_limit_per_task"):
        if type(plan.get(key)) is not int or plan[key] < 1:
            raise ValueError(f"{key} must be positive (bounded first trial)")
    for dim, steps in plan.get("dimension_steps", {}).items():
        if dim not in dims or type(steps) is not int or steps < 1:
            raise ValueError("dimension_steps must contain positive budgets for selected dimensions")
    if plan.get("evaluation_split", "validation") != "validation":
        raise ValueError("First dimension trial uses validation, never the held-out test set")


def generate_candidates(cfg, model, tokenizer, job, device):
    """One result bundle per input even when all sampled responses are invalid."""
    from .rollout import collect_episodes
    signature = tokenizer_signature(tokenizer, cfg)
    generation = copy.deepcopy(cfg)
    generation["rollout"]["generation_batch_size"] = cfg["inference"]["batch_size"]
    records = [{"id": row["id"], "task_id": row["task_id"], "candidates": []} for row in job["rows"]]
    size = cfg["inference"]["batch_size"]
    for trial in range(job["candidates"]):
        for start in range(0, len(records), size):
            rows = job["rows"][start:start + size]
            episodes = collect_episodes(model, rows, tokenizer, generation, device)
            for offset, (row, (transitions, response)) in enumerate(zip(rows, episodes, strict=True)):
                result = evaluate_response(cfg, row, response)
                if len(transitions) != 1:
                    raise ValueError("Dimension demo pool requires a single teacher decision")
                transition = transitions[0]
                demo = {**row, "id": f"{row['id']}:demo:{job['teacher']}:{trial}:0",
                        "messages": transition["messages"],
                        "response": tokenizer.decode(transition["response_ids"], skip_special_tokens=True),
                        "response_token_ids": transition["response_ids"],
                        "prompt_token_ids": transition["prompt_ids"], "tokenizer_signature": signature,
                        "demo_teacher": job["teacher"], "demo_trial": trial, "demo_quality": result,
                        "source_id": row.get("source_id", row["id"]), "demo_input_id": row["id"]}
                records[start + offset]["candidates"].append(demo)
    return records


def select_demos(cfg, inputs, output):
    """Compare versions only on this training prompt, preserving token provenance."""
    grouped = defaultdict(list)
    counts = defaultdict(lambda: {"inputs": 0, "generated": 0, "invalid": 0, "rejected": 0,
                                  "accepted_inputs": 0, "accepted_turns": 0})
    for path in inputs:
        for record in read_rows(path):
            grouped[(record["task_id"], record["id"])].extend(record["candidates"])
    accepted = []
    for (task, row_id), demos in sorted(grouped.items()):
        counts[task]["inputs"] += 1
        threshold = cfg["quality"]["task_min_scores"].get(task, cfg["quality"]["min_score"])
        passing = []
        for demo in demos:
            if (demo["task_id"] != task or demo["demo_input_id"] != row_id or
                    demo["demo_teacher"] not in cfg["routing"]["tasks"][task]["teachers"]):
                raise ValueError("Corrupt teacher demonstration provenance")
            counts[task]["generated"] += 1
            result = demo["demo_quality"]
            if not result["valid"]:
                counts[task]["invalid"] += 1
                continue
            if not result["constraint_pass"] or scalar_score(cfg, result) < threshold or not demo["response"].strip():
                counts[task]["rejected"] += 1
                continue
            passing.append(demo)
        # Reproducible ties without systematic preference for manifest order.
        seed = int(fingerprint([cfg["seed"], task, row_id])[:16], 16)
        random.Random(seed).shuffle(passing)
        passing.sort(key=lambda d: scalar_score(cfg, d["demo_quality"]), reverse=True)
        seen = set()
        for demo in passing:
            tokens = tuple(demo["response_token_ids"])
            if tokens in seen:
                continue
            seen.add(tokens)
            accepted.append(demo)
            counts[task]["accepted_turns"] += 1
            if len(seen) >= cfg["quality"]["keep_per_prompt"]:
                break
        counts[task]["accepted_inputs"] += bool(seen)
    missing = sorted(set(cfg["routing"]["tasks"]) - {r["task_id"] for r in accepted})
    report = {"tasks": dict(counts), "accepted_turns": len(accepted), "missing_tasks": missing,
              "thresholds": cfg["quality"]["task_min_scores"], "default_threshold": cfg["quality"]["min_score"],
              "selection": "objective-filtered, ranked and deduplicated across all compatible checkpoint versions",
              "evaluator": cfg["quality"]["evaluator"],
              "input_hashes": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs}}
    atomic_json(Path(str(output) + ".report.json"), report)
    if missing:
        raise ValueError(f"No accepted teacher demonstrations for {missing}; inspect {output}.report.json. "
                         "Do not silently drop tasks; check teacher quality, labels, lengths and sampling budget.")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    write_rows(temporary, accepted)
    if destination.exists():
        archive_incomplete(destination)
    temporary.replace(destination)
    report["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    atomic_json(Path(str(output) + ".report.json"), report)
    return report


def bounded_rows(rows, limit, seed):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["task_id"]].append(row)
    selected = []
    for task, values in sorted(buckets.items()):
        random.Random(int(fingerprint([seed, task])[:16], 16)).shuffle(values)
        selected.extend(values[:limit])
    return selected


def run_dimensions(plan, resume=False, dry_run=False):
    validate_plan(plan)
    runner = Runner(plan, resume, dry_run)
    data = Path(plan.get("prepared_data") or runner.root / "data")
    cfg = base_config(plan, data)
    if cfg["qgpi"]["enabled"]:
        raise ValueError("Disable QGPI for dimension off-policy distillation")
    provenance = {"plan": plan, "data": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(data.glob("*.jsonl"))}}
    signature = fingerprint(provenance)
    manifest = runner.root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["signature"] != signature:
        raise ValueError("Dimension plan/data changed; use a new output directory")
    atomic_json(manifest, {"signature": signature, **provenance})
    for key in ("recipe_alignment", "expert_inventory"):
        if plan.get(key):
            atomic_json(runner.root / (key + ".json"), plan[key])
    if not dry_run:
        data = tokenized_data(cfg, data, runner)
        cfg = base_config(plan, data)
    rows = load_data(cfg["data"]["train_file"])
    eval_rows = load_data(cfg["data"]["eval_file"])
    tasks = set(cfg["routing"]["tasks"])
    for split, source in (("train", rows), ("validation", eval_rows)):
        if tasks != {r["task_id"] for r in source}:
            raise ValueError(f"Dimension {split} must cover exactly the routed tasks after length filtering")
        if any(len(r["dimensions"]) != 1 or r["dimensions"][0] not in plan["dimensions"] for r in source):
            raise ValueError("Each objective row must label exactly one selected dimension")
        if cfg["quality"]["evaluator"] == "opd.objective:objective_response":
            from .objective import TASK_AXES
            if any(r["dimensions"] != [TASK_AXES[r["task_id"]]] for r in source):
                raise ValueError("Prepared task labels differ from the audited objective dimension")
    budgets = {d: plan.get("dimension_steps", {}).get(d, plan["steps"]["distill"]) for d in plan["dimensions"]}
    results, stages = {}, []
    for seed in plan["seeds"]:
        stem = f"seed_{seed}"
        base = copy.deepcopy(cfg)
        base["seed"] = seed
        training_inputs = bounded_rows(rows, plan["demo_limit_per_task"], seed)
        validation = bounded_rows(eval_rows, plan["validation_limit_per_task"], seed)
        eval_path = runner.root / "eval_data" / (stem + ".jsonl")
        write_rows(eval_path, validation)
        base["data"]["eval_file"] = str(eval_path)
        requests = []
        for task in sorted(tasks):
            task_path = runner.root / "demo_inputs" / stem / (task + ".jsonl")
            write_rows(task_path, [r for r in training_inputs if r["task_id"] == task])
            for teacher in base["routing"]["tasks"][task]["teachers"]:
                name = stem + "/" + task + "/" + teacher
                requests.append({"name": name, "teacher": teacher, "eval_file": str(task_path),
                                 "mode": "demonstrations", "candidates": plan["demo_candidates"],
                                 "output": str(runner.root / "demo_candidates" / (name + ".jsonl"))})
        if plan.get("val_before_train", False) and plan.get("evaluate", True):
            baselines = [{"name": stem + "/base", "model": base["model"]["base_model"],
                          "eval_file": str(eval_path), "output": str(runner.root / "evaluation" / stem / "base.jsonl")}]
            for request in requests:
                task = request["name"].split("/")[1]
                path = runner.root / "eval_data" / stem / (task + ".jsonl")
                write_rows(path, [r for r in validation if r["task_id"] == task])
                baselines.append({"name": request["name"], "teacher": request["teacher"], "eval_file": str(path),
                                  "output": str(runner.root / "evaluation" / (request["name"] + ".jsonl"))})
            runner.baseline_pool(stem, base, baselines)
        runner.baseline_pool(stem + "/teacher_demonstrations", base, requests)
        demos = runner.root / "demos" / (stem + ".jsonl")
        input_file = runner.root / "configs" / (stem + "_demo_inputs.json")
        runner.write(input_file, json.dumps([r["output"] for r in requests]))
        path = runner.config(stem + "/demo_selection", base)
        record = runner.state.get("demo_selection/" + stem)
        if resume and record and record["status"] == "complete":
            try:
                saved = json.loads(Path(str(demos) + ".report.json").read_text())
                valid = (saved["sha256"] == hashlib.sha256(demos.read_bytes()).hexdigest() and
                         saved["input_hashes"] == {r["output"]: hashlib.sha256(Path(r["output"]).read_bytes()).hexdigest()
                                                   for r in requests})
            except (OSError, ValueError, KeyError):
                valid = False
            if not valid:
                record["status"] = "failed"
        runner.command("demo_selection/" + stem, [sys.executable, "-m", "opd.dimension_workflow", "select",
                       "--config", path, "--inputs", input_file, "--output", demos],
                       [demos, str(demos) + ".report.json"], {"config": base, "requests": requests})
        results[stem] = {}
        for dimension in plan["dimensions"]:
            dim_cfg = dimension_config(base, dimension, rows, base["model"]["lora_rank"], seed)
            dim_cfg["model"]["lora_alpha"] = base["model"]["lora_alpha"]
            train_cfg = configure_method(dim_cfg, "offline_forward", str(demos), {"distill": budgets[dimension]})
            name = stem + "/dimension_" + dimension + "/offline_forward"
            stages.append({"stage": name, "dimension": dimension, "steps": budgets[dimension],
                           "tasks": sorted(dim_cfg["routing"]["tasks"]), "trajectory_source": "teacher"})
            model, _ = runner.train(name, train_cfg)
            result = {"model": model, "tasks": stages[-1]["tasks"], "steps": budgets[dimension]}
            if plan.get("evaluate", True):
                result["evaluation"] = runner.evaluate(name, dim_cfg, model=model)
            results[stem][dimension] = result
    summary = {"workflow": "dimension_offline", "dry_run": dry_run, "output_dir": str(runner.root),
               "training_order": ["reuse_task_experts", "teacher_demo_pool", "objective_selection",
                                  *["dimension_" + d for d in plan["dimensions"]]],
               "dimension_generation_experts": True, "active_dimensions": plan["dimensions"],
               "excluded_dimensions": ["T", "C"], "pending_dimensions": {"U": "No integrated objective supervision"},
               "tasks": sorted(tasks), "active_checkpoints": len(cfg["teachers"]),
               "methods": ["offline_forward"], "automatic_fusion": False, "training": stages,
               "optimizer_steps_per_seed": sum(budgets.values()), "commands_this_invocation": len(runner.commands),
               "validation_before_train": plan.get("val_before_train", False),
               "demo_limit_per_task": plan["demo_limit_per_task"], "demo_candidates_per_checkpoint": plan["demo_candidates"],
               "scope": "N is MirrorBench first-turn lexical proxy, not full behavioral realism; U is not trained."}
    if cfg["quality"]["evaluator"] == "opd.objective:objective_response":
        from .objective import METRICS
        summary.update(requires_llm_judge=False, task_metrics={t: METRICS[t] for t in sorted(tasks)})
    else:
        summary["warning"] = "Synthetic mechanics fixture only; not an objective capability result"
    atomic_json(runner.root / "plan_summary.json", summary)
    if not dry_run:
        atomic_json(runner.root / "results.json", results)
    return summary


def main():
    from .config import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["select"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(select_demos(load_config(args.config), json.loads(Path(args.inputs).read_text()), args.output), indent=2))


if __name__ == "__main__":
    main()
