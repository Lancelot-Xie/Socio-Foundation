"""Teacher demonstrations, paired calibration, evaluation and model export."""

import copy
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from accelerate.utils import set_seed

from .data import load_data, normalize_row, read_rows, tokenizer_signature, write_rows
from .evaluation import evaluate_response, scalar_score
from .models import TeacherBank, adapter_path, load_causal, load_student, load_tokenizer
from .rollout import collect_episode, collect_episodes
from .routing import Router


def inference_device():
    # CPU fallback is intentional for tiny reproducible smoke tests, including on macOS.
    cpu = os.environ.get("ACCELERATE_USE_CPU", "").lower() in ("1", "true", "yes")
    return torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")


def new_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def prepare_data(source, output, task_id=None, dimensions=None, messages_field="messages"):
    rows = []
    for i, original in enumerate(read_rows(source)):
        row = dict(original)
        if task_id:
            row["task_id"] = task_id
        if dimensions is not None:
            row["dimensions"] = dimensions
        value = original
        for part in messages_field.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is not None:
            row["messages"] = value
        rows.append(normalize_row(row, i))
    new_output(output)
    write_rows(output, rows)
    return {"rows": len(rows), "output": output}


def build_demos(cfg, output, candidates=1):
    if candidates < 1:
        raise ValueError("candidates must be positive")
    new_output(output)
    set_seed(cfg["seed"])
    rows = load_data(cfg["data"]["train_file"], cfg["data"]["dimension"])
    tokenizer, device = load_tokenizer(cfg), inference_device()
    signature = tokenizer_signature(tokenizer, cfg)
    bank, router = TeacherBank(cfg, tokenizer, device), Router(cfg)
    accepted, counts = [], defaultdict(int)
    rng = random.Random(cfg["seed"])
    task_counts = defaultdict(lambda: {"inputs": 0, "accepted_inputs": 0, "accepted_turns": 0})
    for row in rows:
        task_counts[row["task_id"]]["inputs"] += 1
        weights, strength = router.resolve(row)
        if strength == 0:
            counts["inactive"] += 1
            continue
        options = []
        for teacher in weights:
            model = bank.activate(teacher)
            for trial in range(candidates):
                transitions, response = collect_episode(model, row, tokenizer, cfg, bank.device)
                result = (evaluate_response(cfg, row, response) if cfg["quality"]["filter_demos"] else
                          {"valid": False, "reason": "Unfiltered teacher sampling; quality not evaluated"})
                counts["generated"] += 1
                if cfg["quality"]["filter_demos"] and not result["valid"]:
                    counts["judge_invalid"] += 1
                    continue
                score = scalar_score(cfg, result) if cfg["quality"]["filter_demos"] else 0.0
                if cfg["quality"]["filter_demos"] and (not result["constraint_pass"] or score < cfg["quality"]["min_score"]):
                    counts["quality_rejected"] += 1
                    continue
                options.append((score, teacher, trial, transitions, result))
        if not options:
            continue
        rng.shuffle(options)  # Avoid favoring the first registered teacher when scores tie.
        retained, seen = [], set()
        for option in sorted(options, key=lambda x: x[0], reverse=True):
            key = tuple(tuple(t["response_ids"]) for t in option[3])
            if key not in seen:
                retained.append(option)
                seen.add(key)
            if len(retained) >= cfg["quality"]["keep_per_prompt"]:
                break
        task_counts[row["task_id"]]["accepted_inputs"] += 1
        for _, teacher, trial, transitions, result in retained:
            for transition in transitions:
                # Preserve EXACT rendered teacher prefix for subsequent SFT tokenization.
                # Multi-turn histories come from the actual teacher/environment interaction.
                demo = {**row, "id": f"{row['id']}:demo:{teacher}:{trial}:{transition['turn']}",
                        "messages": transition["messages"],
                        "response": tokenizer.decode(transition["response_ids"], skip_special_tokens=True),
                        "response_token_ids": transition["response_ids"],
                        "prompt_token_ids": transition["prompt_ids"], "tokenizer_signature": signature,
                        "demo_teacher": teacher, "demo_trial": trial, "demo_quality": result,
                        "source_id": row.get("source_id", row["id"])}
                if demo["response"].strip():
                    accepted.append(demo)
                    task_counts[row["task_id"]]["accepted_turns"] += 1
    if not accepted:
        raise ValueError(f"No demonstrations passed quality filtering: {dict(counts)}")
    write_rows(output, accepted)
    report = {**dict(counts), "accepted_turns": len(accepted), "output": str(output),
              "filter_demos": cfg["quality"]["filter_demos"], "tasks": dict(task_counts)}
    Path(str(output) + ".report.json").write_text(json.dumps(report, indent=2))
    return report


def paired_summary(differences, min_samples, margin):
    """Each number is the mean paired difference for one distinct sample, not one rollout."""
    n = len(differences)
    mean = statistics.mean(differences) if n else 0.0
    # A normal-approximation diagnostic, not a claim of small-sample exact coverage.
    error = statistics.stdev(differences) / math.sqrt(n) if n > 1 else None
    lower = mean - 1.96 * error if error is not None and n >= min_samples else None
    gain = max(0.0, lower-margin) if lower is not None else 0.0
    return {"samples": n, "mean_gain": mean, "lower_bound_approx": lower, "positive_gain": gain}


def calibrate(cfg, output, repeats=2, min_samples=8):
    if repeats < 1 or min_samples < 2:
        raise ValueError("Need repeats>=1 and min_samples>=2")
    output = new_output(output)
    if not cfg["data"].get("calibration_file"):
        raise ValueError("data.calibration_file is required; do not calibrate on the final test set")
    set_seed(cfg["seed"])
    rows = load_data(cfg["data"]["calibration_file"], cfg["data"]["dimension"])
    tokenizer, device = load_tokenizer(cfg), inference_device()
    student = load_student(cfg).to(device).requires_grad_(False).eval()
    bank = TeacherBank(cfg, tokenizer, device)
    scores = defaultdict(lambda: defaultdict(list))
    invalid, violations, details = defaultdict(int), defaultdict(int), []
    for index, row in enumerate(rows):
        task = row["task_id"]
        if task not in cfg["routing"]["tasks"]:
            raise ValueError(f"No candidate teachers for calibration task {task}")
        candidate_ids = list(cfg["routing"]["tasks"][task]["teachers"])
        paired = defaultdict(list)
        for trial in range(repeats):
            seed = cfg["seed"] + index * repeats + trial
            set_seed(seed)
            _, response = collect_episode(student, row, tokenizer, cfg, device)
            student_eval = evaluate_response(cfg, row, response)
            for teacher in candidate_ids:
                set_seed(seed)
                _, response = collect_episode(bank.activate(teacher), row, tokenizer, cfg, bank.device)
                teacher_eval = evaluate_response(cfg, row, response)
                details.append({"id": row["id"], "task_id": task, "teacher": teacher, "trial": trial,
                                "student": student_eval, "teacher_result": teacher_eval})
                if not student_eval["valid"] or not teacher_eval["valid"]:
                    invalid[(task, teacher)] += 1
                    continue
                if not teacher_eval["constraint_pass"]:
                    violations[(task, teacher)] += 1
                paired[teacher].append(scalar_score(cfg, teacher_eval) - scalar_score(cfg, student_eval))
        for teacher, diffs in paired.items():
            if diffs:
                scores[task][teacher].append(statistics.mean(diffs))
    routes = {}
    for task, route in cfg["routing"]["tasks"].items():
        summaries, gains = {}, {}
        for teacher in route["teachers"]:
            summary = paired_summary(scores[task][teacher], min_samples, cfg["quality"]["min_gain"])
            summary["judge_invalid"] = invalid[(task, teacher)]
            summary["constraint_failures"] = violations[(task, teacher)]
            summaries[teacher] = summary
            gains[teacher] = summary["positive_gain"] if not violations[(task, teacher)] else 0.0
        # A configurable scale keeps alpha normalization from canceling the absolute quality gain.
        scale = cfg["quality"].get("gain_for_full_strength", 0.1)
        if scale <= 0:
            raise ValueError("gain_for_full_strength must be positive")
        routes[task] = {"gains": gains, "strength": min(1.0, max(gains.values(), default=0.0)/scale),
                        "diagnostics": summaries}
    result = {"dimension": cfg["data"]["dimension"], "student_init": cfg["model"].get("student_init", cfg["model"]["base_model"]),
              "calibration_file": cfg["data"]["calibration_file"], "repeats": repeats,
              "min_distinct_samples": min_samples, "routes": routes}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    write_rows(str(output) + ".details.jsonl", details)
    return result


def evaluate_model(cfg, output, teacher_id=None, model_path=None):
    from .inference import inference_workers
    with inference_workers() as workers:
        return _evaluate_model(cfg, output, teacher_id, model_path, workers)


def _evaluate_model(cfg, output, teacher_id, model_path, workers):
    started = time.monotonic()
    new_output(output)
    cfg = copy.deepcopy(cfg)
    if model_path and teacher_id:
        raise ValueError("Choose either --model or --teacher")
    if not cfg["data"].get("eval_file"):
        raise ValueError("data.eval_file is required")
    set_seed(cfg["seed"])
    rows = load_data(cfg["data"]["eval_file"], cfg["data"]["dimension"])
    tokenizer, device = load_tokenizer(cfg), inference_device()
    if teacher_id:
        teacher_cfg = copy.deepcopy(cfg)
        teacher_cfg["teachers"] = [t for t in cfg["teachers"] if t["id"] == teacher_id]
        bank = TeacherBank(teacher_cfg, tokenizer, device)
        model, device = bank.activate(teacher_id), bank.device
    else:
        trained = Path(cfg["output_dir"]) / "final"
        if model_path:
            cfg["model"]["student_init"] = str(Path(model_path).resolve()) if Path(model_path).exists() else model_path
        elif (trained / "opd_metadata.json").exists():
            cfg["model"]["student_init"] = str(trained)
        model = load_student(cfg).to(device).requires_grad_(False).eval()
    local_rows = workers.shard(rows)
    print(f"[inference] rank={workers.rank}/{workers.world_size} device={device} rows={len(local_rows)} "
          f"batch={cfg['inference']['batch_size']}", flush=True)
    records = workers.gather(evaluate_rows(cfg, model, tokenizer, local_rows, device))
    if not workers.main:
        return {}
    order = {row["id"]: i for i, row in enumerate(rows)}
    if len(records) != len(rows) or len({r["id"] for r in records}) != len(rows):
        raise ValueError("Distributed evaluation has missing or duplicate rows")
    records.sort(key=lambda r: order[r["id"]])
    report = summarize_evaluation(cfg, records)
    write_rows(output, records)
    Path(str(output) + ".summary.json").write_text(json.dumps(report, indent=2))
    Path(str(output) + ".model.json").write_text(json.dumps({"teacher_id": teacher_id,
        "student_checkpoint": None if teacher_id else cfg["model"].get("student_init", cfg["model"]["base_model"]),
        "elapsed_seconds": time.monotonic() - started, "world_size": workers.world_size,
        "batch_size_per_rank": cfg["inference"]["batch_size"]}, indent=2))
    return report


@torch.no_grad()
def evaluate_rows(cfg, model, tokenizer, rows, device):
    """Same greedy rollout and evaluator for baseline, periodic and final reports."""
    cfg = copy.deepcopy(cfg)
    size = cfg["inference"]["batch_size"]
    cfg["rollout"]["generation_batch_size"] = size
    records = []

    def episodes():
        for start in range(0, len(rows), size):
            chunk = rows[start:start + size]
            yield from zip(chunk, collect_episodes(model, chunk, tokenizer, cfg, device, sample=False), strict=True)

    for row, (transitions, response) in episodes():
        if cfg.get("qgpi", {}).get("enabled"):
            from .quality_basis import evaluate_candidates
            trajectory = [{"messages": t["messages"], "response": tokenizer.decode(t["response_ids"], skip_special_tokens=True),
                           **{k: t[k] for k in ("next_messages", "done", "reward") if k in t}} for t in transitions]
            spec, measured = evaluate_candidates(cfg, row, transitions[0]["messages"],
                                                 [{"response": response, "trajectory": trajectory}])
            result = {**measured[0], "applicability": spec}
            result["reliable"] = (result["valid"] and result["confidence"] >= cfg["qgpi"]["min_confidence"] and
                                  result["disagreement"] <= cfg["qgpi"]["max_disagreement"])
        else:
            result = evaluate_response(cfg, row, response)
        records.append({"id": row["id"], "task_id": row["task_id"], "response": response, "evaluation": result,
                        "group_id": row.get("group_id", row["id"])})
        if cfg["qgpi"]["evaluator"] == "opd.objective:objective_candidates" and row["task_id"] == "userllm":
            info = row.get("evaluator_context", {}).get("original_row", {}).get("extra_info", {})
            records[-1]["metric_context"] = {"is_first_turn": info.get("conversation_history") == ""}
    return records


def summarize_evaluation(cfg, records):
    groups = defaultdict(list)
    for record in records:
        groups[record["task_id"]].append(record["evaluation"])
    report = {}
    for task, results in groups.items():
        valid = [r for r in results if r["valid"]]
        dimensions = sorted({k for r in valid for k in r["scores"]})
        report[task] = {"total": len(results), "judge_valid": len(valid),
                        "judge_reliable": sum(r.get("reliable", r["valid"]) for r in results),
                        "constraint_failures": sum(not r["constraint_pass"] for r in valid),
                        "scores": {d: statistics.mean([r["scores"][d] for r in valid if d in r["scores"]])
                                   for d in dimensions},
                        "score_counts": {d: sum(d in r["scores"] for r in valid) for d in dimensions}}
        task_metrics = sorted({k for r in valid for k, value in r.get("task_metrics", {}).items()
                               if isinstance(value, (int, float)) and math.isfinite(value)})
        report[task]["task_metrics"] = {k: statistics.mean(r["task_metrics"][k] for r in valid
            if k in r.get("task_metrics", {}) and isinstance(r["task_metrics"][k], (int, float))
            and math.isfinite(r["task_metrics"][k])) for k in task_metrics}
        if (cfg["qgpi"]["evaluator"] == "opd.objective:objective_candidates" or
                cfg["quality"]["evaluator"] == "opd.objective:objective_response"):
            from .objective_metrics import aggregate_partial_metrics
            report[task]["aggregate_metrics"] = aggregate_partial_metrics(task, [r for r in records if r["task_id"] == task])
            report[task]["requires_llm_judge"] = False
    return report


def merge_adapter(base, adapter, output, dtype="float32"):
    from peft import PeftModel
    new_output(output)
    model = PeftModel.from_pretrained(load_causal(base, dtype), adapter_path(adapter)).merge_and_unload()
    model.save_pretrained(output, safe_serialization=True)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(base).save_pretrained(output)
    return {"output": str(output)}


def inspect_config(cfg):
    rows = load_data(cfg["data"]["train_file"], cfg["data"]["dimension"], cfg["train"]["stage"] == "warmup")
    router = Router(cfg)
    active = 0
    if cfg["train"]["stage"] == "opd":
        active = sum(router.resolve(r)[1] > 0 for r in rows)
    adapters = {t["id"]: adapter_path(t["adapter"]) for t in cfg.get("teachers", []) if "adapter" in t}
    return {"rows": len(rows), "tasks": sorted({r["task_id"] for r in rows}), "active_rows": active,
            "adapters": adapters, "output": cfg["output_dir"], "student_mode": cfg["model"]["student_mode"]}
