"""Strict configuration loading; all paths are relative to the YAML file."""

import copy
import math
import os
from pathlib import Path

import yaml

DEFAULTS = {
    "seed": 42,
    "output_dir": "../outputs/run",
    "model": {"student_mode": "full", "dtype": "bfloat16", "trust_remote_code": False,
              "gradient_checkpointing": True, "lora_rank": 64, "lora_alpha": 128,
              "lora_targets": "all-linear", "lora_dropout": 0.0,
              "chat_template_kwargs": {"enable_thinking": False}},
    "teacher": {"device": "auto", "dtype": "bfloat16"},
    "data": {"dimension": None, "task_weights": {}, "demo_file": None},
    "routing": {"tasks": {}},
    "rollout": {"max_prompt_tokens": 2048, "max_new_tokens": 256, "context_length": None, "temperature": 1.0,
                "max_turns": 1, "environment_factory": None, "generation_batch_size": 1},
    "inference": {"batch_size": 1},
    "train": {"stage": "opd", "max_steps": 1000, "batch_size": 1,
              "gradient_accumulation_steps": 8, "learning_rate": 2e-6, "weight_decay": 0.0,
              "max_grad_norm": 1.0, "warmup_steps": 20, "save_every": 100,
              "global_prompt_batch": None, "eval_every": 0, "eval_limit_per_task": 0,
              "objective": "sampled_reverse_kl", "teacher_top_k": 64,
              "advantage_clip": 5.0, "anchor_coef": 0.0, "resume_from": None,
              "trajectory_source": "student", "teacher_fraction": 0.5, "sft_coef": 0.0},
    "quality": {"evaluator": None, "min_score": 0.0, "min_gain": 0.0,
                "filter_demos": True, "keep_per_prompt": 1, "task_min_scores": {}},
    "qgpi": {"enabled": False, "evaluator": "opd.quality_basis:hybrid_candidates",
             "profiles": {}, "shortlist": 2, "candidates_per_teacher": 1,
             "judge_repeats": 2, "min_confidence": 0.7, "max_disagreement": 0.15,
             "min_gain": 0.05, "uncertainty_coef": 0.5, "gain_scale": 0.2,
             "imitation_coef": 1.0, "kl_coef": 1.0, "rl_coef": 0.0,
             "reward_baseline": 0.0, "discount": 1.0, "branch_horizon": 1,
             "registry_file": None, "audit_candidates": True},
}


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _path(value, root, local=False):
    if value is None:
        return None
    value = os.path.expanduser(os.path.expandvars(str(value)))
    if "$" in value:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    if local or value.startswith((".", "/")) or (root / value).exists():
        return str((root / value).resolve())
    return value  # Hugging Face model id, e.g. Qwen/Qwen3-8B


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = set(raw) - set(DEFAULTS) - {"teachers"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    optional_keys = {"model": {"base_model", "student_init", "tokenizer"},
                     "data": {"train_file", "calibration_file", "eval_file"},
                     "routing": {"calibration_file"}, "quality": {"gain_for_full_strength"}}
    for section, value in raw.items():
        if section in DEFAULTS and isinstance(DEFAULTS[section], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{section} must be a mapping")
            unexpected = set(value) - set(DEFAULTS[section]) - optional_keys.get(section, set())
            if unexpected:
                raise ValueError(f"Unknown {section} options: {sorted(unexpected)}")
    cfg = merge(DEFAULTS, raw)
    root = path.parent
    cfg["output_dir"] = _path(cfg["output_dir"], root, True)
    for key in ("base_model", "student_init", "tokenizer"):
        if key in cfg["model"]:
            cfg["model"][key] = _path(cfg["model"][key], root)
    for key in ("train_file", "calibration_file", "eval_file", "demo_file"):
        if key in cfg["data"]:
            cfg["data"][key] = _path(cfg["data"][key], root, True)
    cfg["train"]["resume_from"] = _path(cfg["train"]["resume_from"], root, True)
    cfg["qgpi"]["registry_file"] = _path(cfg["qgpi"]["registry_file"], root, True)
    if cfg["routing"].get("calibration_file"):
        cfg["routing"]["calibration_file"] = _path(cfg["routing"]["calibration_file"], root, True)
    for teacher in cfg.get("teachers", []):
        for key in ("adapter", "model", "base_model"):
            if key in teacher:
                teacher[key] = _path(teacher[key], root, key == "adapter")
    validate(cfg)
    return cfg


def validate(cfg):
    from .quality_basis import validate_quality_config
    validate_quality_config(cfg)
    if cfg["quality"]["evaluator"] == "opd.objective:objective_response":
        from .objective import validate_objective_config
        validate_objective_config(cfg)
    if not cfg["model"].get("base_model"):
        raise ValueError("model.base_model must identify the COMMON Mixed-SFT checkpoint")
    if cfg["model"]["student_mode"] not in ("full", "lora"):
        raise ValueError("student_mode must be full or lora")
    for name in ("lora_rank", "lora_alpha"):
        if type(cfg["model"][name]) is not int or cfg["model"][name] < 1:
            raise ValueError(f"model.{name} must be a positive integer")
    if cfg["model"]["lora_dropout"] != 0.0:
        raise ValueError("model.lora_dropout must be 0: on-policy generation/scoring requires a dropout-free policy")
    if cfg["train"]["stage"] not in ("opd", "warmup"):
        raise ValueError("stage must be opd or warmup")
    if cfg["train"]["objective"] not in ("sampled_reverse_kl", "forward_kl"):
        raise ValueError("objective must be sampled_reverse_kl or forward_kl")
    if cfg["data"]["dimension"] not in (None, "F", "S", "U", "T", "N", "C"):
        raise ValueError("data.dimension must be null or F/S/U/T/N/C")
    for section in ("model", "teacher"):
        if cfg[section]["dtype"] not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Unsupported {section}.dtype")
    source = cfg["train"]["trajectory_source"]
    if source not in ("student", "teacher", "mixed"):
        raise ValueError("trajectory_source must be student, teacher, or mixed")
    if cfg["train"]["stage"] != "warmup" and source != "student":
        if cfg["train"]["objective"] != "forward_kl":
            raise ValueError("Teacher/off-policy trajectories require forward_kl; sampled reverse surrogate is on-policy only")
        if not cfg["data"]["demo_file"]:
            raise ValueError("teacher/mixed trajectories require data.demo_file")
    fraction = cfg["train"]["teacher_fraction"]
    if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("teacher_fraction must be strictly between zero and one")
    if not isinstance(cfg["quality"]["filter_demos"], bool):
        raise ValueError("quality.filter_demos must be boolean")
    thresholds = cfg["quality"]["task_min_scores"]
    if not isinstance(thresholds, dict) or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in [cfg["quality"]["min_score"], *thresholds.values()]):
        raise ValueError("Demo quality thresholds must be finite values in [0,1]")
    if type(cfg["quality"]["keep_per_prompt"]) is not int or cfg["quality"]["keep_per_prompt"] < 1:
        raise ValueError("quality.keep_per_prompt must be positive")
    for name in ("anchor_coef", "weight_decay", "advantage_clip", "max_grad_norm", "sft_coef"):
        if not math.isfinite(cfg["train"][name]) or cfg["train"][name] < 0:
            raise ValueError(f"train.{name} must be finite and nonnegative")
    if cfg["train"]["sft_coef"] and (source == "student" or cfg["train"]["stage"] == "warmup"):
        raise ValueError("sft_coef is an auxiliary loss on teacher trajectories, not student-generated labels")
    if cfg["train"]["anchor_coef"] and source != "student":
        raise ValueError("The sampled anchor requires on-policy student trajectories")
    if not isinstance(cfg["train"]["warmup_steps"], int) or cfg["train"]["warmup_steps"] < 0:
        raise ValueError("train.warmup_steps must be a nonnegative integer")
    for key in ("eval_every", "eval_limit_per_task"):
        if type(cfg["train"][key]) is not int or cfg["train"][key] < 0:
            raise ValueError(f"train.{key} must be a nonnegative integer")
    batch = cfg["train"]["global_prompt_batch"]
    if batch is not None and (type(batch) is not int or batch < 1):
        raise ValueError("train.global_prompt_batch must be null or a positive integer")
    if cfg["train"]["eval_every"] and not cfg["data"].get("eval_file"):
        raise ValueError("Periodic validation requires data.eval_file")
    context = cfg["rollout"]["context_length"]
    if context is not None:
        if type(context) is not int or context < 1:
            raise ValueError("rollout.context_length must be null or a positive integer")
        if cfg["rollout"]["max_prompt_tokens"] + cfg["rollout"]["max_new_tokens"] > context:
            raise ValueError("Prompt + response exceeds CONTEXT_LEN / rollout.context_length")
    for section, names in {"train": ["max_steps", "batch_size", "gradient_accumulation_steps",
                                     "teacher_top_k", "save_every"],
                           "rollout": ["max_prompt_tokens", "max_new_tokens", "max_turns", "generation_batch_size"],
                           "inference": ["batch_size"]}.items():
        for name in names:
            if not isinstance(cfg[section][name], int) or cfg[section][name] < 1:
                raise ValueError(f"{section}.{name} must be a positive integer")
    for section, name in [("rollout", "temperature"), ("train", "learning_rate")]:
        if not math.isfinite(cfg[section][name]) or cfg[section][name] <= 0:
            raise ValueError(f"{section}.{name} must be positive and finite")
    if cfg["rollout"]["max_turns"] > 1 and not cfg["rollout"]["environment_factory"]:
        raise ValueError("Multi-turn rollout requires an environment_factory; history alone is not an environment")
    ids = [t.get("id") for t in cfg.get("teachers", [])]
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("Each teacher needs a unique id")
    for t in cfg.get("teachers", []):
        if set(t) - {"id", "adapter", "model", "base_model"}:
            raise ValueError(f"Unknown teacher options for {t['id']}")
        if bool(t.get("adapter")) == bool(t.get("model")):
            raise ValueError(f"Teacher {t['id']} must specify exactly one of adapter or full model")
        if t.get("base_model", cfg["model"]["base_model"]) != cfg["model"]["base_model"]:
            raise ValueError("Adapter teachers must share the configured base_model")
    for task, route in cfg["routing"]["tasks"].items():
        if set(route) - {"teachers", "strength"}:
            raise ValueError(f"Unknown routing options for {task}")
        if not route.get("teachers") or set(route["teachers"]) - set(ids):
            raise ValueError(f"Route {task} contains missing or unknown teachers")
        weights = list(route["teachers"].values())
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(f"Route {task} needs nonnegative finite weights with a positive sum")
        strength = route.get("strength", 1.0)
        if not math.isfinite(strength) or strength < 0:
            raise ValueError(f"Invalid strength for {task}")
    for weight in cfg["data"]["task_weights"].values():
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Task sampling weights must be positive and finite")
