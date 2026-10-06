"""Explicit, opt-in mapping of the supplied expert recipe to unified QGPI.

Only load_plan applies environment overrides; resolved trainer YAMLs are immutable.
PPO and vLLM knobs are recorded as inapplicable, never silently wired into OPD KL.
"""

import math
import os
from pathlib import Path

import yaml


ENV_FIELDS = {
    "CONTEXT_LEN": ("rollout", "context_length", int),
    "MAX_PROMPT_LEN": ("rollout", "max_prompt_tokens", int),
    "MAX_RESP_LEN": ("rollout", "max_new_tokens", int),
    "ACTOR_LR": ("train", "learning_rate", float),
    "ACTOR_LR_WARMUP_STEPS": ("train", "warmup_steps", int),
    "ACTOR_WEIGHT_DECAY": ("train", "weight_decay", float),
    "SAVE_FREQ": ("train", "save_every", int),
    "TEST_FREQ": ("train", "eval_every", int),
    "OPD_EVAL_LIMIT_PER_TASK": ("train", "eval_limit_per_task", int),
    "OPD_MICRO_BATCH": ("train", "batch_size", int),
    "OPD_ROLLOUT_BATCH_SIZE": ("rollout", "generation_batch_size", int),
    "OPD_INFERENCE_BATCH_SIZE": ("inference", "batch_size", int),
    "OPD_ANCHOR_COEF": ("train", "anchor_coef", float),
    "N_RESP": ("qgpi", "candidates_per_teacher", int),
    "OPD_CANDIDATES_PER_TEACHER": ("qgpi", "candidates_per_teacher", int),
    "LORA_RANK": ("model", "lora_rank", int),
    "LORA_ALPHA": ("model", "lora_alpha", int),
    "LORA_DROPOUT": ("model", "lora_dropout", float),
    "OPD_STUDENT_MODE": ("model", "student_mode", str),
    "TOTAL_STEPS": ("steps", "final", int),
}
INAPPLICABLE = {
    "DAPO_ENABLED": "QGPI imitation/teacher forward-KL, not DAPO",
    "PPO_MINI": "No PPO epochs or old-policy minibatch reuse",
    "USE_KL_IN_REWARD": "No RL reward-KL; OPD_ANCHOR_COEF controls the optional base anchor",
    "USE_KL_LOSS": "No PPO reference-KL; teacher distillation KL remains enabled",
    "KL_COEF": "Not the teacher distillation KL coefficient",
    "ACTOR_KL_COEF": "Not the teacher distillation KL coefficient",
    "ENTROPY_FROM_LOGITS_WITH_CHUNKING": "No VERL entropy metrics in this trainer",
    "ROLLOUT_GPU_MEMORY_UTILIZATION": "Transformers generation, not a vLLM service",
    "ROLLOUT_MAX_NUM_SEQS": "Use OPD_ROLLOUT_BATCH_SIZE / OPD_INFERENCE_BATCH_SIZE; not a vLLM scheduler",
    "ROLLOUT_MAX_MODEL_LEN": "Use CONTEXT_LEN/MAX_PROMPT_LEN/MAX_RESP_LEN here",
    "AGENT_NUM_WORKERS": "No remote agent-loop worker pool",
    "CONTINUE_ON_ERROR": "One unified workflow; failure always stops for safe resume",
    "TASKS": "Task scope comes from the checkpoint manifest and YAML, not the old RL loop",
    "TRAIN_FILES": "Data exported from OPD_DATA_ROOT with objective eligibility checks",
    "VAL_FILES": "Validation split comes from the OPD data preparation plan",
    "USE_LORA": "Only OPD_STUDENT_MODE changes the student; saved teachers are frozen",
}


def _positive(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if str(parsed) != str(value) or parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def apply_expert_recipe(plan):
    if not plan.get("expert_recipe"):
        return
    dimensions = plan.get("workflow") == "dimension_offline"
    if plan.get("workflow") not in ("qgpi", "dimension_offline"):
        raise ValueError("expert_recipe supports qgpi or dimension_offline")
    recipe = plan["expert_recipe"]
    if not isinstance(recipe, dict) or set(recipe) != {"global_prompt_batch"}:
        raise ValueError("expert_recipe must contain global_prompt_batch only")
    overrides = {}
    for variable, (section, key, convert) in ENV_FIELDS.items():
        if variable not in os.environ:
            continue
        try:
            value = convert(os.environ[variable])
        except ValueError as error:
            raise ValueError(f"Invalid {variable}: {os.environ[variable]!r}") from error
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{variable} must be finite")
        if dimensions and variable == "TOTAL_STEPS":
            section, key = "steps", "distill"
            plan["dimension_steps"] = {}  # TOTAL_STEPS explicitly resets all dimension budgets.
        if dimensions and variable in ("N_RESP", "OPD_CANDIDATES_PER_TEACHER"):
            plan["demo_candidates"] = value
        else:
            plan.setdefault(section, {})[key] = value
        overrides[variable] = value
    if dimensions:
        for dim in plan.get("dimensions", ["F", "S", "N"]):
            variable = "OPD_" + dim + "_STEPS"
            if variable in os.environ:
                plan.setdefault("dimension_steps", {})[dim] = _positive(os.environ[variable], variable)
                overrides[variable] = plan["dimension_steps"][dim]
        for variable, field in (("OPD_DEMO_LIMIT_PER_TASK", "demo_limit_per_task"),
                                ("OPD_DEMO_CANDIDATES", "demo_candidates")):
            if variable in os.environ:
                plan[field] = _positive(os.environ[variable], variable)
                overrides[variable] = plan[field]
    if "VAL_BEFORE_TRAIN" in os.environ:
        value = os.environ["VAL_BEFORE_TRAIN"].lower()
        if value not in ("true", "false", "1", "0"):
            raise ValueError("VAL_BEFORE_TRAIN must be true/false or 1/0")
        plan["val_before_train"] = value in ("true", "1")
        overrides["VAL_BEFORE_TRAIN"] = plan["val_before_train"]
    if type(plan.get("val_before_train", True)) is not bool:
        raise ValueError("val_before_train must be boolean")
    if plan["train"].get("eval_every", 0) and plan.get("evaluation_split", "validation") != "validation":
        raise ValueError("Periodic validation must not select the held-out eval/test split")
    processes = 1
    if plan.get("accelerate_config"):
        launch = yaml.safe_load(Path(plan["accelerate_config"]).read_text())
        if int(launch.get("num_machines", 1)) != 1:
            raise ValueError("expert_recipe supports single-node training only")
        processes = _positive(os.environ.get("OPD_NUM_PROCESSES", launch.get("num_processes", 1)), "OPD_NUM_PROCESSES")
    batch = _positive(os.environ.get("TRAIN_BATCH", recipe["global_prompt_batch"]), "TRAIN_BATCH")
    micro = _positive(plan["train"]["batch_size"], "OPD_MICRO_BATCH")
    if batch % (processes * micro):
        raise ValueError(f"TRAIN_BATCH={batch} must be divisible by processes({processes}) * microbatch({micro})")
    plan["train"].update(global_prompt_batch=batch, gradient_accumulation_steps=batch // (processes * micro))
    step_key = "distill" if dimensions else "final"
    plan["train"]["max_steps"] = plan["steps"][step_key]
    inference_processes = os.environ.get("OPD_INFERENCE_PROCESSES", plan.get("inference_num_processes", 1))
    plan["inference_num_processes"] = processes if inference_processes == "auto" else _positive(
        inference_processes, "OPD_INFERENCE_PROCESSES")
    plan["baseline_scheduler"] = os.environ.get("OPD_BASELINE_SCHEDULER", plan.get("baseline_scheduler", "distributed"))
    if plan["baseline_scheduler"] not in ("distributed", "task_pool"):
        raise ValueError("OPD_BASELINE_SCHEDULER must be distributed or task_pool")
    plan["baseline_chunk_size"] = _positive(os.environ.get("OPD_BASELINE_CHUNK_SIZE", plan.get("baseline_chunk_size", 32)),
                                             "OPD_BASELINE_CHUNK_SIZE")
    plan["baseline_workers_per_gpu"] = _positive(os.environ.get("OPD_BASELINE_WORKERS_PER_GPU",
        plan.get("baseline_workers_per_gpu", 1)), "OPD_BASELINE_WORKERS_PER_GPU")
    for variable, field in (("OPD_BASELINE_SCHEDULER", "baseline_scheduler"), ("OPD_BASELINE_CHUNK_SIZE", "baseline_chunk_size"),
                            ("OPD_BASELINE_WORKERS_PER_GPU", "baseline_workers_per_gpu")):
        if variable in os.environ:
            overrides[variable] = plan[field]
    # The student preset never alters actual teacher metadata.
    for key in ("lora_rank", "lora_alpha"):
        _positive(plan["model"][key], key)
    plan["recipe_alignment"] = {
        "source": "expert LoRA architecture + recommended objective OPD trial budgets (not an exact RL recipe copy)",
        "processes": processes, "global_prompt_batch": batch, "microbatch_per_rank": micro,
        "inference_processes": plan["inference_num_processes"],
        "baseline_scheduler": plan["baseline_scheduler"], "baseline_chunk_size": plan["baseline_chunk_size"],
        "baseline_workers_per_gpu": plan["baseline_workers_per_gpu"],
        "gradient_accumulation_steps": plan["train"]["gradient_accumulation_steps"],
        "student_mode": plan["model"]["student_mode"], "environment_overrides": overrides,
        "candidate_semantics": ("Fixed teacher-only demonstrations per compatible checkpoint; no student candidate" if dimensions else
                                "1 student + candidates_per_teacher per compatible shortlisted teacher; not an RL group"),
        "step_semantics": "per dimension; dimension_steps overrides distill" if dimensions else "unified student",
        "inapplicable_rl_options": INAPPLICABLE,
        "inherited_but_not_applied": {k: os.environ[k] for k in INAPPLICABLE if k in os.environ},
        "logging": "local JSONL/log files; WANDB_MODE does not enable a W&B integration",
    }
    print(f"[recipe] prompts={batch} = {processes} ranks x {micro} microbatch x "
          f"{plan['train']['gradient_accumulation_steps']} accumulation; "
          f"student={plan['model']['student_mode']}; steps={plan['steps'][step_key]} ({'per dimension' if dimensions else 'unified'}); "
          f"save={plan['train']['save_every']}; validate={plan['train']['eval_every']}", flush=True)
    print(f"[parallel] inference={plan['inference_num_processes']} workers x "
          f"{plan.get('inference', {}).get('batch_size', 1)} batch; "
          f"rollout_batch_per_rank={plan['rollout'].get('generation_batch_size', 1)}; "
          f"baseline={plan['baseline_scheduler']} chunk={plan['baseline_chunk_size']}", flush=True)
    if plan["baseline_scheduler"] == "task_pool":
        print(f"[baseline] {plan['inference_num_processes']} GPUs x {plan['baseline_workers_per_gpu']} model processes/GPU "
              f"= {plan['inference_num_processes'] * plan['baseline_workers_per_gpu']} concurrent slots; "
              + ("also used for fixed teacher demos; training ranks unchanged" if dimensions else
                 "calibration/final inference and training process counts unchanged"), flush=True)
    for name in plan["recipe_alignment"]["inherited_but_not_applied"]:
        print(f"[recipe] {name} not applied: {INAPPLICABLE[name]}", flush=True)
