"""Strict local-only task metrics. No rubric, judge, API or neutral-score fallback."""

from .upstream import canonical_task, task_or_rubric

TASK_AXES = {"lifechoices": "F", "alignx": "F", "humanllm": "F", "behavior_chain": "F",
             "fantom": "S", "social_r1": "S", "userllm": "T", "mirrorbench": "N"}
METRICS = {
    "lifechoices": "choice_accuracy",
    "alignx": "seeded_preference_pair_accuracy",
    "humanllm": "reciprocal_rank_at_5",
    "behavior_chain": "node_choice_accuracy_not_CumScore",
    "fantom": "question_type_accuracy_or_set_F1_or_token_F1",
    "social_r1": "choice_accuracy",
    "userllm": "PRISM_termination_accuracy_not_full_T",
    "mirrorbench": "first_turn_lexical_similarity_not_GTEval_or_full_N",
}


def backend_hashes():
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("objective.py", "objective_metrics.py", "quality_basis.py", "upstream.py")}


def objective_response(row, response):
    task = canonical_task(row["task_id"])
    if task not in TASK_AXES:
        raise ValueError(f"{task} has no enabled audited objective metric; no judge fallback is allowed")
    if canonical_task(row.get("original_task_id", task)) != task:
        raise ValueError("Objective scoring forbids a different original_task_id")
    axis = TASK_AXES[task]
    if set(row.get("dimensions", [axis])) != {axis}:
        raise ValueError(f"{task} objectively measures only {axis}; do not copy its score into other axes")
    raw = row.get("evaluator_context", {}).get("original_row", {}).get("extra_info", {})
    if task == "fantom" and (raw.get("correct_answer") is None or not str(raw.get("correct_answer", "")).strip()):
        # Upstream uses a neutral 0.5 for an empty fact label. In this strict
        # objective phase missing labels are invalid, never positive evidence.
        return {"valid": False, "reason": "FANToM gold answer missing; neutral reward is not objective evidence"}
    if task == "lifechoices" and not str(raw.get("Multiple Choice Question", {}).get("Correct Answer", "")).strip():
        return {"valid": False, "reason": "LifeChoices gold answer missing"}
    if task == "social_r1" and not (raw.get("answer_letter") or raw.get("answer_text")):
        return {"valid": False, "reason": "Social-R1 gold answer missing"}
    if task in ("userllm", "mirrorbench"):
        from .objective_metrics import partial_response
        result = partial_response({**row, "task_id": task}, response)
    else:
        result = task_or_rubric({**row, "task_id": task, "dimensions": [axis]}, response)
    return {**result, "score_kind": "objective_task_proxy", "metric": METRICS[task], "requires_llm_judge": False}


def objective_candidates(row, candidates, spec):
    task = canonical_task(row["task_id"])
    if task not in TASK_AXES or spec["dimensions"] != [TASK_AXES[task]]:
        raise ValueError(f"No audited objective-only evaluator for {task}/{spec['dimensions']}")
    results = []
    for candidate in candidates:
        result = objective_response({**row, "dimensions": spec["dimensions"]}, candidate["response"])
        results.append({**result, "confidence": 1.0})
    return results


def validate_objective_config(cfg):
    """Enforce the contract before loading models or starting generation."""
    from .quality_basis import PROFILES
    for task in cfg["routing"]["tasks"]:
        if task not in TASK_AXES:
            raise ValueError(f"Objective-only route cannot include {task}")
        profile = cfg["qgpi"]["profiles"].get(task, PROFILES.get(task, {}))
        if profile.get("dimensions") != [TASK_AXES[task]]:
            raise ValueError(f"Objective-only {task} requires exactly [{TASK_AXES[task]}]")
    if cfg["rollout"]["max_turns"] != 1 or cfg["rollout"]["environment_factory"]:
        raise ValueError("Current objective metrics require static single-decision rollouts")
    if cfg["quality"]["evaluator"] != "opd.objective:objective_response":
        raise ValueError("Objective-only workflows require objective_response for quality filtering")
    return sorted({TASK_AXES[t] for t in cfg["routing"]["tasks"]})
