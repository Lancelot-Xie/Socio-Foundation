"""Five independent quality axes, evidence-aware applicability and conservative gates.

C is intentionally absent. Judge reliability here is a configurable heuristic,
not a statistically calibrated probability of a teacher being superior.
"""

import copy
import math
import random
import re
import statistics

from .rollout import load_callable

DIMENSIONS = ("F", "S", "U", "T", "N")
RUBRICS = {
    "F": "Individual fidelity: match this individual's stated identity, preferences, values and persona; not generic helpfulness.",
    "S": "Social-state grounding: respect the actor's visible knowledge, others' beliefs, relationships and social context; avoid omniscient leakage.",
    "U": "Intent and outcome realization: advance the explicitly evidenced actor goal or intent, using actual branch outcomes when available; do not invent goals or confuse compliance with success.",
    "T": "Sequential and narrative coherence: preserve causal and storyline continuity, commitments and facts across turns, and terminate at the correct boundary; distinguish narrow termination metrics from full trajectory coherence.",
    "N": "Human-likeness and behavioral realism: match human expression, actions, information disclosure, response patterns and behavioral distributions; natural wording alone is narrow evidence, not complete realism.",
}
PROFILES = {
    "lifechoices": {"dimensions": ["F"], "core": {"F": .6}},
    "alignx": {"dimensions": ["F"], "core": {"F": .6}},
    "fantom": {"dimensions": ["S"], "core": {"S": .6}},
    "userllm": {"dimensions": ["F", "U", "T", "N"], "core": {"F": .6, "U": .5, "T": .6}},
    "humanllm": {"dimensions": ["F"], "core": {"F": .6}},
    "behavior_chain": {"dimensions": ["F"], "core": {"F": .6}},
    "social_r1": {"dimensions": ["S"], "core": {"S": .6}},
    "mirrorbench": {"dimensions": ["N"], "core": {"N": .5}},
    "coser": {"dimensions": ["F", "S", "U", "T", "N"], "core": {"F": .6, "S": .6, "T": .6},
              "require_evidence": ["U"]},
    "sotopia": {"dimensions": ["F", "S", "U", "T", "N"], "core": {"S": .6, "U": .5, "T": .6},
                "require_evidence": ["F", "U"]},
}
# These are explicit task-metric proxies, never replicated across all axes.
VERIFIABLE_AXIS = {"lifechoices": "F", "fantom": "S", "alignx": "F", "humanllm": "F",
                   "behavior_chain": "F", "social_r1": "S"}


def validate_quality_config(cfg):
    q = cfg.get("qgpi", {})
    if type(q.get("enabled", False)) is not bool:
        raise ValueError("qgpi.enabled must be boolean")
    if not q.get("enabled"):
        return
    if type(q["audit_candidates"]) is not bool or not isinstance(q["profiles"], dict):
        raise ValueError("qgpi.audit_candidates must be boolean and profiles must be a mapping")
    if cfg["train"]["stage"] != "opd" or cfg["train"]["objective"] != "forward_kl":
        raise ValueError("QGPI requires stage=opd and objective=forward_kl")
    if cfg["data"]["dimension"] is not None or cfg["train"]["trajectory_source"] != "student":
        raise ValueError("QGPI trains a unified student on student-visited states, not dimension/offline experts")
    if cfg["train"]["sft_coef"]:
        raise ValueError("Use qgpi.imitation_coef for gated teacher imitation")
    if not q.get("evaluator"):
        raise ValueError("QGPI needs a candidate evaluator")
    if q["evaluator"] == "opd.objective:objective_candidates":
        from .objective import validate_objective_config
        validate_objective_config(cfg)
    for key in ("shortlist", "candidates_per_teacher", "judge_repeats", "branch_horizon"):
        if type(q[key]) is not int or q[key] < 1:
            raise ValueError(f"qgpi.{key} must be a positive integer")
    for key in ("min_confidence", "max_disagreement", "min_gain", "discount"):
        if not finite(q[key]) or not 0 <= q[key] <= 1:
            raise ValueError(f"qgpi.{key} must be in [0,1]")
    for key in ("uncertainty_coef", "imitation_coef", "kl_coef", "rl_coef", "gain_scale"):
        if not finite(q[key]) or q[key] < 0 or (key == "gain_scale" and q[key] == 0):
            raise ValueError(f"qgpi.{key} must be finite and nonnegative (gain_scale positive)")
    if not finite(q["reward_baseline"]):
        raise ValueError("reward_baseline must be finite")
    if not (q["imitation_coef"] or q["kl_coef"] or q["rl_coef"] or cfg["train"]["anchor_coef"]):
        raise ValueError("QGPI has no enabled loss")
    if (q["branch_horizon"] > 1 or q["rl_coef"]) and not cfg["rollout"]["environment_factory"]:
        raise ValueError("Branch continuation / environment RL requires an actual environment_factory")
    for task, profile in q["profiles"].items():
        if not isinstance(profile, dict):
            raise ValueError(f"QGPI profile {task} must be a mapping")
        if set(profile) - {"dimensions", "weights", "core", "require_evidence", "regression_tolerance", "rubrics"}:
            raise ValueError(f"Unknown QGPI profile option for {task}")
        dims = profile.get("dimensions", PROFILES.get(task, {}).get("dimensions", []))
        if not dims or len(set(dims)) != len(dims) or set(dims) - set(DIMENSIONS):
            raise ValueError(f"QGPI profile {task} must select F/S/U/T/N only; C is disabled")
        for key in ("core", "weights", "rubrics"):
            if set(profile.get(key, {})) - set(dims):
                raise ValueError(f"Profile {task}.{key} contains an inactive dimension")
        for key in ("core", "weights"):
            if any(not finite(v) or v < 0 or (key == "core" and v > 1) or
                   (key == "weights" and v == 0) for v in profile.get(key, {}).values()):
                raise ValueError(f"Invalid {task}.{key}")
        tolerance = profile.get("regression_tolerance", .05)
        if not finite(tolerance) or not 0 <= tolerance <= 1:
            raise ValueError("regression_tolerance must be in [0,1]")
        if set(profile.get("require_evidence", [])) - set(dims):
            raise ValueError("require_evidence must select active dimensions")


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def visible_quality_evidence(task, messages, declared=None):
    evidence = copy.deepcopy(declared or {})
    if task == "coser" and "U" not in evidence:
        # Exact audited upstream get_character_prompt section: this is already
        # visible to this actor. Never take another character's private thought.
        for message in messages:
            if message["role"] != "system":
                continue
            match = re.search(r"===Your Inner Thoughts===\s*\n(.*?)(?=\n\s*===|$)", message["content"], re.S)
            if match and match.group(1).strip():
                evidence["U"] = {"source": "actor_visible_motivation", "text": match.group(1).strip(),
                                 "scope": "intent_progress_proxy_not_observed_outcome"}
                break
    return evidence


def quality_spec(cfg, row, messages, candidates):
    task = row["task_id"]
    profile = copy.deepcopy(PROFILES.get(task, {"dimensions": [d for d in row["dimensions"] if d in DIMENSIONS]}))
    profile.update(copy.deepcopy(cfg["qgpi"]["profiles"].get(task, {})))
    dims = list(profile["dimensions"])
    evidence = visible_quality_evidence(task, messages, row.get("evaluator_context", {}).get("quality_evidence"))
    omitted = {}
    objective = cfg["qgpi"]["evaluator"] == "opd.objective:objective_candidates"
    if objective:
        from .objective_metrics import eligible
        usable, reason = eligible(task, row.get("evaluator_context", {}).get("original_row", {}))
        if not usable:
            omitted.update(dict.fromkeys(dims, reason))
    for dim in profile.get("require_evidence", []):
        if not evidence.get(dim):
            omitted[dim] = "No explicit task/row evidence"
    history = any(m["role"] == "assistant" for m in messages)
    future = bool(candidates) and all(len(c.get("trajectory", [])) > 1 for c in candidates)
    boundary = objective and task == "userllm" and "T" not in omitted
    if "T" in dims and not (history or future or boundary):
        omitted["T"] = "No prior actor turn or observed multi-turn branch"
    dims = [d for d in dims if d not in omitted]
    weights = {d: profile.get("weights", {}).get(d, 1.0) for d in dims}
    total = sum(weights.values())
    rubrics = {d: row.get("evaluator_context", {}).get("rubrics", {}).get(d, RUBRICS[d]) for d in dims}
    rubrics.update({d: r for d, r in profile.get("rubrics", {}).items() if d in dims})
    return {"dimensions": dims, "weights": {d: w/total for d, w in weights.items()},
            "core": {d: v for d, v in profile.get("core", {}).items() if d in dims},
            "regression_tolerance": profile.get("regression_tolerance", .05), "rubrics": rubrics,
            "quality_evidence": evidence, "outcome_scope": "observed_branch" if future else "intent_progress_only",
            "omitted": omitted, "temporal_scope": "gold_termination_boundary" if boundary else
            "future_branch" if future else "history" if history else "none"}


def hybrid_candidates(row, candidates, spec):
    """Plugin contract: (private row, anonymous candidates, spec) -> ordered results.

    Use original verifiable reward on its one justified axis, a blinded joint
    rubric judge for all remaining applicable axes. The judge sees no model IDs.
    """
    from .judges import quality_candidates_judge
    from .upstream import task_or_rubric
    local_axis = VERIFIABLE_AXIS.get(row["task_id"])
    local_axis = local_axis if local_axis in spec["dimensions"] else None
    judged = [d for d in spec["dimensions"] if d != local_axis]
    results = quality_candidates_judge(row, candidates, {**spec, "dimensions": judged}) if judged else [
        {"valid": True, "scores": {}, "confidence": 1.0} for _ in candidates]
    for candidate, result in zip(candidates, results, strict=True):
        if local_axis and result.get("valid"):
            measured = task_or_rubric({**row, "dimensions": [local_axis]}, candidate["response"])
            if not measured.get("valid"):
                result.update(valid=False, reason="Verifiable task metric unavailable")
            else:
                result["scores"][local_axis] = measured["scores"][local_axis]
                result["task_metrics"] = measured.get("task_metrics", {})
    return results


def evaluate_candidates(cfg, row, messages, candidates):
    spec = quality_spec(cfg, row, messages, candidates)
    if not spec["dimensions"]:
        return spec, [{"valid": False, "reason": "No applicable quality evidence"} for _ in candidates]
    evaluator = load_callable(cfg["qgpi"]["evaluator"])
    row = {**row, "evaluator_context": {**row.get("evaluator_context", {}), "quality_evidence": spec["quality_evidence"]}}
    samples = [[] for _ in candidates]
    # Omit candidate origins and hidden environment snapshots from evaluator input.
    anonymous = [{"response": c["response"], "trajectory": c.get("trajectory", [])} for c in candidates]
    for _ in range(cfg["qgpi"]["judge_repeats"]):
        order = list(range(len(candidates)))
        random.shuffle(order)
        results = evaluator({**row, "messages": copy.deepcopy(messages)},
                            [copy.deepcopy(anonymous[i]) for i in order], copy.deepcopy(spec))
        if not isinstance(results, list) or len(results) != len(order):
            raise ValueError("Candidate evaluator must return one ordered result per candidate")
        for i, result in zip(order, results, strict=True):
            if not isinstance(result, dict) or type(result.get("valid")) is not bool:
                raise ValueError("Candidate evaluation requires boolean valid")
            if result["valid"]:
                scores = result.get("scores", {})
                if any(d not in scores or not finite(scores[d]) or not 0 <= scores[d] <= 1 for d in spec["dimensions"]):
                    raise ValueError("Candidate evaluator must measure every applicable axis in [0,1]")
                if not finite(result.get("confidence")) or not 0 <= result["confidence"] <= 1:
                    raise ValueError("Candidate evaluator requires confidence in [0,1]")
            samples[i].append(result)
    combined = []
    for trials in samples:
        if not all(r["valid"] for r in trials):
            combined.append({"valid": False, "reason": "Missing evidence or failed judge trial"})
            continue
        scores = {d: statistics.mean(r["scores"][d] for r in trials) for d in spec["dimensions"]}
        spread = max(max(r["scores"][d] for r in trials)-min(r["scores"][d] for r in trials)
                     for d in spec["dimensions"])
        combined.append({"valid": True, "constraint_pass": True, "scores": scores,
                         "confidence": min(r["confidence"] for r in trials), "disagreement": spread,
                         "utility": sum(spec["weights"][d]*scores[d] for d in scores)})
        # Preserve original task diagnostics rather than reducing MRR/hit@K to
        # an undocumented generic quality number. Trials may differ for plugins.
        diagnostics = [r["task_metrics"] for r in trials if "task_metrics" in r]
        if diagnostics:
            combined[-1]["task_metrics"] = {k: statistics.mean(r[k] for r in diagnostics)
                for k in set.intersection(*(set(r) for r in diagnostics))
                if all(finite(r[k]) for r in diagnostics)}
        for key in ("metric", "score_kind", "requires_llm_judge"):
            if all(key in r and r[key] == trials[0].get(key) for r in trials):
                combined[-1][key] = trials[0][key]
    return spec, combined


def select_winner(cfg, spec, evaluations):
    """Index 0 is the student's baseline, included even when it fails core floors."""
    q = cfg["qgpi"]
    def reliable(r):
        return (r.get("valid", False) and r["confidence"] >= q["min_confidence"] and
                r["disagreement"] <= q["max_disagreement"])
    student = evaluations[0]
    decision = {"winner": None, "advantage": 0.0, "weight": 0.0, "reason": "student_unreliable"}
    if not reliable(student):
        return decision
    decision["reason"] = "no_reliable_positive_advantage"
    for index, candidate in enumerate(evaluations[1:], 1):
        if not reliable(candidate):
            continue
        if any(candidate["scores"][d] < floor or
               candidate["scores"][d] < student["scores"][d] - spec["regression_tolerance"]
               for d, floor in spec["core"].items()):
            continue
        advantage = candidate["utility"] - student["utility"] - q["uncertainty_coef"] * (
            candidate["disagreement"] + student["disagreement"])
        if advantage <= q["min_gain"] + 1e-12 or advantage <= decision["advantage"]:
            continue
        decision.update(winner=index, advantage=advantage, weight=min(1.0, advantage/q["gain_scale"]), reason="accepted")
    return decision
