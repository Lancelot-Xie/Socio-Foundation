"""Explicit task evaluators; private rubric/reference fields never enter generation."""

import math
import re

from .rollout import load_callable


def builtin_evaluator(row, response):
    rubric = row.get("evaluation", {})
    kind = rubric.get("type")
    dimension = rubric.get("dimension")
    if kind == "exact_match":
        score = float(response.strip().casefold() == str(rubric["answer"]).strip().casefold())
    elif kind == "choice":
        match = re.search(r"<answer>\s*([A-D])\s*</answer>", response, re.I)
        prediction = match.group(1).upper() if match else response.strip().upper()
        score = float(prediction == str(rubric["answer"]).strip().upper())
    elif kind == "nonempty_smoke_only":
        score = float(bool(response.strip()))
        return {"valid": True, "constraint_pass": True,
                "scores": {d: score for d in row.get("dimensions", [dimension])}}
    else:
        return {"valid": False, "reason": "No supported task rubric; configure a custom evaluator"}
    if dimension not in ("F", "S", "U", "T", "N", "C"):
        return {"valid": False, "reason": "evaluation.dimension must be explicit"}
    return {"valid": True, "constraint_pass": True, "scores": {dimension: score}}


def evaluate_response(cfg, row, response):
    spec = cfg["quality"]["evaluator"]
    evaluator = load_callable(spec) if spec else builtin_evaluator
    result = evaluator(row, response)
    if not isinstance(result, dict) or not isinstance(result.get("valid"), bool):
        raise ValueError("Evaluator must return a dict with boolean valid")
    if not result["valid"]:
        return result
    if not isinstance(result.get("constraint_pass"), bool):
        raise ValueError("Valid evaluation must contain boolean constraint_pass")
    scores = result.get("scores", {})
    if not scores or any(k not in ("F", "S", "U", "T", "N", "C") or
                         not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                         for k, v in scores.items()):
        raise ValueError("Evaluator scores must be finite normalized values in [0,1] for named dimensions")
    required = cfg["data"]["dimension"]
    if required and required not in scores:
        return {"valid": False, "reason": f"Evaluator did not measure required dimension {required}"}
    return result


def scalar_score(cfg, result):
    if not result["valid"]:
        raise ValueError("An invalid judge result is not a zero task reward")
    if not result["constraint_pass"]:
        return 0.0
    dimension = cfg["data"]["dimension"]
    if dimension:
        return result["scores"][dimension]
    # Within a sample only. Do not pool raw task metrics across tasks.
    return sum(result["scores"].values()) / len(result["scores"])
