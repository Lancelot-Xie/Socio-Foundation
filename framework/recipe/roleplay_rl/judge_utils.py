"""Utilities for lower-variance, failure-aware LLM judge rewards."""

from __future__ import annotations

import math
from statistics import mean, median, pstdev
from typing import Iterable


def robust_judge_aggregate(values: Iterable[float | None], expected_samples: int | None = None) -> dict[str, float]:
    """Aggregate repeated judge scores and expose a reliability estimate.

    Three or more samples use a median/trimmed-mean blend.  The confidence is
    reduced both by missing parses and by judge disagreement.
    """

    raw = list(values)
    valid = [max(0.0, min(1.0, float(v))) for v in raw if v is not None]
    expected = max(int(expected_samples or len(raw) or 1), 1)
    valid_fraction = min(1.0, len(valid) / expected)
    if not valid:
        return {"score": 0.5, "std": 0.0, "valid_fraction": 0.0, "confidence": 0.0}

    ordered = sorted(valid)
    if len(ordered) >= 5:
        trim = max(1, int(len(ordered) * 0.2))
        trimmed = ordered[trim:-trim] or ordered
        score = 0.5 * mean(trimmed) + 0.5 * median(ordered)
    elif len(ordered) >= 3:
        score = 0.5 * mean(ordered) + 0.5 * median(ordered)
    else:
        score = mean(ordered)

    std = pstdev(ordered) if len(ordered) > 1 else 0.0
    confidence = valid_fraction * math.exp(-4.0 * std)
    return {
        "score": max(0.0, min(1.0, score)),
        "std": std,
        "valid_fraction": valid_fraction,
        "confidence": max(0.0, min(1.0, confidence)),
    }


def calibrated_score(score: float, confidence: float, neutral: float = 0.5) -> float:
    """Shrink an unreliable judge score toward a neutral reward."""

    confidence = max(0.0, min(1.0, float(confidence)))
    return max(0.0, min(1.0, neutral + confidence * (float(score) - neutral)))
