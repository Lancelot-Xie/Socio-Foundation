"""Driver-side health checks for rewards that depend on external judges."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


RELIABILITY_SUFFIXES = (
    "/judge_parse_success",
    "/parse_success",
    "/judge_valid_fraction",
    "/judge_confidence",
)
FAILURE_SUFFIXES = ("/judge_parse_fail",)


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _numeric_mean(values: Any) -> float | None:
    array = np.asarray(values, dtype=object).reshape(-1)
    numeric = []
    for value in array:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numeric.append(max(0.0, min(1.0, number)))
    return float(np.mean(numeric)) if numeric else None


class JudgeHealthMonitor:
    """Stop training when an external judge stays unhealthy across batches."""

    def __init__(self, config: Any):
        self.enabled = bool(_cfg_get(config, "enabled", True))
        self.minimum = float(_cfg_get(config, "min_valid_fraction", 0.5))
        self.patience = int(_cfg_get(config, "failure_patience", 2))
        if not 0.0 <= self.minimum <= 1.0:
            raise ValueError("judge_health.min_valid_fraction must be in [0, 1]")
        if self.patience < 1:
            raise ValueError("judge_health.failure_patience must be at least 1")
        self.consecutive_failures = 0

    @classmethod
    def from_algorithm_config(cls, algorithm_config: Any):
        return cls(_cfg_get(algorithm_config, "judge_health", {}))

    def check(self, reward_extra_info: dict[str, Any]) -> dict[str, float]:
        if not self.enabled:
            return {}
        reliability: dict[str, float] = {}
        for key, values in reward_extra_info.items():
            mean = _numeric_mean(values)
            if mean is None:
                continue
            if key.endswith(RELIABILITY_SUFFIXES):
                reliability[key] = mean
            elif key.endswith(FAILURE_SUFFIXES):
                reliability[key] = 1.0 - mean
        if not reliability:
            return {}

        # A task can expose both parse success and a softer confidence score.
        # The minimum is intentional: every required judge channel must remain
        # healthy for the resulting reward to be trustworthy.
        valid_fraction = min(reliability.values())
        if valid_fraction < self.minimum:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

        metrics = {
            "judge_health/valid_fraction": valid_fraction,
            "judge_health/consecutive_failures": float(self.consecutive_failures),
        }
        if self.consecutive_failures >= self.patience:
            details = ", ".join(f"{key}={value:.3f}" for key, value in sorted(reliability.items()))
            raise RuntimeError(
                "External judge health stayed below threshold: "
                f"minimum={self.minimum:.3f}, patience={self.patience}, {details}"
            )
        return metrics
