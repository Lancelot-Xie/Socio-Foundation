"""Load task-specific role-playing reward profiles."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROFILE_PATH = Path(__file__).with_name("config") / "task_profiles.yaml"
VALID_STRATEGIES = {"crpo", "dapo", "grpo"}
VALID_CHANNELS = {"task", "style"}


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _validate_profile(name: str, profile: dict[str, Any]) -> None:
    strategy = str(profile.get("strategy", "dapo")).lower()
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Task profile {name!r} has unsupported strategy {strategy!r}; "
            f"expected one of {sorted(VALID_STRATEGIES)}"
        )

    mix = _finite_float(profile.get("task_reward_weight", 1.0), f"{name}.task_reward_weight")
    if not 0.0 <= mix <= 1.0:
        raise ValueError(f"{name}.task_reward_weight must be in [0, 1], got {mix}")

    dimensions = profile.get("dimensions", []) or []
    if not isinstance(dimensions, list):
        raise ValueError(f"{name}.dimensions must be a list")
    seen: set[tuple[str, str]] = set()
    for position, spec in enumerate(dimensions):
        if not isinstance(spec, dict):
            raise ValueError(f"{name}.dimensions[{position}] must be a mapping")
        key = str(spec.get("key", "")).strip()
        channel = str(spec.get("channel", "task")).lower()
        if not key:
            raise ValueError(f"{name}.dimensions[{position}] is missing a non-empty key")
        if channel not in VALID_CHANNELS:
            raise ValueError(
                f"{name}.{key} has unsupported channel {channel!r}; "
                f"expected one of {sorted(VALID_CHANNELS)}"
            )
        identity = (key, channel)
        if identity in seen:
            raise ValueError(f"{name} repeats dimension {key!r} in channel {channel!r}")
        seen.add(identity)

        lo = _finite_float(spec.get("min", 0.0), f"{name}.{key}.min")
        hi = _finite_float(spec.get("max", 1.0), f"{name}.{key}.max")
        weight = _finite_float(spec.get("weight", 1.0), f"{name}.{key}.weight")
        if hi <= lo:
            raise ValueError(f"{name}.{key} has invalid range [{lo}, {hi}]")
        if weight <= 0.0:
            raise ValueError(f"{name}.{key}.weight must be positive, got {weight}")
        if "mask_key" in spec and not str(spec["mask_key"]).strip():
            raise ValueError(f"{name}.{key}.mask_key must be a non-empty string")

    judge = profile.get("judge") or {}
    if not isinstance(judge, dict):
        raise ValueError(f"{name}.judge must be a mapping")

    for position, penalty in enumerate(profile.get("penalties", []) or []):
        if not isinstance(penalty, dict) or not str(penalty.get("key", "")).strip():
            raise ValueError(f"{name}.penalties[{position}] must define a non-empty key")
        factor = _finite_float(penalty.get("factor", 1.0), f"{name}.penalties[{position}].factor")
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"{name}.penalties[{position}].factor must be in [0, 1], got {factor}")


def load_profiles(path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(defaults, profiles_by_data_source)``.

    Aliases are expanded at load time so the hot reward path only performs one
    dictionary lookup per sequence.
    """

    profile_path = Path(path).expanduser() if path else DEFAULT_PROFILE_PATH
    if not profile_path.is_absolute() and not profile_path.exists():
        profile_path = Path(__file__).resolve().parents[2] / profile_path
    with profile_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid roleplay profile file: {profile_path}")

    defaults = dict(raw.get("defaults") or {})
    _validate_profile("defaults", defaults)
    profiles: dict[str, Any] = {}
    for name, task_cfg in (raw.get("tasks") or {}).items():
        merged = deepcopy(defaults)
        merged.update(deepcopy(task_cfg or {}))
        merged["name"] = name
        _validate_profile(str(name), merged)
        aliases = [name, *(merged.pop("aliases", []) or [])]
        for alias in aliases:
            if str(alias) in profiles:
                raise ValueError(f"Duplicate task profile alias: {alias!r}")
            profiles[str(alias)] = deepcopy(merged)
    return defaults, profiles


def normalize_dimension(value: Any, spec: dict[str, Any]) -> float | None:
    """Normalize one configured dimension to ``[0, 1]``."""

    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None

    lo = float(spec.get("min", 0.0))
    hi = float(spec.get("max", 1.0))
    if hi <= lo:
        raise ValueError(f"Invalid dimension range for {spec.get('key')}: [{lo}, {hi}]")
    normalized = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    return 1.0 - normalized if bool(spec.get("invert", False)) else normalized
