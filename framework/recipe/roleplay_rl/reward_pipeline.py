"""AnthroDial-style adaptive rewards and judge calibration."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from recipe.roleplay_rl.judge_utils import calibrated_score
from recipe.roleplay_rl.profiles import load_profiles, normalize_dimension


@dataclass
class RunningStat:
    mean: float = 0.5
    variance: float = 0.0625
    count: int = 0

    def update(self, value: float, alpha: float) -> None:
        value = float(value)
        if self.count == 0:
            self.mean = value
            self.variance = 0.0625
        else:
            delta = value - self.mean
            self.mean += alpha * delta
            self.variance = (1.0 - alpha) * (self.variance + alpha * delta * delta)
        self.count += 1


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _require_range(name: str, value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}], got {value}")
    return value


def _require_positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}, got {value}")
    return value


def _column(values: Any, length: int, name: str, default: Any) -> np.ndarray:
    if values is None:
        result = np.empty(length, dtype=object)
        result[:] = default
        return result
    result = np.asarray(values, dtype=object)
    if result.ndim == 0:
        raise ValueError(f"Batch field {name!r} must have length {length}, got a scalar")
    result = result.reshape(-1)
    if len(result) != length:
        raise ValueError(f"Batch field {name!r} has length {len(result)}; expected {length}")
    return result


class RoleplayRewardProcessor:
    """Driver-side stateful reward processor.

    The driver is the correct location for adaptive statistics: unlike agent-loop
    workers, it sees every reward batch and has a single consistent state.
    """

    def __init__(self, config: Any):
        self.config = config
        path = _cfg_get(config, "profile_path", None)
        self.defaults, self.profiles = load_profiles(path)
        self.ema_alpha = _require_range(
            "ema_alpha", float(_cfg_get(config, "ema_alpha", 0.05)), 1e-12, 1.0
        )
        self.adaptive_strength = _require_range(
            "adaptive_strength", float(_cfg_get(config, "adaptive_strength", 0.7)), 0.0, 1.0
        )
        self.minimum_weight = _require_positive(
            "minimum_dimension_weight",
            float(_cfg_get(config, "minimum_dimension_weight", 0.2)),
            allow_zero=True,
        )
        self.deficit_power = _require_positive(
            "deficit_power", float(_cfg_get(config, "deficit_power", 1.0))
        )
        self.zpd_sigma = _require_positive("zpd_sigma", float(_cfg_get(config, "zpd_sigma", 0.25)))
        self.zpd_floor = _require_range(
            "zpd_floor", float(_cfg_get(config, "zpd_floor", 0.25)), 0.0, 1.0
        )
        self.neutral = _require_range(
            "judge_neutral", float(_cfg_get(config, "judge_neutral", 0.5)), 0.0, 1.0
        )
        self.min_style_history = int(_cfg_get(config, "min_style_history", 8))
        if self.min_style_history < 1:
            raise ValueError(f"min_style_history must be at least 1, got {self.min_style_history}")
        self.preserve_penalties = bool(_cfg_get(config, "preserve_base_penalties", True))
        self.dimension_stats: dict[tuple[str, str], RunningStat] = defaultdict(RunningStat)
        self.character_style_stats: dict[str, RunningStat] = defaultdict(RunningStat)

    @classmethod
    def from_algorithm_config(cls, algorithm_config: Any):
        cfg = _cfg_get(algorithm_config, "roleplay_rl", {})
        if not bool(_cfg_get(cfg, "enabled", False)):
            return None
        return cls(cfg)

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-serializable adaptive state for exact checkpoint resume."""

        dimensions = [
            {
                "source": source,
                "key": key,
                "mean": stat.mean,
                "variance": stat.variance,
                "count": stat.count,
            }
            for (source, key), stat in sorted(self.dimension_stats.items())
            if stat.count > 0
        ]
        characters = [
            {
                "character_id": character_id,
                "mean": stat.mean,
                "variance": stat.variance,
                "count": stat.count,
            }
            for character_id, stat in sorted(self.character_style_stats.items())
            if stat.count > 0
        ]
        return {"version": 1, "dimensions": dimensions, "characters": characters}

    @staticmethod
    def _restore_stat(payload: dict[str, Any], label: str) -> RunningStat:
        mean = _safe_float(payload.get("mean"), None)
        variance = _safe_float(payload.get("variance"), None)
        try:
            count = int(payload.get("count", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.count must be a non-negative integer") from exc
        if mean is None or variance is None or variance < 0.0 or count < 0:
            raise ValueError(f"Invalid adaptive reward checkpoint entry: {label}")
        return RunningStat(mean=mean, variance=variance, count=count)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""

        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Unsupported roleplay reward checkpoint format")
        dimensions = state.get("dimensions", [])
        characters = state.get("characters", [])
        if not isinstance(dimensions, list) or not isinstance(characters, list):
            raise ValueError("Roleplay reward checkpoint entries must be lists")

        restored_dimensions: dict[tuple[str, str], RunningStat] = {}
        for i, payload in enumerate(dimensions):
            if not isinstance(payload, dict):
                raise ValueError(f"dimensions[{i}] must be a mapping")
            source = str(payload.get("source", "")).strip()
            key = str(payload.get("key", "")).strip()
            if not source or not key:
                raise ValueError(f"dimensions[{i}] must define source and key")
            restored_dimensions[(source, key)] = self._restore_stat(
                payload, f"dimensions[{i}]"
            )

        restored_characters: dict[str, RunningStat] = {}
        for i, payload in enumerate(characters):
            if not isinstance(payload, dict):
                raise ValueError(f"characters[{i}] must be a mapping")
            character_id = str(payload.get("character_id", "")).strip()
            if not character_id:
                raise ValueError(f"characters[{i}] must define character_id")
            restored_characters[character_id] = self._restore_stat(
                payload, f"characters[{i}]"
            )

        self.dimension_stats = defaultdict(RunningStat, restored_dimensions)
        self.character_style_stats = defaultdict(RunningStat, restored_characters)

    def _profile(self, source: str) -> dict[str, Any]:
        return self.profiles.get(source, self.defaults)

    def _confidence(self, row: dict[str, Any], profile: dict[str, Any]) -> float:
        judge_cfg = profile.get("judge") or {}
        if not judge_cfg.get("enabled", False):
            return 1.0

        confidence = 1.0
        success_key = judge_cfg.get("success_key")
        if success_key:
            success = _safe_float(row.get(success_key), None)
            confidence *= max(0.0, min(1.0, success)) if success is not None else 0.0
        failure_key = judge_cfg.get("failure_key")
        if failure_key:
            failure = _safe_float(row.get(failure_key), None)
            confidence *= 1.0 - max(0.0, min(1.0, failure)) if failure is not None else 0.0
        valid_fraction_key = judge_cfg.get("valid_fraction_key")
        if valid_fraction_key:
            valid_fraction = _safe_float(row.get(valid_fraction_key), None)
            confidence *= max(0.0, min(1.0, valid_fraction)) if valid_fraction is not None else 0.0
        confidence_key = judge_cfg.get("confidence_key")
        if confidence_key:
            explicit = _safe_float(row.get(confidence_key), None)
            confidence *= max(0.0, min(1.0, explicit)) if explicit is not None else 0.0
        std_key = judge_cfg.get("std_key")
        if std_key:
            std = _safe_float(row.get(std_key), None)
            if std is not None:
                confidence *= math.exp(-4.0 * max(0.0, std))
        return max(0.0, min(1.0, confidence))

    def _adaptive_weight(self, source: str, spec: dict[str, Any], score: float) -> float:
        stat = self.dimension_stats[(source, str(spec["key"]))]
        capability = stat.mean if stat.count else 0.5
        deficit = self.minimum_weight + (1.0 - capability) ** self.deficit_power
        adaptive = (1.0 - self.adaptive_strength) + self.adaptive_strength * deficit
        distance = score - capability
        zpd = math.exp(-(distance * distance) / (2.0 * self.zpd_sigma * self.zpd_sigma))
        zpd = self.zpd_floor + (1.0 - self.zpd_floor) * zpd
        return max(1e-6, float(spec.get("weight", 1.0)) * adaptive * zpd)

    def _channel_score(
        self,
        source: str,
        dimensions: list[tuple[dict[str, Any], float]],
        channel: str,
        base_reward: float,
    ) -> float:
        selected = [(spec, score) for spec, score in dimensions if spec.get("channel", "task") == channel]
        if not selected:
            return base_reward
        weights = [self._adaptive_weight(source, spec, score) for spec, score in selected]
        return sum(weight * score for weight, (_, score) in zip(weights, selected, strict=True)) / sum(weights)

    @staticmethod
    def _place_sequence_scores(batch, reward_tensor: torch.Tensor, scores: list[float]) -> torch.Tensor:
        shaped = torch.zeros_like(reward_tensor)
        response_width = reward_tensor.shape[-1]
        attention = batch.batch.get("attention_mask")
        if attention is not None:
            response_attention = attention[:, -response_width:]
        else:
            response_attention = batch.batch.get("response_mask")
        if response_attention is None:
            raise ValueError("Roleplay reward shaping requires attention_mask or response_mask")
        if response_attention.shape != reward_tensor.shape:
            raise ValueError(
                "Response attention shape must match reward tensor shape; "
                f"got {tuple(response_attention.shape)} and {tuple(reward_tensor.shape)}"
            )
        for i, score in enumerate(scores):
            valid = torch.nonzero(response_attention[i] > 0, as_tuple=False).flatten()
            if len(valid):
                shaped[i, int(valid[-1])] = float(score)
        return shaped

    def process_batch(self, batch, reward_tensor: torch.Tensor, update_state: bool = True):
        """Return calibrated reward tensor and per-sequence CRPO metadata."""

        if reward_tensor.ndim != 2 or reward_tensor.shape[-1] < 1:
            raise ValueError(
                f"reward_tensor must have shape (batch, response_length>0), got {tuple(reward_tensor.shape)}"
            )
        base_scores = reward_tensor.sum(dim=-1).detach().cpu().tolist()
        n = len(base_scores)
        sources = _column(batch.non_tensor_batch.get("data_source"), n, "data_source", "unknown")
        gen_uids = _column(batch.non_tensor_batch.get("gen_uid"), n, "gen_uid", None)
        for i in range(n):
            if gen_uids[i] is None:
                gen_uids[i] = f"row:{i}"
        character_ids = _column(
            batch.non_tensor_batch.get("roleplay/character_id"), n, "roleplay/character_id", None
        )
        anchor_flags = _column(
            batch.non_tensor_batch.get("roleplay/is_anchor"), n, "roleplay/is_anchor", 0.0
        )
        attention = batch.batch.get("attention_mask")
        if attention is not None:
            if attention.ndim != 2 or attention.shape[0] != n or attention.shape[1] < reward_tensor.shape[1]:
                raise ValueError(
                    "attention_mask must be rank 2, batch-aligned, and at least as wide as reward_tensor"
                )
            response_attention = attention[:, -reward_tensor.shape[-1] :]
        else:
            response_mask = batch.batch.get("response_mask")
            if response_mask is None or response_mask.shape != reward_tensor.shape:
                raise ValueError("response_mask must match reward_tensor when attention_mask is absent")
            response_attention = response_mask
        valid_sequences = response_attention.sum(dim=-1).detach().cpu().bool().tolist()

        task_scores: list[float] = []
        style_scores: list[float] = []
        final_scores: list[float] = []
        style_prior_adv: list[float] = []
        strategies: list[str] = []
        confidences: list[float] = []
        dimension_updates: list[tuple[str, str, float, str]] = []
        style_updates: list[tuple[str, float, str]] = []

        for i in range(n):
            source = str(sources[i] if sources[i] is not None else "unknown")
            profile = self._profile(source)
            strategy = str(profile.get("strategy", "dapo")).lower()
            row = {}
            for key, values in batch.non_tensor_batch.items():
                try:
                    if len(values) == n:
                        row[key] = values[i]
                except TypeError:
                    continue
            base_reward = max(0.0, min(1.0, float(_safe_float(base_scores[i], self.neutral))))
            confidence = self._confidence(row, profile)
            judge_enabled = bool((profile.get("judge") or {}).get("enabled", False))

            normalized_dims: list[tuple[dict[str, Any], float]] = []
            for spec in profile.get("dimensions", []) or []:
                mask_key = spec.get("mask_key")
                if mask_key and not bool(_safe_float(row.get(mask_key), 0.0)):
                    continue
                score = normalize_dimension(row.get(spec.get("key")), spec)
                if score is not None:
                    normalized_dims.append((spec, score))

            raw_task_score = self._channel_score(source, normalized_dims, "task", base_reward)
            raw_style_score = self._channel_score(source, normalized_dims, "style", base_reward)
            mix = float(profile.get("task_reward_weight", 0.55))
            adaptive_score = mix * raw_task_score + (1.0 - mix) * raw_style_score

            if normalized_dims and self.preserve_penalties:
                penalty_factor = 1.0
                reference_key = profile.get("penalty_reference_key")
                reference_reward = _safe_float(row.get(reference_key), None) if reference_key else None
                if reference_reward is not None and base_reward < reference_reward and reference_reward > 1e-6:
                    penalty_factor *= max(0.0, min(1.0, base_reward / reference_reward))
                elif profile.get("infer_penalty_from_base", False):
                    base_core_weights = [float(spec.get("weight", 1.0)) for spec, _ in normalized_dims]
                    base_core = sum(
                        weight * score
                        for weight, (_, score) in zip(base_core_weights, normalized_dims, strict=True)
                    ) / sum(base_core_weights)
                    if base_reward < base_core and base_core > 1e-6:
                        penalty_factor *= max(0.0, min(1.0, base_reward / base_core))
                for penalty in profile.get("penalties", []) or []:
                    active = _safe_float(row.get(penalty.get("key")), 0.0)
                    if active and active > 0.0:
                        penalty_factor *= float(penalty.get("factor", 1.0))
                adaptive_score *= max(0.0, min(1.0, penalty_factor))

            task_score = raw_task_score
            style_score = raw_style_score
            if judge_enabled:
                # CRPO consumes task/style streams directly. Calibrate both streams,
                # not only the final scalar, so failed or unreliable judges cannot
                # inject arbitrary dual-stream advantages or poison style history.
                task_score = calibrated_score(raw_task_score, confidence, self.neutral)
                style_score = calibrated_score(raw_style_score, confidence, self.neutral)
                adaptive_score = calibrated_score(adaptive_score, confidence, self.neutral)

            if strategy != "crpo":
                task_score = base_reward
                style_score = base_reward
                adaptive_score = base_reward

            character_id = str(character_ids[i] or source)
            style_stat = self.character_style_stats[character_id]
            if style_stat.count >= self.min_style_history and style_stat.variance > 1e-8:
                prior_adv = (style_score - style_stat.mean) / math.sqrt(style_stat.variance + 1e-6)
            else:
                prior_adv = float("nan")

            gid = str(gen_uids[i])
            is_anchor = bool(_safe_float(anchor_flags[i], 0.0))
            state_is_reliable = not judge_enabled or confidence > 0.0
            if update_state and not is_anchor and valid_sequences[i] and state_is_reliable:
                for spec, score in normalized_dims:
                    update_score = calibrated_score(score, confidence, self.neutral) if judge_enabled else score
                    dimension_updates.append((source, str(spec["key"]), update_score, gid))
                style_updates.append((character_id, style_score, gid))

            task_scores.append(max(0.0, min(1.0, task_score)))
            style_scores.append(max(0.0, min(1.0, style_score)))
            final_scores.append(max(0.0, min(1.0, adaptive_score)))
            style_prior_adv.append(prior_adv)
            strategies.append(strategy)
            confidences.append(confidence)

        if update_state:
            seen_dims: set[tuple[str, str, str]] = set()
            for source, key, score, gid in dimension_updates:
                dedup_key = (source, key, gid)
                if dedup_key not in seen_dims:
                    self.dimension_stats[(source, key)].update(score, self.ema_alpha)
                    seen_dims.add(dedup_key)
            seen_styles: set[tuple[str, str]] = set()
            for character_id, score, gid in style_updates:
                dedup_key = (character_id, gid)
                if dedup_key not in seen_styles:
                    self.character_style_stats[character_id].update(score, self.ema_alpha)
                    seen_styles.add(dedup_key)

        info = {
            "roleplay/task_score": np.asarray(task_scores, dtype=np.float32),
            "roleplay/style_score": np.asarray(style_scores, dtype=np.float32),
            "roleplay/final_score": np.asarray(final_scores, dtype=np.float32),
            "roleplay/style_prior_adv": np.asarray(style_prior_adv, dtype=np.float32),
            "roleplay/strategy": np.asarray(strategies, dtype=object),
            "roleplay/judge_confidence": np.asarray(confidences, dtype=np.float32),
        }
        return self._place_sequence_scores(batch, reward_tensor, final_scores), info
