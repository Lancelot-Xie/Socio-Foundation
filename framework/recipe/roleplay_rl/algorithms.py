"""CRPO advantage estimator with FoldGRPO-compatible grouping."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch


def _as_object_array(value, length: int, default=None, *, name: str) -> np.ndarray:
    if value is None:
        out = np.empty(length, dtype=object)
        out[:] = default
        return out
    out = np.asarray(value, dtype=object)
    if out.ndim == 0:
        raise ValueError(f"{name} must have length {length}, got a scalar")
    out = out.reshape(-1)
    if len(out) != length:
        raise ValueError(f"{name} has length {len(out)}; expected {length}")
    return out


def _as_score_tensor(value, fallback: torch.Tensor, *, name: str) -> torch.Tensor:
    if value is None:
        return fallback.detach().clone()
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        raise ValueError(f"{name} must have length {len(fallback)}, got a scalar")
    array = array.reshape(-1)
    if len(array) != len(fallback):
        raise ValueError(f"{name} has length {len(array)}; expected {len(fallback)}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return torch.as_tensor(array, dtype=fallback.dtype, device=fallback.device)


def _group_keys(index, agent_role, length: int):
    indices = _as_object_array(index, length, name="index")
    roles = _as_object_array(agent_role, length, "", name="agent_role")
    return [(str(indices[i]), str(roles[i] or "")) for i in range(length)]


def _dedup_group_normalize(
    values: torch.Tensor,
    index: np.ndarray,
    gen_uid: np.ndarray | None,
    agent_role: np.ndarray | None,
    valid_sequences: torch.Tensor,
    epsilon: float,
    normalize_std: bool,
) -> torch.Tensor:
    """GRPO-normalize valid rows after counting each multi-agent rollout once."""

    n = len(values)
    gids = _as_object_array(gen_uid, n, name="gen_uid")
    if gen_uid is None:
        gids = np.asarray([f"row:{i}" for i in range(n)], dtype=object)
    keys = _group_keys(index, agent_role, n)
    grouped: dict[Any, list[torch.Tensor]] = defaultdict(list)
    seen: dict[Any, set[str]] = defaultdict(set)
    for i, key in enumerate(keys):
        if not bool(valid_sequences[i]):
            continue
        gid = str(gids[i])
        if gid in seen[key]:
            continue
        seen[key].add(gid)
        grouped[key].append(values[i])

    means: dict[Any, torch.Tensor] = {}
    stds: dict[Any, torch.Tensor] = {}
    for key, samples in grouped.items():
        stacked = torch.stack(samples)
        # Match verl's GRPO/FoldGRPO singleton behavior: a one-sample group is
        # left as its raw score rather than being centered to zero.
        means[key] = stacked.mean() if len(samples) > 1 else values.new_tensor(0.0)
        stds[key] = stacked.std() if len(samples) > 1 else values.new_tensor(1.0)

    out = torch.zeros_like(values)
    for i, key in enumerate(keys):
        if not bool(valid_sequences[i]):
            continue
        centered = values[i] - means[key]
        out[i] = centered / (stds[key] + epsilon) if normalize_std else centered
    return out


def _roleplay_config(config: Any) -> Any:
    if config is None:
        return {}
    if hasattr(config, "get"):
        return config.get("roleplay_rl", {})
    return getattr(config, "roleplay_rl", {})


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def compute_crpo_foldgrpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    gen_uid: np.ndarray | None = None,
    agent_role: np.ndarray | None = None,
    roleplay_task_scores: np.ndarray | None = None,
    roleplay_style_scores: np.ndarray | None = None,
    roleplay_style_prior_adv: np.ndarray | None = None,
    roleplay_strategy: np.ndarray | None = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine prompt-normalized task advantage and character-style advantage.

    Rows configured as ``grpo`` or ``dapo`` fall back to the final scalar reward.
    DAPO's policy-loss stability features remain configured on the actor side;
    its outcome advantage is intentionally GRPO-compatible.
    """

    if token_level_rewards.ndim != 2 or token_level_rewards.shape[-1] < 1:
        raise ValueError(
            "token_level_rewards must have shape (batch, response_length>0), "
            f"got {tuple(token_level_rewards.shape)}"
        )
    if response_mask.shape != token_level_rewards.shape:
        raise ValueError(
            "response_mask must match token_level_rewards; "
            f"got {tuple(response_mask.shape)} and {tuple(token_level_rewards.shape)}"
        )
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")

    with torch.no_grad():
        final_scores = token_level_rewards.sum(dim=-1).detach()
        if not torch.isfinite(final_scores).all():
            raise ValueError("token_level_rewards produced non-finite sequence rewards")
        n = len(final_scores)
        if index is None:
            raise ValueError("index is required for CRPO/FoldGRPO prompt grouping")
        _as_object_array(index, n, name="index")

        valid_sequences = response_mask.to(dtype=torch.bool).any(dim=-1)
        task_scores = _as_score_tensor(roleplay_task_scores, final_scores, name="roleplay_task_scores")
        style_scores = _as_score_tensor(roleplay_style_scores, final_scores, name="roleplay_style_scores")
        strategies = _as_object_array(roleplay_strategy, n, "dapo", name="roleplay_strategy")
        normalized_strategies = np.asarray(
            [str(strategy).lower() for strategy in strategies], dtype=object
        )
        invalid_strategies = sorted(set(normalized_strategies) - {"crpo", "dapo", "grpo"})
        if invalid_strategies:
            raise ValueError(
                f"roleplay_strategy contains unsupported values: {invalid_strategies}"
            )

        task_adv = _dedup_group_normalize(
            task_scores,
            index,
            gen_uid,
            agent_role,
            valid_sequences,
            epsilon,
            norm_adv_by_std_in_grpo,
        )
        group_style_adv = _dedup_group_normalize(
            style_scores,
            index,
            gen_uid,
            agent_role,
            valid_sequences,
            epsilon,
            norm_adv_by_std_in_grpo,
        )
        fallback_adv = _dedup_group_normalize(
            final_scores,
            index,
            gen_uid,
            agent_role,
            valid_sequences,
            epsilon,
            norm_adv_by_std_in_grpo,
        )

        style_adv = group_style_adv.clone()
        if roleplay_style_prior_adv is not None:
            prior = np.asarray(roleplay_style_prior_adv, dtype=float)
            if prior.ndim == 0:
                raise ValueError(f"roleplay_style_prior_adv must have length {n}, got a scalar")
            prior = prior.reshape(-1)
            if len(prior) != n:
                raise ValueError(f"roleplay_style_prior_adv has length {len(prior)}; expected {n}")
            valid_prior = np.isfinite(prior) & valid_sequences.detach().cpu().numpy()
            if valid_prior.any():
                prior_tensor = torch.as_tensor(prior, dtype=final_scores.dtype, device=final_scores.device)
                valid_prior_tensor = torch.as_tensor(valid_prior, dtype=torch.bool, device=final_scores.device)
                style_adv[valid_prior_tensor] = prior_tensor[valid_prior_tensor]

        role_cfg = _roleplay_config(config)
        task_lambda = float(_cfg_get(role_cfg, "task_adv_weight", 0.55))
        style_clip = float(_cfg_get(role_cfg, "style_adv_clip", 5.0))
        if not math.isfinite(task_lambda) or not 0.0 <= task_lambda <= 1.0:
            raise ValueError(f"task_adv_weight must be finite and in [0, 1], got {task_lambda}")
        if not math.isfinite(style_clip) or style_clip < 0.0:
            raise ValueError(f"style_adv_clip must be finite and non-negative, got {style_clip}")
        style_adv = style_adv.clamp(-style_clip, style_clip)

        combined = task_lambda * task_adv + (1.0 - task_lambda) * style_adv
        crpo_mask = torch.as_tensor(
            normalized_strategies == "crpo",
            dtype=torch.bool,
            device=final_scores.device,
        )
        sequence_adv = torch.where(crpo_mask, combined, fallback_adv)
        sequence_adv = sequence_adv.unsqueeze(-1) * response_mask
    return sequence_adv, sequence_adv
