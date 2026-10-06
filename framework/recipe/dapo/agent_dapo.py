"""DAPO helpers for Simulation's variable-length agent-loop trajectories."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch


def dynamic_sampling_exhaustion_action(
    *,
    pending_prompt_count: int,
    target_prompt_count: int,
    num_gen_batches: int,
    max_num_gen_batches: int,
    exhaustion_strategy: str,
) -> str:
    """Resolve bounded Dynamic Sampling into one explicit trainer action."""

    strategy = str(exhaustion_strategy).lower()
    if strategy not in {"error", "use_partial"}:
        raise ValueError(
            "exhaustion_strategy must be 'error' or 'use_partial', "
            f"got {strategy!r}"
        )
    if target_prompt_count <= 0:
        raise ValueError("target_prompt_count must be positive")
    if pending_prompt_count >= target_prompt_count:
        return "ready"
    if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
        return "continue"
    if strategy == "error":
        return "error"
    return "use_partial" if pending_prompt_count > 0 else "use_unfiltered"


def apply_overlong_penalty(
    reward_tensor: torch.Tensor,
    valid_response_mask: torch.Tensor,
    *,
    max_response_length: int,
    buffer_length: int,
    penalty_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply DAPO's linear overlong penalty at each sequence's terminal token."""

    if buffer_length <= 0:
        raise ValueError("DAPO overlong buffer_length must be positive")
    if max_response_length < buffer_length:
        raise ValueError("max_response_length must be at least buffer_length")
    if penalty_factor < 0:
        raise ValueError("DAPO overlong penalty_factor must be non-negative")
    if reward_tensor.shape != valid_response_mask.shape:
        raise ValueError("reward_tensor and valid_response_mask must have identical shapes")

    adjusted = reward_tensor.clone()
    lengths = valid_response_mask.to(torch.long).sum(dim=-1)
    expected_length = max_response_length - buffer_length
    excess = (lengths - expected_length).clamp_min(0).to(adjusted.dtype)
    penalties = -(excess / float(buffer_length)) * float(penalty_factor)

    nonempty = lengths > 0
    if nonempty.any():
        rows = torch.arange(len(lengths), device=adjusted.device)[nonempty]
        terminal = (lengths[nonempty] - 1).to(torch.long)
        adjusted[rows, terminal] += penalties[nonempty]
    return adjusted, penalties, lengths


def informative_prompt_uids(
    prompt_uids: Iterable[object],
    rollout_uids: Iterable[object],
    sequence_rewards: Iterable[float],
    *,
    epsilon: float = 0.0,
    agent_roles: Iterable[object] | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Return prompt groups whose distinct rollout rewards have nonzero range.

    Simulation may emit several sub-sequences for one rollout.  Their shared
    ``gen_uid`` is collapsed with ``max`` before measuring variation across the
    n rollouts belonging to one prompt ``uid``. When roles are present, each
    role is measured independently, matching FoldGRPO's ``(uid, role)`` groups;
    a prompt is informative when any trainable role has nonzero reward range.
    """

    prompt_uids = [str(value) for value in prompt_uids]
    rollout_uids = [str(value) for value in rollout_uids]
    rewards = [float(value) for value in sequence_rewards]
    roles = (
        [str(value) if value is not None else "" for value in agent_roles]
        if agent_roles is not None
        else [""] * len(prompt_uids)
    )
    if not (len(prompt_uids) == len(rollout_uids) == len(rewards) == len(roles)):
        raise ValueError("uid, gen_uid, reward, and role arrays must have equal lengths")

    rollout_scores: dict[tuple[str, str, str], float] = {}
    prompt_order: list[str] = []
    seen_prompts: set[str] = set()
    for prompt_uid, rollout_uid, reward, role in zip(prompt_uids, rollout_uids, rewards, roles, strict=True):
        if prompt_uid not in seen_prompts:
            seen_prompts.add(prompt_uid)
            prompt_order.append(prompt_uid)
        key = (prompt_uid, role, rollout_uid)
        rollout_scores[key] = max(rollout_scores.get(key, -float("inf")), reward)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (prompt_uid, role, _), reward in rollout_scores.items():
        grouped[(prompt_uid, role)].append(reward)

    informative = []
    ranges = {}
    for prompt_uid in prompt_order:
        role_ranges = []
        for (group_uid, _role), values in grouped.items():
            if group_uid != prompt_uid:
                continue
            role_range = max(values) - min(values) if len(values) > 1 else 0.0
            role_ranges.append(role_range)
        reward_range = max(role_ranges, default=0.0)
        ranges[prompt_uid] = reward_range
        if reward_range > epsilon:
            informative.append(prompt_uid)
    return informative, ranges


def select_prompt_rows(prompt_uids: Iterable[object], selected_uids: Iterable[object]) -> np.ndarray:
    """Return row indices belonging to selected prompt groups, preserving order."""

    selected = {str(value) for value in selected_uids}
    return np.asarray(
        [index for index, uid in enumerate(prompt_uids) if str(uid) in selected],
        dtype=np.int64,
    )


def select_first_valid_prompt_groups(
    prompt_uids: Iterable[object],
    valid_rows: Iterable[bool],
    max_prompt_groups: int,
) -> tuple[np.ndarray, list[str]]:
    """Select valid rows from the first prompt groups that produced responses."""

    prompt_uids = np.asarray(list(prompt_uids), dtype=object)
    valid_rows = np.asarray(list(valid_rows), dtype=bool)
    if len(prompt_uids) != len(valid_rows):
        raise ValueError("prompt_uids and valid_rows must have equal lengths")
    if max_prompt_groups < 0:
        raise ValueError("max_prompt_groups must be non-negative")

    valid_indices = np.flatnonzero(valid_rows)
    prompt_order = list(dict.fromkeys(str(prompt_uids[index]) for index in valid_indices))
    selected_prompt_uids = prompt_order[:max_prompt_groups]
    selected_indices = select_prompt_rows(prompt_uids, selected_prompt_uids)
    return selected_indices[valid_rows[selected_indices]], selected_prompt_uids


def concat_reward_extra_info_chunks(chunks: list[tuple[dict[str, Any], int]]) -> dict[str, Any]:
    """Concatenate row-aligned reward metadata across generation batches."""

    keys = {key for values, _ in chunks for key in values}
    output: dict[str, Any] = {}
    for key in keys:
        merged: list[Any] = []
        row_aligned = True
        for values, size in chunks:
            value = values.get(key)
            if value is None:
                merged.extend([None] * size)
            elif isinstance(value, np.ndarray) and len(value) == size:
                merged.extend(value.tolist())
            elif isinstance(value, list) and len(value) == size:
                merged.extend(value)
            else:
                row_aligned = False
                break
        if row_aligned:
            output[key] = merged
        else:
            output[key] = next(values[key] for values, _ in reversed(chunks) if key in values)
    return output


def fold_agent_rollout_scores(
    sequence_scores: torch.Tensor,
    prompt_uids: Iterable[object],
    rollout_uids: Iterable[object],
    agent_roles: Iterable[object] | None = None,
    sequence_penalties: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[object]]:
    """Aggregate multi-agent sub-sequence scores and map them back to rows.

    Task rewards use ``max`` because Simulation emits both terminal-only and
    repeated reward layouts. DAPO overlong penalties are aggregated
    separately with ``min`` so a negative terminal penalty cannot be hidden by
    another zero-reward sub-sequence from the same rollout.
    """

    prompt_uids = list(prompt_uids)
    rollout_uids = list(rollout_uids)
    roles = list(agent_roles) if agent_roles is not None else [None] * len(prompt_uids)
    if sequence_penalties is not None and len(sequence_penalties) != len(sequence_scores):
        raise ValueError("scores and sequence_penalties must have equal lengths")
    if not (len(sequence_scores) == len(prompt_uids) == len(rollout_uids) == len(roles)):
        raise ValueError("scores, uid, gen_uid, and role arrays must have equal lengths")

    rollout_task_scores: dict[tuple[object, str], torch.Tensor] = {}
    rollout_penalties: dict[tuple[object, str], torch.Tensor] = {}
    row_group_keys: list[object] = []
    row_rollout_keys: list[tuple[object, str]] = []
    penalties = sequence_penalties if sequence_penalties is not None else torch.zeros_like(sequence_scores)
    for score, penalty, uid, rollout_uid, role in zip(
        sequence_scores, penalties, prompt_uids, rollout_uids, roles, strict=True
    ):
        group_key = (uid, str(role) if role is not None else "") if agent_roles is not None else uid
        rollout_key = (group_key, str(rollout_uid))
        row_group_keys.append(group_key)
        row_rollout_keys.append(rollout_key)
        task_score = score - penalty
        if rollout_key in rollout_task_scores:
            rollout_task_scores[rollout_key] = torch.maximum(rollout_task_scores[rollout_key], task_score)
            rollout_penalties[rollout_key] = torch.minimum(rollout_penalties[rollout_key], penalty)
        else:
            rollout_task_scores[rollout_key] = task_score
            rollout_penalties[rollout_key] = penalty

    row_scores = torch.stack(
        [rollout_task_scores[key] + rollout_penalties[key] for key in row_rollout_keys]
    )
    return row_scores, row_group_keys


def align_non_tensor_batch_chunks(
    chunks: list[dict[str, np.ndarray]], sizes: list[int]
) -> list[dict[str, np.ndarray]]:
    """Fill optional non-tensor keys so DataProto chunks can be concatenated."""

    if len(chunks) != len(sizes):
        raise ValueError("chunks and sizes must have equal lengths")
    keys = {key for chunk in chunks for key in chunk}
    aligned = []
    for chunk, size in zip(chunks, sizes, strict=True):
        output = dict(chunk)
        for key in keys - output.keys():
            missing = np.empty(size, dtype=object)
            missing[:] = None
            output[key] = missing
        aligned.append(output)
    return aligned
