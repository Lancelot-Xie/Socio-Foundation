"""Generic no-profile anchor selection and metadata helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


ANCHOR_FLAG = "_roleplay_generic_anchor"


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _anchor_config(config: Any):
    algorithm = _cfg_get(config, "algorithm", {})
    roleplay = _cfg_get(algorithm, "roleplay_rl", {})
    return _cfg_get(roleplay, "generic_anchor", {})


def prepare_rollout_item(item: dict[str, Any], config: Any, is_train: bool) -> dict[str, Any]:
    """Mark one configured rollout slot as the CRPO generic anchor."""

    prepared = dict(item)
    prepared[ANCHOR_FLAG] = False
    if not is_train:
        return prepared

    anchor_cfg = _anchor_config(config)
    if not bool(_cfg_get(anchor_cfg, "enabled", False)):
        return prepared
    tasks = {str(x) for x in (_cfg_get(anchor_cfg, "tasks", []) or [])}
    source = str(prepared.get("data_source", ""))
    rollout_slot = int(prepared.get("rollout_n", -1))
    if source in tasks and rollout_slot == int(_cfg_get(anchor_cfg, "slot", 0)):
        prepared[ANCHOR_FLAG] = True
    return prepared


def is_generic_anchor(data: dict[str, Any]) -> bool:
    return bool(data.get(ANCHOR_FLAG, False))


def stable_character_id(data_source: str, item: dict[str, Any]) -> str:
    """Build a privacy-preserving stable key for CRPO style history."""

    row = item.get("extra_info") if isinstance(item.get("extra_info"), dict) else item
    if isinstance(row.get("raw"), str):
        try:
            parsed_raw = json.loads(row["raw"])
            if isinstance(parsed_raw, dict):
                row = {**row, **parsed_raw}
        except (TypeError, ValueError):
            pass
    candidates: list[Any]
    if data_source == "sotopia":
        position = row.get("eval_position", "agent1")
        candidates = [row.get(f"{position}_name"), row.get("episode_id"), row.get("index")]
    elif data_source == "mirrorbench":
        candidates = [row.get("conversation_id"), row.get("persona"), row.get("task_description")]
    elif data_source.startswith("humanual"):
        candidates = [row.get("persona_id"), row.get("id"), row.get("persona")]
    elif data_source.startswith("sim_arena"):
        candidates = [row.get("user_id"), row.get("id"), row.get("user_profile_text")]
    elif data_source == "userllm":
        candidates = [row.get("user_id"), row.get("id"), row.get("intent")]
    else:
        candidates = [row.get("character_id"), row.get("id"), row.get("index")]
    raw = next((str(value) for value in candidates if value not in (None, "")), data_source)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{data_source}:{digest}"


def mark_outputs(outputs, data_source: str, item: dict[str, Any]):
    """Attach anchor and character metadata to one output or output list."""

    is_anchor = is_generic_anchor(item)
    character_id = stable_character_id(data_source, item)
    output_list = outputs if isinstance(outputs, list) else [outputs]
    for output in output_list:
        output.extra_fields["roleplay/is_anchor"] = is_anchor
        output.extra_fields.setdefault("roleplay/character_id", character_id)
        reward_info = output.extra_fields.setdefault("reward_extra_info", {})
        # Character ID is grouping metadata, not a numeric reward metric.
        reward_info.pop("roleplay/character_id", None)
        reward_info["roleplay/is_anchor"] = float(is_anchor)
    return outputs
