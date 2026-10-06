"""Benchmark-specific generation policy shared by direct and suite evaluation."""
from __future__ import annotations

import copy
from typing import Any, Mapping


from .thinking_budget import apply_thinking_defaults


AGENTSENSE_JUDGE_TEMPERATURE = 0.8
AGENTSENSE_JUDGE_ROLES = ("judge_1", "judge_2", "judge_3")


def apply_benchmark_generation_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    """Apply opt-in thinking defaults and AgentSense judge policy after routing overrides.

    Actor actions, participant interviews and private-information questions are
    routed through evaluated_actor; their temperatures are not changed here.
    """
    result = copy.deepcopy(dict(document))
    for name, role in result.get("roles", {}).items():
        result["roles"][name] = apply_thinking_defaults(role)
    for name in ("target_user", "fixed_assistant"):
        if name in result:
            result[name] = apply_thinking_defaults(result[name])
    if result.get("benchmark_id") != "agentsense":
        return result

    def configure(role: dict[str, Any]) -> None:
        role["generation"] = {
            **dict(role.get("generation") or {}),
            "temperature": AGENTSENSE_JUDGE_TEMPERATURE,
        }
        # Some provider profiles omit sampling parameters unless opted in.
        role["send_sampling_params"] = True
        if isinstance(role.get("fallback"), dict):
            configure(role["fallback"])

    for name in AGENTSENSE_JUDGE_ROLES:
        role = result.get("roles", {}).get(name)
        if isinstance(role, dict):
            configure(role)
    return result
