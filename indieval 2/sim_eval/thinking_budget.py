"""Opt-in output allowances, separate from benchmark answer caps."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .errors import ConfigurationError

DEFAULT_THINKING_TOKEN_BUDGET = 2048
THINKING_BUDGET_REVISION = "vllm-answer-plus-thinking-content-episode-v2"


def thinking_token_budget(role: Mapping[str, Any]) -> int | None:
    """Return a vLLM thinking budget or an explicit relay output allowance.

    The relay allowance raises the total request cap, without instructing the
    service to use a particular reasoning mode or an exact reasoning budget.
    """
    allowance = role.get("reasoning_token_allowance")
    if allowance is not None:
        if role.get("profile") != "relay":
            raise ConfigurationError("reasoning_token_allowance requires profile=relay")
        if isinstance(allowance, bool) or not isinstance(allowance, int) or allowance < 0:
            raise ConfigurationError("reasoning_token_allowance must be a non-negative integer")
        return allowance
    if role.get("profile", role.get("backend")) != "vllm":
        return None
    extra = role.get("extra_body") or {}
    if (extra.get("chat_template_kwargs") or {}).get("enable_thinking") is not True:
        return None
    value = extra.get("thinking_token_budget", DEFAULT_THINKING_TOKEN_BUDGET)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError("extra_body.thinking_token_budget must be a non-negative integer")
    return value


def apply_thinking_defaults(role: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(role))
    budget = thinking_token_budget(result)
    if budget is not None:
        extra = result.get("extra_body") or {}
        if result.get("profile", result.get("backend")) == "vllm":
            extra = result["extra_body"]
            extra["thinking_token_budget"] = budget
            extra["include_reasoning"] = False
        # A raw override would bypass benchmark answer caps and episode guards.
        if "max_tokens" in extra or "max_completion_tokens" in extra:
            raise ConfigurationError("set answer max_tokens in generation, not extra_body, for thinking evaluation")
    return result


def thinking_budget_identity(role: Mapping[str, Any]) -> dict[str, Any]:
    budget = thinking_token_budget(role)
    if budget is None:
        return {}
    if role.get("profile") == "relay":
        return {"evaluated_reasoning_allowance": {
            "revision": "relay-answer-plus-output-allowance-v1",
            "reasoning_token_allowance": budget,
            "reasoning_mode": "provider_default",
            "max_tokens_semantics": "benchmark_answer_cap_plus_allowance",
            "episode_budget_scope": "evaluated_model_content_only",
        }}
    return {"evaluated_thinking_budget": {
        "revision": THINKING_BUDGET_REVISION,
        "thinking_token_budget": budget,
        "include_reasoning": False,
        "max_tokens_semantics": "benchmark_answer_cap_plus_thinking",
        "episode_budget_scope": "evaluated_model_content_only",
    }}
