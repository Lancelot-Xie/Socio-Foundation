"""Sampling parameter resolution shared by agent-loop rollouts."""

from __future__ import annotations

from typing import Any


def resolve_sampling_params(config: Any, *, validate: bool) -> dict[str, Any]:
    params = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": 1.0,
        "logprobs": config.calculate_log_probs,
    }
    if validate:
        params.update(
            temperature=config.val_kwargs.temperature,
            top_p=config.val_kwargs.top_p,
            top_k=config.val_kwargs.top_k,
        )
        # vLLM SamplingParams has no do_sample field. Temperature zero is its
        # greedy-decoding representation.
        if not config.val_kwargs.do_sample:
            params["temperature"] = 0.0
    return params
