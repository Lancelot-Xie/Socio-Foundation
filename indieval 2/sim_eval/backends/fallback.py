"""Narrow model fallback policies for support-role API calls."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..contracts import ModelRequest, ModelResponse
from ..errors import BackendSafetyError, ConfigurationError
from ..interfaces import ModelBackend


class SafetyFallbackBackend(ModelBackend):
    """Use a fallback backend only after an explicit provider safety block."""

    name = "safety_fallback"

    def __init__(
        self,
        *,
        primary: ModelBackend,
        fallback: ModelBackend,
        fallback_request_overrides: Mapping[str, Any],
        primary_identity: Mapping[str, Any],
        fallback_identity: Mapping[str, Any],
        fallback_reason: str,
    ) -> None:
        if not fallback_reason.strip():
            raise ConfigurationError("safety fallback requires a nonempty fallback_reason")
        self.primary = primary
        self.fallback = fallback
        self.fallback_request_overrides = dict(fallback_request_overrides)
        self.primary_identity = dict(primary_identity)
        self.fallback_identity = dict(fallback_identity)
        self.fallback_reason = fallback_reason.strip()

    def generate(self, request: ModelRequest) -> ModelResponse:
        try:
            return self.primary.generate(request)
        except BackendSafetyError as exc:
            fallback_request = replace(request, **self.fallback_request_overrides)
            response = self.fallback.generate(fallback_request)
            raw = dict(response.raw)
            sim_eval = dict(raw.get("_sim_eval") or {})
            sim_eval["safety_fallback"] = {
                "used": True,
                "trigger": "explicit_provider_safety_block",
                "primary_status_code": exc.status_code,
                "primary_error_code": exc.error_code,
                "primary": dict(self.primary_identity),
                "fallback": dict(self.fallback_identity),
                "fallback_reason": self.fallback_reason,
            }
            raw["_sim_eval"] = sim_eval
            return replace(response, raw=raw)


__all__ = ["SafetyFallbackBackend"]
