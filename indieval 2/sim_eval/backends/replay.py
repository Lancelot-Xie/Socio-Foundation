"""Deterministic offline backend for tests and auditable response replay."""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import ModelRequest, ModelResponse, TokenUsage
from ..errors import BackendTimeoutError, InjectedBackendError
from ..interfaces import ModelBackend


class ReplayBackend(ModelBackend):
    name = "replay"

    def __init__(
        self,
        responses: Mapping[str, str | Mapping[str, Any] | ModelResponse] | None = None,
        *,
        errors: Mapping[str, str | Mapping[str, Any]] | None = None,
        default_response: str | Mapping[str, Any] | ModelResponse | None = None,
    ) -> None:
        self._responses = dict(responses or {})
        self._errors = dict(errors or {})
        self._default_response = default_response

    @staticmethod
    def request_key(request: ModelRequest) -> str:
        return request.request_id or request.fingerprint

    def generate(self, request: ModelRequest) -> ModelResponse:
        key = self.request_key(request)
        injected = self._errors.get(key, self._errors.get("*"))
        if injected is not None:
            if isinstance(injected, str):
                kind, message = injected, f"replay backend injected {injected} for {key}"
                retryable = injected == "timeout"
            else:
                kind = str(injected.get("kind", "error"))
                message = str(injected.get("message", f"replay backend injected {kind} for {key}"))
                retryable = bool(injected.get("retryable", kind == "timeout"))
            if kind == "timeout":
                raise BackendTimeoutError(message)
            raise InjectedBackendError(f"{message} (retryable={retryable})")

        value = self._responses.get(key, self._responses.get("*", self._default_response))
        if value is None:
            raise InjectedBackendError(
                f"replay response missing for request key {key}; add a fixture response or explicit default_response"
            )
        if isinstance(value, ModelResponse):
            return value
        if isinstance(value, str):
            return ModelResponse(text=value, finish_reason="replayed", latency_ms=0.0, response_id=key)
        usage_value = value.get("usage")
        usage = None
        if isinstance(usage_value, Mapping):
            usage = TokenUsage(
                prompt_tokens=usage_value.get("prompt_tokens"),
                completion_tokens=usage_value.get("completion_tokens"),
                total_tokens=usage_value.get("total_tokens"),
                cached_tokens=usage_value.get("cached_tokens"),
            )
        return ModelResponse(
            text=str(value.get("text", "")),
            finish_reason=value.get("finish_reason", "replayed"),
            usage=usage,
            latency_ms=float(value.get("latency_ms", 0.0)),
            raw=dict(value.get("raw", {})),
            response_id=str(value.get("response_id", key)),
        )

