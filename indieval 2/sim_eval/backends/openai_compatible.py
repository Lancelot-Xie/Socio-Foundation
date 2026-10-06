"""Capability-aware OpenAI-compatible Chat Completions backends."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..contracts import ModelRequest, ModelResponse, TokenUsage
from ..errors import (
    BackendError,
    BackendHTTPError,
    BackendSafetyError,
    BackendStructuredOutputError,
    BackendTimeoutError,
    ConfigurationError,
)
from ..interfaces import ModelBackend
from ..thinking_budget import apply_thinking_defaults, thinking_token_budget
from .episode_budget import remaining_budget_from_metadata


Transport = Callable[[Mapping[str, Any], Mapping[str, str], float], Mapping[str, Any]]

CHAT_PROFILES = {"openai_compatible", "vllm", "deepseek", "relay", "openai"}
STRUCTURED_MODES = {"auto", "structured_outputs", "json_schema", "json_object", "none"}
_PROFILE_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "openai_compatible": {
        "token_limit_field": "max_tokens",
        "send_sampling_params": True,
        "structured_order": ("json_schema", "json_object", "none"),
    },
    "vllm": {
        "token_limit_field": "max_tokens",
        "send_sampling_params": True,
        "structured_order": ("structured_outputs", "json_schema", "json_object", "none"),
    },
    "deepseek": {
        "token_limit_field": "max_tokens",
        "send_sampling_params": True,
        "structured_order": ("json_schema", "json_object", "none"),
    },
    "relay": {
        "token_limit_field": "max_completion_tokens",
        "send_sampling_params": False,
        "structured_order": ("json_schema", "json_object", "none"),
    },
    "openai": {
        "token_limit_field": "max_completion_tokens",
        "send_sampling_params": False,
        "structured_order": ("json_schema", "json_object", "none"),
    },
}


def _error_fields(payload: Any) -> tuple[str | None, str | None, str]:
    error = payload.get("error") if isinstance(payload, Mapping) else None
    if not isinstance(error, Mapping):
        error = payload if isinstance(payload, Mapping) else {}
    code = str(error.get("code") or "").strip() or None
    error_type = str(error.get("type") or "").strip() or None
    message = str(error.get("message") or "").strip()
    return code, error_type, message


def _is_explicit_safety_error(
    *, status_code: int | None, code: str | None, error_type: str | None, message: str
) -> bool:
    """Conservatively recognize provider-declared safety blocks.

    HTTP failures trigger model fallback only for status 400 plus an explicit
    safety marker. Successful response refusals/content-filter finishes are
    classified separately by response parsers.
    """

    if status_code != 400:
        return False
    machine = " ".join(value.casefold() for value in (code, error_type) if value)
    if any(
        marker in machine
        for marker in (
            "content_filter",
            "content_policy_violation",
            "responsibleaipolicyviolation",
            "safety_violation",
            "moderation_blocked",
        )
    ):
        return True
    lowered = message.casefold()
    return any(
        marker in lowered
        for marker in (
            "blocked by the safety",
            "blocked by safety",
            "violates the safety policy",
            "violates our safety policy",
            "content policy violation",
            "blocked by moderation",
            "unsafe content",
            "安全策略拦截",
            "内容安全拦截",
            "内容审核拒绝",
        )
    )


def _http_error(exc: urllib.error.HTTPError, *, provider: str) -> BackendHTTPError:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        raw = ""
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {}
    code, error_type, api_message = _error_fields(payload)
    raw_detail = re.sub(r"\s+", " ", raw).strip()[:1000]
    detail = api_message or raw_detail or str(exc.reason or "HTTP request failed")
    detail = re.sub(r"\s+", " ", detail).strip()[:1000]
    message = f"{provider} request failed with HTTP {exc.code}: {detail}"
    error_class = (
        BackendSafetyError
        if _is_explicit_safety_error(
            status_code=exc.code,
            code=code,
            error_type=error_type,
            message=api_message or raw_detail,
        )
        else BackendHTTPError
    )
    return error_class(
        message,
        status_code=exc.code,
        error_code=code,
        error_type=error_type,
    )


def _schema_parts(response_format: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], bool]:
    raw = response_format.get("json_schema")
    if isinstance(raw, Mapping):
        schema = raw.get("schema")
        if isinstance(schema, Mapping):
            return (
                str(raw.get("name") or "sim_eval_output")[:64],
                schema,
                bool(raw.get("strict", True)),
            )
    schema = response_format.get("schema")
    if isinstance(schema, Mapping):
        return (
            str(response_format.get("name") or "sim_eval_output")[:64],
            schema,
            bool(response_format.get("strict", True)),
        )
    # vLLM can use this permissive schema to guarantee an object for legacy
    # adapters that currently declare json_object but validate fields locally.
    return "sim_eval_object", {"type": "object"}, False


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise BackendStructuredOutputError("structured response was empty")
    try:
        value = json.loads(stripped)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
        pass
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates.append(stripped)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raise BackendStructuredOutputError("no valid JSON value was found in the model response")


def _structured_parameter_rejected(exc: BackendHTTPError, mode: str) -> bool:
    if exc.status_code != 400:
        return False
    text = " ".join(
        value.casefold()
        for value in (str(exc), exc.error_code or "", exc.error_type or "")
    )
    rejection = any(
        marker in text
        for marker in (
            "unknown parameter",
            "unsupported parameter",
            "not supported",
            "unrecognized request argument",
            "extra inputs are not permitted",
            "invalid parameter",
        )
    )
    fields = {
        "structured_outputs": ("structured_outputs",),
        "json_schema": ("json_schema", "response_format"),
        "json_object": ("json_object", "response_format"),
        "none": (),
    }[mode]
    return rejection and any(field in text for field in fields)


class OpenAICompatibleBackend(ModelBackend):
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
        require_api_key: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        transport: Transport | None = None,
        profile: str = "openai_compatible",
        token_limit_field: str | None = None,
        send_sampling_params: bool | None = None,
        extra_body: Mapping[str, Any] | None = None,
        reasoning_token_allowance: int | None = None,
        structured_output: Mapping[str, Any] | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ConfigurationError("OpenAI-compatible base_url must start with http:// or https://")
        if timeout <= 0:
            raise ConfigurationError("backend timeout must be positive")
        if profile not in CHAT_PROFILES:
            raise ConfigurationError(f"unsupported Chat Completions profile: {profile}")
        defaults = _PROFILE_DEFAULTS[profile]
        token_field = token_limit_field or str(defaults["token_limit_field"])
        if token_field not in {"max_tokens", "max_completion_tokens"}:
            raise ConfigurationError("token_limit_field must be max_tokens or max_completion_tokens")
        structured = dict(structured_output or {})
        mode = str(structured.get("mode") or "auto")
        if mode not in STRUCTURED_MODES:
            raise ConfigurationError(f"unsupported structured output mode: {mode}")
        raw_order = structured.get("fallback_order") or defaults["structured_order"]
        if not isinstance(raw_order, Sequence) or isinstance(raw_order, (str, bytes)):
            raise ConfigurationError("structured_output.fallback_order must be an array")
        order = tuple(str(value) for value in raw_order)
        if not order or any(value not in STRUCTURED_MODES - {"auto"} for value in order):
            raise ConfigurationError("structured output fallback order contains an unsupported mode")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.require_api_key = require_api_key
        self.extra_headers = dict(extra_headers or {})
        self.profile = profile
        self.token_limit_field = token_field
        self.send_sampling_params = (
            bool(defaults["send_sampling_params"])
            if send_sampling_params is None
            else bool(send_sampling_params)
        )
        self.extra_body = apply_thinking_defaults({
            "profile": profile, "extra_body": dict(extra_body or {}),
            "reasoning_token_allowance": reasoning_token_allowance,
        })["extra_body"]
        self.reasoning_token_allowance = reasoning_token_allowance
        self.structured_output_mode = mode
        self.structured_output_order = order
        self._transport = transport or self._http_transport
        self._structured_mode_cache: dict[str, str] = {}
        self._cache_lock = threading.Lock()

    def _candidate_modes(self, request: ModelRequest) -> tuple[str, ...]:
        response_format = request.response_format
        if not isinstance(response_format, Mapping):
            return ("none",)
        if self.structured_output_mode != "auto":
            return (self.structured_output_mode,)
        contract_type = str(response_format.get("type") or "json_object")
        modes = list(self.structured_output_order)
        if contract_type != "json_schema" and self.profile != "vllm":
            modes = [mode for mode in modes if mode not in {"structured_outputs", "json_schema"}]
        cache_key = "json_schema" if contract_type == "json_schema" else "json_object"
        with self._cache_lock:
            cached = self._structured_mode_cache.get(cache_key)
        if cached in modes:
            modes.remove(cached)
            modes.insert(0, cached)
        return tuple(modes or ["none"])

    def build_payload(self, request: ModelRequest, *, structured_mode: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [item.to_chat_dict() for item in request.messages],
        }
        if request.max_tokens is not None:
            payload[self.token_limit_field] = request.max_tokens
        if self.send_sampling_params:
            optional_sampling = {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "seed": request.seed,
            }
            payload.update({key: value for key, value in optional_sampling.items() if value is not None})
        if request.reasoning_effort:
            payload["reasoning_effort"] = request.reasoning_effort
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.tools:
            payload["tools"] = list(request.tools)
            if (
                request.metadata.get("benchmark_id") == "tau_usi"
                and request.metadata.get("actor") == "fixed_assistant"
            ):
                payload["parallel_tool_calls"] = False

        mode = structured_mode or ("json_object" if request.response_format else "none")
        extra_body = copy.deepcopy(self.extra_body)
        if isinstance(request.response_format, Mapping):
            name, schema, strict = _schema_parts(request.response_format)
            if mode == "structured_outputs":
                extra_body["structured_outputs"] = {"json": copy.deepcopy(schema)}
            elif mode == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": copy.deepcopy(schema),
                        "strict": strict,
                    },
                }
            elif mode == "json_object":
                payload["response_format"] = {"type": "json_object"}
            elif mode != "none":
                raise ConfigurationError(f"unsupported structured output mode: {mode}")
        payload.update(extra_body)
        thinking_budget = thinking_token_budget({
            "profile": self.profile, "extra_body": self.extra_body,
            "reasoning_token_allowance": self.reasoning_token_allowance,
        })
        if thinking_budget is not None:
            if request.max_tokens is None:
                raise ConfigurationError("thinking evaluation requires a finite answer max_tokens")
            total_limit = request.max_tokens + thinking_budget
            allowance = remaining_budget_from_metadata(request.metadata)
            if allowance is not None:
                total_limit = min(total_limit, allowance)
            payload[self.token_limit_field] = total_limit
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        token = self.api_key or os.getenv(self.api_key_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif self.require_api_key:
            raise ConfigurationError(
                f"missing API key: pass api_key or set {self.api_key_env}; no request was sent"
            )
        return headers

    def _http_transport(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise _http_error(exc, provider="Chat Completions") from exc
        except TimeoutError as exc:
            raise BackendTimeoutError(f"OpenAI-compatible request timed out after {timeout}s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise BackendTimeoutError(f"OpenAI-compatible request timed out after {timeout}s") from exc
            raise BackendError(f"OpenAI-compatible request failed: {exc.reason}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError("OpenAI-compatible endpoint returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise BackendError("OpenAI-compatible endpoint returned a non-object payload")
        return decoded

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}:
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        return str(content or "")

    @staticmethod
    def _response_diagnostics(
        *,
        message: Mapping[str, Any],
        finish_reason: Any,
        text: str,
    ) -> str:
        reasoning = message.get("reasoning_content")
        tool_calls = message.get("tool_calls")
        preview = re.sub(r"\s+", " ", text).strip()[:240]
        return ", ".join(
            (
                f"finish_reason={finish_reason!r}",
                f"content_chars={len(text)}",
                f"reasoning_content_chars={len(str(reasoning or ''))}",
                f"tool_call_count={len(tool_calls) if isinstance(tool_calls, list) else 0}",
                f"content_preview={preview!r}",
            )
        )

    def _parse_response(
        self,
        result: Mapping[str, Any],
        request: ModelRequest,
        *,
        latency_ms: float,
        structured_mode: str,
    ) -> ModelResponse:
        native_tool_call: dict[str, Any] | None = None
        try:
            choice = result["choices"][0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason")
            if str(finish_reason or "").casefold() == "content_filter" or message.get("refusal"):
                raise BackendSafetyError(
                    "Chat Completions response was blocked by the provider safety filter",
                    status_code=200,
                    error_code="content_filter",
                    error_type="response_safety_filter",
                )
            text = self._content_text(message.get("content", choice.get("text", "")))
            diagnostics = self._response_diagnostics(
                message=message,
                finish_reason=finish_reason,
                text=text,
            )
            output_truncated = str(finish_reason or "").casefold() in {
                "length",
                "max_tokens",
            }
            native_tool_calls = message.get("tool_calls")
            if (
                native_tool_calls
                and request.metadata.get("benchmark_id") == "tau_usi"
                and request.metadata.get("actor") == "fixed_assistant"
            ):
                if not isinstance(native_tool_calls, list) or len(native_tool_calls) != 1:
                    raise BackendError(
                        "tau-USI fixed assistant must return at most one native tool call "
                        f"({diagnostics})"
                    )
                function = native_tool_calls[0].get("function")
                if not isinstance(function, Mapping):
                    raise BackendError("tau-USI native tool call is missing function")
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise BackendError("tau-USI native tool arguments are invalid JSON") from exc
                if not isinstance(arguments, Mapping):
                    raise BackendError("tau-USI native tool arguments must be an object")
                name = str(function.get("name") or "").strip()
                if not name:
                    raise BackendError("tau-USI native tool call is missing function name")
                native_tool_call = {"name": name, "arguments": dict(arguments)}
                # Match tau-bench: a native tool call wins over simultaneous
                # assistant content and is not converted into model-authored text.
                text = ""
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("OpenAI-compatible response is missing choices[0]") from exc

        # A provider-side output cap is a model-generation outcome, not a
        # transport/infrastructure failure.  Preserve the partial text and let
        # the benchmark parser/scorer decide whether it contains a valid answer.
        # In particular, do not turn an unfinished JSON object into a backend
        # structured-output exception merely because the provider truncated it.
        if request.response_format is not None and not output_truncated:
            if structured_mode == "none":
                try:
                    text = _extract_json_text(text)
                except BackendStructuredOutputError as exc:
                    raise BackendStructuredOutputError(f"{exc} ({diagnostics})") from exc
            else:
                try:
                    json.loads(text)
                except json.JSONDecodeError as exc:
                    raise BackendStructuredOutputError(
                        f"{structured_mode} returned invalid JSON ({diagnostics})"
                    ) from exc
        usage_raw = result.get("usage") or {}
        usage = (
            TokenUsage(
                prompt_tokens=usage_raw.get("prompt_tokens"),
                completion_tokens=usage_raw.get("completion_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
                cached_tokens=(usage_raw.get("prompt_tokens_details") or {}).get("cached_tokens"),
            )
            if usage_raw
            else None
        )
        raw = dict(result)
        protocol_metadata = {
            "protocol": "chat_completions",
            "profile": self.profile,
            "structured_output_mode": structured_mode,
            "token_limit_field": self.token_limit_field,
            "sampling_parameters_sent": self.send_sampling_params,
            "output_truncated": output_truncated,
        }
        if self.reasoning_token_allowance is not None:
            sent = self.build_payload(request, structured_mode=structured_mode)
            protocol_metadata["reasoning_allowance"] = {
                "reasoning_token_allowance": self.reasoning_token_allowance,
                "reasoning_mode": "provider_default",
                "requested_total_max_tokens": sent[self.token_limit_field],
            }
        elif thinking_token_budget({"profile": self.profile, "extra_body": self.extra_body}) is not None:
            sent = self.build_payload(request, structured_mode=structured_mode)
            protocol_metadata["thinking_budget"] = {
                "thinking_token_budget": sent["thinking_token_budget"],
                "include_reasoning": sent["include_reasoning"],
                "requested_total_max_tokens": sent[self.token_limit_field],
            }
        if output_truncated:
            protocol_metadata["truncation_reason"] = str(finish_reason)
        if native_tool_call is not None:
            protocol_metadata["native_tool_call"] = native_tool_call
        raw["_sim_eval"] = protocol_metadata
        return ModelResponse(
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            raw=raw,
            response_id=result.get("id"),
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        headers = self._headers()
        modes = self._candidate_modes(request)
        cache_key = (
            "json_schema"
            if isinstance(request.response_format, Mapping)
            and request.response_format.get("type") == "json_schema"
            else "json_object"
        )
        last_error: BackendHTTPError | None = None
        for index, mode in enumerate(modes):
            payload = self.build_payload(request, structured_mode=mode)
            started = time.perf_counter()
            try:
                result = self._transport(payload, headers, self.timeout)
            except BackendHTTPError as exc:
                last_error = exc
                if (
                    self.structured_output_mode == "auto"
                    and index < len(modes) - 1
                    and _structured_parameter_rejected(exc, mode)
                ):
                    continue
                raise
            except BackendError:
                raise
            except TimeoutError as exc:
                raise BackendTimeoutError(
                    f"OpenAI-compatible request timed out after {self.timeout}s"
                ) from exc
            except Exception as exc:
                raise BackendError(
                    f"OpenAI-compatible transport failed: {type(exc).__name__}: {exc}"
                ) from exc
            latency_ms = (time.perf_counter() - started) * 1000.0
            response = self._parse_response(
                result,
                request,
                latency_ms=latency_ms,
                structured_mode=mode,
            )
            if request.response_format is not None and self.structured_output_mode == "auto":
                with self._cache_lock:
                    self._structured_mode_cache[cache_key] = mode
            return response
        assert last_error is not None
        raise last_error


class VLLMBackend(OpenAICompatibleBackend):
    """OpenAI-compatible local vLLM server with no API key requirement."""

    name = "vllm"

    def __init__(self, *, base_url: str = "http://127.0.0.1:8000/v1", **kwargs: Any) -> None:
        kwargs.setdefault("require_api_key", False)
        kwargs.setdefault("profile", "vllm")
        super().__init__(base_url=base_url, **kwargs)


__all__ = [
    "CHAT_PROFILES",
    "OpenAICompatibleBackend",
    "STRUCTURED_MODES",
    "VLLMBackend",
]
