"""Dependency-free implementation of the native OpenAI Responses API."""

from __future__ import annotations

import copy
import json
import os
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
from .openai_compatible import (
    STRUCTURED_MODES,
    _extract_json_text,
    _http_error,
    _schema_parts,
    _structured_parameter_rejected,
)


Transport = Callable[[Mapping[str, Any], Mapping[str, str], float], Mapping[str, Any]]


def _responses_tool(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    if tool.get("type") != "function" or not isinstance(tool.get("function"), Mapping):
        return dict(tool)
    function = tool["function"]
    return {
        "type": "function",
        "name": function.get("name"),
        "description": function.get("description"),
        "parameters": function.get("parameters") or {},
        **({"strict": function["strict"]} if "strict" in function else {}),
    }


class OpenAIResponsesBackend(ModelBackend):
    """Call ``POST /v1/responses`` and normalize it to ``ModelResponse``."""

    name = "openai_responses"

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
        profile: str = "openai",
        send_sampling_params: bool = False,
        extra_body: Mapping[str, Any] | None = None,
        structured_output: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ConfigurationError("OpenAI Responses base_url must start with http:// or https://")
        if timeout <= 0:
            raise ConfigurationError("backend timeout must be positive")
        if profile != "openai":
            raise ConfigurationError("OpenAI Responses backend requires profile=openai")
        structured = dict(structured_output or {})
        mode = str(structured.get("mode") or "auto")
        if mode not in STRUCTURED_MODES - {"structured_outputs"}:
            raise ConfigurationError(f"unsupported Responses structured output mode: {mode}")
        raw_order = structured.get("fallback_order") or ("json_schema", "json_object", "none")
        if not isinstance(raw_order, Sequence) or isinstance(raw_order, (str, bytes)):
            raise ConfigurationError("structured_output.fallback_order must be an array")
        order = tuple(str(value) for value in raw_order)
        if not order or any(value not in {"json_schema", "json_object", "none"} for value in order):
            raise ConfigurationError("Responses structured fallback order is invalid")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.require_api_key = require_api_key
        self.extra_headers = dict(extra_headers or {})
        self.profile = profile
        self.send_sampling_params = bool(send_sampling_params)
        self.extra_body = copy.deepcopy(dict(extra_body or {}))
        self.structured_output_mode = mode
        self.structured_output_order = order
        self._transport = transport or self._http_transport

    def _candidate_modes(self, request: ModelRequest) -> tuple[str, ...]:
        if not isinstance(request.response_format, Mapping):
            return ("none",)
        if self.structured_output_mode != "auto":
            return (self.structured_output_mode,)
        contract_type = str(request.response_format.get("type") or "json_object")
        if contract_type == "json_schema":
            return self.structured_output_order
        modes = tuple(mode for mode in self.structured_output_order if mode != "json_schema")
        return modes or ("json_object", "none")

    def build_payload(self, request: ModelRequest, *, structured_mode: str | None = None) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "tool":
                history.append({"type": "function_call_output", "call_id": message.tool_call_id,
                                "output": message.content})
            elif message.tool_calls:
                if message.content:
                    history.append({"role": message.role, "content": message.content})
                for call in message.tool_calls:
                    history.append({"type": "function_call", "call_id": call["id"],
                                    "name": call["function"]["name"],
                                    "arguments": call["function"]["arguments"]})
            else:
                history.append({"role": message.role, "content": message.content})
        payload: dict[str, Any] = {
            "model": request.model,
            "input": history,
            # Evaluation traffic should not be retained unless an explicit
            # extra_body override opts in.
            "store": False,
        }
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens
        if self.send_sampling_params:
            if request.temperature is not None:
                payload["temperature"] = request.temperature
            if request.top_p is not None:
                payload["top_p"] = request.top_p
        if request.reasoning_effort:
            payload["reasoning"] = {"effort": request.reasoning_effort}
        if request.tools:
            payload["tools"] = [_responses_tool(tool) for tool in request.tools]
            if (
                request.metadata.get("benchmark_id") == "tau_usi"
                and request.metadata.get("actor") == "fixed_assistant"
            ):
                payload["parallel_tool_calls"] = False

        mode = structured_mode or ("json_object" if request.response_format else "none")
        if isinstance(request.response_format, Mapping):
            name, schema, strict = _schema_parts(request.response_format)
            if mode == "json_schema":
                payload["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": name,
                        "schema": copy.deepcopy(schema),
                        "strict": strict,
                    }
                }
            elif mode == "json_object":
                payload["text"] = {"format": {"type": "json_object"}}
            elif mode != "none":
                raise ConfigurationError(f"unsupported Responses structured output mode: {mode}")
        payload.update(copy.deepcopy(self.extra_body))
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
            f"{self.base_url}/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise _http_error(exc, provider="OpenAI Responses") from exc
        except TimeoutError as exc:
            raise BackendTimeoutError(f"OpenAI Responses request timed out after {timeout}s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise BackendTimeoutError(f"OpenAI Responses request timed out after {timeout}s") from exc
            raise BackendError(f"OpenAI Responses request failed: {exc.reason}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError("OpenAI Responses endpoint returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise BackendError("OpenAI Responses endpoint returned a non-object payload")
        return decoded

    @staticmethod
    def _output(
        result: Mapping[str, Any], request: ModelRequest
    ) -> tuple[str, str, dict[str, Any] | None, bool]:
        status = str(result.get("status") or "")
        incomplete = result.get("incomplete_details")
        reason = str(incomplete.get("reason") or "") if isinstance(incomplete, Mapping) else ""
        if reason == "content_filter":
            raise BackendSafetyError(
                "OpenAI Responses output was interrupted by the content filter",
                status_code=200,
                error_code="content_filter",
                error_type="response_safety_filter",
            )
        text_parts: list[str] = []
        function_calls: list[Mapping[str, Any]] = []
        for item in result.get("output") or ():
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "function_call":
                function_calls.append(item)
                continue
            if item.get("type") != "message":
                continue
            for content in item.get("content") or ():
                if not isinstance(content, Mapping):
                    continue
                if content.get("type") == "refusal":
                    raise BackendSafetyError(
                        "OpenAI Responses returned an explicit safety refusal",
                        status_code=200,
                        error_code="refusal",
                        error_type="response_safety_refusal",
                    )
                if content.get("type") == "output_text":
                    text_parts.append(str(content.get("text") or ""))
        output_truncated = status == "incomplete" and reason in {
            "max_output_tokens",
            "max_tokens",
        }
        if status == "incomplete" and not output_truncated:
            raise BackendError(f"OpenAI Responses output was incomplete: {reason or 'unknown reason'}")
        finish_reason = reason or status
        if (
            function_calls
            and request.metadata.get("benchmark_id") == "tau_usi"
            and request.metadata.get("actor") == "fixed_assistant"
        ):
            if len(function_calls) != 1:
                raise BackendError("tau-USI fixed assistant must return at most one native tool call")
            call = function_calls[0]
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise BackendError("tau-USI native tool arguments are invalid JSON") from exc
            if not isinstance(arguments, Mapping):
                raise BackendError("tau-USI native tool arguments must be an object")
            name = str(call.get("name") or "").strip()
            if not name:
                raise BackendError("tau-USI native tool call is missing function name")
            return "", finish_reason, {"name": name, "arguments": dict(arguments)}, output_truncated
        return "".join(text_parts), finish_reason, None, output_truncated

    def _parse_response(
        self,
        result: Mapping[str, Any],
        request: ModelRequest,
        *,
        latency_ms: float,
        structured_mode: str,
    ) -> ModelResponse:
        text, finish_reason, native_tool_call, output_truncated = self._output(result, request)
        # Preserve token-capped partial output for benchmark-level parsing and
        # scoring instead of classifying it as a backend failure.
        if request.response_format is not None and not output_truncated:
            if structured_mode == "none":
                text = _extract_json_text(text)
            else:
                try:
                    json.loads(text)
                except json.JSONDecodeError as exc:
                    raise BackendStructuredOutputError(
                        f"Responses {structured_mode} returned invalid JSON"
                    ) from exc
        usage_raw = result.get("usage") or {}
        usage = (
            TokenUsage(
                prompt_tokens=usage_raw.get("input_tokens"),
                completion_tokens=usage_raw.get("output_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
                cached_tokens=(usage_raw.get("input_tokens_details") or {}).get("cached_tokens"),
            )
            if usage_raw
            else None
        )
        raw = dict(result)
        protocol_metadata = {
            "protocol": "openai_responses",
            "profile": self.profile,
            "structured_output_mode": structured_mode,
            "token_limit_field": "max_output_tokens",
            "sampling_parameters_sent": self.send_sampling_params,
            "output_truncated": output_truncated,
        }
        if output_truncated:
            protocol_metadata["truncation_reason"] = finish_reason
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
                    f"OpenAI Responses request timed out after {self.timeout}s"
                ) from exc
            except Exception as exc:
                raise BackendError(
                    f"OpenAI Responses transport failed: {type(exc).__name__}: {exc}"
                ) from exc
            return self._parse_response(
                result,
                request,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                structured_mode=mode,
            )
        assert last_error is not None
        raise last_error


__all__ = ["OpenAIResponsesBackend"]
