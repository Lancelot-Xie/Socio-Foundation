"""Small benchmark-agnostic helpers for result accounting."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from ..contracts import (
    CaseResult,
    ChatMessage,
    MetricValue,
    ModelRequest,
    ModelResponse,
    ResultStatus,
    TokenUsage,
)
from ..errors import BackendStructuredOutputError, ParseError
from ..interfaces import ModelBackend


ParsedT = TypeVar("ParsedT")
DEFAULT_CONTRACT_RETRIES = 2
_MAX_RETRY_ECHO_CHARS = 2_000


def json_schema_response_format(name: str, schema: Mapping[str, Any]) -> Mapping[str, Any]:
    """Declare one strict JSON transport contract for every supported API backend."""

    return {
        "type": "json_schema",
        "name": name,
        "schema": dict(schema),
        "strict": True,
    }


def parse_json_object(response: ModelResponse, *, label: str) -> Mapping[str, Any]:
    """Parse an object while normalizing JSON decoder failures to the retryable contract error."""

    try:
        value = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"{label} must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ParseError(f"{label} must be a JSON object")
    return value


def merge_system_instructions(
    messages: Sequence[ChatMessage],
    *instructions: str,
) -> tuple[ChatMessage, ...]:
    """Merge benchmark constraints into one leading system message.

    OpenAI-compatible chat templates are not consistently able to interpret a
    system role after user/assistant content.  Benchmark adapters therefore
    keep one official-style system prompt at the front of the request.
    """

    source = tuple(messages)
    if not source or source[0].role != "system":
        raise ValidationError("system instructions require one leading system message")
    if any(message.role == "system" for message in source[1:]):
        raise ValidationError("system instructions must already be consolidated at the front")
    normalized = [instruction.strip() for instruction in instructions]
    if not normalized or any(not instruction for instruction in normalized):
        raise ValidationError("system instructions cannot be empty")
    leading = source[0]
    merged = replace(
        leading,
        content="\n\n".join((leading.content.rstrip(), *normalized)),
    )
    return (merged, *source[1:])


def corrective_retry_request(
    base_request: ModelRequest,
    *,
    attempt: int,
    reason: str,
    contract: str,
    previous_output: str | None = None,
    request_suffix: str = "contract_retry",
) -> ModelRequest:
    """Build a private transport-level correction without changing episode state."""

    messages = list(base_request.messages)
    if previous_output is not None:
        messages.append(
            ChatMessage(
                "assistant",
                previous_output[:_MAX_RETRY_ECHO_CHARS],
                metadata={"visibility": "transport_retry"},
            )
        )
    messages.append(
        ChatMessage(
            "user",
            f"The previous response violated the output contract: {reason}. {contract}",
            metadata={"visibility": "transport_retry"},
        )
    )
    return replace(
        base_request,
        request_id=f"{base_request.request_id or 'request'}:{request_suffix}:{attempt}",
        messages=tuple(messages),
        seed=(base_request.seed + attempt if base_request.seed is not None else None),
        metadata={
            **dict(base_request.metadata),
            "transport_retry_kind": request_suffix,
            "transport_retry_attempt": attempt,
        },
    )


def generate_and_parse_with_contract_retries(
    *,
    backend: ModelBackend,
    request: ModelRequest,
    parser: Callable[[ModelResponse], ParsedT],
    responses: list[ModelResponse],
    contract: str,
    max_retries: int = DEFAULT_CONTRACT_RETRIES,
) -> ParsedT:
    """Retry only structured transport/parse violations, keeping the parser strict."""

    current = request
    for attempt in range(max_retries + 1):
        try:
            response = backend.generate(current)
        except BackendStructuredOutputError as exc:
            if attempt >= max_retries:
                raise
            current = corrective_retry_request(
                request,
                attempt=attempt + 1,
                reason=str(exc),
                contract=contract,
            )
            continue
        responses.append(response)
        try:
            return parser(response)
        except ParseError as exc:
            if attempt >= max_retries:
                raise
            current = corrective_retry_request(
                request,
                attempt=attempt + 1,
                reason=str(exc),
                contract=contract,
                previous_output=response.text,
            )
    raise AssertionError("unreachable structured contract retry loop")


def combine_usage(usages: Iterable[TokenUsage | None]) -> TokenUsage | None:
    present = [usage for usage in usages if usage is not None]
    if not present:
        return None

    def total(field: str) -> int | None:
        values = [getattr(usage, field) for usage in present]
        known = [value for value in values if value is not None]
        return sum(known) if known else None

    return TokenUsage(
        prompt_tokens=total("prompt_tokens"),
        completion_tokens=total("completion_tokens"),
        total_tokens=total("total_tokens"),
        cached_tokens=total("cached_tokens"),
    )


def aggregate_named_metrics(results: Sequence[CaseResult], *, namespace: str) -> Mapping[str, MetricValue]:
    values: dict[str, list[float]] = defaultdict(list)
    unavailable: dict[str, int] = defaultdict(int)
    for result in results:
        for metric in result.metrics:
            if metric.value is None:
                unavailable[metric.name] += 1
            elif isinstance(metric.value, (int, float, bool)):
                values[metric.name].append(float(metric.value))
    metrics: dict[str, MetricValue] = {}
    for name in sorted(set(values) | set(unavailable)):
        available_values = values.get(name, [])
        metrics[name] = MetricValue(
            name=name,
            value=(sum(available_values) / len(available_values)) if available_values else None,
            numerator=sum(available_values) if available_values else None,
            denominator=len(available_values) if available_values else 0,
            metadata={"unavailable_count": unavailable.get(name, 0), "aggregation": "arithmetic_mean_available"},
        )
    completed = sum(result.status == ResultStatus.COMPLETED for result in results)
    failed = sum(result.status == ResultStatus.FAILED for result in results)
    metrics[f"{namespace}.case_completion_rate"] = MetricValue(
        name=f"{namespace}.case_completion_rate",
        value=completed / len(results) if results else None,
        numerator=completed,
        denominator=len(results),
    )
    metrics[f"{namespace}.case_failure_count"] = MetricValue(
        name=f"{namespace}.case_failure_count",
        value=failed,
        direction="lower_is_better",
        unit="cases",
        numerator=failed,
        denominator=len(results),
    )
    return metrics
