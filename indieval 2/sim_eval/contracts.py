"""Dependency-free typed contracts shared by every protocol family."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from .errors import ValidationError
from .json_utils import (
    UNICODE_SANITIZATION_REVISION,
    jsonable,
    sanitize_unicode_scalars,
    sha256_digest,
)


JsonMap = Mapping[str, Any]


class ResultStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    name: str | None = None
    metadata: JsonMap = field(default_factory=dict)
    tool_calls: Sequence[JsonMap] = field(default_factory=tuple)
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if not self.role or not isinstance(self.content, str):
            raise ValidationError("ChatMessage requires a non-empty role and string content")
        if self.tool_calls and self.role != "assistant":
            raise ValidationError("tool_calls require an assistant message")
        if (self.role == "tool") != bool(self.tool_call_id):
            raise ValidationError("tool messages require a tool_call_id, exclusively")

    def to_chat_dict(self) -> dict[str, Any]:
        """Wire representation; private audit metadata never becomes model input."""
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = [dict(call) for call in self.tool_calls]
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValidationError(f"{name} cannot be negative")


@dataclass(frozen=True)
class ErrorState:
    stage: str
    kind: str
    message: str
    retryable: bool = False
    details: JsonMap = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    messages: Sequence[ChatMessage]
    model: str
    request_id: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    stop: Sequence[str] = field(default_factory=tuple)
    tools: Sequence[JsonMap] = field(default_factory=tuple)
    response_format: JsonMap | None = None
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValidationError("ModelRequest.messages cannot be empty")
        if not all(isinstance(message, ChatMessage) for message in self.messages):
            raise ValidationError("ModelRequest.messages must contain only ChatMessage values")
        system_positions = [
            index for index, message in enumerate(self.messages) if message.role == "system"
        ]
        tau_text_tool_history = (
            self.metadata.get("benchmark_id") == "tau_usi"
            and self.metadata.get("actor") == "fixed_assistant"
            and self.metadata.get("tool_protocol") == "supplemental-text-tool-calls-v4"
            and all(self.messages[index].metadata.get("source") == "tau_tool_feedback"
                    for index in system_positions[1:])
        )
        if len(system_positions) > 1 and not tau_text_tool_history:
            raise ValidationError(
                "ModelRequest.messages must contain at most one system message; merge system instructions"
            )
        if system_positions and system_positions[0] != 0:
            raise ValidationError(
                "ModelRequest system message must precede every non-system message"
            )
        pending_tools: set[str] = set()
        seen_tools: set[str] = set()
        for message in self.messages:
            if message.role == "tool":
                if message.tool_call_id not in pending_tools:
                    raise ValidationError("tool observation must match an outstanding tool call")
                pending_tools.remove(message.tool_call_id)
                continue
            if pending_tools:
                raise ValidationError("tool calls must receive observations before the next message")
            for call in message.tool_calls:
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in seen_tools:
                    raise ValidationError("history tool call IDs must be nonempty and unique")
                seen_tools.add(call_id)
                pending_tools.add(call_id)
        if pending_tools:
            raise ValidationError("history tool calls lack observations")
        if not self.model:
            raise ValidationError("ModelRequest.model cannot be empty")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValidationError("ModelRequest.max_tokens must be positive")

    @property
    def fingerprint(self) -> str:
        return sha256_digest(self)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    latency_ms: float | None = None
    raw: JsonMap = field(default_factory=dict)
    response_id: str | None = None

    def __post_init__(self) -> None:
        safe_text, text_stats = sanitize_unicode_scalars(self.text)
        safe_raw, raw_stats = sanitize_unicode_scalars(dict(self.raw))
        unpaired = (
            text_stats["unpaired_surrogates_replaced"]
            + raw_stats["unpaired_surrogates_replaced"]
        )
        paired = (
            text_stats["surrogate_pairs_normalized"]
            + raw_stats["surrogate_pairs_normalized"]
        )
        if not unpaired and not paired:
            return
        raw_payload = dict(safe_raw)
        sim_eval = raw_payload.get("_sim_eval")
        sim_eval_payload = dict(sim_eval) if isinstance(sim_eval, Mapping) else {}
        sim_eval_payload["unicode_sanitization"] = {
            "revision": UNICODE_SANITIZATION_REVISION,
            "unpaired_surrogates_replaced": unpaired,
            "surrogate_pairs_normalized": paired,
        }
        raw_payload["_sim_eval"] = sim_eval_payload
        object.__setattr__(self, "text", safe_text)
        object.__setattr__(self, "raw", raw_payload)


@dataclass(frozen=True)
class BenchmarkCase:
    benchmark_id: str
    case_id: str
    group_id: str
    split: str
    source_revision: str
    input_data: JsonMap
    gold: Any = None
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.benchmark_id, self.case_id, self.group_id, self.split, self.source_revision)):
            raise ValidationError("BenchmarkCase identity fields cannot be empty")


@dataclass(frozen=True)
class TraceEvent:
    turn: int
    actor: str
    kind: str
    content: Any
    visible_to: Sequence[str] = field(default_factory=tuple)
    timestamp: str | None = None
    metadata: JsonMap = field(default_factory=dict)


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: float | int | bool | None
    direction: str = "higher_is_better"
    unit: str | None = None
    numerator: float | int | None = None
    denominator: float | int | None = None
    uncertainty: JsonMap = field(default_factory=dict)
    metadata: JsonMap = field(default_factory=dict)


@dataclass(frozen=True)
class SourceManifest:
    benchmark_id: str
    source_kind: str
    source_revision: str
    split: str
    resolved_population: int
    urls: Sequence[str] = field(default_factory=tuple)
    file_hashes: JsonMap = field(default_factory=dict)
    license: str | None = None
    transformations: Sequence[JsonMap] = field(default_factory=tuple)
    schema_version: str = "1.0"
    metadata: JsonMap = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return sha256_digest(self)


@dataclass(frozen=True)
class SampleManifest:
    benchmark_id: str
    source_revision: str
    split: str
    profile: str
    result_label: str
    algorithm: str
    seed: int
    target: Any
    population_group_count: int
    population_case_count: int
    selected_group_ids: Sequence[str]
    selected_case_ids: Sequence[str]
    strata: Sequence[str] = field(default_factory=tuple)
    quotas: JsonMap = field(default_factory=dict)
    repetition_seeds: JsonMap = field(default_factory=dict)
    exclusions: Sequence[JsonMap] = field(default_factory=tuple)
    source_manifest_digest: str | None = None
    metadata: JsonMap = field(default_factory=dict)
    schema_version: str = "1.0"

    @property
    def digest(self) -> str:
        return sha256_digest(self)


@dataclass(frozen=True)
class RunIdentityInput:
    framework_version: str
    benchmark_id: str
    source_revision: str
    split: str
    profile: str
    sample_manifest_digest: str
    seed: int
    backend: str
    model: str
    decoding: JsonMap
    prompt_revision: str
    scorer_revision: str
    judge: JsonMap = field(default_factory=dict)
    environment: JsonMap = field(default_factory=dict)
    assistant_or_partner: JsonMap = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return sha256_digest(self)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at: str
    identity: RunIdentityInput
    catalog_revision: str
    result_label: str
    requested_case_count: int | None
    selected_case_count: int
    selected_group_count: int
    artifact_schema_version: str = "1.0"
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.run_id != self.identity.run_id:
            raise ValidationError("RunManifest.run_id does not match its identity payload")
        if self.selected_case_count < 0 or self.selected_group_count < 0:
            raise ValidationError("selected counts cannot be negative")

    @classmethod
    def create(
        cls,
        identity: RunIdentityInput,
        *,
        catalog_revision: str,
        result_label: str,
        requested_case_count: int | None,
        selected_case_count: int,
        selected_group_count: int,
        metadata: JsonMap | None = None,
    ) -> "RunManifest":
        return cls(
            run_id=identity.run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            identity=identity,
            catalog_revision=catalog_revision,
            result_label=result_label,
            requested_case_count=requested_case_count,
            selected_case_count=selected_case_count,
            selected_group_count=selected_group_count,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass(frozen=True)
class CaseResult:
    run_id: str
    benchmark_id: str
    case_id: str
    group_id: str
    repetition: int
    status: ResultStatus
    prediction: Any = None
    metrics: Sequence[MetricValue] = field(default_factory=tuple)
    trace: Sequence[TraceEvent] = field(default_factory=tuple)
    model_response: ModelResponse | None = None
    error: ErrorState | None = None
    latency_ms: float | None = None
    token_usage: TokenUsage | None = None
    metadata: JsonMap = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.repetition < 0:
            raise ValidationError("repetition cannot be negative")
        if self.status == ResultStatus.FAILED and self.error is None:
            raise ValidationError("failed CaseResult requires ErrorState")
        if self.status == ResultStatus.COMPLETED and self.error is not None:
            raise ValidationError("completed CaseResult cannot carry ErrorState")

    @property
    def checkpoint_key(self) -> tuple[str, int]:
        return (self.case_id, self.repetition)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


def case_result_from_dict(value: Mapping[str, Any]) -> CaseResult:
    """Rehydrate a checkpoint record for resume-time aggregation."""

    def usage(raw: Any) -> TokenUsage | None:
        return TokenUsage(**raw) if isinstance(raw, Mapping) else None

    response_raw = value.get("model_response")
    response = None
    if isinstance(response_raw, Mapping):
        response = ModelResponse(
            text=str(response_raw.get("text", "")),
            finish_reason=response_raw.get("finish_reason"),
            usage=usage(response_raw.get("usage")),
            latency_ms=response_raw.get("latency_ms"),
            raw=response_raw.get("raw") or {},
            response_id=response_raw.get("response_id"),
        )
    error_raw = value.get("error")
    error = ErrorState(**error_raw) if isinstance(error_raw, Mapping) else None
    return CaseResult(
        run_id=str(value["run_id"]),
        benchmark_id=str(value["benchmark_id"]),
        case_id=str(value["case_id"]),
        group_id=str(value["group_id"]),
        repetition=int(value["repetition"]),
        status=ResultStatus(value["status"]),
        prediction=value.get("prediction"),
        metrics=tuple(MetricValue(**item) for item in value.get("metrics") or ()),
        trace=tuple(TraceEvent(**item) for item in value.get("trace") or ()),
        model_response=response,
        error=error,
        latency_ms=value.get("latency_ms"),
        token_usage=usage(value.get("token_usage")),
        metadata=value.get("metadata") or {},
        schema_version=str(value.get("schema_version", "1.0")),
    )
