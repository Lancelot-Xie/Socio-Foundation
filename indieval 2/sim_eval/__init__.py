"""Independent, protocol-aware human-simulation evaluation framework."""

from .contracts import (
    BenchmarkCase,
    CaseResult,
    ChatMessage,
    ErrorState,
    MetricValue,
    ModelRequest,
    ModelResponse,
    RunIdentityInput,
    RunManifest,
    SampleManifest,
    SourceManifest,
    TokenUsage,
    TraceEvent,
    case_result_from_dict,
)

__all__ = [
    "BenchmarkCase",
    "CaseResult",
    "ChatMessage",
    "ErrorState",
    "MetricValue",
    "ModelRequest",
    "ModelResponse",
    "RunIdentityInput",
    "RunManifest",
    "SampleManifest",
    "SourceManifest",
    "TokenUsage",
    "TraceEvent",
    "case_result_from_dict",
]

__version__ = "0.3.40"
