"""Abstract extension boundaries for sources, protocols, environments, and reports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .contracts import BenchmarkCase, CaseResult, ChatMessage, MetricValue, ModelRequest, ModelResponse, SampleManifest


@dataclass(frozen=True)
class EnvironmentTransition:
    state: Any
    events: Sequence[Any] = field(default_factory=tuple)
    terminal: bool = False
    terminal_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CaseSource(ABC):
    """Acquires or loads cases without deciding how to sample or score them."""

    @abstractmethod
    def load(self) -> Iterable[BenchmarkCase]:
        raise NotImplementedError


class Sampler(ABC):
    """Selects dependency groups and produces an auditable manifest."""

    @abstractmethod
    def select(self, cases: Sequence[BenchmarkCase]) -> tuple[Sequence[BenchmarkCase], SampleManifest]:
        raise NotImplementedError


class ModelBackend(ABC):
    """Executes a model request; it knows nothing about benchmark gold data."""

    name: str

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError


class BenchmarkAdapter(ABC):
    """Owns task-visible prompts and response parsing, but not data acquisition."""

    benchmark_id: str
    prompt_revision: str

    @abstractmethod
    def validate_case(self, case: BenchmarkCase) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, case: BenchmarkCase, response: ModelResponse) -> Any:
        raise NotImplementedError


class InteractiveEnvironment(ABC):
    """Controls visible state, turns, tools, and termination for an episode."""

    environment_revision: str

    @abstractmethod
    def reset(self, case: BenchmarkCase, *, seed: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def observation(self, state: Any, *, actor: str) -> Sequence[ChatMessage]:
        raise NotImplementedError

    @abstractmethod
    def apply(self, state: Any, *, actor: str, action: Any) -> EnvironmentTransition:
        raise NotImplementedError


class Scorer(ABC):
    """Maps a prediction or episode trace to benchmark-native metrics."""

    scorer_revision: str

    @abstractmethod
    def score(self, case: BenchmarkCase, prediction: Any, *, trace: Sequence[Any] = ()) -> Sequence[MetricValue]:
        raise NotImplementedError


class Aggregator(ABC):
    """Aggregates only within a benchmark's declared metric semantics."""

    @abstractmethod
    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        raise NotImplementedError


class ArtifactSink(ABC):
    """Persists manifests and structured checkpoints."""

    @abstractmethod
    def initialize(self, manifest: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_result(self, result: CaseResult) -> None:
        raise NotImplementedError


class Reporter(ABC):
    """Renders already-computed benchmark metrics without redefining them."""

    @abstractmethod
    def render(self, manifest: Any, metrics: Mapping[str, Any], results: Sequence[CaseResult]) -> str:
        raise NotImplementedError

