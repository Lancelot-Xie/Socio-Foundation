"""Strict, auditable primitives shared by static choice benchmarks.

The helpers in this module deliberately stop short of defining a universal
score.  They standardize only option identity, deterministic presentation,
response parsing, and one-request failure accounting.  Each benchmark keeps
its own metric and aggregation semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..contracts import (
    BenchmarkCase,
    CaseResult,
    ErrorState,
    MetricValue,
    ModelRequest,
    ModelResponse,
    ResultStatus,
    TraceEvent,
)
from ..errors import BackendError, ConfigurationError, EpisodeTokenBudgetExhausted, ParseError, ValidationError
from ..interfaces import BenchmarkAdapter, ModelBackend
from .common import combine_usage


_DISPLAY_ID = re.compile(r"^[A-Z][A-Z0-9_]{0,15}$")


@dataclass(frozen=True)
class DisplayedChoice:
    """One public option and its source-stable identity."""

    display_id: str
    source_id: str
    text: str
    source_index: int

    def __post_init__(self) -> None:
        if not _DISPLAY_ID.fullmatch(self.display_id):
            raise ValidationError(f"invalid display choice ID {self.display_id!r}")
        if not self.source_id:
            raise ValidationError("choice source_id cannot be empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValidationError("choice text cannot be empty")
        if self.source_index < 0:
            raise ValidationError("choice source_index cannot be negative")

    def public_dict(self) -> dict[str, str]:
        return {"id": self.display_id, "text": self.text}

    def audit_dict(self) -> dict[str, Any]:
        return {
            "display_id": self.display_id,
            "source_id": self.source_id,
            "source_index": self.source_index,
        }


@dataclass(frozen=True)
class ChoicePrediction:
    display_id: str
    source_id: str
    source_index: int
    ranking_source_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_id": self.display_id,
            "source_id": self.source_id,
            "source_index": self.source_index,
            "ranking_source_ids": list(self.ranking_source_ids),
        }


def alphabetical_labels(count: int) -> tuple[str, ...]:
    if count <= 0 or count > 26:
        raise ValidationError("alphabetical choice labels support 1 through 26 options")
    return tuple(chr(ord("A") + index) for index in range(count))


def numbered_labels(count: int, *, prefix: str = "C") -> tuple[str, ...]:
    if count <= 0 or count > 999:
        raise ValidationError("numbered choice labels support 1 through 999 options")
    width = max(2, len(str(count)))
    return tuple(f"{prefix}{index + 1:0{width}d}" for index in range(count))


def strip_declared_option_prefix(text: str, label: str) -> str:
    """Remove only the matching source label, never an arbitrary prefix."""

    pattern = re.compile(rf"^\s*{re.escape(label)}\s*[.)\]:-]\s*", re.IGNORECASE)
    stripped = pattern.sub("", text, count=1).strip()
    return stripped or text.strip()


def indexed_choices(
    values: Sequence[Any],
    *,
    labels: Sequence[str],
    source_ids: Sequence[str] | None = None,
    strip_matching_prefix: bool = False,
) -> tuple[DisplayedChoice, ...]:
    if isinstance(values, (str, bytes)) or len(values) != len(labels):
        raise ValidationError("choice values and labels must be equally sized arrays")
    resolved_source_ids = tuple(source_ids or (str(index) for index in range(len(values))))
    if len(resolved_source_ids) != len(values) or len(set(resolved_source_ids)) != len(values):
        raise ValidationError("choice source IDs must be unique and aligned with values")
    choices = []
    for index, (value, label, source_id) in enumerate(zip(values, labels, resolved_source_ids)):
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"choice #{index} must be a nonempty string")
        text = strip_declared_option_prefix(value, label) if strip_matching_prefix else value.strip()
        choices.append(DisplayedChoice(str(label), str(source_id), text, index))
    return tuple(choices)


def deterministic_binary_assignment(
    case_id: str,
    *,
    seed: int,
    left_source_id: str,
    left_text: str,
    right_source_id: str,
    right_text: str,
) -> tuple[DisplayedChoice, DisplayedChoice]:
    """Assign a pair to A/B from a seed and case ID without consulting gold."""

    if left_source_id == right_source_id or left_text == right_text:
        raise ValidationError("binary choices require distinct source IDs and texts")
    material = json.dumps(
        {
            "revision": "balanced-hash-side-v1",
            "case_id": case_id,
            "seed": seed,
            "source_ids": sorted((left_source_id, right_source_id)),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    left_first = hashlib.sha256(material).digest()[0] % 2 == 0
    ordered = (
        ((left_source_id, left_text, 0), (right_source_id, right_text, 1))
        if left_first
        else ((right_source_id, right_text, 1), (left_source_id, left_text, 0))
    )
    return tuple(
        DisplayedChoice(label, source_id, text.strip(), source_index)
        for label, (source_id, text, source_index) in zip(("A", "B"), ordered)
    )  # type: ignore[return-value]


def _resolve_exact_token(
    token: Any,
    choices: Sequence[DisplayedChoice],
    *,
    allow_exact_text: bool,
) -> DisplayedChoice:
    if not isinstance(token, str) or not token.strip():
        raise ParseError("choice must be a nonempty string")
    normalized = token.strip().casefold()
    by_label = [choice for choice in choices if choice.display_id.casefold() == normalized]
    if len(by_label) == 1:
        return by_label[0]
    if allow_exact_text:
        by_text = [choice for choice in choices if choice.text.strip().casefold() == normalized]
        if len(by_text) == 1:
            return by_text[0]
        if len(by_text) > 1:
            raise ParseError("choice text is ambiguous because multiple candidates share it")
    raise ParseError(f"choice {token!r} is not one exact displayed option ID")


def parse_strict_choice(
    text: str,
    choices: Sequence[DisplayedChoice],
    *,
    allow_exact_text: bool = False,
    allow_ranking: bool = False,
    allow_answer_tag: bool = False,
) -> ChoicePrediction:
    """Parse an exact label, official ``<answer>`` tag, or small JSON object.

    JSON remains accepted for backward-compatible fixture and artifact replay,
    while benchmark adapters can opt into their released ``<answer>``
    protocol without asking the model for a JSON transport envelope.
    """

    raw = text.strip()
    if not raw:
        raise ParseError("choice response is empty")
    choice_token: Any = raw
    ranking_tokens: Sequence[Any] = ()
    if allow_answer_tag and "<answer" in raw.casefold():
        matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
        if len(matches) != 1:
            raise ParseError("choice response requires exactly one complete <answer> tag")
        answer = matches[0].strip()
        if not answer:
            raise ParseError("choice <answer> content is empty")
        if allow_ranking and "," in answer:
            ranking_tokens = tuple(part.strip() for part in answer.split(","))
            if any(not token for token in ranking_tokens):
                raise ParseError("ranked <answer> contains an empty option ID")
            choice_token = ranking_tokens[0]
        else:
            choice_token = answer
    elif raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ParseError("choice response starts with JSON but is malformed") from exc
        if not isinstance(payload, Mapping):
            raise ParseError("choice JSON must be an object")
        allowed_keys = {"choice", "ranking"} if allow_ranking else {"choice"}
        if set(payload) - allowed_keys or "choice" not in payload:
            raise ParseError(f"choice JSON requires exactly allowed keys {sorted(allowed_keys)}")
        choice_token = payload["choice"]
        if "ranking" in payload:
            ranking_tokens = payload["ranking"]
            if isinstance(ranking_tokens, (str, bytes)) or not isinstance(ranking_tokens, Sequence):
                raise ParseError("ranking must be an array of exact displayed option IDs")
    selected = _resolve_exact_token(choice_token, choices, allow_exact_text=allow_exact_text)
    ranking: list[DisplayedChoice] = []
    if ranking_tokens:
        ranking = [
            _resolve_exact_token(token, choices, allow_exact_text=False)
            for token in ranking_tokens
        ]
        ranking_ids = [item.source_id for item in ranking]
        if len(ranking_ids) != len(set(ranking_ids)):
            raise ParseError("ranking contains a duplicate option")
        if ranking[0].source_id != selected.source_id:
            raise ParseError("ranking must begin with the selected choice")
    return ChoicePrediction(
        display_id=selected.display_id,
        source_id=selected.source_id,
        source_index=selected.source_index,
        ranking_source_ids=tuple(item.source_id for item in ranking),
    )


def exact_binary_accuracy(
    results: Sequence[CaseResult],
    metric_name: str,
) -> tuple[float | None, int, int]:
    """Treat target-model failures as incorrect instead of dropping them."""

    if not results:
        return None, 0, 0
    correct = 0
    for result in results:
        value = next((metric.value for metric in result.metrics if metric.name == metric_name), None)
        correct += int(value == 1 or value is True)
    return correct / len(results), correct, len(results)


def unavailable_complete_metric(
    results: Sequence[CaseResult],
    metric_name: str,
) -> tuple[float | None, int, int, int]:
    """Average an evaluator-backed metric only when every case is available."""

    values: list[float] = []
    unavailable = 0
    for result in results:
        metric = next((item for item in result.metrics if item.name == metric_name), None)
        if result.status != ResultStatus.COMPLETED or metric is None or metric.value is None:
            unavailable += 1
            continue
        values.append(float(metric.value))
    if not results or unavailable:
        return None, len(values), len(results), unavailable
    return sum(values) / len(values), len(values), len(results), 0


def parse_failure_rate(results: Sequence[CaseResult]) -> MetricValue:
    failed = sum(
        (
            isinstance(result.metadata.get("target_output_failure"), Mapping)
            and result.metadata["target_output_failure"].get("stage") == "response_parse"
        )
        or (
            # Backward compatibility for records written before target format
            # violations became terminal, scoreable capability outcomes.
            result.status == ResultStatus.FAILED
            and result.error is not None
            and result.error.stage == "response_parse"
        )
        for result in results
    )
    return MetricValue(
        name="choice.parse_failure_rate",
        value=failed / len(results) if results else None,
        direction="lower_is_better",
        unit="proportion",
        numerator=failed,
        denominator=len(results),
    )


def safe_metric_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return token or "unspecified"


def finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ParseError(f"{field} must be finite")
    return result


class StaticChoiceAdapter(BenchmarkAdapter):
    """Execution shell for one target-model choice request."""

    metric_name: str

    def choices_for_case(self, case: BenchmarkCase, *, seed: int) -> tuple[DisplayedChoice, ...]:
        raise NotImplementedError

    def gold_source_id(self, case: BenchmarkCase) -> str:
        raise NotImplementedError

    def parse_choice_response(
        self,
        case: BenchmarkCase,
        response: ModelResponse,
        choices: Sequence[DisplayedChoice],
    ) -> ChoicePrediction:
        return parse_strict_choice(response.text, choices)

    def parse_response(self, case: BenchmarkCase, response: ModelResponse) -> ChoicePrediction:
        """Compatibility entrypoint; seeded execution uses parse_choice_response directly."""

        return self.parse_choice_response(case, response, self.choices_for_case(case, seed=0))

    def score_response(
        self,
        case: BenchmarkCase,
        prediction: ChoicePrediction,
        response: ModelResponse,
        *,
        seed: int,
    ) -> Sequence[MetricValue]:
        correct = int(prediction.source_id == self.gold_source_id(case))
        return (
            MetricValue(
                name=self.metric_name,
                value=correct,
                unit="proportion",
                numerator=correct,
                denominator=1,
                metadata={"scoring": "exact_source_choice"},
            ),
        )

    def score_parse_failure(
        self,
        case: BenchmarkCase,
        response: ModelResponse,
        *,
        seed: int,
        exc: ParseError,
    ) -> Sequence[MetricValue]:
        """Score a target format violation as an incorrect, terminal item."""

        del case, response, seed
        return (
            MetricValue(
                name=self.metric_name,
                value=0,
                unit="proportion",
                numerator=0,
                denominator=1,
                metadata={
                    "scoring": "target_output_parse_failure_is_incorrect",
                    "parse_error": str(exc),
                },
            ),
        )

    def result_metadata(
        self,
        case: BenchmarkCase,
        choices: Sequence[DisplayedChoice],
        *,
        seed: int,
    ) -> Mapping[str, Any]:
        return {
            "choice_contract": {
                "parser_revision": "strict-exact-choice-json-v1",
                "assignment": [choice.audit_dict() for choice in choices],
                "seed": seed,
            }
        }

    def replay_responses(self, case: BenchmarkCase, *, seed: int) -> Mapping[str, Any]:
        replay = case.metadata.get("replay")
        if not isinstance(replay, Mapping) or "response" not in replay:
            raise ConfigurationError(f"fixture {case.case_id} requires metadata.replay.response")
        return {
            f"{case.case_id}:choice": replay["response"],
        }

    def _failure(
        self,
        case: BenchmarkCase,
        *,
        run_id: str,
        repetition: int,
        stage: str,
        exc: Exception,
        response: ModelResponse | None,
        choices: Sequence[DisplayedChoice],
        seed: int,
    ) -> CaseResult:
        if isinstance(exc, ParseError):
            assert response is not None
            failure = {
                "stage": stage,
                "kind": "parse_failure",
                "message": str(exc),
                "retryable": False,
                "scoring_policy": "target_capability_failure_scores_zero",
            }
            return CaseResult(
                run_id=run_id,
                benchmark_id=self.benchmark_id,
                case_id=case.case_id,
                group_id=case.group_id,
                repetition=repetition,
                status=ResultStatus.COMPLETED,
                prediction={"raw_response": response.text, "parsed": False},
                metrics=tuple(
                    self.score_parse_failure(
                        case,
                        response,
                        seed=seed,
                        exc=exc,
                    )
                ),
                trace=(
                    TraceEvent(
                        turn=0,
                        actor="evaluated_model",
                        kind="target_output_parse_failure",
                        content={"raw_response": response.text, **failure},
                        visible_to=("evaluator",),
                    ),
                ),
                model_response=response,
                latency_ms=response.latency_ms,
                token_usage=response.usage,
                metadata={
                    **dict(self.result_metadata(case, choices, seed=seed)),
                    "evaluation_complete": True,
                    "target_output_failure": failure,
                },
            )
        kind = (
            "token_budget_exhausted" if isinstance(exc, EpisodeTokenBudgetExhausted)
            else "backend_failure" if isinstance(exc, BackendError) else "parse_failure"
        )
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.FAILED,
            prediction={"raw_response": response.text if response else None},
            trace=(
                TraceEvent(
                    turn=0,
                    actor="evaluated_model",
                    kind="response_failure",
                    content=response.text if response else str(exc),
                    visible_to=("evaluator",),
                ),
            ),
            model_response=response,
            error=ErrorState(
                stage=stage,
                kind=kind,
                message=str(exc),
                retryable=isinstance(exc, BackendError),
            ),
            latency_ms=response.latency_ms if response else None,
            token_usage=response.usage if response else None,
            metadata={
                **dict(self.result_metadata(case, choices, seed=seed)),
                "evaluation_complete": False,
                **({"token_budget": exc.details()} if isinstance(exc, EpisodeTokenBudgetExhausted) else {}),
            },
        )

    def execute_case(
        self,
        case: BenchmarkCase,
        *,
        backend: ModelBackend,
        run_id: str,
        seed: int,
        model: str,
        repetition: int = 0,
    ) -> CaseResult:
        self.validate_case(case)
        choices = self.choices_for_case(case, seed=seed)
        request = self.build_request(case, model=model, seed=seed)
        try:
            response = backend.generate(request)
        except (BackendError, EpisodeTokenBudgetExhausted) as exc:
            return self._failure(
                case,
                run_id=run_id,
                repetition=repetition,
                stage="token_budget" if isinstance(exc, EpisodeTokenBudgetExhausted) else "model_backend",
                exc=exc,
                response=None,
                choices=choices,
                seed=seed,
            )
        try:
            prediction = self.parse_choice_response(case, response, choices)
        except ParseError as exc:
            return self._failure(
                case,
                run_id=run_id,
                repetition=repetition,
                stage="response_parse",
                exc=exc,
                response=response,
                choices=choices,
                seed=seed,
            )
        metrics = tuple(self.score_response(case, prediction, response, seed=seed))
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction=prediction.to_dict(),
            metrics=metrics,
            trace=(
                TraceEvent(
                    turn=0,
                    actor="evaluated_model",
                    kind="choice",
                    content=prediction.to_dict(),
                    visible_to=("evaluator",),
                ),
            ),
            model_response=response,
            latency_ms=response.latency_ms,
            token_usage=combine_usage((response.usage,)),
            metadata={
                **dict(self.result_metadata(case, choices, seed=seed)),
                "evaluation_complete": True,
            },
        )


__all__ = [
    "ChoicePrediction",
    "DisplayedChoice",
    "StaticChoiceAdapter",
    "alphabetical_labels",
    "deterministic_binary_assignment",
    "exact_binary_accuracy",
    "finite_float",
    "indexed_choices",
    "numbered_labels",
    "parse_failure_rate",
    "parse_strict_choice",
    "safe_metric_token",
    "strip_declared_option_prefix",
    "unavailable_complete_metric",
]
