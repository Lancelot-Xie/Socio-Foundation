"""FANToM theory-of-mind question and official set aggregation adapter."""

from __future__ import annotations

from ..presentation import material_user_message

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

from ..contracts import BenchmarkCase, CaseResult, ChatMessage, MetricValue, ModelRequest, ModelResponse, ResultStatus
from ..data.schemas import probe_case
from ..errors import ConfigurationError, ParseError, ValidationError
from ..json_utils import canonical_json
from ..registry import adapter
from ..model_adaptation import adapted_choice_text
from .choice import (
    DisplayedChoice,
    StaticChoiceAdapter,
    deterministic_binary_assignment,
    finite_float,
    parse_failure_rate,
    parse_strict_choice,
)
from .common import aggregate_named_metrics


FANTOM_CONTEXT_CONDITIONS = ("short", "full")
FANTOM_SCENARIOS = ("inaccessible", "accessible", "fact")
FANTOM_FORMATS = {
    "fact": ("free_text",),
    "belief": ("free_text", "multiple_choice"),
    "answerability": ("list", "binary"),
    "information_access": ("list", "binary"),
}
FANTOM_SEMANTIC_MODEL = "sentence-transformers/all-roberta-large-v1"
FANTOM_SEMANTIC_CONTRACT = "fantom-belief-distance-evidence-v1"
FANTOM_SET_CONTRACT = "fantom-set-members-v1"


@dataclass(frozen=True)
class FantomPrediction:
    kind: str
    value: Any
    source_id: str | None = None
    display_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "source_id": self.source_id,
            "display_id": self.display_id,
        }


def whitespace_token_f1(reference: str, prediction: str) -> float:
    """Mirror FANToM's lowercase whitespace-token Counter F1."""

    reference_tokens = str(reference).lower().split()
    prediction_tokens = str(prediction).lower().split()
    if not reference_tokens or not prediction_tokens:
        return 0.0
    overlap = Counter(reference_tokens) & Counter(prediction_tokens)
    shared = sum(overlap.values())
    if not shared:
        return 0.0
    precision = shared / len(prediction_tokens)
    recall = shared / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def belief_whitespace_token_f1(reference: str, prediction: str) -> float:
    """Mirror upstream belief overlap, which is whitespace-token case-sensitive."""

    reference_tokens = str(reference).split()
    prediction_tokens = str(prediction).split()
    if not reference_tokens or not prediction_tokens:
        return 0.0
    overlap = Counter(reference_tokens) & Counter(prediction_tokens)
    shared = sum(overlap.values())
    if not shared:
        return 0.0
    precision = shared / len(prediction_tokens)
    recall = shared / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def weighted_binary_f1(references: Sequence[int], predictions: Sequence[int]) -> float | None:
    """Dependency-free equivalent of weighted F1 over the two gold labels."""

    if len(references) != len(predictions):
        raise ValidationError("FANToM binary references and predictions must align")
    if not references:
        return None
    total = len(references)
    score = 0.0
    for label in (0, 1):
        support = sum(reference == label for reference in references)
        if not support:
            continue
        true_positive = sum(reference == label and prediction == label for reference, prediction in zip(references, predictions))
        false_positive = sum(reference != label and prediction == label for reference, prediction in zip(references, predictions))
        false_negative = sum(reference == label and prediction != label for reference, prediction in zip(references, predictions))
        denominator = 2 * true_positive + false_positive + false_negative
        label_f1 = 2 * true_positive / denominator if denominator else 0.0
        score += support / total * label_f1
    return score


def _parse_json_object(text: str, *, required_key: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        raise ParseError("FANToM response is empty")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError("FANToM structured response is not valid JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {required_key}:
        raise ParseError(f"FANToM response must contain exactly the key {required_key!r}")
    return payload[required_key]


def _official_answer_content(text: str) -> str:
    """Extract Supplemental's first complete answer tag, retaining legacy raw replay."""

    raw = str(text or "").strip()
    if not raw:
        raise ParseError("FANToM response is empty")
    match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
    if match is not None:
        content = match.group(1).strip()
        if not content:
            raise ParseError("FANToM <answer> content is empty")
        return content
    return raw


def _binary_label(value: Any, *, allow_long: bool) -> str:
    if not isinstance(value, str):
        raise ValidationError("FANToM binary answer must be text")
    normalized = value.strip().casefold()
    if allow_long and normalized == "no:long":
        return "no"
    if normalized in {"yes", "true"}:
        return "yes"
    if normalized in {"no", "false"}:
        return "no"
    raise ValidationError("FANToM binary answer must be yes/no/true/false or source-only no:long")


def _metric_value(result: CaseResult, name: str) -> float | None:
    metric = next((item for item in result.metrics if item.name == name), None)
    if metric is None or metric.value is None:
        return None
    return float(metric.value)


@adapter("fantom")
class FantomAdapter(StaticChoiceAdapter):
    benchmark_id = "fantom"
    prompt_revision = "fantom-supplemental-answer-tag-set-contract-v3-natural-material-v1"
    scorer_revision = "fantom-v1-official-components-member-closed-set-batched-semantic-v4"
    metric_name = "fantom.item_correct"

    def __init__(
        self,
        *,
        belief_semantic_scorer: Callable[[str, str, str], Mapping[str, Any]] | None = None,
        defer_belief_semantic: bool = False,
    ) -> None:
        self._belief_semantic_scorer = belief_semantic_scorer
        self._defer_belief_semantic = defer_belief_semantic

    @property
    def belief_semantic_batch_size(self) -> int:
        value = getattr(self._belief_semantic_scorer, "batch_size", 1)
        return int(value)

    def belief_semantic_protocol_identity(self) -> Mapping[str, Any]:
        method = getattr(self._belief_semantic_scorer, "protocol_identity", None)
        return dict(method()) if callable(method) else {}

    @staticmethod
    def needs_belief_semantic(case: BenchmarkCase, result: CaseResult) -> bool:
        return (
            result.status == ResultStatus.COMPLETED
            and case.input_data.get("question_family") == "belief"
            and case.input_data.get("answer_format") == "free_text"
            and isinstance(result.prediction, Mapping)
            and isinstance(result.prediction.get("value"), str)
            and bool(str(result.prediction["value"]).strip())
        )

    def finalize_belief_semantic_batch(
        self,
        items: Sequence[tuple[BenchmarkCase, CaseResult, int]],
    ) -> tuple[CaseResult, ...]:
        """Batch-score deferred free-text beliefs and preserve item order."""

        if not items:
            return ()
        score_batch = getattr(self._belief_semantic_scorer, "score_batch", None)
        if not callable(score_batch):
            raise ConfigurationError(
                "deferred FANToM semantic scoring requires a batch-capable scorer"
            )
        requests = tuple(
            (
                str(result.prediction["value"]),
                str(case.gold),
                str(case.input_data["wrong_answer"]),
            )
            for case, result, _ in items
        )
        evidence_items = tuple(score_batch(requests))
        if len(evidence_items) != len(items):
            raise ValidationError(
                "FANToM semantic scorer result count does not match deferred case count"
            )
        finalized = []
        for (case, result, seed), evidence in zip(items, evidence_items):
            if result.model_response is None:
                raise ValidationError(
                    f"completed FANToM case {case.case_id} lacks its model response"
                )
            raw_prediction = result.prediction
            assert isinstance(raw_prediction, Mapping)
            prediction = FantomPrediction(
                kind=str(raw_prediction.get("kind") or "free_text"),
                value=raw_prediction.get("value"),
                source_id=raw_prediction.get("source_id"),
                display_id=raw_prediction.get("display_id"),
            )
            response = replace(
                result.model_response,
                raw={
                    **dict(result.model_response.raw),
                    "fantom_belief_distance": dict(evidence),
                },
            )
            metrics = tuple(self.score_response(case, prediction, response, seed=seed))
            metadata = {
                **dict(result.metadata),
                "belief_semantic": dict(evidence),
            }
            finalized.append(
                replace(
                    result,
                    metrics=metrics,
                    model_response=response,
                    metadata=metadata,
                )
            )
        return tuple(finalized)

    @staticmethod
    def _conversation_text(case: BenchmarkCase) -> str:
        conversation = case.input_data.get("conversation")
        if isinstance(conversation, str) and conversation.strip():
            return conversation.strip()
        if isinstance(conversation, Sequence) and not isinstance(conversation, (str, bytes)):
            if conversation and all(isinstance(item, str) and item.strip() for item in conversation):
                return "\n".join(item.strip() for item in conversation)
        raise ValidationError("FANToM conversation must be nonempty text or an array of nonempty turns")

    @staticmethod
    def _list_answers(case: BenchmarkCase) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
        correct = case.gold
        incorrect = case.input_data.get("incorrect_characters")
        if isinstance(correct, (str, bytes)) or not isinstance(correct, Sequence):
            raise ValidationError("FANToM list gold must be an array of character names")
        if isinstance(incorrect, (str, bytes)) or not isinstance(incorrect, Sequence):
            raise ValidationError("FANToM list questions require incorrect_characters")
        if any(not isinstance(item, str) or not item.strip() for item in (*correct, *incorrect)):
            raise ValidationError("FANToM character names must be nonempty strings")
        correct_names = tuple(str(item).strip() for item in correct)
        incorrect_names = tuple(str(item).strip() for item in incorrect)
        folded = [item.casefold() for item in (*correct_names, *incorrect_names)]
        if len(folded) != len(set(folded)):
            raise ValidationError("FANToM correct and incorrect character lists must be unique and disjoint")
        if not folded:
            raise ValidationError("FANToM list question requires at least one candidate character")
        canonical = {item.casefold(): item for item in (*correct_names, *incorrect_names)}
        return correct_names, incorrect_names, canonical

    def choices_for_case(self, case: BenchmarkCase, *, seed: int) -> tuple[DisplayedChoice, ...]:
        values = case.input_data
        if values.get("question_family") != "belief" or values.get("answer_format") != "multiple_choice":
            return ()
        correct = case.gold
        wrong = values.get("wrong_answer")
        if not isinstance(correct, str) or not correct.strip() or not isinstance(wrong, str) or not wrong.strip():
            raise ValidationError("FANToM belief choice requires nonempty correct and wrong answers")
        return deterministic_binary_assignment(
            case.case_id,
            seed=seed,
            left_source_id="correct",
            left_text=correct.strip(),
            right_source_id="omniscient_or_wrong",
            right_text=wrong.strip(),
        )

    def gold_source_id(self, case: BenchmarkCase) -> str:
        if case.input_data.get("question_family") != "belief" or case.input_data.get("answer_format") != "multiple_choice":
            raise ValidationError("only FANToM belief-choice cases expose a source choice ID")
        return "correct"

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"FantomAdapter cannot run {case.benchmark_id!r}")
        probe_case(case)
        values = case.input_data
        self._conversation_text(case)
        if not isinstance(values.get("question"), str) or not str(values["question"]).strip():
            raise ValidationError("FANToM question must be nonempty text")
        family = values.get("question_family")
        answer_format = values.get("answer_format")
        if family not in FANTOM_FORMATS or answer_format not in FANTOM_FORMATS.get(str(family), ()):
            raise ValidationError("FANToM question_family/answer_format pairing is invalid")
        if values.get("context_condition") not in FANTOM_CONTEXT_CONDITIONS:
            raise ValidationError(f"FANToM context_condition must be one of {FANTOM_CONTEXT_CONDITIONS}")
        scenario = values.get("scenario")
        if scenario not in FANTOM_SCENARIOS:
            raise ValidationError(f"FANToM scenario must be one of {FANTOM_SCENARIOS}")
        if family == "fact" and scenario != "fact":
            raise ValidationError("FANToM fact questions must use scenario=fact")
        if family != "fact" and scenario == "fact":
            raise ValidationError("FANToM ToM questions must be accessible or inaccessible")
        for field in ("conversation_id", "part_id", "set_id"):
            if isinstance(values.get(field), bool) or not isinstance(values.get(field), (str, int)) or not str(values[field]).strip():
                raise ValidationError(f"FANToM {field} must be a stable nonempty identifier")
        if str(values["conversation_id"]) not in str(values["part_id"]):
            raise ValidationError("FANToM part_id must retain the conversation_id")
        if str(values["part_id"]) not in str(values["set_id"]):
            raise ValidationError("FANToM set_id must retain the part_id")
        member_id = values.get("question_member_id")
        contract = values.get("set_contract")
        if (member_id is None) != (contract is None):
            raise ValidationError(
                "FANToM question_member_id and set_contract must either both be present or both be absent"
            )
        if contract is not None:
            if not isinstance(member_id, str) or not member_id.strip():
                raise ValidationError("FANToM question_member_id must be nonempty text")
            if not isinstance(contract, Mapping):
                raise ValidationError("FANToM set_contract must be an object")
            if contract.get("revision") != FANTOM_SET_CONTRACT:
                raise ValidationError("FANToM set_contract revision mismatch")
            if str(contract.get("set_id")) != str(values["set_id"]):
                raise ValidationError("FANToM set_contract set_id mismatch")
            if contract.get("context_condition") != values["context_condition"]:
                raise ValidationError("FANToM set_contract context condition mismatch")
            if not isinstance(contract.get("fully_accessible"), bool):
                raise ValidationError("FANToM set_contract fully_accessible must be boolean")
            members = contract.get("members")
            if isinstance(members, (str, bytes)) or not isinstance(members, Sequence) or not members:
                raise ValidationError("FANToM set_contract members must be a nonempty array")
            normalized_members: dict[str, Mapping[str, Any]] = {}
            for member in members:
                if not isinstance(member, Mapping):
                    raise ValidationError("FANToM set_contract members must be objects")
                expected_fields = {
                    "id",
                    "question_family",
                    "answer_format",
                    "scenario",
                    "short_no_long_excluded",
                }
                if not expected_fields <= set(member):
                    raise ValidationError("FANToM set_contract member is incomplete")
                identifier = member.get("id")
                member_family = member.get("question_family")
                member_format = member.get("answer_format")
                member_scenario = member.get("scenario")
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValidationError("FANToM set_contract member ID must be nonempty text")
                if identifier in normalized_members:
                    raise ValidationError("FANToM set_contract member IDs must be unique")
                if (
                    member_family not in FANTOM_FORMATS
                    or member_format not in FANTOM_FORMATS[str(member_family)]
                ):
                    raise ValidationError("FANToM set_contract member family/format is invalid")
                if member_scenario not in FANTOM_SCENARIOS:
                    raise ValidationError("FANToM set_contract member scenario is invalid")
                if (member_family == "fact") != (member_scenario == "fact"):
                    raise ValidationError("FANToM set_contract fact scenario is inconsistent")
                excluded = member.get("short_no_long_excluded")
                if not isinstance(excluded, bool):
                    raise ValidationError("FANToM set_contract exclusion flag must be boolean")
                if excluded and not (
                    values["context_condition"] == "short" and member_format == "binary"
                ):
                    raise ValidationError("FANToM set_contract has an invalid no:long exclusion")
                normalized_members[identifier] = member
            current = normalized_members.get(member_id)
            if current is None:
                raise ValidationError("FANToM question_member_id is absent from set_contract")
            if (
                current["question_family"] != family
                or current["answer_format"] != answer_format
                or current["scenario"] != scenario
            ):
                raise ValidationError("FANToM current question does not match its set_contract member")
            active_tom = [
                member
                for member in normalized_members.values()
                if member["question_family"] != "fact"
                and not member["short_no_long_excluded"]
            ]
            fully_accessible = bool(active_tom) and all(
                member["scenario"] == "accessible" for member in active_tom
            )
            if contract["fully_accessible"] != fully_accessible:
                raise ValidationError("FANToM set_contract fully_accessible value is inconsistent")
        if family in {"fact", "belief"}:
            if not isinstance(case.gold, str) or not case.gold.strip():
                raise ValidationError("FANToM free/choice answer must be nonempty text")
        if family == "belief":
            wrong = values.get("wrong_answer")
            if not isinstance(wrong, str) or not wrong.strip() or wrong.strip() == str(case.gold).strip():
                raise ValidationError("FANToM belief cases require a distinct nonempty wrong_answer")
            if answer_format == "multiple_choice":
                self.choices_for_case(case, seed=0)
        elif family in {"answerability", "information_access"}:
            target = values.get("target_fact")
            if not isinstance(target, str) or not target.strip():
                raise ValidationError(
                    "FANToM answerability/information-access cases require a nonempty target_fact"
                )
            if answer_format == "list":
                self._list_answers(case)
            else:
                _binary_label(case.gold, allow_long=True)

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        self.validate_case(case)
        values = case.input_data
        family = str(values["question_family"])
        answer_format = str(values["answer_format"])
        payload: dict[str, Any] = {
            "conversation": self._conversation_text(case),
            "question": values["question"],
        }
        if values.get("target_fact"):
            payload["target"] = values["target_fact"]
        instruction: str
        if family == "belief" and answer_format == "multiple_choice":
            choices = self.choices_for_case(case, seed=seed)
            payload["options"] = [choice.public_dict() for choice in choices]
            instruction = "Put the final option letter inside <answer></answer>, for example <answer>A</answer>."
        elif answer_format == "list":
            instruction = "Put the final list of character names inside <answer></answer>."
        elif answer_format == "binary":
            instruction = "Put only yes or no inside <answer></answer>."
        else:
            instruction = "Put the concise free-text answer inside <answer></answer>."
        return ModelRequest(
            request_id=f"{case.case_id}:question",
            messages=(
                ChatMessage(
                    "system",
                    "This is a theory-of-mind test. Please answer the question regarding facts or beliefs, based on the following in-person conversation between individuals who have just met.",
                ),
                material_user_message(payload, {'conversation': 'Context', 'question': 'Question', 'target': 'Target Fact', 'options': 'Options'}, truncatable=('conversation',), suffix="\n" + instruction),
            ),
            model=model,
            temperature=0.0,
            max_tokens=1024,
            seed=seed,
            metadata={
                "benchmark_id": self.benchmark_id,
                "prompt_revision": self.prompt_revision,
                "question_family": family,
                "answer_format": answer_format,
                "context_condition": values["context_condition"],
                "gold_visible": False,
            },
        )

    def parse_choice_response(
        self,
        case: BenchmarkCase,
        response: ModelResponse,
        choices: Sequence[DisplayedChoice],
    ) -> FantomPrediction:
        family = str(case.input_data["question_family"])
        answer_format = str(case.input_data["answer_format"])
        if family == "belief" and answer_format == "multiple_choice":
            parsed = parse_strict_choice(adapted_choice_text(response, choices), choices, allow_answer_tag=True)
            return FantomPrediction("multiple_choice", parsed.display_id, parsed.source_id, parsed.display_id)
        if answer_format == "binary":
            raw = _official_answer_content(response.text)
            value = _parse_json_object(raw, required_key="answer") if raw.startswith("{") else raw
            try:
                label = _binary_label(value, allow_long=False)
            except ValidationError as exc:
                raise ParseError(str(exc)) from exc
            return FantomPrediction("binary", label)
        if answer_format == "list":
            content = _official_answer_content(response.text)
            value: Any = None
            if content.startswith("{"):
                value = _parse_json_object(content, required_key="characters")
                if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                    raise ParseError("FANToM characters must be an array")
            _, _, canonical = self._list_answers(case)
            resolved: list[str] = []
            if value is not None:
                for raw_name in value:
                    if not isinstance(raw_name, str) or not raw_name.strip():
                        raise ParseError("FANToM character predictions must be nonempty strings")
                    name = canonical.get(raw_name.strip().casefold())
                    if name is None:
                        raise ParseError(f"FANToM response names unknown character {raw_name!r}")
                    resolved.append(name)
            else:
                for folded_name, name in canonical.items():
                    if re.search(
                        r"(?<!\w)" + re.escape(folded_name) + r"(?!\w)",
                        content.casefold(),
                    ):
                        resolved.append(name)
            folded = [item.casefold() for item in resolved]
            if len(folded) != len(set(folded)):
                raise ParseError("FANToM character response contains a duplicate")
            return FantomPrediction("list", tuple(resolved))
        value = _official_answer_content(response.text)
        return FantomPrediction("free_text", value)

    @staticmethod
    def _semantic_evidence(response: ModelResponse) -> tuple[int | None, Mapping[str, Any]]:
        raw = response.raw.get("fantom_belief_distance")
        base = {
            "contract_revision": FANTOM_SEMANTIC_CONTRACT,
            "model_id": FANTOM_SEMANTIC_MODEL,
            "metric": "cosine_similarity",
            "official_dependency": True,
        }
        if not isinstance(raw, Mapping):
            return None, {**base, "availability": "unavailable", "reason": "missing backend semantic evidence"}
        try:
            if raw.get("contract_revision") != FANTOM_SEMANTIC_CONTRACT:
                raise ParseError("belief-distance contract revision mismatch")
            if raw.get("model_id") != FANTOM_SEMANTIC_MODEL:
                raise ParseError("belief-distance model ID mismatch")
            if raw.get("metric") != "cosine_similarity":
                raise ParseError("belief-distance metric mismatch")
            revision = raw.get("model_revision")
            if not isinstance(revision, str) or not revision.strip():
                raise ParseError("belief-distance evidence requires a model revision")
            correct_similarity = finite_float(raw.get("correct_similarity"), field="correct_similarity")
            wrong_similarity = finite_float(raw.get("wrong_similarity"), field="wrong_similarity")
        except ParseError as exc:
            return None, {**base, "availability": "unavailable", "reason": str(exc)}
        correct = int(correct_similarity > wrong_similarity)
        batch = raw.get("batch")
        return correct, {
            **base,
            "availability": "available",
            "model_revision": revision,
            "correct_similarity": correct_similarity,
            "wrong_similarity": wrong_similarity,
            "tie_rule": "wrong_greater_or_equal_is_incorrect",
            **({"batch": dict(batch)} if isinstance(batch, Mapping) else {}),
        }

    def score_response(
        self,
        case: BenchmarkCase,
        prediction: FantomPrediction,
        response: ModelResponse,
        *,
        seed: int,
    ) -> Sequence[MetricValue]:
        del seed
        family = str(case.input_data["question_family"])
        answer_format = str(case.input_data["answer_format"])
        if family == "fact":
            value = whitespace_token_f1(str(case.gold), str(prediction.value))
            return (
                MetricValue(
                    "fantom.fact_token_f1",
                    value,
                    numerator=value,
                    denominator=1,
                    metadata={"official_component": True, "tokenizer": "lowercase_whitespace_counter"},
                ),
            )
        if family == "belief" and answer_format == "free_text":
            if (
                "fantom_belief_distance" not in response.raw
                and self._belief_semantic_scorer is not None
                and not self._defer_belief_semantic
            ):
                generated = self._belief_semantic_scorer(
                    str(prediction.value),
                    str(case.gold),
                    str(case.input_data["wrong_answer"]),
                )
                response = ModelResponse(
                    text=response.text,
                    finish_reason=response.finish_reason,
                    usage=response.usage,
                    latency_ms=response.latency_ms,
                    raw={**dict(response.raw), "fantom_belief_distance": dict(generated)},
                    response_id=response.response_id,
                )
            correct, evidence = self._semantic_evidence(response)
            token_f1 = belief_whitespace_token_f1(str(case.gold), str(prediction.value)) if correct == 1 else None
            return (
                MetricValue(
                    "fantom.item_correct",
                    correct,
                    unit="proportion",
                    numerator=correct,
                    denominator=1 if correct is not None else 0,
                    metadata={**dict(evidence), "official_component": "belief_distance_accuracy"},
                ),
                MetricValue(
                    "fantom.belief_distance_accuracy",
                    correct,
                    unit="proportion",
                    numerator=correct,
                    denominator=1 if correct is not None else 0,
                    metadata=dict(evidence),
                ),
                MetricValue(
                    "fantom.belief_token_f1",
                    token_f1,
                    numerator=token_f1,
                    denominator=1 if token_f1 is not None else 0,
                    metadata={
                        "official_component": True,
                        "condition": "distance_correct_only",
                        "tokenizer": "case_sensitive_whitespace_counter",
                    },
                ),
            )
        if family == "belief" and answer_format == "multiple_choice":
            correct = int(prediction.source_id == "correct")
            component = "fantom.belief_choice_accuracy"
        elif answer_format == "list":
            correct_names, _, _ = self._list_answers(case)
            predicted_names = tuple(str(item) for item in prediction.value)
            correct = int(
                len(predicted_names) == len(correct_names)
                and {item.casefold() for item in predicted_names} == {item.casefold() for item in correct_names}
            )
            component = f"fantom.{family}_list_accuracy"
        elif answer_format == "binary":
            expected = _binary_label(case.gold, allow_long=True)
            correct = int(prediction.value == expected)
            component = f"fantom.{family}_binary_accuracy"
        else:
            raise ValidationError("unsupported FANToM scoring branch")
        return (
            MetricValue(
                "fantom.item_correct",
                correct,
                unit="proportion",
                numerator=correct,
                denominator=1,
                metadata={"official_set_component": True},
            ),
            MetricValue(
                component,
                correct,
                unit="proportion",
                numerator=correct,
                denominator=1,
                metadata={"official_component": True},
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
        del response, seed
        family = str(case.input_data["question_family"])
        answer_format = str(case.input_data["answer_format"])
        metadata = {
            "scoring": "target_output_parse_failure_is_incorrect",
            "parse_error": str(exc),
        }
        if family == "fact":
            return (MetricValue("fantom.fact_token_f1", 0.0, numerator=0.0, denominator=1, metadata=metadata),)
        if family == "belief" and answer_format == "free_text":
            return (
                MetricValue("fantom.item_correct", 0, unit="proportion", numerator=0, denominator=1, metadata=metadata),
                MetricValue("fantom.belief_distance_accuracy", 0, unit="proportion", numerator=0, denominator=1, metadata=metadata),
                MetricValue(
                    "fantom.belief_token_f1",
                    None,
                    denominator=0,
                    metadata={**metadata, "condition": "distance_correct_only"},
                ),
            )
        component = (
            "fantom.belief_choice_accuracy"
            if family == "belief" and answer_format == "multiple_choice"
            else f"fantom.{family}_list_accuracy"
            if answer_format == "list"
            else f"fantom.{family}_binary_accuracy"
        )
        return (
            MetricValue("fantom.item_correct", 0, unit="proportion", numerator=0, denominator=1, metadata=metadata),
            MetricValue(component, 0, unit="proportion", numerator=0, denominator=1, metadata=metadata),
        )

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        del case
        return {
            "revision": "fantom-static-question-v1",
            "belief_choice_assignment": "balanced-hash-side-v1",
            "parser_revision": "supplemental-answer-tag-family-specific-with-legacy-json-compat-v2",
            "semantic_evidence_contract": FANTOM_SEMANTIC_CONTRACT,
            "semantic_model": FANTOM_SEMANTIC_MODEL,
            "official_model_revision_status": "not_pinned_by_upstream_evaluator",
            "aggregation_target": "set_id",
        }

    def result_metadata(
        self,
        case: BenchmarkCase,
        choices: Sequence[DisplayedChoice],
        *,
        seed: int,
    ) -> Mapping[str, Any]:
        metadata = dict(super().result_metadata(case, choices, seed=seed))
        values = case.input_data
        binary_reference = None
        short_excluded = False
        if values["answer_format"] == "binary":
            binary_reference = _binary_label(case.gold, allow_long=True)
            short_excluded = values["context_condition"] == "short" and str(case.gold).strip().casefold() == "no:long"
        metadata["fantom"] = {
            "conversation_id": str(values["conversation_id"]),
            "part_id": str(values["part_id"]),
            "set_id": str(values["set_id"]),
            "question_family": values["question_family"],
            "answer_format": values["answer_format"],
            "context_condition": values["context_condition"],
            "scenario": values["scenario"],
            "binary_reference": binary_reference,
            "short_no_long_excluded": short_excluded,
            "aggregation_target": "set_id",
            "gold_visible_to_model": False,
            "question_member_id": values.get("question_member_id"),
            "set_contract": values.get("set_contract"),
        }
        if choices:
            metadata["fantom"]["belief_choice_source_at_A"] = next(
                choice.source_id for choice in choices if choice.display_id == "A"
            )
        return metadata

    def replay_responses(self, case: BenchmarkCase, *, seed: int) -> Mapping[str, Any]:
        replay = case.metadata.get("replay")
        if not isinstance(replay, Mapping):
            raise ConfigurationError(f"fixture {case.case_id} requires metadata.replay")
        family = case.input_data["question_family"]
        answer_format = case.input_data["answer_format"]
        if family == "belief" and answer_format == "multiple_choice":
            choices = self.choices_for_case(case, seed=seed)
            by_source = {choice.source_id: choice.display_id for choice in choices}
            source = str(replay.get("choice_source") or "correct")
            if source not in by_source:
                raise ConfigurationError("FANToM belief-choice replay source must be correct or omniscient_or_wrong")
            response: Any = canonical_json({"choice": by_source[source]})
        elif answer_format == "list":
            characters = replay.get("characters")
            if isinstance(characters, (str, bytes)) or not isinstance(characters, Sequence):
                raise ConfigurationError("FANToM list replay requires a characters array")
            response = canonical_json({"characters": list(characters)})
        elif answer_format == "binary":
            response = canonical_json({"answer": replay.get("answer")})
        else:
            response = replay.get("response")
            if not isinstance(response, str):
                raise ConfigurationError("FANToM free-text replay requires response text")
        if family == "belief" and answer_format == "free_text" and "semantic_scores" in replay:
            scores = replay["semantic_scores"]
            if not isinstance(scores, Mapping):
                raise ConfigurationError("FANToM replay semantic_scores must be an object")
            response = {
                "text": response,
                "finish_reason": "replayed",
                "raw": {
                    "fantom_belief_distance": {
                        "contract_revision": FANTOM_SEMANTIC_CONTRACT,
                        "model_id": FANTOM_SEMANTIC_MODEL,
                        "model_revision": scores.get("model_revision") or "official-code-unpinned",
                        "metric": "cosine_similarity",
                        "correct_similarity": scores.get("correct_similarity"),
                        "wrong_similarity": scores.get("wrong_similarity"),
                    }
                },
            }
        return {f"{case.case_id}:question": response}

    @staticmethod
    def _info(result: CaseResult) -> Mapping[str, Any] | None:
        value = result.metadata.get("fantom")
        return value if isinstance(value, Mapping) else None

    @staticmethod
    def _deterministic_mean(results: Sequence[CaseResult], metric_name: str) -> tuple[float | None, float, int]:
        if not results:
            return None, 0.0, 0
        values = [
            0.0 if result.status != ResultStatus.COMPLETED else float(_metric_value(result, metric_name) or 0.0)
            for result in results
        ]
        return sum(values) / len(values), sum(values), len(values)

    @staticmethod
    def _semantic_mean(results: Sequence[CaseResult], metric_name: str) -> tuple[float | None, float | None, int, int]:
        if not results:
            return None, None, 0, 0
        values: list[float] = []
        unavailable = 0
        for result in results:
            if result.status != ResultStatus.COMPLETED:
                values.append(0.0)
                continue
            value = _metric_value(result, metric_name)
            if value is None:
                unavailable += 1
            else:
                values.append(value)
        if unavailable:
            return None, None, len(results), unavailable
        return sum(values) / len(results), sum(values), len(results), 0

    @staticmethod
    def _group_all(
        results: Sequence[CaseResult],
        required: set[tuple[str, str]],
        *,
        scenario: str,
        accessible_only: bool,
    ) -> tuple[float | None, int | None, int, int, int]:
        groups: dict[tuple[str, int], list[CaseResult]] = defaultdict(list)
        for result in results:
            info = FantomAdapter._info(result)
            if info is not None:
                groups[(str(info["set_id"]), result.repetition)].append(result)
        scores: list[int] = []
        incomplete = 0
        unavailable = 0
        for items in groups.values():
            infos = [FantomAdapter._info(item) for item in items]
            present_infos = [info for info in infos if info is not None]
            contracts = [info.get("set_contract") for info in present_infos]
            has_contract = bool(contracts) and all(
                isinstance(value, Mapping) for value in contracts
            )
            relevant: list[CaseResult]
            if has_contract:
                serialized_contracts = {canonical_json(value) for value in contracts}
                if len(serialized_contracts) != 1:
                    incomplete += 1
                    continue
                contract = contracts[0]
                if accessible_only and not bool(contract.get("fully_accessible")):
                    continue
                expected_ids = {
                    str(member["id"])
                    for member in contract["members"]
                    if member["scenario"] == scenario
                    and not bool(member["short_no_long_excluded"])
                    and (
                        str(member["question_family"]),
                        str(member["answer_format"]),
                    ) in required
                }
                if not expected_ids:
                    continue
                relevant = [
                    item
                    for item in items
                    if (info := FantomAdapter._info(item)) is not None
                    and info.get("scenario") == scenario
                    and not bool(info.get("short_no_long_excluded"))
                    and (
                        str(info["question_family"]),
                        str(info["answer_format"]),
                    ) in required
                ]
                observed_ids = {
                    str(FantomAdapter._info(item).get("question_member_id"))
                    for item in relevant
                }
                if observed_ids != expected_ids or len(relevant) != len(expected_ids):
                    incomplete += 1
                    continue
            else:
                active_infos = [
                    info
                    for info in present_infos
                    if not bool(info.get("short_no_long_excluded"))
                ]
                if accessible_only and any(
                    info.get("scenario") != "accessible" for info in active_infos
                ):
                    continue
                relevant = [
                    item
                    for item in items
                    if (info := FantomAdapter._info(item)) is not None
                    and info.get("scenario") == scenario
                    and not bool(info.get("short_no_long_excluded"))
                    and (
                        str(info["question_family"]),
                        str(info["answer_format"]),
                    ) in required
                ]
                categories = {
                    (
                        str(FantomAdapter._info(item)["question_family"]),
                        str(FantomAdapter._info(item)["answer_format"]),
                    )
                    for item in relevant
                }
                if not relevant:
                    continue
                if not required.issubset(categories):
                    incomplete += 1
                    continue
            item_values: list[float] = []
            missing_evaluator = False
            for item in relevant:
                if item.status != ResultStatus.COMPLETED:
                    item_values.append(0.0)
                    continue
                value = _metric_value(item, "fantom.item_correct")
                if value is None:
                    missing_evaluator = True
                    break
                item_values.append(value)
            if missing_evaluator:
                unavailable += 1
                continue
            scores.append(int(bool(item_values) and all(value == 1.0 for value in item_values)))
        total = len(scores) + incomplete + unavailable
        if not total:
            return None, None, 0, 0, 0
        if incomplete or unavailable:
            return None, None, total, incomplete, unavailable
        return sum(scores) / len(scores), sum(scores), len(scores), 0, 0

    def _condition_metrics(
        self,
        metrics: dict[str, MetricValue],
        results: Sequence[CaseResult],
        *,
        context: str,
        scenario: str,
    ) -> None:
        all_tom = [
            result
            for result in results
            if (info := self._info(result)) is not None
            and info.get("question_family") != "fact"
            and info.get("context_condition") == context
            and not bool(info.get("short_no_long_excluded"))
        ]
        candidates = [
            result
            for result in all_tom
            if self._info(result).get("scenario") == scenario
        ]
        if scenario == "accessible":
            fully_accessible_sets = {
                str(info["set_id"])
                for result in all_tom
                if (info := self._info(result)) is not None
                and (
                    bool(info.get("set_contract", {}).get("fully_accessible"))
                    if isinstance(info.get("set_contract"), Mapping)
                    else all(
                        self._info(other).get("scenario") == "accessible"
                        for other in all_tom
                        if str(self._info(other).get("set_id")) == str(info["set_id"])
                    )
                )
            }
            candidates = [
                result for result in candidates if str(self._info(result).get("set_id")) in fully_accessible_sets
            ]
        prefix = f"fantom.{context}.{scenario}"
        required_all_star = {
            ("belief", "free_text"),
            ("belief", "multiple_choice"),
            ("answerability", "list"),
            ("answerability", "binary"),
            ("information_access", "list"),
            ("information_access", "binary"),
        }
        required_all = required_all_star - {("belief", "free_text")}
        for suffix, required in (("all_star", required_all_star), ("all", required_all)):
            value, correct, total, incomplete, unavailable = self._group_all(
                all_tom,
                required,
                scenario=scenario,
                accessible_only=scenario == "accessible",
            )
            metrics[f"{prefix}.{suffix}"] = MetricValue(
                f"{prefix}.{suffix}",
                value,
                unit="proportion",
                numerator=correct,
                denominator=total,
                metadata={
                    "official": True,
                    "aggregation": "set_level_all_questions_correct",
                    "incomplete_group_count": incomplete,
                    "evaluator_unavailable_group_count": unavailable,
                    "control_task": scenario == "accessible",
                },
            )
            if suffix == "all_star":
                complete = total - incomplete - unavailable
                metrics[f"{prefix}.complete_set_rate"] = MetricValue(
                    f"{prefix}.complete_set_rate",
                    complete / total if total else None,
                    unit="proportion",
                    numerator=complete,
                    denominator=total,
                )

        def subset(family: str, answer_format: str) -> list[CaseResult]:
            return [
                result
                for result in candidates
                if self._info(result).get("question_family") == family
                and self._info(result).get("answer_format") == answer_format
            ]

        choice = subset("belief", "multiple_choice")
        value, numerator, denominator = self._deterministic_mean(choice, "fantom.belief_choice_accuracy")
        metrics[f"{prefix}.belief_choice_accuracy"] = MetricValue(
            f"{prefix}.belief_choice_accuracy", value, unit="proportion", numerator=numerator, denominator=denominator
        )
        belief = subset("belief", "free_text")
        value, numerator, denominator, unavailable = self._semantic_mean(belief, "fantom.belief_distance_accuracy")
        metrics[f"{prefix}.belief_distance_accuracy"] = MetricValue(
            f"{prefix}.belief_distance_accuracy",
            value,
            unit="proportion",
            numerator=numerator,
            denominator=denominator,
            metadata={"unavailable_count": unavailable, "semantic_model": FANTOM_SEMANTIC_MODEL},
        )
        correct_belief_token_values = [
            _metric_value(result, "fantom.belief_token_f1")
            for result in belief
            if result.status == ResultStatus.COMPLETED and _metric_value(result, "fantom.item_correct") == 1.0
        ]
        present_token_values = [value for value in correct_belief_token_values if value is not None]
        token_value = None
        if not unavailable and present_token_values:
            token_value = sum(present_token_values) / len(present_token_values)
        metrics[f"{prefix}.belief_token_f1_correct_only"] = MetricValue(
            f"{prefix}.belief_token_f1_correct_only",
            token_value,
            numerator=sum(present_token_values) if token_value is not None else None,
            denominator=len(present_token_values),
            metadata={"official": True, "condition": "belief_distance_correct_only"},
        )
        metrics[f"{prefix}.belief_distance_evidence_availability"] = MetricValue(
            f"{prefix}.belief_distance_evidence_availability",
            (denominator - unavailable) / denominator if denominator else None,
            unit="proportion",
            numerator=denominator - unavailable,
            denominator=denominator,
        )

        for family, family_token in (("answerability", "answerability"), ("information_access", "information_access")):
            list_results = subset(family, "list")
            list_value, list_correct, list_total = self._deterministic_mean(
                list_results, f"fantom.{family}_list_accuracy"
            )
            metrics[f"{prefix}.{family_token}_list_accuracy"] = MetricValue(
                f"{prefix}.{family_token}_list_accuracy",
                list_value,
                unit="proportion",
                numerator=list_correct,
                denominator=list_total,
            )
            binary_results = subset(family, "binary")
            references: list[int] = []
            predictions: list[int] = []
            for result in binary_results:
                info = self._info(result)
                references.append(1 if info.get("binary_reference") == "yes" else 0)
                value = result.prediction.get("value") if isinstance(result.prediction, Mapping) else None
                # Parse failures are completed target attempts, but are not "no".
                if (
                    result.status != ResultStatus.COMPLETED
                    or result.metadata.get("target_output_failure")
                    or value not in ("yes", "no")
                ):
                    predictions.append(-1)
                else:
                    predictions.append(1 if value == "yes" else 0)
            f1_value = weighted_binary_f1(references, predictions)
            metrics[f"{prefix}.{family_token}_binary_weighted_f1"] = MetricValue(
                f"{prefix}.{family_token}_binary_weighted_f1",
                f1_value,
                numerator=None,
                denominator=len(references),
                metadata={"official": True, "average": "weighted", "parse_or_backend_failure_label": -1},
            )
            required = {(family, "list"), (family, "binary")}
            all_value, all_correct, all_total, incomplete, evaluator_unavailable = self._group_all(
                all_tom,
                required,
                scenario=scenario,
                accessible_only=scenario == "accessible",
            )
            metrics[f"{prefix}.{family_token}_all"] = MetricValue(
                f"{prefix}.{family_token}_all",
                all_value,
                unit="proportion",
                numerator=all_correct,
                denominator=all_total,
                metadata={
                    "official": True,
                    "incomplete_group_count": incomplete,
                    "evaluator_unavailable_group_count": evaluator_unavailable,
                },
            )

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        metrics = dict(aggregate_named_metrics(results, namespace=self.benchmark_id))
        contexts = sorted(
            {
                str(info["context_condition"])
                for result in results
                if (info := self._info(result)) is not None
            }
        )
        for context in contexts:
            for scenario in ("inaccessible", "accessible"):
                self._condition_metrics(metrics, results, context=context, scenario=scenario)
            fact_results = [
                result
                for result in results
                if (info := self._info(result)) is not None
                and info.get("context_condition") == context
                and info.get("question_family") == "fact"
            ]
            fact_value, fact_sum, fact_total = self._deterministic_mean(fact_results, "fantom.fact_token_f1")
            metrics[f"fantom.{context}.fact_token_f1"] = MetricValue(
                f"fantom.{context}.fact_token_f1",
                fact_value,
                numerator=fact_sum,
                denominator=fact_total,
                metadata={"official_control_component": True, "failures_are_zero": True},
            )

        main_contexts = [
            context
            for context in contexts
            if any(
                (info := self._info(result)) is not None
                and info.get("context_condition") == context
                and info.get("scenario") == "inaccessible"
                and info.get("question_family") != "fact"
                for result in results
            )
        ]
        for suffix in ("all_star", "all"):
            source = metrics.get(f"fantom.{main_contexts[0]}.inaccessible.{suffix}") if len(main_contexts) == 1 else None
            metrics[f"fantom.{suffix}"] = MetricValue(
                f"fantom.{suffix}",
                source.value if source else None,
                unit="proportion",
                numerator=source.numerator if source else None,
                denominator=source.denominator if source else 0,
                metadata={
                    "official_primary": suffix == "all_star",
                    "leaderboard_order": "All* then All",
                    "context_condition": main_contexts[0] if len(main_contexts) == 1 else None,
                    "availability": "available" if source else "unavailable_mixed_or_missing_context_condition",
                },
            )
        parse_metric = parse_failure_rate(results)
        metrics["fantom.parse_failure_rate"] = MetricValue(
            "fantom.parse_failure_rate",
            parse_metric.value,
            direction="lower_is_better",
            unit="proportion",
            numerator=parse_metric.numerator,
            denominator=parse_metric.denominator,
        )
        return metrics


__all__ = [
    "FANTOM_CONTEXT_CONDITIONS",
    "FANTOM_FORMATS",
    "FANTOM_SCENARIOS",
    "FANTOM_SEMANTIC_CONTRACT",
    "FANTOM_SEMANTIC_MODEL",
    "FANTOM_SET_CONTRACT",
    "FantomAdapter",
    "FantomPrediction",
    "belief_whitespace_token_f1",
    "weighted_binary_f1",
    "whitespace_token_f1",
]
