"""LifeChoice literary-character decision adapter."""

from __future__ import annotations

from ..presentation import material_user_message

from collections import defaultdict
from typing import Any, Mapping, Sequence

from ..contracts import BenchmarkCase, CaseResult, ChatMessage, MetricValue, ModelRequest, ModelResponse
from ..data.schemas import probe_case
from ..errors import ValidationError
from ..json_utils import canonical_json
from ..registry import adapter
from ..model_adaptation import adapted_choice_text
from .choice import (
    ChoicePrediction,
    DisplayedChoice,
    StaticChoiceAdapter,
    alphabetical_labels,
    exact_binary_accuracy,
    indexed_choices,
    parse_failure_rate,
    parse_strict_choice,
    safe_metric_token,
)
from .common import aggregate_named_metrics


@adapter("lifechoices")
class LifeChoicesAdapter(StaticChoiceAdapter):
    benchmark_id = "lifechoices"
    prompt_revision = "lifechoices-supplemental-answer-tag-v3-natural-material-v1"
    scorer_revision = "lifechoice-exact-option-accuracy-v1"
    metric_name = "lifechoices.accuracy"

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"LifeChoicesAdapter cannot run {case.benchmark_id!r}")
        probe_case(case)
        values = case.input_data
        for field in ("character_profile", "decision_context", "book_id"):
            if not isinstance(values.get(field), str) or not str(values[field]).strip():
                raise ValidationError(f"LifeChoice {field} must be a nonempty string")
        choices = self.choices_for_case(case, seed=0)
        if len({choice.text.casefold() for choice in choices}) != 4:
            raise ValidationError("LifeChoice options must be textually distinct")
        self.gold_source_id(case)
        strata = case.metadata.get("strata") or {}
        if not str(strata.get("context_condition", "")).strip():
            raise ValidationError("LifeChoice context_condition cannot be empty")

    def choices_for_case(self, case: BenchmarkCase, *, seed: int) -> tuple[DisplayedChoice, ...]:
        del seed
        options = case.input_data.get("options")
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)) or len(options) != 4:
            raise ValidationError("LifeChoice requires exactly four options")
        labels = alphabetical_labels(4)
        return indexed_choices(
            options,
            labels=labels,
            source_ids=labels,
            strip_matching_prefix=True,
        )

    def gold_source_id(self, case: BenchmarkCase) -> str:
        gold = case.gold
        if isinstance(gold, int) and not isinstance(gold, bool) and 0 <= gold < 4:
            return alphabetical_labels(4)[gold]
        if isinstance(gold, str):
            normalized = gold.strip().upper().rstrip(".)")
            if normalized in alphabetical_labels(4):
                return normalized
        raise ValidationError("LifeChoice gold must be an option index or one exact label A-D")

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        self.validate_case(case)
        choices = self.choices_for_case(case, seed=seed)
        values = case.input_data
        payload = {
            "character": values.get("character_name") or values.get("character_id") or "assigned character",
            "profile": values["character_profile"],
            "scenario": values["decision_context"],
            "question": values.get("question") or "Which action would this character take?",
            "options": [choice.public_dict() for choice in choices],
        }
        return ModelRequest(
            request_id=f"{case.case_id}:choice",
            messages=(
                ChatMessage(
                    "system",
                    "Reason from the supplied character profile and only pre-decision context. "
                    "Select exactly one displayed option. You may think step by step before answering.",
                ),
                material_user_message(
                    payload, {'character': 'Character', 'profile': 'Profile', 'scenario': 'Scenario', 'question': 'Question', 'options': 'Options'},
                    truncatable=('profile', 'scenario'), suffix="\nOutput the final choice as <answer>X</answer>, where X is exactly one of A, B, C, or D.",
                ),
            ),
            model=model,
            temperature=0.0,
            max_tokens=1024,
            seed=seed,
            response_format=None,
            metadata={
                "benchmark_id": self.benchmark_id,
                "prompt_revision": self.prompt_revision,
                "gold_visible": False,
            },
        )

    def parse_choice_response(
        self,
        case: BenchmarkCase,
        response: ModelResponse,
        choices: Sequence[DisplayedChoice],
    ) -> ChoicePrediction:
        del case
        return parse_strict_choice(adapted_choice_text(response, choices), choices, allow_answer_tag=True)

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        del case
        return {
            "revision": "static-choice-v1",
            "option_policy": "preserve_official_source_order",
            "parser_revision": "supplemental-answer-tag-with-legacy-json-compat-v1",
        }

    def result_metadata(
        self,
        case: BenchmarkCase,
        choices: Sequence[DisplayedChoice],
        *,
        seed: int,
    ) -> Mapping[str, Any]:
        metadata = dict(super().result_metadata(case, choices, seed=seed))
        strata = case.metadata.get("strata") or {}
        metadata["lifechoices"] = {
            "book_id": case.input_data["book_id"],
            "character_id": case.input_data.get("character_id") or case.input_data.get("character_name"),
            "context_condition": strata.get("context_condition"),
            "entity_replacement_revision": case.input_data.get("entity_replacement_revision"),
            "profile_method": case.input_data.get("profile_method") or strata.get("context_condition"),
            "gold_visible_to_model": False,
        }
        return metadata

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        metrics = dict(aggregate_named_metrics(results, namespace=self.benchmark_id))
        value, correct, denominator = exact_binary_accuracy(results, self.metric_name)
        metrics[self.metric_name] = MetricValue(
            self.metric_name,
            value,
            unit="proportion",
            numerator=correct,
            denominator=denominator,
            metadata={
                "official_primary": True,
                "aggregation": "micro_over_all_decisions_failures_are_incorrect",
            },
        )
        parse_metric = parse_failure_rate(results)
        metrics["lifechoices.parse_failure_rate"] = MetricValue(
            "lifechoices.parse_failure_rate",
            parse_metric.value,
            direction="lower_is_better",
            unit="proportion",
            numerator=parse_metric.numerator,
            denominator=parse_metric.denominator,
        )
        by_condition: dict[str, list[CaseResult]] = defaultdict(list)
        by_book: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            info = result.metadata.get("lifechoices")
            if not isinstance(info, Mapping):
                continue
            by_condition[str(info.get("context_condition") or "unspecified")].append(result)
            by_book[str(info.get("book_id") or result.group_id)].append(result)
        for condition, items in sorted(by_condition.items()):
            condition_value, condition_correct, condition_total = exact_binary_accuracy(items, self.metric_name)
            name = f"lifechoices.accuracy.context_condition.{safe_metric_token(condition)}"
            metrics[name] = MetricValue(
                name,
                condition_value,
                unit="proportion",
                numerator=condition_correct,
                denominator=condition_total,
                metadata={"context_condition": condition},
            )
        book_values = [exact_binary_accuracy(items, self.metric_name)[0] for items in by_book.values()]
        present_book_values = [float(item) for item in book_values if item is not None]
        metrics["lifechoices.book_macro_accuracy"] = MetricValue(
            "lifechoices.book_macro_accuracy",
            sum(present_book_values) / len(present_book_values) if present_book_values else None,
            unit="proportion",
            numerator=sum(present_book_values) if present_book_values else None,
            denominator=len(present_book_values),
            metadata={
                "diagnostic": True,
                "official_primary": False,
                "aggregation": "mean_of_per_book_accuracies",
            },
        )
        return metrics


__all__ = ["LifeChoicesAdapter"]
