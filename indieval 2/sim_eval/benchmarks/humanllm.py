"""HumanLLM 20-way Item Selection adapter."""

from __future__ import annotations

from ..presentation import material_user_message

from collections import defaultdict
import json
from typing import Any, Mapping, Sequence

from ..contracts import BenchmarkCase, CaseResult, ChatMessage, MetricValue, ModelRequest, ModelResponse
from ..data.schemas import probe_case
from ..errors import ParseError, ValidationError
from ..json_utils import canonical_json
from ..registry import adapter
from ..model_adaptation import adapted_choice_text
from .choice import (
    ChoicePrediction,
    DisplayedChoice,
    StaticChoiceAdapter,
    alphabetical_labels,
    exact_binary_accuracy,
    parse_failure_rate,
    parse_strict_choice,
    safe_metric_token,
)
from .common import aggregate_named_metrics


@adapter("humanllm")
class HumanLLMItemSelectionAdapter(StaticChoiceAdapter):
    benchmark_id = "humanllm"
    prompt_revision = "humanllm-20-rank5-supplemental-answer-tag-v2-natural-material-v1"
    scorer_revision = "humanllm-official-top1-plus-supplemental-rank5-v1"
    metric_name = "humanllm.top1_accuracy"

    @staticmethod
    def _candidate_identity(value: Any, index: int) -> tuple[str, str]:
        if isinstance(value, str):
            if not value.strip():
                raise ValidationError(f"HumanLLM candidate #{index} is empty")
            return f"source_index_{index}", value.strip()
        if isinstance(value, Mapping):
            source_id = value.get("item_id") or value.get("id")
            title = value.get("title") or value.get("name")
            if isinstance(source_id, bool) or not isinstance(source_id, (str, int)) or not str(source_id).strip():
                raise ValidationError(f"HumanLLM candidate #{index} object requires item_id")
            if not isinstance(title, str) or not title.strip():
                raise ValidationError(f"HumanLLM candidate #{index} object requires title")
            return str(source_id).strip(), title.strip()
        raise ValidationError(f"HumanLLM candidate #{index} must be a string or item object")

    def choices_for_case(self, case: BenchmarkCase, *, seed: int) -> tuple[DisplayedChoice, ...]:
        del seed
        raw = case.input_data.get("candidates")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 20:
            raise ValidationError("HumanLLM Item Selection requires exactly 20 candidates")
        identities = [self._candidate_identity(value, index) for index, value in enumerate(raw)]
        source_ids = [item[0] for item in identities]
        if len(set(source_ids)) != len(source_ids):
            raise ValidationError("HumanLLM candidate item IDs must be unique")
        return tuple(
            DisplayedChoice(label, source_id, title, index)
            for index, (label, (source_id, title)) in enumerate(zip(alphabetical_labels(20), identities))
        )

    def gold_source_id(self, case: BenchmarkCase) -> str:
        choices = self.choices_for_case(case, seed=0)
        gold = case.gold
        if isinstance(gold, int) and not isinstance(gold, bool) and 0 <= gold < len(choices):
            return choices[gold].source_id
        if isinstance(gold, str) and gold.strip():
            token = gold.strip().casefold()
            by_source = [choice for choice in choices if choice.source_id.casefold() == token]
            if len(by_source) == 1:
                return by_source[0].source_id
            by_title = [choice for choice in choices if choice.text.casefold() == token]
            if len(by_title) == 1:
                return by_title[0].source_id
            if len(by_title) > 1:
                raise ValidationError("HumanLLM gold title is ambiguous; use an item ID")
        raise ValidationError("HumanLLM gold must be a valid source index, item ID, or unique exact title")

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"HumanLLMItemSelectionAdapter cannot run {case.benchmark_id!r}")
        probe_case(case)
        profile = case.input_data.get("user_profile")
        history = case.input_data.get("purchase_history")
        has_profile = (
            isinstance(profile, str) and bool(profile.strip())
        ) or (isinstance(profile, Mapping) and bool(profile))
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise ValidationError("HumanLLM purchase_history must be an array")
        if len(history) > 30:
            raise ValidationError("HumanLLM purchase_history cannot contain more than 30 items")
        for index, item in enumerate(history):
            valid = (isinstance(item, str) and bool(item.strip())) or (
                isinstance(item, Mapping) and bool(item)
            )
            if not valid:
                raise ValidationError(f"HumanLLM purchase history item #{index} is empty or invalid")
        has_history = bool(history)
        if not has_profile and not has_history:
            raise ValidationError("HumanLLM requires a nonempty user profile or purchase history")
        self.choices_for_case(case, seed=0)
        self.gold_source_id(case)

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        self.validate_case(case)
        choices = self.choices_for_case(case, seed=seed)
        payload = {
            "user_profile": case.input_data.get("user_profile"),
            "purchase_history": list(case.input_data.get("purchase_history") or ()),
            "product_category": (case.metadata.get("strata") or {}).get("product_category_if_available"),
            "candidates": [choice.public_dict() for choice in choices],
        }
        return ModelRequest(
            request_id=f"{case.case_id}:choice",
            messages=(
                ChatMessage(
                    "system",
                    "Infer the user's next purchase from only the supplied profile and prior purchases. "
                    "Rank the five most likely purchases from the 20 candidates. You may think concisely before answering. "
                    "Your final answer must use <answer>X1,X2,X3,X4,X5</answer> with five distinct letters from A through T.",
                ),
                material_user_message(
                    payload, {'user_profile': 'About This User', 'purchase_history': 'Purchase History', 'product_category': 'Product Category', 'candidates': 'Candidate Items'},
                    truncatable=('user_profile', 'purchase_history'), suffix="\nRank the five candidates this user is most likely to buy next. "
                    "Return the final ranking as <answer>X1,X2,X3,X4,X5</answer>: exactly five distinct A-T letters, "
                    "comma-separated with no spaces, ordered from most to least likely.",
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
                "paper_generation_temperature": 0.7,
                "configured_generation_temperature": 0.0,
                "candidate_count": 20,
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
        response_text = adapted_choice_text(response, choices, ranking=True)
        raw = response_text.strip()
        if raw.startswith("{"):
            # Historical indieval artifacts used C01..C20.  Normalize those
            # stable legacy IDs before applying the released A..T protocol.
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, Mapping):
                def official_label(value: Any) -> Any:
                    if isinstance(value, str) and len(value) == 3 and value.startswith("C") and value[1:].isdigit():
                        index = int(value[1:]) - 1
                        if 0 <= index < 20:
                            return alphabetical_labels(20)[index]
                    return value

                normalized = dict(payload)
                normalized["choice"] = official_label(normalized.get("choice"))
                ranking = normalized.get("ranking")
                if isinstance(ranking, Sequence) and not isinstance(ranking, (str, bytes)):
                    normalized["ranking"] = [official_label(value) for value in ranking]
                response_text = canonical_json(normalized)
        prediction = parse_strict_choice(
            response_text,
            choices,
            allow_exact_text=True,
            allow_ranking=True,
            allow_answer_tag=True,
        )
        if len(prediction.ranking_source_ids) != 5:
            raise ParseError("HumanLLM Supplemental-compatible output requires exactly five ranked candidates")
        if len(set(prediction.ranking_source_ids)) != 5:
            raise ParseError("HumanLLM ranking must contain five distinct candidates")
        return prediction

    def score_response(
        self,
        case: BenchmarkCase,
        prediction: ChoicePrediction,
        response: ModelResponse,
        *,
        seed: int,
    ) -> Sequence[MetricValue]:
        del response, seed
        gold = self.gold_source_id(case)
        correct = int(prediction.source_id == gold)
        rank = prediction.ranking_source_ids.index(gold) + 1 if gold in prediction.ranking_source_ids else None
        hit_at_5 = int(rank is not None)
        reciprocal_rank = 1.0 / rank if rank is not None else 0.0
        supplemental_meta = {
            "official_primary": False,
            "protocol": "supplemental_20_candidates_rank_5",
            "rank": rank,
        }
        return (
            MetricValue(
                self.metric_name,
                correct,
                unit="proportion",
                numerator=correct,
                denominator=1,
                metadata={"official_primary": True, "candidate_count": 20, "scoring": "strict_top1_exact"},
            ),
            MetricValue(
                "humanllm.supplemental.hit_at_5",
                hit_at_5,
                unit="proportion",
                numerator=hit_at_5,
                denominator=1,
                metadata=supplemental_meta,
            ),
            MetricValue(
                "humanllm.supplemental.reciprocal_rank",
                reciprocal_rank,
                numerator=reciprocal_rank,
                denominator=1,
                metadata=supplemental_meta,
            ),
            # Backward-compatible diagnostic names retained for existing reports.
            MetricValue(
                "humanllm.diagnostic.top5_accuracy",
                hit_at_5,
                unit="proportion",
                numerator=hit_at_5,
                denominator=1,
                metadata=supplemental_meta,
            ),
            MetricValue(
                "humanllm.diagnostic.reciprocal_rank",
                reciprocal_rank,
                numerator=reciprocal_rank,
                denominator=1,
                metadata=supplemental_meta,
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
        del case, response, seed
        metadata = {
            "scoring": "target_output_parse_failure_is_incorrect",
            "parse_error": str(exc),
        }
        return (
            MetricValue(self.metric_name, 0, unit="proportion", numerator=0, denominator=1, metadata=metadata),
            MetricValue("humanllm.supplemental.hit_at_5", 0, unit="proportion", numerator=0, denominator=1, metadata=metadata),
            MetricValue("humanllm.supplemental.reciprocal_rank", 0.0, numerator=0.0, denominator=1, metadata=metadata),
            MetricValue("humanllm.diagnostic.top5_accuracy", 0, unit="proportion", numerator=0, denominator=1, metadata=metadata),
            MetricValue("humanllm.diagnostic.reciprocal_rank", 0.0, numerator=0.0, denominator=1, metadata=metadata),
        )

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        del case
        return {
            "revision": "humanllm-static-choice-v2",
            "candidate_policy": "preserve_authorized_source_order",
            "candidate_count": 20,
            "parser_revision": "supplemental-answer-tag-rank5-with-legacy-json-compat-v1",
            "protocols": {
                "official_paper": "20_candidates_choose_1_from_rank_1",
                "supplemental_compatibility": "20_candidates_rank_5_reciprocal_rank",
            },
            "paper_generation_temperature": 0.7,
            "configured_generation_temperature": 0.0,
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
        metadata["humanllm"] = {
            "user": strata.get("user") or case.group_id,
            "history_length": len(case.input_data.get("purchase_history") or ()),
            "history_length_bucket": strata.get("history_length_bucket"),
            "product_category": strata.get("product_category_if_available"),
            "candidate_count": len(choices),
            "conditioning_mode": (
                "profile_and_history"
                if case.input_data.get("user_profile") and case.input_data.get("purchase_history")
                else "profile_only"
                if case.input_data.get("user_profile")
                else "history_only"
            ),
            "candidate_order_policy": "preserve_authorized_source_order",
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
                "candidate_count": 20,
                "aggregation": "micro_over_all_items_target_failures_are_incorrect",
            },
        )
        parse_metric = parse_failure_rate(results)
        metrics["humanllm.parse_failure_rate"] = MetricValue(
            "humanllm.parse_failure_rate",
            parse_metric.value,
            direction="lower_is_better",
            unit="proportion",
            numerator=parse_metric.numerator,
            denominator=parse_metric.denominator,
        )
        by_user: dict[str, list[CaseResult]] = defaultdict(list)
        by_category: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            info = result.metadata.get("humanllm")
            if not isinstance(info, Mapping):
                continue
            by_user[str(info.get("user") or result.group_id)].append(result)
            by_category[str(info.get("product_category") or "unspecified")].append(result)
        user_values = [exact_binary_accuracy(items, self.metric_name)[0] for items in by_user.values()]
        user_values = [float(item) for item in user_values if item is not None]
        metrics["humanllm.diagnostic.user_macro_top1_accuracy"] = MetricValue(
            "humanllm.diagnostic.user_macro_top1_accuracy",
            sum(user_values) / len(user_values) if user_values else None,
            unit="proportion",
            numerator=sum(user_values) if user_values else None,
            denominator=len(user_values),
            metadata={"official_primary": False, "aggregation": "mean_of_per_user_top1"},
        )
        for category, items in sorted(by_category.items()):
            category_value, category_correct, category_total = exact_binary_accuracy(items, self.metric_name)
            name = f"humanllm.top1_accuracy.category.{safe_metric_token(category)}"
            metrics[name] = MetricValue(
                name,
                category_value,
                unit="proportion",
                numerator=category_correct,
                denominator=category_total,
                metadata={"product_category": category},
            )
        return metrics


__all__ = ["HumanLLMItemSelectionAdapter"]
