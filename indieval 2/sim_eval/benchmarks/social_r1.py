"""Social-R1's held-out ToMBench-Hard multiple-choice evaluation adapter."""

from __future__ import annotations

from collections import defaultdict
import re
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
    indexed_choices,
    parse_failure_rate,
    parse_strict_choice,
    safe_metric_token,
)
from .common import aggregate_named_metrics


SOCIAL_R1_ALIASES = ("Social-R1", "ToMBench-Hard")
SOCIAL_R1_CANONICAL_VARIANT = "tombench_hard_official_v1"
SOCIAL_R1_COMPAT_VARIANT = "supplemental_compat_v1"
ATOMS_DIMENSIONS = (
    "belief",
    "desire",
    "emotion",
    "intention",
    "knowledge",
    "non_literal_communication",
)

ODYSIM_COMPAT_SYSTEM_PROMPT = (
    "You are solving a social reasoning multiple-choice question.\n"
    "Read the story, question, and options carefully.\n"
    "Think concisely, then output your final answer in the format:\n"
    "<answer>X</answer>\n"
    "where X is exactly one option letter from the provided choices."
)


def social_r1_protocol_variant(case: BenchmarkCase) -> str:
    """Return the explicit compatibility variant or canonical default."""

    value = case.metadata.get("protocol_variant", SOCIAL_R1_CANONICAL_VARIANT)
    if value not in {SOCIAL_R1_CANONICAL_VARIANT, SOCIAL_R1_COMPAT_VARIANT}:
        raise ValidationError(f"unknown Social-R1 protocol_variant {value!r}")
    return str(value)


def parse_social_r1_response(text: str, choices: Sequence[DisplayedChoice]) -> ChoicePrediction:
    """Accept an exact label/JSON or the paper's complete XML-style output."""

    raw = str(text or "").strip()
    xml = re.fullmatch(
        r"(?is)(?:<(think|thinking)>(.*?)</\1>\s*)?<answer>\s*([A-Z])\s*</answer>",
        raw,
    )
    if xml:
        reasoning = xml.group(2) or ""
        if re.search(r"(?is)</?answer\b", reasoning):
            raise ParseError("Social-R1 reasoning must not contain another answer tag")
        raw = xml.group(3).upper()
    return parse_strict_choice(raw, choices)


def parse_social_r1_compat_response(
    text: str,
    choices: Sequence[DisplayedChoice],
) -> ChoicePrediction:
    """Mirror Supplemental's project-side parser: strip think blocks, then find an answer tag."""

    raw = str(text or "")
    raw = re.sub(r"<seed:think>.*?</seed:think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"<answer>\s*([A-Z])\s*</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        raise ParseError("Supplemental-compatible Social-R1 response requires <answer>X</answer>")
    return parse_strict_choice(match.group(1).upper(), choices)


@adapter("social_r1")
class SocialR1Adapter(StaticChoiceAdapter):
    benchmark_id = "social_r1"
    aliases = SOCIAL_R1_ALIASES
    prompt_revision = "indieval-social-r1-variant-aware-v2"
    scorer_revision = "social-r1-variant-aware-exact-accuracy-v2"
    metric_name = "social_r1.accuracy"

    def choices_for_case(self, case: BenchmarkCase, *, seed: int) -> tuple[DisplayedChoice, ...]:
        del seed
        options = case.input_data.get("options")
        if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
            raise ValidationError("Social-R1 requires an options array")
        variant = social_r1_protocol_variant(case)
        expected = {4} if variant == SOCIAL_R1_CANONICAL_VARIANT else {2, 3, 4, 5, 6}
        if len(options) not in expected:
            if variant == SOCIAL_R1_CANONICAL_VARIANT:
                raise ValidationError("canonical Social-R1 / ToMBench-Hard requires exactly four options")
            raise ValidationError("Supplemental-compatible Social-R1 requires 2 through 6 options")
        labels = alphabetical_labels(len(options))
        choices = indexed_choices(
            options,
            labels=labels,
            source_ids=labels,
            strip_matching_prefix=True,
        )
        if len({choice.text.casefold() for choice in choices}) != len(choices):
            raise ValidationError("Social-R1 options must be textually distinct")
        return choices

    def gold_source_id(self, case: BenchmarkCase) -> str:
        options = case.input_data.get("options")
        option_count = len(options) if isinstance(options, Sequence) and not isinstance(options, (str, bytes)) else 0
        labels = alphabetical_labels(option_count) if option_count else ()
        if isinstance(case.gold, int) and not isinstance(case.gold, bool) and 0 <= case.gold < option_count:
            return labels[case.gold]
        if isinstance(case.gold, str):
            match = re.fullmatch(r"\s*([A-Za-z])(?:\s*[.)])?\s*", case.gold)
            if match and match.group(1).upper() in labels:
                return match.group(1).upper()
        raise ValidationError(
            "Social-R1 gold must be one exact displayed label or a zero-based source index"
        )

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"SocialR1Adapter cannot run {case.benchmark_id!r}")
        probe_case(case)
        if not isinstance(case.input_data.get("question"), str) or not str(case.input_data["question"]).strip():
            raise ValidationError("Social-R1 question must be nonempty text")
        strata = case.metadata.get("strata")
        if not isinstance(strata, Mapping):
            raise ValidationError("Social-R1 requires metadata.strata")
        variant = social_r1_protocol_variant(case)
        if variant == SOCIAL_R1_CANONICAL_VARIANT:
            dimension = case.input_data.get("atoms_dimension")
            if dimension not in ATOMS_DIMENSIONS:
                raise ValidationError(f"Social-R1 atoms_dimension must be one of {ATOMS_DIMENSIONS}")
            if strata.get("atoms_dimension") != dimension:
                raise ValidationError(
                    "canonical Social-R1 metadata.strata.atoms_dimension must match input.atoms_dimension"
                )
        else:
            if case.metadata.get("source_kind") != "local_compatibility":
                raise ValidationError(
                    "Supplemental-compatible Social-R1 must use source_kind='local_compatibility'"
                )
            if "atoms_dimension" in case.input_data or "atoms_dimension" in strata:
                raise ValidationError(
                    "Supplemental-compatible Social-R1 has no released ATOMS labels; do not infer or insert one"
                )
            prompt_text = case.input_data.get("prompt_text")
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                raise ValidationError("Supplemental-compatible Social-R1 requires the exact source prompt_text")
            if case.input_data.get("num_options") != len(case.input_data["options"]):
                raise ValidationError(
                    "Supplemental-compatible Social-R1 input.num_options must match the options array"
                )
            if strata.get("protocol_variant") != SOCIAL_R1_COMPAT_VARIANT:
                raise ValidationError(
                    "Supplemental-compatible Social-R1 requires protocol_variant in metadata.strata"
                )
            if strata.get("num_options") != len(case.input_data["options"]):
                raise ValidationError(
                    "Supplemental-compatible Social-R1 metadata.strata.num_options must match options"
                )
        self.choices_for_case(case, seed=0)
        self.gold_source_id(case)

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        self.validate_case(case)
        choices = self.choices_for_case(case, seed=seed)
        variant = social_r1_protocol_variant(case)
        if variant == SOCIAL_R1_COMPAT_VARIANT:
            return ModelRequest(
                request_id=f"{case.case_id}:choice",
                messages=(
                    ChatMessage("system", ODYSIM_COMPAT_SYSTEM_PROMPT),
                    ChatMessage("user", str(case.input_data["prompt_text"])),
                ),
                model=model,
                temperature=0.0,
                max_tokens=2048,
                seed=seed,
                metadata={
                    "benchmark_id": self.benchmark_id,
                    "benchmark_identity": "Supplemental Social-R1 author-project compatibility snapshot",
                    "protocol_variant": variant,
                    "prompt_revision": self.prompt_revision,
                    "gold_visible": False,
                    "canonical_score_eligible": False,
                },
            )
        payload = {
            "story_and_question": case.input_data["question"],
            "options": [choice.public_dict() for choice in choices],
        }
        return ModelRequest(
            request_id=f"{case.case_id}:choice",
            messages=(
                ChatMessage(
                    "system",
                    "Solve this social-reasoning multiple-choice item from the story evidence. Select exactly one option.",
                ),
                ChatMessage("user", canonical_json(payload) + "\nReturn only <answer>A</answer> with the chosen label."),
            ),
            model=model,
            temperature=0.0,
            max_tokens=2048,
            seed=seed,
            metadata={
                "benchmark_id": self.benchmark_id,
                "benchmark_identity": "Social-R1/ToMBench-Hard held-out test",
                "protocol_variant": variant,
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
        if social_r1_protocol_variant(case) == SOCIAL_R1_COMPAT_VARIANT:
            return parse_social_r1_compat_response(adapted_choice_text(response, choices), choices)
        return parse_social_r1_response(response.text, choices)

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        variant = social_r1_protocol_variant(case)
        if variant == SOCIAL_R1_COMPAT_VARIANT:
            return {
                "revision": "supplemental-social-r1-author-project-static-choice-v1",
                "protocol_variant": variant,
                "split": "test",
                "canonical_tombench_hard": False,
                "option_policy": "preserve_author_project_source_order_dynamic_A_through_F",
                "parser_revision": "supplemental-remove-think-first-answer-tag-v1",
                "availability_gate": "local_compatibility_manifest_required",
            }
        return {
            "revision": "tombench-hard-static-choice-v1",
            "protocol_variant": variant,
            "split": "test",
            "expected_population": 100,
            "parser_revision": "strict-label-json-or-complete-answer-xml-v2",
            "option_policy": "preserve_authorized_source_order",
            "availability_gate": "author_release_or_authorized_local_copy_required",
        }

    def result_metadata(
        self,
        case: BenchmarkCase,
        choices: Sequence[DisplayedChoice],
        *,
        seed: int,
    ) -> Mapping[str, Any]:
        metadata = dict(super().result_metadata(case, choices, seed=seed))
        variant = social_r1_protocol_variant(case)
        choice_contract = dict(metadata["choice_contract"])
        choice_contract["parser_revision"] = (
            "supplemental-remove-think-first-answer-tag-v1"
            if variant == SOCIAL_R1_COMPAT_VARIANT
            else "strict-label-json-or-complete-answer-xml-v2"
        )
        metadata["choice_contract"] = choice_contract
        if variant == SOCIAL_R1_COMPAT_VARIANT:
            metadata["social_r1"] = {
                "benchmark_identity": "Supplemental Social-R1 author-project compatibility snapshot",
                "aliases": ["Social-R1 (Supplemental compatibility)"],
                "protocol_variant": variant,
                "split": case.split,
                "num_options": len(choices),
                "canonical_tombench_hard": False,
                "canonical_score_eligible": False,
                "official_data_available": False,
                "source_status": "author_project_local_compatibility",
                "contamination_policy": case.metadata.get("contamination_policy", "none"),
                "gold_visible_to_model": False,
            }
        else:
            metadata["social_r1"] = {
                "benchmark_identity": "Social-R1/ToMBench-Hard",
                "aliases": list(SOCIAL_R1_ALIASES),
                "protocol_variant": variant,
                "split": "test" if case.split != "fixture" else "fixture",
                "atoms_dimension": case.input_data["atoms_dimension"],
                "official_test_population": 100,
                "official_data_available": case.metadata.get("source_kind") in {"official", "authorized_local"},
                "placeholder_dataset_revision": "539f50b8a6c35e643628ebae0f842681d7079cf3",
                "gold_visible_to_model": False,
            }
        return metadata

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        metrics = dict(aggregate_named_metrics(results, namespace=self.benchmark_id))
        variants = {
            str(info.get("protocol_variant"))
            for result in results
            if isinstance((info := result.metadata.get("social_r1")), Mapping)
            and info.get("protocol_variant")
        }
        if len(variants) > 1:
            raise ValidationError(
                f"cannot pool incompatible Social-R1 protocol variants: {sorted(variants)!r}"
            )
        variant = next(iter(variants), SOCIAL_R1_CANONICAL_VARIANT)
        value, correct, total = exact_binary_accuracy(results, self.metric_name)
        metrics[self.metric_name] = MetricValue(
            self.metric_name,
            value,
            unit="proportion",
            numerator=correct,
            denominator=total,
            metadata={
                "official_primary": variant == SOCIAL_R1_CANONICAL_VARIANT,
                "aggregation": "micro_over_held_out_test_failures_are_incorrect",
                "benchmark_identity": (
                    "Social-R1/ToMBench-Hard"
                    if variant == SOCIAL_R1_CANONICAL_VARIANT
                    else "Supplemental Social-R1 author-project compatibility snapshot"
                ),
                "protocol_variant": variant,
                "canonical_score_eligible": variant == SOCIAL_R1_CANONICAL_VARIANT,
            },
        )
        parse_metric = parse_failure_rate(results)
        metrics["social_r1.parse_failure_rate"] = MetricValue(
            "social_r1.parse_failure_rate",
            parse_metric.value,
            direction="lower_is_better",
            unit="proportion",
            numerator=parse_metric.numerator,
            denominator=parse_metric.denominator,
        )
        if variant == SOCIAL_R1_COMPAT_VARIANT:
            by_option_count: dict[int, list[CaseResult]] = defaultdict(list)
            for result in results:
                info = result.metadata.get("social_r1")
                if isinstance(info, Mapping) and isinstance(info.get("num_options"), int):
                    by_option_count[int(info["num_options"])].append(result)
            for option_count in sorted(by_option_count):
                items = by_option_count[option_count]
                slice_value, slice_correct, slice_total = exact_binary_accuracy(items, self.metric_name)
                name = f"social_r1.compat.accuracy.num_options.{option_count}"
                metrics[name] = MetricValue(
                    name,
                    slice_value,
                    unit="proportion",
                    numerator=slice_correct,
                    denominator=slice_total,
                    metadata={"num_options": option_count, "official_breakdown": False},
                )
            return metrics

        by_dimension: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            info = result.metadata.get("social_r1")
            if isinstance(info, Mapping):
                by_dimension[str(info.get("atoms_dimension"))].append(result)
        for dimension in ATOMS_DIMENSIONS:
            items = by_dimension.get(dimension, ())
            dimension_value, dimension_correct, dimension_total = exact_binary_accuracy(items, self.metric_name)
            name = f"social_r1.accuracy.atoms_dimension.{safe_metric_token(dimension)}"
            metrics[name] = MetricValue(
                name,
                dimension_value,
                unit="proportion",
                numerator=dimension_correct,
                denominator=dimension_total,
                metadata={"atoms_dimension": dimension, "official_breakdown": True},
            )
            count_name = name.replace(".accuracy.", ".count.")
            metrics[count_name] = MetricValue(
                count_name,
                dimension_total,
                unit="questions",
                numerator=dimension_total,
                denominator=dimension_total,
                metadata={"atoms_dimension": dimension},
            )
        return metrics


__all__ = [
    "ATOMS_DIMENSIONS",
    "ODYSIM_COMPAT_SYSTEM_PROMPT",
    "SOCIAL_R1_ALIASES",
    "SOCIAL_R1_CANONICAL_VARIANT",
    "SOCIAL_R1_COMPAT_VARIANT",
    "SocialR1Adapter",
    "parse_social_r1_compat_response",
    "parse_social_r1_response",
    "social_r1_protocol_variant",
]
