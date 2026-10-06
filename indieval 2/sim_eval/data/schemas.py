"""Normalized schema probes for the benchmark families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..contracts import BenchmarkCase
from ..errors import ValidationError


@dataclass(frozen=True)
class CaseSchema:
    benchmark_id: str
    required_input_fields: Sequence[str]
    required_strata: Sequence[str]
    topology: str
    gold_optional: bool = False


SCHEMAS: dict[str, CaseSchema] = {
    "fantom": CaseSchema(
        "fantom",
        (
            "conversation",
            "question",
            "question_family",
            "answer_format",
            "context_condition",
            "scenario",
            "conversation_id",
            "part_id",
            "set_id",
        ),
        ("question_family", "answer_format", "context_condition"),
        "static",
    ),
    # Social-R1 has two deliberately non-interchangeable source contracts. The
    # canonical ToMBench-Hard release requires atoms_dimension; the Supplemental
    # author-project compatibility snapshot does not expose it. Conditional
    # validation therefore lives in SocialR1Adapter rather than this shape-only
    # probe, so the compatibility loader never invents an ATOMS annotation.
    "social_r1": CaseSchema("social_r1", ("question", "options"), (), "static"),
    "lifechoices": CaseSchema("lifechoices", ("character_profile", "decision_context", "options", "book_id"), ("book", "context_condition"), "static"),
    "behaviorchain": CaseSchema("behaviorchain", ("persona", "history", "task_mode", "chain_index", "chain_length", "prior_nodes", "current_context"), ("chain_length_bucket", "key_behavior_status", "task_mode"), "static_or_generation"),
    "alignx": CaseSchema("alignx", ("prompt", "chosen", "rejected", "variant"), ("test_variant", "user"), "static"),
    "humanllm": CaseSchema("humanllm", ("user_profile", "purchase_history", "candidates"), ("user", "history_length_bucket", "product_category_if_available"), "static"),
    "humanual": CaseSchema(
        "humanual",
        ("domain", "persona", "prompt", "target_user_id", "post_id", "turn_id"),
        ("domain", "user"),
        "static_generation_with_judges",
    ),
    # UserLM contains two different protocols.  Section 3 intrinsic probes are
    # single next-user-turn generations and do not have LiC task/shard fields;
    # Section 4 validates those fields conditionally in UserLMAdapter.
    "userlm": CaseSchema("userlm", ("intent", "variant"), ("source_task", "intent", "required_information_pattern"), "static_or_interactive", gold_optional=True),
    "tau_usi": CaseSchema("tau_usi", ("domain", "user_goal", "initial_state"), ("domain", "difficulty_bin"), "interactive", gold_optional=True),
    "mirrorbench": CaseSchema("mirrorbench", ("dataset", "dialogue_context", "assistant_config"), ("dataset", "conversation_length", "domain_or_task_label"), "interactive", gold_optional=True),
    "coser": CaseSchema("coser", ("characters", "plot", "scene_context"), ("in_domain_status", "book", "plot"), "multi_character", gold_optional=True),
    "sotopia": CaseSchema("sotopia", ("scenario", "agents", "goals"), ("scenario", "role_assignment", "goal", "relationship"), "multi_agent", gold_optional=True),
    "agentsense": CaseSchema("agentsense", ("scenario", "agents", "profiles", "private_information"), ("scenario_domain", "private_information_presence"), "multi_agent", gold_optional=True),
}


def _missing(mapping: Mapping[str, Any], required: Sequence[str]) -> list[str]:
    return [key for key in required if key not in mapping]


def probe_case(case: BenchmarkCase) -> None:
    from ..extensions.supplemental import BENCHMARK_IDS as SUPPLEMENTAL_IDS
    if case.benchmark_id in SUPPLEMENTAL_IDS:
        from ..extensions.supplemental.adapter import validate_case
        validate_case(case)
        return
    try:
        schema = SCHEMAS[case.benchmark_id]
    except KeyError as exc:
        raise ValidationError(f"no normalized schema registered for {case.benchmark_id!r}") from exc
    missing_input = _missing(case.input_data, schema.required_input_fields)
    if missing_input:
        raise ValidationError(f"{case.benchmark_id}/{case.case_id} missing input fields: {missing_input}")
    strata = case.metadata.get("strata")
    if not isinstance(strata, Mapping):
        raise ValidationError(f"{case.benchmark_id}/{case.case_id} requires metadata.strata")
    missing_strata = _missing(strata, schema.required_strata)
    if missing_strata:
        raise ValidationError(f"{case.benchmark_id}/{case.case_id} missing strata: {missing_strata}")
    if not schema.gold_optional and case.gold is None:
        raise ValidationError(f"{case.benchmark_id}/{case.case_id} requires a gold value")
    values = case.input_data
    options = values.get("options")
    if case.benchmark_id in {"social_r1", "lifechoices"}:
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
            raise ValidationError(f"{case.benchmark_id}/{case.case_id} requires an options array")
    if case.benchmark_id == "lifechoices" and len(options) != 4:
        raise ValidationError(f"{case.benchmark_id}/{case.case_id} requires exactly four options")
    if case.benchmark_id == "humanllm":
        candidates = values.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or len(candidates) != 20:
            raise ValidationError(f"humanllm/{case.case_id} requires exactly 20 candidates")
    if case.benchmark_id == "humanual":
        if values.get("domain") not in {"news", "book", "opinion", "politics", "chat", "email"}:
            raise ValidationError(f"humanual/{case.case_id} has an unsupported domain")
        prompt = values.get("prompt")
        if isinstance(prompt, (str, bytes)) or not isinstance(prompt, Sequence) or not prompt:
            raise ValidationError(f"humanual/{case.case_id} requires a nonempty prompt-message array")
        for index, message in enumerate(prompt):
            if not isinstance(message, Mapping):
                raise ValidationError(f"humanual/{case.case_id} prompt message #{index} must be an object")
            if not isinstance(message.get("content"), str):
                raise ValidationError(f"humanual/{case.case_id} prompt message #{index} requires string content")
            if not isinstance(message.get("role"), str) or not message.get("role", "").strip():
                raise ValidationError(f"humanual/{case.case_id} prompt message #{index} requires a source role")
    if case.benchmark_id == "behaviorchain":
        mode = values.get("task_mode")
        if mode in {"prediction", "multiple_choice"}:
            candidates = values.get("candidates")
            if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or len(candidates) != 4:
                raise ValidationError(
                    f"behaviorchain/{case.case_id} prediction requires exactly four candidates"
                )
        elif mode != "generation":
            raise ValidationError(
                f"behaviorchain/{case.case_id} task_mode must be prediction/multiple_choice or generation"
            )
    if case.benchmark_id == "alignx" and values.get("chosen") == values.get("rejected"):
        raise ValidationError(f"alignx/{case.case_id} chosen and rejected responses must differ")
    if case.benchmark_id == "coser" and len(values.get("characters") or ()) < 1:
        raise ValidationError(f"coser/{case.case_id} requires at least one character")
    if case.benchmark_id == "sotopia" and len(values.get("agents") or ()) != 2:
        raise ValidationError(f"sotopia/{case.case_id} requires exactly two agents")
    if case.benchmark_id == "agentsense" and not 2 <= len(values.get("agents") or ()) <= 5:
        raise ValidationError(f"agentsense/{case.case_id} requires between two and five agents")
