"""tau-USI task-oriented user simulation and distribution-level scoring.

The target model controls only the customer.  A separately identified fixed
assistant controls customer-service actions and a task environment owns tool
state.  Human references are evaluator-only derived inputs and are never
placed in either model prompt.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..contracts import (
    BenchmarkCase,
    CaseResult,
    ChatMessage,
    ErrorState,
    MetricValue,
    ModelRequest,
    ModelResponse,
    ResultStatus,
)
from ..data.schemas import probe_case
from ..environments.tau_usi import (
    ASSISTANT_ROLE,
    TAU_USER_STOP_TOKEN,
    USER_ROLE,
    TauAssistantAction,
    TauTaskEnvironment,
    TauTaskState,
    parse_assistant_response,
    parse_user_action,
)
from ..errors import (
    BackendError,
    BackendStructuredOutputError,
    BackendTimeoutError,
    ConfigurationError,
    EpisodeTokenBudgetExhausted,
    ParseError,
    ValidationError,
)
from ..interfaces import BenchmarkAdapter, ModelBackend
from ..json_utils import canonical_json
from ..integrations.tau_usi_protocol import (
    SURVEY_SCHEMA_REVISION, SURVEY_FIELD_NAMES, FIELD_ORDINAL,
    survey_prompt, parse_survey_options, fixture_survey_options, _parse_json_object, tool_call_text,
)
from ..registry import adapter
from .common import (
    combine_usage,
    corrective_retry_request,
    merge_system_instructions,
)


_USER_ACTION_CONTRACT = (
    "Return only one raw user utterance as a single line, with no JSON wrapper, role prefix, notes, or analysis. "
    f"When the task is complete, return {TAU_USER_STOP_TOKEN!r} as a standalone message without anything else."
)
_USER_BEHAVIOR_INSTRUCTION = (
    "Act as a realistic customer and send only one natural user message at a time. Reveal only information needed "
    "for the current step instead of giving away the entire private goal at once. Never invent information absent "
    "from your private context, and do not repeat the goal verbatim. Once the goal is satisfied, or after a "
    f"successful transfer to a human is confirmed, immediately output {TAU_USER_STOP_TOKEN!r}."
)
_ASSISTANT_ACTION_CONTRACT = "Use the text function-call format provided in the system prompt."
_SURVEY_CONTRACT = "Return a JSON object mapping each official survey field to exactly one listed option string."



FEATURE_DIMENSIONS: Mapping[str, tuple[str, ...]] = {
    "d1_communication": (
        "polite_turn_rate",
        "short_turn_rate",
        "dash_turn_rate",
        "ack_only_turn_rate",
        "verbosity_cv",
        "repeated_trigram_interaction_rate",
        "identity_confusion_interaction_rate",
    ),
    "d2_information": (
        "frontload_word_rate",
        "identifiers_per_turn",
        "words_per_turn",
        "opening_words",
    ),
    "d3_clarification": (
        "uncertainty_turn_rate",
        "certainty_turn_rate",
        "pushback_question_turn_rate",
        "clarification_question_turn_rate",
        "information_question_turn_rate",
    ),
    "d4_error_reaction": (
        "emotion_turn_rate",
        "accusation_turn_rate",
        "pivot_turn_rate",
    ),
}

SURVEY_SCALES: Mapping[str, int] = {
    "task_success": 4,
    "efficiency": 3,
    # Semantic desirability, not the UI position: both "too few" and "too many"
    # map to 0, while "about right" maps to 1.
    "question_amount": 1,
    "answer_effort": 2,
    "human_likeness": 2,
    "interaction_flow": 3,
    "overall": 4,
    "reuse_intent": 4,
}

OFFICIAL_BATCH_COUNT = 3
ECE_CUTOFFS = (0.2, 0.4, 0.6, 0.8)
FEATURE_EXTRACTOR_REVISION = "indieval-supplemental-tau-usi-feature-extractor-v1"
MAX_USER_TURNS = 60
ASSISTANT_STEP_LIMIT_POLICY = "yield_last_assistant_text_to_user-v2"


@dataclass(frozen=True)
class TauRuntimeProvenance:
    fixed_assistant_model: str
    fixed_assistant_revision: str
    assistant_policy_revision: str
    environment_revision: str
    tool_schema_revision: str
    max_user_turns: int = MAX_USER_TURNS
    max_assistant_steps_per_user_turn: int = 64
    request_timeout_seconds: float = 120.0
    max_retries: int = 2
    survey_required: bool = True
    token_accounting: str = "backend_total_tokens_else_whitespace_request_plus_response"
    source: str = "configured"

    def __post_init__(self) -> None:
        required = (
            self.fixed_assistant_model,
            self.fixed_assistant_revision,
            self.assistant_policy_revision,
            self.environment_revision,
            self.tool_schema_revision,
            self.token_accounting,
            self.source,
        )
        if not all(required):
            raise ConfigurationError("tau-USI runtime provenance fields cannot be empty")
        if self.max_user_turns <= 0 or self.max_assistant_steps_per_user_turn <= 0:
            raise ConfigurationError("tau-USI turn and assistant-step limits must be positive")
        if self.max_user_turns > MAX_USER_TURNS:
            raise ConfigurationError(
                f"tau-USI max_user_turns cannot exceed the project protocol cap of {MAX_USER_TURNS}"
            )
        if (
            self.request_timeout_seconds <= 0
            or not math.isfinite(self.request_timeout_seconds)
            or self.max_retries < 0
        ):
            raise ConfigurationError("tau-USI timeout must be positive and retries nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed_assistant_model": self.fixed_assistant_model,
            "fixed_assistant_revision": self.fixed_assistant_revision,
            "assistant_policy_revision": self.assistant_policy_revision,
            "environment_revision": self.environment_revision,
            "tool_schema_revision": self.tool_schema_revision,
            "max_user_turns": self.max_user_turns,
            "max_assistant_steps_per_user_turn": self.max_assistant_steps_per_user_turn,
            "assistant_step_limit_policy": ASSISTANT_STEP_LIMIT_POLICY,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_retries": self.max_retries,
            "survey_required": self.survey_required,
            "token_accounting": self.token_accounting,
            "source": self.source,
        }


@dataclass(frozen=True)
class TauScoringProvenance:
    annotation_revision: str
    feature_extractor_revision: str
    survey_schema_revision: str
    difficulty_revision: str
    human_batch_ids: tuple[str, ...]
    expected_task_count: int
    source: str = "configured"
    protocol_status: str = "official_equations_with_configured_derived_inputs"

    def __post_init__(self) -> None:
        required = (
            self.annotation_revision,
            self.feature_extractor_revision,
            self.survey_schema_revision,
            self.difficulty_revision,
            self.source,
            self.protocol_status,
        )
        if not all(required):
            raise ConfigurationError("tau-USI scoring provenance fields cannot be empty")
        if len(self.human_batch_ids) != OFFICIAL_BATCH_COUNT or len(set(self.human_batch_ids)) != OFFICIAL_BATCH_COUNT:
            raise ConfigurationError("tau-USI requires exactly three distinct human annotation batch IDs")
        if self.expected_task_count <= 0:
            raise ConfigurationError("tau-USI expected_task_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_revision": self.annotation_revision,
            "feature_extractor_revision": self.feature_extractor_revision,
            "survey_schema_revision": self.survey_schema_revision,
            "missing_survey_policy": "supplemental-uniform-ordinal-numpy-rng42-v1",
            "difficulty_revision": self.difficulty_revision,
            "human_batch_ids": list(self.human_batch_ids),
            "expected_task_count": self.expected_task_count,
            "source": self.source,
            "protocol_status": self.protocol_status,
        }


_AGENT_MARKUP = re.compile(
    r"<\|think\|>.*?<\|/think\|>|<function=.*?(?:</function>|$)|<\|tool\|>.*?<\|/tool\|>",
    re.DOTALL,
)
_POLITE = re.compile(
    r"\b(please|thank you|thanks|sorry|excuse me|appreciate|grateful|"
    r"kind of you|wonderful|courteous|considerate|i understand|"
    r"no worries|my apologies|pardon)\b"
)
_DASH = re.compile(r"[\u2014\u2013]")
_ACK_ONLY = re.compile(
    r"^(yes|yeah|yea|yep|yup|ok|okay|okey|sure|right|fine|great|perfect|"
    r"awesome|cool|agree|agreed|correct|absolutely|indeed|alright|"
    r"got it|go ahead|sounds good|that works|proceed|confirm|do it|"
    r"no problem|thank you|thanks|please|mhm|uh-huh|aight)[\s!.\-,]*$"
)
_IDENTIFIER = re.compile(
    r"(\b[a-z0-9]{6,}\b|#?\w*\d{5,}\w*|"
    r"\w+_\w+_\d{3,}|"
    r"gift_card_\w+|credit_card_\w+|paypal_\w+|"
    r"\w+@\w+\.\w+)"
)
_UNCERTAIN = re.compile(
    r"\b(i think|not sure|i don'?t remember|i don'?t recall|i believe|probably|"
    r"maybe|i forgot|don'?t know|i guess|i'?m not sure|if i recall|"
    r"i don'?t have|can'?t recall|can'?t find|i'?m unsure|i might|"
    r"perhaps|apparently|sort of|kind of|somewhat|partly|possibly|"
    r"vaguely|roughly|hesitant|doubtful|confused|uncertain|dunno)\b"
)
_CERTAIN = re.compile(
    r"\b(definitely|absolutely|certainly|exactly|precisely|clearly|always|"
    r"never|completely|entirely|fully|obvious|obviously|undoubtedly|"
    r"without a doubt|for sure|of course|guaranteed|no doubt)\b"
)
_PUSHBACK = re.compile(
    r"\b(are you sure|that'?s (not right|wrong|incorrect|not what)|"
    r"i already (told|said|gave|provided|mentioned)|you already (asked|have)|"
    r"try again|check again|that can'?t be (right|correct)|"
    r"that'?s (crazy|nonsense|ridiculous|absurd)|this is (not fair|nonsense)|"
    r"why (can'?t|won'?t|didn'?t|isn'?t|aren'?t|would)|"
    r"doesn'?t (seem|sound|look|make sense)|not what i (asked|said|meant|wanted)|"
    r"do(n'?t| not) ask me .* again|i would never)\b"
)
_CLARIFICATION = re.compile(
    r"\b(what do you mean|what does that mean|what did you mean|"
    r"could you (clarify|elaborate|specify)|can you (explain|clarify|elaborate)|"
    r"i don'?t understand|i dont understand|"
    r"what'?s the difference|whats the difference|"
    r"what exactly|what specifically|"
    r"which (one|reservation|order|flight|option|plan|account|item|product) (is|was|do|should|did|would)|"
    r"i'?m not sure which|im not sure which|which is which|"
    r"you (said|mentioned|told me|wrote|indicated) .{0,30}\?|"
    r"that mean|what does that (involve|include|entail|look like)|"
    r"(sorry|wait),? (what|which|how|i don'?t)|"
    r"can you (be more specific|give me more detail|break that down|walk me through)|"
    r"i'?m (confused|lost)|im (confused|lost)|"
    r"(how|what) (exactly|specifically) (does|do|is|are|would|will|should)|"
    r"could you (repeat|say) that|can you (repeat|say) that|"
    r"what (is|are) the (options|choices|alternatives|details)|"
    r"i need (more|some) (info|information|details|clarification)|"
    r"(so|wait),? (you'?re saying|does that mean|is that)|"
    r"meaning\??|how so\??|in what (way|sense)\??)\b"
)
_INFO_QUESTION = re.compile(
    r"\b(what is (the|my)|what'?s (the|my)|how much|how many|"
    r"when (will|does|is|can|did)|where (is|are|was|can)|"
    r"how (do|can|would|should) i|what are (the|my)|"
    r"can you (check|tell|find|look|see|show|help|list)|"
    r"do you (have|know|see)|is (there|it|that) (a |any )?|"
    r"what options|how long|what.s the (status|price|cost|total|balance))\b"
)
_EMOTION = re.compile(
    r"\b(frustrated|upset|angry|confused|worried|annoyed|furious|"
    r"ridiculous|terrible|horrible|awful|ugh|"
    r"stressed|anxious|nervous|uncomfortable|panic|disappointed|disappointing|"
    r"irritated|irritating|aggravated|outraged|outrageous|exasperated|"
    r"bothered|disgusted|miserable|devastated|heartbroken|desperate)\b"
)
_ACCUSATION = re.compile(
    r"\b(blame|fault|ridiculous|wrong|useless|incompetent|failure|"
    r"misleading|irresponsible|unacceptable|disgrace|insult|"
    r"scam|shame|stupid|trouble|disappointing|ruined)\b"
)
_PIVOT = re.compile(
    r"\b((wait|actually) (can|let|could|would)|i('?d| would) (like|prefer|rather)|"
    r"(can you|can we|could you) just|"
    r"instead|rather than|how about|what about|what if|"
    r"on second thought|"
    r"is there (a |any |another )?way to|"
    r"(let'?s|can we|lets) (try|do|go with|switch|change)|"
    r"maybe (we|i|you) (should|could|can)|"
    r"(change|switch|try) (it |that )?(to|something|a different))\b"
)
_IDENTITY_CONFUSION = re.compile(
    r"\b(how can i (help|assist)|how may i (help|assist)|"
    r"what can i (do|help|assist) (for|with)|"
    r"do you (want|require|wish|prefer)|would you (like|prefer)|"
    r"let me (check|look|verify|find|pull up|search|assist|help)|"
    r"i('?ll| will) (check|look into|verify|process|handle|assist|help)|"
    r"i('?m| am) (happy|glad|here) to (help|assist)|"
    r"is there anything else i can|"
    r"thank you for (calling|contacting|reaching|choosing|your patience)|"
    r"for (security|verification) (purposes|reasons)|"
    r"(may|could) i (verify)|"
    r"allow me to|permit me to|"
    r"your (order|account|reservation|booking|request|ticket|case|inquiry)|"
    r"i (see|understand) (that|your)|"
    r"according to (our|the) (records|system|policy)|"
    r"(our|the) (policy|system|records) (shows?|indicates?)|"
    r"i (can|could) (offer|suggest|recommend)|"
    r"(have you|did you) (tried?|considered?)|"
    r"i apologize for (the|any) (inconvenience|confusion|delay|trouble))\b"
)


def _words(text: str) -> tuple[str, ...]:
    cleaned = _AGENT_MARKUP.sub("", text)
    return tuple(token for token in cleaned.split() if token)


def _rate(flags: Iterable[bool]) -> float:
    values = tuple(flags)
    return sum(values) / len(values) if values else 0.0


def extract_behavior_features(user_messages: Sequence[str]) -> dict[str, float]:
    """Compute the Supplemental-compatible tau-USI behavioral feature vector."""

    if not user_messages or not all(isinstance(message, str) and message.strip() for message in user_messages):
        raise ValidationError("tau-USI behavior extraction requires non-empty user messages")
    turns = tuple(_AGENT_MARKUP.sub("", message.strip()) for message in user_messages)
    tokenized = tuple(_words(message) for message in turns)
    counts = tuple(len(tokens) for tokens in tokenized)
    total_words = sum(counts)
    mean_words = total_words / len(turns)
    verbosity_cv = statistics.pstdev(counts) / mean_words if mean_words else 0.0
    trigram_counts: Counter[tuple[str, str, str]] = Counter()
    for tokens in tokenized:
        lowered = tuple(token.casefold() for token in tokens)
        trigram_counts.update(tuple(lowered[index : index + 3]) for index in range(max(0, len(lowered) - 2)))
    lowered = tuple(message.lower() for message in turns)

    question_kinds = []
    for message in lowered:
        if _PUSHBACK.search(message):
            question_kinds.append("pushback")
        elif _CLARIFICATION.search(message):
            question_kinds.append("clarification")
        elif _INFO_QUESTION.search(message):
            question_kinds.append("information")
        else:
            question_kinds.append("none")

    return {
        "words_per_turn": mean_words,
        "short_turn_rate": _rate(count <= 3 for count in counts),
        "polite_turn_rate": _rate(bool(_POLITE.search(message)) for message in lowered),
        "dash_turn_rate": _rate(bool(_DASH.search(message)) for message in turns),
        "ack_only_turn_rate": _rate(bool(_ACK_ONLY.match(message.strip())) for message in lowered),
        "verbosity_cv": verbosity_cv,
        "repeated_trigram_interaction_rate": float(any(count > 10 for count in trigram_counts.values())),
        "identity_confusion_interaction_rate": float(any(_IDENTITY_CONFUSION.search(message) for message in lowered)),
        "frontload_word_rate": sum(counts[:2]) / total_words if total_words else 0.0,
        "identifiers_per_turn": sum(len(_IDENTIFIER.findall(message)) for message in lowered) / len(turns),
        "opening_words": float(counts[0]),
        "uncertainty_turn_rate": _rate(bool(_UNCERTAIN.search(message)) for message in lowered),
        "certainty_turn_rate": _rate(bool(_CERTAIN.search(message)) for message in lowered),
        "pushback_question_turn_rate": _rate(kind == "pushback" for kind in question_kinds),
        "clarification_question_turn_rate": _rate(kind == "clarification" for kind in question_kinds),
        "information_question_turn_rate": _rate(kind == "information" for kind in question_kinds),
        "emotion_turn_rate": _rate(bool(_EMOTION.search(message)) for message in lowered),
        "accusation_turn_rate": _rate(bool(_ACCUSATION.search(message)) for message in lowered),
        "pivot_turn_rate": _rate(bool(_PIVOT.search(message)) for message in lowered),
    }


def dice_alignment(model_value: float, human_value: float) -> float:
    if not (math.isfinite(model_value) and math.isfinite(human_value)):
        raise ValidationError("tau-USI Dice inputs must be finite")
    if model_value < 0 or human_value < 0:
        raise ValidationError("tau-USI Dice inputs cannot be negative")
    if model_value == 0 and human_value == 0:
        return 100.0
    return 2.0 * min(model_value, human_value) / (model_value + human_value) * 100.0


def difficulty_bin(score: float) -> int:
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValidationError("tau-USI difficulty score must be finite numeric")
    value = float(score)
    if not 0 <= value <= 1:
        raise ValidationError("tau-USI difficulty score must be in [0,1]")
    return sum(value >= cutoff for cutoff in ECE_CUTOFFS)


def expected_calibration_error(rows: Sequence[tuple[float, float, float]]) -> float:
    """Return five-bin ECE from (difficulty, simulator_success, human_success)."""

    if not rows:
        raise ValidationError("tau-USI ECE requires at least one paired task")
    bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for difficulty, simulator, human in rows:
        for label, value in (("simulator success", simulator), ("human success", human)):
            if isinstance(value, bool):
                value = float(value)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValidationError(f"tau-USI {label} must be in [0,1]")
        bins[difficulty_bin(float(difficulty))].append((float(simulator), float(human)))
    total = len(rows)
    return sum(
        len(values) / total
        * abs(sum(sim for sim, _ in values) / len(values) - sum(human for _, human in values) / len(values))
        for values in bins.values()
    )


def normalize_survey(raw: Mapping[str, Any]) -> dict[str, float]:
    if set(raw) != set(SURVEY_SCALES):
        raise ParseError(
            "tau-USI survey fields differ; "
            f"missing={sorted(set(SURVEY_SCALES)-set(raw))}, extra={sorted(set(raw)-set(SURVEY_SCALES))}"
        )
    normalized: dict[str, float] = {}
    for name, maximum in SURVEY_SCALES.items():
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
            raise ParseError(f"tau-USI survey {name} must be an integer from 0 to {maximum}")
        normalized[name] = value / maximum
    return normalized


def evaluative_alignment(
    simulator_surveys: Sequence[Mapping[str, float]], human_surveys: Sequence[Mapping[str, float]]
) -> tuple[float, Mapping[str, float]]:
    if not simulator_surveys or len(simulator_surveys) != len(human_surveys):
        raise ValidationError("tau-USI Eval requires equally sized non-empty paired surveys")
    per_dimension: dict[str, float] = {}
    for name in SURVEY_SCALES:
        differences = []
        for simulator, human in zip(simulator_surveys, human_surveys):
            if name not in simulator or name not in human:
                raise ValidationError(f"tau-USI Eval missing survey field {name}")
            differences.append(abs(float(simulator[name]) - float(human[name])))
        per_dimension[name] = sum(differences) / len(differences)
    mae = sum(per_dimension.values()) / len(per_dimension)
    return (1.0 - mae) * 100.0, per_dimension


def user_sim_index(
    d1: float, d2: float, d3: float, d4: float, ece: float, evaluation: float | None = None
) -> float:
    components = [float(d1), float(d2), float(d3), float(d4), (1.0 - float(ece)) * 100.0]
    if evaluation is not None:
        components.append(float(evaluation))
    return sum(components) / len(components)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _mean_vectors(vectors: Sequence[Mapping[str, Any]], names: Sequence[str]) -> dict[str, float] | None:
    if not vectors:
        return None
    result: dict[str, float] = {}
    for name in names:
        values: list[float] = []
        for vector in vectors:
            value = vector.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return None
            values.append(float(value))
        result[name] = _mean(values)
    return result


def _batch_metric(
    name: str,
    per_batch: Mapping[str, float],
    *,
    direction: str = "higher_is_better",
    unit: str,
    metadata: Mapping[str, Any],
) -> MetricValue:
    values = list(per_batch.values())
    if len(values) != OFFICIAL_BATCH_COUNT:
        return MetricValue(
            name,
            None,
            direction=direction,
            unit=unit,
            metadata={
                **dict(metadata),
                "availability": "unavailable",
                "reason": "all three human annotation batches are required",
                "available_batches": sorted(per_batch),
            },
        )
    return MetricValue(
        name,
        _mean(values),
        direction=direction,
        unit=unit,
        uncertainty={
            "kind": "population_std_across_three_human_batches",
            "std": statistics.pstdev(values),
            "batch_count": OFFICIAL_BATCH_COUNT,
        },
        metadata={**dict(metadata), "per_batch": dict(per_batch), "aggregation": "mean_across_three_batches"},
    )


class TauUSIScorer:
    scorer_revision = "tau-usi-supplemental-compatible-dice-ece-survey-usi-v5-text-options-rng42"

    @staticmethod
    def _base_metrics(results: Sequence[CaseResult], *, complete: bool, reason: str | None) -> dict[str, MetricValue]:
        completed = sum(result.status == ResultStatus.COMPLETED for result in results)
        failed = sum(result.status == ResultStatus.FAILED for result in results)
        assistant_limited = sum(
            any(event.kind == "budget_limit" and event.actor == ASSISTANT_ROLE for event in result.trace)
            for result in results
        )
        tau_payloads = [TauUSIScorer._tau_payload(result) for result in results]
        stop_compliance = [
            float(value)
            for value in (payload.get("stop_compliance") for payload in tau_payloads)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        strict_success = [
            float(value)
            for value in (payload.get("strict_episode_success") for payload in tau_payloads)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        return {
            "tau_usi.assistant_step_limit_rate": MetricValue(
                "tau_usi.assistant_step_limit_rate",
                assistant_limited / len(results) if results else None,
                direction="lower_is_better",
                unit="0_to_1",
                numerator=assistant_limited,
                denominator=len(results),
                metadata={
                    "official_primary": False,
                    "eligibility": "all selected case results, including post-limit evaluation failures",
                    "policy_revision": ASSISTANT_STEP_LIMIT_POLICY,
                },
            ),
            "tau_usi.case_completion_rate": MetricValue(
                "tau_usi.case_completion_rate",
                completed / len(results) if results else None,
                numerator=completed,
                denominator=len(results),
            ),
            "tau_usi.case_failure_count": MetricValue(
                "tau_usi.case_failure_count",
                failed,
                direction="lower_is_better",
                unit="cases",
                numerator=failed,
                denominator=len(results),
            ),
            "tau_usi.suite_complete": MetricValue(
                "tau_usi.suite_complete",
                complete,
                unit="boolean",
                metadata={"reason": reason},
            ),
            "tau_usi.stop_compliance_rate": MetricValue(
                "tau_usi.stop_compliance_rate",
                _mean(stop_compliance) if stop_compliance else None,
                unit="0_to_1",
                numerator=sum(stop_compliance),
                denominator=len(stop_compliance),
                metadata={
                    "availability": "available" if stop_compliance else "unavailable",
                    "eligibility": "episodes ending by user_stop or user_turn_limit",
                    "official_primary": False,
                    "does_not_modify_environment_reward_or_usi": True,
                },
            ),
            "tau_usi.diagnostic.strict_episode_success_rate": MetricValue(
                "tau_usi.diagnostic.strict_episode_success_rate",
                _mean(strict_success) if strict_success else None,
                unit="0_to_1",
                numerator=sum(strict_success),
                denominator=len(strict_success),
                metadata={
                    "availability": "available" if strict_success else "unavailable",
                    "formula": "environment_reward * stop_compliance",
                    "official_primary": False,
                    "nonofficial_diagnostic": True,
                    "does_not_modify_ece_outcome_alignment_or_usi": True,
                },
            ),
        }

    @staticmethod
    def _unavailable_distribution_metrics(reason: str) -> dict[str, MetricValue]:
        result: dict[str, MetricValue] = {}
        for dimension in FEATURE_DIMENSIONS:
            result[f"tau_usi.{dimension}"] = MetricValue(
                f"tau_usi.{dimension}", None, unit="score_0_to_100", metadata={"availability": "unavailable", "reason": reason}
            )
        for feature in sorted({feature for names in FEATURE_DIMENSIONS.values() for feature in names}):
            result[f"tau_usi.feature_alignment.{feature}"] = MetricValue(
                f"tau_usi.feature_alignment.{feature}",
                None,
                unit="score_0_to_100",
                metadata={"availability": "unavailable", "reason": reason},
            )
        for name, direction, unit in (
            ("tau_usi.ece", "lower_is_better", "0_to_1"),
            ("tau_usi.outcome_alignment", "higher_is_better", "score_0_to_100"),
            ("tau_usi.eval", "higher_is_better", "score_0_to_100"),
            ("tau_usi.usi", "higher_is_better", "score_0_to_100"),
            ("tau_usi.usi_without_eval", "higher_is_better", "score_0_to_100"),
        ):
            result[name] = MetricValue(
                name, None, direction=direction, unit=unit, metadata={"availability": "unavailable", "reason": reason}
            )
        return result

    @staticmethod
    def _tau_payload(result: CaseResult) -> Mapping[str, Any]:
        payload = result.metadata.get("tau_usi")
        return payload if isinstance(payload, Mapping) else {}

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        if not results:
            reason = "no case results"
            return {**self._base_metrics(results, complete=False, reason=reason), **self._unavailable_distribution_metrics(reason)}

        provenance_values = [self._tau_payload(result).get("scoring_provenance") for result in results]
        provenance = next((value for value in provenance_values if isinstance(value, Mapping)), None)
        if provenance is None:
            reason = "scoring provenance is absent"
            return {**self._base_metrics(results, complete=False, reason=reason), **self._unavailable_distribution_metrics(reason)}
        if any(value != provenance for value in provenance_values):
            reason = "case results have incompatible scoring provenance"
            return {**self._base_metrics(results, complete=False, reason=reason), **self._unavailable_distribution_metrics(reason)}
        expected_count = provenance.get("expected_task_count")
        batch_ids = tuple(provenance.get("human_batch_ids") or ())
        grouped: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            grouped[result.case_id].append(result)
        incomplete_reasons = []
        if not isinstance(expected_count, int) or expected_count <= 0:
            incomplete_reasons.append("invalid expected task count")
        elif len(grouped) != expected_count:
            incomplete_reasons.append(f"expected {expected_count} tasks but observed {len(grouped)}")
        if len(batch_ids) != OFFICIAL_BATCH_COUNT or len(set(batch_ids)) != OFFICIAL_BATCH_COUNT:
            incomplete_reasons.append("exactly three declared human batches are required")
        if any(result.status != ResultStatus.COMPLETED for result in results):
            incomplete_reasons.append("one or more task rollouts failed or were skipped")
        if any(not bool(self._tau_payload(result).get("episode_complete")) for result in results):
            incomplete_reasons.append("one or more episodes are incomplete")
        suite_complete = not incomplete_reasons
        reason = "; ".join(incomplete_reasons) if incomplete_reasons else None
        metrics = self._base_metrics(results, complete=suite_complete, reason=reason)
        if not suite_complete:
            metrics.update(self._unavailable_distribution_metrics(reason or "incomplete suite"))
            return metrics

        all_features = tuple(sorted({feature for names in FEATURE_DIMENSIONS.values() for feature in names}))
        simulator_by_task: dict[str, Mapping[str, float]] = {}
        simulator_survey_by_task: dict[str, list[Mapping[str, float]]] = {}
        simulator_reward_by_task: dict[str, float] = {}
        difficulty_by_task: dict[str, float] = {}
        references_by_task: dict[str, Mapping[str, Mapping[str, Any]]] = {}
        for case_id, repetitions in grouped.items():
            repetitions.sort(key=lambda result: result.repetition)
            payloads = [self._tau_payload(result) for result in repetitions]
            feature_vector = _mean_vectors(
                [value for value in (payload.get("simulator_features") for payload in payloads) if isinstance(value, Mapping)],
                all_features,
            )
            if feature_vector is not None and len(payloads) == len(
                [value for value in (payload.get("simulator_features") for payload in payloads) if isinstance(value, Mapping)]
            ):
                simulator_by_task[case_id] = feature_vector
            survey_values = [payload.get("simulator_survey") for payload in payloads]
            if all(isinstance(value, Mapping) for value in survey_values):
                # Complete fields are averaged only after per-repetition scoring
                # below. Keep partial answers separate for official imputation.
                simulator_survey_by_task[case_id] = survey_values
            rewards = [payload.get("environment_reward") for payload in payloads]
            if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in rewards):
                simulator_reward_by_task[case_id] = _mean(
                    [1.0 if float(value) > 0.0 else 0.0 for value in rewards]
                )
            difficulty_values = [payload.get("difficulty_score") for payload in payloads]
            if difficulty_values and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in difficulty_values
            ) and len({float(value) for value in difficulty_values}) == 1:
                difficulty_by_task[case_id] = float(difficulty_values[0])
            reference_values = [payload.get("human_references") for payload in payloads]
            if reference_values and all(isinstance(value, Mapping) for value in reference_values) and all(
                value == reference_values[0] for value in reference_values[1:]
            ):
                references_by_task[case_id] = reference_values[0]  # type: ignore[assignment]

        simulator_population = _mean_vectors(list(simulator_by_task.values()), all_features)
        feature_per_batch: dict[str, dict[str, float]] = {}
        dimension_per_batch: dict[str, dict[str, float]] = {name: {} for name in FEATURE_DIMENSIONS}
        for batch_id in batch_ids:
            human_vectors = []
            for case_id in grouped:
                reference = references_by_task.get(case_id, {}).get(batch_id)
                features = reference.get("features") if isinstance(reference, Mapping) else None
                if isinstance(features, Mapping):
                    human_vectors.append(features)
            human_population = _mean_vectors(human_vectors, all_features) if len(human_vectors) == len(grouped) else None
            if simulator_population is None or human_population is None:
                continue
            alignments = {
                feature: dice_alignment(simulator_population[feature], human_population[feature])
                for feature in all_features
            }
            feature_per_batch[batch_id] = alignments
            for dimension, names in FEATURE_DIMENSIONS.items():
                dimension_per_batch[dimension][batch_id] = _mean([alignments[name] for name in names])

        for feature in all_features:
            per_batch = {
                batch_id: values[feature]
                for batch_id, values in feature_per_batch.items()
                if feature in values
            }
            dimension_names = [name for name, features in FEATURE_DIMENSIONS.items() if feature in features]
            metrics[f"tau_usi.feature_alignment.{feature}"] = _batch_metric(
                f"tau_usi.feature_alignment.{feature}",
                per_batch,
                unit="score_0_to_100",
                metadata={
                    "formula": "2*min(M,H)/(M+H)*100; both zero => 100",
                    "dimensions": dimension_names,
                    "population_level": True,
                    "feature_extractor_revision": provenance.get("feature_extractor_revision"),
                },
            )
        for dimension, per_batch in dimension_per_batch.items():
            metrics[f"tau_usi.{dimension}"] = _batch_metric(
                f"tau_usi.{dimension}",
                per_batch,
                unit="score_0_to_100",
                metadata={
                    "formula": "arithmetic mean of constituent feature Dice scores",
                    "features": list(FEATURE_DIMENSIONS[dimension]),
                    "population_level": True,
                },
            )

        ece_per_batch: dict[str, float] = {}
        for batch_id in batch_ids:
            rows = []
            for case_id in grouped:
                reference = references_by_task.get(case_id, {}).get(batch_id)
                human_reward = reference.get("reward") if isinstance(reference, Mapping) else None
                if (
                    case_id in difficulty_by_task
                    and case_id in simulator_reward_by_task
                    and isinstance(human_reward, (int, float))
                    and not isinstance(human_reward, bool)
                ):
                    rows.append(
                        (
                            difficulty_by_task[case_id],
                            simulator_reward_by_task[case_id],
                            1.0 if float(human_reward) > 0.0 else 0.0,
                        )
                    )
            if len(rows) == len(grouped):
                ece_per_batch[batch_id] = expected_calibration_error(rows)
        metrics["tau_usi.ece"] = _batch_metric(
            "tau_usi.ece",
            ece_per_batch,
            direction="lower_is_better",
            unit="0_to_1",
            metadata={
                "formula": "sum_b |S_b|/N * abs(mean_sim_success_b - mean_human_success_b)",
                "bins": [0.0, *ECE_CUTOFFS, 1.0],
                "difficulty_revision": provenance.get("difficulty_revision"),
                "difficulty_is_fixed_independently_of_current_scoring_run": True,
            },
        )
        outcome_per_batch = {batch_id: (1.0 - value) * 100.0 for batch_id, value in ece_per_batch.items()}
        metrics["tau_usi.outcome_alignment"] = _batch_metric(
            "tau_usi.outcome_alignment",
            outcome_per_batch,
            unit="score_0_to_100",
            metadata={"formula": "(1-ECE)*100"},
        )

        eval_per_batch: dict[str, float] = {}
        eval_dimension_mae: dict[str, Mapping[str, float]] = {}
        # One RNG stream across batches, tasks sorted by official task key, then
        # fields in FIELD_ORDINAL order. This matches Supplemental's scorer exactly.
        import numpy as np
        rng = np.random.default_rng(42)
        imputed_counts: dict[str, int] = {}
        allow_imputation = provenance.get("missing_survey_policy") == "supplemental-uniform-ordinal-numpy-rng42-v1"
        ordered_cases = sorted(grouped, key=lambda key: self._tau_payload(grouped[key][0]).get("task_key", key))
        for batch_id in batch_ids:
            task_scores = []
            field_differences: dict[str, list[float]] = {name: [] for name in SURVEY_SCALES}
            imputed_counts[batch_id] = 0
            for case_id in ordered_cases:
                reference = references_by_task.get(case_id, {}).get(batch_id)
                human_survey = reference.get("survey") if isinstance(reference, Mapping) else None
                if case_id not in simulator_survey_by_task or not isinstance(human_survey, Mapping):
                    continue
                field_values = {name: [] for name in SURVEY_SCALES}
                for simulator in simulator_survey_by_task[case_id]:
                    for official_name, options in FIELD_ORDINAL.items():
                        name = SURVEY_FIELD_NAMES[official_name]
                        hv = human_survey.get(name)
                        if hv is None:
                            continue
                        lv = simulator.get(name)
                        if lv is None:
                            if not allow_imputation:
                                continue
                            values = sorted(set(options.values()))
                            lv = (rng.choice(values) - min(values)) / (max(values) - min(values))
                            imputed_counts[batch_id] += 1
                        field_values[name].append(float(lv))
                # Preserve the framework's existing across-repetition pooling.
                differences = []
                expected_repetitions = len(simulator_survey_by_task[case_id])
                for name, values in field_values.items():
                    if name in human_survey and len(values) == expected_repetitions:
                        difference = abs(float(human_survey[name]) - _mean(values))
                        differences.append(difference)
                        field_differences[name].append(difference)
                expected_fields = sum(name in human_survey for name in SURVEY_SCALES)
                if differences and len(differences) == expected_fields:
                    task_scores.append((1.0 - _mean(differences)) * 100)
            if len(task_scores) == len(grouped):
                eval_per_batch[batch_id] = _mean(task_scores)
                eval_dimension_mae[batch_id] = {name: _mean(values) for name, values in field_differences.items() if values}
        metrics["tau_usi.eval"] = _batch_metric(
            "tau_usi.eval",
            eval_per_batch,
            unit="score_0_to_100",
            metadata={
                "formula": "(1 - mean absolute error over paired tasks and eight normalized survey fields)*100",
                "survey_fields": list(SURVEY_SCALES),
                "survey_schema_revision": provenance.get("survey_schema_revision"),
                "per_batch_field_mae": eval_dimension_mae,
                "missing_answer_policy": provenance.get("missing_survey_policy"),
                "imputed_field_count_per_batch": imputed_counts,
                "imputation_seed": 42,
                "raw_answers_unchanged": True,
            },
        )

        usi_without_eval: dict[str, float] = {}
        usi: dict[str, float] = {}
        # Supplemental first aggregates Eval across the three annotation batches,
        # then uses that shared mean in each batch-specific USI calculation.
        # This leaves the USI mean unchanged but is necessary to reproduce its
        # reported across-batch population standard deviation.
        eval_mean = _mean(list(eval_per_batch.values())) if len(eval_per_batch) == OFFICIAL_BATCH_COUNT else None
        for batch_id in batch_ids:
            components = [dimension_per_batch[name].get(batch_id) for name in FEATURE_DIMENSIONS]
            if all(value is not None for value in components) and batch_id in ece_per_batch:
                d1, d2, d3, d4 = (float(value) for value in components)
                usi_without_eval[batch_id] = user_sim_index(d1, d2, d3, d4, ece_per_batch[batch_id])
                if eval_mean is not None:
                    usi[batch_id] = user_sim_index(
                        d1, d2, d3, d4, ece_per_batch[batch_id], eval_mean
                    )
        metrics["tau_usi.usi_without_eval"] = _batch_metric(
            "tau_usi.usi_without_eval",
            usi_without_eval,
            unit="score_0_to_100",
            metadata={
                "formula": "(D1+D2+D3+D4+(1-ECE)*100)/5",
                "protocol_status": "paper-compatible five-component variant when survey evaluation is unavailable",
            },
        )
        metrics["tau_usi.usi"] = _batch_metric(
            "tau_usi.usi",
            usi,
            unit="score_0_to_100",
            metadata={
                "formula": "(D1+D2+D3+D4+(1-ECE)*100+Eval)/6",
                "eval_component": "mean_eval_across_three_human_batches_as_in_supplemental",
                "summary_not_substitute_for_components": True,
            },
        )
        return metrics


@adapter("tau_usi")
class TauUSIAdapter(BenchmarkAdapter):
    benchmark_id = "tau_usi"
    prompt_revision = "tau-usi-supplemental-text-tools-original-history-survey-v10"
    scorer_revision = TauUSIScorer.scorer_revision

    def __init__(
        self,
        *,
        runtime_provenance: TauRuntimeProvenance | None = None,
        scoring_provenance: TauScoringProvenance | None = None,
        feature_extractor: Callable[[Sequence[str]], Mapping[str, float]] = extract_behavior_features,
        feature_extractor_revision: str = FEATURE_EXTRACTOR_REVISION,
        human_reference_resolver: Callable[[BenchmarkCase], Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        self.environment = TauTaskEnvironment()
        self.scorer = TauUSIScorer()
        self._runtime_provenance = runtime_provenance
        self._scoring_provenance = scoring_provenance
        self._feature_extractor = feature_extractor
        self._feature_extractor_revision = feature_extractor_revision
        self._human_reference_resolver = human_reference_resolver

    @staticmethod
    def _metadata_mapping(case: BenchmarkCase, key: str) -> Mapping[str, Any] | None:
        direct = case.metadata.get(key)
        if isinstance(direct, Mapping):
            return direct
        replay = case.metadata.get("replay")
        value = replay.get(key) if isinstance(replay, Mapping) else None
        return value if isinstance(value, Mapping) else None

    def runtime_for_case(self, case: BenchmarkCase) -> TauRuntimeProvenance:
        if self._runtime_provenance is not None:
            return self._runtime_provenance
        raw = self._metadata_mapping(case, "runtime_provenance")
        if raw is None:
            raise ConfigurationError("tau-USI requires fixed-assistant and environment runtime provenance")
        return TauRuntimeProvenance(
            fixed_assistant_model=str(raw.get("fixed_assistant_model") or ""),
            fixed_assistant_revision=str(raw.get("fixed_assistant_revision") or ""),
            assistant_policy_revision=str(raw.get("assistant_policy_revision") or ""),
            environment_revision=str(raw.get("environment_revision") or ""),
            tool_schema_revision=str(raw.get("tool_schema_revision") or ""),
            max_user_turns=min(int(raw.get("max_user_turns", MAX_USER_TURNS)), MAX_USER_TURNS),
            max_assistant_steps_per_user_turn=int(raw.get("max_assistant_steps_per_user_turn", 64)),
            request_timeout_seconds=float(raw.get("request_timeout_seconds", 120.0)),
            max_retries=int(raw.get("max_retries", 2)),
            survey_required=bool(raw.get("survey_required", True)),
            token_accounting=str(
                raw.get("token_accounting")
                or "backend_total_tokens_else_whitespace_request_plus_response"
            ),
            source=str(raw.get("source") or "configured"),
        )

    def provenance_for_case(self, case: BenchmarkCase) -> TauScoringProvenance:
        if self._scoring_provenance is not None:
            provenance = self._scoring_provenance
        else:
            raw = self._metadata_mapping(case, "scoring_provenance")
            if raw is None:
                raise ConfigurationError("tau-USI requires annotation, survey, and difficulty scoring provenance")
            provenance = TauScoringProvenance(
                annotation_revision=str(raw.get("annotation_revision") or ""),
                feature_extractor_revision=str(raw.get("feature_extractor_revision") or ""),
                survey_schema_revision=str(raw.get("survey_schema_revision") or ""),
                difficulty_revision=str(raw.get("difficulty_revision") or ""),
                human_batch_ids=tuple(str(value) for value in raw.get("human_batch_ids") or ()),
                expected_task_count=int(raw.get("expected_task_count", 0)),
                source=str(raw.get("source") or "configured"),
                protocol_status=str(
                    raw.get("protocol_status") or "official_equations_with_configured_derived_inputs"
                ),
            )
        if provenance.feature_extractor_revision != self._feature_extractor_revision:
            raise ConfigurationError(
                "tau-USI scoring provenance feature extractor does not match the configured executable extractor"
            )
        return provenance

    def assistant_or_partner_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        runtime = self.runtime_for_case(case)
        return {
            "role": ASSISTANT_ROLE,
            "model": runtime.fixed_assistant_model,
            "model_revision": runtime.fixed_assistant_revision,
            "policy_revision": runtime.assistant_policy_revision,
            "source": runtime.source,
            "held_fixed_across_simulator_conditions": True,
            "human_reference_assistant": "gpt-5.2",
            "comparison_scope": "relative_under_configured_assistant",
        }

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        runtime = self.runtime_for_case(case)
        return {
            "role": "task_environment",
            "revision": runtime.environment_revision,
            "adapter_revision": self.environment.environment_revision,
            "tool_schema_revision": runtime.tool_schema_revision,
            "adapter_tool_schema_revision": self.environment.tool_schema_revision,
            "max_user_turns": runtime.max_user_turns,
            "max_assistant_steps_per_user_turn": runtime.max_assistant_steps_per_user_turn,
            "assistant_step_limit_policy": ASSISTANT_STEP_LIMIT_POLICY,
            "request_timeout_seconds": runtime.request_timeout_seconds,
            "max_retries": runtime.max_retries,
            "survey_required": runtime.survey_required,
            "token_accounting": runtime.token_accounting,
        }

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"TauUSIAdapter cannot run {case.benchmark_id!r}")
        probe_case(case)
        runtime = self.runtime_for_case(case)
        scoring = self.provenance_for_case(case)
        if runtime.environment_revision != self.environment.environment_revision:
            raise ConfigurationError(
                f"tau-USI runtime environment revision {runtime.environment_revision!r} does not match "
                f"adapter {self.environment.environment_revision!r}"
            )
        legacy_fixture_protocol = (
            runtime.source == "repository_owned_synthetic_fixture"
            and runtime.tool_schema_revision == "single-tool-call-priority-json-v2"
        )
        if runtime.tool_schema_revision != self.environment.tool_schema_revision and not legacy_fixture_protocol:
            raise ConfigurationError(
                f"tau-USI runtime tool schema revision {runtime.tool_schema_revision!r} does not match "
                f"adapter {self.environment.tool_schema_revision!r}"
            )
        self.environment.reset(case, seed=0)
        evaluation = self._evaluation_for_case(case)
        difficulty = evaluation.get("difficulty")
        if difficulty is not None:
            if not isinstance(difficulty, Mapping):
                raise ValidationError("tau-USI evaluation.difficulty must be an object")
            score = difficulty.get("score")
            computed_bin = difficulty_bin(score)  # type: ignore[arg-type]
            declared_bin = difficulty.get("bin")
            if declared_bin is not None and declared_bin != computed_bin:
                raise ValidationError("tau-USI declared difficulty bin does not match fixed score cutoffs")
            if difficulty.get("revision") != scoring.difficulty_revision:
                raise ValidationError("tau-USI task difficulty revision differs from scoring provenance")
        reward = evaluation.get("environment_reward")
        if reward is not None:
            self.environment.finalize_reward(self.environment.reset(case, seed=0), reward)
        self._derived_human_references(case, scoring)

    @staticmethod
    def _evaluation_for_case(case: BenchmarkCase) -> Mapping[str, Any]:
        value = case.metadata.get("evaluation")
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValidationError("tau-USI metadata.evaluation must be an object")
        return value

    def build_user_request(
        self,
        case: BenchmarkCase,
        state: TauTaskState,
        *,
        model: str,
        seed: int,
    ) -> ModelRequest:
        messages = merge_system_instructions(
            self.environment.observation(state, actor=USER_ROLE),
            _USER_BEHAVIOR_INSTRUCTION,
            f"{_USER_ACTION_CONTRACT} Do not emit assistant or tool actions.",
        )
        return ModelRequest(
            request_id=f"{case.case_id}:user:{state.user_turn_count}",
            messages=tuple(messages),
            model=model,
            temperature=0.0,
            max_tokens=256,
            seed=seed + len(state.trace),
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": USER_ROLE,
                "prompt_revision": self.prompt_revision,
                "human_reference_visible": False,
                "environment_private_state_visible": False,
            },
        )

    def build_assistant_request(
        self,
        case: BenchmarkCase,
        state: TauTaskState,
        runtime: TauRuntimeProvenance,
        *,
        seed: int,
    ) -> ModelRequest:
        messages = self.environment.observation(state, actor=ASSISTANT_ROLE)
        return ModelRequest(
            request_id=(
                f"{case.case_id}:assistant:user{state.user_turn_count}:step{state.assistant_step_count}"
            ),
            messages=tuple(messages),
            model=runtime.fixed_assistant_model,
            temperature=0,
            max_tokens=512,
            seed=seed + len(state.trace),
            tools=(),
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": ASSISTANT_ROLE,
                "tool_protocol": self.environment.tool_schema_revision,
                "fixed_assistant_revision": runtime.fixed_assistant_revision,
                "assistant_policy_revision": runtime.assistant_policy_revision,
                "human_reference_visible": False,
                "evaluator_reward_visible": False,
            },
        )

    def build_survey_request(
        self,
        case: BenchmarkCase,
        state: TauTaskState,
        *,
        model: str,
        seed: int,
    ) -> ModelRequest:
        # Reuse the evaluated user's actual history, including its terminal
        # response. Tool observations remain private to the fixed assistant.
        messages = merge_system_instructions(
            self.environment.observation(state, actor=USER_ROLE, allow_terminal=True),
            _USER_BEHAVIOR_INSTRUCTION,
            f"{_USER_ACTION_CONTRACT} Do not emit assistant or tool actions.",
        )
        return ModelRequest(
            request_id=f"{case.case_id}:survey",
            messages=(*messages, ChatMessage("user", survey_prompt())),
            model=model, temperature=0, max_tokens=256, seed=seed,
            metadata={
                "benchmark_id": self.benchmark_id, "actor": USER_ROLE,
                "stage": "post_interaction_survey", "survey_schema_revision": SURVEY_SCHEMA_REVISION,
                "human_reference_visible": False,
            },
        )

    @staticmethod
    def build_survey_repair_request(case, text, runtime, *, seed):
        # Supplemental uses a helper only to extract answers from malformed JSON.
        # Route that helper to the already configured fixed DeepSeek assistant.
        return ModelRequest(
            request_id=f"{case.case_id}:survey_repair",
            model=runtime.fixed_assistant_model, temperature=0, max_tokens=512, seed=seed,
            messages=(ChatMessage("user",
                "A user just filled out a survey but the output was not valid JSON. "
                "Extract the survey answers and return ONLY a JSON object with these keys: "
                + ", ".join(FIELD_ORDINAL)
                + ". Do not infer or invent answers that are not present.\n\n" + text),),
            metadata={"benchmark_id": "tau_usi", "actor": ASSISTANT_ROLE, "stage": "survey_repair"},
        )

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        self.validate_case(case)
        state = self.environment.reset(case, seed=seed)
        return self.build_user_request(case, state, model=model, seed=seed)

    def parse_response(self, case: BenchmarkCase, response: ModelResponse) -> Any:
        return parse_user_action(response.text)

    @staticmethod
    def _accounted_tokens(request: ModelRequest, response: ModelResponse) -> int:
        if response.usage is not None and response.usage.total_tokens is not None:
            return response.usage.total_tokens
        return sum(len(message.content.split()) for message in request.messages) + len(response.text.split())

    @staticmethod
    def _generate_with_retries(
        backend: ModelBackend,
        request: ModelRequest,
        runtime: TauRuntimeProvenance,
        audits: list[Mapping[str, Any]],
    ) -> ModelResponse:
        current_request = request
        for attempt in range(runtime.max_retries + 1):
            try:
                response = backend.generate(current_request)
                if response.latency_ms is not None and response.latency_ms > runtime.request_timeout_seconds * 1000:
                    raise BackendTimeoutError(
                        f"response latency {response.latency_ms}ms exceeded {runtime.request_timeout_seconds}s"
                    )
                audits.append(
                    {
                        "request_id": request.request_id,
                        "actual_request_id": current_request.request_id,
                        "actor": request.metadata.get("actor"),
                        "attempt": attempt + 1,
                        "status": "completed",
                    }
                )
                return response
            except BackendError as exc:
                audits.append(
                    {
                        "request_id": request.request_id,
                        "actual_request_id": current_request.request_id,
                        "actor": request.metadata.get("actor"),
                        "attempt": attempt + 1,
                        "status": "retry" if attempt < runtime.max_retries else "failed",
                        "kind": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                if attempt == runtime.max_retries:
                    raise
                contract_error = isinstance(exc, BackendStructuredOutputError) or (
                    "tau-USI fixed assistant must return at most one native tool call" in str(exc)
                )
                if contract_error:
                    current_request = corrective_retry_request(
                        request,
                        attempt=attempt + 1,
                        reason=str(exc),
                        contract=(
                            _ASSISTANT_ACTION_CONTRACT
                            if request.metadata.get("actor") == ASSISTANT_ROLE
                            else _USER_ACTION_CONTRACT
                        ),
                    )
        raise AssertionError("unreachable tau-USI retry loop")

    def _derived_human_references(
        self, case: BenchmarkCase, scoring: TauScoringProvenance
    ) -> Mapping[str, Mapping[str, Any]]:
        raw = self._evaluation_for_case(case).get("human_references")
        if raw is None and self._human_reference_resolver is not None:
            raw = self._human_reference_resolver(case)
        if raw is None:
            return {}
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValidationError("tau-USI evaluation.human_references must be an array")
        derived: dict[str, Mapping[str, Any]] = {}
        all_features = tuple(sorted({feature for names in FEATURE_DIMENSIONS.values() for feature in names}))
        for index, reference in enumerate(raw):
            if not isinstance(reference, Mapping):
                raise ValidationError(f"tau-USI human reference #{index} must be an object")
            batch_id = str(reference.get("batch_id") or "")
            if not batch_id or batch_id in derived:
                raise ValidationError("tau-USI human batch IDs must be nonempty and unique per task")
            if batch_id not in scoring.human_batch_ids:
                raise ValidationError(f"undeclared tau-USI human batch ID {batch_id!r}")
            output: dict[str, Any] = {}
            features = reference.get("features")
            user_messages = reference.get("user_messages")
            if isinstance(features, Mapping):
                validated = _mean_vectors([features], all_features)
                if validated is None:
                    raise ValidationError(f"tau-USI human reference {batch_id} has invalid features")
                if any(value < 0 for value in validated.values()):
                    raise ValidationError(f"tau-USI human reference {batch_id} features cannot be negative")
                output["features"] = validated
            elif isinstance(user_messages, Sequence) and not isinstance(user_messages, (str, bytes)):
                if not all(isinstance(message, str) for message in user_messages):
                    raise ValidationError(f"tau-USI human reference {batch_id} messages must be text")
                output["features"] = dict(self._feature_extractor(tuple(user_messages)))
            survey = reference.get("survey")
            if isinstance(survey, Mapping):
                output["survey"] = normalize_survey(survey)
            reward = reference.get("reward")
            if reward is not None:
                if isinstance(reward, bool):
                    reward = float(reward)
                if not isinstance(reward, (int, float)) or not 0 <= float(reward) <= 1:
                    raise ValidationError(f"tau-USI human reference {batch_id} reward must be in [0,1]")
                output["reward"] = float(reward)
            derived[batch_id] = output
        return derived

    def replay_responses(self, case: BenchmarkCase, *, seed: int) -> Mapping[str, Any]:
        self.validate_case(case)
        replay = case.metadata.get("replay")
        if not isinstance(replay, Mapping):
            raise ConfigurationError(f"fixture {case.case_id} has no tau-USI replay metadata")
        steps = replay.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise ConfigurationError(f"fixture {case.case_id} replay.steps must be an array")
        runtime = self.runtime_for_case(case)
        state = self.environment.reset(case, seed=seed)
        responses: dict[str, Any] = {}
        for index, step in enumerate(steps):
            if state.terminal:
                raise ConfigurationError(f"fixture {case.case_id} has replay step after termination")
            if not isinstance(step, Mapping):
                raise ConfigurationError(f"fixture {case.case_id} replay step #{index} must be an object")
            actor = str(step.get("actor") or "")
            if actor != state.next_actor:
                raise ConfigurationError(
                    f"fixture {case.case_id} replay actor #{index} is {actor!r}, expected {state.next_actor!r}"
                )
            output = step.get("output")
            if not isinstance(output, Mapping):
                raise ConfigurationError(f"fixture {case.case_id} replay step #{index} needs output object")
            if actor == USER_ROLE:
                request = self.build_user_request(case, state, model="offline-replay", seed=seed)
                if set(output) != {"action", "message"}:
                    raise ConfigurationError(
                        f"fixture {case.case_id} replay user step #{index} has an invalid legacy envelope"
                    )
                legacy_action = output.get("action")
                legacy_message = output.get("message")
                if legacy_action == "stop":
                    text = TAU_USER_STOP_TOKEN
                elif legacy_action == "send" and isinstance(legacy_message, str) and legacy_message.strip():
                    text = legacy_message
                else:
                    raise ConfigurationError(
                        f"fixture {case.case_id} replay user step #{index} has no raw user utterance"
                    )
                action = parse_user_action(text)
            else:
                request = self.build_assistant_request(case, state, runtime, seed=seed)
                if set(output) != {"message", "tool_call", "done"}:
                    raise ConfigurationError(
                        f"fixture {case.case_id} replay assistant step #{index} has an invalid legacy envelope"
                    )
                legacy_message = output.get("message")
                legacy_tool = output.get("tool_call")
                if not isinstance(legacy_message, str) or not isinstance(output.get("done"), bool):
                    raise ConfigurationError(
                        f"fixture {case.case_id} replay assistant step #{index} has invalid legacy fields"
                    )
                raw: dict[str, Any] = {"_sim_eval": {"protocol": "legacy_fixture_translation"}}
                text = legacy_message
                if legacy_tool is not None:
                    if not isinstance(legacy_tool, Mapping) or set(legacy_tool) != {"name", "arguments"}:
                        raise ConfigurationError(
                            f"fixture {case.case_id} replay assistant step #{index} has an invalid tool call"
                        )
                    text = tool_call_text(legacy_tool["name"], legacy_tool["arguments"])
                replay_response = ModelResponse(text=text, raw=raw)
                action = parse_assistant_response(replay_response)
            responses[request.request_id or ""] = {
                "text": text,
                "finish_reason": "replayed",
                "usage": step.get("usage") or {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
            if actor == ASSISTANT_ROLE:
                responses[request.request_id or ""]["raw"] = raw
            self.environment.apply(state, actor=actor, action=action)
        if not state.terminal:
            raise ConfigurationError(f"fixture {case.case_id} replay steps do not terminate the tau-USI episode")
        if runtime.survey_required:
            survey = replay.get("survey")
            if not isinstance(survey, Mapping):
                raise ConfigurationError(f"fixture {case.case_id} requires replay.survey")
            normalize_survey(survey)
            request = self.build_survey_request(case, state, model="offline-replay", seed=seed)
            responses[request.request_id or ""] = {
                "text": canonical_json(fixture_survey_options(survey)),
                "finish_reason": "replayed",
                "usage": replay.get("survey_usage")
                or {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
        return responses

    def _failure_result(
        self,
        case: BenchmarkCase,
        state: TauTaskState,
        *,
        run_id: str,
        repetition: int,
        responses: Sequence[ModelResponse],
        runtime: TauRuntimeProvenance,
        scoring: TauScoringProvenance,
        request_audits: Sequence[Mapping[str, Any]],
        accounted_tokens: int,
        actor: str,
        stage: str,
        kind: str,
        message: str,
        retryable: bool,
    ) -> CaseResult:
        if not state.terminal:
            self.environment.fail(state, actor=actor, stage=stage, kind=kind)
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.FAILED,
            trace=tuple(state.trace),
            model_response=responses[-1] if responses else None,
            error=ErrorState(
                stage,
                kind,
                message,
                retryable=retryable,
                details={"request_audits": list(request_audits)},
            ),
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "environment_revision": self.environment.environment_revision,
                "episode_complete": False,
                "terminal_reason": state.terminal_reason,
                "request_audits": list(request_audits),
                "accounted_tokens": accounted_tokens,
                "tau_usi": {
                    "episode_complete": False,
                    "scoring_provenance": scoring.to_dict(),
                    "runtime_provenance": runtime.to_dict(),
                    "failure_stage": stage,
                },
            },
        )

    def _target_capability_result(
        self,
        case: BenchmarkCase,
        state: TauTaskState,
        *,
        run_id: str,
        repetition: int,
        responses: Sequence[ModelResponse],
        runtime: TauRuntimeProvenance,
        scoring: TauScoringProvenance,
        request_audits: Sequence[Mapping[str, Any]],
        accounted_tokens: int,
        stage: str,
        kind: str,
        message: str,
        preserve_environment_reward: bool = False,
        simulator_features: Mapping[str, float] | None = None,
        termination_details: Mapping[str, Any] | None = None,
    ) -> CaseResult:
        if not state.terminal:
            self.environment.terminate_capability_limit(
                state,
                actor=USER_ROLE,
                stage=stage,
                kind=kind,
            )
        budget_exhausted = kind == "token_budget_exhausted"
        failure = {
            "stage": stage,
            "kind": kind,
            "message": message,
            "retryable": False,
            "scoring_policy": (
                "partial_trajectory_scored_at_budget_limit"
                if budget_exhausted
                else "target_capability_failure_scores_zero"
            ),
            **(
                {"budget": dict(termination_details)}
                if termination_details is not None
                else {}
            ),
        }
        outcome_metadata_key = (
            "budget_termination" if budget_exhausted else "target_output_failure"
        )
        all_features = tuple(sorted({feature for names in FEATURE_DIMENSIONS.values() for feature in names}))
        if simulator_features is None:
            user_messages = [
                str(event.content)
                for event in state.trace
                if event.actor == USER_ROLE and event.kind == "message"
            ]
            # Capability-format failures include a just-returned target response
            # that was not applied to the environment, so retain that text for
            # behavioral feature scoring. Budget exhaustion occurs before a
            # response exists; in that case the latest response may be the fixed
            # assistant and must not be misattributed to the evaluated user.
            if responses and not budget_exhausted:
                raw = responses[-1].text.strip()
                if raw and raw != TAU_USER_STOP_TOKEN:
                    user_messages.append(raw)
            try:
                candidate = dict(self._feature_extractor(tuple(user_messages))) if user_messages else {}
                validated = _mean_vectors([candidate], all_features) if candidate else None
                simulator_features = validated if validated is not None else {name: 0.0 for name in all_features}
            except (ValidationError, ValueError):
                simulator_features = {name: 0.0 for name in all_features}
        simulator_survey = None  # no survey response was received on this failure path
        evaluation = self._evaluation_for_case(case)
        environment_reward = (
            self.environment.finalize_reward(state, evaluation.get("environment_reward"))
            if preserve_environment_reward
            else 0.0
        )
        difficulty = evaluation.get("difficulty")
        difficulty_score = float(difficulty["score"]) if isinstance(difficulty, Mapping) else None
        human_references = self._derived_human_references(case, scoring)
        public_transcript = [
            {"turn": event.turn, "actor": event.actor, "kind": event.kind, "content": event.content}
            for event in state.public_transcript
        ]
        metrics = (
            MetricValue(
                "tau_usi.environment_reward",
                environment_reward,
                unit="0_to_1",
                metadata={outcome_metadata_key: failure, "computed_from_final_environment_state": preserve_environment_reward},
            ),
            MetricValue(
                "tau_usi.stop_compliance",
                0.0,
                unit="0_to_1",
                metadata={outcome_metadata_key: failure},
            ),
            MetricValue(
                "tau_usi.diagnostic.strict_episode_success",
                0.0,
                unit="0_to_1",
                metadata={outcome_metadata_key: failure, "nonofficial_diagnostic": True},
            ),
        )
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction={
                "public_transcript": public_transcript,
                "terminal_reason": state.terminal_reason,
                "user_turn_count": state.user_turn_count,
                "tool_call_count": sum(event.kind == "tool_call" for event in state.trace),
                "parsed": False,
            },
            metrics=metrics,
            trace=tuple(state.trace),
            model_response=responses[-1] if responses else None,
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "environment_revision": self.environment.environment_revision,
                "prompt_revision": self.prompt_revision,
                "scorer_revision": self.scorer_revision,
                "episode_complete": True,
                "natural_termination": False,
                "terminal_reason": state.terminal_reason,
                "request_audits": list(request_audits),
                "accounted_tokens": accounted_tokens,
                "evaluated_model_role": USER_ROLE,
                "fixed_assistant_identity": self.assistant_or_partner_identity_for_case(case),
                "environment_identity": self.environment_identity_for_case(case),
                "human_reference_visible_to_models": False,
                **({"budget_termination": failure} if budget_exhausted else {"target_output_failure": failure}),
                "capability_termination": failure,
                "tau_usi": {
                    "episode_complete": True,
                    "capability_termination": failure,
                    "simulator_features": dict(simulator_features),
                    "simulator_survey": simulator_survey,
                    "environment_reward": environment_reward,
                    "stop_compliance": 0.0,
                    "strict_episode_success": 0.0,
                    "difficulty_score": difficulty_score,
                    "human_references": human_references,
                    "scoring_provenance": scoring.to_dict(),
                    "runtime_provenance": runtime.to_dict(),
                },
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
        runtime = self.runtime_for_case(case)
        scoring = self.provenance_for_case(case)
        state = self.environment.reset(case, seed=seed)
        responses: list[ModelResponse] = []
        request_audits: list[Mapping[str, Any]] = []
        accounted_tokens = 0
        capability_termination: Mapping[str, Any] | None = None

        while not state.terminal:
            actor = state.next_actor
            assert actor is not None
            if actor == ASSISTANT_ROLE and state.assistant_step_count >= runtime.max_assistant_steps_per_user_turn:
                self.environment.handoff_assistant_step_limit(
                    state, limit=runtime.max_assistant_steps_per_user_turn
                )
                actor = USER_ROLE
            request = (
                self.build_user_request(case, state, model=model, seed=seed)
                if actor == USER_ROLE
                else self.build_assistant_request(case, state, runtime, seed=seed)
            )
            stage_prefix = "user" if actor == USER_ROLE else "assistant"
            try:
                response = self._generate_with_retries(backend, request, runtime, request_audits)
                responses.append(response)
                accounted_tokens += self._accounted_tokens(request, response)
            except EpisodeTokenBudgetExhausted as exc:
                return self._target_capability_result(
                    case,
                    state,
                    run_id=run_id,
                    repetition=repetition,
                    responses=responses,
                    runtime=runtime,
                    scoring=scoring,
                    request_audits=request_audits,
                    accounted_tokens=accounted_tokens,
                    stage="token_budget_exhausted",
                    kind="token_budget_exhausted",
                    message=str(exc),
                    preserve_environment_reward=True,
                    termination_details=exc.details(),
                )
            except BackendStructuredOutputError as exc:
                if actor == USER_ROLE:
                    return self._target_capability_result(
                        case,
                        state,
                        run_id=run_id,
                        repetition=repetition,
                        responses=responses,
                        runtime=runtime,
                        scoring=scoring,
                        request_audits=request_audits,
                        accounted_tokens=accounted_tokens,
                        stage="user_parse",
                        kind="invalid_structured_output",
                        message=str(exc),
                    )
                return self._failure_result(
                    case,
                    state,
                    run_id=run_id,
                    repetition=repetition,
                    responses=responses,
                    runtime=runtime,
                    scoring=scoring,
                    request_audits=request_audits,
                    accounted_tokens=accounted_tokens,
                    actor=actor,
                    stage=f"{stage_prefix}_backend",
                    kind=type(exc).__name__,
                    message=str(exc),
                    retryable=False,
                )
            except BackendError as exc:
                return self._failure_result(
                    case,
                    state,
                    run_id=run_id,
                    repetition=repetition,
                    responses=responses,
                    runtime=runtime,
                    scoring=scoring,
                    request_audits=request_audits,
                    accounted_tokens=accounted_tokens,
                    actor=actor,
                    stage=f"{stage_prefix}_backend",
                    kind=type(exc).__name__,
                    message=str(exc),
                    retryable=True,
                )
            try:
                action: TauAssistantAction | Any
                action = (
                    parse_user_action(response.text)
                    if actor == USER_ROLE
                    else parse_assistant_response(response)
                )
            except ParseError as exc:
                if actor == USER_ROLE:
                    return self._target_capability_result(
                        case,
                        state,
                        run_id=run_id,
                        repetition=repetition,
                        responses=responses,
                        runtime=runtime,
                        scoring=scoring,
                        request_audits=request_audits,
                        accounted_tokens=accounted_tokens,
                        stage="user_parse",
                        kind="invalid_action",
                        message=str(exc),
                    )
                return self._failure_result(
                    case,
                    state,
                    run_id=run_id,
                    repetition=repetition,
                    responses=responses,
                    runtime=runtime,
                    scoring=scoring,
                    request_audits=request_audits,
                    accounted_tokens=accounted_tokens,
                    actor=actor,
                    stage=f"{stage_prefix}_parse",
                    kind="invalid_action",
                    message=str(exc),
                    retryable=False,
                )
            reached_turn_limit = (
                actor == USER_ROLE and action.action == "send"
                and state.user_turn_count >= runtime.max_user_turns
            )
            try:
                self.environment.apply(state, actor=actor, action=action)
            except ValidationError as exc:
                kind = "invalid_tool" if "invalid_tool" in str(exc) else "invalid_transition"
                if actor == USER_ROLE:
                    return self._target_capability_result(
                        case,
                        state,
                        run_id=run_id,
                        repetition=repetition,
                        responses=responses,
                        runtime=runtime,
                        scoring=scoring,
                        request_audits=request_audits,
                        accounted_tokens=accounted_tokens,
                        stage="user_transition",
                        kind=kind,
                        message=str(exc),
                    )
                return self._failure_result(
                    case,
                    state,
                    run_id=run_id,
                    repetition=repetition,
                    responses=responses,
                    runtime=runtime,
                    scoring=scoring,
                    request_audits=request_audits,
                    accounted_tokens=accounted_tokens,
                    actor=actor,
                    stage="environment_tool" if kind == "invalid_tool" else f"{stage_prefix}_transition",
                    kind=kind,
                    message=str(exc),
                    retryable=False,
                )

            if reached_turn_limit:
                capability_termination = {
                    "stage": "user_turn_limit", "kind": "stop_instruction_not_followed",
                    "max_user_turns": runtime.max_user_turns,
                    "final_user_reply_retained": True,
                    "environment_reward_policy": "final_environment_state",
                }
                self.environment.terminate_capability_limit(
                    state, actor=USER_ROLE, stage="user_turn_limit", kind="stop_instruction_not_followed"
                )
                break

        user_messages = [
            str(event.content)
            for event in state.trace
            if event.actor == USER_ROLE and event.kind == "message"
        ]
        try:
            all_features = tuple(sorted({feature for names in FEATURE_DIMENSIONS.values() for feature in names}))
            simulator_features = (dict(self._feature_extractor(tuple(user_messages))) if user_messages
                                  else {name: 0.0 for name in all_features})
            validated_features = _mean_vectors([simulator_features], all_features)
            if validated_features is None or any(value < 0 for value in validated_features.values()):
                raise ValidationError("configured tau-USI feature extractor returned an invalid feature vector")
            simulator_features = validated_features
        except (ValidationError, ValueError) as exc:
            return self._failure_result(
                case,
                state,
                run_id=run_id,
                repetition=repetition,
                responses=responses,
                runtime=runtime,
                scoring=scoring,
                request_audits=request_audits,
                accounted_tokens=accounted_tokens,
                actor="scorer",
                stage="behavior_features",
                kind="feature_extraction_error",
                message=str(exc),
                retryable=False,
            )

        simulator_survey: Mapping[str, float] | None = None
        survey_error: Mapping[str, Any] | None = None
        survey_raw: Mapping[str, Any] | None = None
        survey_response: ModelResponse | None = None
        survey_repaired = False
        survey_missing_fields: list[str] = []
        if runtime.survey_required:
            survey_request = self.build_survey_request(case, state, model=model, seed=seed)
            try:
                survey_response = self._generate_with_retries(backend, survey_request, runtime, request_audits)
                responses.append(survey_response)
                accounted_tokens += self._accounted_tokens(survey_request, survey_response)
                survey_raw = _parse_json_object(survey_response.text)
                if survey_raw is None:
                    repair = self.build_survey_repair_request(case, survey_response.text, runtime, seed=seed)
                    try:
                        repaired = self._generate_with_retries(backend, repair, runtime, request_audits)
                        responses.append(repaired)
                        accounted_tokens += self._accounted_tokens(repair, repaired)
                        survey_raw = _parse_json_object(repaired.text)
                        survey_repaired = survey_raw is not None
                    except BackendError as exc:
                        # The actual candidate answer exists and is malformed;
                        # a failed extraction cannot create answers on its behalf.
                        survey_error = {"stage": "survey_repair", "kind": type(exc).__name__, "message": str(exc)}
                simulator_survey = parse_survey_options(survey_raw or {})
                survey_missing_fields = [name for name in SURVEY_SCALES if name not in simulator_survey]
            except EpisodeTokenBudgetExhausted as exc:
                survey_error = {"stage": "survey_budget", "kind": type(exc).__name__, "message": str(exc)}
                # No answer was generated: do not impute an infrastructure/budget
                # failure as a missing field in a successfully received survey.
            except BackendError as exc:
                return self._failure_result(
                    case, state, run_id=run_id, repetition=repetition, responses=responses,
                    runtime=runtime, scoring=scoring, request_audits=request_audits,
                    accounted_tokens=accounted_tokens, actor=USER_ROLE, stage="survey_backend",
                    kind=type(exc).__name__, message=str(exc), retryable=True,
                )

        evaluation = self._evaluation_for_case(case)
        environment_reward = self.environment.finalize_reward(state, evaluation.get("environment_reward"))
        stop_compliance = (
            1.0
            if state.terminal_reason == "user_stop"
            else 0.0
            if capability_termination is not None
            and capability_termination.get("kind") == "stop_instruction_not_followed"
            else None
        )
        strict_episode_success = (
            environment_reward * stop_compliance
            if environment_reward is not None and stop_compliance is not None
            else None
        )
        difficulty = evaluation.get("difficulty")
        difficulty_score = float(difficulty["score"]) if isinstance(difficulty, Mapping) else None
        human_references = self._derived_human_references(case, scoring)
        public_transcript = [
            {"turn": event.turn, "actor": event.actor, "kind": event.kind, "content": event.content}
            for event in state.public_transcript
        ]
        reward_metric = MetricValue(
            "tau_usi.environment_reward",
            environment_reward,
            unit="0_to_1",
            metadata={
                "availability": "available" if environment_reward is not None else "unavailable",
                "role": "automatic_task_environment_outcome_not_human_feedback",
                "not_a_substitute_for_usi": True,
                "computed_from_final_environment_state": True,
                "independent_of_stop_compliance": True,
            },
        )
        stop_metric = MetricValue(
            "tau_usi.stop_compliance",
            stop_compliance,
            unit="0_to_1",
            metadata={
                "availability": "available" if stop_compliance is not None else "not_applicable",
                "formula": "1 for user_stop; 0 for user_turn_limit after another send",
                "official_primary": False,
                "does_not_modify_environment_reward_or_usi": True,
            },
        )
        strict_success_metric = MetricValue(
            "tau_usi.diagnostic.strict_episode_success",
            strict_episode_success,
            unit="0_to_1",
            metadata={
                "availability": (
                    "available" if strict_episode_success is not None else "not_applicable"
                ),
                "formula": "environment_reward * stop_compliance",
                "official_primary": False,
                "nonofficial_diagnostic": True,
                "does_not_modify_ece_outcome_alignment_or_usi": True,
            },
        )
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction={
                "public_transcript": public_transcript,
                "terminal_reason": state.terminal_reason,
                "user_turn_count": state.user_turn_count,
                "tool_call_count": sum(event.kind == "tool_call" for event in state.trace),
            },
            metrics=(reward_metric, stop_metric, strict_success_metric),
            trace=tuple(state.trace),
            model_response=survey_response or (responses[-1] if responses else None),
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "environment_revision": self.environment.environment_revision,
                "prompt_revision": self.prompt_revision,
                "scorer_revision": self.scorer_revision,
                "episode_complete": True,
                "terminal_reason": state.terminal_reason,
                "request_audits": request_audits,
                "accounted_tokens": accounted_tokens,
                "evaluated_model_role": USER_ROLE,
                "fixed_assistant_identity": self.assistant_or_partner_identity_for_case(case),
                "environment_identity": self.environment_identity_for_case(case),
                "human_reference_visible_to_models": False,
                "capability_termination": capability_termination,
                "post_interaction_evaluation": {
                    "survey_availability": "available" if simulator_survey is not None else "unavailable",
                    "survey_missing_fields": survey_missing_fields,
                    "survey_error": survey_error,
                },
                "tau_usi": {
                    "episode_complete": True,
                    "capability_termination": capability_termination,
                    "simulator_features": simulator_features,
                    "simulator_survey": simulator_survey,
                    "simulator_survey_raw": survey_raw,
                    "survey_answer_source": "fixed_assistant_json_extraction" if survey_repaired else "candidate_text",
                    "survey_missing_fields": survey_missing_fields,
                    "task_key": f"{state.domain}_{case.input_data.get('task_index', case.case_id)}",
                    "environment_reward": environment_reward,
                    "stop_compliance": stop_compliance,
                    "strict_episode_success": strict_episode_success,
                    "difficulty_score": difficulty_score,
                    "human_references": human_references,
                    "scoring_provenance": scoring.to_dict(),
                    "runtime_provenance": runtime.to_dict(),
                },
            },
        )

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        return self.scorer.aggregate(results)


__all__ = [
    "ECE_CUTOFFS",
    "FEATURE_DIMENSIONS",
    "FEATURE_EXTRACTOR_REVISION",
    "SURVEY_SCALES",
    "TauRuntimeProvenance",
    "TauScoringProvenance",
    "TauUSIAdapter",
    "TauUSIScorer",
    "dice_alignment",
    "difficulty_bin",
    "evaluative_alignment",
    "expected_calibration_error",
    "extract_behavior_features",
    "normalize_survey",
    "user_sim_index",
]
