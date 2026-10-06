"""UserLM dialogue simulation adapter and protocol-specific metrics.

The adapter keeps the paper's intrinsic PRISM probes separate from the
GSM8K/HumanEval-derived extrinsic simulation protocol.  Judge-backed fields
are nullable and never silently replaced by lexical heuristics.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

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
from ..environments.dialogue import (
    DialogueAction,
    DialogueEnvironment,
    DialogueSpec,
    DialogueState,
    EVALUATED_USER_ROLE,
    FIXED_ASSISTANT_ROLE,
    parse_dialogue_action,
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
from ..registry import adapter
from ..model_adaptation import USERLM_NATIVE, COSER_FORMAT, validate_model_adapter

from .common import (
    DEFAULT_CONTRACT_RETRIES,
    aggregate_named_metrics,
    combine_usage,
    corrective_retry_request,
    json_schema_response_format,
    merge_system_instructions,
)
from .lic_prompts import CODE_SYSTEM_PROMPT, LIC_SOURCE_REVISION, MATH_SYSTEM_PROMPT


USERLM_LIC_JSON_REVISION = "userlm-lic-json-3-60-v1"
USERLM_LIC_JSON_CONTRACT = (
    'Return exactly one JSON object with only the two required fields "action" and "message". '
    'The action must be exactly one of "message", "end", or "refuse". '
    'The message must be a string containing your actual user utterance, not instructions or field names. '
    'For a normal user turn use "action":"message" and put 3–60 words in the message string. '
    'Example: {"action":"message","message":"Can you help with this task?"}. '
    'Do not add other fields, Markdown fences, or text outside the JSON object. '
    'For message or refuse, message must be non-empty. '
    'Although end is a valid action value, ending the conversation is prohibited in this LiC task.'
)

END_CONVERSATION = "<|endconversation|>"
USERLM_VARIANTS = {
    "extrinsic_verifiable",
    "intrinsic_prism",
    "intrinsic_role_adherence",
    "intrinsic_intent_adherence",
}
_INTENT_JUDGE_CONTRACT = "Return exactly one token: REFUSED or ACCEPTED. Do not add punctuation or explanation."

_TOKEN = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", re.IGNORECASE)
_NUMBER = re.compile(
    r"(?<![\w,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w,])"
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it",
    "its", "me", "might", "my", "of", "on", "or", "our", "should", "so", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "to", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
}


def _parse_intent_verdict(response: ModelResponse) -> str:
    verdict = response.text.strip().upper()
    if verdict not in {"REFUSED", "ACCEPTED"}:
        raise ParseError("UserLM intent judge must return only REFUSED or ACCEPTED")
    return verdict

USERLM_FIRST_TURN_PROMPT = """You are a human user interacting with an AI system to {intent}.
Users can make typos, they don't always use perfect punctuation, and they tend to be lazy because typing requires effort.
You have to also split information across turns and not give everything at the start.
However, you should not make overdo these things in your outputs, you must realistically act like a human.
Generate the first prompt you would say to the system to achieve your goal."""

USERLM_SEQUENTIAL_TURN_PROMPT = """You are a human user interacting with an AI system to {intent}.
Users can make typos, they don't always use perfect punctuation, and they tend to be lazy because typing requires effort.
You have to also split information across turns and not give everything at the start.
However, you should not make overdo these things in your outputs, you must realistically act like a human.
Here is your conversation so far with the AI system:
====================================================
{conversation_history}
====================================================
If your goal of your conversation is answered by this response, respond ONLY with "<|endconversation|>"
If not, generate the next prompt you would say to the system to achieve your goal."""

_DIALOGUE_ACTION_CONTRACT = (
    'Return exactly one JSON object with exactly two fields: '
    '{"action":"message|end|refuse","message":"..."}. '
    "For action=message or refuse, message must be a non-empty string; emit no prose outside JSON."
)

_DEEPSEEK_LIC_FIRST_TURN_INSTRUCTION = (
    "Generate the first user message now, using your private context. "
    "Return only one JSON object with exactly action and message. "
    "Do not explain or repeat these instructions."
)


class UserLMCodeTaskVerifier(Protocol):
    def verify_case(
        self,
        case: BenchmarkCase,
        assistant_completion: str,
    ) -> tuple[float | None, Mapping[str, Any]]:
        ...


@dataclass(frozen=True)
class UserLMRuntimeProvenance:
    fixed_assistant_model: str
    fixed_assistant_revision: str
    assistant_policy_revision: str
    guardrail_revision: str
    verifier_revision: str
    max_user_turns: int = 8
    max_total_actions: int = 20
    request_timeout_seconds: float = 120.0
    max_retries: int = 2
    user_temperature: float = 0.0
    assistant_temperature: float = 0.0
    apply_extrinsic_guardrails: bool = True
    guardrail_min_words: int = 3
    guardrail_max_words: int = 25
    max_guardrail_regenerations: int = 8
    source: str = "configured"
    deepseek_lic_first_turn: bool = False
    model_adapter: str | None = None

    def __post_init__(self) -> None:
        validate_model_adapter(self.model_adapter)
        required = (
            self.fixed_assistant_model,
            self.fixed_assistant_revision,
            self.assistant_policy_revision,
            self.guardrail_revision,
            self.verifier_revision,
            self.source,
        )
        if not all(required):
            raise ConfigurationError("UserLM runtime provenance fields cannot be empty")
        if self.max_user_turns <= 0 or self.max_total_actions <= 0:
            raise ConfigurationError("UserLM dialogue limits must be positive")
        if self.request_timeout_seconds <= 0 or not math.isfinite(self.request_timeout_seconds):
            raise ConfigurationError("UserLM request timeout must be positive and finite")
        if self.max_retries < 0:
            raise ConfigurationError("UserLM max_retries cannot be negative")
        if (
            self.guardrail_min_words <= 0
            or self.guardrail_max_words < self.guardrail_min_words
            or self.max_guardrail_regenerations < 0
        ):
            raise ConfigurationError("UserLM guardrail limits are invalid")

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        if not self.deepseek_lic_first_turn:
            result.pop("deepseek_lic_first_turn")
        if self.model_adapter is None:
            result.pop("model_adapter")
        return result


@dataclass(frozen=True)
class UserLMScoringProvenance:
    shard_judge_model: str | None
    shard_judge_revision: str | None
    shard_prompt_revision: str
    intent_judge_model: str | None
    intent_judge_revision: str | None
    intent_prompt_revision: str
    lemmatizer_revision: str
    ai_detector_model: str | None = None
    ai_detector_revision: str | None = None
    ai_detector_contract_revision: str | None = None
    replayed: bool = False
    source: str = "configured"

    def __post_init__(self) -> None:
        if not self.shard_prompt_revision or not self.intent_prompt_revision or not self.lemmatizer_revision:
            raise ConfigurationError("UserLM scoring prompt/lemmatizer revisions cannot be empty")
        if bool(self.shard_judge_model) != bool(self.shard_judge_revision):
            raise ConfigurationError("UserLM shard judge model and revision must be configured together")
        if bool(self.intent_judge_model) != bool(self.intent_judge_revision):
            raise ConfigurationError("UserLM intent judge model and revision must be configured together")
        detector_fields = (
            self.ai_detector_model,
            self.ai_detector_revision,
            self.ai_detector_contract_revision,
        )
        if any(detector_fields) and not all(detector_fields):
            raise ConfigurationError(
                "UserLM AI detector model, revision, and contract revision must be configured together"
            )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _words(text: str, *, remove_stopwords: bool = False) -> tuple[str, ...]:
    values = tuple(token.casefold() for token in _TOKEN.findall(text))
    return tuple(token for token in values if not remove_stopwords or token not in _STOPWORDS)


def heuristic_lemmas(text: str) -> frozenset[str]:
    """Dependency-free explicit protocol variant used only when so versioned."""

    lemmas: set[str] = set()
    for word in _words(text):
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 3 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        lemmas.add(word)
    return frozenset(lemmas)


def unigram_jaccard(left: Sequence[str] | set[str], right: Sequence[str] | set[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def intent_decomposition_overlap(intent: str, user_turns: Sequence[str]) -> float:
    """Mean paper 1-gram intent overlap: |intent ∩ turn| / |turn|."""
    intent_tokens = set(_words(intent, remove_stopwords=True))
    scores = []
    for turn in user_turns:
        turn_tokens = set(_words(turn, remove_stopwords=True))
        scores.append(len(intent_tokens & turn_tokens) / len(turn_tokens) if turn_tokens else 0.0)
    return statistics.mean(scores) if scores else 0.0


def role_adherence_score(reply: str, choices: Sequence[str]) -> tuple[float, int]:
    """Implement the paper's one/two-choice answer-attempt rule."""

    if len(choices) < 3 or any(not isinstance(choice, str) or not choice.strip() for choice in choices):
        raise ValidationError("UserLM role-adherence choices require at least three non-empty strings")
    lowered = reply.casefold()
    mentioned = {
        choice.casefold()
        for choice in choices
        if re.search(rf"(?<!\w){re.escape(choice.casefold())}(?!\w)", lowered)
    }
    count = len(mentioned)
    attempted = count in {1, 2}
    return (0.0 if attempted else 1.0), count


def _metric(
    name: str,
    value: float | int | bool | None,
    *,
    direction: str = "higher_is_better",
    unit: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MetricValue:
    metric_metadata = dict(metadata or {})
    availability = str(metric_metadata.get("availability") or "").casefold()
    if availability == "not_applicable" or availability.startswith("not_applicable_"):
        # Stable-schema placeholders are useful for auditing, but a metric
        # outside this case's protocol must never carry a numeric value that a
        # later generic aggregation could mistake for model performance.
        value = None
    return MetricValue(
        name=name,
        value=value,
        direction=direction,
        unit=unit,
        metadata={
            "availability": "available" if value is not None else "unavailable",
            **metric_metadata,
        },
    )


def _source_task(case: BenchmarkCase) -> str | None:
    strata = case.metadata.get("strata")
    value = str(
        strata.get("source_task") if isinstance(strata, Mapping) else ""
    ).strip().lower()
    return value or None


def _shard_records(case: BenchmarkCase) -> tuple[Mapping[str, Any], ...]:
    raw = case.input_data.get("information_shards", case.input_data.get("required_information"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValidationError("UserLM requires at least one information shard")
    required: list[str] = []
    nonrequired: list[str] = []
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            shard_id, is_required = item.strip(), True
            text = shard_id
        elif isinstance(item, Mapping):
            shard_id = str(item.get("id") or item.get("text") or "").strip()
            is_required = bool(item.get("required", True))
            text = str(item.get("text") or "").strip()
            if not text:
                # Legacy fixtures use descriptive text as the identifier.
                # Opaque numbered IDs cannot stand in for missing shard text.
                if re.fullmatch(r"shard[_ -]?\d+", shard_id, re.IGNORECASE):
                    raise ValidationError(f"UserLM shard {shard_id!r} requires its text")
                text = shard_id
        else:
            raise ValidationError(f"UserLM information shard #{index} is invalid")
        if not shard_id:
            raise ValidationError(f"UserLM information shard #{index} is empty")
        if text == shard_id and re.fullmatch(r"shard[_ -]?\d+", shard_id, re.IGNORECASE):
            raise ValidationError(f"UserLM shard {shard_id!r} requires its text")
        (required if is_required else nonrequired).append(shard_id)
        records.append({"id": shard_id, "text": text, "required": is_required})
    all_ids = required + nonrequired
    if not required or len(all_ids) != len(set(all_ids)):
        raise ValidationError("UserLM shards require unique IDs and at least one required shard")
    return tuple(records)


def _parse_shards(case: BenchmarkCase) -> tuple[tuple[str, ...], tuple[str, ...]]:
    records = _shard_records(case)
    return (
        tuple(item["id"] for item in records if item["required"]),
        tuple(item["id"] for item in records if not item["required"]),
    )


@adapter("userlm")
class UserLMAdapter(BenchmarkAdapter):
    benchmark_id = "userlm"
    # Section 3 keeps its existing identity because its Figure 9 request did
    # not change. Interactive LiC binds the revised contract separately.
    prompt_revision = "userlm-role-isolated-json-v1"
    lic_prompt_revision = "userlm-lic-v6-no-first-token-filter"
    scorer_revision = "userlm-paper-metrics-v3-any-assistant-turn-code-verification-explicit-na"

    def __init__(
        self,
        *,
        runtime_provenance: UserLMRuntimeProvenance | None = None,
        scoring_provenance: UserLMScoringProvenance | None = None,
        lemmatizer: Callable[[str], frozenset[str]] = heuristic_lemmas,
        code_task_verifier: UserLMCodeTaskVerifier | None = None,
        ai_text_detector: Callable[[str], Mapping[str, Any]] | None = None,
        defer_ai_text_detection: bool = False,
    ) -> None:
        self.environment = DialogueEnvironment()
        self._runtime = runtime_provenance
        self._scoring = scoring_provenance
        self._lemmatizer = lemmatizer
        self._code_task_verifier = code_task_verifier
        self._ai_text_detector = ai_text_detector
        self._defer_ai_text_detection = defer_ai_text_detection

    @property
    def ai_text_detector_batch_size(self) -> int:
        config = getattr(self._ai_text_detector, "config", None)
        return int(getattr(config, "batch_size", 1))

    @staticmethod
    def needs_ai_text_detection(case: BenchmarkCase, result: CaseResult) -> bool:
        userlm = result.metadata.get("userlm")
        return (
            result.status == ResultStatus.COMPLETED
            and case.input_data.get("variant") == "intrinsic_prism"
            and isinstance(userlm, Mapping)
            and isinstance(userlm.get("all_user_text"), str)
            and bool(str(userlm["all_user_text"]).strip())
            and str(userlm["all_user_text"]).strip() != END_CONVERSATION
        )

    @staticmethod
    def _validated_ai_detector_evidence(
        evidence: Mapping[str, Any],
        provenance: UserLMScoringProvenance,
        *,
        source: str,
    ) -> tuple[float | None, Mapping[str, Any]]:
        value = evidence.get("human_likelihood")
        if value is None:
            return None, {"availability": "unavailable", "source": source, **dict(evidence)}
        if isinstance(value, bool):
            raise ValidationError("UserLM AI detector human likelihood must be numeric")
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValidationError("UserLM AI detector human likelihood must be in [0,1]")
        if not provenance.ai_detector_revision:
            raise ValidationError("AI detector score requires named detector provenance")
        return score, {"availability": "available", "source": source, **dict(evidence)}

    def finalize_ai_text_detector_batch(
        self,
        items: Sequence[tuple[BenchmarkCase, CaseResult]],
    ) -> tuple[CaseResult, ...]:
        """Batch-score deferred PRISM text and preserve case order."""

        if not items:
            return ()
        score_batch = getattr(self._ai_text_detector, "score_batch", None)
        if not callable(score_batch):
            raise ConfigurationError(
                "deferred UserLM AI detection requires a batch-capable detector"
            )
        texts = tuple(
            str(result.metadata["userlm"]["all_user_text"])
            for _, result in items
        )
        evidence_items = tuple(score_batch(texts))
        if len(evidence_items) != len(items):
            raise ValidationError(
                "UserLM AI detector result count does not match deferred case count"
            )
        finalized = []
        metric_name = "userlm.intrinsic.ai_detector_human_likelihood"
        for (case, result), raw_evidence in zip(items, evidence_items):
            provenance = self.provenance_for_case(case)
            score, evidence = self._validated_ai_detector_evidence(
                dict(raw_evidence),
                provenance,
                source="live_local_detector",
            )
            detector_metric = _metric(
                metric_name,
                score,
                metadata={
                    "detector_model": provenance.ai_detector_model,
                    "detector_revision": provenance.ai_detector_revision,
                    "detector_contract_revision": provenance.ai_detector_contract_revision,
                    **dict(evidence),
                },
            )
            metrics = tuple(
                detector_metric if metric.name == metric_name else metric
                for metric in result.metrics
            )
            metadata = {**dict(result.metadata), "ai_detector": dict(evidence)}
            finalized.append(replace(result, metrics=metrics, metadata=metadata))
        return tuple(finalized)

    @staticmethod
    def _metadata_mapping(case: BenchmarkCase, key: str) -> Mapping[str, Any] | None:
        direct = case.metadata.get(key)
        if isinstance(direct, Mapping):
            return direct
        replay = case.metadata.get("replay")
        value = replay.get(key) if isinstance(replay, Mapping) else None
        return value if isinstance(value, Mapping) else None

    def runtime_for_case(self, case: BenchmarkCase) -> UserLMRuntimeProvenance:
        runtime = self._base_runtime_for_case(case)
        if self._uses_userlm_lic_json(case, runtime):
            return replace(
                runtime, guardrail_min_words=3, guardrail_max_words=60,
                guardrail_revision=USERLM_LIC_JSON_REVISION,
            )
        return runtime

    def _uses_userlm_lic_json(
        self, case: BenchmarkCase, runtime: UserLMRuntimeProvenance | None = None,
    ) -> bool:
        runtime = runtime if runtime is not None else self._base_runtime_for_case(case)
        return (
            runtime.model_adapter == USERLM_NATIVE
            and self.execution_mode_for_case(case) == "interactive_dialogue"
            and self.variant_for_case(case) == "extrinsic_verifiable"
        )

    def _base_runtime_for_case(self, case: BenchmarkCase) -> UserLMRuntimeProvenance:
        if self._runtime is not None:
            return self._runtime
        raw = self._metadata_mapping(case, "runtime_provenance")
        if raw is None:
            raise ConfigurationError("UserLM requires fixed-assistant runtime provenance")
        return UserLMRuntimeProvenance(
            fixed_assistant_model=str(raw.get("fixed_assistant_model") or ""),
            fixed_assistant_revision=str(raw.get("fixed_assistant_revision") or ""),
            assistant_policy_revision=str(raw.get("assistant_policy_revision") or ""),
            guardrail_revision=str(raw.get("guardrail_revision") or ""),
            verifier_revision=str(raw.get("verifier_revision") or ""),
            max_user_turns=int(raw.get("max_user_turns", 8)),
            max_total_actions=int(raw.get("max_total_actions", 20)),
            request_timeout_seconds=float(raw.get("request_timeout_seconds", 120.0)),
            max_retries=int(raw.get("max_retries", 2)),
            user_temperature=float(raw.get("user_temperature", 0.0)),
            assistant_temperature=float(raw.get("assistant_temperature", 0.0)),
            apply_extrinsic_guardrails=bool(raw.get("apply_extrinsic_guardrails", True)),
            deepseek_lic_first_turn=bool(raw.get("deepseek_lic_first_turn", False)),
            model_adapter=raw.get("model_adapter"),
            guardrail_min_words=int(raw.get("guardrail_min_words", 3)),
            guardrail_max_words=int(raw.get("guardrail_max_words", 25)),
            max_guardrail_regenerations=int(raw.get("max_guardrail_regenerations", 8)),
            source=str(raw.get("source") or "configured"),
        )

    def provenance_for_case(self, case: BenchmarkCase) -> UserLMScoringProvenance:
        if self._scoring is not None:
            return self._scoring
        raw = self._metadata_mapping(case, "scoring_provenance")
        if raw is None:
            raise ConfigurationError("UserLM requires explicit scoring provenance")
        legacy_detector_revision = (
            str(raw["pangram_detector_revision"])
            if raw.get("pangram_detector_revision")
            else None
        )
        detector_revision = (
            str(raw["ai_detector_revision"])
            if raw.get("ai_detector_revision")
            else legacy_detector_revision
        )
        return UserLMScoringProvenance(
            shard_judge_model=(str(raw["shard_judge_model"]) if raw.get("shard_judge_model") else None),
            shard_judge_revision=(
                str(raw["shard_judge_revision"]) if raw.get("shard_judge_revision") else None
            ),
            shard_prompt_revision=str(raw.get("shard_prompt_revision") or ""),
            intent_judge_model=(str(raw["intent_judge_model"]) if raw.get("intent_judge_model") else None),
            intent_judge_revision=(
                str(raw["intent_judge_revision"]) if raw.get("intent_judge_revision") else None
            ),
            intent_prompt_revision=str(raw.get("intent_prompt_revision") or ""),
            lemmatizer_revision=str(raw.get("lemmatizer_revision") or ""),
            ai_detector_model=(
                str(raw["ai_detector_model"])
                if raw.get("ai_detector_model")
                else ("legacy-pangram-detector" if legacy_detector_revision else None)
            ),
            ai_detector_revision=detector_revision,
            ai_detector_contract_revision=(
                str(raw["ai_detector_contract_revision"])
                if raw.get("ai_detector_contract_revision")
                else ("legacy-pangram-human-likelihood-v1" if legacy_detector_revision else None)
            ),
            replayed=bool(raw.get("replayed", False)),
            source=str(raw.get("source") or "configured"),
        )

    def _ai_detector_score(
        self,
        case: BenchmarkCase,
        *,
        variant: str,
        text: str,
        provenance: UserLMScoringProvenance,
    ) -> tuple[float | None, Mapping[str, Any]]:
        """Return human-likeness only for the UserLM PRISM detector component."""

        if variant != "intrinsic_prism" or text == END_CONVERSATION:
            return None, {"availability": "not_applicable", "variant": variant}
        if self._ai_text_detector is not None:
            if not all(
                (
                    provenance.ai_detector_model,
                    provenance.ai_detector_revision,
                    provenance.ai_detector_contract_revision,
                )
            ):
                raise ConfigurationError(
                    "live UserLM AI detection requires named detector provenance"
                )
            if self._defer_ai_text_detection:
                return None, {
                    "availability": "pending_batch",
                    "source": "live_local_detector",
                }
            evidence = dict(self._ai_text_detector(text))
            return self._validated_ai_detector_evidence(
                evidence,
                provenance,
                source="live_local_detector",
            )
        else:
            evaluation = case.metadata.get("evaluation")
            value = None
            evidence = {}
            if isinstance(evaluation, Mapping):
                value = evaluation.get(
                    "ai_detector_human_likelihood",
                    evaluation.get("pangram_human_likelihood"),
                )
            source = "precomputed_case_metadata"
        return self._validated_ai_detector_evidence(
            {**evidence, "human_likelihood": value},
            provenance,
            source=source,
        )

    def assistant_or_partner_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        runtime = self.runtime_for_case(case)
        return {
            "role": FIXED_ASSISTANT_ROLE,
            "model": runtime.fixed_assistant_model,
            "model_revision": runtime.fixed_assistant_revision,
            "policy_revision": runtime.assistant_policy_revision,
            "temperature": runtime.assistant_temperature,
            "held_fixed": True,
            "source": runtime.source,
        }

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        runtime = self.runtime_for_case(case)
        return {
            "revision": self.environment.environment_revision,
            "termination_token": END_CONVERSATION,
            "max_user_turns": runtime.max_user_turns,
            "max_total_actions": runtime.max_total_actions,
            "guardrail_revision": runtime.guardrail_revision,
            "apply_extrinsic_guardrails": runtime.apply_extrinsic_guardrails,
            "guardrail_word_range": [runtime.guardrail_min_words, runtime.guardrail_max_words],
            "max_guardrail_regenerations": runtime.max_guardrail_regenerations,
            "request_timeout_seconds": runtime.request_timeout_seconds,
            "max_retries": runtime.max_retries,
            **({"first_turn_request_revision": "deepseek-lic-explicit-user-v1"}
               if self._uses_deepseek_lic_first_turn(case) else {}),
        }

    def _uses_deepseek_lic_first_turn(self, case: BenchmarkCase) -> bool:
        return (
            self.execution_mode_for_case(case) == "interactive_dialogue"
            and self.variant_for_case(case) == "extrinsic_verifiable"
            and self.runtime_for_case(case).deepseek_lic_first_turn
        )

    @staticmethod
    def variant_for_case(case: BenchmarkCase) -> str:
        variant = str(case.input_data.get("variant") or "extrinsic_verifiable")
        if variant not in USERLM_VARIANTS:
            raise ValidationError(f"unknown UserLM variant {variant!r}")
        return variant

    @staticmethod
    def _assistant_system_prompt(case: BenchmarkCase) -> str:
        task = case.input_data.get("assistant_task") or {}
        kind = task.get("kind") if isinstance(task, Mapping) else None
        if kind in {"arithmetic", "math"}:
            return MATH_SYSTEM_PROMPT
        if kind == "code":
            return CODE_SYSTEM_PROMPT
        if UserLMAdapter.variant_for_case(case) == "extrinsic_verifiable":
            raise ValidationError(f"unsupported LiC assistant task kind {kind!r}")
        return str(case.input_data.get("assistant_policy") or "Help the user complete the stated task.")

    def dialogue_spec_for_case(self, case: BenchmarkCase) -> DialogueSpec:
        runtime = self.runtime_for_case(case)
        variant = self.variant_for_case(case)
        target_turns = int(case.input_data.get("target_user_turns", 2))
        user_may_end = variant != "extrinsic_verifiable"
        return DialogueSpec(
            user_context={
                "generic_intent": str(case.input_data.get("intent") or ""),
                "information_shards": list(_shard_records(case)),
                "initial_user_message_hint": case.input_data.get("initial_user_message"),
                "variant": variant,
                "termination_token": END_CONVERSATION,
            },
            assistant_context={
                # Task payloads, answers and tests belong only to verification.
                "fixed_policy": self._assistant_system_prompt(case),
            },
            max_user_turns=runtime.max_user_turns,
            max_total_actions=runtime.max_total_actions,
            target_user_turns=target_turns,
            termination_token=END_CONVERSATION,
            user_may_end=user_may_end,
            assistant_may_end=False,
        )

    @staticmethod
    def execution_mode_for_case(case: BenchmarkCase) -> str:
        return str(case.input_data.get("execution_mode") or "interactive_dialogue")

    def prompt_revision_for_case(self, case: BenchmarkCase) -> str:
        adaptation = self.runtime_for_case(case).model_adapter
        if self._uses_userlm_lic_json(case):
            return f"{self.lic_prompt_revision}-{adaptation}-{USERLM_LIC_JSON_REVISION}"
        if adaptation:
            base = self.prompt_revision if self.execution_mode_for_case(case) == "section3_single_user_turn" else self.lic_prompt_revision
            return f"{base}-{adaptation}"
        if self._uses_deepseek_lic_first_turn(case):
            return f"{self.lic_prompt_revision}-deepseek-first-turn-v1"
        return (
            self.prompt_revision
            if self.execution_mode_for_case(case) == "section3_single_user_turn"
            else self.lic_prompt_revision
        )

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"UserLMAdapter cannot run {case.benchmark_id!r}")
        probe_case(case)
        if not str(case.input_data.get("intent") or "").strip():
            raise ValidationError("UserLM intent cannot be empty")
        variant = self.variant_for_case(case)
        execution_mode = self.execution_mode_for_case(case)
        if execution_mode == "section3_single_user_turn":
            if variant == "extrinsic_verifiable":
                raise ValidationError("UserLM Section 3 single-turn mode cannot use extrinsic_verifiable")
            if not isinstance(case.input_data.get("conversation_history"), str):
                raise ValidationError("UserLM Section 3 cases require conversation_history text")
            if variant == "intrinsic_role_adherence":
                choices = case.input_data.get("choices")
                if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
                    raise ValidationError("UserLM role-adherence case requires choices")
                role_adherence_score("", [str(item) for item in choices])
            if variant == "intrinsic_intent_adherence":
                for field in ("question", "assistant_suggestion_turn"):
                    if not str(case.input_data.get(field) or "").strip():
                        raise ValidationError(f"UserLM intent-adherence case requires {field}")
        elif execution_mode == "interactive_dialogue":
            _parse_shards(case)
        else:
            raise ValidationError(f"unknown UserLM execution_mode {execution_mode!r}")
        self.provenance_for_case(case)
        if execution_mode == "section3_single_user_turn":
            self.runtime_for_case(case)
            return
        state = self.environment.reset_with_spec(case, self.dialogue_spec_for_case(case), seed=0)
        if state.next_actor != EVALUATED_USER_ROLE:
            raise ValidationError("UserLM episodes must begin with the evaluated user")

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        self.validate_case(case)
        if self.execution_mode_for_case(case) == "section3_single_user_turn":
            return self.build_section3_request(case, model=model, seed=seed)
        state = self.environment.reset_with_spec(case, self.dialogue_spec_for_case(case), seed=seed)
        return self.build_user_request(case, state, model=model, seed=seed)

    def parse_response(self, case: BenchmarkCase, response: ModelResponse) -> Any:
        if self.execution_mode_for_case(case) == "section3_single_user_turn":
            output = response.text.strip()
            if not output:
                raise ParseError("UserLM Section 3 user turn is empty")
            return output
        return self._parse_user_action(case, response.text)

    def build_section3_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        history = str(case.input_data.get("conversation_history") or "")
        intent = str(case.input_data["intent"]).strip()
        prompt = (
            USERLM_FIRST_TURN_PROMPT.format(intent=intent)
            if not history.strip()
            else USERLM_SEQUENTIAL_TURN_PROMPT.format(intent=intent, conversation_history=history.rstrip())
        )
        return ModelRequest(
            request_id=f"{case.case_id}:section3:user_turn",
            messages=(ChatMessage("user", prompt, metadata={"visibility": EVALUATED_USER_ROLE}),),
            model=model,
            temperature=self.runtime_for_case(case).user_temperature,
            max_tokens=200,
            seed=seed,
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": EVALUATED_USER_ROLE,
                "variant": self.variant_for_case(case),
                "execution_mode": "section3_single_user_turn",
                "prompt_source": "UserLM Appendix Figure 9",
                "prompt_revision": "userlm-paper-figure9-transcribed-v1",
                "gold_visible": False,
            },
        )

    def build_section3_intent_judge_request(
        self,
        case: BenchmarkCase,
        output: str,
        provenance: UserLMScoringProvenance,
        *,
        seed: int,
    ) -> ModelRequest:
        if not provenance.intent_judge_model:
            raise ConfigurationError("UserLM intent-adherence judge is not configured")
        prompt = (
            "You are given a conversation between a user and an AI system.\n"
            f"User Question: {case.input_data['question']}\n"
            f"AI Response: {case.input_data['assistant_suggestion_turn']}\n"
            f"User Reply: {output}\n"
            "Classify if the user accepted the AI's suggestion or repeated their original question.\n"
            "If the user's reply repeats or rephrases their original question, output only: REFUSED\n"
            "If the user's reply follows the AI's suggestion, output only: ACCEPTED\n"
        )
        return ModelRequest(
            request_id=f"{case.case_id}:judge:intent_adherence",
            messages=(ChatMessage("user", prompt, metadata={"visibility": "evaluator"}),),
            model=provenance.intent_judge_model,
            temperature=0,
            max_tokens=16,
            seed=seed,
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": "intent_judge",
                "judge_revision": provenance.intent_judge_revision,
                "prompt_revision": provenance.intent_prompt_revision,
                "prompt_source": "UserLM Appendix Figure 10",
                "output_contract": _INTENT_JUDGE_CONTRACT,
            },
        )

    def build_user_request(
        self,
        case: BenchmarkCase,
        state: DialogueState,
        *,
        model: str,
        seed: int,
    ) -> ModelRequest:
        runtime = self.runtime_for_case(case)
        role_instruction = (
            "Act as a realistic human user pursuing the generic intent in your private context. "
            "Users may be informal and concise. Reveal relevant information naturally across turns instead "
            "of copying the full intent or all information at once. Never answer the task yourself and never "
            "write assistant, judge, or evaluator text."
        )
        if self.variant_for_case(case) == "extrinsic_verifiable" and runtime.apply_extrinsic_guardrails:
            role_instruction += (
                f" Every user message must contain {runtime.guardrail_min_words}–"
                f"{runtime.guardrail_max_words} words, must not copy "
                "the full intent verbatim, and must not repeat an earlier user message. Do not emit the "
                f"termination token {END_CONVERSATION} in this extrinsic protocol."
            )
        userlm_lic_json = self._uses_userlm_lic_json(case, runtime)
        native = runtime.model_adapter == USERLM_NATIVE and not userlm_lic_json
        adapted_json = runtime.model_adapter == COSER_FORMAT or userlm_lic_json
        action_contract = (
            "Generate only the next user utterance as plain text. Do not output JSON, field names, "
            "role labels, or explanations of these instructions."
            if native else (
                'Return exactly one JSON object with exactly action and message. '
                'Choose ONE action: "message", "end", or "refuse". '
                'Example: {"action":"message","message":"Can you help with this task?"}. '
                'For message or refuse, message must be non-empty.'
                if adapted_json else _DIALOGUE_ACTION_CONTRACT
            )
        )
        if userlm_lic_json:
            action_contract = USERLM_LIC_JSON_CONTRACT
        if native and not (
            self.variant_for_case(case) == "extrinsic_verifiable" and runtime.apply_extrinsic_guardrails
        ):
            action_contract += f" When ending the conversation, output exactly {END_CONVERSATION}."
        messages = merge_system_instructions(
            self.environment.observation(state, actor=EVALUATED_USER_ROLE, self_action_envelope=not native),
            role_instruction,
            action_contract if native or userlm_lic_json else (
                f"{action_contract} Use action=end with an empty message only when ending "
                f"via {END_CONVERSATION}."
            ),
        )
        # This is a private API instruction, not a simulated dialogue event.
        # Later turns already end with the fixed assistant's reply as API user.
        if (self._uses_deepseek_lic_first_turn(case) or runtime.model_adapter == COSER_FORMAT) and not state.public_transcript:
            messages = (*messages, ChatMessage(
                "user", _DEEPSEEK_LIC_FIRST_TURN_INSTRUCTION,
                metadata={"visibility": "transport_instruction"},
            ))
        return ModelRequest(
            request_id=f"{case.case_id}:user:{state.user_turn_count}",
            messages=tuple(messages),
            model=model,
            temperature=runtime.user_temperature,
            max_tokens=512,
            seed=seed + state.action_count,
            response_format=(None if native else json_schema_response_format(
                "user_dialogue_action", {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["message", "end", "refuse"]},
                        "message": {"type": "string"},
                    },
                    "required": ["action", "message"],
                    "additionalProperties": False,
                },
            ) if adapted_json else {"type": "json_object"}),
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": EVALUATED_USER_ROLE,
                "variant": self.variant_for_case(case),
                "gold_visible": False,
            },
        )

    def _parse_user_action(self, case: BenchmarkCase, text: str) -> DialogueAction:
        if self.runtime_for_case(case).model_adapter != USERLM_NATIVE or self._uses_userlm_lic_json(case):
            return parse_dialogue_action(text, termination_token=END_CONVERSATION)
        stripped = text.strip()
        if stripped == END_CONVERSATION:
            return DialogueAction(action="end", message="")
        if not stripped:
            raise ParseError("dialogue message requires a non-empty message")
        return DialogueAction(action="message", message=stripped)

    def build_assistant_request(
        self,
        case: BenchmarkCase,
        state: DialogueState,
        *,
        seed: int,
    ) -> ModelRequest:
        runtime = self.runtime_for_case(case)
        messages = self.environment.observation(
            state, actor=FIXED_ASSISTANT_ROLE,
            system_instruction=self._assistant_system_prompt(case),
            include_private_context=False,
        )
        return ModelRequest(
            request_id=f"{case.case_id}:assistant:{state.assistant_turn_count}",
            messages=tuple(messages),
            model=runtime.fixed_assistant_model,
            temperature=runtime.assistant_temperature,
            max_tokens=1024,
            seed=seed + state.action_count,
            response_format=None,
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": FIXED_ASSISTANT_ROLE,
                "output_protocol": "natural_language_message",
                "assistant_prompt_source_revision": LIC_SOURCE_REVISION,
                "task_payload_visible": False,
            },
        )

    @staticmethod
    def _parse_assistant_message(text: str) -> DialogueAction:
        message = text.strip()
        if not message:
            raise ParseError("UserLM fixed assistant response must be non-empty natural language")
        return DialogueAction(action="message", message=message)

    def build_shard_judge_request(
        self,
        case: BenchmarkCase,
        state: DialogueState,
        provenance: UserLMScoringProvenance,
        *,
        seed: int,
    ) -> ModelRequest:
        if not provenance.shard_judge_model:
            raise ConfigurationError("UserLM shard judge is not configured")
        required, nonrequired = _parse_shards(case)
        payload = {
            "intent": case.input_data["intent"],
            "information_shards": list(_shard_records(case)),
            "required_shards": list(required),
            "nonrequired_shards": list(nonrequired),
            "user_turns": [
                {"turn_index": index, "text": text}
                for index, text in enumerate(self._user_messages(state))
            ],
            "schema": {
                "turn_labels": [
                    {"turn_index": "integer", "revealed_shards": ["shard_id"], "additional_demand": "boolean"}
                ]
            },
        }
        output_contract = (
            "Return only one JSON object with exactly `turn_labels`. It must contain exactly one entry for every "
            f"turn_index from 0 through {len(self._user_messages(state)) - 1}. Every entry must contain exactly "
            "turn_index (integer), revealed_shards (an array containing only supplied shard IDs, without "
            "duplicates), and additional_demand (JSON boolean). Do not emit markdown or outside prose."
        )
        schema = {
            "type": "object",
            "properties": {
                "turn_labels": {
                    "type": "array",
                    "minItems": len(self._user_messages(state)),
                    "maxItems": len(self._user_messages(state)),
                    "items": {
                        "type": "object",
                        "properties": {
                            "turn_index": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": max(0, len(self._user_messages(state)) - 1),
                            },
                            "revealed_shards": {
                                "type": "array",
                                "items": {"type": "string", "enum": list(required) + list(nonrequired)},
                            },
                            "additional_demand": {"type": "boolean"},
                        },
                        "required": ["turn_index", "revealed_shards", "additional_demand"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["turn_labels"],
            "additionalProperties": False,
        }
        return ModelRequest(
            request_id=f"{case.case_id}:judge:shards",
            messages=(
                ChatMessage(
                    "system",
                    "For each user turn, label only the supplied intent shards that the turn explicitly or "
                    "unambiguously reveals. A shard may appear on multiple turns when repeated. Mark "
                    "additional_demand=true only when the turn asks for a genuinely new requirement not represented "
                    f"by any supplied shard. {output_contract}",
                    metadata={"visibility": "evaluator"},
                ),
                ChatMessage("user", canonical_json(payload), metadata={"visibility": "evaluator"}),
            ),
            model=provenance.shard_judge_model,
            temperature=0,
            max_tokens=1024,
            seed=seed,
            response_format=json_schema_response_format("userlm_shard_labels", schema),
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": "shard_judge",
                "judge_revision": provenance.shard_judge_revision,
                "prompt_revision": provenance.shard_prompt_revision,
                "replayed": provenance.replayed,
                "prompt_source": "indieval reconstruction of UserLM extrinsic shard metrics",
                "output_contract": output_contract,
            },
        )

    def build_intent_judge_request(
        self,
        case: BenchmarkCase,
        state: DialogueState,
        provenance: UserLMScoringProvenance,
        *,
        seed: int,
    ) -> ModelRequest:
        if not provenance.intent_judge_model:
            raise ConfigurationError("UserLM intent-adherence judge is not configured")
        turns = list(state.public_transcript)
        if len(turns) < 3:
            raise ValidationError("intent-adherence probe needs user, assistant, user turns")
        prompt = (
            "You are given a conversation between a user and an AI system.\n\n"
            f"User Question: {turns[0].content}\n"
            f"AI Response: {turns[1].content}\n"
            f"User Reply: {turns[2].content}\n\n"
            "Classify if the user accepted the AI's suggestion or repeated their original question.\n"
            "If the user's reply repeats or rephrases their original question, output only: REFUSED\n"
            "If the user's reply follows the AI's suggestion, output only: ACCEPTED"
        )
        return ModelRequest(
            request_id=f"{case.case_id}:judge:intent_adherence",
            messages=(ChatMessage("user", prompt, metadata={"visibility": "evaluator"}),),
            model=provenance.intent_judge_model,
            temperature=0,
            max_tokens=16,
            seed=seed,
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": "intent_judge",
                "judge_revision": provenance.intent_judge_revision,
                "prompt_revision": provenance.intent_prompt_revision,
                "replayed": provenance.replayed,
                "prompt_source": "UserLM Appendix Figure 10",
                "output_contract": _INTENT_JUDGE_CONTRACT,
            },
        )

    @staticmethod
    def _user_messages(state: DialogueState) -> tuple[str, ...]:
        return tuple(
            str(event.content)
            for event in state.public_transcript
            if event.actor == EVALUATED_USER_ROLE and event.kind in {"message", "seed_message"}
        )

    @staticmethod
    def _assistant_messages(state: DialogueState) -> tuple[str, ...]:
        return tuple(
            str(event.content)
            for event in state.public_transcript
            if event.actor == FIXED_ASSISTANT_ROLE and event.kind in {"message", "seed_message", "end"}
        )

    @staticmethod
    def _guardrail_rejection(
        case: BenchmarkCase,
        state: DialogueState,
        action: Any,
        runtime: UserLMRuntimeProvenance,
    ) -> str | None:
        """Return the paper guardrail violated by an extrinsic user output."""

        if not runtime.apply_extrinsic_guardrails:
            return None
        if action.action == "end":
            return "termination_token_prohibited"
        if action.action != "message":
            return None
        words = _words(action.message)
        if not runtime.guardrail_min_words <= len(words) <= runtime.guardrail_max_words:
            return "word_count_out_of_range"
        normalize = lambda value: " ".join(_words(value))
        normalized = normalize(action.message)
        if normalized == normalize(str(case.input_data.get("intent") or "")):
            return "verbatim_intent_copy"
        if normalized in {normalize(message) for message in UserLMAdapter._user_messages(state)}:
            return "verbatim_user_turn_repeat"
        return None

    @staticmethod
    def _guardrail_feedback(
        rejection: str,
        action: Any,
        runtime: UserLMRuntimeProvenance,
    ) -> str:
        if rejection == "word_count_out_of_range":
            return (
                f"word_count_out_of_range: received {len(_words(action.message))} words; "
                f"required {runtime.guardrail_min_words}–{runtime.guardrail_max_words}"
            )
        explanations = {
            "termination_token_prohibited": "termination_token_prohibited: send a user message instead",
            "verbatim_intent_copy": "verbatim_intent_copy: reveal only part of the intent naturally",
            "verbatim_user_turn_repeat": "verbatim_user_turn_repeat: write a new user message",
        }
        return explanations.get(rejection, rejection)

    def _generate_with_retries(
        self,
        backend: ModelBackend,
        request: ModelRequest,
        runtime: UserLMRuntimeProvenance,
        audits: list[Mapping[str, Any]],
    ) -> ModelResponse:
        last_error: BackendError | None = None
        for attempt in range(runtime.max_retries + 1):
            try:
                response = backend.generate(request)
                if response.latency_ms is not None and response.latency_ms > runtime.request_timeout_seconds * 1000:
                    raise BackendTimeoutError(
                        f"response latency {response.latency_ms}ms exceeded {runtime.request_timeout_seconds}s"
                    )
                audits.append({"request_id": request.request_id, "attempt": attempt, "status": "completed"})
                return response
            except BackendError as exc:
                last_error = exc
                audits.append(
                    {
                        "request_id": request.request_id,
                        "attempt": attempt,
                        "status": "retry" if attempt < runtime.max_retries else "failed",
                        "kind": type(exc).__name__,
                    }
                )
        assert last_error is not None
        raise last_error

    def _generate_and_parse_contract(
        self,
        *,
        backend: ModelBackend,
        request: ModelRequest,
        runtime: UserLMRuntimeProvenance,
        audits: list[Mapping[str, Any]],
        responses: list[ModelResponse],
        parser: Callable[[ModelResponse], Any],
        contract: str,
    ) -> Any:
        current_request = request
        for contract_attempt in range(DEFAULT_CONTRACT_RETRIES + 1):
            response = self._generate_with_retries(backend, current_request, runtime, audits)
            responses.append(response)
            try:
                return parser(response)
            except ParseError as exc:
                audits.append(
                    {
                        "request_id": request.request_id,
                        "contract_attempt": contract_attempt,
                        "status": (
                            "contract_retry"
                            if contract_attempt < DEFAULT_CONTRACT_RETRIES
                            else "contract_failed"
                        ),
                        "kind": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                if contract_attempt >= DEFAULT_CONTRACT_RETRIES:
                    raise
                current_request = corrective_retry_request(
                    request,
                    attempt=contract_attempt + 1,
                    reason=str(exc),
                    contract=contract,
                    previous_output=response.text,
                    request_suffix="judge_contract_retry",
                )
        raise AssertionError("unreachable UserLM judge contract retry loop")

    def replay_responses(self, case: BenchmarkCase, *, seed: int) -> Mapping[str, Any]:
        self.validate_case(case)
        replay = case.metadata.get("replay")
        if not isinstance(replay, Mapping):
            raise ConfigurationError(f"fixture {case.case_id} lacks UserLM replay metadata")
        steps = replay.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise ConfigurationError("UserLM replay.steps must be an array")
        runtime = self.runtime_for_case(case)
        scoring = self.provenance_for_case(case)
        state = self.environment.reset_with_spec(case, self.dialogue_spec_for_case(case), seed=seed)
        responses: dict[str, Any] = {}
        for index, step in enumerate(steps):
            if state.terminal or not isinstance(step, Mapping):
                raise ConfigurationError(f"invalid UserLM replay step #{index}")
            actor = str(step.get("actor") or "")
            if actor != state.next_actor:
                raise ConfigurationError(f"UserLM replay actor {actor!r} does not match {state.next_actor!r}")
            output = step.get("output")
            if not isinstance(output, Mapping):
                raise ConfigurationError(f"UserLM replay step #{index} requires output object")
            request = (
                self.build_user_request(case, state, model="offline-replay", seed=seed)
                if actor == EVALUATED_USER_ROLE
                else self.build_assistant_request(case, state, seed=seed)
            )
            if actor == FIXED_ASSISTANT_ROLE:
                if set(output) != {"action", "message"} or output.get("action") != "message":
                    raise ConfigurationError(
                        f"UserLM fixed-assistant replay step #{index} must contain one message action"
                    )
                message = output.get("message")
                if not isinstance(message, str) or not message.strip():
                    raise ConfigurationError(
                        f"UserLM fixed-assistant replay step #{index} requires non-empty message text"
                    )
                text = message
                action = self._parse_assistant_message(text)
            else:
                text = canonical_json(output)
                action = parse_dialogue_action(text, termination_token=END_CONVERSATION)
            responses[request.request_id or ""] = {
                "text": text,
                "finish_reason": "replayed",
                "usage": step.get("usage") or {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
            self.environment.apply(
                state,
                actor=actor,
                action=action,
            )
        if not state.terminal or not state.protocol_complete:
            raise ConfigurationError("UserLM replay steps must produce a complete dialogue")
        judges = replay.get("judges") or {}
        if not isinstance(judges, Mapping):
            raise ConfigurationError("UserLM replay.judges must be an object")
        if self.variant_for_case(case) == "extrinsic_verifiable" and scoring.shard_judge_model:
            request = self.build_shard_judge_request(case, state, scoring, seed=seed)
            if "shards" not in judges:
                raise ConfigurationError("UserLM fixture requires a shards judge replay")
            responses[request.request_id or ""] = {"text": canonical_json(judges["shards"]), "finish_reason": "replayed"}
        if self.variant_for_case(case) == "intrinsic_intent_adherence" and scoring.intent_judge_model:
            request = self.build_intent_judge_request(case, state, scoring, seed=seed)
            if "intent_adherence" not in judges:
                raise ConfigurationError("UserLM fixture requires an intent-adherence judge replay")
            responses[request.request_id or ""] = {"text": str(judges["intent_adherence"]), "finish_reason": "replayed"}
        return responses

    @staticmethod
    def _parse_shard_judgment(
        case: BenchmarkCase,
        state: DialogueState,
        text: str,
    ) -> Mapping[str, Any]:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError("UserLM shard judge must return JSON") from exc
        if not isinstance(value, Mapping) or set(value) != {"turn_labels"}:
            raise ParseError("UserLM shard judge requires exactly turn_labels")
        labels = value.get("turn_labels")
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise ParseError("UserLM turn_labels must be an array")
        user_messages = UserLMAdapter._user_messages(state)
        required, nonrequired = _parse_shards(case)
        allowed = set(required) | set(nonrequired)
        parsed = []
        seen_indices: set[int] = set()
        for raw in labels:
            if not isinstance(raw, Mapping) or set(raw) != {"turn_index", "revealed_shards", "additional_demand"}:
                raise ParseError("each UserLM turn label has an invalid schema")
            turn_index = raw.get("turn_index")
            shards = raw.get("revealed_shards")
            additional = raw.get("additional_demand")
            if isinstance(turn_index, bool) or not isinstance(turn_index, int) or not 0 <= turn_index < len(user_messages):
                raise ParseError("UserLM judge turn_index is out of range")
            if turn_index in seen_indices or not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
                raise ParseError("UserLM judge has duplicate turn indices or invalid revealed_shards")
            shard_values = [str(shard) for shard in shards]
            if len(shard_values) != len(set(shard_values)) or not set(shard_values) <= allowed:
                raise ParseError("UserLM judge returned duplicate or unknown shard IDs")
            if not isinstance(additional, bool):
                raise ParseError("UserLM additional_demand must be boolean")
            seen_indices.add(turn_index)
            parsed.append(
                {"turn_index": turn_index, "revealed_shards": shard_values, "additional_demand": additional}
            )
        if seen_indices != set(range(len(user_messages))):
            raise ParseError("UserLM shard judge must label every user turn exactly once")
        return {"turn_labels": sorted(parsed, key=lambda item: item["turn_index"])}

    @staticmethod
    def _shard_metrics(case: BenchmarkCase, judgment: Mapping[str, Any]) -> Mapping[str, Any]:
        required, nonrequired = _parse_shards(case)
        counts: Counter[str] = Counter()
        additional = False
        for label in judgment["turn_labels"]:
            counts.update(label["revealed_shards"])
            additional = additional or bool(label["additional_demand"])
        return {
            "intent_coverage": sum(counts[shard] > 0 for shard in required) / len(required),
            "repeat_required": any(counts[shard] >= 2 for shard in required),
            "skip_nonrequired": (any(counts[shard] == 0 for shard in nonrequired) if nonrequired else None),
            "additional_demands": additional,
            "shard_counts": dict(counts),
        }

    def _task_verifier(self, case: BenchmarkCase, state: DialogueState, runtime: UserLMRuntimeProvenance) -> tuple[float | None, Mapping[str, Any]]:
        assistant_messages = self._assistant_messages(state)
        assistant_text = "\n".join(assistant_messages)
        task = case.input_data.get("assistant_task")
        if not isinstance(task, Mapping):
            return None, {"availability": "unavailable", "reason": "missing_assistant_task"}
        if task.get("kind") == "arithmetic" and isinstance(case.gold, Mapping) and "task_answer" in case.gold:
            # GSM8K answers and model responses may use conventional thousands
            # separators (for example ``114,200``).  Removing commas only from
            # literals that matched the strict grouped-number grammar preserves
            # exact numeric scoring without accepting malformed punctuation.
            expected_text = str(case.gold["task_answer"]).strip()
            if _NUMBER.fullmatch(expected_text) is None:
                raise ValidationError(
                    f"UserLM arithmetic task_answer is not a valid numeric literal: {expected_text!r}"
                )
            expected = float(expected_text.replace(",", ""))
            observed = [float(value.replace(",", "")) for value in _NUMBER.findall(assistant_text)]
            passed = any(math.isclose(value, expected, rel_tol=0, abs_tol=1e-9) for value in observed)
            return float(passed), {
                "availability": "available",
                "verifier_revision": "indieval-exact-numeric-v1",
                "configured_suite_verifier": runtime.verifier_revision,
            }
        if task.get("kind") == "code" and self._code_task_verifier is not None:
            if not assistant_messages:
                return None, {
                    "availability": "unavailable",
                    "reason": "no_assistant_completion",
                    "verifier_revision": runtime.verifier_revision,
                }
            attempts: list[tuple[int, float | None, Mapping[str, Any]]] = []
            for message_index, assistant_message in enumerate(assistant_messages):
                score, verification = self._code_task_verifier.verify_case(case, assistant_message)
                attempts.append((message_index, score, dict(verification)))

            passing = [attempt for attempt in attempts if attempt[1] == 1.0]
            unavailable = [attempt for attempt in attempts if attempt[1] is None]
            if passing:
                aggregate_score: float | None = 1.0
                representative = passing[0]
            elif unavailable:
                # Without a passing turn, one unavailable verification means that
                # the official "any assistant solution passes" predicate cannot be
                # decided without conflating infrastructure failure with model failure.
                aggregate_score = None
                representative = unavailable[0]
            else:
                aggregate_score = 0.0
                # Preserve the former last-message evidence at the top level when
                # every assistant turn was successfully checked and failed.
                representative = attempts[-1]

            summary_keys = (
                "status",
                "passed",
                "verifier_passed",
                "reason",
                "completion_sha256",
                "completion_mode",
                "cache_key",
                "cache_hit",
                "runtime_ms",
                "exit_code",
                "executor_reason",
            )
            per_message_results = [
                {
                    "assistant_message_index": message_index,
                    "score": score,
                    "availability": "available" if score is not None else "unavailable",
                    **{key: verification[key] for key in summary_keys if key in verification},
                }
                for message_index, score, verification in attempts
            ]
            selected_index, _, selected_verification = representative
            return aggregate_score, {
                **dict(selected_verification),
                "availability": "available" if aggregate_score is not None else "unavailable",
                "aggregation": "one_if_any_assistant_message_passes",
                "assistant_message_count": len(assistant_messages),
                "evaluated_message_count": len(attempts),
                "passing_message_indices": [message_index for message_index, _, _ in passing],
                "selected_message_index": selected_index,
                "per_message_results": per_message_results,
            }
        evaluation = case.metadata.get("evaluation")
        if isinstance(evaluation, Mapping) and "verifier_passed" in evaluation:
            if evaluation.get("verifier_revision") != runtime.verifier_revision:
                raise ValidationError("UserLM verifier result revision does not match runtime provenance")
            passed = evaluation["verifier_passed"]
            if not isinstance(passed, bool):
                raise ValidationError("UserLM verifier_passed must be boolean")
            return float(passed), {
                "availability": "available",
                "verifier_revision": runtime.verifier_revision,
                "source": "trusted_derived_input",
            }
        return None, {
            "availability": "unavailable",
            "reason": "no executable or trusted task verifier result",
            "verifier_revision": runtime.verifier_revision,
        }

    def _failure_result(
        self,
        case: BenchmarkCase,
        state: DialogueState,
        *,
        run_id: str,
        repetition: int,
        responses: Sequence[ModelResponse],
        audits: Sequence[Mapping[str, Any]],
        actor: str,
        stage: str,
        kind: str,
        message: str,
        retryable: bool,
    ) -> CaseResult:
        if not state.terminal:
            self.environment.force_terminate(
                state,
                actor=actor,
                reason=kind,
                stage=stage,
                kind=kind,
            )
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.FAILED,
            prediction={"public_transcript": self._public_prediction(state), "terminal_reason": state.terminal_reason},
            trace=tuple(state.trace),
            model_response=responses[-1] if responses else None,
            error=ErrorState(stage, kind, message, retryable=retryable, details={"request_audits": list(audits)}),
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "episode_complete": False,
                "terminal_reason": state.terminal_reason,
                "variant": self.variant_for_case(case),
                "source_task": _source_task(case),
                "request_audits": list(audits),
                "partial_user_turn_count": state.user_turn_count,
            },
        )

    def _target_capability_metrics(
        self,
        case: BenchmarkCase,
        *,
        user_turn_count: int,
        failure: Mapping[str, Any],
    ) -> tuple[MetricValue, ...]:
        variant = self.variant_for_case(case)
        extrinsic = variant == "extrinsic_verifiable"
        prism = variant == "intrinsic_prism"
        role = variant == "intrinsic_role_adherence"
        intent = variant == "intrinsic_intent_adherence"
        metadata = {"target_output_failure": dict(failure)}
        not_applicable = {"availability": "not_applicable"}
        has_nonrequired_shards = False
        if extrinsic:
            _required_shards, nonrequired_shards = _parse_shards(case)
            has_nonrequired_shards = bool(nonrequired_shards)
        return (
            _metric(
                "userlm.extrinsic.intent_coverage",
                0.0 if extrinsic else None,
                metadata={**metadata, **({} if extrinsic else not_applicable)},
            ),
            _metric(
                "userlm.extrinsic.repeat_required",
                0.0 if extrinsic else None,
                metadata={**metadata, **({} if extrinsic else not_applicable)},
            ),
            _metric(
                "userlm.extrinsic.skip_non_required",
                0.0 if extrinsic and has_nonrequired_shards else None,
                metadata={
                    **metadata,
                    **({} if extrinsic and has_nonrequired_shards else not_applicable),
                    "not_applicable_when_all_shards_required": True,
                },
            ),
            _metric(
                "userlm.extrinsic.additional_demands",
                0.0 if extrinsic else None,
                metadata={**metadata, **({} if extrinsic else not_applicable)},
            ),
            _metric(
                "userlm.extrinsic.assistant_task_score",
                0.0 if extrinsic else None,
                metadata={**metadata, **({} if extrinsic else not_applicable)},
            ),
            _metric(
                "userlm.intrinsic.intent_decomposition_overlap",
                1.0 if prism else None,
                direction="lower_is_better",
                metadata={
                    **metadata,
                    **({} if prism else not_applicable),
                    "invalid_output_worst_case_value": 1.0,
                },
            ),
            _metric(
                "userlm.intrinsic.role_adherence",
                1.0 if role else None,
                metadata={
                    **metadata,
                    **({} if role else not_applicable),
                    "invalid_output_matches_no_valid_choice_attempt": True,
                },
            ),
            _metric(
                "userlm.intrinsic.intent_adherence",
                0.0 if intent else None,
                metadata={**metadata, **({} if intent else not_applicable)},
            ),
            _metric(
                "userlm.intrinsic.ai_detector_human_likelihood",
                0.0 if prism else None,
                metadata={**metadata, **({} if prism else not_applicable)},
            ),
            _metric("userlm.episode.user_turn_count", user_turn_count, unit="turns", direction="descriptive"),
        )

    def _target_capability_result(
        self,
        case: BenchmarkCase,
        state: DialogueState,
        *,
        run_id: str,
        repetition: int,
        responses: Sequence[ModelResponse],
        audits: Sequence[Mapping[str, Any]],
        stage: str,
        kind: str,
        message: str,
    ) -> CaseResult:
        if not state.terminal:
            self.environment.force_terminate(
                state,
                actor=EVALUATED_USER_ROLE,
                reason=kind,
                stage=stage,
                kind=kind,
            )
        failure = {
            "stage": stage,
            "kind": kind,
            "message": message,
            "retryable": False,
            "scoring_policy": "target_capability_failure_scores_zero",
        }
        user_messages = self._user_messages(state)
        raw = responses[-1].text.strip() if responses else ""
        all_user_text = " ".join(user_messages) or raw
        variant = self.variant_for_case(case)
        actual_end = bool(case.input_data.get("is_last_turn")) if variant == "intrinsic_prism" else None
        metrics = self._target_capability_metrics(
            case,
            user_turn_count=state.user_turn_count,
            failure=failure,
        )
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction={"public_transcript": self._public_prediction(state), "terminal_reason": state.terminal_reason, "parsed": False},
            metrics=metrics,
            trace=tuple(state.trace),
            model_response=responses[-1] if responses else None,
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "episode_complete": True,
                "natural_termination": False,
                "terminal_reason": state.terminal_reason,
                "variant": variant,
                "source_task": _source_task(case),
                "evaluated_model_role": EVALUATED_USER_ROLE,
                "fixed_assistant_identity": self.assistant_or_partner_identity_for_case(case),
                "environment_identity": self.environment_identity_for_case(case),
                "scoring_provenance": self.provenance_for_case(case).to_dict(),
                "request_audits": list(audits),
                "target_output_failure": failure,
                "judge_records": {},
                "judge_errors": {},
                "userlm": {
                    "first_turn": user_messages[0] if user_messages else raw,
                    "all_user_text": all_user_text,
                    "user_turn_count": state.user_turn_count,
                    "termination_actual": actual_end,
                    "termination_predicted": False,
                    "shard_metrics": {},
                },
            },
        )

    @staticmethod
    def _public_prediction(state: DialogueState) -> list[Mapping[str, Any]]:
        return [
            {"turn": event.turn, "actor": event.actor, "kind": event.kind, "message": event.content}
            for event in state.public_transcript
        ]

    def _execute_section3_case(
        self,
        case: BenchmarkCase,
        *,
        backend: ModelBackend,
        run_id: str,
        seed: int,
        model: str,
        repetition: int,
        previous_result: CaseResult | None = None,
    ) -> CaseResult:
        runtime = self.runtime_for_case(case)
        provenance = self.provenance_for_case(case)
        variant = self.variant_for_case(case)
        request = self.build_section3_request(case, model=model, seed=seed)
        audits: list[Mapping[str, Any]] = []
        responses: list[ModelResponse] = []
        try:
            if previous_result is None:
                response = self._generate_with_retries(backend, request, runtime, audits)
                responses.append(response)
                output = str(self.parse_response(case, response))
            else:
                output = str(previous_result.prediction["user_turn"])
        except EpisodeTokenBudgetExhausted as exc:
            failure = {
                "stage": "token_budget_exhausted",
                "kind": "token_budget_exhausted",
                "message": str(exc),
                "retryable": False,
                "scoring_policy": "no_generation_scores_as_target_capability_outcome",
            }
            actual_end = (
                bool(case.input_data.get("is_last_turn"))
                if variant == "intrinsic_prism"
                else None
            )
            return CaseResult(
                run_id=run_id,
                benchmark_id=self.benchmark_id,
                case_id=case.case_id,
                group_id=case.group_id,
                repetition=repetition,
                status=ResultStatus.COMPLETED,
                prediction={"user_turn": "", "predicted_end": False, "parsed": False},
                metrics=self._target_capability_metrics(
                    case, user_turn_count=0, failure=failure
                ),
                latency_ms=sum(item.latency_ms or 0 for item in responses),
                token_usage=combine_usage(item.usage for item in responses),
                metadata={
                    "episode_complete": True,
                    "natural_termination": False,
                    "terminal_reason": "token_budget_exhausted",
                    "budget_termination": exc.details(),
                    "variant": variant,
                    "source_task": _source_task(case),
                    "execution_mode": "section3_single_user_turn",
                    "evaluated_model_role": EVALUATED_USER_ROLE,
                    "scoring_provenance": provenance.to_dict(),
                    "request_audits": audits,
                    "target_output_failure": failure,
                    "judge_records": {},
                    "judge_errors": {},
                    "userlm": {
                        "first_turn": "",
                        "all_user_text": "",
                        "user_turn_count": 0,
                        "source_turn_index": int(case.input_data.get("turn", 0)),
                        "include_first_turn_diversity": False,
                        "termination_actual": actual_end,
                        "termination_predicted": False,
                        "shard_metrics": {},
                    },
                },
            )
        except BackendError as exc:
            return CaseResult(
                run_id=run_id,
                benchmark_id=self.benchmark_id,
                case_id=case.case_id,
                group_id=case.group_id,
                repetition=repetition,
                status=ResultStatus.FAILED,
                error=ErrorState("section3_user_generation", type(exc).__name__, str(exc), retryable=isinstance(exc, BackendError)),
                model_response=responses[-1] if responses else None,
                latency_ms=sum(item.latency_ms or 0 for item in responses),
                token_usage=combine_usage(item.usage for item in responses),
                metadata={
                    "variant": variant,
                    "source_task": _source_task(case),
                    "execution_mode": "section3_single_user_turn",
                    "request_audits": audits,
                },
            )
        except ParseError as exc:
            failure = {
                "stage": "section3_user_generation",
                "kind": "parse_failure",
                "message": str(exc),
                "retryable": False,
                "scoring_policy": "target_capability_failure_scores_zero",
            }
            raw = responses[-1].text if responses else ""
            actual_end = bool(case.input_data.get("is_last_turn")) if variant == "intrinsic_prism" else None
            return CaseResult(
                run_id=run_id,
                benchmark_id=self.benchmark_id,
                case_id=case.case_id,
                group_id=case.group_id,
                repetition=repetition,
                status=ResultStatus.COMPLETED,
                prediction={"user_turn": raw, "predicted_end": False, "parsed": False},
                metrics=self._target_capability_metrics(case, user_turn_count=0, failure=failure),
                model_response=responses[-1] if responses else None,
                latency_ms=sum(item.latency_ms or 0 for item in responses),
                token_usage=combine_usage(item.usage for item in responses),
                metadata={
                    "episode_complete": True,
                    "natural_termination": False,
                    "variant": variant,
                    "source_task": _source_task(case),
                    "execution_mode": "section3_single_user_turn",
                    "evaluated_model_role": EVALUATED_USER_ROLE,
                    "scoring_provenance": provenance.to_dict(),
                    "request_audits": audits,
                    "target_output_failure": failure,
                    "judge_records": {},
                    "judge_errors": {},
                    "userlm": {
                        "first_turn": raw,
                        "all_user_text": raw,
                        "user_turn_count": 0,
                        "source_turn_index": int(case.input_data.get("turn", 0)),
                        "include_first_turn_diversity": variant == "intrinsic_prism" and int(case.input_data.get("turn", 0)) == 0,
                        "termination_actual": actual_end,
                        "termination_predicted": False,
                        "shard_metrics": {},
                    },
                },
            )

        intent_adherence: float | None = None
        judge_records: dict[str, Any] = {}
        judge_errors: dict[str, Any] = {}
        if variant == "intrinsic_intent_adherence" and provenance.intent_judge_model:
            judge_request = self.build_section3_intent_judge_request(
                case, output, provenance, seed=seed
            )
            try:
                verdict = (previous_result.metadata.get("judge_records", {}).get("intent_adherence", {}).get("verdict") if previous_result else None)
                if verdict is None:
                    verdict = self._generate_and_parse_contract(
                        backend=backend,
                        request=judge_request,
                        runtime=runtime,
                        audits=audits,
                        responses=responses,
                        parser=_parse_intent_verdict,
                        contract=_INTENT_JUDGE_CONTRACT,
                    )
                intent_adherence = 1.0 if verdict == "REFUSED" else 0.0
                judge_records["intent_adherence"] = {"status": "available", "verdict": verdict}
            except (BackendError, ParseError) as exc:
                judge_errors["intent_adherence"] = {"kind": type(exc).__name__, "message": str(exc)}
                judge_records["intent_adherence"] = {"status": "unavailable"}

        decomposition = (
            intent_decomposition_overlap(str(case.input_data["intent"]), [output])
            if variant == "intrinsic_prism" and output != END_CONVERSATION
            else None
        )
        role_score: float | None = None
        mentioned_choices: int | None = None
        if variant == "intrinsic_role_adherence":
            role_score, mentioned_choices = role_adherence_score(
                output, [str(item) for item in case.input_data["choices"]]
            )
        predicted_end = output == END_CONVERSATION
        actual_end = bool(case.input_data.get("is_last_turn")) if variant == "intrinsic_prism" else None
        if previous_result is None:
            ai_detector_score, ai_detector_evidence = self._ai_detector_score(
                case,
                variant=variant,
                text=output,
                provenance=provenance,
            )
        else:
            ai_detector_score = next((m.value for m in previous_result.metrics if m.name == "userlm.intrinsic.ai_detector_human_likelihood"), None)
            ai_detector_evidence = previous_result.metadata.get("ai_detector", {})

        metrics = (
            _metric("userlm.extrinsic.intent_coverage", None),
            _metric("userlm.extrinsic.repeat_required", None),
            _metric("userlm.extrinsic.skip_non_required", None),
            _metric("userlm.extrinsic.additional_demands", None),
            _metric("userlm.extrinsic.assistant_task_score", None),
            _metric(
                "userlm.intrinsic.intent_decomposition_overlap",
                decomposition,
                direction="lower_is_better",
                metadata={
                    "formula": "mean_per_turn_stopword_filtered_unique_1gram_intersection_over_user_turn",
                    "protocol_status": "UserLM_paper_metric_reproduced",
                },
            ),
            _metric(
                "userlm.intrinsic.role_adherence",
                role_score,
                metadata={"mentioned_choice_count": mentioned_choices, "attempt_rule": "exactly_one_or_two_choice_texts"},
            ),
            _metric(
                "userlm.intrinsic.intent_adherence",
                intent_adherence,
                metadata={
                    "judge_model": provenance.intent_judge_model,
                    "judge_revision": provenance.intent_judge_revision,
                    "prompt_revision": provenance.intent_prompt_revision,
                    "prompt_source": "UserLM Appendix Figure 10",
                },
            ),
            _metric(
                "userlm.intrinsic.ai_detector_human_likelihood",
                ai_detector_score,
                metadata={
                    "detector_model": provenance.ai_detector_model,
                    "detector_revision": provenance.ai_detector_revision,
                    "detector_contract_revision": provenance.ai_detector_contract_revision,
                    **dict(ai_detector_evidence),
                },
            ),
            _metric("userlm.episode.user_turn_count", 1, unit="turns", direction="descriptive"),
        )
        turn_index = int(case.input_data.get("turn", 0))
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction={"user_turn": output, "predicted_end": predicted_end},
            metrics=metrics,
            model_response=responses[-1] if responses else None,
            latency_ms=sum(item.latency_ms or 0 for item in responses),
            token_usage=combine_usage(item.usage for item in responses),
            metadata={
                "episode_complete": True,
                "variant": variant,
                "source_task": _source_task(case),
                "execution_mode": "section3_single_user_turn",
                "evaluated_model_role": EVALUATED_USER_ROLE,
                "prompt_provenance": {
                    "generation": "UserLM Appendix Figure 9 / transcribed v1",
                    "intent_judge": "UserLM Appendix Figure 10 / transcribed v1",
                },
                "scoring_provenance": provenance.to_dict(),
                "request_audits": audits,
                "judge_records": judge_records,
                "judge_errors": judge_errors,
                "ai_detector": dict(ai_detector_evidence),
                "userlm": {
                    "first_turn": output,
                    "all_user_text": output,
                    "user_turn_count": 1,
                    "source_turn_index": turn_index,
                    "include_first_turn_diversity": variant == "intrinsic_prism" and turn_index == 0,
                    "termination_actual": actual_end,
                    "termination_predicted": predicted_end,
                    "shard_metrics": {},
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
        previous_result: CaseResult | None = None,
    ) -> CaseResult:
        self.validate_case(case)
        if self.execution_mode_for_case(case) == "section3_single_user_turn":
            return self._execute_section3_case(
                case,
                backend=backend,
                run_id=run_id,
                seed=seed,
                model=model,
                repetition=repetition,
                previous_result=previous_result,
            )
        runtime = self.runtime_for_case(case)
        provenance = self.provenance_for_case(case)
        variant = self.variant_for_case(case)
        state = self.environment.reset_with_spec(case, self.dialogue_spec_for_case(case), seed=seed)
        responses: list[ModelResponse] = []
        audits: list[Mapping[str, Any]] = []
        budget_termination: Mapping[str, Any] | None = None

        if previous_result is not None:
            from ..judge_resume import restore_judge_state
            restore_judge_state(state, previous_result)
        while previous_result is None and not state.terminal:
            actor = state.next_actor
            assert actor is not None
            if actor == EVALUATED_USER_ROLE and state.user_turn_count >= runtime.max_user_turns:
                return self._target_capability_result(
                    case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                    stage="user_turn_limit", kind="turn_limit",
                    message=f"evaluated user exceeded {runtime.max_user_turns} turns",
                )
            base_request = (
                self.build_user_request(case, state, model=model, seed=seed)
                if actor == EVALUATED_USER_ROLE
                else self.build_assistant_request(case, state, seed=seed)
            )
            stage_prefix = "user" if actor == EVALUATED_USER_ROLE else "assistant"
            guardrail_attempt = 0
            guardrail_feedback: str | None = None
            previous_guardrail_output: str | None = None
            while True:
                request = (
                    base_request
                    if guardrail_attempt == 0
                    else corrective_retry_request(
                        base_request,
                        attempt=guardrail_attempt,
                        reason=guardrail_feedback or "guardrail violation",
                        contract=(
                            USERLM_LIC_JSON_CONTRACT
                            if self._uses_userlm_lic_json(case, runtime)
                            else
                            "Generate a different user utterance as plain text, without JSON."
                            if runtime.model_adapter == USERLM_NATIVE
                            else ('Return a different valid action JSON. Choose ONE action: "message", '
                                  '"end", or "refuse"; include a message string and no other fields.')
                            if runtime.model_adapter == COSER_FORMAT
                            else f"Generate a different valid user action. {_DIALOGUE_ACTION_CONTRACT}"
                        ),
                        previous_output=previous_guardrail_output,
                        request_suffix="guardrail_regen",
                    )
                )
                try:
                    response = self._generate_with_retries(backend, request, runtime, audits)
                    responses.append(response)
                except EpisodeTokenBudgetExhausted as exc:
                    budget_termination = exc.details()
                    self.environment.terminate_capability_limit(
                        state,
                        actor=actor,
                        reason="token_budget_exhausted",
                        details=budget_termination,
                    )
                    break
                except BackendStructuredOutputError as exc:
                    if actor == EVALUATED_USER_ROLE:
                        return self._target_capability_result(
                            case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                            stage="user_parse", kind="invalid_structured_output", message=str(exc),
                        )
                    return self._failure_result(
                        case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                        actor=actor, stage=f"{stage_prefix}_backend", kind=type(exc).__name__,
                        message=str(exc), retryable=False,
                    )
                except BackendError as exc:
                    return self._failure_result(
                        case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                        actor=actor, stage=f"{stage_prefix}_backend", kind=type(exc).__name__,
                        message=str(exc), retryable=True,
                    )
                try:
                    action = (
                        self._parse_user_action(case, response.text)
                        if actor == EVALUATED_USER_ROLE
                        else self._parse_assistant_message(response.text)
                    )
                except ParseError as exc:
                    kind = "empty_turn" if "non-empty" in str(exc) else "invalid_action"
                    if actor == EVALUATED_USER_ROLE:
                        return self._target_capability_result(
                            case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                            stage="user_parse", kind=kind, message=str(exc),
                        )
                    return self._failure_result(
                        case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                        actor=actor, stage=f"{stage_prefix}_parse", kind=kind, message=str(exc), retryable=False,
                    )
                rejection = (
                    self._guardrail_rejection(case, state, action, runtime)
                    if actor == EVALUATED_USER_ROLE and variant == "extrinsic_verifiable"
                    else None
                )
                if rejection is None:
                    break
                audits.append(
                    {
                        "request_id": request.request_id,
                        "status": "guardrail_rejected",
                        "guardrail": rejection,
                        "regeneration": guardrail_attempt,
                        "observed_word_count": len(_words(action.message)) if action.action == "message" else 0,
                    }
                )
                if guardrail_attempt >= runtime.max_guardrail_regenerations:
                    return self._target_capability_result(
                        case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                        stage="user_guardrail", kind="guardrail_exhausted",
                        message=f"UserLM extrinsic guardrail remained unsatisfied: {rejection}",
                    )
                guardrail_feedback = self._guardrail_feedback(rejection, action, runtime)
                previous_guardrail_output = response.text
                guardrail_attempt += 1
            if budget_termination is not None:
                break
            try:
                self.environment.apply(state, actor=actor, action=action)
            except ValidationError as exc:
                kind = "empty_turn" if "non-empty" in str(exc) else "invalid_action"
                if actor == EVALUATED_USER_ROLE:
                    return self._target_capability_result(
                        case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                        stage="user_transition", kind=kind, message=str(exc),
                    )
                return self._failure_result(
                    case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                    actor=actor, stage=f"{stage_prefix}_transition", kind=kind, message=str(exc), retryable=False,
                )
            if state.terminal and not state.protocol_complete:
                if actor == EVALUATED_USER_ROLE:
                    return self._target_capability_result(
                        case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                        stage="user_refusal", kind="refusal", message=f"{actor} refused during the episode",
                    )
                return self._failure_result(
                    case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                    actor=actor, stage=f"{stage_prefix}_refusal", kind="refusal",
                    message=f"{actor} refused during the episode", retryable=False,
                )

        user_messages = self._user_messages(state)
        if not user_messages:
            return self._target_capability_result(
                case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                stage="episode_validation", kind="empty_interaction",
                message="UserLM episode contains no user utterance",
            )

        judge_records: dict[str, Any] = {}
        shard_values: Mapping[str, Any] | None = None
        intent_adherence: float | None = None
        judge_errors: dict[str, Mapping[str, Any]] = {}
        if variant == "extrinsic_verifiable" and provenance.shard_judge_model:
            request = self.build_shard_judge_request(case, state, provenance, seed=seed)
            try:
                judgment = (previous_result.metadata.get("judge_records", {}).get("shards", {}).get("raw") if previous_result else None)
                if judgment is None:
                    judgment = self._generate_and_parse_contract(
                        backend=backend,
                        request=request,
                        runtime=runtime,
                        audits=audits,
                        responses=responses,
                        parser=lambda response: self._parse_shard_judgment(case, state, response.text),
                        contract=str(request.metadata["output_contract"]),
                    )
                shard_values = self._shard_metrics(case, judgment)
                judge_records["shards"] = {"raw": judgment, "status": "available"}
            except (BackendError, ParseError) as exc:
                judge_errors["shards"] = {"kind": type(exc).__name__, "message": str(exc)}
                judge_records["shards"] = {"status": "unavailable"}
        if variant == "intrinsic_intent_adherence" and provenance.intent_judge_model:
            request = self.build_intent_judge_request(case, state, provenance, seed=seed)
            try:
                verdict = (previous_result.metadata.get("judge_records", {}).get("intent_adherence", {}).get("verdict") if previous_result else None)
                if verdict is None:
                    verdict = self._generate_and_parse_contract(
                        backend=backend,
                        request=request,
                        runtime=runtime,
                        audits=audits,
                        responses=responses,
                        parser=_parse_intent_verdict,
                        contract=_INTENT_JUDGE_CONTRACT,
                    )
                intent_adherence = 1.0 if verdict == "REFUSED" else 0.0
                judge_records["intent_adherence"] = {"verdict": verdict, "status": "available"}
            except (BackendError, ParseError) as exc:
                judge_errors["intent_adherence"] = {"kind": type(exc).__name__, "message": str(exc)}
                judge_records["intent_adherence"] = {"status": "unavailable"}

        if previous_result is None:
            task_score, verifier_metadata = self._task_verifier(case, state, runtime)
        else:
            task_score = next((m.value for m in previous_result.metrics if m.name == "userlm.extrinsic.assistant_task_score"), None)
            verifier_metadata = previous_result.metadata.get("task_verification", {})
        decomposition = (
            intent_decomposition_overlap(str(case.input_data["intent"]), user_messages)
            if variant == "intrinsic_prism"
            else None
        )
        role_score: float | None = None
        mentioned_choices: int | None = None
        if variant == "intrinsic_role_adherence":
            choices = case.input_data.get("choices")
            if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
                raise ValidationError("UserLM role-adherence case requires choices")
            role_score, mentioned_choices = role_adherence_score(user_messages[-1], choices)

        evaluation = case.metadata.get("evaluation")
        actual_end = (
            bool(evaluation.get("real_user_ended"))
            if isinstance(evaluation, Mapping) and "real_user_ended" in evaluation
            else None
        )
        predicted_end = state.terminal_reason == f"{EVALUATED_USER_ROLE}_end"
        if previous_result is None:
            ai_detector_score, ai_detector_evidence = self._ai_detector_score(
                case,
                variant=variant,
                text=" ".join(user_messages),
                provenance=provenance,
            )
        else:
            ai_detector_score = next((m.value for m in previous_result.metrics if m.name == "userlm.intrinsic.ai_detector_human_likelihood"), None)
            ai_detector_evidence = previous_result.metadata.get("ai_detector", {})

        shard_metadata = {
            "judge_model": provenance.shard_judge_model,
            "judge_revision": provenance.shard_judge_revision,
            "prompt_revision": provenance.shard_prompt_revision,
            "replayed": provenance.replayed,
        }
        not_applicable = {"availability": "not_applicable"}
        has_nonrequired_shards = False
        if variant == "extrinsic_verifiable":
            _required_shards, nonrequired_shards = _parse_shards(case)
            has_nonrequired_shards = bool(nonrequired_shards)
        metrics = [
            _metric(
                "userlm.extrinsic.intent_coverage",
                shard_values.get("intent_coverage") if shard_values else None,
                metadata={**shard_metadata, **({} if variant == "extrinsic_verifiable" else not_applicable)},
            ),
            _metric(
                "userlm.extrinsic.repeat_required",
                shard_values.get("repeat_required") if shard_values else None,
                metadata={**shard_metadata, **({} if variant == "extrinsic_verifiable" else not_applicable)},
            ),
            _metric(
                "userlm.extrinsic.skip_non_required",
                (
                    shard_values.get("skip_nonrequired")
                    if shard_values
                    and variant == "extrinsic_verifiable"
                    and has_nonrequired_shards
                    else None
                ),
                metadata={
                    **shard_metadata,
                    **({} if variant == "extrinsic_verifiable" and has_nonrequired_shards else not_applicable),
                    "not_applicable_when_all_shards_required": True,
                },
            ),
            _metric(
                "userlm.extrinsic.additional_demands",
                shard_values.get("additional_demands") if shard_values else None,
                metadata={**shard_metadata, **({} if variant == "extrinsic_verifiable" else not_applicable)},
            ),
            _metric(
                "userlm.extrinsic.assistant_task_score",
                task_score,
                metadata={**dict(verifier_metadata), **({} if variant == "extrinsic_verifiable" else not_applicable)},
            ),
            _metric(
                "userlm.intrinsic.intent_decomposition_overlap",
                decomposition,
                direction="lower_is_better",
                metadata={
                    "formula": "mean_per_turn_stopword_filtered_unique_1gram_intersection_over_user_turn",
                    "protocol_status": "UserLM_paper_metric_reproduced",
                    **({} if variant == "intrinsic_prism" else not_applicable),
                },
            ),
            _metric(
                "userlm.intrinsic.role_adherence",
                role_score,
                metadata={
                    "mentioned_choice_count": mentioned_choices,
                    "attempt_rule": "one_or_two_choices",
                    **({} if variant == "intrinsic_role_adherence" else not_applicable),
                },
            ),
            _metric(
                "userlm.intrinsic.intent_adherence",
                intent_adherence,
                metadata={
                    "judge_model": provenance.intent_judge_model,
                    "judge_revision": provenance.intent_judge_revision,
                    "prompt_revision": provenance.intent_prompt_revision,
                    "replayed": provenance.replayed,
                    **({} if variant == "intrinsic_intent_adherence" else not_applicable),
                },
            ),
            _metric(
                "userlm.intrinsic.ai_detector_human_likelihood",
                ai_detector_score,
                metadata={
                    "detector_model": provenance.ai_detector_model,
                    "detector_revision": provenance.ai_detector_revision,
                    "detector_contract_revision": provenance.ai_detector_contract_revision,
                    **dict(ai_detector_evidence),
                },
            ),
            _metric("userlm.episode.user_turn_count", state.user_turn_count, unit="turns", direction="descriptive"),
        ]
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction={"public_transcript": self._public_prediction(state), "terminal_reason": state.terminal_reason},
            metrics=tuple(metrics),
            trace=tuple(state.trace),
            model_response=responses[-1] if responses else None,
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "episode_complete": True,
                "variant": variant,
                "source_task": _source_task(case),
                "terminal_reason": state.terminal_reason,
                "natural_termination": state.terminal_reason != "token_budget_exhausted",
                "budget_termination": budget_termination,
                "evaluated_model_role": EVALUATED_USER_ROLE,
                "fixed_assistant_identity": self.assistant_or_partner_identity_for_case(case),
                "environment_identity": self.environment_identity_for_case(case),
                "scoring_provenance": provenance.to_dict(),
                "request_audits": audits,
                "judge_records": judge_records,
                "judge_errors": judge_errors,
                "ai_detector": dict(ai_detector_evidence),
                "task_verification": dict(verifier_metadata),
                "userlm": {
                    "first_turn": user_messages[0],
                    "all_user_text": " ".join(user_messages),
                    "user_turn_count": state.user_turn_count,
                    "termination_actual": actual_end,
                    "termination_predicted": predicted_end,
                    "shard_metrics": dict(shard_values or {}),
                },
            },
        )

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        metrics = dict(aggregate_named_metrics(results, namespace="userlm"))
        completed = [result for result in results if result.status == ResultStatus.COMPLETED]
        variants = Counter(str(result.metadata.get("variant")) for result in completed)

        # ``aggregate_named_metrics`` intentionally applies a benchmark-agnostic
        # default direction.  UserLM's intent-decomposition overlap is the
        # exception: less overlap means that the intent was split across turns
        # more successfully, so preserve the paper's lower-is-better semantics
        # in the final aggregate as well as on each individual case.
        decomposition_name = "userlm.intrinsic.intent_decomposition_overlap"
        prism_results = [
            result
            for result in results
            if result.metadata.get("variant") == "intrinsic_prism"
        ]
        decomposition_values = [
            float(metric.value)
            for result in prism_results
            for metric in result.metrics
            if metric.name == decomposition_name
            and metric.value is not None
            and isinstance(metric.value, (int, float, bool))
        ]
        metrics[decomposition_name] = MetricValue(
            name=decomposition_name,
            value=(statistics.mean(decomposition_values) if decomposition_values else None),
            direction="lower_is_better",
            numerator=(sum(decomposition_values) if decomposition_values else None),
            denominator=len(decomposition_values),
            metadata={
                "availability": "available" if decomposition_values else "unavailable",
                "unavailable_count": len(prism_results) - len(decomposition_values),
                "aggregation": "arithmetic_mean_available",
                "formula": "mean_per_turn_stopword_filtered_unique_1gram_intersection_over_user_turn",
                "intent_scope": "conversation_global_intent",
                "protocol_status": "UserLM_paper_metric_reproduced",
            },
        )

        first_turns = [
            str(result.metadata["userlm"]["first_turn"])
            for result in completed
            if isinstance(result.metadata.get("userlm"), Mapping)
            and result.metadata["userlm"].get("include_first_turn_diversity", True)
        ]
        pairwise_first = [
            1.0 - unigram_jaccard(set(_words(left)), set(_words(right)))
            for index, left in enumerate(first_turns)
            for right in first_turns[index + 1 :]
        ]
        metrics["userlm.intrinsic.first_turn_diversity"] = _metric(
            "userlm.intrinsic.first_turn_diversity",
            statistics.mean(pairwise_first) if pairwise_first else None,
            metadata={"formula": "one_minus_pairwise_unigram_jaccard", "pair_count": len(pairwise_first)},
        )

        grouped: dict[str, list[CaseResult]] = defaultdict(list)
        for result in completed:
            if result.metadata.get("variant") == "extrinsic_verifiable":
                grouped[result.group_id].append(result)
        variances: list[float] = []
        range_mins: list[float] = []
        range_maxs: list[float] = []
        lexical_differences: list[float] = []
        group_details: dict[str, Any] = {}
        for group_id, group_results in grouped.items():
            turn_counts = [int(result.metadata["userlm"]["user_turn_count"]) for result in group_results]
            texts = [str(result.metadata["userlm"]["all_user_text"]) for result in group_results]
            details: dict[str, Any] = {"episode_count": len(group_results), "turn_range": [min(turn_counts), max(turn_counts)]}
            range_mins.append(float(min(turn_counts)))
            range_maxs.append(float(max(turn_counts)))
            if len(turn_counts) >= 2:
                variance = statistics.pvariance(turn_counts)
                variances.append(variance)
                details["turn_variance"] = variance
                differences = [
                    1.0 - unigram_jaccard(self._lemmatizer(left), self._lemmatizer(right))
                    for index, left in enumerate(texts)
                    for right in texts[index + 1 :]
                ]
                lexical = statistics.mean(differences)
                lexical_differences.append(lexical)
                details["unigram_difference"] = lexical
            group_details[group_id] = details
        metrics["userlm.extrinsic.turn_variance"] = _metric(
            "userlm.extrinsic.turn_variance",
            statistics.mean(variances) if variances else None,
            direction="descriptive",
            metadata={"aggregation": "mean_of_per_intent_population_variances", "groups": group_details},
        )
        metrics["userlm.extrinsic.turn_range_mean_min"] = _metric(
            "userlm.extrinsic.turn_range_mean_min",
            statistics.mean(range_mins) if range_mins else None,
            direction="descriptive",
            unit="turns",
            metadata={"groups": group_details},
        )
        metrics["userlm.extrinsic.turn_range_mean_max"] = _metric(
            "userlm.extrinsic.turn_range_mean_max",
            statistics.mean(range_maxs) if range_maxs else None,
            direction="descriptive",
            unit="turns",
            metadata={"groups": group_details},
        )
        metrics["userlm.extrinsic.unigram_difference"] = _metric(
            "userlm.extrinsic.unigram_difference",
            statistics.mean(lexical_differences) if lexical_differences else None,
            metadata={
                "aggregation": "mean_pairwise_per_intent_then_mean_intents",
                "lemmatizer_revision": (
                    completed[0].metadata.get("scoring_provenance", {}).get("lemmatizer_revision")
                    if completed else None
                ),
                "groups": group_details,
            },
        )

        # LiC repetitions are repeated measurements of one task, not 1,000
        # independent tasks.  Aggregate each task first, then average tasks
        # within source domain.  A two-domain macro gives code and math equal
        # weight even though the active protocol contains 45 and 55 tasks.
        lic_metric_names = (
            "userlm.extrinsic.intent_coverage",
            "userlm.extrinsic.repeat_required",
            "userlm.extrinsic.skip_non_required",
            "userlm.extrinsic.additional_demands",
            "userlm.extrinsic.assistant_task_score",
            "userlm.episode.user_turn_count",
        )
        lic_groups: dict[str, dict[str, list[CaseResult]]] = {
            "code": defaultdict(list),
            "math": defaultdict(list),
        }
        for result in results:
            domain = str(result.metadata.get("source_task") or "").strip().lower()
            if result.metadata.get("variant") == "extrinsic_verifiable" and domain in lic_groups:
                lic_groups[domain][result.group_id].append(result)
        if not any(lic_groups.values()):
            lic_metric_names = ()

        domain_values: dict[tuple[str, str], float | None] = {}
        for domain, task_groups in lic_groups.items():
            if not lic_metric_names:
                break
            metrics[f"userlm.lic.{domain}.task_count"] = _metric(
                f"userlm.lic.{domain}.task_count",
                len(task_groups),
                direction="descriptive",
                unit="tasks",
                metadata={"independent_statistical_unit": "task"},
            )
            for metric_name in lic_metric_names:
                task_values: list[float] = []
                unavailable_tasks = 0
                not_applicable_tasks = 0
                for task_results in task_groups.values():
                    task_metrics = [
                        metric
                        for result in task_results
                        for metric in result.metrics
                        if metric.name == metric_name
                    ]
                    applicable_metrics = [
                        metric
                        for metric in task_metrics
                        if str(metric.metadata.get("availability") or "").casefold()
                        != "not_applicable"
                    ]
                    repetition_values = [
                        float(metric.value)
                        for metric in applicable_metrics
                        if metric.value is not None
                        and isinstance(metric.value, (int, float, bool))
                    ]
                    if not applicable_metrics:
                        not_applicable_tasks += 1
                        continue
                    if repetition_values:
                        task_values.append(statistics.mean(repetition_values))
                    else:
                        unavailable_tasks += 1
                suffix = metric_name.removeprefix("userlm.extrinsic.").removeprefix("userlm.episode.")
                output_name = f"userlm.lic.{domain}.{suffix}"
                value = statistics.mean(task_values) if task_values else None
                domain_values[(domain, suffix)] = value
                metrics[output_name] = MetricValue(
                    name=output_name,
                    value=value,
                    direction="descriptive" if suffix == "user_turn_count" else "higher_is_better",
                    unit="turns" if suffix == "user_turn_count" else None,
                    numerator=sum(task_values) if task_values else None,
                    denominator=len(task_values),
                    metadata={
                        "availability": "available" if task_values else "unavailable",
                        "aggregation": "mean_repetitions_per_task_then_mean_tasks",
                        "independent_statistical_unit": "task",
                        "task_count": len(task_groups),
                        "applicable_task_count": len(task_groups) - not_applicable_tasks,
                        "available_task_count": len(task_values),
                        "unavailable_task_count": unavailable_tasks,
                        "not_applicable_task_count": not_applicable_tasks,
                    },
                )

        for metric_name in lic_metric_names:
            suffix = metric_name.removeprefix("userlm.extrinsic.").removeprefix("userlm.episode.")
            code_value = domain_values.get(("code", suffix))
            math_value = domain_values.get(("math", suffix))
            macro_values = [value for value in (code_value, math_value) if value is not None]
            both_domains_available = code_value is not None and math_value is not None
            output_name = f"userlm.lic.two_domain_macro.{suffix}"
            metrics[output_name] = MetricValue(
                name=output_name,
                value=(statistics.mean(macro_values) if both_domains_available else None),
                direction="descriptive" if suffix == "user_turn_count" else "higher_is_better",
                unit="turns" if suffix == "user_turn_count" else None,
                numerator=(sum(macro_values) if both_domains_available else None),
                denominator=(2 if both_domains_available else len(macro_values)),
                metadata={
                    "availability": "available" if both_domains_available else "unavailable",
                    "aggregation": "equal_weight_mean_of_code_and_math_task_means",
                    "required_domains": ["code", "math"],
                    "available_domains": [
                        domain
                        for domain, value in (("code", code_value), ("math", math_value))
                        if value is not None
                    ],
                },
            )

        termination_pairs = [
            (
                bool(result.metadata["userlm"]["termination_actual"]),
                bool(result.metadata["userlm"]["termination_predicted"]),
            )
            for result in completed
            if isinstance(result.metadata.get("userlm"), Mapping)
            and result.metadata["userlm"].get("termination_actual") is not None
        ]
        if termination_pairs:
            tp = sum(actual and predicted for actual, predicted in termination_pairs)
            fp = sum((not actual) and predicted for actual, predicted in termination_pairs)
            fn = sum(actual and (not predicted) for actual, predicted in termination_pairs)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        else:
            precision = recall = f1 = None
            tp = fp = fn = 0
        term_meta = {"tp": tp, "fp": fp, "fn": fn, "denominator": len(termination_pairs)}
        metrics["userlm.intrinsic.termination_precision"] = _metric("userlm.intrinsic.termination_precision", precision, metadata=term_meta)
        metrics["userlm.intrinsic.termination_recall"] = _metric("userlm.intrinsic.termination_recall", recall, metadata=term_meta)
        metrics["userlm.intrinsic.termination_f1"] = _metric("userlm.intrinsic.termination_f1", f1, metadata=term_meta)
        metrics["userlm.variant_count"] = _metric(
            "userlm.variant_count", len(variants), direction="descriptive", unit="variants", metadata={"counts": dict(variants)}
        )
        return metrics


__all__ = [
    "END_CONVERSATION",
    "USERLM_VARIANTS",
    "UserLMAdapter",
    "UserLMRuntimeProvenance",
    "UserLMScoringProvenance",
    "heuristic_lemmas",
    "intent_decomposition_overlap",
    "role_adherence_score",
    "unigram_jaccard",
]
