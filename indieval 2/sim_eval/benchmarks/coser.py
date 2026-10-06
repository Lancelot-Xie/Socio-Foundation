"""CoSER GCA adapter with upstream profile visibility and audited critic scoring."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
from ..environments.coser import (
    END_SCENE,
    ENVIRONMENT_ROLE,
    CoserEnvironment,
    CoserState,
    compose_coser_profile,
    deterministic_token_count,
    deterministic_tokens,
    parse_coser_role_output,
    parse_next_speaker,
)
from ..environments.social import JudgeProvenance
from ..errors import (
    BackendError,
    ConfigurationError,
    ContextLimitError,
    EpisodeTokenBudgetExhausted,
    ParseError,
    ValidationError,
)
from ..interfaces import BenchmarkAdapter, ModelBackend
from ..json_utils import canonical_json
from ..registry import adapter
from .common import (
    aggregate_named_metrics,
    combine_usage,
    generate_and_parse_with_contract_retries,
    json_schema_response_format,
    parse_json_object,
)


COSER_DIMENSIONS: Mapping[str, str] = {
    "storyline_consistency": "Storyline Consistency",
    "anthropomorphism": "Anthropomorphism",
    "character_fidelity": "Character Fidelity",
    "storyline_quality": "Storyline Quality",
}
OFFICIAL_COSER_CRITIC_PROMPT_REVISION = "coser-self-play-deduct-template-upstream-exact-v1"
OFFICIAL_COSER_CRITIC_TEMPLATE = """You are a literary critic specializing in character analysis and dialogue evaluation. Given a simulated conversation for a plot in {book}, your task is to evaluate this conversation via the following steps:

1. Read and understand the provided materials about {book}:
   * Story context and scenario.
   * Profiles of the main characters, including {major_characters}.
   * The original conversation from {book} in the same scenario as a reference.

  2. Evaluate the simulated conversation in terms of {dimension_name}, i.e., {dimension_brief}. 
   Note that, each character message is composed of speech, action (wrapped within (...) ), and inner thoughts (wrapped within [...] ). The inner thoughts are not spoken aloud and are thus invisible to other characters. 
   The detailed evaluation criteria will be provided below.
   {additional_instructions}

## Scenario

### Plot Summary

{plot_summary}

### Current Scenario

{scenario}

## Character Profiles

{character_profiles}

## Original Conversation

{original_conversation}

## Evaluation Criteria

To evaluate the simulated conversation, identify the following types of flaws:

{dimension_criteria}

## Scoring Guidelines

1. Identify all instances of flaws occurred in the simulated conversation.
      
2. For each flaw identified, determine its level of severity into 1 to 5, where 1 indicates minor, 3 indicates moderate, and 5 indicates severe.
   
## Output Requirements

Provide your evaluation in JSON format:

Example Output:
{
    "{dimension_name}": {
        "flaws": [ 
          {
            "instance": <comment on the flaw instance>, 
            "type": <flaw type>, 
            "severity": <range from 1 (minor) to 5 (severe)>
          },
    },
}
===Dialogue Content===
"""
OFFICIAL_COSER_DIMENSION_DETAILS: Mapping[str, Mapping[str, str]] = {
    "storyline_consistency": {
        "brief": "Whether the storyline and characters' reactions in the simulated conversation align well with those in the reference conversation",
        "criteria": """### Storyline Consistency
   - Type: Storyline Consistency
     * Characters' reactions (emotions, attitudes, behaviors) in the simulated conversation deviate from those in the original conversation""",
    },
    "anthropomorphism": {
        "brief": "How human-like and natural the characters behave",
        "criteria": """### Anthropomorphism
   - Type: Self-identity
     * Lacks initiative and goals
     * Does not make independent decisions
     * Lacks clear preferences and dislikes
     * Behaves like a 'helpful AI assistant' by being overly verbose, helpful, didactic, moralistic, submissive or easily persuaded if it is not the character's personality

   - Type: Emotional Depth
     * Lacks psychological complexity and exhibits rigid, superficial reactions
     * Directly speaks out all thoughts and feelings, instead of using subtext

   - Type: Persona Coherence
     * Shows inconsistent or rapidly changing personality traits and emotional patterns

   - Type: Social Interaction
     * Shows a lack of understanding of others' thoughts and feelings
     * Reacts rigidly to others without considering the context.
     * Demonstrate a lack of appropriate social skills.""",
    },
    "character_fidelity": {
        "brief": "How well the characters match their established profiles from the book",
        "criteria": """### Character Fidelity
   (Only apply to the main characters: {major_characters})
   - Type: Character Language
     * Uses vocabulary, expressions, and tone that are not appropriate for the characters' traits or  social/educational background

   - Type: Knowledge & Background
     * Fails to demonstrate character-specific knowledge, background or experiences
     * Includes future information beyond the character's current stage

   - Type: Personality & Behavior
     * Shows emotions, thoughts, behaviors, values, beliefs, and decisions that conflict with their personality and background
     * Shows interest in topics that are uninteresting and unrelated to the character
     * Character's thoughts, emotions, and behaviors demonstrate contrasting personality traits compared to the reference conversation
     * Exhibits contrasting reactions compared to those in the reference conversation if situated in similar contexts. (Such flaws should be counted both in the "Storyline Consistency" dimension and the "Character Fidelity" dimension.) 

   - Type: Relationship & Social Status
     * Interacts inappropriately with other characters regarding their background, relationship and social status""",
    },
    "storyline_quality": {
        "brief": "How well the conversation maintains logical consistency and narrative quality",
        "criteria": """### Storyline Quality
   - Type: Flow & Progression
     * Shows unnatural progression or lacks meaningful developments
     * Dialogue is verbose and redundant
     * Repeats others' viewpoints or previously mentioned information
     * Mechanically repeats one's own words or phrases. More repetitions lead to higher severity (up to 10). 

   - Type: Logical Consistency
     * Contains factual contradictions between statements or perspectives""",
    },
}
LENGTH_CORRECTION_PER_ACTOR_TURN = 1.5
COSER_CHARACTER_MAX_TOKENS = 2048
OFFICIAL_HIDDEN_THOUGHT_POLICY = "role_private_removed_from_other_roles_and_critic"
_ROLE_OUTPUT_CONTRACT = (
    "Include thought, speech, and action in natural role-play text. Put private thoughts in "
    "[square brackets] and publicly visible actions in (parentheses), following the official CoSER protocol."
)
_ENVIRONMENT_OUTPUT_CONTRACT = (
    "Describe only the resulting environment changes in 1-3 concise sentences; do not emit private thoughts "
    "or dictate a main character's dialogue or actions."
)
_NEXT_SPEAKER_CONTRACT = (
    "Use the official two-line protocol: `1. Reasoning: ...` followed by "
    "`2. Next Speaker: <one allowed character, Environment, random, or <END CHAT>>`."
)


@dataclass(frozen=True)
class CoserRuntimeProvenance:
    environment_model: str
    environment_revision: str
    next_speaker_model: str
    next_speaker_revision: str
    retrieval_mode: str = "none"
    retrieval_k: int = 0
    hidden_thought_policy: str = OFFICIAL_HIDDEN_THOUGHT_POLICY
    max_judge_context_tokens: int = 131072
    source: str = "configured"

    def __post_init__(self) -> None:
        required = (
            self.environment_model,
            self.environment_revision,
            self.next_speaker_model,
            self.next_speaker_revision,
            self.hidden_thought_policy,
        )
        if not all(required):
            raise ConfigurationError("CoSER runtime provenance fields cannot be empty")
        if self.retrieval_k < 0 or self.max_judge_context_tokens <= 0:
            raise ConfigurationError("CoSER retrieval_k must be nonnegative and judge context budget positive")
        if self.hidden_thought_policy != OFFICIAL_HIDDEN_THOUGHT_POLICY:
            raise ConfigurationError(
                "CoSER official GCA requires private thoughts to be removed from other roles and critics"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_model": self.environment_model,
            "environment_revision": self.environment_revision,
            "next_speaker_model": self.next_speaker_model,
            "next_speaker_revision": self.next_speaker_revision,
            "retrieval_mode": self.retrieval_mode,
            "retrieval_k": self.retrieval_k,
            "hidden_thought_policy": self.hidden_thought_policy,
            "max_judge_context_tokens": self.max_judge_context_tokens,
            "source": self.source,
        }


def _safe_metric_component(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.-]+", "_", value.casefold()).strip("_")
    return normalized or "character"


def _flaws(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ParseError(f"CoSER critic {label} must be a flaws array")
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ParseError(f"CoSER critic {label} flaw #{index} must be an object")
        instance = item.get("instance")
        flaw_type = item.get("type")
        severity = item.get("severity")
        if not isinstance(instance, str) or not instance.strip():
            raise ParseError(f"CoSER critic {label} flaw #{index} requires nonempty instance")
        if not isinstance(flaw_type, str) or not flaw_type.strip():
            raise ParseError(f"CoSER critic {label} flaw #{index} requires nonempty type")
        if isinstance(severity, bool) or not isinstance(severity, int) or not 1 <= severity <= 5:
            raise ParseError(f"CoSER critic {label} flaw #{index} severity must be an integer from 1 to 5")
        parsed.append({"instance": instance.strip(), "type": flaw_type.strip(), "severity": severity})
    return parsed


def coser_length_corrected_score(flaws: Sequence[Mapping[str, Any]], actor_rounds: int) -> float:
    if actor_rounds < 0:
        raise ValidationError("CoSER actor_rounds cannot be negative")
    severity_sum = sum(int(item["severity"]) for item in flaws)
    return max(0.0, min(100.0, 100.0 - severity_sum * 5.0 + actor_rounds * LENGTH_CORRECTION_PER_ACTOR_TURN))


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1)))


def dependency_free_bleu(reference: str, hypothesis: str) -> float:
    """BLEU-4 compatibility metric using the framework's audited tokenizer."""

    ref = tuple(token.casefold() for token in deterministic_tokens(reference))
    hyp = tuple(token.casefold() for token in deterministic_tokens(hypothesis))
    if not ref or not hyp:
        return 0.0
    precisions: list[float] = []
    for n in range(1, 5):
        hyp_counts = _ngrams(hyp, n)
        denominator = sum(hyp_counts.values())
        if denominator == 0:
            return 0.0
        ref_counts = _ngrams(ref, n)
        numerator = sum(min(count, ref_counts.get(gram, 0)) for gram, count in hyp_counts.items())
        if numerator == 0:
            return 0.0
        precisions.append(numerator / denominator)
    brevity_penalty = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / len(hyp))
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / 4.0)


def dependency_free_rouge_l(reference: str, hypothesis: str) -> float:
    ref = tuple(token.casefold() for token in deterministic_tokens(reference))
    hyp = tuple(token.casefold() for token in deterministic_tokens(hypothesis))
    if not ref or not hyp:
        return 0.0
    previous = [0] * (len(hyp) + 1)
    for ref_token in ref:
        current = [0]
        for index, hyp_token in enumerate(hyp, start=1):
            if ref_token == hyp_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class CoserScorer:
    scorer_revision = "coser-gca-official-envelope-scene-score-no-character-attribution-v2"

    def unavailable_dimension(
        self,
        dimension: str,
        *,
        reason: str,
        state: CoserState,
        provenance: JudgeProvenance | None,
    ) -> tuple[MetricValue, ...]:
        provenance_data = provenance.to_dict() if provenance else None
        metrics = [
            MetricValue(
                f"coser.scene.{dimension}",
                None,
                unit="score_0_to_100",
                metadata={"availability": "unavailable", "reason": reason, "judge_provenance": provenance_data},
            )
        ]
        for character_id, character in state.characters.items():
            metrics.append(
                MetricValue(
                    f"coser.character.{_safe_metric_component(character_id)}.{dimension}",
                    None,
                    unit="score_0_to_100",
                    metadata={
                        "availability": "unavailable",
                        "reason": reason,
                        "character_id": character_id,
                        "character_name": character.name,
                        "judge_provenance": provenance_data,
                        "protocol_status": "indieval_character_diagnostic_not_official_scene_score",
                    },
                )
            )
        return tuple(metrics)

    def score_dimension(
        self,
        payload: Mapping[str, Any],
        *,
        dimension: str,
        state: CoserState,
        provenance: JudgeProvenance,
    ) -> tuple[MetricValue, ...]:
        if dimension not in COSER_DIMENSIONS:
            raise ValidationError(f"unknown CoSER critic dimension {dimension!r}")
        expected = {"dimension", "scene_flaws", "character_flaws"}
        if set(payload) != expected:
            raise ParseError(
                f"CoSER critic response fields differ; missing={sorted(expected-set(payload))}, "
                f"extra={sorted(set(payload)-expected)}"
            )
        if payload["dimension"] not in {dimension, COSER_DIMENSIONS[dimension]}:
            raise ParseError(f"CoSER critic returned dimension {payload['dimension']!r}, expected {dimension!r}")
        scene_flaws = _flaws(payload["scene_flaws"], label=f"scene/{dimension}")
        character_raw = payload["character_flaws"]
        if not isinstance(character_raw, Mapping) or set(character_raw) != set(state.characters):
            got = sorted(character_raw) if isinstance(character_raw, Mapping) else type(character_raw).__name__
            raise ParseError(
                f"CoSER critic character IDs differ for {dimension}; expected={sorted(state.characters)}, got={got}"
            )
        public_events = state.public_transcript
        actor_rounds = sum(event.actor != ENVIRONMENT_ROLE for event in public_events)
        provenance_data = provenance.to_dict()
        formula = "clamp(100 - 5*sum(severity) + 1.5*actor_rounds, 0, 100)"
        metrics: list[MetricValue] = [
            MetricValue(
                f"coser.scene.{dimension}",
                coser_length_corrected_score(scene_flaws, actor_rounds),
                unit="score_0_to_100",
                metadata={
                    "official_dimension_name": COSER_DIMENSIONS[dimension],
                    "flaws": scene_flaws,
                    "flaw_visibility": "evaluator_only",
                    "actor_rounds": actor_rounds,
                    "length_correction_per_actor_turn": LENGTH_CORRECTION_PER_ACTOR_TURN,
                    "formula": formula,
                    "judge_provenance": provenance_data,
                    "scope": "scene_official_gca_compatible",
                },
            )
        ]
        for character_id, character in state.characters.items():
            flaws = _flaws(character_raw[character_id], label=f"character/{character_id}/{dimension}")
            character_rounds = sum(event.actor == character_id for event in public_events)
            metrics.append(
                MetricValue(
                    f"coser.character.{_safe_metric_component(character_id)}.{dimension}",
                    coser_length_corrected_score(flaws, character_rounds),
                    unit="score_0_to_100",
                    metadata={
                        "character_id": character_id,
                        "character_name": character.name,
                        "flaws": flaws,
                        "flaw_visibility": "evaluator_only",
                        "actor_rounds": character_rounds,
                        "length_correction_per_actor_turn": LENGTH_CORRECTION_PER_ACTOR_TURN,
                        "formula": formula,
                        "judge_provenance": provenance_data,
                        "scope": "character",
                        "protocol_status": "indieval_character_diagnostic_not_official_scene_score",
                    },
                )
            )
        return tuple(metrics)

    def score_official_dimension(
        self,
        payload: Mapping[str, Any],
        *,
        dimension: str,
        state: CoserState,
        provenance: JudgeProvenance,
    ) -> tuple[MetricValue, ...]:
        """Score the exact released CoSER critic envelope without inventing character attribution."""

        official_name = COSER_DIMENSIONS[dimension]
        if set(payload) != {official_name}:
            raise ParseError(f"CoSER official critic requires exactly the key {official_name!r}")
        body = payload[official_name]
        if not isinstance(body, Mapping) or set(body) != {"flaws"}:
            raise ParseError(f"CoSER official critic {official_name} requires exactly the flaws field")
        flaws = _flaws(body["flaws"], label=f"scene/{dimension}")
        actor_rounds = sum(event.actor != ENVIRONMENT_ROLE for event in state.public_transcript)
        provenance_data = provenance.to_dict()
        formula = "clamp(100 - 5*sum(severity) + 1.5*actor_rounds, 0, 100)"
        metrics: list[MetricValue] = [
            MetricValue(
                f"coser.scene.{dimension}",
                coser_length_corrected_score(flaws, actor_rounds),
                unit="score_0_to_100",
                metadata={
                    "official_dimension_name": official_name,
                    "flaws": flaws,
                    "flaw_visibility": "evaluator_only",
                    "actor_rounds": actor_rounds,
                    "length_correction_per_actor_turn": LENGTH_CORRECTION_PER_ACTOR_TURN,
                    "formula": formula,
                    "judge_provenance": provenance_data,
                    "scope": "scene_official_gca",
                    "output_protocol": "upstream_dimension_flaws_envelope",
                },
            )
        ]
        for character_id, character in state.characters.items():
            metrics.append(
                MetricValue(
                    f"coser.character.{_safe_metric_component(character_id)}.{dimension}",
                    None,
                    unit="score_0_to_100",
                    metadata={
                        "availability": "unavailable",
                        "reason": "official CoSER scene critic does not attribute flaws to individual characters",
                        "character_id": character_id,
                        "character_name": character.name,
                        "judge_provenance": provenance_data,
                        "protocol_status": "indieval_character_diagnostic_not_produced_by_official_output",
                    },
                )
            )
        return tuple(metrics)

    @staticmethod
    def add_averages(metrics: Sequence[MetricValue], state: CoserState) -> tuple[MetricValue, ...]:
        result = list(metrics)

        def average_for(prefix: str, metadata: Mapping[str, Any]) -> MetricValue:
            values = []
            missing = []
            for dimension in COSER_DIMENSIONS:
                name = f"{prefix}.{dimension}"
                match = next((metric for metric in metrics if metric.name == name), None)
                if match is None or not isinstance(match.value, (int, float)):
                    missing.append(dimension)
                else:
                    values.append(float(match.value))
            return MetricValue(
                f"{prefix}.critic_average",
                sum(values) / len(values) if len(values) == len(COSER_DIMENSIONS) else None,
                unit="score_0_to_100",
                numerator=sum(values) if len(values) == len(COSER_DIMENSIONS) else None,
                denominator=len(values),
                metadata={
                    **dict(metadata),
                    "aggregation": "arithmetic_mean_four_dimensions",
                    "missing_dimensions": missing,
                    "availability": "available" if not missing else "unavailable_incomplete_judge_dimensions",
                },
            )

        result.append(average_for("coser.scene", {"scope": "scene_official_gca_average"}))
        for character_id, character in state.characters.items():
            result.append(
                average_for(
                    f"coser.character.{_safe_metric_component(character_id)}",
                    {
                        "scope": "character",
                        "character_id": character_id,
                        "character_name": character.name,
                        "protocol_status": "indieval_character_diagnostic_not_official_scene_score",
                    },
                )
            )
        return tuple(result)


def _public_dialogue(state: CoserState) -> list[dict[str, Any]]:
    return [
        {"turn": event.turn, "speaker": event.actor, "speech": event.content["speech"], "action": event.content["action"]}
        for event in state.public_transcript
    ]


def _remove_inner_thoughts(text: str) -> str:
    """Mirror CoSER ``gca_evaluation.utils.remove_inner_thoughts``."""

    cleaned = re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    cleaned = re.sub(r"\n+", "\n", cleaned)
    for tag in ("system", "role", ""):
        prefix = rf"{tag}[_\s]+" if tag else r"[_\s]+"
        cleaned = re.sub(
            rf"\s*<\s*{prefix}think(?:ing)?\s*>.*?</\s*{prefix}think(?:ing)?\s*>\s*",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
    return cleaned.strip()


def _critic_reference_dialogue(case: BenchmarkCase) -> list[dict[str, Any]]:
    cleaned_dialogue: list[dict[str, Any]] = []
    for item in CoserAdapter._reference_dialogue(case):
        cleaned = dict(item)
        cleaned.pop("inner_thought", None)
        speaker = str(cleaned.get("speaker") or cleaned.get("character") or "")
        if speaker != ENVIRONMENT_ROLE:
            for field in ("content", "message", "speech"):
                if isinstance(cleaned.get(field), str):
                    cleaned[field] = _remove_inner_thoughts(str(cleaned[field]))
        cleaned_dialogue.append(cleaned)
    return cleaned_dialogue


def _dialogue_text(dialogue: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for item in dialogue:
        speaker = str(item.get("speaker") or item.get("character") or "")
        content = item.get("content")
        if content is None:
            inner_thought = str(item.get("inner_thought") or "").strip()
            speech = str(item.get("speech") or "").strip()
            action = str(item.get("action") or "").strip()
            content = " ".join(
                part
                for part in (
                    f"[{inner_thought}]" if inner_thought else "",
                    speech,
                    f"({action})" if action else "",
                )
                if part
            )
        lines.append(f"{speaker}: {str(content).strip()}")
    return "\n\n".join(lines)


def build_official_coser_critic_prompt(case: BenchmarkCase, dimension: str) -> str:
    """Render the revision-frozen upstream CoSER critic system prompt."""

    if dimension not in OFFICIAL_COSER_DIMENSION_DETAILS:
        raise ValidationError(f"unsupported CoSER dimension {dimension!r}")
    characters = case.input_data.get("characters")
    if isinstance(characters, (str, bytes)) or not isinstance(characters, Sequence) or not characters:
        raise ValidationError("CoSER case requires characters")
    names: list[str] = []
    profiles: list[str] = []
    plot = case.input_data.get("plot")
    if not isinstance(plot, Mapping):
        raise ValidationError("CoSER plot must be an object")
    for index, raw in enumerate(characters):
        if not isinstance(raw, Mapping):
            raise ValidationError(f"CoSER character #{index} must be an object")
        name = str(raw.get("name") or raw.get("id") or "").strip()
        profile = compose_coser_profile(plot, name, str(raw.get("profile") or ""))
        if not name:
            raise ValidationError(f"CoSER character #{index} lacks a name")
        names.append(name)
        if profile:
            profiles.append(f"### {name}\n\n{profile}")
    raw_major = case.input_data.get("major_characters")
    if raw_major is None:
        major_characters = names
    elif isinstance(raw_major, Sequence) and not isinstance(raw_major, (str, bytes)):
        major_characters = [str(name).strip() for name in raw_major if str(name).strip()]
        if not major_characters or not set(major_characters).issubset(set(names)):
            raise ValidationError("CoSER major_characters must be nonempty speaking character names")
    else:
        raise ValidationError("CoSER major_characters must be an array")
    replacements = {
        "{book}": str(case.input_data.get("story_identity") or "").strip(),
        "{major_characters}": ", ".join(major_characters),
        "{dimension_name}": COSER_DIMENSIONS[dimension],
        "{dimension_brief}": OFFICIAL_COSER_DIMENSION_DETAILS[dimension]["brief"],
        "{additional_instructions}": "",
        "{plot_summary}": str(plot.get("summary") or "").strip(),
        "{scenario}": str(case.input_data.get("scene_context") or "").strip(),
        "{character_profiles}": "\n\n".join(profiles),
        "{original_conversation}": _dialogue_text(_critic_reference_dialogue(case)),
        "{dimension_criteria}": OFFICIAL_COSER_DIMENSION_DETAILS[dimension]["criteria"].replace(
            "{major_characters}", ", ".join(major_characters)
        ),
    }
    required = ("{book}", "{plot_summary}", "{scenario}", "{original_conversation}")
    if not all(replacements[key] for key in required):
        raise ValidationError("CoSER official critic context is incomplete")
    prompt = OFFICIAL_COSER_CRITIC_TEMPLATE
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


@adapter("coser")
class CoserAdapter(BenchmarkAdapter):
    benchmark_id = "coser"
    prompt_revision = "coser-gca-upstream-exact-critic-prompt-v7"
    scorer_revision = CoserScorer.scorer_revision

    def __init__(
        self,
        *,
        judge_provenance: JudgeProvenance | None = None,
        runtime_provenance: CoserRuntimeProvenance | None = None,
    ) -> None:
        self.environment = CoserEnvironment()
        self.scorer = CoserScorer()
        self._judge_provenance = judge_provenance
        self._runtime_provenance = runtime_provenance

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"CoserAdapter cannot run {case.benchmark_id!r}")
        probe_case(case)
        self.environment.reset(case, seed=0)

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        state = self.environment.reset(case, seed=seed)
        runtime = self.runtime_for_case(case)
        return self.build_role_request(case, state, state.current_speaker, model=model, runtime=runtime, seed=seed)

    def parse_response(self, case: BenchmarkCase, response: ModelResponse) -> Any:
        return parse_coser_role_output(response.text)

    @staticmethod
    def _mapping_from_replay(case: BenchmarkCase, key: str) -> Mapping[str, Any] | None:
        replay = case.metadata.get("replay")
        raw = replay.get(key) if isinstance(replay, Mapping) else None
        return raw if isinstance(raw, Mapping) else None

    def provenance_for_case(self, case: BenchmarkCase) -> JudgeProvenance | None:
        if self._judge_provenance is not None:
            return self._judge_provenance
        raw = self._mapping_from_replay(case, "judge_provenance")
        if raw is None:
            return None
        provenance = JudgeProvenance(
            judge_models=tuple(raw.get("judge_models") or ()),
            judge_revisions=tuple(raw.get("judge_revisions") or ()),
            rubric_revision=str(raw.get("rubric_revision") or ""),
            calls_per_output=int(raw.get("calls_per_output", len(COSER_DIMENSIONS))),
            source=str(raw.get("source") or "synthetic_fixture"),
        )
        if len(provenance.judge_models) != 1 or provenance.calls_per_output != len(COSER_DIMENSIONS):
            raise ConfigurationError("CoSER GCA requires one pinned critic and four calls per scene")
        return provenance

    def runtime_for_case(self, case: BenchmarkCase) -> CoserRuntimeProvenance:
        if self._runtime_provenance is not None:
            return self._runtime_provenance
        raw = self._mapping_from_replay(case, "runtime_provenance")
        if raw is None:
            raise ConfigurationError("CoSER requires environment and next-speaker runtime provenance")
        return CoserRuntimeProvenance(
            environment_model=str(raw.get("environment_model") or ""),
            environment_revision=str(raw.get("environment_revision") or ""),
            next_speaker_model=str(raw.get("next_speaker_model") or ""),
            next_speaker_revision=str(raw.get("next_speaker_revision") or ""),
            retrieval_mode=str(raw.get("retrieval_mode") or "none"),
            retrieval_k=int(raw.get("retrieval_k", 0)),
            hidden_thought_policy=str(
                raw.get("hidden_thought_policy") or OFFICIAL_HIDDEN_THOUGHT_POLICY
            ),
            max_judge_context_tokens=int(raw.get("max_judge_context_tokens", 131072)),
            source=str(raw.get("source") or "synthetic_fixture"),
        )

    def assistant_or_partner_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        return self.runtime_for_case(case).to_dict()

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        state = self.environment.reset(case, seed=0)
        return {
            "revision": self.environment.environment_revision,
            "context_policy_revision": self.environment.context_policy_revision,
            "tokenizer_revision": self.environment.tokenizer_revision,
            "max_turns": state.max_turns,
            "min_end_turns": state.min_end_turns,
            "max_context_tokens": state.max_context_tokens,
            "include_environment": ENVIRONMENT_ROLE in state.speaking_roles,
        }

    def build_role_request(
        self,
        case: BenchmarkCase,
        state: CoserState,
        actor: str,
        *,
        model: str,
        runtime: CoserRuntimeProvenance,
        seed: int,
    ) -> ModelRequest:
        selection = self.environment.assemble_context(
            state,
            actor=actor,
            add_output_example=actor != ENVIRONMENT_ROLE and "coser" not in model.casefold(),
        )
        request_model = runtime.environment_model if actor == ENVIRONMENT_ROLE else model
        return ModelRequest(
            request_id=f"{case.case_id}:turn:{state.turn_count}:{actor}",
            messages=selection.messages,
            model=request_model,
            temperature=0,
            max_tokens=4096 if actor == ENVIRONMENT_ROLE else COSER_CHARACTER_MAX_TOKENS,
            seed=seed + state.turn_count,
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": actor,
                "route_role": "environment_actor" if actor == ENVIRONMENT_ROLE else "evaluated_actor",
                "actor_kind": "environment" if actor == ENVIRONMENT_ROLE else "character",
                "prompt_revision": self.prompt_revision,
                "context_provenance": {"request_kind": "role", **dict(selection.provenance)},
            },
        )

    def build_nsp_request(
        self,
        case: BenchmarkCase,
        state: CoserState,
        runtime: CoserRuntimeProvenance,
        *,
        seed: int,
    ) -> ModelRequest:
        allowed_speakers = [character.name for character in state.characters.values()]
        if ENVIRONMENT_ROLE in state.speaking_roles:
            allowed_speakers.append("Environment")
        system_content = (
            "Your task is to predict the next speaker for a role-playing game. That is, you need to determine "
            "which character (or the Environment) might act next based on their previous interactions. The "
            "Environment is a special role that provides the environmental feedback. Choose a name from this "
            f"list: {allowed_speakers}. If it's unclear who should act next, output \"random\". If you believe "
            "the scene or conversation should conclude, output \"<END CHAT>\".\n\n"
            f"===The scenario is as follows===\n{state.scene_context}\n\n"
            "===Output Format===\n"
            "You must structure your response as follows:\n"
            "1. Reasoning: Explain your thought process.\n"
            f"2. Next Speaker: Select exactly one character name from the list: {allowed_speakers}.\n"
        )
        user_content = "===Conversation Start===\n\n" + _dialogue_text(_public_dialogue(state))
        return ModelRequest(
            request_id=f"{case.case_id}:nsp:{state.turn_count}",
            messages=(
                ChatMessage(
                    "system",
                    system_content,
                    metadata={"visibility": "public_scheduler"},
                ),
                ChatMessage("user", user_content, metadata={"visibility": "public_scheduler"}),
            ),
            model=runtime.next_speaker_model,
            temperature=0,
            seed=seed + state.turn_count,
            metadata={
                "benchmark_id": self.benchmark_id,
                "stage": "next_speaker",
                "route_role": "next_speaker",
                "context_provenance": {
                    "request_kind": "next_speaker",
                    "visibility": "public_scheduler",
                    "private_context_included": False,
                    "memory_ids": [],
                    "public_transcript_turns": [item["turn"] for item in _public_dialogue(state)],
                    "tokenizer_revision": self.environment.tokenizer_revision,
                    "used_tokens": deterministic_token_count(system_content)
                    + deterministic_token_count(user_content),
                    "truncated": False,
                },
            },
        )

    @staticmethod
    def _reference_dialogue(case: BenchmarkCase) -> Sequence[Mapping[str, Any]]:
        reference = case.gold.get("reference_dialogue") if isinstance(case.gold, Mapping) else None
        if reference is None:
            return ()
        if not isinstance(reference, Sequence) or isinstance(reference, (str, bytes)):
            raise ValidationError("CoSER gold.reference_dialogue must be an array when present")
        if not all(isinstance(item, Mapping) for item in reference):
            raise ValidationError("each CoSER reference dialogue item must be an object")
        return reference

    def build_judge_request(
        self,
        case: BenchmarkCase,
        state: CoserState,
        provenance: JudgeProvenance,
        runtime: CoserRuntimeProvenance,
        dimension: str,
        *,
        seed: int,
    ) -> ModelRequest:
        system_prompt = build_official_coser_critic_prompt(case, dimension)
        dialogue = _dialogue_text(_public_dialogue(state))
        used_tokens = deterministic_token_count(system_prompt) + deterministic_token_count(dialogue)
        if used_tokens > runtime.max_judge_context_tokens:
            raise ContextLimitError(
                f"protected CoSER judge context needs {used_tokens} accounting tokens, "
                f"budget is {runtime.max_judge_context_tokens}; reference/profile context was not truncated"
            )
        flaw_schema = {
            "type": "object",
            "properties": {
                "instance": {"type": "string", "minLength": 1},
                "type": {"type": "string", "minLength": 1},
                "severity": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["instance", "type", "severity"],
            "additionalProperties": False,
        }
        official_name = COSER_DIMENSIONS[dimension]
        response_schema = {
            "type": "object",
            "properties": {
                official_name: {
                    "type": "object",
                    "properties": {"flaws": {"type": "array", "items": flaw_schema}},
                    "required": ["flaws"],
                    "additionalProperties": False,
                }
            },
            "required": [official_name],
            "additionalProperties": False,
        }
        output_contract = (
            "Return only the official CoSER JSON envelope with exactly this shape: "
            f'{{"{official_name}":{{"flaws":[]}}}}. '
            "Every flaw object must contain exactly instance (non-empty text), type (non-empty text), and "
            "severity (integer 1-5). Use an empty array when no flaw exists; never invent placeholder flaws."
        )
        return ModelRequest(
            request_id=f"{case.case_id}:judge:{dimension}",
            messages=(
                ChatMessage(
                    "system",
                    system_prompt,
                    metadata={"visibility": "evaluator_only"},
                ),
                ChatMessage("user", dialogue, metadata={"visibility": "evaluator_only"}),
            ),
            model=provenance.judge_models[0],
            temperature=0,
            seed=seed,
            response_format=json_schema_response_format(f"coser_{dimension}_critic", response_schema),
            metadata={
                "benchmark_id": self.benchmark_id,
                "route_role": "judge",
                "dimension": dimension,
                "rubric_revision": provenance.rubric_revision,
                "prompt_source": "CoSER gca_evaluation/prompts.py self-play-deduct-template",
                "critic_prompt_revision": OFFICIAL_COSER_CRITIC_PROMPT_REVISION,
                "input_protocol": "upstream_system_prompt_and_plain_simulation_user_message",
                "explicit_character_goals_included": False,
                "full_structured_plot_included": False,
                "output_contract": output_contract,
                "context_policy": "protected_evaluator_context_fail_if_over_budget",
                "accounting_tokens": used_tokens,
                "context_provenance": {
                    "request_kind": "judge",
                    "visibility": "evaluator_only",
                    "dimension": dimension,
                    "tokenizer_revision": self.environment.tokenizer_revision,
                    "budget_tokens": runtime.max_judge_context_tokens,
                    "used_tokens": used_tokens,
                    "protected_context_truncated": False,
                    "hidden_thoughts_included": False,
                },
            },
        )

    def replay_responses(self, case: BenchmarkCase, *, seed: int) -> Mapping[str, Any]:
        replay = case.metadata.get("replay")
        if not isinstance(replay, Mapping):
            raise ConfigurationError(f"fixture {case.case_id} has no CoSER replay metadata")
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
            actor = str(step.get("speaker") or "")
            if actor != state.current_speaker:
                raise ConfigurationError(
                    f"fixture {case.case_id} replay speaker #{index} is {actor!r}, expected {state.current_speaker!r}"
                )
            request = self.build_role_request(
                case, state, actor, model="offline-replay", runtime=runtime, seed=seed
            )
            output = step.get("output")
            if not isinstance(output, Mapping):
                raise ConfigurationError(f"fixture {case.case_id} replay step #{index} needs output object")
            text = canonical_json(output)
            responses[request.request_id or ""] = {"text": text, "finish_reason": "replayed"}
            parsed = parse_coser_role_output(text, allow_inner_thought=actor != ENVIRONMENT_ROLE)
            self.environment.apply(state, actor=actor, action=parsed)
            if state.terminal:
                if "next_speaker" in step:
                    raise ConfigurationError(f"fixture {case.case_id} max-turn step cannot include next_speaker")
                continue
            next_speaker = step.get("next_speaker")
            if not isinstance(next_speaker, str) or not next_speaker:
                raise ConfigurationError(f"fixture {case.case_id} replay step #{index} needs next_speaker")
            nsp_request = self.build_nsp_request(case, state, runtime, seed=seed)
            nsp_text = canonical_json({"next_speaker": next_speaker})
            responses[nsp_request.request_id or ""] = {"text": nsp_text, "finish_reason": "replayed"}
            self.environment.choose_next(state, raw_next_speaker=next_speaker)
        if not state.terminal:
            raise ConfigurationError(f"fixture {case.case_id} replay steps do not terminate the CoSER scene")
        judges = replay.get("judges")
        if isinstance(judges, Mapping):
            for dimension in COSER_DIMENSIONS:
                if dimension not in judges:
                    raise ConfigurationError(f"fixture {case.case_id} lacks CoSER judge response for {dimension}")
                fixture_payload = judges[dimension]
                if not isinstance(fixture_payload, Mapping):
                    raise ConfigurationError(f"fixture {case.case_id} CoSER judge response must be an object")
                # Historical synthetic fixtures stored an extended internal envelope. Replay only the scene flaws
                # through the exact released critic output contract.
                if "scene_flaws" in fixture_payload:
                    fixture_payload = {
                        COSER_DIMENSIONS[dimension]: {"flaws": fixture_payload["scene_flaws"]}
                    }
                responses[f"{case.case_id}:judge:{dimension}"] = {
                    "text": canonical_json(fixture_payload),
                    "finish_reason": "replayed",
                }
        return responses

    @staticmethod
    def _failure_result(
        case: BenchmarkCase,
        state: CoserState,
        *,
        run_id: str,
        repetition: int,
        responses: Sequence[ModelResponse],
        context_audits: Sequence[Mapping[str, Any]],
        stage: str,
        kind: str,
        message: str,
        retryable: bool,
    ) -> CaseResult:
        return CaseResult(
            run_id=run_id,
            benchmark_id="coser",
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.FAILED,
            trace=tuple(state.transcript),
            model_response=responses[-1] if responses else None,
            error=ErrorState(stage, kind, message, retryable=retryable),
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "environment_revision": CoserEnvironment.environment_revision,
                "terminal_reason": state.terminal_reason,
                "episode_complete": False,
                "turn_count": state.turn_count,
                "context_audits": list(context_audits),
            },
        )

    def _target_parse_failure_result(
        self,
        case: BenchmarkCase,
        state: CoserState,
        *,
        run_id: str,
        repetition: int,
        responses: Sequence[ModelResponse],
        context_audits: Sequence[Mapping[str, Any]],
        stage: str,
        kind: str,
        message: str,
        runtime: CoserRuntimeProvenance,
    ) -> CaseResult:
        failure = {
            "stage": stage,
            "kind": kind,
            "message": message,
            "retryable": False,
            "scoring_policy": "target_capability_failure_scores_zero",
        }
        metrics: list[MetricValue] = []
        for dimension in COSER_DIMENSIONS:
            metrics.append(
                MetricValue(
                    f"coser.scene.{dimension}",
                    0,
                    unit="score_0_to_100",
                    numerator=0,
                    denominator=1,
                    metadata={"target_output_failure": failure, "scope": "scene_official_gca"},
                )
            )
            for character_id, character in state.characters.items():
                metrics.append(
                    MetricValue(
                        f"coser.character.{_safe_metric_component(character_id)}.{dimension}",
                        None,
                        unit="score_0_to_100",
                        metadata={
                            "availability": "unavailable",
                            "reason": "official CoSER critic has no per-character score after target output failure",
                            "character_id": character_id,
                            "character_name": character.name,
                        },
                    )
                )
        metrics.append(
            MetricValue(
                "coser.scene.critic_average",
                0,
                unit="score_0_to_100",
                numerator=0,
                denominator=len(COSER_DIMENSIONS),
                metadata={"target_output_failure": failure},
            )
        )
        for character_id, character in state.characters.items():
            metrics.append(
                MetricValue(
                    f"coser.character.{_safe_metric_component(character_id)}.critic_average",
                    None,
                    unit="score_0_to_100",
                    metadata={
                        "availability": "unavailable",
                        "reason": "official CoSER critic has no per-character score after target output failure",
                        "character_id": character_id,
                        "character_name": character.name,
                    },
                )
            )
        for name in ("bleu", "rouge_l"):
            metrics.append(
                MetricValue(
                    f"coser.scene.{name}",
                    0.0,
                    unit="0_to_1",
                    numerator=0.0,
                    denominator=1,
                    metadata={"target_output_failure": failure},
                )
            )
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction={
                "public_dialogue": _public_dialogue(state),
                "terminal_reason": state.terminal_reason,
                "turn_count": state.turn_count,
                "parsed": False,
            },
            metrics=tuple(metrics),
            trace=tuple(state.transcript),
            model_response=responses[-1] if responses else None,
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "environment_revision": self.environment.environment_revision,
                "prompt_revision": self.prompt_revision,
                "scorer_revision": self.scorer_revision,
                "terminal_reason": state.terminal_reason,
                "episode_complete": True,
                "natural_termination": False,
                "turn_count": state.turn_count,
                "context_audits": list(context_audits),
                "target_output_failure": failure,
                "judge_status": {dimension: "not_called_target_output_failure" for dimension in COSER_DIMENSIONS},
                "judge_errors": {},
                "judge_call_count": 0,
                "judge_provenance": (
                    self.provenance_for_case(case).to_dict() if self.provenance_for_case(case) else None
                ),
                "runtime_provenance": runtime.to_dict(),
                "hidden_thoughts_shared_with_critic": False,
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
        runtime = self.runtime_for_case(case)
        state = self.environment.reset(case, seed=seed)
        responses: list[ModelResponse] = []
        context_audits: list[Mapping[str, Any]] = []
        budget_termination: Mapping[str, Any] | None = None
        if previous_result is not None:
            from ..judge_resume import restore_judge_state
            restore_judge_state(state, previous_result)
        while previous_result is None and not state.terminal:
            actor = state.current_speaker
            stage_prefix = "environment" if actor == ENVIRONMENT_ROLE else "character"
            try:
                request = self.build_role_request(case, state, actor, model=model, runtime=runtime, seed=seed)
                context_audits.append(dict(request.metadata["context_provenance"]))
            except ContextLimitError as exc:
                self.environment.fail(state, actor=actor, stage="context_assembly", failure_kind="context_limit")
                return self._failure_result(
                    case,
                    state,
                    run_id=run_id,
                    repetition=repetition,
                    responses=responses,
                    context_audits=context_audits,
                    stage="context_assembly",
                    kind="context_limit",
                    message=str(exc),
                    retryable=False,
                )
            try:
                output = generate_and_parse_with_contract_retries(
                    backend=backend,
                    request=request,
                    parser=lambda response: parse_coser_role_output(
                        response.text,
                        allow_inner_thought=actor != ENVIRONMENT_ROLE,
                    ),
                    responses=responses,
                    contract=(
                        _ENVIRONMENT_OUTPUT_CONTRACT if actor == ENVIRONMENT_ROLE else _ROLE_OUTPUT_CONTRACT
                    ),
                )
            except EpisodeTokenBudgetExhausted as exc:
                budget_termination = exc.details()
                self.environment.terminate_capability_limit(
                    state,
                    actor=actor,
                    reason="token_budget_exhausted",
                    details=budget_termination,
                )
                break
            except BackendError as exc:
                self.environment.fail(state, actor=actor, stage=f"{stage_prefix}_backend", failure_kind=type(exc).__name__)
                return self._failure_result(
                    case,
                    state,
                    run_id=run_id,
                    repetition=repetition,
                    responses=responses,
                    context_audits=context_audits,
                    stage=f"{stage_prefix}_backend",
                    kind=type(exc).__name__,
                    message=str(exc),
                    retryable=True,
                )
            except ParseError as exc:
                kind = "empty_action" if "empty_action" in str(exc) else "invalid_role_output"
                self.environment.fail(state, actor=actor, stage=f"{stage_prefix}_parse", failure_kind=kind)
                if actor != ENVIRONMENT_ROLE:
                    return self._target_parse_failure_result(
                        case,
                        state,
                        run_id=run_id,
                        repetition=repetition,
                        responses=responses,
                        context_audits=context_audits,
                        stage=f"{stage_prefix}_parse",
                        kind=kind,
                        message=str(exc),
                        runtime=runtime,
                    )
                return self._failure_result(
                    case,
                    state,
                    run_id=run_id,
                    repetition=repetition,
                    responses=responses,
                    context_audits=context_audits,
                    stage=f"{stage_prefix}_parse",
                    kind=kind,
                    message=str(exc),
                    retryable=False,
                )
            self.environment.apply(state, actor=actor, action=output)
            if state.terminal:
                break
            nsp_request = self.build_nsp_request(case, state, runtime, seed=seed)
            context_audits.append(dict(nsp_request.metadata["context_provenance"]))
            try:
                next_speaker = generate_and_parse_with_contract_retries(
                    backend=backend,
                    request=nsp_request,
                    parser=lambda response: parse_next_speaker(response.text),
                    responses=responses,
                    contract=_NEXT_SPEAKER_CONTRACT,
                )
                self.environment.choose_next(state, raw_next_speaker=next_speaker)
            except BackendError as exc:
                self.environment.fail(
                    state, actor="next_speaker_predictor", stage="nsp_backend", failure_kind=type(exc).__name__
                )
                return self._failure_result(
                    case,
                    state,
                    run_id=run_id,
                    repetition=repetition,
                    responses=responses,
                    context_audits=context_audits,
                    stage="nsp_backend",
                    kind=type(exc).__name__,
                    message=str(exc),
                    retryable=True,
                )
            except ParseError as exc:
                self.environment.fail(state, actor="next_speaker_predictor", stage="nsp_parse", failure_kind="invalid_nsp")
                return self._failure_result(
                    case,
                    state,
                    run_id=run_id,
                    repetition=repetition,
                    responses=responses,
                    context_audits=context_audits,
                    stage="nsp_parse",
                    kind="invalid_nsp",
                    message=str(exc),
                    retryable=False,
                )

        provenance = self.provenance_for_case(case)
        critic_metrics: list[MetricValue] = []
        judge_status: dict[str, str] = {}
        judge_errors: dict[str, Mapping[str, str]] = {}
        judge_call_count = 0
        if provenance is None:
            for dimension in COSER_DIMENSIONS:
                judge_status[dimension] = "unavailable"
                critic_metrics.extend(
                    self.scorer.unavailable_dimension(
                        dimension, reason="judge configuration is required", state=state, provenance=None
                    )
                )
        else:
            for dimension in COSER_DIMENSIONS:
                old_dimension = [m for m in previous_result.metrics if m.name.endswith("." + dimension)] if previous_result else []
                if any(m.name == f"coser.scene.{dimension}" and m.value is not None for m in old_dimension):
                    critic_metrics.extend(old_dimension)
                    judge_status[dimension] = "available"
                    continue
                try:
                    judge_request = self.build_judge_request(
                        case, state, provenance, runtime, dimension, seed=seed
                    )
                    context_audits.append(dict(judge_request.metadata["context_provenance"]))
                    judge_call_count += 1

                    def parse_critic_response(response: ModelResponse) -> Mapping[str, Any]:
                        candidate = parse_json_object(response, label="CoSER critic response")
                        # Run the full strict semantic validator inside the retry loop.
                        self.scorer.score_official_dimension(
                            candidate,
                            dimension=dimension,
                            state=state,
                            provenance=provenance,
                        )
                        return candidate

                    payload = generate_and_parse_with_contract_retries(
                        backend=backend,
                        request=judge_request,
                        parser=parse_critic_response,
                        responses=responses,
                        contract=str(judge_request.metadata["output_contract"]),
                    )
                    critic_metrics.extend(
                        self.scorer.score_official_dimension(
                            payload, dimension=dimension, state=state, provenance=provenance
                        )
                    )
                    judge_status[dimension] = "available"
                except (BackendError, ContextLimitError, ParseError) as exc:
                    judge_status[dimension] = "unavailable"
                    judge_errors[dimension] = {"kind": type(exc).__name__, "message": str(exc)}
                    critic_metrics.extend(
                        self.scorer.unavailable_dimension(
                            dimension,
                            reason=f"judge failure: {type(exc).__name__}",
                            state=state,
                            provenance=provenance,
                        )
                    )
        metrics = list(self.scorer.add_averages(critic_metrics, state))
        reference = _critic_reference_dialogue(case)
        generated_dialogue = _public_dialogue(state)
        if reference:
            reference_text = _dialogue_text(reference)
            generated_text = _dialogue_text(generated_dialogue)
            metrics.extend(
                (
                    MetricValue(
                        "coser.scene.bleu",
                        dependency_free_bleu(reference_text, generated_text),
                        unit="0_to_1",
                        metadata={
                            "protocol_status": "dependency_free_tokenizer_variant_of_official_nltk_bleu",
                            "tokenizer_revision": self.environment.tokenizer_revision,
                        },
                    ),
                    MetricValue(
                        "coser.scene.rouge_l",
                        dependency_free_rouge_l(reference_text, generated_text),
                        unit="0_to_1",
                        metadata={
                            "protocol_status": "dependency_free_tokenizer_variant_of_official_rouge_l_f1",
                            "tokenizer_revision": self.environment.tokenizer_revision,
                        },
                    ),
                )
            )
        else:
            for name in ("bleu", "rouge_l"):
                metrics.append(
                    MetricValue(
                        f"coser.scene.{name}",
                        None,
                        unit="0_to_1",
                        metadata={"availability": "unavailable", "reason": "reference dialogue is absent"},
                    )
                )
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction={
                "public_dialogue": generated_dialogue,
                "terminal_reason": state.terminal_reason,
                "turn_count": state.turn_count,
            },
            metrics=tuple(metrics),
            trace=tuple(state.transcript),
            model_response=responses[-1] if responses else None,
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "environment_revision": self.environment.environment_revision,
                "prompt_revision": self.prompt_revision,
                "scorer_revision": self.scorer_revision,
                "terminal_reason": state.terminal_reason,
                "episode_complete": True,
                "natural_termination": state.terminal_reason == "nsp_end",
                "budget_termination": budget_termination,
                "turn_count": state.turn_count,
                "context_audits": context_audits,
                "judge_status": judge_status,
                "judge_errors": judge_errors,
                "judge_call_count": judge_call_count,
                "judge_provenance": provenance.to_dict() if provenance else None,
                "runtime_provenance": runtime.to_dict(),
                "hidden_thoughts_shared_with_critic": False,
            },
        )

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        return aggregate_named_metrics(results, namespace="coser")


__all__ = [
    "COSER_DIMENSIONS",
    "OFFICIAL_COSER_CRITIC_PROMPT_REVISION",
    "OFFICIAL_COSER_CRITIC_TEMPLATE",
    "CoserAdapter",
    "CoserRuntimeProvenance",
    "CoserScorer",
    "LENGTH_CORRECTION_PER_ACTOR_TURN",
    "build_official_coser_critic_prompt",
    "coser_length_corrected_score",
    "dependency_free_bleu",
    "dependency_free_rouge_l",
]
