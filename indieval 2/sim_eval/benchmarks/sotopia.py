"""SOTOPIA social-scene adapter with native dimension ranges and judge provenance."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import replace
from typing import Any, Mapping, Sequence

from ..backends.replay import ReplayBackend
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
from ..environments.social import (
    JudgeProvenance,
    RoleIsolatedSocialEnvironment,
    SocialAction,
    SocialState,
    parse_social_action,
)
from ..errors import (
    BackendError,
    BackendSafetyError,
    BackendStructuredOutputError,
    BackendTokenLimitError,
    ConfigurationError,
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


SOTOPIA_DIMENSIONS: Mapping[str, tuple[int, int]] = {
    "believability": (0, 10),
    "relationship": (-5, 5),
    "knowledge": (0, 10),
    "secret": (-10, 0),
    "social_rules": (-10, 0),
    "financial_and_material_benefits": (-5, 5),
    "goal": (0, 10),
}
SOTOPIA_RUBRICS: Mapping[str, str] = {
    "believability": (
        "Reasoning requirement: 1. Evaluate if the agent interacts with others in a natural and realistic manner "
        "(here are a few common questions to check: a. whether the agent is confusing with its own identity? "
        "b. whether the agent repeats others' words/actions without any reason? c. whether the agent is being "
        "overly polite considering the context?). Start the analysis with tag <naturalness>. 2. Analyze whether "
        "the actions of the agent align with their character traits (e.g., personality, values, and etc.). Start "
        "the analysis with tag <consistency>. Output your reasoning process to the 'reasoning' field. Output an "
        "integer score ranging from 0 and 10 in the 'score' field. A higher score indicates that the agent is "
        "more believable."
    ),
    "relationship": (
        "Please first analyze what relationship the participant has with the other agent(s) before the "
        "interaction. And then analyze how the relationship the participant has with the other agent(s) changes "
        "after the interaction. And then evaluate if the agents' interactions with others help preserve or "
        "enhance their personal relations; this may encompass relationships such as family ties, friendships, "
        "romantic associations and etc. Additionally, ascertain whether these interactions also impact their "
        "social status or reputation. In the 'reasoning' field, provide a comprehensive account of the logic or "
        "thought process that led you to your conclusion. Further, provide an integer score ranging from -5 to 5 "
        "in the 'score' field. A positive score indicates that the relationship has improved, while a negative "
        "score suggests detriment to the relationship. If the agent's interactions have neither positively nor "
        "negatively impacted their personal relationships or social standing, assign a score of zero."
    ),
    "knowledge": (
        "Please first assess what information the agent has gained through the interaction. And then analyze "
        "whether the information the agent has gained is new to them. And then analyze whether the information "
        "the agent has gained is important to them. In the 'reasoning' field, provide a comprehensive account of "
        "the logic or thought process that led you to your conclusion. Further, provide an integer score ranging "
        "from 0 and 10 in the 'score' field. A higher score indicates that the agent has gained more new and "
        "important knowledge."
    ),
    "secret": (
        "First please analyze what secret or secretive intention the participant wants to keep. And then analyze "
        "whether they keep it successfully. Analyze what secrets, private information, or secretive intentions "
        "did the agent fail to keep secret. Please output the reasoning in the reasoning field. Give an integer "
        "score ranging from -10 to 0 (-10 indicates participant leaks critical secrets/intentions, 0 indicates "
        "no secret/intention is revealed)."
    ),
    "social_rules": (
        "Please evaluate whether the agent violates any moral rules or laws in their interactions with others. "
        "In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you "
        "to your conclusion. Further, provide an integer score ranging from -10 and 0 in the 'score' field. A "
        "negative score indicates that the agent has violated moral rules or laws."
    ),
    "financial_and_material_benefits": (
        "Please evaluate whether the agent's interactions with others contribute towards financial and material "
        "benefits. Analyze what the agent would gain/lose after the interactions. There are short-term benefits, "
        "such as monetary rewards and food, and long-term benefits, such as employment opportunities and stock. "
        "In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you "
        "to your conclusion. Further, provide an integer score ranging from -5 and 5 in the 'score' field. "
        "Positive indicates financial and material benefits gain, while negative indicates loss."
    ),
    "goal": (
        "Please first reiterate agent's social goals. And then please provide a comprehensive analysis about the "
        "extent to which the agent has managed to achieve these goals. In the 'reasoning' field, provide a "
        "comprehensive account of the logic or thought process that led you to your conclusion. Further, provide "
        "an integer score ranging from 0 and 10 in the 'score' field. 0 represents minimal goals achievement, 10 "
        "represents complete goal achievement, and a higher score indicates that the agent is making progress "
        "towards their social goals."
    ),
}

_SOCIAL_ACTION_CONTRACT = (
    'Return exactly one JSON object with exactly two fields: '
    '{"action_type":"none|speak|non-verbal communication|action|leave","argument":"..."}. '
    "action_type and argument must both be strings; emit no prose outside the JSON object."
)


def normalize_sotopia_dimension(name: str, value: int) -> float:
    try:
        low, high = SOTOPIA_DIMENSIONS[name]
    except KeyError as exc:
        raise ValidationError(f"unknown SOTOPIA dimension {name!r}") from exc
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValidationError(f"SOTOPIA {name} must be an integer in [{low}, {high}], got {value!r}")
    return (value - low) / (high - low)


class SotopiaScorer:
    scorer_revision = "sotopia-dimensions-role-isolated-complete-pair-aggregation-v4"

    def unavailable(self, reason: str, provenance: JudgeProvenance | None) -> tuple[MetricValue, ...]:
        provenance_data = provenance.to_dict() if provenance else None
        metrics: list[MetricValue] = []
        for dimension in SOTOPIA_DIMENSIONS:
            metadata = {
                "availability": "unavailable",
                "reason": reason,
                "judge_provenance": provenance_data,
                "aggregation": "evaluated_agent_only",
                "official_primary": True,
            }
            metrics.append(MetricValue(f"sotopia.evaluated_agent.{dimension}", None, metadata=metadata))
            metrics.append(
                MetricValue(
                    f"sotopia.evaluated_agent.{dimension}.normalized",
                    None,
                    unit="0_to_1",
                    metadata={**metadata, "official_primary": False, "native_metric": dimension},
                )
            )
        metrics.append(
            MetricValue(
                "sotopia.evaluated_agent.normalized_dimension_mean",
                None,
                unit="0_to_1",
                metadata={
                    "availability": "unavailable",
                    "reason": reason,
                    "judge_provenance": provenance_data,
                    "official_primary": False,
                },
            )
        )
        return tuple(metrics)

    def score_payload(
        self,
        payload: Mapping[str, Any],
        *,
        agent_ids: Sequence[str],
        provenance: JudgeProvenance,
        evaluated_agent_id: str | None,
        partner_agent_id: str | None,
    ) -> tuple[MetricValue, ...]:
        scores = payload.get("scores")
        if not isinstance(scores, Mapping):
            raise ParseError("SOTOPIA judge response requires a scores object")
        if set(scores) != set(agent_ids):
            raise ParseError(f"SOTOPIA judge score agents differ: expected={sorted(agent_ids)}, got={sorted(scores)}")
        if evaluated_agent_id is not None and evaluated_agent_id not in agent_ids:
            raise ParseError(f"SOTOPIA evaluated agent {evaluated_agent_id!r} is absent from judge scores")
        if partner_agent_id is not None and partner_agent_id not in agent_ids:
            raise ParseError(f"SOTOPIA partner agent {partner_agent_id!r} is absent from judge scores")
        if evaluated_agent_id is not None and evaluated_agent_id == partner_agent_id:
            raise ParseError("SOTOPIA evaluated and partner agent IDs must differ")

        metrics: list[MetricValue] = []
        native_by_agent: dict[str, dict[str, int]] = {agent_id: {} for agent_id in agent_ids}
        normalized_by_agent: dict[str, dict[str, float]] = {agent_id: {} for agent_id in agent_ids}
        provenance_data = provenance.to_dict()
        for agent_id in agent_ids:
            agent_scores = scores[agent_id]
            if not isinstance(agent_scores, Mapping):
                raise ParseError(f"SOTOPIA judge scores for {agent_id} must be an object")
            if set(agent_scores) != set(SOTOPIA_DIMENSIONS):
                raise ParseError(
                    f"SOTOPIA dimensions for {agent_id} differ; "
                    f"missing={sorted(set(SOTOPIA_DIMENSIONS)-set(agent_scores))}, "
                    f"extra={sorted(set(agent_scores)-set(SOTOPIA_DIMENSIONS))}"
                )
            for dimension, (low, high) in SOTOPIA_DIMENSIONS.items():
                item = agent_scores[dimension]
                if not isinstance(item, Mapping):
                    raise ParseError(f"SOTOPIA {agent_id}/{dimension} requires reasoning and score")
                reasoning = item.get("reasoning")
                value = item.get("score")
                if not isinstance(reasoning, str) or not reasoning.strip():
                    raise ParseError(f"SOTOPIA {agent_id}/{dimension} requires nonempty reasoning")
                try:
                    normalized = normalize_sotopia_dimension(dimension, value)
                except ValidationError as exc:
                    raise ParseError(str(exc)) from exc
                native_by_agent[agent_id][dimension] = value
                normalized_by_agent[agent_id][dimension] = normalized
                role = (
                    "evaluated_agent"
                    if agent_id == evaluated_agent_id
                    else "partner_agent" if agent_id == partner_agent_id else "unassigned_agent"
                )
                metadata = {
                    "native_range": [low, high],
                    "reasoning": reasoning,
                    "reasoning_visibility": "evaluator_only",
                    "judge_provenance": provenance_data,
                    "agent_role": role,
                    "agent_id": agent_id,
                    "official_primary": False,
                }
                metrics.append(MetricValue(f"sotopia.agent.{agent_id}.{dimension}", value, metadata=metadata))
                metrics.append(
                    MetricValue(
                        f"sotopia.agent.{agent_id}.{dimension}.normalized",
                        normalized,
                        unit="0_to_1",
                        metadata={**metadata, "native_metric": dimension},
                    )
                )
        for dimension in SOTOPIA_DIMENSIONS:
            values = [native_by_agent[agent_id][dimension] for agent_id in agent_ids]
            normalized_values = [normalized_by_agent[agent_id][dimension] for agent_id in agent_ids]
            metrics.append(
                MetricValue(
                    f"sotopia.diagnostic.mean_across_agents.{dimension}",
                    sum(values) / len(values),
                    numerator=sum(values),
                    denominator=len(values),
                    metadata={
                        "aggregation": "mean_across_agents",
                        "official_primary": False,
                        "judge_provenance": provenance_data,
                    },
                )
            )
            metrics.append(
                MetricValue(
                    f"sotopia.diagnostic.mean_across_agents.{dimension}.normalized",
                    sum(normalized_values) / len(normalized_values),
                    unit="0_to_1",
                    numerator=sum(normalized_values),
                    denominator=len(normalized_values),
                    metadata={
                        "aggregation": "mean_across_agents",
                        "official_primary": False,
                        "judge_provenance": provenance_data,
                    },
                )
            )

        if evaluated_agent_id is None:
            metrics.extend(self.unavailable("evaluated_agent_id is required for formal aggregation", provenance))
            return tuple(metrics)

        def add_role_metrics(agent_id: str, role: str) -> None:
            is_evaluated = role == "evaluated_agent"
            for dimension, (low, high) in SOTOPIA_DIMENSIONS.items():
                native = native_by_agent[agent_id][dimension]
                normalized = normalized_by_agent[agent_id][dimension]
                metadata = {
                    "aggregation": "selected_agent_identity",
                    "agent_id": agent_id,
                    "agent_role": role,
                    "native_range": [low, high],
                    "official_primary": is_evaluated,
                    "judge_provenance": provenance_data,
                }
                metrics.append(MetricValue(f"sotopia.{role}.{dimension}", native, metadata=metadata))
                metrics.append(
                    MetricValue(
                        f"sotopia.{role}.{dimension}.normalized",
                        normalized,
                        unit="0_to_1",
                        metadata={**metadata, "official_primary": False, "native_metric": dimension},
                    )
                )
            dimension_values = list(normalized_by_agent[agent_id].values())
            metrics.append(
                MetricValue(
                    f"sotopia.{role}.normalized_dimension_mean",
                    sum(dimension_values) / len(dimension_values),
                    unit="0_to_1",
                    numerator=sum(dimension_values),
                    denominator=len(dimension_values),
                    metadata={
                        "aggregation": "diagnostic_mean_across_normalized_dimensions",
                        "agent_id": agent_id,
                        "agent_role": role,
                        "official_primary": False,
                        "judge_provenance": provenance_data,
                    },
                )
            )

        add_role_metrics(evaluated_agent_id, "evaluated_agent")
        if partner_agent_id is not None:
            add_role_metrics(partner_agent_id, "partner_agent")
        return tuple(metrics)


@adapter("sotopia")
class SotopiaAdapter(BenchmarkAdapter):
    benchmark_id = "sotopia"
    prompt_revision = "sotopia-official-agent-action-and-dimension-prompts-v6-none-fallback"
    scorer_revision = SotopiaScorer.scorer_revision

    def __init__(self, *, judge_provenance: JudgeProvenance | None = None) -> None:
        self.environment = RoleIsolatedSocialEnvironment(
            randomize_turn_order=False,
            default_max_turns=20,
            max_stale_turns=2,
        )
        self.scorer = SotopiaScorer()
        self._judge_provenance = judge_provenance

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"SotopiaAdapter cannot run {case.benchmark_id!r}")
        probe_case(case)
        position_fields = (
            "evaluated_agent_id",
            "evaluated_role_position",
            "partner_agent_id",
        )
        present = [field in case.input_data for field in position_fields]
        if any(present) and not all(present):
            raise ValidationError(
                "SOTOPIA evaluated-position cases require evaluated_agent_id, "
                "evaluated_role_position, and partner_agent_id together"
            )
        if all(present):
            agent_ids = {str(agent.get("id")) for agent in case.input_data["agents"]}
            evaluated = str(case.input_data["evaluated_agent_id"])
            partner = str(case.input_data["partner_agent_id"])
            if evaluated not in agent_ids or partner not in agent_ids or evaluated == partner:
                raise ValidationError(
                    "SOTOPIA evaluated_agent_id and partner_agent_id must name the two distinct agents"
                )
            if case.input_data["evaluated_role_position"] not in {"agent1", "agent2"}:
                raise ValidationError(
                    "SOTOPIA evaluated_role_position must be agent1 or agent2"
                )

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        self.validate_case(case)
        return self.environment.identity()

    def assistant_or_partner_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        self.validate_case(case)
        if "evaluated_agent_id" not in case.input_data:
            return {"mode": "joint_self_play", "position_designated": False}
        return {
            "mode": "target_role_vs_separately_configured_partner_required",
            "position_designated": True,
            "evaluated_agent_id": str(case.input_data["evaluated_agent_id"]),
            "evaluated_role_position": str(case.input_data["evaluated_role_position"]),
            "partner_agent_id": str(case.input_data["partner_agent_id"]),
        }

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        state = self.environment.reset(case, seed=seed)
        assert state.next_actor is not None
        return self.build_actor_request(case, state, state.next_actor, model=model, seed=seed)

    def build_actor_request(
        self,
        case: BenchmarkCase,
        state: SocialState,
        actor: str,
        *,
        model: str,
        seed: int,
    ) -> ModelRequest:
        observation = self.environment.observation(state, actor=actor)
        agent = state.agents[actor]
        other_names = [candidate.role for candidate in state.agents.values() if candidate.agent_id != actor]
        action_list = " ".join(
            "non-verbal communication" if action == "nonverbal" else action
            for action in state.allowed_actions
        )
        official_actor_prompt = f"""Imagine you are {agent.role}, your task is to act/speak as {agent.role} would, keeping in mind {agent.role}'s social goal.
You can find {agent.role}'s goal (or background) in the 'Here is the context of the interaction' field.
Note that {agent.role}'s goal is only visible to you.
You should try your best to achieve {agent.role}'s goal in a way that align with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).

Here is the context of the interaction:

Scenario: {state.public_background}
Participant: {agent.role}
Other participant(s): {', '.join(other_names)}
{agent.role}'s background: {agent.profile}
{agent.role}'s goal: {agent.goal}
Private information visible only to {agent.role}: {agent.private_information}

You are at Turn #{state.turn_count}. Your available action types are {action_list}.
Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
{{"action_type": "<one available action type>", "argument": "<your utterance or action description>"}}"""
        system_message = ChatMessage(
            role="system",
            content=official_actor_prompt,
            metadata=observation[0].metadata,
        )
        history_lines = []
        for event in state.transcript:
            if event.metadata.get("visibility") != "public_transcript":
                continue
            speaker = state.agents[event.actor].role
            if event.kind == "speak":
                rendered = f'{speaker} said: "{event.content}"'
            elif event.kind == "none":
                rendered = f"{speaker} did nothing"
            elif event.kind == "nonverbal":
                rendered = f"{speaker} [non-verbal communication] {event.content}"
            elif event.kind == "action":
                rendered = f"{speaker} [action] {event.content}"
            elif event.kind == "leave":
                rendered = f"{speaker} left the conversation"
            else:
                rendered = f"{speaker}: {event.content}"
            history_lines.append(rendered)
        history = "\n".join(history_lines) if history_lines else "No previous interaction."
        messages = (
            system_message,
            ChatMessage(
                role="user",
                content=f"Interaction history:\n{history}\n\n{_SOCIAL_ACTION_CONTRACT}",
                metadata={"visibility": "public_transcript"},
            ),
        )
        return ModelRequest(
            request_id=f"{case.case_id}:actor:{state.turn_count}:{actor}",
            messages=messages,
            model=model,
            temperature=0.0,
            seed=seed + state.turn_count,
            response_format={"type": "json_object"},
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": actor,
                "route_role": (
                    "partner_agent"
                    if case.input_data.get("partner_agent_id") == actor
                    else "evaluated_agent"
                ),
                "prompt_revision": self.prompt_revision,
            },
        )

    def parse_response(self, case: BenchmarkCase, response: ModelResponse) -> Any:
        return parse_social_action(response.text, self.environment.allowed_actions)

    def provenance_for_case(self, case: BenchmarkCase) -> JudgeProvenance | None:
        if self._judge_provenance is not None:
            return self._judge_provenance
        replay = case.metadata.get("replay")
        raw = replay.get("judge_provenance") if isinstance(replay, Mapping) else None
        if not isinstance(raw, Mapping):
            return None
        return JudgeProvenance(
            judge_models=tuple(raw.get("judge_models") or ()),
            judge_revisions=tuple(raw.get("judge_revisions") or ()),
            rubric_revision=str(raw.get("rubric_revision") or ""),
            calls_per_output=int(raw.get("calls_per_output", 1)),
            source=str(raw.get("source", "synthetic_fixture")),
        )

    def build_judge_request(
        self,
        case: BenchmarkCase,
        state: SocialState,
        provenance: JudgeProvenance,
        *,
        seed: int,
    ) -> ModelRequest:
        evaluator_view = {
            "scenario": state.public_background,
            "agents": {
                agent_id: {
                    "role": agent.role,
                    "profile": agent.profile,
                    "goal": agent.goal,
                    "private_information": agent.private_information,
                }
                for agent_id, agent in state.agents.items()
            },
            "transcript": [
                {"turn": event.turn, "actor": event.actor, "action_type": event.kind, "content": event.content}
                for event in state.transcript
                if event.metadata.get("visibility") == "public_transcript"
            ],
            "dimensions": {name: {"min": bounds[0], "max": bounds[1]} for name, bounds in SOTOPIA_DIMENSIONS.items()},
        }
        score_item_schemas = {
            dimension: {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "minLength": 1},
                    "score": {"type": "integer", "minimum": low, "maximum": high},
                },
                "required": ["reasoning", "score"],
                "additionalProperties": False,
            }
            for dimension, (low, high) in SOTOPIA_DIMENSIONS.items()
        }
        agent_schema = {
            "type": "object",
            "properties": score_item_schemas,
            "required": list(SOTOPIA_DIMENSIONS),
            "additionalProperties": False,
        }
        response_schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "properties": {agent_id: agent_schema for agent_id in state.agents},
                    "required": list(state.agents),
                    "additionalProperties": False,
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        }
        rubric_text = "\n".join(
            f"- {name} [{low}, {high}]: {SOTOPIA_RUBRICS[name]}"
            for name, (low, high) in SOTOPIA_DIMENSIONS.items()
        )
        output_contract = (
            "Return only one JSON object with exactly one top-level field `scores`. `scores` must contain exactly "
            f"these participant IDs: {list(state.agents)}. Every participant must contain exactly these seven "
            f"dimensions: {list(SOTOPIA_DIMENSIONS)}. Every dimension must be exactly "
            '{"reasoning":"non-empty evidence-based text","score":<integer in that dimension range>}. '
            "Do not add summaries, markdown, or fields outside this schema."
        )
        return ModelRequest(
            request_id=f"{case.case_id}:judge",
            messages=[
                ChatMessage(
                    "system",
                    "Based on the interaction, evaluate how well every participant performed and achieved their "
                    "goals. Judge each participant independently using the official SOTOPIA dimensions below. "
                    "Ground every score in the scenario, private evaluator context, goals, and transcript.\n"
                    + rubric_text
                    + "\n"
                    + output_contract,
                    metadata={"visibility": "evaluator_only"},
                ),
                ChatMessage("user", canonical_json(evaluator_view), metadata={"visibility": "evaluator_only"}),
            ],
            model=provenance.judge_models[0],
            temperature=0,
            seed=seed,
            response_format=json_schema_response_format("sotopia_episode_evaluation", response_schema),
            metadata={
                "benchmark_id": self.benchmark_id,
                "route_role": "judge",
                "rubric_revision": provenance.rubric_revision,
                "prompt_source": "SOTOPIA EpisodeLLMEvaluator and SotopiaDimensions",
                "output_contract": output_contract,
            },
        )

    def replay_responses(self, case: BenchmarkCase, *, seed: int) -> Mapping[str, Any]:
        replay = case.metadata.get("replay")
        if not isinstance(replay, Mapping):
            raise ConfigurationError(f"fixture {case.case_id} has no replay metadata")
        actions = replay.get("actions")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise ConfigurationError(f"fixture {case.case_id} replay.actions must be an array")
        state = self.environment.reset(case, seed=seed)
        responses: dict[str, Any] = {}
        for action in actions:
            if state.terminal or state.next_actor is None:
                break
            actor = state.next_actor
            request = self.build_actor_request(case, state, actor, model="offline-replay", seed=seed)
            text = canonical_json(action)
            responses[request.request_id or ""] = {"text": text, "finish_reason": "replayed"}
            parsed = parse_social_action(text, self.environment.allowed_actions)
            self.environment.apply(state, actor=actor, action=parsed)
        if not state.terminal:
            raise ConfigurationError(f"fixture {case.case_id} replay actions do not terminate the episode")
        if "judge" in replay:
            responses[f"{case.case_id}:judge"] = {"text": canonical_json(replay["judge"]), "finish_reason": "replayed"}
        return responses

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
        state = self.environment.reset(case, seed=seed)
        responses: list[ModelResponse] = []
        actor_output_fallbacks: list[Mapping[str, Any]] = []
        budget_termination: Mapping[str, Any] | None = None
        if previous_result is not None:
            from ..judge_resume import restore_judge_state
            restore_judge_state(state, previous_result)
        while previous_result is None and not state.terminal:
            assert state.next_actor is not None
            actor = state.next_actor
            request = self.build_actor_request(case, state, actor, model=model, seed=seed)
            try:
                action = generate_and_parse_with_contract_retries(
                    backend=backend,
                    request=request,
                    parser=lambda response: self.parse_response(case, response),
                    responses=responses,
                    contract=_SOCIAL_ACTION_CONTRACT,
                )
                self.environment.apply(state, actor=actor, action=action)
            except ParseError as exc:
                actor_output_fallbacks.append(
                    {
                        "turn": state.turn_count,
                        "actor": actor,
                        "kind": "invalid_action",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "applied_action": {"action_type": "none", "argument": ""},
                    }
                )
                self.environment.apply(state, actor=actor, action=SocialAction("none", ""))
            except (BackendStructuredOutputError, BackendTokenLimitError, BackendSafetyError) as exc:
                actor_output_fallbacks.append(
                    {
                        "turn": state.turn_count,
                        "actor": actor,
                        "kind": "actor_output_failure",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "applied_action": {"action_type": "none", "argument": ""},
                    }
                )
                self.environment.apply(state, actor=actor, action=SocialAction("none", ""))
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
                self.environment.fail_actor(state, actor=actor, failure_kind=type(exc).__name__)
                return CaseResult(
                    run_id=run_id,
                    benchmark_id=self.benchmark_id,
                    case_id=case.case_id,
                    group_id=case.group_id,
                    repetition=repetition,
                    status=ResultStatus.FAILED,
                    trace=tuple(state.transcript),
                    model_response=responses[-1] if responses else None,
                    error=ErrorState("actor_backend", type(exc).__name__, str(exc), retryable=True),
                    latency_ms=sum(response.latency_ms or 0 for response in responses),
                    token_usage=combine_usage(response.usage for response in responses),
                    metadata={
                        "terminal_reason": state.terminal_reason,
                        "episode_complete": False,
                        "actor_output_fallbacks": actor_output_fallbacks,
                    },
                )

        provenance = self.provenance_for_case(case)
        payload = previous_result.metadata.get("judge_payload") if previous_result else None
        judge_status = "unavailable"
        judge_error = None
        metrics: tuple[MetricValue, ...]
        if provenance is None:
            metrics = self.scorer.unavailable("judge configuration is required", None)
        else:
            judge_request = self.build_judge_request(case, state, provenance, seed=seed)
            try:
                def parse_judge_response(response: ModelResponse) -> Mapping[str, Any]:
                    candidate = parse_json_object(response, label="SOTOPIA judge response")
                    self.scorer.score_payload(
                        candidate,
                        agent_ids=tuple(state.agents),
                        provenance=provenance,
                        evaluated_agent_id=(
                            str(case.input_data["evaluated_agent_id"])
                            if "evaluated_agent_id" in case.input_data
                            else None
                        ),
                        partner_agent_id=(
                            str(case.input_data["partner_agent_id"])
                            if "partner_agent_id" in case.input_data
                            else None
                        ),
                    )
                    return candidate

                if payload is None:
                    payload = generate_and_parse_with_contract_retries(
                        backend=backend,
                        request=judge_request,
                        parser=parse_judge_response,
                        responses=responses,
                        contract=str(judge_request.metadata["output_contract"]),
                    )
                metrics = self.scorer.score_payload(
                    payload,
                    agent_ids=tuple(state.agents),
                    provenance=provenance,
                    evaluated_agent_id=(
                        str(case.input_data["evaluated_agent_id"])
                        if "evaluated_agent_id" in case.input_data
                        else None
                    ),
                    partner_agent_id=(
                        str(case.input_data["partner_agent_id"])
                        if "partner_agent_id" in case.input_data
                        else None
                    ),
                )
                judge_status = "available"
            except (BackendError, ParseError) as exc:
                judge_error = {"kind": type(exc).__name__, "message": str(exc)}
                metrics = self.scorer.unavailable(f"judge failure: {type(exc).__name__}", provenance)
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction={
                "terminal_reason": state.terminal_reason,
                "turn_count": state.turn_count,
                "public_transcript": [
                    {"turn": event.turn, "actor": event.actor, "action_type": event.kind, "content": event.content}
                    for event in state.transcript
                    if event.metadata.get("visibility") == "public_transcript"
                ],
            },
            metrics=metrics,
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
                "natural_termination": state.terminal_reason != "token_budget_exhausted",
                "budget_termination": budget_termination,
                "judge_status": judge_status,
                "judge_payload": payload,
                "judge_error": judge_error,
                "actor_output_fallbacks": actor_output_fallbacks,
                "judge_provenance": provenance.to_dict() if provenance else None,
                "formal_score_subject": {
                    "aggregation": "evaluated_agent_only",
                    "evaluated_agent_id": case.input_data.get("evaluated_agent_id"),
                    "evaluated_role_position": case.input_data.get("evaluated_role_position"),
                    "partner_agent_id": case.input_data.get("partner_agent_id"),
                    "position_designated": "evaluated_agent_id" in case.input_data,
                },
            },
        )

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        # Keep original per-case judge outputs and continuation contracts intact.
        # Only aggregate persona slices under explicit evaluated/partner roles.
        separated_results = []
        persona_examples: dict[str, MetricValue] = {}
        for result in results:
            by_name = {metric.name: metric for metric in result.metrics}
            subject = result.metadata.get("formal_score_subject") or {}
            if not isinstance(subject, Mapping):
                subject = {}
            separated = []
            for metric in result.metrics:
                if metric.name.startswith("sotopia.agent."):
                    tail = metric.name.removeprefix("sotopia.agent.")
                    agent_id = tail.removesuffix(".normalized").rsplit(".", 1)[0]
                    native = by_name.get(metric.name.removesuffix(".normalized"), metric)
                    role = native.metadata.get("agent_role")
                    if role not in {"evaluated_agent", "partner_agent", "unassigned_agent"}:
                        role = ("evaluated_agent" if agent_id == subject.get("evaluated_agent_id")
                                else "partner_agent" if agent_id == subject.get("partner_agent_id")
                                else "unassigned_agent")
                    metric = replace(metric, name=f"sotopia.{role}_by_id.{tail}",
                                     metadata={**metric.metadata, "agent_role": role, "agent_id": agent_id})
                    persona_examples[metric.name] = metric
                separated.append(metric)
            separated_results.append(replace(result, metrics=tuple(separated)))
        metrics = dict(aggregate_named_metrics(separated_results, namespace="sotopia"))
        for name, example in persona_examples.items():
            base = metrics[name]
            metrics[name] = replace(base, unit=example.unit, direction=example.direction, metadata={
                **base.metadata, "agent_role": example.metadata["agent_role"],
                "agent_id": example.metadata["agent_id"], "official_primary": False,
                "case_count": (base.denominator or 0) + base.metadata["unavailable_count"],
                "available_case_count": base.denominator or 0,
                "independent_statistical_unit": "case",
            })
        configuration_groups: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            configuration_groups[result.group_id].append(result)
        metrics["sotopia.configuration_count"] = MetricValue(
            "sotopia.configuration_count",
            len(configuration_groups),
            direction="descriptive",
            unit="configurations",
            numerator=len(configuration_groups),
            denominator=len(configuration_groups),
            metadata={"independent_statistical_unit": "configuration"},
        )
        def opposite_roles(group: Sequence[CaseResult]) -> bool:
            if len(group) != 2 or group[0].case_id == group[1].case_id or group[0].repetition != group[1].repetition:
                return False
            subjects = [result.metadata.get("formal_score_subject") or {} for result in group]
            if not all(isinstance(subject, Mapping) for subject in subjects):
                return False
            first, second = subjects
            evaluated, partner = first.get("evaluated_agent_id"), first.get("partner_agent_id")
            if not evaluated or not partner or evaluated == partner:
                return False
            if (evaluated, partner) != (second.get("partner_agent_id"), second.get("evaluated_agent_id")):
                return False
            positions = [subject.get("evaluated_role_position") for subject in subjects]
            # Older completed records stored both identities but not positions.
            return positions in ([None, None], ["agent1", "agent2"], ["agent2", "agent1"])

        def paired_values(group: Sequence[CaseResult], name: str) -> list[float]:
            if not opposite_roles(group):
                return []
            values = []
            for result in group:
                matches = [metric for metric in result.metrics if metric.name == name]
                if result.status != ResultStatus.COMPLETED or len(matches) != 1:
                    return []
                value = matches[0].value
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    return []
                values.append(float(value))
            return values

        complete_pairs = sum(
            all(paired_values(group, f"sotopia.evaluated_agent.{dimension}") for dimension in SOTOPIA_DIMENSIONS)
            for group in configuration_groups.values()
        )
        metrics["sotopia.complete_role_pair_rate"] = MetricValue(
            "sotopia.complete_role_pair_rate",
            complete_pairs / len(configuration_groups) if configuration_groups else None,
            unit="proportion",
            numerator=complete_pairs,
            denominator=len(configuration_groups),
            metadata={
                "expected_role_positions_per_configuration": 2,
                "independent_statistical_unit": "configuration",
                "requires": "opposite_role_identities_and_all_seven_evaluated_dimensions_available",
            },
        )

        evaluated_metric_names = sorted(
            name for name in metrics if name.startswith("sotopia.evaluated_agent.")
        )
        for metric_name in evaluated_metric_names:
            configuration_values: list[float] = []
            complete_value_pairs = 0
            for group_results in configuration_groups.values():
                values = paired_values(group_results, metric_name)
                if values:
                    configuration_values.append(statistics.mean(values))
                    complete_value_pairs += 1
            suffix = metric_name.removeprefix("sotopia.evaluated_agent.")
            output_name = f"sotopia.configuration_mean.{suffix}"
            base = metrics[metric_name]
            metrics[output_name] = MetricValue(
                name=output_name,
                value=(
                    statistics.mean(configuration_values)
                    if configuration_values
                    else None
                ),
                direction=base.direction,
                unit=base.unit,
                numerator=sum(configuration_values) if configuration_values else None,
                denominator=len(configuration_values),
                metadata={
                    "aggregation": "mean_complete_role_pairs_then_mean_available_configurations",
                    "missing_role_policy": "exclude_incomplete_pair_without_zero_imputation",
                    "availability": "available" if configuration_values else "unavailable_no_complete_role_pairs",
                    "independent_statistical_unit": "configuration",
                    "configuration_count": len(configuration_groups),
                    "available_configuration_count": len(configuration_values),
                    "complete_value_pair_count": complete_value_pairs,
                    "incomplete_or_unavailable_configuration_count": (
                        len(configuration_groups) - complete_value_pairs
                    ),
                },
            )
        judge_available = sum(
            result.metadata.get("judge_status") == "available"
            for result in results
        )
        judge_unavailable = len(results) - judge_available
        judge_failures = sum(bool(result.metadata.get("judge_error")) for result in results)
        metrics["sotopia.judge_score_availability_rate"] = MetricValue(
            "sotopia.judge_score_availability_rate",
            judge_available / len(results) if results else None,
            unit="proportion",
            numerator=judge_available,
            denominator=len(results),
            metadata={
                "aggregation": "available_judge_scores_over_all_selected_case_results",
                "unavailable_count": judge_unavailable,
                "episode_scores_average_available_cases_only": True,
                "configuration_scores_require_complete_role_pairs": True,
            },
        )
        metrics["sotopia.judge_score_unavailable_count"] = MetricValue(
            "sotopia.judge_score_unavailable_count",
            judge_unavailable,
            direction="lower_is_better",
            unit="cases",
            numerator=judge_unavailable,
            denominator=len(results),
        )
        metrics["sotopia.judge_failure_count"] = MetricValue(
            "sotopia.judge_failure_count",
            judge_failures,
            direction="lower_is_better",
            unit="cases",
            numerator=judge_failures,
            denominator=len(results),
            metadata={
                "aggregation": "case_results_with_nonempty_metadata_judge_error",
                "differs_from_unavailable_count": True,
            },
        )
        return metrics


__all__ = ["SOTOPIA_DIMENSIONS", "SotopiaAdapter", "SotopiaScorer", "normalize_sotopia_dimension"]
