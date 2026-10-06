"""AgentSense adapter for active social goals and passive private-information MCQs."""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
from ..environments.social import JudgeProvenance, RoleIsolatedSocialEnvironment, SocialAction, SocialState
from ..errors import (
    BackendError,
    ConfigurationError,
    EpisodeTokenBudgetExhausted,
    ParseError,
    ValidationError,
)
from ..interfaces import BenchmarkAdapter, ModelBackend
from ..generation_policy import AGENTSENSE_JUDGE_TEMPERATURE
from ..json_utils import canonical_json
from ..registry import adapter
from .common import (
    aggregate_named_metrics,
    combine_usage,
    generate_and_parse_with_contract_retries,
    json_schema_response_format,
    parse_json_object,
)


_CHOICE = re.compile(r"^\s*\(?([A-Z])\)?[.)]?\s*$")
_INITIAL_MESSAGE = "Hi, there!"
_JUDGE_SYSTEM_PROMPT = (
    "You are skilled at evaluating whether a person has achieved his or her social goal during social interactions."
)


def parse_information_choice(
    text: str,
    option_count: int,
    options: Sequence[str] | None = None,
) -> int:
    match = _CHOICE.fullmatch(text)
    if match:
        index = ord(match.group(1)) - ord("A")
    else:
        parenthesized = re.findall(r"\(([A-Z])\)", text)
        if parenthesized:
            index = ord(parenthesized[0]) - ord("A")
        elif options is not None:
            prediction = text.strip()
            prefixed = re.match(r"^[A-Z][.)]\s*(.+)$", prediction, flags=re.DOTALL)
            if prefixed:
                prediction = prefixed.group(1).strip()
            matches = [
                option_index
                for option_index, option in enumerate(options)
                if prediction == option
                or (option in prediction and not any(other in prediction for other in options if other != option))
                or (prediction and prediction in option and not any(prediction in other for other in options if other != option))
            ]
            if len(matches) != 1:
                raise ParseError("AgentSense information answer does not identify one unambiguous option")
            index = matches[0]
        else:
            raise ParseError("AgentSense information answer must contain exactly one option label")
    if index < 0 or index >= option_count:
        raise ParseError("AgentSense information option is out of range")
    return index


def _yes_no(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value.strip().casefold() in {"yes", "no"}:
        return int(value.strip().casefold() == "yes")
    raise ParseError(f"goal judgment must be exact Yes/No or boolean, got {value!r}")


def _parse_goal_answer(response: ModelResponse) -> Mapping[str, str]:
    # Match AgentSense's GoalMetric exactly: any case-sensitive ``Yes`` in the
    # free-form interview answer is positive; every other answer is negative.
    text = response.text.strip()
    return {"reasoning": text, "answer": "Yes" if "Yes" in text else "No"}


class AgentSenseScorer:
    scorer_revision = "agentsense-official-goal-info-psi-sample-std-per-judge-v2"

    def unavailable_goal_metrics(
        self,
        reason: str,
        provenance: JudgeProvenance | None,
    ) -> tuple[MetricValue, ...]:
        provenance_data = provenance.to_dict() if provenance else None
        names = (
            "agentsense.episode.self_goal_completion",
            "agentsense.episode.other_goal_completion",
            "agentsense.episode.judge_average",
            "agentsense.episode.judge_majority",
        )
        return tuple(
            MetricValue(
                name=name,
                value=None,
                metadata={"availability": "unavailable", "reason": reason, "judge_provenance": provenance_data},
            )
            for name in names
        )

    def score_goals(
        self,
        payload: Mapping[str, Any],
        *,
        agent_ids: Sequence[str],
        provenance: JudgeProvenance,
    ) -> tuple[MetricValue, ...]:
        raw = payload.get("goal_evaluations")
        if not isinstance(raw, Mapping) or set(raw) != set(agent_ids):
            raise ParseError("AgentSense judge response requires goal_evaluations for exactly every agent")
        expected_judges = provenance.logical_judge_ids
        per_agent: dict[str, dict[str, float]] = {}
        metrics: list[MetricValue] = []
        provenance_data = provenance.to_dict()
        for agent_id in agent_ids:
            evaluations = raw[agent_id]
            if not isinstance(evaluations, Sequence) or isinstance(evaluations, (str, bytes)) or not evaluations:
                raise ParseError(f"AgentSense {agent_id} requires a nonempty goal-evaluation array")
            self_values: list[int] = []
            other_values: list[int] = []
            judge_values: dict[str, list[int]] = {judge: [] for judge in expected_judges}
            majority_values: list[int] = []
            for evaluation in evaluations:
                if not isinstance(evaluation, Mapping):
                    raise ParseError("AgentSense goal evaluation entries must be objects")
                self_values.append(_yes_no(evaluation.get("self")))
                others = evaluation.get("others")
                if not isinstance(others, Sequence) or isinstance(others, (str, bytes)) or not others:
                    raise ParseError("AgentSense goal evaluation requires a nonempty others array")
                other_values.extend(_yes_no(item) for item in others)
                judgments = evaluation.get("judges")
                if not isinstance(judgments, Mapping) or set(judgments) != set(expected_judges):
                    raise ParseError(
                        f"AgentSense judge keys differ; expected={sorted(expected_judges)}, "
                        f"got={sorted(judgments) if isinstance(judgments, Mapping) else 'non-object'}"
                    )
                current = []
                for judge in expected_judges:
                    value = _yes_no(judgments[judge])
                    judge_values[judge].append(value)
                    current.append(value)
                majority_values.append(int(sum(current) > len(current) / 2))
            values = {
                "self_goal_completion": sum(self_values) / len(self_values),
                "other_goal_completion": sum(other_values) / len(other_values),
                "judge_average": sum(sum(items) for items in judge_values.values()) / sum(len(items) for items in judge_values.values()),
                "judge_majority": sum(majority_values) / len(majority_values),
            }
            for judge, items in judge_values.items():
                values[f"judge.{judge}"] = sum(items) / len(items)
            per_agent[agent_id] = values
            for name, value in values.items():
                metrics.append(
                    MetricValue(
                        f"agentsense.agent.{agent_id}.{name}",
                        value,
                        unit="proportion",
                        metadata={"judge_provenance": provenance_data, "goal_count": len(evaluations)},
                    )
                )
        episode_names = sorted(set.intersection(*(set(values) for values in per_agent.values())))
        for name in episode_names:
            values = [per_agent[agent_id][name] for agent_id in agent_ids]
            metrics.append(
                MetricValue(
                    f"agentsense.episode.{name}",
                    sum(values) / len(values),
                    unit="proportion",
                    numerator=sum(values),
                    denominator=len(values),
                    metadata={"aggregation": "mean_across_agents", "judge_provenance": provenance_data},
                )
            )
        return tuple(metrics)


@adapter("agentsense")
class AgentSenseAdapter(BenchmarkAdapter):
    benchmark_id = "agentsense"
    prompt_revision = "agentsense-official-conversation-and-interviews-v6-inline-speakers"
    scorer_revision = AgentSenseScorer.scorer_revision

    def __init__(
        self,
        *,
        judge_provenance: JudgeProvenance | None = None,
        judge_role_by_id: Mapping[str, str] | None = None,
        judge_max_workers: int = 1,
    ) -> None:
        self.environment = RoleIsolatedSocialEnvironment(
            randomize_turn_order=True,
            speaker_selection_method="random_no_repeat",
            default_max_turns=15,
            allowed_actions=("speak",),
            min_agent_count=2,
            max_agent_count=5,
        )
        self.scorer = AgentSenseScorer()
        self._judge_provenance = judge_provenance
        self._judge_role_by_id = dict(judge_role_by_id or {})
        if (
            isinstance(judge_max_workers, bool)
            or not isinstance(judge_max_workers, int)
            or not 1 <= judge_max_workers <= 3
        ):
            raise ConfigurationError("AgentSense judge_max_workers must be an integer in [1,3]")
        self.judge_max_workers = judge_max_workers
        if self._judge_role_by_id and judge_provenance is None:
            raise ConfigurationError("AgentSense judge routing requires judge provenance")
        if judge_provenance is not None and self._judge_role_by_id and (
            set(self._judge_role_by_id) != set(judge_provenance.logical_judge_ids)
        ):
            raise ConfigurationError("AgentSense judge routing must cover every logical judge ID exactly")

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"AgentSenseAdapter cannot run {case.benchmark_id!r}")
        probe_case(case)

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        self.validate_case(case)
        return {
            **self.environment.identity(),
            "initial_message": _INITIAL_MESSAGE,
            "initial_message_speaker": "seeded_random_participant",
            "actor_output": "one_free_form_paragraph",
        }

    @staticmethod
    def _scenario_parts(state: SocialState) -> tuple[str, str]:
        background, separator, description = state.public_background.partition("\n\n")
        if not separator:
            return state.public_background, ""
        return background.strip(), description.strip()

    @staticmethod
    def _goal_text(goal: str) -> str:
        # Official initialization joins the per-character goal list with spaces.
        # Normalized records preserve the same list as one bullet-delimited string.
        parts = [line.strip().removeprefix("- ").strip() for line in goal.splitlines() if line.strip()]
        return " ".join(parts) if parts else goal.strip()

    def _actor_system_prompt(self, state: SocialState, actor: str) -> str:
        if actor not in state.agents:
            raise ValidationError(f"unknown AgentSense actor {actor!r}")
        agent = state.agents[actor]
        background, description = self._scenario_parts(state)
        private_information = agent.private_information if agent.private_information else "N/A"
        return (
            f"You are {actor}. {agent.profile}\n\n"
            f"Now you are in a social scene. {background}\n{description}\n\n"
            f"In this scene, as {actor}, you have some social goals and secrets that you don't want anyone else to know.\n\n"
            f"SOCIAL GOALS: {self._goal_text(agent.goal)}\n\n"
            f"SECRETS: {private_information}.\n"
            "Please advance your social goals while hiding your secrets as much as possible.\n\n"
            "Based on the interactions below, what would you like to say next? Please answer in a paragraph."
        )

    def _actor_chat_messages(self, state: SocialState, actor: str) -> tuple[ChatMessage, ...]:
        messages = [
            ChatMessage(
                "system",
                self._actor_system_prompt(state, actor),
                metadata={"visibility": f"actor:{actor}"},
            )
        ]
        for event in state.transcript:
            if event.metadata.get("visibility") != "public_transcript":
                continue
            messages.append(
                ChatMessage(
                    "assistant" if event.actor == actor else "user",
                    f"{event.actor}: {event.content}",
                    metadata={"visibility": "public_transcript", "turn": event.turn},
                )
            )
        return tuple(messages)

    def reset_official_state(self, case: BenchmarkCase, *, seed: int) -> SocialState:
        """Seed the official random participant's ``Hi, there!`` opening turn."""

        state = self.environment.reset(case, seed=seed)
        assert state.next_actor is not None
        self.environment.apply(
            state,
            actor=state.next_actor,
            action=SocialAction("speak", _INITIAL_MESSAGE),
        )
        return state

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        state = self.reset_official_state(case, seed=seed)
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
        return ModelRequest(
            request_id=f"{case.case_id}:actor:{state.turn_count}:{actor}",
            messages=self._actor_chat_messages(state, actor),
            model=model,
            temperature=0.0,
            max_tokens=1024,
            seed=seed + state.turn_count,
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": actor,
                "route_role": "evaluated_actor",
                "prompt_revision": self.prompt_revision,
                "prompt_source": "AgentSense configs/prompt_template.json",
                "output_contract": "one free-form paragraph",
            },
        )

    def parse_response(self, case: BenchmarkCase, response: ModelResponse) -> Any:
        del case
        content = response.text.strip()
        if not content:
            raise ParseError("AgentSense actor response must be a non-empty paragraph")
        return SocialAction("speak", content)

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
            judge_ids=tuple(raw.get("judge_ids") or ()),
        )

    def build_judge_request(
        self,
        case: BenchmarkCase,
        state: SocialState,
        provenance: JudgeProvenance,
        *,
        judge_model: str,
        judge_id: str | None = None,
        judge_index: int,
        seed: int,
    ) -> ModelRequest:
        logical_judge_id = str(judge_id or judge_model)
        evaluation_schema = {
            "type": "object",
            "properties": {
                "goal_id": {"type": "string"},
                "self": {"type": "string", "enum": ["Yes", "No"]},
                "others": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "enum": ["Yes", "No"]},
                },
                "judges": {
                    "type": "object",
                    "properties": {
                        current: {"type": "string", "enum": ["Yes", "No"]}
                        for current in provenance.logical_judge_ids
                    },
                    "required": list(provenance.logical_judge_ids),
                    "additionalProperties": False,
                },
            },
            "required": ["self", "others", "judges"],
            "additionalProperties": False,
        }
        response_schema = {
            "type": "object",
            "properties": {
                "goal_evaluations": {
                    "type": "object",
                    "properties": {
                        agent_id: {"type": "array", "minItems": 1, "items": evaluation_schema}
                        for agent_id in state.agents
                    },
                    "required": list(state.agents),
                    "additionalProperties": False,
                }
            },
            "required": ["goal_evaluations"],
            "additionalProperties": False,
        }
        evaluator_view = {
            "scenario": state.public_background,
            "agents": {
                agent_id: {
                    "profile": agent.profile,
                    "goal": agent.goal,
                    "private_information": agent.private_information,
                }
                for agent_id, agent in state.agents.items()
            },
            "transcript": [
                {"actor": event.actor, "content": event.content}
                for event in state.transcript
                if event.metadata.get("visibility") == "public_transcript"
            ],
            "current_judge": logical_judge_id,
            "required_judges": list(provenance.logical_judge_ids),
        }
        output_contract = (
            "Return only one JSON object with exactly `goal_evaluations`. It must contain every agent ID exactly. "
            "Each goal item must contain `self` (Yes/No), a non-empty `others` array of Yes/No answers, and "
            f"`judges` with exactly these keys: {list(provenance.logical_judge_ids)}. `goal_id` is optional. "
            "Do not add prose outside the JSON object."
        )
        return ModelRequest(
            request_id=f"{case.case_id}:judge:{judge_index}",
            messages=[
                ChatMessage(
                    "system",
                    "You are skilled at evaluating whether a person has achieved their social goal during social "
                    f"interactions. Evaluate every explicit goal from the supplied evidence. {output_contract}",
                    metadata={"visibility": "evaluator_only"},
                ),
                ChatMessage("user", canonical_json(evaluator_view), metadata={"visibility": "evaluator_only"}),
            ],
            model=judge_model,
            temperature=AGENTSENSE_JUDGE_TEMPERATURE,
            seed=seed + 2000 + judge_index,
            response_format=json_schema_response_format("agentsense_legacy_goal_matrix", response_schema),
            metadata={
                "benchmark_id": self.benchmark_id,
                "route_role": self._judge_role_by_id.get(logical_judge_id, "judge"),
                "judge_id": logical_judge_id,
                "rubric_revision": provenance.rubric_revision,
                "prompt_source": "AgentSense judge_prompt_template; legacy batched fixture compatibility",
                "output_contract": output_contract,
            },
        )

    def build_goal_evaluation_request(
        self,
        case: BenchmarkCase,
        state: SocialState,
        *,
        target_agent_id: str,
        goal_index: int,
        dimension: str,
        question_index: int,
        evaluator_id: str,
        question: str,
        model: str,
        seed: int,
        route_role: str,
    ) -> ModelRequest:
        """Build one official-topology AgentSense interview question."""

        if target_agent_id not in state.agents:
            raise ValidationError(f"AgentSense goal question targets unknown agent {target_agent_id!r}")
        if dimension in {"self", "others"}:
            if evaluator_id not in state.agents:
                raise ValidationError(f"AgentSense goal question names unknown evaluator {evaluator_id!r}")
            messages = list(self._actor_chat_messages(state, evaluator_id))
            visibility = f"actor:{evaluator_id}"
            prompt_source = "AgentSense participant generate_reply(chat_history + question)"
        elif dimension == "judge":
            messages = [
                ChatMessage(
                    "system",
                    _JUDGE_SYSTEM_PROMPT,
                    metadata={"visibility": "evaluator_only"},
                )
            ]
            for event in state.transcript:
                if event.metadata.get("visibility") != "public_transcript":
                    continue
                messages.append(
                    ChatMessage(
                        "user",
                        str(event.content),
                        name=event.actor,
                        metadata={"visibility": "public_transcript", "turn": event.turn},
                    )
                )
            visibility = "evaluator_only"
            prompt_source = "AgentSense configs/prompt_template.json judge_prompt_template"
        else:
            raise ValidationError(f"unsupported AgentSense goal-evaluation dimension {dimension!r}")
        messages.append(ChatMessage("user", question, metadata={"visibility": visibility}))
        return ModelRequest(
            request_id=(
                f"{case.case_id}:goal:{target_agent_id}:{goal_index}:{dimension}:"
                f"{question_index}:{evaluator_id}"
            ),
            messages=tuple(messages),
            model=model,
            temperature=AGENTSENSE_JUDGE_TEMPERATURE if dimension == "judge" else 0.0,
            max_tokens=1024 if route_role == "evaluated_actor" else 4096,
            seed=seed,
            metadata={
                "benchmark_id": self.benchmark_id,
                "route_role": route_role,
                "target_agent_id": target_agent_id,
                "goal_index": goal_index,
                "evaluation_dimension": dimension,
                "evaluator_id": evaluator_id,
                "prompt_source": prompt_source,
                "output_contract": "official free-form short explanation ending in Yes or No",
                "preserve_request_sampling": dimension in {"self", "others"},
            },
        )

    def _parse_legacy_goal_matrix(
        self,
        response: ModelResponse,
        *,
        state: SocialState,
        provenance: JudgeProvenance,
    ) -> Mapping[str, Any]:
        candidate = parse_json_object(response, label="AgentSense legacy goal matrix")
        self.scorer.score_goals(candidate, agent_ids=tuple(state.agents), provenance=provenance)
        return candidate

    def _evaluate_official_goal_questions(
        self,
        case: BenchmarkCase,
        state: SocialState,
        provenance: JudgeProvenance,
        *,
        backend: ModelBackend,
        evaluated_model: str,
        seed: int,
        responses: list[ModelResponse],
        cache: dict[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, int]]:
        raw_questions = case.metadata.get("goal_evaluation_questions")
        if not isinstance(raw_questions, Mapping) or set(raw_questions) != set(state.agents):
            raise ValidationError("AgentSense official goal questions must cover every agent exactly")
        result: dict[str, list[dict[str, Any]]] = {}
        call_counts = {"self": 0, "others": 0, "judge": 0}
        for target_agent_id in state.agents:
            goals = raw_questions[target_agent_id]
            if not isinstance(goals, Sequence) or isinstance(goals, (str, bytes)) or not goals:
                raise ValidationError(f"AgentSense {target_agent_id} requires non-empty official goal questions")
            target_results: list[dict[str, Any]] = []
            for goal_index, raw_goal in enumerate(goals):
                if not isinstance(raw_goal, Mapping) or set(raw_goal) != {"self", "others", "judge"}:
                    raise ValidationError("AgentSense goal questions require exactly self, others, and judge")
                self_specs = raw_goal["self"]
                other_specs = raw_goal["others"]
                judge_specs = raw_goal["judge"]
                if (
                    not isinstance(self_specs, Sequence)
                    or isinstance(self_specs, (str, bytes))
                    or len(self_specs) != 1
                    or not isinstance(other_specs, Sequence)
                    or isinstance(other_specs, (str, bytes))
                    or not other_specs
                    or not isinstance(judge_specs, Sequence)
                    or isinstance(judge_specs, (str, bytes))
                    or len(judge_specs) != 1
                ):
                    raise ValidationError(
                        "AgentSense official topology requires one self question, one-or-more other questions, "
                        "and one judge question per goal"
                    )

                reasoning: dict[str, Any] = {"self": None, "others": [], "judges": {}}

                def generate_answer(
                    spec: Any,
                    *,
                    dimension: str,
                    question_index: int,
                    evaluator_id: str,
                    request_model: str,
                    route_role: str,
                    seed_offset: int,
                ) -> tuple[ModelResponse, Mapping[str, str]]:
                    if not isinstance(spec, Mapping):
                        raise ValidationError("AgentSense goal question entry must be an object")
                    question = spec.get("question")
                    if not isinstance(question, str) or not question.strip():
                        raise ValidationError("AgentSense goal question text must be non-empty")
                    request = self.build_goal_evaluation_request(
                        case,
                        state,
                        target_agent_id=target_agent_id,
                        goal_index=goal_index,
                        dimension=dimension,
                        question_index=question_index,
                        evaluator_id=evaluator_id,
                        question=question.strip(),
                        model=request_model,
                        seed=seed + 3000 + seed_offset,
                        route_role=route_role,
                    )
                    cached = cache.get(request.request_id)
                    if cached is not None:
                        return ModelResponse(text=canonical_json(cached)), cached
                    response = backend.generate(request)
                    parsed = _parse_goal_answer(response)
                    cache[request.request_id] = parsed
                    return response, parsed

                def ask(
                    spec: Any,
                    *,
                    dimension: str,
                    question_index: int,
                    evaluator_id: str,
                    request_model: str,
                    route_role: str,
                    seed_offset: int,
                ) -> str:
                    response, parsed = generate_answer(
                        spec,
                        dimension=dimension,
                        question_index=question_index,
                        evaluator_id=evaluator_id,
                        request_model=request_model,
                        route_role=route_role,
                        seed_offset=seed_offset,
                    )
                    responses.append(response)
                    call_counts[dimension] += 1
                    if dimension == "self":
                        reasoning["self"] = parsed["reasoning"]
                    elif dimension == "others":
                        reasoning["others"].append(parsed["reasoning"])
                    else:
                        reasoning["judges"][evaluator_id] = parsed["reasoning"]
                    return parsed["answer"]

                self_spec = self_specs[0]
                if not isinstance(self_spec, Mapping):
                    raise ValidationError("AgentSense self question must be an object")
                self_evaluator = str(self_spec.get("obj") or target_agent_id)
                self_answer = ask(
                    self_spec,
                    dimension="self",
                    question_index=0,
                    evaluator_id=self_evaluator,
                    request_model=evaluated_model,
                    route_role="evaluated_actor",
                    seed_offset=goal_index * 100,
                )
                other_answers = []
                for question_index, spec in enumerate(other_specs):
                    if not isinstance(spec, Mapping):
                        raise ValidationError("AgentSense other question must be an object")
                    other_evaluator = str(spec.get("obj") or "")
                    other_answers.append(
                        ask(
                            spec,
                            dimension="others",
                            question_index=question_index,
                            evaluator_id=other_evaluator,
                            request_model=evaluated_model,
                            route_role="evaluated_actor",
                            seed_offset=goal_index * 100 + 10 + question_index,
                        )
                    )
                judge_answers = {}
                judge_spec = judge_specs[0]
                judge_jobs = tuple(enumerate(
                    zip(provenance.logical_judge_ids, provenance.judge_models)
                ))

                def generate_judge_answer(
                    job: tuple[int, tuple[str, str]],
                ) -> tuple[str, ModelResponse, Mapping[str, str]]:
                    judge_index, (judge_id, judge_model) = job
                    response, parsed = generate_answer(
                        judge_spec,
                        dimension="judge",
                        question_index=0,
                        evaluator_id=judge_id,
                        request_model=judge_model,
                        route_role=self._judge_role_by_id.get(judge_id, "judge"),
                        seed_offset=goal_index * 100 + 50 + judge_index,
                    )
                    return judge_id, response, parsed

                if self.judge_max_workers == 1:
                    judge_results = tuple(generate_judge_answer(job) for job in judge_jobs)
                else:
                    with ThreadPoolExecutor(
                        max_workers=min(self.judge_max_workers, len(judge_jobs)),
                        thread_name_prefix="sim-eval-agentsense-judge",
                    ) as pool:
                        judge_results = tuple(pool.map(generate_judge_answer, judge_jobs))
                for judge_id, response, parsed in judge_results:
                    responses.append(response)
                    call_counts["judge"] += 1
                    reasoning["judges"][judge_id] = parsed["reasoning"]
                    judge_answers[judge_id] = parsed["answer"]
                target_results.append(
                    {
                        "goal_id": f"{target_agent_id}:goal:{goal_index}",
                        "self": self_answer,
                        "others": other_answers,
                        "judges": judge_answers,
                        "reasoning": reasoning,
                    }
                )
            result[target_agent_id] = target_results
        return {"goal_evaluations": result}, call_counts

    def build_information_request(
        self,
        case: BenchmarkCase,
        state: SocialState,
        question: Mapping[str, Any],
        *,
        index: int,
        model: str,
        seed: int,
    ) -> ModelRequest:
        actor = str(question.get("agent_id"))
        if actor not in state.agents:
            raise ValidationError(f"information question references unknown agent {actor!r}")
        options = question.get("options")
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)) or len(options) < 2:
            raise ValidationError("information question requires at least two options")
        question_text = question.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            raise ValidationError("information question text must be non-empty")
        labels = [chr(ord("A") + option_index) for option_index in range(len(options))]
        rendered_options = "; ".join(
            f"({label}) {option}" for label, option in zip(labels, options)
        )
        official_question = (
            "Please answer the question and only output your choice.\n"
            f"{question_text.strip()} Options: {rendered_options}."
        )
        messages = list(self._actor_chat_messages(state, actor))
        messages.append(
            ChatMessage("user", official_question, metadata={"visibility": f"actor:{actor}"})
        )
        return ModelRequest(
            request_id=f"{case.case_id}:info:{index}:{actor}",
            messages=tuple(messages),
            model=model,
            temperature=0,
            max_tokens=1024,
            seed=seed + 1000 + index,
            metadata={
                "benchmark_id": self.benchmark_id,
                "task": "private_information",
                "actor": actor,
                "prompt_source": "AgentSense question_with_options + interview_agent(chat_history)",
            },
        )

    def replay_responses(self, case: BenchmarkCase, *, seed: int) -> Mapping[str, Any]:
        replay = case.metadata.get("replay")
        if not isinstance(replay, Mapping):
            raise ConfigurationError(f"fixture {case.case_id} has no replay metadata")
        actions = replay.get("actions")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise ConfigurationError(f"fixture {case.case_id} replay.actions must be an array")
        state = self.reset_official_state(case, seed=seed)
        responses: dict[str, Any] = {}
        for action in actions:
            if state.terminal or state.next_actor is None:
                break
            actor = state.next_actor
            request = self.build_actor_request(case, state, actor, model="offline-replay", seed=seed)
            if not isinstance(action, Mapping):
                raise ConfigurationError("fixture AgentSense replay actions must be objects")
            content = action.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ConfigurationError("fixture AgentSense replay actions require non-empty content")
            text = content.strip()
            responses[request.request_id or ""] = {"text": text, "finish_reason": "replayed"}
            self.environment.apply(
                state,
                actor=actor,
                action=SocialAction("speak", text),
            )
        if not state.terminal:
            raise ConfigurationError(f"fixture {case.case_id} replay actions do not terminate the episode")
        if "judge" in replay:
            provenance = self.provenance_for_case(case)
            if provenance is None:
                raise ConfigurationError("fixture judge payload requires judge provenance")
            for judge_index, _judge_model in enumerate(provenance.judge_models):
                responses[f"{case.case_id}:judge:{judge_index}"] = {
                    "text": canonical_json(replay["judge"]),
                    "finish_reason": "replayed",
                }
        info_responses = replay.get("information_responses") or []
        questions = case.input_data.get("information_questions") or []
        if len(info_responses) != len(questions):
            raise ConfigurationError("fixture information response count does not match questions")
        for index, (question, response) in enumerate(zip(questions, info_responses)):
            actor = str(question.get("agent_id"))
            responses[f"{case.case_id}:info:{index}:{actor}"] = {"text": str(response), "finish_reason": "replayed"}
        return responses

    def _failed_result(
        self,
        case: BenchmarkCase,
        state: SocialState,
        *,
        run_id: str,
        repetition: int,
        stage: str,
        exc: Exception,
        responses: Sequence[ModelResponse],
    ) -> CaseResult:
        if isinstance(exc, ParseError):
            provenance = self.provenance_for_case(case)
            if provenance is not None:
                zero_payload = {
                    "goal_evaluations": {
                        agent_id: [
                            {
                                "goal_id": "target_output_failure",
                                "self": "No",
                                "others": ["No"],
                                "judges": {
                                    judge_id: "No" for judge_id in provenance.logical_judge_ids
                                },
                            }
                        ]
                        for agent_id in state.agents
                    }
                }
                goal_metrics = self.scorer.score_goals(
                    zero_payload,
                    agent_ids=tuple(state.agents),
                    provenance=provenance,
                )
            else:
                goal_metrics = tuple(
                    MetricValue(
                        f"agentsense.episode.{name}",
                        0,
                        unit="proportion",
                        numerator=0,
                        denominator=1,
                        metadata={"scoring": "target_output_parse_failure_is_incorrect"},
                    )
                    for name in (
                        "self_goal_completion",
                        "other_goal_completion",
                        "judge_average",
                        "judge_majority",
                    )
                )
            info_metrics: list[MetricValue] = []
            for index, question in enumerate(case.input_data.get("information_questions") or ()):
                question_id = str(question.get("id") or index) if isinstance(question, Mapping) else str(index)
                info_metrics.append(
                    MetricValue(
                        f"agentsense.information.{question_id}.accuracy",
                        0,
                        numerator=0,
                        denominator=1,
                        metadata={"parse_status": "not_attempted_after_target_output_failure"},
                    )
                )
            info_metrics.append(
                MetricValue(
                    "agentsense.episode.private_information_accuracy",
                    0 if info_metrics else None,
                    numerator=0 if info_metrics else None,
                    denominator=len(info_metrics),
                    metadata={
                        "scoring": "target_output_parse_failure_is_incorrect"
                        if info_metrics
                        else "not_applicable_no_private_information_questions"
                    },
                )
            )
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
                prediction={"terminal_reason": state.terminal_reason, "turn_count": state.turn_count, "parsed": False},
                metrics=tuple(goal_metrics) + tuple(info_metrics),
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
                    "target_output_failure": failure,
                    "judge_status": "not_called_target_output_failure",
                    "judge_provenance": provenance.to_dict() if provenance else None,
                    "judge_call_count": 0,
                    "profile_id": str(case.metadata.get("profile_id") or case.metadata.get("source_id") or case.case_id),
                },
            )
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.FAILED,
            trace=tuple(state.transcript),
            error=ErrorState(stage, type(exc).__name__, str(exc), retryable=isinstance(exc, BackendError)),
            model_response=responses[-1] if responses else None,
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={"terminal_reason": state.terminal_reason, "episode_complete": False},
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
        state = self.reset_official_state(case, seed=seed)
        responses: list[ModelResponse] = []
        budget_termination: Mapping[str, Any] | None = None
        if previous_result is not None:
            from ..judge_resume import restore_judge_state
            restore_judge_state(state, previous_result)
        while previous_result is None and not state.terminal:
            assert state.next_actor is not None
            actor = state.next_actor
            try:
                request = self.build_actor_request(case, state, actor, model=model, seed=seed)
                response = backend.generate(request)
                responses.append(response)
                action = self.parse_response(case, response)
                self.environment.apply(state, actor=actor, action=action)
            except EpisodeTokenBudgetExhausted as exc:
                budget_termination = exc.details()
                self.environment.terminate_capability_limit(
                    state,
                    actor=actor,
                    reason="token_budget_exhausted",
                    details=budget_termination,
                )
                break
            except ParseError as exc:
                self.environment.fail_actor(state, actor=actor, failure_kind="invalid_action")
                return self._failed_result(
                    case, state, run_id=run_id, repetition=repetition, stage="action_parse", exc=exc, responses=responses
                )
            except BackendError as exc:
                self.environment.fail_actor(state, actor=actor, failure_kind=type(exc).__name__)
                return self._failed_result(
                    case, state, run_id=run_id, repetition=repetition, stage="actor_backend", exc=exc, responses=responses
                )

        goal_cache = dict(previous_result.metadata.get("goal_evaluation_cache") or {}) if previous_result else {}
        provenance = self.provenance_for_case(case)
        goal_metrics: tuple[MetricValue, ...]
        judge_status = "unavailable"
        judge_error = None
        goal_evaluation_call_counts = {"self": 0, "others": 0, "judge": 0}
        if provenance is None:
            goal_metrics = self.scorer.unavailable_goal_metrics("judge configuration is required", None)
        else:
            try:
                if isinstance(case.metadata.get("goal_evaluation_questions"), Mapping):
                    merged, goal_evaluation_call_counts = self._evaluate_official_goal_questions(
                        case,
                        state,
                        provenance,
                        backend=backend,
                        evaluated_model=model,
                        seed=seed,
                        responses=responses,
                        cache=goal_cache,
                    )
                else:
                    # Synthetic fixtures created before official question metadata was preserved.
                    judge_jobs = tuple(enumerate(
                        zip(provenance.logical_judge_ids, provenance.judge_models)
                    ))

                    def generate_legacy_judgment(
                        job: tuple[int, tuple[str, str]],
                    ) -> tuple[Mapping[str, Any], tuple[ModelResponse, ...]]:
                        judge_index, (judge_id, judge_model) = job
                        request = self.build_judge_request(
                            case,
                            state,
                            provenance,
                            judge_model=judge_model,
                            judge_id=judge_id,
                            judge_index=judge_index,
                            seed=seed,
                        )
                        if judge_id in goal_cache:
                            return goal_cache[judge_id], ()
                        local_responses: list[ModelResponse] = []
                        payload = generate_and_parse_with_contract_retries(
                            backend=backend,
                            request=request,
                            parser=lambda response: self._parse_legacy_goal_matrix(
                                response, state=state, provenance=provenance
                            ),
                            responses=local_responses,
                            contract=str(request.metadata["output_contract"]),
                        )
                        goal_cache[judge_id] = payload
                        return payload, tuple(local_responses)

                    if self.judge_max_workers == 1:
                        legacy_results = tuple(
                            generate_legacy_judgment(job) for job in judge_jobs
                        )
                    else:
                        with ThreadPoolExecutor(
                            max_workers=min(self.judge_max_workers, len(judge_jobs)),
                            thread_name_prefix="sim-eval-agentsense-judge",
                        ) as pool:
                            legacy_results = tuple(
                                pool.map(generate_legacy_judgment, judge_jobs)
                            )
                    judge_payloads = [payload for payload, _ in legacy_results]
                    for _, judge_responses in legacy_results:
                        responses.extend(judge_responses)
                    first_payload = judge_payloads[0]
                    first_evaluations = first_payload.get("goal_evaluations")
                    if not isinstance(first_evaluations, Mapping):
                        raise ParseError("AgentSense judge response requires goal_evaluations")
                    merged = json.loads(canonical_json(first_payload))
                    for agent_id in state.agents:
                        base_items = merged["goal_evaluations"][agent_id]
                        for item_index, base_item in enumerate(base_items):
                            for judge_id, payload in zip(provenance.logical_judge_ids, judge_payloads):
                                try:
                                    value = payload["goal_evaluations"][agent_id][item_index]["judges"][judge_id]
                                except (KeyError, IndexError, TypeError) as exc:
                                    raise ParseError(
                                        f"AgentSense response from {judge_id} lacks its aligned goal judgment"
                                    ) from exc
                                base_item["judges"][judge_id] = value
                    goal_evaluation_call_counts["judge"] = len(judge_payloads)
                goal_metrics = self.scorer.score_goals(merged, agent_ids=tuple(state.agents), provenance=provenance)
                judge_status = "available"
            except EpisodeTokenBudgetExhausted as exc:
                budget_termination = budget_termination or exc.details()
                judge_error = {
                    "kind": "token_budget_exhausted",
                    "message": str(exc),
                }
                goal_metrics = self.scorer.unavailable_goal_metrics(
                    "evaluated actor token budget exhausted during official self/other interview",
                    provenance,
                )
            except (BackendError, ParseError, ValidationError) as exc:
                judge_error = {"kind": type(exc).__name__, "message": str(exc)}
                goal_metrics = self.scorer.unavailable_goal_metrics(f"judge failure: {type(exc).__name__}", provenance)

        info_metrics: list[MetricValue] = []
        info_predictions: list[dict[str, Any]] = []
        questions = [] if previous_result is not None else (case.input_data.get("information_questions") or [])
        answer_map = case.gold.get("information_answers", {}) if isinstance(case.gold, Mapping) else {}
        for index, question in enumerate(questions):
            if not isinstance(question, Mapping):
                raise ValidationError("AgentSense information question must be an object")
            question_id = str(question.get("id") or index)
            options = question.get("options")
            if budget_termination is not None:
                prediction_index = None
                correct = 0
                parse_status = "token_budget_exhausted"
            else:
                try:
                    response = backend.generate(
                        self.build_information_request(
                            case, state, question, index=index, model=model, seed=seed
                        )
                    )
                    responses.append(response)
                    prediction_index = parse_information_choice(
                        response.text, len(options), options
                    )
                    answer = answer_map.get(question_id)
                    answer_index = (
                        parse_information_choice(str(answer), len(options))
                        if not isinstance(answer, int)
                        else answer
                    )
                    correct = int(prediction_index == answer_index)
                    parse_status = "parsed"
                except EpisodeTokenBudgetExhausted as exc:
                    budget_termination = exc.details()
                    prediction_index = None
                    correct = 0
                    parse_status = "token_budget_exhausted"
                except (BackendError, ParseError) as exc:
                    prediction_index = None
                    correct = 0
                    parse_status = f"failed:{type(exc).__name__}"
            info_predictions.append(
                {"question_id": question_id, "prediction_index": prediction_index, "parse_status": parse_status}
            )
            info_metrics.append(
                MetricValue(
                    f"agentsense.information.{question_id}.accuracy",
                    correct,
                    numerator=correct,
                    denominator=1,
                    metadata={"parse_status": parse_status, "evaluator_visibility": "private_task"},
                )
            )
        if info_metrics:
            correct_sum = sum(int(metric.value or 0) for metric in info_metrics)
            info_metrics.append(
                MetricValue(
                    "agentsense.episode.private_information_accuracy",
                    correct_sum / (len(info_metrics)),
                    numerator=correct_sum,
                    denominator=len(info_metrics),
                )
            )
        else:
            info_metrics.append(
                MetricValue(
                    "agentsense.episode.private_information_accuracy",
                    None,
                    metadata={"availability": "not_applicable", "reason": "scenario has no private-information question"},
                )
            )
        profile_id = str(case.metadata.get("profile_id") or case.metadata.get("source_id") or case.case_id)
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
                "information": info_predictions,
            },
            metrics=tuple(goal_metrics) + tuple(info_metrics),
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
                "judge_error": judge_error,
                "judge_provenance": provenance.to_dict() if provenance else None,
                "judge_call_count": goal_evaluation_call_counts["judge"] if judge_status == "available" else 0,
                "goal_evaluation_call_counts": goal_evaluation_call_counts,
                "goal_evaluation_cache": goal_cache,
                "goal_evaluation_topology": (
                    "official_per_goal_self_others_three_judges"
                    if isinstance(case.metadata.get("goal_evaluation_questions"), Mapping)
                    else "legacy_batched_fixture_compatibility"
                ),
                "profile_id": profile_id,
            },
        )

    @staticmethod
    def _profile_sensitivity(
        results: Sequence[CaseResult],
        metric_name: str,
        *,
        suffix: str,
    ) -> MetricValue:
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for result in results:
            profile_id = str(result.metadata.get("profile_id", result.case_id))
            match = next((metric for metric in result.metrics if metric.name == metric_name), None)
            if match is not None and isinstance(match.value, (int, float, bool)):
                grouped[result.group_id][profile_id].append(float(match.value))
        deviations: list[float] = []
        eligible = 0
        for profiles in grouped.values():
            values = [sum(items) / len(items) for items in profiles.values()]
            if len(values) >= 2:
                eligible += 1
                deviations.append(statistics.stdev(values) * 100.0)
        return MetricValue(
            f"agentsense.profile_sensitivity_index.{suffix}",
            sum(deviations) / len(deviations) if deviations else None,
            direction="lower_is_better",
            unit="percentage_points",
            numerator=sum(deviations) if deviations else None,
            denominator=len(deviations),
            metadata={
                "aggregation": "mean_within_template_sample_standard_deviation",
                "standard_deviation_ddof": 1,
                "source_metric": metric_name,
                "eligible_template_count": eligible,
                "availability": "available" if deviations else "unavailable_requires_multiple_profiles_per_template",
            },
        )

    @staticmethod
    def _goal_profile_sensitivity(results: Sequence[CaseResult]) -> MetricValue:
        prefix = "agentsense.episode.judge."
        judge_names = sorted(
            {
                metric.name
                for result in results
                for metric in result.metrics
                if metric.name.startswith(prefix)
                and metric.name not in {
                    "agentsense.episode.judge_average",
                    "agentsense.episode.judge_majority",
                }
            }
        )
        grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for result in results:
            profile_id = str(result.metadata.get("profile_id", result.case_id))
            metric_by_name = {metric.name: metric for metric in result.metrics}
            for judge_name in judge_names:
                match = metric_by_name.get(judge_name)
                if match is not None and isinstance(match.value, (int, float, bool)):
                    grouped[result.group_id][judge_name][profile_id].append(float(match.value))

        template_deviations: list[float] = []
        per_judge_deviations: dict[str, list[float]] = defaultdict(list)
        incomplete_templates = 0
        for judges in grouped.values():
            current: list[float] = []
            for judge_name in judge_names:
                profiles = judges.get(judge_name, {})
                values = [sum(items) / len(items) for items in profiles.values()]
                if len(values) < 2:
                    current = []
                    break
                deviation = statistics.stdev(values) * 100.0
                current.append(deviation)
                per_judge_deviations[judge_name].append(deviation)
            if current and len(current) == len(judge_names):
                template_deviations.append(sum(current) / len(current))
            else:
                incomplete_templates += 1

        per_judge_means = {
            judge_name.removeprefix(prefix): sum(values) / len(values)
            for judge_name, values in per_judge_deviations.items()
            if values
        }
        return MetricValue(
            "agentsense.profile_sensitivity_index.goal",
            (
                sum(template_deviations) / len(template_deviations)
                if template_deviations
                else None
            ),
            direction="lower_is_better",
            unit="percentage_points",
            numerator=sum(template_deviations) if template_deviations else None,
            denominator=len(template_deviations),
            metadata={
                "aggregation": "mean_templates_of_mean_judges_within_template_sample_standard_deviation",
                "standard_deviation_ddof": 1,
                "judge_metric_names": judge_names,
                "per_judge_mean_sample_standard_deviation": per_judge_means,
                "eligible_template_count": len(template_deviations),
                "incomplete_template_count": incomplete_templates,
                "availability": (
                    "available"
                    if template_deviations
                    else "unavailable_requires_two_profiles_for_every_judge_within_template"
                ),
            },
        )

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        metrics = dict(aggregate_named_metrics(results, namespace="agentsense"))
        goal_psi = self._goal_profile_sensitivity(results)
        metrics[goal_psi.name] = goal_psi
        information_psi = self._profile_sensitivity(
            results,
            "agentsense.episode.private_information_accuracy",
            suffix="information",
        )
        metrics[information_psi.name] = information_psi
        return metrics


__all__ = ["AgentSenseAdapter", "AgentSenseScorer", "parse_information_choice"]
