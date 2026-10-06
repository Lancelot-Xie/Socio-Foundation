"""Dependency-free, role-isolated social-scene environment."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts import BenchmarkCase, ChatMessage, TraceEvent
from ..errors import ConfigurationError, ParseError, ValidationError
from ..interfaces import EnvironmentTransition, InteractiveEnvironment


@dataclass(frozen=True)
class JudgeProvenance:
    judge_models: tuple[str, ...]
    judge_revisions: tuple[str, ...]
    rubric_revision: str
    calls_per_output: int = 1
    source: str = "configured"
    judge_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.judge_models or len(self.judge_models) != len(self.judge_revisions):
            raise ConfigurationError("judge provenance requires equally sized, nonempty model and revision lists")
        if not self.rubric_revision:
            raise ConfigurationError("judge provenance requires rubric_revision")
        if self.calls_per_output <= 0:
            raise ConfigurationError("judge calls_per_output must be positive")
        if self.judge_ids and len(self.judge_ids) != len(self.judge_models):
            raise ConfigurationError("judge provenance judge_ids must match the model list length")
        if self.judge_ids and len(set(self.judge_ids)) != len(self.judge_ids):
            raise ConfigurationError("judge provenance judge_ids must be distinct")

    @property
    def logical_judge_ids(self) -> tuple[str, ...]:
        """Stable scoring identities, defaulting to the legacy model-name keys."""

        return self.judge_ids or self.judge_models

    def to_dict(self) -> dict[str, Any]:
        result = {
            "judge_models": list(self.judge_models),
            "judge_revisions": list(self.judge_revisions),
            "rubric_revision": self.rubric_revision,
            "calls_per_output": self.calls_per_output,
            "source": self.source,
        }
        # Keep legacy provenance and run identities byte-for-byte stable when
        # the optional logical IDs are not configured.
        if self.judge_ids:
            result["judge_ids"] = list(self.judge_ids)
        return result


@dataclass(frozen=True)
class SocialAgent:
    agent_id: str
    role: str
    profile: str
    goal: str
    private_information: Any = None


@dataclass(frozen=True)
class SocialAction:
    action_type: str
    content: str = ""


@dataclass
class SocialState:
    benchmark_id: str
    scenario_id: str
    public_background: str
    agents: Mapping[str, SocialAgent]
    turn_order: tuple[str, ...]
    max_turns: int
    allowed_actions: tuple[str, ...]
    seed: int
    speaker_selection_method: str
    turn_count: int = 0
    next_actor_index: int = 0
    transcript: list[TraceEvent] = field(default_factory=list)
    terminal: bool = False
    terminal_reason: str | None = None

    @property
    def next_actor(self) -> str | None:
        if self.terminal:
            return None
        return self.turn_order[self.next_actor_index]


_ACTION_ALIASES = {
    "speak": "speak",
    "none": "none",
    "nonverbal": "nonverbal",
    "non-verbal": "nonverbal",
    "non-verbal communication": "nonverbal",
    "action": "action",
    "physical_action": "action",
    "leave": "leave",
}


def parse_social_action(text: str, allowed_actions: Sequence[str]) -> SocialAction:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError("social action must be one JSON object") from exc
    if not isinstance(payload, dict):
        raise ParseError("social action must be a JSON object")
    raw_type = payload.get("action_type")
    if not isinstance(raw_type, str):
        raise ParseError("social action requires string action_type")
    action_type = _ACTION_ALIASES.get(raw_type.strip().lower())
    if action_type is None or action_type not in allowed_actions:
        raise ParseError(f"unsupported social action_type {raw_type!r}; allowed={sorted(allowed_actions)}")
    content = payload.get("content", payload.get("argument", ""))
    if not isinstance(content, str):
        raise ParseError("social action content/argument must be a string")
    content = content.strip()
    if action_type in {"speak", "nonverbal", "action"} and not content:
        raise ParseError(f"{action_type} action requires non-empty content")
    return SocialAction(action_type=action_type, content=content)


class RoleIsolatedSocialEnvironment(InteractiveEnvironment):
    """N-agent scheduler whose observations expose only the current role's private state.

    The default remains exactly two agents so existing dyadic benchmarks stay
    fail-closed.  A benchmark must opt in explicitly to a wider bounded range.
    """

    environment_revision = "role-isolated-social-v4-optional-stale-termination"

    def __init__(
        self,
        *,
        randomize_turn_order: bool,
        default_max_turns: int,
        allowed_actions: Sequence[str] = ("none", "speak", "nonverbal", "action", "leave"),
        min_agent_count: int = 2,
        max_agent_count: int = 2,
        speaker_selection_method: str | None = None,
        max_stale_turns: int | None = None,
    ) -> None:
        if default_max_turns <= 0:
            raise ConfigurationError("default_max_turns must be positive")
        if isinstance(min_agent_count, bool) or not isinstance(min_agent_count, int) or min_agent_count < 2:
            raise ConfigurationError("min_agent_count must be an integer greater than or equal to two")
        if (
            isinstance(max_agent_count, bool)
            or not isinstance(max_agent_count, int)
            or max_agent_count < min_agent_count
        ):
            raise ConfigurationError("max_agent_count must be an integer greater than or equal to min_agent_count")
        normalized = tuple(_ACTION_ALIASES.get(action, action) for action in allowed_actions)
        if not normalized or any(action not in set(_ACTION_ALIASES.values()) for action in normalized):
            raise ConfigurationError("invalid social allowed_actions")
        selection_method = speaker_selection_method or (
            "initial_shuffle_round_robin" if randomize_turn_order else "round_robin"
        )
        if selection_method not in {"round_robin", "initial_shuffle_round_robin", "random_no_repeat"}:
            raise ConfigurationError(f"unsupported social speaker_selection_method {selection_method!r}")
        if max_stale_turns is not None and (
            isinstance(max_stale_turns, bool)
            or not isinstance(max_stale_turns, int)
            or max_stale_turns < 0
        ):
            raise ConfigurationError("max_stale_turns must be a nonnegative integer or null")
        self.randomize_turn_order = randomize_turn_order
        self.speaker_selection_method = selection_method
        self.default_max_turns = default_max_turns
        self.allowed_actions = normalized
        self.min_agent_count = min_agent_count
        self.max_agent_count = max_agent_count
        self.max_stale_turns = max_stale_turns

    def identity(self) -> dict[str, Any]:
        return {
            "revision": self.environment_revision,
            "randomize_turn_order": self.randomize_turn_order,
            "allowed_actions": list(self.allowed_actions),
            "min_agent_count": self.min_agent_count,
            "max_agent_count": self.max_agent_count,
            "turn_scheduler": self.speaker_selection_method,
            "allow_repeat_speaker": False,
            "max_stale_turns": self.max_stale_turns,
            "private_state_visibility": "current_actor_only",
        }

    def reset(self, case: BenchmarkCase, *, seed: int) -> SocialState:
        raw_agents = case.input_data.get("agents")
        if not isinstance(raw_agents, Sequence) or isinstance(raw_agents, (str, bytes)):
            raise ValidationError("role-isolated social environment requires an agents array")
        agent_count = len(raw_agents)
        if not self.min_agent_count <= agent_count <= self.max_agent_count:
            if self.min_agent_count == self.max_agent_count:
                requirement = f"exactly {self.min_agent_count} agents"
            else:
                requirement = f"between {self.min_agent_count} and {self.max_agent_count} agents"
            raise ValidationError(f"role-isolated social environment requires {requirement}; got {agent_count}")
        profiles = case.input_data.get("profiles") or {}
        goals = case.input_data.get("goals") or {}
        private = case.input_data.get("private_information") or {}
        if not all(isinstance(value, Mapping) for value in (profiles, goals, private)):
            raise ValidationError("profiles, goals, and private_information must be mappings when present")
        agents: dict[str, SocialAgent] = {}
        for index, value in enumerate(raw_agents):
            if not isinstance(value, Mapping):
                raise ValidationError("each social agent must be an object")
            agent_id = str(value.get("id") or f"agent_{index}")
            if agent_id in agents:
                raise ValidationError(f"duplicate social agent ID {agent_id!r}")
            agents[agent_id] = SocialAgent(
                agent_id=agent_id,
                role=str(value.get("role") or agent_id),
                profile=str(profiles.get(agent_id, value.get("profile", value.get("role", agent_id)))),
                goal=str(goals.get(agent_id, value.get("goal", ""))),
                private_information=private.get(agent_id),
            )
        turn_order = list(agents)
        next_actor_index = 0
        if self.speaker_selection_method == "initial_shuffle_round_robin":
            random.Random(seed).shuffle(turn_order)
        elif self.speaker_selection_method == "random_no_repeat":
            next_actor_index = random.Random(seed).randrange(len(turn_order))
        max_turns = int(case.input_data.get("max_turns", self.default_max_turns))
        if max_turns <= 0:
            raise ValidationError("social max_turns must be positive")
        return SocialState(
            benchmark_id=case.benchmark_id,
            scenario_id=case.case_id,
            public_background=str(case.input_data.get("scenario", "")),
            agents=agents,
            turn_order=tuple(turn_order),
            max_turns=max_turns,
            allowed_actions=self.allowed_actions,
            seed=seed,
            speaker_selection_method=self.speaker_selection_method,
            next_actor_index=next_actor_index,
        )

    def observation(self, state: SocialState, *, actor: str) -> Sequence[ChatMessage]:
        if actor not in state.agents:
            raise ValidationError(f"unknown social actor {actor!r}")
        if state.terminal:
            raise ValidationError("cannot observe a terminated social episode")
        if actor != state.next_actor:
            raise ValidationError(f"out-of-turn observation for {actor!r}; next actor is {state.next_actor!r}")
        agent = state.agents[actor]
        other_public = [
            {"agent_id": candidate.agent_id, "role": candidate.role}
            for candidate in state.agents.values()
            if candidate.agent_id != actor
        ]
        role_view = {
            "public_background": state.public_background,
            "your_agent_id": agent.agent_id,
            "your_role": agent.role,
            "your_profile": agent.profile,
            "your_goal": agent.goal,
            "your_private_information": agent.private_information,
            "other_participants_public": other_public,
            "allowed_actions": list(state.allowed_actions),
            "turn": state.turn_count,
            "max_turns": state.max_turns,
        }
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Act only as the assigned social participant. You may use your private information, "
                    "but you do not know another participant's private information unless they disclose it.\n"
                    + json.dumps(role_view, ensure_ascii=False, sort_keys=True)
                ),
                metadata={"visibility": f"actor:{actor}", "environment_revision": self.environment_revision},
            )
        ]
        for event in state.transcript:
            messages.append(
                ChatMessage(
                    role="assistant" if event.actor == actor else "user",
                    name=event.actor,
                    content=f"[{event.kind}] {event.content}",
                    metadata={"visibility": "public_transcript", "turn": event.turn},
                )
            )
        return messages

    def apply(self, state: SocialState, *, actor: str, action: Any) -> EnvironmentTransition:
        if state.terminal:
            raise ValidationError("cannot apply an action to a terminated social episode")
        if actor != state.next_actor:
            raise ValidationError(f"out-of-turn action for {actor!r}; next actor is {state.next_actor!r}")
        if not isinstance(action, SocialAction):
            raise ValidationError("social environment requires a parsed SocialAction")
        if action.action_type not in state.allowed_actions:
            raise ValidationError(f"action type {action.action_type!r} is not allowed")
        if action.action_type in {"speak", "nonverbal", "action"} and not action.content:
            raise ValidationError(f"{action.action_type} action requires content")
        event = TraceEvent(
            turn=state.turn_count,
            actor=actor,
            kind=action.action_type,
            content=action.content,
            visible_to=tuple(state.agents),
            metadata={"visibility": "public_transcript"},
        )
        state.transcript.append(event)
        state.turn_count += 1
        stale_turn_count = 0
        if self.max_stale_turns is not None:
            for prior_event in reversed(state.transcript):
                if prior_event.kind != "none":
                    break
                stale_turn_count += 1
        if action.action_type == "leave":
            state.terminal = True
            state.terminal_reason = "agent_left"
        elif state.turn_count >= state.max_turns:
            state.terminal = True
            state.terminal_reason = "max_turns"
        elif self.max_stale_turns is not None and stale_turn_count > self.max_stale_turns:
            state.terminal = True
            state.terminal_reason = "stale_turns"
        elif state.speaker_selection_method == "random_no_repeat":
            candidates = [index for index in range(len(state.turn_order)) if index != state.next_actor_index]
            random_seed = int.from_bytes(
                hashlib.sha256(
                    f"{state.seed}\0{state.turn_count}\0{actor}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            state.next_actor_index = random.Random(random_seed).choice(candidates)
        else:
            state.next_actor_index = (state.next_actor_index + 1) % len(state.turn_order)
        return EnvironmentTransition(
            state=state,
            events=(event,),
            terminal=state.terminal,
            terminal_reason=state.terminal_reason,
            metadata={"turn_count": state.turn_count, "stale_turn_count": stale_turn_count},
        )

    def fail_actor(self, state: SocialState, *, actor: str, failure_kind: str) -> EnvironmentTransition:
        if state.terminal:
            raise ValidationError("cannot fail an actor in a terminated episode")
        event = TraceEvent(
            turn=state.turn_count,
            actor=actor,
            kind="actor_failure",
            content=failure_kind,
            visible_to=(),
            metadata={"visibility": "evaluator_only"},
        )
        state.transcript.append(event)
        state.terminal = True
        state.terminal_reason = "actor_failure"
        return EnvironmentTransition(
            state=state,
            events=(event,),
            terminal=True,
            terminal_reason=state.terminal_reason,
        )

    def terminate_capability_limit(
        self,
        state: SocialState,
        *,
        actor: str,
        reason: str,
        details: Mapping[str, Any],
    ) -> EnvironmentTransition:
        """Stop at a target-model limit while retaining a scoreable transcript."""

        if state.terminal:
            raise ValidationError("cannot capability-terminate a terminated episode")
        event = TraceEvent(
            turn=state.turn_count,
            actor=actor,
            kind="capability_limit",
            content={"reason": reason, **dict(details)},
            visible_to=(),
            metadata={"visibility": "evaluator_only", "scored_outcome": True},
        )
        state.transcript.append(event)
        state.terminal = True
        state.terminal_reason = reason
        return EnvironmentTransition(
            state=state,
            events=(event,),
            terminal=True,
            terminal_reason=state.terminal_reason,
            metadata={"scored_outcome": True},
        )
