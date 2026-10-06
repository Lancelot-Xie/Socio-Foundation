"""Role-isolated two-party dialogue environment shared by dialogue benchmarks.

The environment deliberately owns only turn taking, visibility, and terminal
state. Benchmark adapters remain responsible for constructing private
contexts and for deciding which terminal states count as valid completions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts import BenchmarkCase, ChatMessage, TraceEvent
from ..errors import ParseError, ValidationError
from ..interfaces import EnvironmentTransition, InteractiveEnvironment
from ..json_utils import canonical_json
from ..presentation import material_text


EVALUATED_USER_ROLE = "evaluated_user"
FIXED_ASSISTANT_ROLE = "fixed_assistant"
EVALUATOR_ROLE = "evaluator"


@dataclass(frozen=True)
class DialogueAction:
    """One typed utterance/control action emitted by a dialogue actor."""

    action: str
    message: str = ""


@dataclass(frozen=True)
class DialogueSpec:
    """Validated visibility and turn-taking inputs for one episode."""

    user_context: Mapping[str, Any]
    assistant_context: Mapping[str, Any]
    scheduled_actors: tuple[str, ...] = ()
    initial_messages: tuple[Mapping[str, Any], ...] = ()
    max_user_turns: int = 8
    max_total_actions: int = 32
    target_user_turns: int | None = None
    termination_token: str = "<|endconversation|>"
    user_may_end: bool = True
    assistant_may_end: bool = True

    def __post_init__(self) -> None:
        allowed = {EVALUATED_USER_ROLE, FIXED_ASSISTANT_ROLE}
        if any(actor not in allowed for actor in self.scheduled_actors):
            raise ValidationError("dialogue schedule contains an unknown actor")
        if self.max_user_turns <= 0 or self.max_total_actions <= 0:
            raise ValidationError("dialogue limits must be positive")
        if self.target_user_turns is not None:
            if self.target_user_turns <= 0:
                raise ValidationError("target_user_turns must be positive")
        if not isinstance(self.termination_token, str) or not self.termination_token:
            raise ValidationError("dialogue termination token must be non-empty text")


@dataclass
class DialogueState:
    benchmark_id: str
    case_id: str
    spec: DialogueSpec
    current_actor: str | None
    schedule_index: int = 0
    user_turn_count: int = 0
    assistant_turn_count: int = 0
    action_count: int = 0
    trace: list[TraceEvent] = field(default_factory=list)
    terminal: bool = False
    terminal_reason: str | None = None
    protocol_complete: bool = False

    @property
    def next_actor(self) -> str | None:
        return None if self.terminal else self.current_actor

    @property
    def public_transcript(self) -> tuple[TraceEvent, ...]:
        return tuple(
            event
            for event in self.trace
            if event.metadata.get("visibility") == "public_transcript"
        )


def parse_dialogue_action(
    text: str,
    *,
    termination_token: str = "<|endconversation|>",
) -> DialogueAction:
    """Parse the strict harness envelope, accepting the native end token too."""

    stripped = text.strip()
    if stripped == termination_token:
        return DialogueAction(action="end", message="")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ParseError("dialogue output must be one JSON object or the exact termination token") from exc
    if not isinstance(value, Mapping) or set(value) != {"action", "message"}:
        raise ParseError("dialogue output requires exactly action and message")
    action = value.get("action")
    message = value.get("message")
    if not isinstance(action, str) or action not in {"message", "end", "refuse"}:
        raise ParseError("dialogue action must be message, end, or refuse")
    if not isinstance(message, str):
        raise ParseError("dialogue message must be text")
    message = message.strip()
    if action in {"message", "refuse"} and not message:
        raise ParseError(f"dialogue {action} action requires a non-empty message")
    return DialogueAction(action=action, message=message)


class DialogueEnvironment(InteractiveEnvironment):
    """Two-party state machine with actor-specific private observations."""

    environment_revision = "role-isolated-dialogue-v1"
    randomize_turn_order = False
    allowed_actions = (
        "evaluated_user.message",
        "evaluated_user.end",
        "evaluated_user.refuse",
        "fixed_assistant.message",
        "fixed_assistant.end",
        "fixed_assistant.refuse",
    )

    def reset(self, case: BenchmarkCase, *, seed: int) -> DialogueState:
        raw = case.input_data.get("dialogue_environment")
        if not isinstance(raw, Mapping):
            raise ValidationError(
                "generic dialogue reset requires input.dialogue_environment; adapters may use reset_with_spec"
            )
        return self.reset_with_spec(
            case,
            DialogueSpec(
                user_context=dict(raw.get("user_context") or {}),
                assistant_context=dict(raw.get("assistant_context") or {}),
                scheduled_actors=tuple(raw.get("scheduled_actors") or ()),
                initial_messages=tuple(raw.get("initial_messages") or ()),
                max_user_turns=int(raw.get("max_user_turns", 8)),
                max_total_actions=int(raw.get("max_total_actions", 32)),
                target_user_turns=(
                    int(raw["target_user_turns"])
                    if raw.get("target_user_turns") is not None
                    else None
                ),
                termination_token=str(raw.get("termination_token") or "<|endconversation|>"),
                user_may_end=bool(raw.get("user_may_end", True)),
                assistant_may_end=bool(raw.get("assistant_may_end", True)),
            ),
            seed=seed,
        )

    def reset_with_spec(
        self,
        case: BenchmarkCase,
        spec: DialogueSpec,
        *,
        seed: int,
    ) -> DialogueState:
        del seed
        current_actor = spec.scheduled_actors[0] if spec.scheduled_actors else EVALUATED_USER_ROLE
        state = DialogueState(
            benchmark_id=case.benchmark_id,
            case_id=case.case_id,
            spec=spec,
            current_actor=current_actor,
        )
        for index, raw in enumerate(spec.initial_messages):
            if not isinstance(raw, Mapping):
                raise ValidationError(f"initial dialogue message #{index} must be an object")
            actor = str(raw.get("actor") or raw.get("role") or "")
            message = raw.get("message", raw.get("content"))
            if actor not in {EVALUATED_USER_ROLE, FIXED_ASSISTANT_ROLE}:
                raise ValidationError(f"initial dialogue message #{index} has an unknown actor")
            if not isinstance(message, str) or not message.strip():
                raise ValidationError(f"initial dialogue message #{index} must contain text")
            state.trace.append(
                TraceEvent(
                    turn=len(state.trace),
                    actor=actor,
                    kind="seed_message",
                    content=message.strip(),
                    visible_to=(EVALUATED_USER_ROLE, FIXED_ASSISTANT_ROLE),
                    metadata={"visibility": "public_transcript", "seeded": True},
                )
            )
        return state

    @staticmethod
    def _public_payload(state: DialogueState) -> list[Mapping[str, Any]]:
        return [
            {
                "turn": event.turn,
                "actor": event.actor,
                "kind": event.kind,
                "message": event.content,
            }
            for event in state.public_transcript
        ]

    def observation(
        self, state: DialogueState, *, actor: str,
        system_instruction: str | None = None,
        self_action_envelope: bool = False,
        include_private_context: bool = True,
    ) -> Sequence[ChatMessage]:
        if state.terminal:
            raise ValidationError("cannot observe a terminated dialogue")
        if actor != state.next_actor:
            raise ValidationError(f"out-of-turn dialogue observation for {actor!r}; expected {state.next_actor!r}")
        if actor == EVALUATED_USER_ROLE:
            context = state.spec.user_context
            instruction = "Act only as the evaluated user. Never write assistant, judge, or evaluator output."
        elif actor == FIXED_ASSISTANT_ROLE:
            context = state.spec.assistant_context
            instruction = "Act only as the fixed assistant. Never impersonate the evaluated user or evaluator."
        else:
            raise ValidationError(f"unknown dialogue actor {actor!r}")
        messages = [ChatMessage(
            "system", (instruction if system_instruction is None else system_instruction)
            + ("\n\n# Your Private Context\n" + material_text(context) if include_private_context else ""),
            metadata={"visibility": actor},
        )]
        for event in state.public_transcript:
            content = str(event.content)
            if event.actor == actor and self_action_envelope and event.kind in {"message", "refuse"}:
                # LiC still asks this actor for JSON actions. Restore that same
                # response envelope in its own assistant history, not the peer's.
                content = canonical_json({"action": event.kind, "message": content})
            elif event.kind == "refuse":
                content = "[refuse] " + content
            messages.append(ChatMessage(
                "assistant" if event.actor == actor else "user", content,
                metadata={"visibility": "public_transcript", "turn": event.turn,
                          "actor": event.actor, "kind": event.kind},
            ))
        return tuple(messages)

    def apply(
        self,
        state: DialogueState,
        *,
        actor: str,
        action: Any,
    ) -> EnvironmentTransition:
        if state.terminal or actor != state.current_actor:
            raise ValidationError("dialogue actor acted out of turn or after termination")
        if not isinstance(action, DialogueAction):
            raise ValidationError("dialogue transition requires DialogueAction")
        if state.action_count >= state.spec.max_total_actions:
            raise ValidationError("dialogue action limit reached")
        if actor == EVALUATED_USER_ROLE and state.user_turn_count >= state.spec.max_user_turns:
            raise ValidationError("dialogue user-turn limit reached")
        if action.action == "end":
            may_end = state.spec.user_may_end if actor == EVALUATED_USER_ROLE else state.spec.assistant_may_end
            if not may_end:
                raise ValidationError(f"{actor} termination is disabled for this protocol")

        event = TraceEvent(
            turn=len(state.trace),
            actor=actor,
            kind=action.action,
            content=action.message,
            visible_to=(EVALUATED_USER_ROLE, FIXED_ASSISTANT_ROLE),
            metadata={"visibility": "public_transcript" if action.message else "control"},
        )
        state.trace.append(event)
        state.action_count += 1
        if actor == EVALUATED_USER_ROLE and action.action == "message":
            state.user_turn_count += 1
        if actor == FIXED_ASSISTANT_ROLE and action.action == "message":
            state.assistant_turn_count += 1

        if action.action == "refuse":
            state.terminal = True
            state.protocol_complete = False
            state.terminal_reason = f"{actor}_refusal"
        elif action.action == "end":
            state.terminal = True
            state.protocol_complete = True
            state.terminal_reason = f"{actor}_end"
        else:
            self._advance(state, actor)

        return EnvironmentTransition(
            state=state,
            events=(event,),
            terminal=state.terminal,
            terminal_reason=state.terminal_reason,
            metadata={
                "protocol_complete": state.protocol_complete,
                "user_turn_count": state.user_turn_count,
                "assistant_turn_count": state.assistant_turn_count,
            },
        )

    @staticmethod
    def _advance(state: DialogueState, actor: str) -> None:
        if state.spec.scheduled_actors:
            state.schedule_index += 1
            if state.schedule_index >= len(state.spec.scheduled_actors):
                state.terminal = True
                state.protocol_complete = True
                state.terminal_reason = "scheduled_complete"
                state.current_actor = None
                return
            state.current_actor = state.spec.scheduled_actors[state.schedule_index]
            return
        if (
            actor == FIXED_ASSISTANT_ROLE
            and state.spec.target_user_turns is not None
            and state.user_turn_count >= state.spec.target_user_turns
        ):
            state.terminal = True
            state.protocol_complete = True
            state.terminal_reason = "target_user_turns_complete"
            state.current_actor = None
            return
        state.current_actor = FIXED_ASSISTANT_ROLE if actor == EVALUATED_USER_ROLE else EVALUATED_USER_ROLE

    def force_terminate(
        self,
        state: DialogueState,
        *,
        actor: str,
        reason: str,
        stage: str,
        kind: str,
    ) -> TraceEvent:
        if state.terminal:
            raise ValidationError("cannot force-terminate an already terminal dialogue")
        event = TraceEvent(
            turn=len(state.trace),
            actor=actor,
            kind="failure",
            content={"reason": reason, "stage": stage, "kind": kind},
            visible_to=(EVALUATOR_ROLE,),
            metadata={"visibility": "evaluator_only"},
        )
        state.trace.append(event)
        state.terminal = True
        state.protocol_complete = False
        state.terminal_reason = reason
        state.current_actor = None
        return event

    def terminate_capability_limit(
        self,
        state: DialogueState,
        *,
        actor: str,
        reason: str,
        details: Mapping[str, Any],
    ) -> TraceEvent:
        """Normally stop a partial dialogue at an evaluated-policy limit."""

        if state.terminal:
            raise ValidationError("cannot capability-terminate an already terminal dialogue")
        event = TraceEvent(
            turn=len(state.trace),
            actor=actor,
            kind="capability_limit",
            content={"reason": reason, **dict(details)},
            visible_to=(EVALUATOR_ROLE,),
            metadata={"visibility": "evaluator_only", "scored_outcome": True},
        )
        state.trace.append(event)
        state.terminal = True
        state.protocol_complete = False
        state.terminal_reason = reason
        state.current_actor = None
        return event


__all__ = [
    "DialogueAction",
    "DialogueEnvironment",
    "DialogueSpec",
    "DialogueState",
    "EVALUATED_USER_ROLE",
    "EVALUATOR_ROLE",
    "FIXED_ASSISTANT_ROLE",
    "parse_dialogue_action",
]
