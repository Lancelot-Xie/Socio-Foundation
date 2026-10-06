"""Task-oriented user/fixed-assistant/tool environment for tau-USI episodes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts import BenchmarkCase, ChatMessage, ModelResponse, TraceEvent
from ..errors import ParseError, ValidationError
from ..interfaces import EnvironmentTransition, InteractiveEnvironment
from ..json_utils import canonical_json
from ..integrations.tau_usi_protocol import (
    TOOL_PROMPT, convert_tools_to_description, extract_fn_call, coerce_tool_arguments, tool_call_text,
)


USER_ROLE = "evaluated_user"
ASSISTANT_ROLE = "fixed_assistant"
ENVIRONMENT_ROLE = "task_environment"
INITIAL_ASSISTANT_MESSAGE = "Hi! How can I help you today?"
TAU_USER_STOP_TOKEN = "###STOP###"
_TAGGED_REASONING = re.compile(
    r"<(?:seed:)?think>.*?</(?:seed:)?think>|<(?:seed:)?think>.*\Z",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class TauUserAction:
    action: str
    message: str = ""


@dataclass(frozen=True)
class TauToolCall:
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class TauAssistantAction:
    message: str
    tool_call: TauToolCall | None
    tool_calls: tuple[TauToolCall, ...] = ()
    format_error: str | None = None


@dataclass(frozen=True)
class TauToolSpec:
    name: str
    description: str
    required_arguments: tuple[str, ...]
    parameters: Mapping[str, Any] | None = None

    def to_model_tool(self) -> Mapping[str, Any]:
        parameters = self.parameters or {
            "type": "object",
            "properties": {name: {"type": "string"} for name in self.required_arguments},
            "required": list(self.required_arguments),
            "additionalProperties": True,
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(parameters),
            },
        }


@dataclass(frozen=True)
class TauToolReplay:
    name: str
    arguments: Mapping[str, Any]
    observation: Any
    state_patch: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TauTaskState:
    benchmark_id: str
    case_id: str
    domain: str
    user_goal: str
    initial_user_state: Mapping[str, Any]
    assistant_policy: str
    tools: Mapping[str, TauToolSpec]
    tool_replays: tuple[TauToolReplay, ...]
    environment_state: dict[str, Any]
    current_actor: str = USER_ROLE
    user_turn_count: int = 0
    assistant_step_count: int = 0
    trace: list[TraceEvent] = field(default_factory=list)
    terminal: bool = False
    terminal_reason: str | None = None
    environment_reward: float | None = None
    local_runtime: Any = None

    @property
    def next_actor(self) -> str | None:
        return None if self.terminal else self.current_actor

    @property
    def public_transcript(self) -> tuple[TraceEvent, ...]:
        return tuple(event for event in self.trace if event.metadata.get("visibility") == "public_transcript")


def parse_user_action(text: str) -> TauUserAction:
    """Parse the upstream tau-bench raw user-message protocol."""

    message = _TAGGED_REASONING.sub("", text).strip()
    if not message:
        return TauUserAction(action="empty", message="")
    # Upstream asks for a standalone marker but terminates on marker presence.
    # Matching that execution rule prevents an otherwise finished task from
    # continuing merely because a model added a short closing phrase.
    if TAU_USER_STOP_TOKEN in message:
        return TauUserAction(action="stop", message=message)
    return TauUserAction(action="send", message=message)


def parse_assistant_response(response: ModelResponse) -> TauAssistantAction:
    """Parse Supplemental text calls; retain native metadata for legacy replay consumers."""

    raw = response.raw if isinstance(response.raw, Mapping) else {}
    protocol = raw.get("_sim_eval") if isinstance(raw, Mapping) else None
    native = protocol.get("native_tool_call") if isinstance(protocol, Mapping) else None
    if native is not None:
        if not isinstance(native, Mapping) or set(native) != {"name", "arguments"}:
            raise ParseError("tau-USI native tool call requires exactly name and arguments")
        name = native.get("name")
        arguments = native.get("arguments")
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, Mapping):
            raise ParseError("tau-USI native tool name must be text and arguments must be an object")
        # As in tau-bench, a valid tool call takes precedence over any
        # simultaneous assistant content.
        return TauAssistantAction(
            message="",
            tool_call=TauToolCall(name=name.strip(), arguments=dict(arguments)),
        )
    message = response.text
    parsed = extract_fn_call(message)
    if isinstance(parsed, dict):
        return TauAssistantAction(message=message, tool_call=None, format_error=parsed["error"])
    calls = tuple(TauToolCall(name=item["name"], arguments=item["arguments"]) for item in (parsed or ()))
    return TauAssistantAction(message=message, tool_call=None, tool_calls=calls)



class TauTaskEnvironment(InteractiveEnvironment):
    """Separates user-visible, assistant-visible, and evaluator-only task state."""

    environment_revision = "tau-usi-task-loop-v4-supplemental-text-tools-user-handoff"
    tool_schema_revision = "supplemental-text-tool-calls-v4"
    tool_error_policy_revision = "tau-bench-environment-observation-v1"
    randomize_turn_order = False
    allowed_actions = ("user.send", "user.stop", "assistant.message", "assistant.tool_call")

    def __init__(self) -> None:
        pass

    @staticmethod
    def _parse_tools(raw: Any) -> Mapping[str, TauToolSpec]:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            raise ValidationError("tau-USI input.tools must be a non-empty array")
        tools: dict[str, TauToolSpec] = {}
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ValidationError(f"tau-USI tool #{index} must be an object")
            name = str(item.get("name") or "").strip()
            required = item.get("required_arguments") or []
            parameters = item.get("parameters")
            if not name or not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
                raise ValidationError(f"invalid tau-USI tool #{index}")
            if parameters is not None and not isinstance(parameters, Mapping):
                raise ValidationError(f"tau-USI tool {name!r} parameters must be an object")
            required_names = tuple(str(field).strip() for field in required)
            if any(not field for field in required_names) or len(set(required_names)) != len(required_names):
                raise ValidationError(f"tau-USI tool {name!r} has empty or duplicate required arguments")
            if name in tools:
                raise ValidationError(f"duplicate tau-USI tool {name!r}")
            tools[name] = TauToolSpec(
                name=name,
                description=str(item.get("description") or ""),
                required_arguments=required_names,
                parameters=dict(parameters) if isinstance(parameters, Mapping) else None,
            )
        return tools

    @staticmethod
    def _parse_tool_replays(case: BenchmarkCase) -> tuple[TauToolReplay, ...]:
        replay = case.metadata.get("replay")
        raw = replay.get("tool_results") if isinstance(replay, Mapping) else None
        if raw is None:
            return ()
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValidationError("tau-USI replay.tool_results must be an array")
        values = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ValidationError(f"tau-USI tool result #{index} must be an object")
            name = str(item.get("name") or "").strip()
            arguments = item.get("arguments")
            state_patch = item.get("state_patch") or {}
            if not name or not isinstance(arguments, Mapping) or not isinstance(state_patch, Mapping):
                raise ValidationError(f"invalid tau-USI tool result #{index}")
            values.append(
                TauToolReplay(
                    name=name,
                    arguments=dict(arguments),
                    observation=item.get("observation"),
                    state_patch=dict(state_patch),
                )
            )
        return tuple(values)

    def reset(self, case: BenchmarkCase, *, seed: int) -> TauTaskState:
        domain = str(case.input_data.get("domain") or "")
        if domain not in {"retail", "airline"}:
            raise ValidationError("tau-USI domain must be retail or airline")
        user_goal = str(case.input_data.get("user_goal") or "").strip()
        assistant_policy = str(case.input_data.get("assistant_policy") or "").strip()
        if not user_goal or not assistant_policy:
            raise ValidationError("tau-USI user_goal and assistant_policy must be non-empty")
        initial = case.input_data.get("initial_state")
        if not isinstance(initial, Mapping):
            raise ValidationError("tau-USI initial_state must be an object")
        environment_state = case.input_data.get("environment_state") or {}
        if not isinstance(environment_state, Mapping):
            raise ValidationError("tau-USI environment_state must be an object")
        tools = self._parse_tools(case.input_data.get("tools"))
        tool_replays = self._parse_tool_replays(case)
        local_runtime = None
        runtime_config = case.input_data.get("environment_runtime")
        if runtime_config is not None:
            if not isinstance(runtime_config, Mapping):
                raise ValidationError("tau-USI environment_runtime must be an object")
            if runtime_config.get("kind") != "local_tau_bench_v1":
                raise ValidationError("unsupported tau-USI environment_runtime.kind")
            task_index = runtime_config.get("task_index")
            if isinstance(task_index, bool) or not isinstance(task_index, int):
                raise ValidationError("local tau-bench task_index must be an integer")
            from ..integrations.tau_bench_local import TauBenchLocalSession

            local_runtime = TauBenchLocalSession(
                domain,
                task_index,
                expected_runtime_digest=str(runtime_config.get("runtime_digest") or "") or None,
                expected_instruction=user_goal,
            )
            runtime_specs = {
                str(item["name"]): item
                for item in local_runtime.repository.tool_specs(domain)
            }
            if set(runtime_specs) != set(tools):
                raise ValidationError("tau-USI declared tools differ from the pinned local tau-bench runtime")
            for name, spec in tools.items():
                runtime_parameters = runtime_specs[name].get("parameters")
                if spec.parameters is None or canonical_json(spec.parameters) != canonical_json(runtime_parameters):
                    raise ValidationError(f"tau-USI tool schema differs from local runtime for {name!r}")
        for replay in tool_replays:
            spec = tools.get(replay.name)
            if spec is None:
                raise ValidationError(f"tau-USI replay result names undeclared tool {replay.name!r}")
            missing = [name for name in spec.required_arguments if name not in replay.arguments]
            if missing:
                raise ValidationError(
                    f"tau-USI replay result for {replay.name!r} is missing required arguments {missing}"
                )
        state = TauTaskState(
            benchmark_id=case.benchmark_id,
            case_id=case.case_id,
            domain=domain,
            user_goal=user_goal,
            initial_user_state=dict(initial),
            assistant_policy=assistant_policy,
            tools=tools,
            tool_replays=tool_replays,
            environment_state=dict(environment_state),
            local_runtime=local_runtime,
        )
        state.trace.append(
            TraceEvent(
                turn=0,
                actor=ASSISTANT_ROLE,
                kind="message",
                content=INITIAL_ASSISTANT_MESSAGE,
                visible_to=(USER_ROLE, ASSISTANT_ROLE),
                metadata={
                    "visibility": "public_transcript",
                    "seeded": True,
                    "source": "tau-bench LLMUserSimulationEnv.reset",
                },
            )
        )
        return state

    @staticmethod
    def _public_messages(state: TauTaskState) -> list[Mapping[str, Any]]:
        return [
            {"turn": event.turn, "role": event.actor, "message": event.content}
            for event in state.public_transcript
        ]

    def observation(self, state: TauTaskState, *, actor: str, allow_terminal: bool = False) -> Sequence[ChatMessage]:
        if not allow_terminal and (state.terminal or actor != state.next_actor):
            raise ValidationError("out-of-turn tau-USI observation or terminated task")
        if actor == USER_ROLE:
            instruction = "Role-play only the assigned customer. The fixed assistant and tool environment are external actors."
            private = "# Domain\n" + state.domain + "\n\n# Your Goal\n" + state.user_goal
            system = instruction + "\n\n" + private
        elif actor == ASSISTANT_ROLE:
            system = state.assistant_policy + "\n\n" + TOOL_PROMPT.format(
                description=convert_tools_to_description([tool.to_model_tool() for tool in state.tools.values()])
            )
        else:
            raise ValidationError(f"unknown tau-USI actor {actor!r}")
        messages = [ChatMessage("system", system, metadata={"visibility": actor})]
        for event in state.trace:
            if actor not in event.visible_to or event.kind in {"tool_call", "budget_limit"}:
                continue
            if event.kind == "tool_observation":
                # Individual calls are audited separately; the combined response
                # is emitted once, exactly as Supplemental's inner assistant loop.
                continue
            if event.kind == "tool_feedback":
                messages.append(ChatMessage("system", str(event.content), metadata={"source": "tau_tool_feedback", "turn": event.turn}))
            elif event.kind == "tool_format_error":
                messages.append(ChatMessage("user", str(event.content), metadata={"turn": event.turn}))
            elif event.kind in {"message", "assistant_text", "assistant_handoff", "stop", "empty"}:
                messages.append(ChatMessage("assistant" if event.actor == actor else "user", str(event.content), metadata={"turn": event.turn}))
        return tuple(messages)

    def apply(self, state: TauTaskState, *, actor: str, action: Any) -> EnvironmentTransition:
        if actor == USER_ROLE:
            if not isinstance(action, TauUserAction):
                raise ValidationError("tau-USI user transition requires TauUserAction")
            return self.apply_user(state, action)
        if actor == ASSISTANT_ROLE:
            if not isinstance(action, TauAssistantAction):
                raise ValidationError("tau-USI assistant transition requires TauAssistantAction")
            return self.apply_assistant(state, action)
        raise ValidationError(f"unknown tau-USI actor {actor!r}")

    def apply_user(self, state: TauTaskState, action: TauUserAction) -> EnvironmentTransition:
        if state.terminal or state.current_actor != USER_ROLE:
            raise ValidationError("tau-USI user acted out of turn or after termination")
        if action.action in {"stop", "empty"}:
            event = TraceEvent(
                turn=len(state.trace),
                actor=USER_ROLE,
                kind=action.action,
                content=action.message,
                visible_to=(USER_ROLE,),
                metadata={"visibility": "control"},
            )
            state.trace.append(event)
            state.terminal = True
            state.terminal_reason = "user_stop" if action.action == "stop" else "empty_user_response"
            return EnvironmentTransition(state=state, events=(event,), terminal=True, terminal_reason=state.terminal_reason)
        event = TraceEvent(
            turn=len(state.trace),
            actor=USER_ROLE,
            kind="message",
            content=action.message,
            visible_to=(USER_ROLE, ASSISTANT_ROLE),
            metadata={"visibility": "public_transcript"},
        )
        state.trace.append(event)
        state.user_turn_count += 1
        state.current_actor = ASSISTANT_ROLE
        state.assistant_step_count = 0
        return EnvironmentTransition(state=state, events=(event,), metadata={"user_turn_count": state.user_turn_count})

    def apply_assistant(self, state: TauTaskState, action: TauAssistantAction) -> EnvironmentTransition:
        if state.terminal or state.current_actor != ASSISTANT_ROLE:
            raise ValidationError("tau-USI assistant acted out of turn or after termination")
        state.assistant_step_count += 1
        calls = action.tool_calls or ((action.tool_call,) if action.tool_call is not None else ())
        if calls or action.format_error:
            text = action.message or "\n\n".join(tool_call_text(call.name, call.arguments) for call in calls)
            before = len(state.trace)
            state.trace.append(TraceEvent(
                turn=before, actor=ASSISTANT_ROLE, kind="assistant_text", content=text,
                visible_to=(ASSISTANT_ROLE,), metadata={"visibility": "assistant_and_evaluator"},
            ))
            if action.format_error:
                state.trace.append(TraceEvent(
                    turn=len(state.trace), actor=ENVIRONMENT_ROLE, kind="tool_format_error",
                    content=action.format_error, visible_to=(ASSISTANT_ROLE,),
                    metadata={"visibility": "assistant_and_evaluator"},
                ))
            else:
                observations = []
                for call in calls:
                    transition = self.execute_tool(state, call)
                    observations.append(str(transition.events[-1].content))
                state.trace.append(TraceEvent(
                    turn=len(state.trace), actor=ENVIRONMENT_ROLE, kind="tool_feedback",
                    content="\n\n".join(observations), visible_to=(ASSISTANT_ROLE,),
                    metadata={"visibility": "assistant_and_evaluator"},
                ))
            return EnvironmentTransition(state=state, events=tuple(state.trace[before:]))
        event = TraceEvent(
            turn=len(state.trace),
            actor=ASSISTANT_ROLE,
            kind="message",
            content=action.message,
            visible_to=(USER_ROLE, ASSISTANT_ROLE),
            metadata={"visibility": "public_transcript"},
        )
        state.trace.append(event)
        if state.local_runtime is not None:
            state.local_runtime.record_assistant_message(action.message)
        state.current_actor = USER_ROLE
        return EnvironmentTransition(state=state, events=(event,))

    def execute_tool(self, state: TauTaskState, call: TauToolCall) -> EnvironmentTransition:
        spec = state.tools.get(call.name)
        missing = (
            [name for name in spec.required_arguments if name not in call.arguments]
            if spec is not None
            else []
        )
        tool_error_kind: str | None = None
        try:
            arguments = coerce_tool_arguments(call.arguments, [tool.to_model_tool() for tool in state.tools.values()], call.name)
            argument_error = None
        except ValueError as exc:
            arguments, argument_error = call.arguments, f"Error: {exc}"
        if argument_error is not None:
            observation, terminal, state_patch = argument_error, False, {}
        elif state.local_runtime is not None:
            observation, terminal = state.local_runtime.execute_tool(call.name, arguments)
            state_patch: Mapping[str, Any] = {}
        else:
            if spec is None:
                # Match tau-bench Env.step: an unknown action is returned to the
                # assistant as an observation instead of terminating the task.
                observation = f"Unknown action {call.name}"
                state_patch = {}
                terminal = False
            elif missing:
                # The live tau-bench tool invocation exposes Python argument
                # errors as ``Error: ...`` observations.  Replay-only fixtures
                # cannot invoke the tool, so preserve the same public behavior
                # with a deterministic equivalent.
                observation = f"Error: {call.name} missing required arguments {missing}"
                state_patch = {}
                terminal = False
            else:
                match = next(
                    (
                        item
                        for item in state.tool_replays
                        if item.name == call.name
                        and canonical_json(item.arguments) == canonical_json(arguments)
                    ),
                    None,
                )
                if match is None:
                    raise ValidationError(
                        "replay_configuration_error: no authorized result for "
                        f"{call.name}({canonical_json(call.arguments)})"
                    )
                observation = match.observation
                state_patch = match.state_patch
                terminal = False
        if spec is None:
            tool_error_kind = "unknown_action"
        elif missing:
            tool_error_kind = "missing_required_arguments"
        elif isinstance(observation, str) and observation.startswith("Error:"):
            tool_error_kind = "tool_exception"
        call_event = TraceEvent(
            turn=len(state.trace),
            actor=ASSISTANT_ROLE,
            kind="tool_call",
            content={"name": call.name, "arguments": dict(call.arguments)},
            visible_to=(ASSISTANT_ROLE,),
            metadata={"visibility": "assistant_and_evaluator"},
        )
        observation_event = TraceEvent(
            turn=len(state.trace) + 1,
            actor=ENVIRONMENT_ROLE,
            kind="tool_observation",
            content=observation,
            visible_to=(ASSISTANT_ROLE,),
            metadata={
                "visibility": "assistant_and_evaluator",
                "tool": call.name,
                "tool_error_kind": tool_error_kind,
                "tool_error_policy_revision": self.tool_error_policy_revision,
            },
        )
        state.trace.extend((call_event, observation_event))
        state.environment_state.update(state_patch)
        state.current_actor = ASSISTANT_ROLE
        # The tool runtime can terminate, but Supplemental continues the dialogue
        # until the user sees the assistant's confirmation and emits STOP.
        return EnvironmentTransition(
            state=state,
            events=(call_event, observation_event),
            terminal=False,
            metadata={"tool_environment_terminal": terminal},
        )

    def finalize_reward(self, state: TauTaskState, value: Any) -> float | None:
        if state.local_runtime is not None:
            value = state.local_runtime.calculate_reward()
        if value is None:
            state.environment_reward = None
            return None
        if isinstance(value, bool):
            reward = float(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            reward = float(value)
        else:
            raise ValidationError("tau-USI environment reward must be numeric or null")
        if not 0 <= reward <= 1:
            raise ValidationError("tau-USI environment reward must be in [0,1]")
        state.environment_reward = reward
        return reward

    def handoff_assistant_step_limit(self, state: TauTaskState, *, limit: int) -> EnvironmentTransition:
        """Yield the final actual assistant text after exhausting tool rounds."""
        if state.terminal or state.current_actor != ASSISTANT_ROLE or state.assistant_step_count < limit:
            raise ValidationError("assistant handoff requires exhausted pending assistant work")
        last = next(event for event in reversed(state.trace)
                    if event.actor == ASSISTANT_ROLE and event.kind in {"assistant_text", "message"})
        event = TraceEvent(
            turn=len(state.trace), actor=ASSISTANT_ROLE, kind="budget_limit",
            content={"stage": "assistant_step_limit", "kind": "step_limit",
                     "max_assistant_steps_per_user_turn": limit,
                     "assistant_step_count": state.assistant_step_count,
                     "user_turn_count": state.user_turn_count,
                     "continuation": "yield_last_assistant_text_to_user"},
            visible_to=(), metadata={"visibility": "evaluator_only"},
        )
        handoff = TraceEvent(
            turn=len(state.trace) + 1, actor=ASSISTANT_ROLE, kind="assistant_handoff",
            content=last.content, visible_to=(USER_ROLE,),
            metadata={"visibility": "public_transcript", "source_turn": last.turn},
        )
        state.trace.extend((event, handoff))
        state.current_actor = USER_ROLE
        return EnvironmentTransition(state=state, events=(event, handoff))

    def terminate_capability_limit(
        self,
        state: TauTaskState,
        *,
        actor: str,
        stage: str,
        kind: str,
    ) -> EnvironmentTransition:
        """End a valid rollout whose evaluated policy exhausted a protocol limit."""

        if state.terminal:
            raise ValidationError("cannot capability-terminate a completed tau-USI task")
        event = TraceEvent(
            turn=len(state.trace),
            actor=actor,
            kind="capability_limit",
            content={"stage": stage, "kind": kind},
            visible_to=(),
            metadata={"visibility": "evaluator_only", "scored_outcome": True},
        )
        state.trace.append(event)
        state.terminal = True
        state.terminal_reason = stage
        return EnvironmentTransition(
            state=state,
            events=(event,),
            terminal=True,
            terminal_reason=state.terminal_reason,
            metadata={
                "scored_outcome": True,
                "environment_reward_policy": "final_environment_state",
            },
        )

    def fail(self, state: TauTaskState, *, actor: str, stage: str, kind: str) -> EnvironmentTransition:
        if state.terminal:
            raise ValidationError("cannot fail a terminated tau-USI task")
        event = TraceEvent(
            turn=len(state.trace),
            actor=actor,
            kind="failure",
            content={"stage": stage, "kind": kind},
            visible_to=(),
            metadata={"visibility": "evaluator_only"},
        )
        state.trace.append(event)
        state.terminal = True
        state.terminal_reason = "failure"
        return EnvironmentTransition(state=state, events=(event,), terminal=True, terminal_reason=state.terminal_reason)


__all__ = [
    "ASSISTANT_ROLE",
    "ENVIRONMENT_ROLE",
    "INITIAL_ASSISTANT_MESSAGE",
    "TAU_USER_STOP_TOKEN",
    "USER_ROLE",
    "TauAssistantAction",
    "TauTaskEnvironment",
    "TauTaskState",
    "TauToolCall",
    "TauToolReplay",
    "TauToolSpec",
    "TauUserAction",
    "parse_assistant_response",
    "parse_user_action",
]
