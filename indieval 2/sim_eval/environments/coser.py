"""CoSER multi-character state and deterministic official-style context assembly."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts import BenchmarkCase, ChatMessage, TraceEvent
from ..errors import ConfigurationError, ContextLimitError, ParseError, ValidationError
from ..interfaces import EnvironmentTransition, InteractiveEnvironment
from ..json_utils import canonical_json, sha256_digest


ENVIRONMENT_ROLE = "environment"
END_SCENE = "<END CHAT>"
OFFICIAL_MAX_TURNS = 20
_TOKEN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_INNER_THOUGHT = re.compile(r"\[([^\[\]]*?)\]", flags=re.DOTALL)
_ACTION = re.compile(r"\(([^()]*)\)", flags=re.DOTALL)
_NEXT_SPEAKER = re.compile(
    r"(?:\*\*|#)?\s*Next\s+Speaker\s*(?:\*\*|#)?\s*(?:[:\-]|\bis\b)?\s*\*?([^\n\*]+)",
    flags=re.IGNORECASE,
)


def deterministic_tokens(text: str) -> tuple[str, ...]:
    """Dependency-free accounting tokens; these are not provider billing tokens."""

    return tuple(_TOKEN.findall(text))


def deterministic_token_count(text: str) -> int:
    return len(deterministic_tokens(text))


@dataclass(frozen=True)
class CoserMemory:
    memory_id: str
    text: str
    source_revision: str
    visibility: str = "role_private"

    def __post_init__(self) -> None:
        if not self.memory_id or not self.text or not self.source_revision:
            raise ValidationError("CoSER memory requires ID, text, and source revision")
        if self.visibility != "role_private":
            raise ValidationError("CoSER character memory must use role_private visibility")


@dataclass(frozen=True)
class CoserCharacter:
    character_id: str
    name: str
    profile: str
    goal: str
    private_context: Any
    memories: tuple[CoserMemory, ...] = ()


@dataclass(frozen=True)
class CoserRoleOutput:
    speech: str = ""
    action: str = ""
    inner_thought: str = ""

    @property
    def public_payload(self) -> dict[str, str]:
        return {"speech": self.speech, "action": self.action}


@dataclass(frozen=True)
class ContextSelection:
    messages: tuple[ChatMessage, ...]
    provenance: Mapping[str, Any]


@dataclass
class CoserState:
    benchmark_id: str
    scene_id: str
    story_identity: str
    plot: Any
    scene_context: str
    characters: Mapping[str, CoserCharacter]
    major_characters: tuple[str, ...]
    speaking_roles: tuple[str, ...]
    current_speaker: str
    max_turns: int
    min_end_turns: int
    max_context_tokens: int
    seed: int
    turn_count: int = 0
    transcript: list[TraceEvent] = field(default_factory=list)
    terminal: bool = False
    terminal_reason: str | None = None

    @property
    def next_actor(self) -> str | None:
        return None if self.terminal else self.current_speaker

    @property
    def public_transcript(self) -> tuple[TraceEvent, ...]:
        return tuple(event for event in self.transcript if event.metadata.get("visibility") == "public_transcript")


def parse_coser_role_output(text: str, *, allow_inner_thought: bool = True) -> CoserRoleOutput:
    """Parse the official free-text protocol, with strict legacy JSON replay support.

    Upstream CoSER marks private thoughts with ``[...]`` and public actions with
    ``(...)``.  Existing offline fixtures used an earlier JSON transport, so a
    valid JSON object remains accepted without being requested from live models.
    """

    parsed_json = False
    try:
        value = json.loads(text)
        parsed_json = True
    except json.JSONDecodeError:
        value = None
        if text.lstrip().startswith("{"):
            raise ParseError("malformed legacy CoSER JSON role output")
    if parsed_json:
        if not isinstance(value, Mapping):
            raise ParseError("legacy CoSER role output must be a JSON object")
        allowed = {"speech", "action", "inner_thought"}
        extra = set(value) - allowed
        if extra:
            raise ParseError(f"CoSER role output has unsupported fields: {sorted(extra)}")
        fields: dict[str, str] = {}
        for name in allowed:
            raw = value.get(name, "")
            if not isinstance(raw, str):
                raise ParseError(f"CoSER role output field {name!r} must be a string")
            fields[name] = raw.strip()
        if fields["inner_thought"] and not allow_inner_thought:
            raise ParseError("the environment role cannot emit private inner_thought")
        if not fields["speech"] and not fields["action"]:
            raise ParseError("empty_action: CoSER role output needs public speech or action")
        return CoserRoleOutput(**fields)

    stripped = text.strip()
    if not stripped:
        raise ParseError("empty_action: CoSER role output is empty")
    thoughts = [item.strip() for item in _INNER_THOUGHT.findall(stripped) if item.strip()]
    if thoughts and not allow_inner_thought:
        raise ParseError("the environment role cannot emit private inner_thought")
    without_thoughts = _INNER_THOUGHT.sub("", stripped)
    actions = [item.strip() for item in _ACTION.findall(without_thoughts) if item.strip()]
    speech = _ACTION.sub("", without_thoughts)
    speech = "\n".join(line.strip() for line in speech.splitlines() if line.strip()).strip()
    action = "\n".join(actions)
    inner_thought = "\n".join(thoughts)
    if not speech and not action:
        raise ParseError("empty_action: CoSER role output needs public speech or action")
    return CoserRoleOutput(speech=speech, action=action, inner_thought=inner_thought)


def parse_next_speaker(text: str) -> str:
    """Parse upstream ``Next Speaker: ...`` text plus legacy JSON fixtures."""

    parsed_json = False
    try:
        value = json.loads(text)
        parsed_json = True
    except json.JSONDecodeError:
        value = None
        if text.lstrip().startswith("{"):
            raise ParseError("malformed legacy CoSER next-speaker JSON output")
    if parsed_json:
        if not isinstance(value, Mapping) or set(value) != {"next_speaker"}:
            raise ParseError("legacy CoSER next-speaker output requires exactly the next_speaker field")
        speaker = value["next_speaker"]
        if not isinstance(speaker, str) or not speaker.strip():
            raise ParseError("CoSER next_speaker must be a non-empty string")
        return speaker.strip()
    matches = _NEXT_SPEAKER.findall(text)
    if matches:
        speaker = matches[-1].strip().rstrip('.,!"\'').replace("*", "").strip()
        if speaker:
            return speaker
    fallback = text.split(":", 1)[0].strip() if ":" in text else text.strip()
    if not fallback:
        raise ParseError("CoSER next-speaker output is empty")
    return fallback


def compose_coser_profile(plot: Any, character_name: str, global_profile: str) -> str:
    """Compose the plot-specific description and global profile as upstream does."""

    description = ""
    if isinstance(plot, Mapping):
        key_characters = plot.get("key_characters")
        if isinstance(key_characters, Sequence) and not isinstance(key_characters, (str, bytes)):
            matches = [
                item
                for item in key_characters
                if isinstance(item, Mapping) and str(item.get("name") or "") == character_name
            ]
            if matches:
                description = str(matches[0].get("description") or "").strip()
    return "\n\n".join(part for part in (description, global_profile.strip()) if part)


class CoserEnvironment(InteractiveEnvironment):
    """Multi-role GCA scheduler matching upstream profile visibility."""

    environment_revision = "coser-upstream-profile-visible-gca-v4-own-private-history"
    context_policy_revision = "official-fixed-20-turn-own-private-history-v4"
    tokenizer_revision = "unicode-regex-token-v1"
    allowed_actions = ("speech", "action", "inner_thought")
    randomize_turn_order = False

    def __init__(
        self,
        *,
        default_max_turns: int = 20,
        default_min_end_turns: int = 5,
        default_max_context_tokens: int = 4096,
        include_environment: bool = True,
    ) -> None:
        if default_max_turns <= 0 or default_min_end_turns < 0 or default_max_context_tokens <= 0:
            raise ConfigurationError("invalid CoSER environment defaults")
        self.default_max_turns = default_max_turns
        self.default_min_end_turns = default_min_end_turns
        self.default_max_context_tokens = default_max_context_tokens
        self.include_environment = include_environment

    @staticmethod
    def _parse_memories(raw: Any, *, character_id: str, source_revision: str) -> tuple[CoserMemory, ...]:
        if raw in (None, []):
            return ()
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValidationError(f"CoSER memories for {character_id!r} must be an array")
        memories: list[CoserMemory] = []
        for index, item in enumerate(raw):
            if isinstance(item, str):
                memory = CoserMemory(f"{character_id}-memory-{index}", item.strip(), source_revision)
            elif isinstance(item, Mapping):
                memory = CoserMemory(
                    memory_id=str(item.get("id") or f"{character_id}-memory-{index}"),
                    text=str(item.get("text") or "").strip(),
                    source_revision=str(item.get("source_revision") or source_revision),
                    visibility=str(item.get("visibility") or "role_private"),
                )
            else:
                raise ValidationError(f"CoSER memory #{index} for {character_id!r} must be text or an object")
            memories.append(memory)
        if len({memory.memory_id for memory in memories}) != len(memories):
            raise ValidationError(f"duplicate CoSER memory IDs for {character_id!r}")
        return tuple(memories)

    def reset(self, case: BenchmarkCase, *, seed: int) -> CoserState:
        raw_characters = case.input_data.get("characters")
        if not isinstance(raw_characters, Sequence) or isinstance(raw_characters, (str, bytes)):
            raise ValidationError("CoSER characters must be an array")
        if len(raw_characters) < 1:
            raise ValidationError("CoSER requires at least one character")
        characters: dict[str, CoserCharacter] = {}
        for index, raw in enumerate(raw_characters):
            if not isinstance(raw, Mapping):
                raise ValidationError("each CoSER character must be an object")
            name = str(raw.get("name") or "").strip()
            character_id = str(raw.get("id") or name or f"character_{index}").strip()
            if not name or not character_id:
                raise ValidationError("each CoSER character requires a name and stable ID")
            if character_id in characters or character_id == ENVIRONMENT_ROLE:
                raise ValidationError(f"duplicate or reserved CoSER character ID {character_id!r}")
            characters[character_id] = CoserCharacter(
                character_id=character_id,
                name=name,
                profile=str(raw.get("profile") or raw.get("traits") or ""),
                goal=str(raw.get("goal") or ""),
                private_context=raw.get("private_context"),
                memories=self._parse_memories(
                    raw.get("memories"), character_id=character_id, source_revision=case.source_revision
                ),
            )
        include_environment = bool(case.input_data.get("include_environment", self.include_environment))
        default_roles = [*characters, *([ENVIRONMENT_ROLE] if include_environment else [])]
        raw_role_order = case.input_data.get("speaking_role_order")
        if raw_role_order is None:
            roles = default_roles
        else:
            if not isinstance(raw_role_order, Sequence) or isinstance(raw_role_order, (str, bytes)):
                raise ValidationError("CoSER speaking_role_order must be an array")
            roles = [
                ENVIRONMENT_ROLE if str(role).casefold() == "environment" else str(role)
                for role in raw_role_order
            ]
            if len(roles) != len(set(roles)) or set(roles) != set(default_roles):
                raise ValidationError("CoSER speaking_role_order must contain every speaking role exactly once")
        if len(roles) < 2:
            raise ValidationError("CoSER requires at least two speaking roles including Environment")
        raw_major = case.input_data.get("major_characters")
        if raw_major is None:
            major_characters = tuple(character.name for character in characters.values())
        else:
            if not isinstance(raw_major, Sequence) or isinstance(raw_major, (str, bytes)):
                raise ValidationError("CoSER major_characters must be an array")
            major_characters = tuple(str(name).strip() for name in raw_major if str(name).strip())
            known_names = {character.name for character in characters.values()}
            if not major_characters or len(major_characters) != len(set(major_characters)):
                raise ValidationError("CoSER major_characters must contain unique nonempty names")
            if not set(major_characters).issubset(known_names):
                raise ValidationError("CoSER major_characters must be speaking character names")
        initial = str(case.input_data.get("initial_speaker") or roles[0])
        if initial not in roles:
            raise ValidationError(f"unknown CoSER initial_speaker {initial!r}")
        declared_max_turns = int(case.input_data.get("max_turns", self.default_max_turns))
        max_turns = OFFICIAL_MAX_TURNS
        min_end_turns = int(case.input_data.get("min_end_turns", self.default_min_end_turns))
        max_context_tokens = int(case.input_data.get("max_context_tokens", self.default_max_context_tokens))
        if declared_max_turns <= 0 or min_end_turns < 0 or min_end_turns > max_turns or max_context_tokens <= 0:
            raise ValidationError("invalid CoSER max_turns/min_end_turns/max_context_tokens")
        return CoserState(
            benchmark_id=case.benchmark_id,
            scene_id=case.case_id,
            story_identity=str(case.input_data.get("story_identity") or case.metadata.get("book") or "unknown"),
            plot=case.input_data.get("plot"),
            scene_context=str(case.input_data.get("scene_context") or ""),
            characters=characters,
            major_characters=major_characters,
            speaking_roles=tuple(roles),
            current_speaker=initial,
            max_turns=max_turns,
            min_end_turns=min_end_turns,
            max_context_tokens=max_context_tokens,
            seed=seed,
        )

    @staticmethod
    def _display_name(state: CoserState, actor: str) -> str:
        return "Environment" if actor == ENVIRONMENT_ROLE else state.characters[actor].name

    @staticmethod
    def _private_text(value: Any) -> str:
        if value in (None, ""):
            return ""
        return value.strip() if isinstance(value, str) else canonical_json(value)

    @staticmethod
    def effective_profile(state: CoserState, character: CoserCharacter) -> str:
        """Reproduce upstream's plot-description plus global-profile composition."""
        return compose_coser_profile(state.plot, character.name, character.profile)

    @staticmethod
    def _public_event_text(event: TraceEvent) -> str:
        speech = str(event.content.get("speech") or "").strip()
        action = str(event.content.get("action") or "").strip()
        return " ".join(part for part in (speech, f"({action})" if action else "") if part)

    def _system_payload(
        self,
        state: CoserState,
        actor: str,
        *,
        add_output_example: bool,
    ) -> Mapping[str, Any]:
        if actor == ENVIRONMENT_ROLE:
            return {
                "role": "environment_narrator",
                "story_identity": state.story_identity,
                "scene_context": state.scene_context,
                "participants": [
                    {"character_id": character.character_id, "name": character.name}
                    for character in state.characters.values()
                ],
                "prompt_protocol": "official_get_environment_prompt",
            }
        character = state.characters[actor]
        return {
            "role": "character",
            "story_identity": state.story_identity,
            "scene_context": state.scene_context,
            "your_character": {
                "character_id": character.character_id,
                "name": character.name,
                "profile": self.effective_profile(state, character),
                "motivation": self._private_text(character.private_context) or character.goal,
            },
            "other_character_profiles": {
                candidate.name: self.effective_profile(state, candidate)
                for candidate_id, candidate in state.characters.items()
                if candidate_id != actor
            },
            "visibility": f"role_private:{actor}",
            "prompt_protocol": "official_get_character_prompt_fixed_template",
            "add_output_example": add_output_example,
        }

    def _system_content(self, state: CoserState, actor: str, *, add_output_example: bool) -> str:
        if actor == ENVIRONMENT_ROLE:
            names = list(state.major_characters)
            return (
                "You are an environment model for a role-playing game. Your task is to provide the "
                "environmental feedback: Based on the characters' interactions, dialogues, and actions, "
                "describe the resulting changes in the environment. This includes:\n"
                "   - Physical changes in the setting\n"
                "   - Reactions of background characters or crowds\n"
                "   - Ambient sounds, weather changes, or atmospheric shifts\n"
                "   - Any other relevant environmental details\n\n"
                "Your descriptions should be vivid and help set the scene, but avoid dictating the actions "
                f"or dialogue of the main characters (including {names}).\n\n"
                "Important notes:\n"
                f"- You may include actions and reactions of minor characters or crowds, as long as they're not main characters (including {names}).\n"
                "- Keep your environmental descriptions concise but impactful, typically 1-3 sentences.\n"
                "- Respond to subtle cues in the characters' interactions to create a dynamic, reactive environment.\n"
                "- Your output should match the tone, setting, and cultural context of the scenario.\n\n"
                f"===The scenario is as follows===\n{state.scene_context}"
            )
        character = state.characters[actor]
        other_profiles = "\n\n".join(
            f"{other.name}: {self.effective_profile(state, other)}"
            for other_id, other in state.characters.items()
            if other_id != actor
        )
        motivation = self._private_text(character.private_context) or character.goal
        output_format = (
            "Your output should include **thought**, **speech**, and **action**. "
            "Use [your thought] for thoughts, which others can't see. "
            "Use (your action) for actions, which others can see."
        )
        if add_output_example:
            output_format = (
                "Your output should include **thought**, **speech**, and **action**. "
                "Use [your thought] for thoughts, which others can't see, e.g. "
                "[I'm terrified, but I must appear strong.]. Use (your action) for actions, "
                "which others can see, such as (watches silently, trying to control her fear and anger)."
            )
        sections = [
            f"You are {character.name} from {state.story_identity}.",
            f"==={character.name}'s Profile===\n{self.effective_profile(state, character)}",
            f"===Current Scenario===\n{state.scene_context}",
        ]
        if other_profiles:
            sections.append(f"===Information about the other Characters===\n{other_profiles}")
        if motivation:
            sections.append(f"===Your Inner Thoughts===\n{motivation}")
        sections.append(f"===Requirements===\n{output_format}")
        return "\n\n".join(sections) + "\n\n"

    def observation(self, state: CoserState, *, actor: str) -> Sequence[ChatMessage]:
        return self.assemble_context(state, actor=actor).messages

    def assemble_context(
        self,
        state: CoserState,
        *,
        actor: str,
        add_output_example: bool = True,
    ) -> ContextSelection:
        if state.terminal:
            raise ValidationError("cannot observe a terminated CoSER scene")
        if actor != state.next_actor:
            raise ValidationError(f"out-of-turn CoSER observation for {actor!r}; expected {state.next_actor!r}")
        if actor not in state.speaking_roles:
            raise ValidationError(f"unknown CoSER role {actor!r}")
        system_payload = self._system_payload(
            state,
            actor,
            add_output_example=add_output_example,
        )
        system_content = self._system_content(state, actor, add_output_example=add_output_example)
        selected_events: list[TraceEvent] = []
        selected_memories: list[CoserMemory] = []
        own_thoughts = {
            event.turn: str(event.content)
            for event in state.transcript
            if event.actor == actor and event.kind == "inner_thought"
            and actor in event.visible_to
        }

        def selected_messages() -> tuple[ChatMessage, ...]:
            messages: list[ChatMessage] = [
                ChatMessage("system", system_content, metadata={"visibility": f"role:{actor}"})
            ]
            if selected_memories:
                memory_text = "\n\n".join(memory.text for memory in selected_memories)
                messages.append(
                    ChatMessage(
                        "user",
                        f"===Relevant Background Information===\n{memory_text}",
                        metadata={"visibility": f"role:{actor}", "source": "declared_memory"},
                    )
                )
            messages.append(
                ChatMessage("user", "===Conversation Start===\n\n", metadata={"visibility": "public_transcript"})
            )
            for event in sorted(selected_events, key=lambda item: item.turn):
                content = self._public_event_text(event)
                if event.actor == actor:
                    thought = own_thoughts.get(event.turn)
                    if thought:
                        content = f"[{thought}] {content}".strip()
                    messages.append(
                        ChatMessage(
                            "assistant",
                            content,
                            metadata={"visibility": f"role_private:{actor}" if thought else "public_transcript",
                                      "turn": event.turn},
                        )
                    )
                else:
                    speaker = self._display_name(state, event.actor)
                    messages.append(
                        ChatMessage(
                            "user",
                            f"{speaker}: {content}",
                            metadata={"visibility": "public_transcript", "turn": event.turn},
                        )
                    )
            return tuple(messages)

        def selected_token_count() -> int:
            return sum(deterministic_token_count(message.content) for message in selected_messages())

        base_tokens = selected_token_count()
        if base_tokens > state.max_context_tokens:
            raise ContextLimitError(
                f"protected CoSER context needs {base_tokens} accounting tokens, budget is {state.max_context_tokens}; "
                "mandatory story/scene/role context was not truncated"
            )
        for event in reversed(state.public_transcript):
            selected_events.append(event)
            used = selected_token_count()
            if used > state.max_context_tokens:
                selected_events.pop()
        memories = state.characters[actor].memories if actor in state.characters else ()
        for memory in memories:
            selected_memories.append(memory)
            used = selected_token_count()
            if used > state.max_context_tokens:
                selected_memories.pop()
        final_messages = selected_messages()
        used_tokens = sum(deterministic_token_count(message.content) for message in final_messages)
        included_turns = [event.turn for event in sorted(selected_events, key=lambda x: x.turn)]
        all_turns = [event.turn for event in state.public_transcript]
        included_memory_ids = [memory.memory_id for memory in selected_memories]
        all_memory_ids = [memory.memory_id for memory in memories]
        provenance = {
            "policy_revision": self.context_policy_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "budget_tokens": state.max_context_tokens,
            "used_tokens": used_tokens,
            "provider_token_count": "unknown",
            "protected_context_digest": sha256_digest(
                {"system_payload": system_payload, "conversation_start": "===Conversation Start==="}
            ),
            "protected_context_truncated": False,
            "other_character_profiles_visible": actor != ENVIRONMENT_ROLE,
            "included_transcript_turns": included_turns,
            "included_own_private_thought_turns": [turn for turn in included_turns if turn in own_thoughts],
            "omitted_transcript_turns": [turn for turn in all_turns if turn not in set(included_turns)],
            "included_memory_ids": included_memory_ids,
            "omitted_memory_ids": [memory_id for memory_id in all_memory_ids if memory_id not in set(included_memory_ids)],
            "memory_sources": {
                memory.memory_id: memory.source_revision for memory in memories
            },
            "truncated": len(included_turns) != len(all_turns) or len(included_memory_ids) != len(all_memory_ids),
            "actor": actor,
        }
        return ContextSelection(
            messages=final_messages,
            provenance=provenance,
        )

    def apply(self, state: CoserState, *, actor: str, action: Any) -> EnvironmentTransition:
        if state.terminal:
            raise ValidationError("cannot act in a terminated CoSER scene")
        if actor != state.next_actor:
            raise ValidationError(f"out-of-turn CoSER action for {actor!r}; expected {state.next_actor!r}")
        if not isinstance(action, CoserRoleOutput):
            raise ValidationError("CoSER environment requires parsed CoserRoleOutput")
        public_event = TraceEvent(
            turn=state.turn_count,
            actor=actor,
            kind="environment" if actor == ENVIRONMENT_ROLE else "character",
            content=action.public_payload,
            visible_to=state.speaking_roles,
            metadata={"visibility": "public_transcript"},
        )
        events: list[TraceEvent] = [public_event]
        state.transcript.append(public_event)
        if action.inner_thought:
            thought = TraceEvent(
                turn=state.turn_count,
                actor=actor,
                kind="inner_thought",
                content=action.inner_thought,
                visible_to=(actor,),
                metadata={"visibility": f"role_private:{actor}"},
            )
            state.transcript.append(thought)
            events.append(thought)
        state.turn_count += 1
        if state.turn_count >= state.max_turns:
            state.terminal = True
            state.terminal_reason = "max_turns"
        return EnvironmentTransition(
            state=state,
            events=tuple(events),
            terminal=state.terminal,
            terminal_reason=state.terminal_reason,
            metadata={"turn_count": state.turn_count},
        )

    @staticmethod
    def _fallback_speaker(state: CoserState, previous: str) -> str:
        candidates = [role for role in state.speaking_roles if role != previous]
        if not candidates:
            raise ValidationError("CoSER scene has no fallback next speaker")
        ranked = sorted(
            candidates,
            key=lambda role: hashlib.sha256(
                f"{state.seed}\0{state.turn_count}\0{previous}\0{role}".encode("utf-8")
            ).hexdigest(),
        )
        return ranked[0]

    def choose_next(self, state: CoserState, *, raw_next_speaker: str) -> EnvironmentTransition:
        if state.terminal:
            raise ValidationError("cannot choose a next speaker for a terminated CoSER scene")
        previous = state.current_speaker
        normalized = raw_next_speaker.strip()
        if normalized.casefold() == "environment":
            resolved = ENVIRONMENT_ROLE
        else:
            matches = [
                character_id
                for character_id, character in state.characters.items()
                if normalized.casefold() in {character_id.casefold(), character.name.casefold()}
            ]
            resolved = matches[0] if len(matches) == 1 else normalized
        fallback = False
        requested_end_too_early = resolved == END_SCENE and state.turn_count < state.min_end_turns
        if resolved == END_SCENE and not requested_end_too_early:
            state.terminal = True
            state.terminal_reason = "nsp_end"
            chosen = END_SCENE
        elif resolved in state.speaking_roles and resolved != previous:
            chosen = resolved
            state.current_speaker = chosen
        else:
            fallback = True
            chosen = self._fallback_speaker(state, previous)
            state.current_speaker = chosen
        event = TraceEvent(
            turn=state.turn_count,
            actor="next_speaker_predictor",
            kind="next_speaker",
            content=chosen,
            visible_to=(),
            metadata={
                "visibility": "evaluator_only",
                "raw_next_speaker": raw_next_speaker,
                "fallback": fallback,
                "requested_end_too_early": requested_end_too_early,
            },
        )
        state.transcript.append(event)
        return EnvironmentTransition(
            state=state,
            events=(event,),
            terminal=state.terminal,
            terminal_reason=state.terminal_reason,
            metadata={"next_speaker": chosen, "fallback": fallback},
        )

    def fail(self, state: CoserState, *, actor: str, stage: str, failure_kind: str) -> EnvironmentTransition:
        if state.terminal:
            raise ValidationError("cannot fail a role in a terminated CoSER scene")
        event = TraceEvent(
            turn=state.turn_count,
            actor=actor,
            kind="role_failure",
            content={"stage": stage, "kind": failure_kind},
            visible_to=(),
            metadata={"visibility": "evaluator_only"},
        )
        state.transcript.append(event)
        state.terminal = True
        state.terminal_reason = "role_failure"
        return EnvironmentTransition(state=state, events=(event,), terminal=True, terminal_reason=state.terminal_reason)

    def terminate_capability_limit(
        self,
        state: CoserState,
        *,
        actor: str,
        reason: str,
        details: Mapping[str, Any],
    ) -> EnvironmentTransition:
        """Stop a partial scene without classifying it as a role/backend failure."""

        if state.terminal:
            raise ValidationError("cannot capability-terminate a completed CoSER scene")
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


__all__ = [
    "CoserCharacter",
    "CoserEnvironment",
    "CoserMemory",
    "CoserRoleOutput",
    "CoserState",
    "ContextSelection",
    "END_SCENE",
    "ENVIRONMENT_ROLE",
    "OFFICIAL_MAX_TURNS",
    "deterministic_token_count",
    "deterministic_tokens",
    "parse_coser_role_output",
    "parse_next_speaker",
]
