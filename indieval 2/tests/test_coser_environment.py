import json
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.coser import CoserAdapter
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.environments.coser import (
    END_SCENE,
    ENVIRONMENT_ROLE,
    CoserRoleOutput,
    OFFICIAL_MAX_TURNS,
    deterministic_token_count,
    parse_coser_role_output,
    parse_next_speaker,
)
from sim_eval.errors import ParseError, ValidationError


ROOT = Path(__file__).resolve().parents[1]


class CoserEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = load_fixture_suite(ROOT / "tests" / "fixtures")["coser"][0][0]
        cls.adapter = CoserAdapter()

    def test_role_observations_include_other_profiles_but_exclude_other_private_state(self) -> None:
        state = self.adapter.environment.reset(self.case, seed=17)
        runtime = self.adapter.runtime_for_case(self.case)
        for actor, character in state.characters.items():
            state.current_speaker = actor
            request = self.adapter.build_role_request(
                self.case, state, actor, model="fixture", runtime=runtime, seed=17
            )
            self.assertIsNone(request.response_format)
            text = "\n".join(message.content for message in request.messages)
            self.assertIn(character.profile, text)
            self.assertIn(str(character.private_context), text)
            for memory in character.memories:
                self.assertIn(memory.text, text)
            for other_id, other in state.characters.items():
                if other_id == actor:
                    continue
                self.assertIn(other.profile, text)
                self.assertNotIn(str(other.private_context), text)
                for memory in other.memories:
                    self.assertNotIn(memory.text, text)
            provenance = request.metadata["context_provenance"]
            self.assertFalse(provenance["protected_context_truncated"])
            self.assertEqual(provenance["actor"], actor)
            self.assertTrue(provenance["other_character_profiles_visible"])
            self.assertEqual(
                provenance["included_memory_ids"], [memory.memory_id for memory in character.memories]
            )

        state.current_speaker = ENVIRONMENT_ROLE
        environment_request = self.adapter.build_role_request(
            self.case, state, ENVIRONMENT_ROLE, model="fixture", runtime=runtime, seed=17
        )
        environment_text = "\n".join(message.content for message in environment_request.messages)
        for character in state.characters.values():
            self.assertNotIn(str(character.private_context), environment_text)
            for memory in character.memories:
                self.assertNotIn(memory.text, environment_text)

    def test_inner_thought_is_private_and_never_enters_public_observation(self) -> None:
        state = self.adapter.environment.reset(self.case, seed=3)
        actor = state.current_speaker
        self.adapter.environment.apply(
            state,
            actor=actor,
            action=CoserRoleOutput(speech="Public line", action="opens a case", inner_thought="private-thought-needle"),
        )
        next_actor = next(candidate for candidate in state.speaking_roles if candidate != actor)
        self.adapter.environment.choose_next(state, raw_next_speaker=next_actor)
        messages = self.adapter.environment.observation(state, actor=next_actor)
        text = "\n".join(message.content for message in messages)
        self.assertIn("Public line", text)
        self.assertNotIn("private-thought-needle", text)
        thoughts = [event for event in state.transcript if event.kind == "inner_thought"]
        self.assertEqual(len(thoughts), 1)
        self.assertEqual(tuple(thoughts[0].visible_to), (actor,))

    def test_early_end_uses_deterministic_fallback_and_late_end_terminates(self) -> None:
        state = self.adapter.environment.reset(self.case, seed=91)
        actor = state.current_speaker
        self.adapter.environment.apply(state, actor=actor, action=CoserRoleOutput(speech="First"))
        transition = self.adapter.environment.choose_next(state, raw_next_speaker=END_SCENE)
        self.assertFalse(transition.terminal)
        self.assertTrue(transition.metadata["fallback"])
        first_fallback = state.current_speaker

        duplicate = self.adapter.environment.reset(self.case, seed=91)
        actor = duplicate.current_speaker
        self.adapter.environment.apply(duplicate, actor=actor, action=CoserRoleOutput(speech="First"))
        self.adapter.environment.choose_next(duplicate, raw_next_speaker=END_SCENE)
        self.assertEqual(duplicate.current_speaker, first_fallback)

        state.turn_count = state.min_end_turns
        transition = self.adapter.environment.choose_next(state, raw_next_speaker=END_SCENE)
        self.assertTrue(transition.terminal)
        self.assertEqual(transition.terminal_reason, "nsp_end")

    def test_source_declared_shorter_turn_limit_does_not_override_official_twenty(self) -> None:
        limited = replace(
            self.case,
            input_data={**self.case.input_data, "max_turns": 2, "min_end_turns": 2},
        )
        state = self.adapter.environment.reset(limited, seed=8)
        self.assertEqual(state.max_turns, OFFICIAL_MAX_TURNS)

    def test_single_character_scene_uses_fixed_environment_role(self) -> None:
        first_character = dict(self.case.input_data["characters"][0])
        singleton = replace(
            self.case,
            input_data={
                **self.case.input_data,
                "characters": [first_character],
                "major_characters": [first_character["name"]],
                "speaking_role_order": [first_character["id"], ENVIRONMENT_ROLE],
                "include_environment": True,
                "initial_speaker": first_character["id"],
            },
        )
        self.adapter.validate_case(singleton)
        state = self.adapter.environment.reset(singleton, seed=8)
        self.assertEqual(tuple(state.characters), (first_character["id"],))
        self.assertEqual(state.speaking_roles, (first_character["id"], ENVIRONMENT_ROLE))
        self.assertEqual(state.major_characters, (first_character["name"],))

    def test_source_declared_turn_limit_is_capped_at_official_twenty(self) -> None:
        oversized = replace(
            self.case,
            input_data={**self.case.input_data, "max_turns": 32, "min_end_turns": 5},
        )
        state = self.adapter.environment.reset(oversized, seed=8)
        self.assertEqual(OFFICIAL_MAX_TURNS, 20)
        self.assertEqual(state.max_turns, OFFICIAL_MAX_TURNS)

    def test_context_truncation_is_deterministic_and_protected_context_remains(self) -> None:
        characters = []
        for raw in self.case.input_data["characters"]:
            value = dict(raw)
            if value["id"] == "mara":
                value["memories"] = [
                    {
                        "id": f"long-{index}",
                        "text": (f"memory segment {index} " * 18).strip(),
                        "source_revision": "synthetic-long-v1",
                    }
                    for index in range(12)
                ]
            characters.append(value)
        case = replace(
            self.case,
            input_data={**self.case.input_data, "characters": characters, "max_context_tokens": 350},
        )
        state = self.adapter.environment.reset(case, seed=5)
        runtime = self.adapter.runtime_for_case(case)
        first = self.adapter.build_role_request(case, state, "mara", model="fixture", runtime=runtime, seed=5)
        second = self.adapter.build_role_request(case, state, "mara", model="fixture", runtime=runtime, seed=5)
        self.assertEqual(first.metadata["context_provenance"], second.metadata["context_provenance"])
        provenance = first.metadata["context_provenance"]
        self.assertTrue(provenance["truncated"])
        self.assertTrue(provenance["omitted_memory_ids"])
        self.assertFalse(provenance["protected_context_truncated"])
        self.assertLessEqual(provenance["used_tokens"], provenance["budget_tokens"])
        text = "\n".join(message.content for message in first.messages)
        self.assertIn(str(state.characters["mara"].private_context), text)
        self.assertEqual(
            sum(deterministic_token_count(message.content) for message in first.messages),
            provenance["used_tokens"],
        )

    def test_protected_context_over_budget_is_failed_not_silently_truncated(self) -> None:
        case = replace(self.case, input_data={**self.case.input_data, "max_context_tokens": 1})
        result = self.adapter.execute_case(
            case,
            backend=ReplayBackend(default_response="{}"),
            run_id="run",
            seed=2,
            model="fixture",
        )
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.error.stage, "context_assembly")
        self.assertEqual(result.error.kind, "context_limit")
        self.assertFalse(result.metadata["episode_complete"])

    def test_strict_role_and_next_speaker_parsers(self) -> None:
        official = parse_coser_role_output("[I should stay calm.] Hello there. (opens the door)")
        self.assertEqual(official.inner_thought, "I should stay calm.")
        self.assertEqual(official.speech, "Hello there.")
        self.assertEqual(official.action, "opens the door")
        self.assertEqual(
            parse_next_speaker("1. Reasoning: Mara was addressed.\n2. Next Speaker: Mara"),
            "Mara",
        )
        with self.assertRaises(ParseError):
            parse_coser_role_output('{"speech":"","action":"","inner_thought":"only private"}')
        with self.assertRaises(ParseError):
            parse_coser_role_output("null")
        with self.assertRaises(ParseError):
            parse_coser_role_output('{"speech":"hello","unknown":"x"}')
        with self.assertRaises(ParseError):
            parse_coser_role_output('{"speech":"narration","inner_thought":"secret"}', allow_inner_thought=False)
        with self.assertRaises(ParseError):
            parse_next_speaker('{"next_speaker":"mara","reason":"extra"}')
        with self.assertRaises(ParseError):
            parse_next_speaker("null")
        with self.assertRaises(ValidationError):
            state = self.adapter.environment.reset(self.case, seed=1)
            wrong = next(role for role in state.speaking_roles if role != state.current_speaker)
            self.adapter.environment.apply(state, actor=wrong, action=CoserRoleOutput(speech="wrong"))


if __name__ == "__main__":
    unittest.main()
