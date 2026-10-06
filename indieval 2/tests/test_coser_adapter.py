import json
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.coser import (
    COSER_DIMENSIONS,
    CoserAdapter,
    build_official_coser_critic_prompt,
    coser_length_corrected_score,
    dependency_free_bleu,
    dependency_free_rouge_l,
)
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.environments.coser import ENVIRONMENT_ROLE, parse_coser_role_output
from sim_eval.errors import ParseError


ROOT = Path(__file__).resolve().parents[1]


def metric(result, name):
    return next(item for item in result.metrics if item.name == name)


class CoserAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_fixture_suite(ROOT / "tests" / "fixtures")["coser"][0]

    def execute(self, case=None, responses=None):
        case = case or self.cases[0]
        adapter = CoserAdapter()
        responses = responses or adapter.replay_responses(case, seed=20260812)
        return adapter.execute_case(
            case,
            backend=ReplayBackend(responses),
            run_id="run",
            seed=20260812,
            model="offline-replay",
        )

    def test_three_character_multi_turn_replay_and_four_call_critic(self) -> None:
        result = self.execute()
        self.assertEqual(result.status.value, "completed")
        self.assertEqual(result.metadata["terminal_reason"], "nsp_end")
        self.assertEqual(result.metadata["turn_count"], 6)
        self.assertEqual(result.metadata["judge_call_count"], 4)
        self.assertEqual(set(result.metadata["judge_status"]), set(COSER_DIMENSIONS))
        self.assertTrue(all(value == "available" for value in result.metadata["judge_status"].values()))
        speakers = {item["speaker"] for item in result.prediction["public_dialogue"]}
        self.assertTrue({"mara", "ivo", "tess", ENVIRONMENT_ROLE}.issubset(speakers))
        request_kinds = [item["request_kind"] for item in result.metadata["context_audits"]]
        self.assertEqual(request_kinds.count("role"), 6)
        self.assertEqual(request_kinds.count("next_speaker"), 6)
        self.assertEqual(request_kinds.count("judge"), 4)
        self.assertNotIn("inner_thought", json.dumps(result.prediction))
        self.assertTrue(result.metadata["natural_termination"])
        self.assertFalse(result.metadata["hidden_thoughts_shared_with_critic"])

        for dimension in COSER_DIMENSIONS:
            self.assertIsNotNone(metric(result, f"coser.scene.{dimension}").value)
            for character in ("mara", "ivo", "tess"):
                item = metric(result, f"coser.character.{character}.{dimension}")
                self.assertIsNone(item.value)
                self.assertIn("not_produced_by_official_output", item.metadata["protocol_status"])
        self.assertIsNotNone(metric(result, "coser.scene.critic_average").value)
        self.assertEqual(metric(result, "coser.scene.bleu").value, 1.0)
        self.assertEqual(metric(result, "coser.scene.rouge_l").value, 1.0)

    def test_replay_is_deterministic(self) -> None:
        first = self.execute()
        second = self.execute()
        self.assertEqual(first.prediction, second.prediction)
        self.assertEqual(
            [(item.name, item.value) for item in first.metrics],
            [(item.name, item.value) for item in second.metrics],
        )
        self.assertEqual(first.metadata["context_audits"], second.metadata["context_audits"])

    def test_judge_input_removes_inner_thoughts_and_private_role_context(self) -> None:
        case = self.cases[0]
        adapter = CoserAdapter()
        runtime = adapter.runtime_for_case(case)
        provenance = adapter.provenance_for_case(case)
        state = adapter.environment.reset(case, seed=20260812)
        for step in case.metadata["replay"]["steps"]:
            output = parse_coser_role_output(
                json.dumps(step["output"]), allow_inner_thought=step["speaker"] != ENVIRONMENT_ROLE
            )
            adapter.environment.apply(state, actor=step["speaker"], action=output)
            if not state.terminal:
                adapter.environment.choose_next(state, raw_next_speaker=step["next_speaker"])
        request = adapter.build_judge_request(
            case, state, provenance, runtime, "character_fidelity", seed=20260812
        )
        system, dialogue = (message.content for message in request.messages)
        text = f"{system}\n{dialogue}"
        for step in case.metadata["replay"]["steps"]:
            thought = step["output"].get("inner_thought")
            if thought:
                self.assertNotIn(thought, text)
        for character in state.characters.values():
            self.assertNotIn(str(character.private_context), text)
            for memory in character.memories:
                self.assertNotIn(memory.text, text)
        self.assertTrue(system.startswith("You are a literary critic specializing in character analysis"))
        self.assertIn("## Original Conversation", system)
        self.assertIn("## Evaluation Criteria", system)
        self.assertIn("===Dialogue Content===", system)
        self.assertIn("mara:", dialogue)
        self.assertNotIn('"simulated_dialogue"', dialogue)
        self.assertEqual(request.metadata["critic_prompt_revision"], "coser-self-play-deduct-template-upstream-exact-v1")
        self.assertEqual(
            request.metadata["input_protocol"],
            "upstream_system_prompt_and_plain_simulation_user_message",
        )
        self.assertFalse(request.metadata["explicit_character_goals_included"])
        self.assertFalse(request.metadata["full_structured_plot_included"])
        self.assertFalse(request.metadata["context_provenance"]["hidden_thoughts_included"])
        self.assertEqual(request.response_format["type"], "json_schema")
        self.assertIn("Character Fidelity", request.response_format["schema"]["properties"])
        self.assertIn('"Character Fidelity":{"flaws":[]}', request.metadata["output_contract"])

    def test_critic_uses_plot_specific_profiles_and_declared_major_characters(self) -> None:
        case = self.cases[0]
        updated_plot = {
            **case.input_data["plot"],
            "key_characters": [
                {"name": "Mara", "description": "Plot-specific Mara description."},
                {"name": "Ivo", "description": "Plot-specific Ivo description."},
            ],
        }
        updated = replace(
            case,
            input_data={
                **case.input_data,
                "plot": updated_plot,
                "major_characters": ["Mara"],
            },
        )
        prompt = build_official_coser_critic_prompt(updated, "character_fidelity")
        self.assertIn(
            "### Mara\n\nPlot-specific Mara description.\n\n"
            "A patient apprentice clockmaker",
            prompt,
        )
        self.assertIn("including Mara.", prompt)
        self.assertIn("Only apply to the main characters: Mara", prompt)
        self.assertNotIn("Only apply to the main characters: Mara, Ivo", prompt)

    def test_reference_inner_thought_is_also_removed_before_critic(self) -> None:
        case = self.cases[0]
        case = replace(
            case,
            gold={
                **case.gold,
                "reference_dialogue": [
                    {
                        "speaker": "mara",
                        "content": "[reference-only secret] The public repair can begin.",
                    }
                ],
            },
        )
        adapter = CoserAdapter()
        state = adapter.environment.reset(case, seed=20260812)
        request = adapter.build_judge_request(
            case,
            state,
            adapter.provenance_for_case(case),
            adapter.runtime_for_case(case),
            "character_fidelity",
            seed=20260812,
        )
        text = "\n".join(message.content for message in request.messages)
        self.assertNotIn("reference-only secret", text)
        self.assertIn("The public repair can begin.", text)

    def test_one_judge_failure_is_unavailable_and_not_fabricated(self) -> None:
        case = self.cases[0]
        adapter = CoserAdapter()
        responses = dict(adapter.replay_responses(case, seed=20260812))
        responses[f"{case.case_id}:judge:anthropomorphism"] = "not-json"
        result = self.execute(case, responses)
        self.assertEqual(result.status.value, "completed")
        self.assertEqual(result.metadata["judge_status"]["anthropomorphism"], "unavailable")
        self.assertIsNone(metric(result, "coser.scene.anthropomorphism").value)
        self.assertIsNone(metric(result, "coser.scene.critic_average").value)
        self.assertIsNotNone(metric(result, "coser.scene.storyline_quality").value)
        self.assertEqual(result.metadata["judge_call_count"], 4)

    def test_character_backend_failure_is_structured_and_incomplete(self) -> None:
        case = self.cases[0]
        adapter = CoserAdapter()
        state = adapter.environment.reset(case, seed=7)
        request = adapter.build_role_request(
            case,
            state,
            state.current_speaker,
            model="fixture",
            runtime=adapter.runtime_for_case(case),
            seed=7,
        )
        self.assertEqual(request.temperature, 0.0)
        self.assertEqual(request.max_tokens, 2048)
        result = adapter.execute_case(
            case,
            backend=ReplayBackend(errors={request.request_id: "timeout"}),
            run_id="run",
            seed=7,
            model="fixture",
        )
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.error.stage, "character_backend")
        self.assertFalse(result.metadata["episode_complete"])
        self.assertEqual(result.trace[-1].kind, "role_failure")

    def test_empty_character_action_is_terminal_zero_score_capability_outcome(self) -> None:
        case = self.cases[0]
        adapter = CoserAdapter()
        state = adapter.environment.reset(case, seed=11)
        request = adapter.build_role_request(
            case,
            state,
            state.current_speaker,
            model="fixture",
            runtime=adapter.runtime_for_case(case),
            seed=11,
        )
        invalid = json.dumps(
            {"speech": "", "action": "", "inner_thought": "private text alone is not an action"}
        )
        result = adapter.execute_case(
            case,
            backend=ReplayBackend(
                responses={
                    request.request_id: invalid,
                },
                default_response=invalid,
            ),
            run_id="run",
            seed=11,
            model="fixture",
        )
        self.assertEqual(result.status.value, "completed")
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["target_output_failure"]["stage"], "character_parse")
        self.assertEqual(result.metadata["target_output_failure"]["kind"], "empty_action")
        self.assertTrue(result.metadata["episode_complete"])
        self.assertEqual(metric(result, "coser.scene.critic_average").value, 0)

    def test_next_speaker_contract_retry_can_recover(self) -> None:
        case = self.cases[0]
        adapter = CoserAdapter()
        responses = dict(adapter.replay_responses(case, seed=20260812))
        first_nsp = next(key for key in responses if ":nsp:" in key)
        valid = responses[first_nsp]
        responses[first_nsp] = '{"speaker":"wrong-field"}'
        responses[f"{first_nsp}:contract_retry:1"] = valid
        result = self.execute(case, responses)
        self.assertEqual(result.status.value, "completed")
        request = adapter.build_nsp_request(
            case,
            adapter.environment.reset(case, seed=20260812),
            adapter.runtime_for_case(case),
            seed=20260812,
        )
        self.assertIn("2. Next Speaker:", request.messages[0].content)
        self.assertIsNone(request.response_format)

    def test_critic_severity_validation_and_length_correction_boundaries(self) -> None:
        self.assertEqual(coser_length_corrected_score([], 0), 100.0)
        self.assertEqual(coser_length_corrected_score([{"severity": 5}] * 9, 0), 0.0)
        self.assertEqual(coser_length_corrected_score([{"severity": 2}], 1), 91.5)

        case = self.cases[0]
        adapter = CoserAdapter()
        state = adapter.environment.reset(case, seed=1)
        bad = {
            "dimension": "storyline_quality",
            "scene_flaws": [{"instance": "x", "type": "flow", "severity": 6}],
            "character_flaws": {character_id: [] for character_id in state.characters},
        }
        with self.assertRaises(ParseError):
            adapter.scorer.score_dimension(
                bad,
                dimension="storyline_quality",
                state=state,
                provenance=adapter.provenance_for_case(case),
            )

    def test_dependency_free_overlap_metric_boundaries(self) -> None:
        text = "mara repairs the brass tide clock before the ferry bell rings"
        self.assertEqual(dependency_free_bleu(text, text), 1.0)
        self.assertEqual(dependency_free_rouge_l(text, text), 1.0)
        self.assertEqual(dependency_free_bleu("alpha beta gamma delta", "one two three four"), 0.0)
        self.assertEqual(dependency_free_rouge_l("alpha beta", "one two"), 0.0)


if __name__ == "__main__":
    unittest.main()
