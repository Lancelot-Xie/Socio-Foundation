import json
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.alignx import ALIGNX_VARIANTS, AlignXAdapter
from sim_eval.benchmarks.behaviorchain import BehaviorChainAdapter, normalized_cumulative_chain_score
from sim_eval.benchmarks.choice import (
    deterministic_binary_assignment,
    indexed_choices,
    parse_strict_choice,
)
from sim_eval.benchmarks.humanllm import HumanLLMItemSelectionAdapter
from sim_eval.benchmarks.lifechoices import LifeChoicesAdapter
from sim_eval.contracts import BenchmarkCase, CaseResult, MetricValue, ModelResponse, ResultStatus
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.errors import BackendStructuredOutputError, ParseError, ValidationError
from sim_eval.interfaces import ModelBackend


ROOT = Path(__file__).resolve().parents[1]


def metric(result, name):
    return next(item for item in result.metrics if item.name == name)


def run_replay(adapter, case, *, seed=17, responses=None):
    replay = responses if responses is not None else adapter.replay_responses(case, seed=seed)
    return adapter.execute_case(
        case,
        backend=ReplayBackend(responses=replay),
        run_id="decision-test-run",
        seed=seed,
        model="fixture-model",
    )


class StrictChoiceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.choices = indexed_choices(
            ("first exact option", "second exact option", "third exact option"),
            labels=("A", "B", "C"),
            source_ids=("s0", "s1", "s2"),
        )

    def test_exact_label_and_small_json_are_accepted(self) -> None:
        self.assertEqual(parse_strict_choice("A", self.choices).source_id, "s0")
        prediction = parse_strict_choice(
            '{"choice":"B","ranking":["B","A"]}',
            self.choices,
            allow_ranking=True,
        )
        self.assertEqual(prediction.source_id, "s1")
        self.assertEqual(prediction.ranking_source_ids, ("s1", "s0"))

    def test_substring_duplicate_and_unknown_outputs_are_rejected(self) -> None:
        malformed = (
            ("A because it looks right", {}),
            ('{"choice":"A","ranking":["A","A"]}', {"allow_ranking": True}),
            ('{"choice":"D"}', {}),
            ('{"choice":"A","extra":true}', {}),
        )
        for text, kwargs in malformed:
            with self.subTest(text=text), self.assertRaises(ParseError):
                parse_strict_choice(text, self.choices, **kwargs)

    def test_binary_assignment_is_seeded_stable_and_gold_independent(self) -> None:
        first = deterministic_binary_assignment(
            "pair-1",
            seed=91,
            left_source_id="chosen",
            left_text="preferred",
            right_source_id="rejected",
            right_text="not preferred",
        )
        again = deterministic_binary_assignment(
            "pair-1",
            seed=91,
            left_source_id="chosen",
            left_text="preferred",
            right_source_id="rejected",
            right_text="not preferred",
        )
        self.assertEqual(first, again)
        self.assertEqual({item.source_id for item in first}, {"chosen", "rejected"})
        alternatives = {
            tuple(item.source_id for item in deterministic_binary_assignment(
                "pair-1",
                seed=seed,
                left_source_id="chosen",
                left_text="preferred",
                right_source_id="rejected",
                right_text="not preferred",
            ))
            for seed in range(32)
        }
        self.assertEqual(alternatives, {("chosen", "rejected"), ("rejected", "chosen")})


class LifeChoicesAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_fixture_suite(ROOT / "tests" / "fixtures")["lifechoices"][0]
        cls.adapter = LifeChoicesAdapter()

    def test_original_fixture_executes_and_preserves_source_option_identity(self) -> None:
        result = run_replay(self.adapter, self.cases[0])
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertEqual(metric(result, "lifechoices.accuracy").value, 1)
        choices = self.adapter.choices_for_case(self.cases[0], seed=999)
        self.assertEqual(choices[0].display_id, "A")
        self.assertFalse(choices[0].text.startswith("A."))
        self.assertFalse(result.metadata["lifechoices"]["gold_visible_to_model"])
        tagged = self.adapter.parse_choice_response(
            self.cases[0],
            ModelResponse(text="<think>brief</think><answer>A</answer>"),
            choices,
        )
        self.assertEqual(tagged.display_id, "A")

    def test_request_exposes_the_official_scenario_once(self) -> None:
        case = self.cases[0]
        request = self.adapter.build_request(case, model="fixture-model", seed=17)
        payload_text = request.messages[1].content
        self.assertIn("# Scenario\n" + case.input_data["decision_context"], payload_text)
        self.assertNotIn("decision_context", payload_text)
        self.assertEqual(payload_text.count(case.input_data["decision_context"]), 1)
        self.assertIn("<answer>X</answer>", payload_text)
        self.assertIsNone(request.response_format)

    def test_option_permutation_is_scored_by_updated_source_label(self) -> None:
        base = self.cases[0]
        options = list(base.input_data["options"])
        permuted = replace(
            base,
            input_data={**base.input_data, "options": [options[1], options[0], options[2], options[3]]},
            gold="B",
            metadata={**base.metadata, "replay": {"response": '{"choice":"B"}'}},
        )
        result = run_replay(self.adapter, permuted)
        self.assertEqual(metric(result, "lifechoices.accuracy").value, 1)

    def test_duplicate_options_and_invalid_gold_fail_validation(self) -> None:
        base = self.cases[0]
        options = list(base.input_data["options"])
        with self.assertRaisesRegex(ValidationError, "distinct"):
            self.adapter.validate_case(replace(base, input_data={**base.input_data, "options": [options[0]] * 4}))
        with self.assertRaisesRegex(ValidationError, "gold"):
            self.adapter.validate_case(replace(base, gold="E"))

    def test_parse_failure_is_structured_and_counts_as_incorrect(self) -> None:
        bad = replace(self.cases[0], metadata={**self.cases[0].metadata, "replay": {"response": "A because"}})
        failed = run_replay(self.adapter, bad)
        passed = run_replay(self.adapter, self.cases[1])
        self.assertEqual(failed.status, ResultStatus.COMPLETED)
        self.assertIsNone(failed.error)
        self.assertEqual(failed.metadata["target_output_failure"]["stage"], "response_parse")
        aggregate = self.adapter.aggregate((failed, passed))
        self.assertEqual(aggregate["lifechoices.accuracy"].value, 0.5)
        self.assertEqual(aggregate["lifechoices.accuracy"].denominator, 2)


class BehaviorChainAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_fixture_suite(ROOT / "tests" / "fixtures")["behaviorchain"][0]
        cls.prediction = [case for case in cls.cases if case.input_data["task_mode"] == "prediction"]
        cls.generation = [case for case in cls.cases if case.input_data["task_mode"] == "generation"]
        cls.adapter = BehaviorChainAdapter()

    def test_released_cumulative_formula_golden_values(self) -> None:
        self.assertAlmostEqual(normalized_cumulative_chain_score((1, 1, 0, 1)), 0.4)
        self.assertEqual(normalized_cumulative_chain_score((1, 1)), 1.0)
        self.assertEqual(normalized_cumulative_chain_score((0, 0)), 0.0)
        with self.assertRaises(ValidationError):
            normalized_cumulative_chain_score(())

    def test_complete_prediction_and_generation_chains_execute(self) -> None:
        results = [run_replay(self.adapter, case, seed=20 + index) for index, case in enumerate(self.cases)]
        self.assertTrue(all(result.status == ResultStatus.COMPLETED for result in results))
        aggregate = self.adapter.aggregate(results)
        self.assertEqual(aggregate["behaviorchain.prediction.avg_score"].value, 1.0)
        self.assertEqual(aggregate["behaviorchain.prediction.cum_score"].value, 1.0)
        self.assertEqual(aggregate["behaviorchain.generation.avg_score"].value, 1.0)
        self.assertEqual(aggregate["behaviorchain.generation.cum_score"].value, 1.0)
        self.assertEqual(aggregate["behaviorchain.chain_structure_complete_rate"].value, 1.0)

    def test_prediction_uses_supplemental_answer_tag_without_json_mode(self) -> None:
        case = self.prediction[0]
        request = self.adapter.build_request(case, model="fixture-model", seed=17)
        self.assertIsNone(request.response_format)
        self.assertIn("<answer>A</answer>", request.messages[-1].content)
        parsed = self.adapter.parse_response(case, ModelResponse(text="<answer>A</answer>"))
        self.assertEqual(parsed.display_id, "A")

    def test_target_parse_failure_is_zero_not_dropped_from_complete_chain(self) -> None:
        first = self.prediction[0]
        failed = run_replay(
            self.adapter,
            first,
            responses={f"{first.case_id}:choice": "A because this is best"},
        )
        passed = run_replay(self.adapter, self.prediction[1])
        aggregate = self.adapter.aggregate((failed, passed))
        self.assertEqual(failed.status, ResultStatus.COMPLETED)
        self.assertEqual(failed.metadata["target_output_failure"]["stage"], "response_parse")
        self.assertEqual(aggregate["behaviorchain.prediction.avg_score"].value, 0.5)
        self.assertAlmostEqual(aggregate["behaviorchain.prediction.cum_score"].value, 1 / 3)

    def test_backend_wrapped_target_json_violation_is_zero_not_failed(self) -> None:
        class StructuredOutputFailureBackend(ModelBackend):
            def generate(self, request):
                del request
                raise BackendStructuredOutputError("candidate emitted invalid JSON")

        case = self.generation[0]
        result = self.adapter.execute_case(
            case,
            backend=StructuredOutputFailureBackend(),
            run_id="run",
            seed=17,
            model="candidate",
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["target_output_failure"]["kind"], "parse_failure")
        self.assertEqual(metric(result, "behaviorchain.node_score").value, 0)

    def test_generation_judge_failure_is_unavailable_not_fabricated(self) -> None:
        case = self.generation[0]
        responses = dict(self.adapter.replay_responses(case, seed=8))
        responses[f"{case.case_id}:judge"] = "not-json"
        result = run_replay(self.adapter, case, seed=8, responses=responses)
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertIsNone(metric(result, "behaviorchain.node_score").value)
        self.assertTrue(result.metadata["behaviorchain"]["judge_error"])

    def test_missing_judge_configuration_is_explicitly_unavailable(self) -> None:
        base = self.generation[0]
        replay = dict(base.metadata["replay"])
        replay.pop("judge")
        case = replace(base, metadata={**base.metadata, "judge_provenance": None, "replay": replay})
        result = run_replay(self.adapter, case)
        self.assertIsNone(metric(result, "behaviorchain.generation.node_judge_score").value)
        self.assertEqual(result.metadata["behaviorchain"]["judge_error"]["kind"], "missing_configuration")

    def test_chain_gap_and_prediction_candidate_count_fail_validation(self) -> None:
        base = self.prediction[1]
        with self.assertRaisesRegex(ValidationError, "prior_nodes count"):
            self.adapter.validate_case(replace(base, input_data={**base.input_data, "prior_nodes": []}))
        with self.assertRaisesRegex(ValidationError, "requires exactly context and behavior"):
            self.adapter.validate_case(
                replace(base, input_data={**base.input_data, "prior_nodes": [{"context": "only context"}]})
            )
        with self.assertRaisesRegex(ValidationError, "four candidates"):
            self.adapter.validate_case(replace(base, input_data={**base.input_data, "candidates": ["a", "b", "c"]}))

    def test_chain_macro_is_not_node_micro(self) -> None:
        def result(group, index, length, value):
            return CaseResult(
                run_id="run",
                benchmark_id="behaviorchain",
                case_id=f"{group}-{index}",
                group_id=group,
                repetition=0,
                status=ResultStatus.COMPLETED,
                metrics=(MetricValue("behaviorchain.node_score", value),),
                metadata={
                    "behaviorchain": {
                        "chain_index": index,
                        "chain_length": length,
                        "task_mode": "prediction",
                        "key_behavior_status": "key",
                    }
                },
            )

        results = [result("short", 0, 1, 1)] + [result("long", index, 3, 0) for index in range(3)]
        aggregate = self.adapter.aggregate(results)
        self.assertEqual(aggregate["behaviorchain.prediction.avg_score"].value, 0.5)
        self.assertEqual(aggregate["behaviorchain.diagnostic.node_micro_score"].value, 0.25)

    def test_incomplete_chain_fails_closed(self) -> None:
        result = run_replay(self.adapter, self.prediction[0])
        aggregate = self.adapter.aggregate((result,))
        self.assertIsNone(aggregate["behaviorchain.prediction.avg_score"].value)
        self.assertEqual(aggregate["behaviorchain.chain_structure_complete_rate"].value, 0.0)


class AlignXAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_fixture_suite(ROOT / "tests" / "fixtures")["alignx"][0]
        cls.adapter = AlignXAdapter()

    def test_all_five_conditioning_variants_validate(self) -> None:
        base = self.cases[0]
        fields = {
            "Reddit_DEMO": {"demographic_information": "prefers short practical explanations"},
            "Reddit_PAIR": {"pairwise_feedback": [{"preferred": "examples", "rejected": "jargon"}]},
            "Reddit_UGC": {"user_generated_content": ["I repair old stools on weekends."]},
            "Reddit_arbitrary": {"persona_components": {"style": "concise", "topic": "repair"}},
            "Reddit_history16": {"history16": [f"interaction-{index}" for index in range(16)]},
        }
        for variant in ALIGNX_VARIANTS:
            values = {
                key: value
                for key, value in base.input_data.items()
                if key not in {"user_context", "demographic_information", "pairwise_feedback", "user_generated_content", "persona_components", "history16"}
            }
            values.update(fields[variant])
            values["variant"] = variant
            strata = {**base.metadata["strata"], "test_variant": variant}
            with self.subTest(variant=variant):
                self.adapter.validate_case(replace(base, input_data=values, metadata={**base.metadata, "strata": strata}))

    def test_reference_adjusted_margin_and_direct_choice_are_separate(self) -> None:
        result = run_replay(self.adapter, self.cases[0], seed=44)
        self.assertEqual(metric(result, "alignx.alignment_accuracy").value, 1)
        self.assertEqual(metric(result, "alignx.direct_choice_accuracy").value, 1)
        details = metric(result, "alignx.alignment_accuracy").metadata
        self.assertEqual(details["availability"], "available")
        self.assertTrue(details["official_primary"])
        self.assertFalse(metric(result, "alignx.direct_choice_accuracy").metadata["official_primary"])

    def test_scoring_request_has_separate_candidate_contract_without_gold(self) -> None:
        request = self.adapter.build_request(self.cases[0], model="fixture", seed=41)
        contract = request.metadata["candidate_scoring"]
        self.assertEqual(contract["contract_revision"], "alignx-candidate-sequence-logprob-v1")
        self.assertEqual(set(contract["candidate_sequences"]), {"A", "B"})
        self.assertNotIn("gold", json.dumps(contract).casefold())
        self.assertFalse(request.metadata["gold_visible"])
        self.assertIsNone(request.response_format)
        self.assertIn("<answer>A</answer>", request.messages[-1].content)

    def test_missing_sequence_scores_fail_official_metric_closed(self) -> None:
        case = self.cases[0]
        choices = self.adapter.choices_for_case(case, seed=3)
        chosen_label = next(item.display_id for item in choices if item.source_id == "chosen")
        response = ModelResponse(text=json.dumps({"choice": chosen_label}))
        prediction = self.adapter.parse_choice_response(case, response, choices)
        scored = self.adapter.score_response(case, prediction, response, seed=3)
        self.assertIsNone(next(item for item in scored if item.name == "alignx.alignment_accuracy").value)
        self.assertEqual(next(item for item in scored if item.name == "alignx.direct_choice_accuracy").value, 1)

    def test_one_unavailable_reference_score_makes_suite_metric_unavailable(self) -> None:
        good = run_replay(self.adapter, self.cases[0], seed=5)
        case = self.cases[1]
        choices = self.adapter.choices_for_case(case, seed=6)
        chosen_label = next(item.display_id for item in choices if item.source_id == "chosen")
        missing = run_replay(
            self.adapter,
            case,
            seed=6,
            responses={f"{case.case_id}:choice": json.dumps({"choice": chosen_label})},
        )
        aggregate = self.adapter.aggregate((good, missing))
        self.assertIsNone(aggregate["alignx.alignment_accuracy"].value)
        self.assertEqual(aggregate["alignx.alignment_score_availability_rate"].value, 0.5)

    def test_assignment_metadata_is_deterministic_and_gold_is_validated(self) -> None:
        case = self.cases[0]
        first = self.adapter.result_metadata(case, self.adapter.choices_for_case(case, seed=22), seed=22)
        second = self.adapter.result_metadata(case, self.adapter.choices_for_case(case, seed=22), seed=22)
        self.assertEqual(first, second)
        self.assertIn(first["alignx"]["chosen_position"], {"A", "B"})
        with self.assertRaisesRegex(ValidationError, "gold"):
            self.adapter.validate_case(replace(case, gold="rejected"))

    def test_history16_more_than_sixteen_is_rejected(self) -> None:
        base = self.cases[0]
        values = {**base.input_data, "variant": "Reddit_history16", "history16": [str(i) for i in range(17)]}
        values.pop("demographic_information", None)
        values.pop("user_context", None)
        with self.assertRaisesRegex(ValidationError, "more than 16"):
            self.adapter.validate_case(replace(base, input_data=values))

    def test_missing_prompt_or_variant_specific_conditioning_is_rejected(self) -> None:
        base = self.cases[0]
        with self.assertRaisesRegex(ValidationError, "prompt"):
            self.adapter.validate_case(replace(base, input_data={**base.input_data, "prompt": ""}))
        values = {
            key: value
            for key, value in base.input_data.items()
            if key != "demographic_information"
        }
        with self.assertRaisesRegex(ValidationError, "demographic_description"):
            self.adapter.validate_case(replace(base, input_data=values))


class HumanLLMItemSelectionAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_fixture_suite(ROOT / "tests" / "fixtures")["humanllm"][0]
        cls.adapter = HumanLLMItemSelectionAdapter()

    def test_twenty_candidate_fixture_uses_strict_item_identity(self) -> None:
        result = run_replay(self.adapter, self.cases[0])
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertEqual(metric(result, "humanllm.top1_accuracy").value, 1)
        self.assertEqual(result.prediction["source_id"], "carry-01")
        self.assertEqual(result.prediction["display_id"], "A")
        self.assertEqual(len(result.metadata["choice_contract"]["assignment"]), 20)
        request = self.adapter.build_request(self.cases[0], model="fixture-model", seed=17)
        self.assertIsNone(request.response_format)
        self.assertIn("<answer>X1,X2,X3,X4,X5</answer>", request.messages[-1].content)

    def test_official_top1_and_supplemental_rank5_remain_separate(self) -> None:
        results = [run_replay(self.adapter, case, seed=30 + index) for index, case in enumerate(self.cases)]
        self.assertEqual(metric(results[1], "humanllm.top1_accuracy").value, 0)
        self.assertEqual(metric(results[1], "humanllm.supplemental.hit_at_5").value, 1)
        self.assertEqual(metric(results[1], "humanllm.supplemental.reciprocal_rank").value, 0.5)
        self.assertEqual(metric(results[1], "humanllm.diagnostic.top5_accuracy").value, 1)
        self.assertEqual(metric(results[1], "humanllm.diagnostic.reciprocal_rank").value, 0.5)
        aggregate = self.adapter.aggregate(results)
        self.assertEqual(aggregate["humanllm.top1_accuracy"].value, 0.5)

    def test_candidate_count_and_duplicate_item_ids_are_rejected(self) -> None:
        base = self.cases[0]
        with self.assertRaisesRegex(ValidationError, "exactly 20"):
            self.adapter.validate_case(replace(base, input_data={**base.input_data, "candidates": base.input_data["candidates"][:19]}))
        candidates = [dict(item) for item in base.input_data["candidates"]]
        candidates[1]["item_id"] = candidates[0]["item_id"]
        with self.assertRaisesRegex(ValidationError, "IDs must be unique"):
            self.adapter.validate_case(replace(base, input_data={**base.input_data, "candidates": candidates}))

    def test_duplicate_exact_titles_are_ambiguous_but_labels_remain_valid(self) -> None:
        base = self.cases[0]
        candidates = [dict(item) for item in base.input_data["candidates"]]
        candidates[1]["title"] = candidates[0]["title"]
        case = replace(base, input_data={**base.input_data, "candidates": candidates})
        self.adapter.validate_case(case)
        choices = self.adapter.choices_for_case(case, seed=0)
        with self.assertRaisesRegex(ParseError, "ambiguous"):
            self.adapter.parse_choice_response(case, ModelResponse(text=candidates[0]["title"]), choices)
        parsed = self.adapter.parse_choice_response(
            case,
            ModelResponse(text='{"choice":"C01","ranking":["C01","C02","C03","C04","C05"]}'),
            choices,
        )
        self.assertEqual(parsed.source_id, candidates[0]["item_id"])
        tagged = self.adapter.parse_choice_response(
            case,
            ModelResponse(text="<think>brief</think><answer>A,B,C,D,E</answer>"),
            choices,
        )
        self.assertEqual(tagged.source_id, candidates[0]["item_id"])

    def test_duplicate_ranking_and_permissive_substring_are_rejected(self) -> None:
        case = self.cases[0]
        choices = self.adapter.choices_for_case(case, seed=0)
        for text in (
            '{"choice":"C01","ranking":["C01","C01"]}',
            '{"choice":"C01","ranking":["C01","C02"]}',
            '{"choice":"C01","ranking":["C01","C02","C03","C04","C05","C06"]}',
            "I would buy C01 because it is useful",
        ):
            with self.subTest(text=text), self.assertRaises(ParseError):
                self.adapter.parse_choice_response(case, ModelResponse(text=text), choices)

    def test_empty_profile_and_history_are_rejected(self) -> None:
        base = self.cases[0]
        with self.assertRaisesRegex(ValidationError, "profile or purchase history"):
            self.adapter.validate_case(replace(base, input_data={**base.input_data, "user_profile": "", "purchase_history": []}))

    def test_history_is_capped_at_official_thirty_item_context(self) -> None:
        base = self.cases[0]
        with self.assertRaisesRegex(ValidationError, "more than 30"):
            self.adapter.validate_case(
                replace(base, input_data={**base.input_data, "purchase_history": [f"item-{i}" for i in range(31)]})
            )


if __name__ == "__main__":
    unittest.main()
