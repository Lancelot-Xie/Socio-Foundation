import json
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.fantom import (
    FANTOM_SEMANTIC_CONTRACT,
    FANTOM_SEMANTIC_MODEL,
    FANTOM_SET_CONTRACT,
    FantomAdapter,
    belief_whitespace_token_f1,
    weighted_binary_f1,
    whitespace_token_f1,
)
from sim_eval.benchmarks.social_r1 import (
    ATOMS_DIMENSIONS,
    SOCIAL_R1_ALIASES,
    SocialR1Adapter,
    parse_social_r1_response,
)
from sim_eval.contracts import ErrorState, ModelResponse, ResultStatus
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.errors import ParseError, ValidationError
from sim_eval.registry import registered_adapters


ROOT = Path(__file__).resolve().parents[1]


def metric(result, name):
    return next(item for item in result.metrics if item.name == name)


def run_replay(adapter, case, *, seed=17, responses=None):
    replay = responses if responses is not None else adapter.replay_responses(case, seed=seed)
    return adapter.execute_case(
        case,
        backend=ReplayBackend(responses=replay),
        run_id="reasoning-test-run",
        seed=seed,
        model="fixture-model",
    )


class ReasoningRegistryTests(unittest.TestCase):
    def test_reasoning_adapters_are_registered(self) -> None:
        self.assertTrue({"fantom", "social_r1"} <= set(registered_adapters()))


class FantomAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_fixture_suite(ROOT / "tests" / "fixtures")["fantom"][0]
        cls.adapter = FantomAdapter()
        cls.main = [
            case
            for case in cls.cases
            if case.input_data["scenario"] in {"inaccessible", "fact"}
            and case.input_data["conversation_id"] == "m1"
        ]
        cls.control = [case for case in cls.cases if case.input_data["conversation_id"] == "c1"]

    @staticmethod
    def by_shape(cases, family, answer_format):
        return next(
            case
            for case in cases
            if case.input_data["question_family"] == family
            and case.input_data["answer_format"] == answer_format
        )

    def test_original_fixture_executes_official_main_and_accessible_control_separately(self) -> None:
        results = [run_replay(self.adapter, case, seed=31) for case in self.cases]
        self.assertTrue(all(result.status == ResultStatus.COMPLETED for result in results))
        aggregate = self.adapter.aggregate(results)
        self.assertEqual(aggregate["fantom.all_star"].value, 1.0)
        self.assertEqual(aggregate["fantom.all"].value, 1.0)
        self.assertEqual(aggregate["fantom.short.inaccessible.all_star"].denominator, 1)
        self.assertEqual(aggregate["fantom.short.accessible.all_star"].value, 1.0)
        self.assertTrue(aggregate["fantom.short.accessible.all_star"].metadata["control_task"])
        self.assertEqual(aggregate["fantom.short.fact_token_f1"].value, 1.0)

    def test_official_token_and_binary_f1_golden_values(self) -> None:
        self.assertAlmostEqual(whitespace_token_f1("a a b", "a b b"), 2 / 3)
        self.assertEqual(whitespace_token_f1("alpha", "beta"), 0.0)
        self.assertEqual(belief_whitespace_token_f1("Pax Knows", "pax Knows"), 0.5)
        self.assertAlmostEqual(weighted_binary_f1((1, 1, 0, 0), (1, 0, 0, -1)), 7 / 12)
        self.assertIsNone(weighted_binary_f1((), ()))
        with self.assertRaises(ValidationError):
            weighted_binary_f1((1,), ())

    def test_prompt_hides_source_gold_fields_and_mc_identity(self) -> None:
        free = self.by_shape(self.main, "belief", "free_text")
        request = self.adapter.build_request(free, model="fixture", seed=9)
        payload = request.messages[-1].content
        self.assertIn("# Context\n", payload)
        self.assertIn("# Question\n" + free.input_data["question"], payload)
        self.assertNotIn("# Options", payload)
        self.assertNotIn("# Target Fact", payload)
        self.assertFalse(request.metadata["gold_visible"])
        serialized = json.dumps(request.metadata)
        self.assertNotIn("wrong_answer", serialized)
        self.assertNotIn("omniscient_or_wrong", serialized)

        choice = self.by_shape(self.main, "belief", "multiple_choice")
        choice_request = self.adapter.build_request(choice, model="fixture", seed=9)
        for option in self.adapter.choices_for_case(choice, seed=9):
            self.assertIn(f"{option.display_id}. {option.text}", choice_request.messages[-1].content)
        self.assertNotIn("source_id", choice_request.messages[-1].content)
        self.assertIsNone(choice_request.response_format)
        self.assertIn("<answer>A</answer>", choice_request.messages[-1].content)

    def test_belief_choice_assignment_is_seeded_stable_and_uses_both_orders(self) -> None:
        case = self.by_shape(self.main, "belief", "multiple_choice")
        first = self.adapter.choices_for_case(case, seed=47)
        self.assertEqual(first, self.adapter.choices_for_case(case, seed=47))
        assignments = {
            tuple(choice.source_id for choice in self.adapter.choices_for_case(case, seed=seed))
            for seed in range(64)
        }
        self.assertEqual(
            assignments,
            {("correct", "omniscient_or_wrong"), ("omniscient_or_wrong", "correct")},
        )

    def test_list_and_binary_parsers_accept_official_answer_tags_and_legacy_json(self) -> None:
        list_case = self.by_shape(self.main, "answerability", "list")
        parsed = self.adapter.parse_choice_response(
            list_case,
            ModelResponse(text='{"characters":["uma","LIN"]}'),
            (),
        )
        self.assertEqual(parsed.value, ("Uma", "Lin"))
        tagged = self.adapter.parse_choice_response(
            list_case,
            ModelResponse(text="<answer>Lin and Uma</answer>"),
            (),
        )
        self.assertEqual(set(tagged.value), {"Uma", "Lin"})
        for response in (
            '{"characters":["Lin","Lin"]}',
            '{"characters":["Lin","Visitor"]}',
            '{"characters":["Lin"],"extra":true}',
            "<answer></answer>",
        ):
            with self.subTest(response=response), self.assertRaises(ParseError):
                self.adapter.parse_choice_response(list_case, ModelResponse(text=response), ())

        binary_case = self.by_shape(self.main, "answerability", "binary")
        for response in ('{"answer":"yes"}', "NO", "true", "false"):
            self.adapter.parse_choice_response(binary_case, ModelResponse(text=response), ())
        for response in ("", "yes because", "maybe", '{"answer":"yes","extra":1}', '{"answer":"no:long"}'):
            with self.subTest(response=response), self.assertRaises(ParseError):
                self.adapter.parse_choice_response(binary_case, ModelResponse(text=response), ())

    def test_missing_or_malformed_semantic_evidence_is_unavailable_not_fabricated(self) -> None:
        belief = self.by_shape(self.main, "belief", "free_text")
        missing = run_replay(
            self.adapter,
            belief,
            responses={f"{belief.case_id}:question": str(belief.gold)},
        )
        self.assertEqual(missing.status, ResultStatus.COMPLETED)
        self.assertIsNone(metric(missing, "fantom.belief_distance_accuracy").value)
        self.assertEqual(
            metric(missing, "fantom.belief_distance_accuracy").metadata["availability"],
            "unavailable",
        )

        malformed_response = {
            "text": str(belief.gold),
            "raw": {
                "fantom_belief_distance": {
                    "contract_revision": FANTOM_SEMANTIC_CONTRACT,
                    "model_id": FANTOM_SEMANTIC_MODEL,
                    "metric": "cosine_similarity",
                    "correct_similarity": 0.9,
                    "wrong_similarity": 0.1,
                }
            },
        }
        malformed = run_replay(
            self.adapter,
            belief,
            responses={f"{belief.case_id}:question": malformed_response},
        )
        self.assertIsNone(metric(malformed, "fantom.belief_distance_accuracy").value)
        self.assertIn("model revision", metric(malformed, "fantom.belief_distance_accuracy").metadata["reason"])

        other_results = [run_replay(self.adapter, case) for case in self.main if case.case_id != belief.case_id]
        aggregate = self.adapter.aggregate((missing, *other_results))
        self.assertIsNone(aggregate["fantom.all_star"].value)
        self.assertEqual(aggregate["fantom.all"].value, 1.0)
        self.assertEqual(aggregate["fantom.short.inaccessible.belief_distance_evidence_availability"].value, 0.0)

    def test_semantic_tie_is_incorrect_and_token_f1_stays_unavailable(self) -> None:
        belief = self.by_shape(self.main, "belief", "free_text")
        response = {
            "text": str(belief.gold),
            "raw": {
                "fantom_belief_distance": {
                    "contract_revision": FANTOM_SEMANTIC_CONTRACT,
                    "model_id": FANTOM_SEMANTIC_MODEL,
                    "model_revision": "fixture-tie-v1",
                    "metric": "cosine_similarity",
                    "correct_similarity": 0.5,
                    "wrong_similarity": 0.5,
                }
            },
        }
        tied = run_replay(
            self.adapter,
            belief,
            responses={f"{belief.case_id}:question": response},
        )
        self.assertEqual(metric(tied, "fantom.belief_distance_accuracy").value, 0)
        self.assertIsNone(metric(tied, "fantom.belief_token_f1").value)
        others = [run_replay(self.adapter, case) for case in self.main if case.case_id != belief.case_id]
        aggregate = self.adapter.aggregate((tied, *others))
        self.assertEqual(aggregate["fantom.all_star"].value, 0.0)
        self.assertEqual(aggregate["fantom.all"].value, 1.0)

    def test_incomplete_set_fails_closed(self) -> None:
        omitted = self.by_shape(self.main, "information_access", "list")
        results = [run_replay(self.adapter, case) for case in self.main if case.case_id != omitted.case_id]
        aggregate = self.adapter.aggregate(results)
        self.assertIsNone(aggregate["fantom.all_star"].value)
        self.assertIsNone(aggregate["fantom.all"].value)
        self.assertEqual(aggregate["fantom.short.inaccessible.complete_set_rate"].value, 0.0)
        self.assertEqual(
            aggregate["fantom.short.inaccessible.all_star"].metadata["incomplete_group_count"],
            1,
        )

    def test_member_contract_detects_one_missing_binary_even_when_type_remains(self) -> None:
        members = [
            {
                "id": case.metadata["source_id"],
                "question_family": case.input_data["question_family"],
                "answer_format": case.input_data["answer_format"],
                "scenario": case.input_data["scenario"],
                "short_no_long_excluded": False,
            }
            for case in self.main
        ]
        contract = {
            "revision": FANTOM_SET_CONTRACT,
            "set_id": "m1-p1-s1",
            "context_condition": "short",
            "fully_accessible": False,
            "members": members,
        }
        contracted = [
            replace(
                case,
                input_data={
                    **case.input_data,
                    "question_member_id": case.metadata["source_id"],
                    "set_contract": contract,
                },
            )
            for case in self.main
        ]
        omitted = next(
            case
            for case in contracted
            if case.input_data["question_family"] == "answerability"
            and case.input_data["answer_format"] == "binary"
        )
        results = [
            run_replay(self.adapter, case)
            for case in contracted
            if case.case_id != omitted.case_id
        ]
        aggregate = self.adapter.aggregate(results)
        self.assertIsNone(aggregate["fantom.all_star"].value)
        self.assertIsNone(aggregate["fantom.all"].value)
        self.assertEqual(
            aggregate["fantom.short.inaccessible.all_star"].metadata["incomplete_group_count"],
            1,
        )

    def test_mixed_set_scores_exact_official_scenario_members(self) -> None:
        remapped = []
        for case in self.main:
            scenario = (
                "accessible"
                if case.input_data["question_family"] in {"answerability", "information_access"}
                else case.input_data["scenario"]
            )
            remapped.append(replace(case, input_data={**case.input_data, "scenario": scenario}))
        members = [
            {
                "id": case.metadata["source_id"],
                "question_family": case.input_data["question_family"],
                "answer_format": case.input_data["answer_format"],
                "scenario": case.input_data["scenario"],
                "short_no_long_excluded": False,
            }
            for case in remapped
        ]
        contract = {
            "revision": FANTOM_SET_CONTRACT,
            "set_id": "m1-p1-s1",
            "context_condition": "short",
            "fully_accessible": False,
            "members": members,
        }
        contracted = [
            replace(
                case,
                input_data={
                    **case.input_data,
                    "question_member_id": case.metadata["source_id"],
                    "set_contract": contract,
                },
            )
            for case in remapped
        ]
        results = [run_replay(self.adapter, case) for case in contracted]
        aggregate = self.adapter.aggregate(results)
        self.assertEqual(aggregate["fantom.short.inaccessible.all_star"].value, 1.0)
        self.assertEqual(aggregate["fantom.short.inaccessible.all"].value, 1.0)
        self.assertIsNone(aggregate["fantom.short.accessible.all_star"].value)

    def test_target_parse_failure_is_zero_and_remains_in_set_denominator(self) -> None:
        target = self.by_shape(self.main, "answerability", "list")
        failed = run_replay(
            self.adapter,
            target,
            responses={f"{target.case_id}:question": "<answer></answer>"},
        )
        others = [run_replay(self.adapter, case) for case in self.main if case.case_id != target.case_id]
        aggregate = self.adapter.aggregate((failed, *others))
        self.assertEqual(failed.status, ResultStatus.COMPLETED)
        self.assertEqual(failed.metadata["target_output_failure"]["stage"], "response_parse")
        self.assertEqual(aggregate["fantom.all_star"].value, 0.0)
        self.assertEqual(aggregate["fantom.all_star"].denominator, 1)
        self.assertEqual(aggregate["fantom.short.inaccessible.answerability_list_accuracy"].value, 0.0)

    def test_binary_parse_failures_never_receive_negative_label_credit(self) -> None:
        for family in ("answerability", "information_access"):
            base = self.by_shape(self.main, family, "binary")
            name = f"fantom.short.inaccessible.{family}_binary_weighted_f1"
            for gold in ("yes", "no"):
                case = replace(base, gold=gold)
                for text in ("<answer></answer>", "I cannot determine the answer."):
                    with self.subTest(family=family, gold=gold, text=text):
                        result = run_replay(self.adapter, case, responses={f"{case.case_id}:question": text})
                        self.assertEqual(result.status, ResultStatus.COMPLETED)
                        self.assertIn("target_output_failure", result.metadata)
                        aggregate = self.adapter.aggregate([result])
                        self.assertEqual(aggregate[name].value, 0.0)
                        self.assertEqual(aggregate[name].denominator, 1)
                        self.assertEqual(metric(result, "fantom.item_correct").value, 0.0)

    def test_binary_f1_preserves_valid_predictions_and_failure_denominators(self) -> None:
        for family in ("answerability", "information_access"):
            base = self.by_shape(self.main, family, "binary")
            name = f"fantom.short.inaccessible.{family}_binary_weighted_f1"
            results = []
            for index, (gold, answer) in enumerate((("yes", "yes"), ("yes", "no"), ("no", "no"), ("no", ""))):
                case = replace(base, case_id=f"{base.case_id}-{index}", gold=gold)
                result = run_replay(self.adapter, case, responses={f"{case.case_id}:question": f"<answer>{answer}</answer>"})
                results.append(result)
                self.assertEqual(self.adapter.aggregate([result])[name].value, float(gold == answer))
            aggregate = self.adapter.aggregate(results)
            self.assertAlmostEqual(aggregate[name].value, 7 / 12)
            self.assertEqual(aggregate[name].denominator, 4)
            valid_no = results[2]
            invalid_variants = (
                replace(valid_no, status=ResultStatus.FAILED, error=ErrorState(stage="backend", kind="test_failure", message="legacy failure")),
                replace(valid_no, prediction=None),
                replace(valid_no, prediction={"parsed": False}),
                replace(valid_no, prediction={"value": "unknown"}),
                replace(valid_no, metadata={**valid_no.metadata, "target_output_failure": {"stage": "response_parse"}}),
            )
            for result in invalid_variants:
                with self.subTest(family=family, prediction=result.prediction, status=result.status):
                    self.assertEqual(self.adapter.aggregate([result])[name].value, 0.0)
                    self.assertEqual(self.adapter.aggregate([result])[name].denominator, 1)

    def test_short_no_long_binary_row_is_excluded_from_short_metrics(self) -> None:
        target = next(
            case
            for case in self.main
            if case.input_data["question_family"] == "answerability"
            and case.input_data["answer_format"] == "binary"
            and str(case.gold).casefold() == "no"
        )
        no_long = replace(
            target,
            gold="no:long",
            metadata={**target.metadata, "replay": {"answer": "no"}},
        )
        result = run_replay(self.adapter, no_long)
        self.assertTrue(result.metadata["fantom"]["short_no_long_excluded"])
        results = [
            result if case.case_id == target.case_id else run_replay(self.adapter, case)
            for case in self.main
        ]
        aggregate = self.adapter.aggregate(results)
        self.assertEqual(
            aggregate["fantom.short.inaccessible.answerability_binary_weighted_f1"].denominator,
            2,
        )
        self.assertEqual(aggregate["fantom.all_star"].value, 1.0)

    def test_mixed_context_conditions_keep_slices_but_disable_one_top_level_score(self) -> None:
        short_results = [run_replay(self.adapter, case) for case in self.main]
        full_results = []
        for result in short_results:
            info = dict(result.metadata["fantom"])
            info["context_condition"] = "full"
            info["set_id"] = f"full-{info['set_id']}"
            full_results.append(
                replace(
                    result,
                    case_id=f"full-{result.case_id}",
                    metadata={**result.metadata, "fantom": info},
                )
            )
        aggregate = self.adapter.aggregate((*short_results, *full_results))
        self.assertIsNone(aggregate["fantom.all_star"].value)
        self.assertEqual(
            aggregate["fantom.all_star"].metadata["availability"],
            "unavailable_mixed_or_missing_context_condition",
        )
        self.assertEqual(aggregate["fantom.short.inaccessible.all_star"].value, 1.0)
        self.assertEqual(aggregate["fantom.full.inaccessible.all_star"].value, 1.0)

    def test_invalid_family_format_and_unstable_hierarchy_are_rejected(self) -> None:
        base = self.by_shape(self.main, "belief", "free_text")
        with self.assertRaisesRegex(ValidationError, "pairing"):
            self.adapter.validate_case(
                replace(base, input_data={**base.input_data, "answer_format": "list"})
            )
        with self.assertRaisesRegex(ValidationError, "retain the conversation_id"):
            self.adapter.validate_case(
                replace(base, input_data={**base.input_data, "part_id": "other-p1", "set_id": "other-p1-s1"})
            )
        with self.assertRaisesRegex(ValidationError, "distinct"):
            self.adapter.validate_case(
                replace(base, input_data={**base.input_data, "wrong_answer": base.gold})
            )
        answerability = self.by_shape(self.main, "answerability", "binary")
        values = {key: value for key, value in answerability.input_data.items() if key != "target_fact"}
        with self.assertRaisesRegex(ValidationError, "target_fact"):
            self.adapter.validate_case(replace(answerability, input_data=values))


class SocialR1AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_fixture_suite(ROOT / "tests" / "fixtures")["social_r1"][0]
        cls.adapter = SocialR1Adapter()

    def test_original_fixture_executes_and_exposes_authoritative_aliases(self) -> None:
        results = [run_replay(self.adapter, case) for case in self.cases]
        self.assertTrue(all(result.status == ResultStatus.COMPLETED for result in results))
        aggregate = self.adapter.aggregate(results)
        self.assertEqual(aggregate["social_r1.accuracy"].value, 1.0)
        self.assertEqual(tuple(self.adapter.aliases), SOCIAL_R1_ALIASES)
        self.assertFalse(results[0].metadata["social_r1"]["official_data_available"])
        self.assertEqual(
            results[0].metadata["social_r1"]["placeholder_dataset_revision"],
            "539f50b8a6c35e643628ebae0f842681d7079cf3",
        )

    def test_parser_accepts_complete_paper_xml_or_exact_choice_only(self) -> None:
        case = self.cases[1]
        choices = self.adapter.choices_for_case(case, seed=0)
        accepted = (
            "A",
            '{"choice":"A"}',
            "<answer>A</answer>",
            "<think>brief reason</think><answer>A</answer>",
            "<thinking>brief reason</thinking>\n<answer>a</answer>",
        )
        for response in accepted:
            with self.subTest(response=response):
                self.assertEqual(parse_social_r1_response(response, choices).source_id, "A")
        malformed = (
            "",
            "A because it follows",
            "<answer>A</answer> because",
            "<answer>A</answer><answer>B</answer>",
            "<think><answer>B</answer></think><answer>A</answer>",
            "<think>x</thinking><answer>A</answer>",
            "<answer>E</answer>",
        )
        for response in malformed:
            with self.subTest(response=response), self.assertRaises(ParseError):
                parse_social_r1_response(response, choices)

    def test_validation_rejects_option_dimension_and_gold_errors(self) -> None:
        base = self.cases[0]
        with self.assertRaisesRegex(ValidationError, "four options"):
            self.adapter.validate_case(
                replace(base, input_data={**base.input_data, "options": base.input_data["options"][:3]})
            )
        with self.assertRaisesRegex(ValidationError, "distinct"):
            self.adapter.validate_case(
                replace(base, input_data={**base.input_data, "options": ["same"] * 4})
            )
        with self.assertRaisesRegex(ValidationError, "atoms_dimension"):
            self.adapter.validate_case(
                replace(base, input_data={**base.input_data, "atoms_dimension": "invented"})
            )
        with self.assertRaisesRegex(ValidationError, "gold"):
            self.adapter.validate_case(replace(base, gold="E"))

    def test_target_parse_failure_is_incorrect_not_dropped(self) -> None:
        target = self.cases[0]
        failed = run_replay(
            self.adapter,
            target,
            responses={f"{target.case_id}:choice": "<answer>B</answer> explanation"},
        )
        passed = run_replay(self.adapter, self.cases[1])
        aggregate = self.adapter.aggregate((failed, passed))
        self.assertEqual(failed.status, ResultStatus.COMPLETED)
        self.assertEqual(failed.metadata["target_output_failure"]["stage"], "response_parse")
        self.assertEqual(aggregate["social_r1.accuracy"].value, 0.5)
        self.assertEqual(aggregate["social_r1.accuracy"].denominator, 2)

    def test_all_six_atoms_dimensions_have_explicit_slices(self) -> None:
        base = self.cases[1]
        results = []
        for index, dimension in enumerate(ATOMS_DIMENSIONS):
            case = replace(
                base,
                case_id=f"social-dimension-{index}",
                group_id=f"social-dimension-group-{index}",
                input_data={**base.input_data, "atoms_dimension": dimension},
                metadata={
                    **base.metadata,
                    "strata": {**base.metadata["strata"], "atoms_dimension": dimension},
                },
            )
            results.append(run_replay(self.adapter, case))
        aggregate = self.adapter.aggregate(results)
        for dimension in ATOMS_DIMENSIONS:
            name = f"social_r1.accuracy.atoms_dimension.{dimension}"
            self.assertEqual(aggregate[name].value, 1.0)
            self.assertEqual(aggregate[name].denominator, 1)


if __name__ == "__main__":
    unittest.main()
