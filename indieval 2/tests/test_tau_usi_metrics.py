import math
import statistics
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.tau_usi import (
    FEATURE_DIMENSIONS,
    SURVEY_SCALES,
    TauRuntimeProvenance,
    TauUSIAdapter,
    dice_alignment,
    difficulty_bin,
    evaluative_alignment,
    expected_calibration_error,
    extract_behavior_features,
    normalize_survey,
    user_sim_index,
)
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.errors import ConfigurationError, ParseError, ValidationError


ROOT = Path(__file__).resolve().parents[1]


class TauUSIMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_fixture_suite(ROOT / "tests" / "fixtures")["tau_usi"][0]
        cls.adapter = TauUSIAdapter()
        cls.results = []
        for case in cls.cases:
            responses = cls.adapter.replay_responses(case, seed=20260812)
            cls.results.append(
                cls.adapter.execute_case(
                    case,
                    backend=ReplayBackend(responses),
                    run_id="tau-metric-fixture",
                    seed=20260812,
                    model="offline-replay",
                )
            )

    def test_dice_formula_and_zero_boundary(self) -> None:
        self.assertEqual(dice_alignment(0.0, 0.0), 100.0)
        self.assertEqual(dice_alignment(1.0, 3.0), 50.0)
        self.assertEqual(dice_alignment(2.5, 2.5), 100.0)
        with self.assertRaises(ValidationError):
            dice_alignment(-1.0, 1.0)

    def test_fixed_difficulty_bins_use_declared_cutoffs(self) -> None:
        values = [0.0, 0.199, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.assertEqual([difficulty_bin(value) for value in values], [0, 0, 1, 2, 3, 4, 4])
        with self.assertRaises(ValidationError):
            difficulty_bin(1.01)

    def test_ece_is_weighted_over_paired_tasks(self) -> None:
        rows = [
            (0.1, 1.0, 0.0),
            (0.1, 1.0, 1.0),
            (0.5, 0.0, 1.0),
        ]
        self.assertAlmostEqual(expected_calibration_error(rows), 2.0 / 3.0)

    def test_survey_normalization_and_eval_formula(self) -> None:
        minimum = {name: 0 for name in SURVEY_SCALES}
        maximum = dict(SURVEY_SCALES)
        self.assertTrue(all(value == 0 for value in normalize_survey(minimum).values()))
        self.assertTrue(all(value == 1 for value in normalize_survey(maximum).values()))
        score, per_field = evaluative_alignment(
            [normalize_survey(maximum)], [normalize_survey(minimum)]
        )
        self.assertEqual(score, 0.0)
        self.assertTrue(all(value == 1.0 for value in per_field.values()))
        with self.assertRaises(ParseError):
            normalize_survey({**minimum, "extra": 1})

    def test_usi_six_and_five_component_equations(self) -> None:
        self.assertEqual(user_sim_index(100, 100, 100, 100, 0, 100), 100.0)
        self.assertEqual(user_sim_index(100, 100, 100, 100, 0), 100.0)
        self.assertAlmostEqual(user_sim_index(50, 60, 70, 80, 0.2, 90), 430 / 6)

    def test_project_runtime_defaults_are_60_by_64(self) -> None:
        runtime = TauRuntimeProvenance(
            fixed_assistant_model="fixed-assistant",
            fixed_assistant_revision="immutable-revision",
            assistant_policy_revision="policy-v1",
            environment_revision="tau-usi-task-loop-v3-official-greeting-no-user-state-leak",
            tool_schema_revision="single-native-tool-call-v3",
        )
        self.assertEqual(runtime.max_user_turns, 60)
        self.assertEqual(runtime.max_assistant_steps_per_user_turn, 64)
        with self.assertRaisesRegex(ConfigurationError, "cannot exceed"):
            replace(runtime, max_user_turns=61)

    def test_question_categories_are_mutually_exclusive_by_priority(self) -> None:
        features = extract_behavior_features(
            ["Are you sure? Can you clarify?", "Maybe, what is the status?"]
        )
        self.assertEqual(features["pushback_question_turn_rate"], 0.5)
        self.assertEqual(features["clarification_question_turn_rate"], 0.0)
        self.assertEqual(features["information_question_turn_rate"], 0.5)
        self.assertEqual(features["uncertainty_turn_rate"], 0.5)
        self.assertEqual(
            set(features), {feature for names in FEATURE_DIMENSIONS.values() for feature in names}
        )

    def test_supplemental_dimension_membership_does_not_double_weight_words_per_turn(self) -> None:
        self.assertNotIn("words_per_turn", FEATURE_DIMENSIONS["d1_communication"])
        self.assertIn("words_per_turn", FEATURE_DIMENSIONS["d2_information"])
        self.assertEqual(len(FEATURE_DIMENSIONS["d1_communication"]), 7)

    def test_repeated_trigram_is_an_interaction_indicator(self) -> None:
        ten_repetitions = " ".join(["red blue green"] * 10)
        eleven_repetitions = " ".join(["red blue green"] * 11)
        self.assertEqual(
            extract_behavior_features([ten_repetitions])["repeated_trigram_interaction_rate"],
            0.0,
        )
        self.assertEqual(
            extract_behavior_features([eleven_repetitions])["repeated_trigram_interaction_rate"],
            1.0,
        )

    def test_fixture_aggregation_matches_each_batch_equation(self) -> None:
        metrics = self.adapter.aggregate(self.results)
        self.assertTrue(metrics["tau_usi.suite_complete"].value)
        self.assertAlmostEqual(metrics["tau_usi.ece"].value, 1.0 / 3.0)
        self.assertAlmostEqual(
            metrics["tau_usi.ece"].uncertainty["std"], statistics.pstdev([0.5, 0.0, 0.5])
        )
        eval_mean = metrics["tau_usi.eval"].value
        for batch_id in ("fixture_h1", "fixture_h2", "fixture_h3"):
            d_values = [metrics[f"tau_usi.{name}"].metadata["per_batch"][batch_id] for name in FEATURE_DIMENSIONS]
            ece = metrics["tau_usi.ece"].metadata["per_batch"][batch_id]
            expected = (sum(d_values) + (1 - ece) * 100 + eval_mean) / 6
            self.assertAlmostEqual(metrics["tau_usi.usi"].metadata["per_batch"][batch_id], expected)

    def test_missing_human_inputs_are_unavailable_not_synthesized(self) -> None:
        stripped = []
        for result in self.results:
            metadata = dict(result.metadata)
            tau = dict(metadata["tau_usi"])
            tau["human_references"] = {}
            tau["difficulty_score"] = None
            metadata["tau_usi"] = tau
            stripped.append(replace(result, metadata=metadata))
        metrics = self.adapter.aggregate(stripped)
        self.assertTrue(metrics["tau_usi.suite_complete"].value)
        for name in (
            "tau_usi.d1_communication",
            "tau_usi.d2_information",
            "tau_usi.d3_clarification",
            "tau_usi.d4_error_reaction",
            "tau_usi.ece",
            "tau_usi.eval",
            "tau_usi.usi",
            "tau_usi.usi_without_eval",
        ):
            self.assertIsNone(metrics[name].value, name)
            self.assertEqual(metrics[name].metadata["availability"], "unavailable")

    def test_missing_simulator_survey_preserves_five_component_variant(self) -> None:
        without_survey = []
        for result in self.results:
            metadata = dict(result.metadata)
            tau = dict(metadata["tau_usi"])
            tau["simulator_survey"] = None
            metadata["tau_usi"] = tau
            without_survey.append(replace(result, metadata=metadata))
        metrics = self.adapter.aggregate(without_survey)
        self.assertIsNone(metrics["tau_usi.eval"].value)
        self.assertIsNone(metrics["tau_usi.usi"].value)
        self.assertIsNotNone(metrics["tau_usi.usi_without_eval"].value)

    def test_incomplete_task_population_nulls_all_distribution_scores(self) -> None:
        metrics = self.adapter.aggregate(self.results[:1])
        self.assertFalse(metrics["tau_usi.suite_complete"].value)
        self.assertIn("expected 2 tasks", metrics["tau_usi.suite_complete"].metadata["reason"])
        self.assertIsNone(metrics["tau_usi.usi"].value)
        self.assertIsNone(metrics["tau_usi.ece"].value)


if __name__ == "__main__":
    unittest.main()
