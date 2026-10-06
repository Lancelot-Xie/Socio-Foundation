import unittest
from pathlib import Path

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.agentsense import AgentSenseAdapter
from sim_eval.benchmarks.coser import CoserAdapter
from sim_eval.benchmarks.mirrorbench import MirrorBenchAdapter
from sim_eval.benchmarks.sotopia import SotopiaAdapter
from sim_eval.benchmarks.tau_usi import TauUSIAdapter, extract_behavior_features
from sim_eval.benchmarks.userlm import UserLMAdapter
from sim_eval.contracts import ResultStatus
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.errors import EpisodeTokenBudgetExceeded
from sim_eval.interfaces import ModelBackend


ROOT = Path(__file__).resolve().parents[1]


class ExhaustEvaluatedModelAfter(ModelBackend):
    """Delegate replay calls, then stop only requests for the target model."""

    name = "budget_exhaustion_fixture"

    def __init__(self, responses, *, target_model: str, completed_target_calls: int):
        self.delegate = ReplayBackend(responses=responses)
        self.target_model = target_model
        self.remaining = completed_target_calls

    def generate(self, request):
        if request.model == self.target_model:
            if self.remaining <= 0:
                raise EpisodeTokenBudgetExceeded(
                    "fixture exhausted evaluated-model token budget",
                    scope="episode_output",
                    episode_remaining_tokens=0,
                )
            self.remaining -= 1
        return self.delegate.generate(request)


class BudgetTerminationAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_fixture_suite(ROOT / "tests" / "fixtures")

    @staticmethod
    def execute(adapter, case, *, model, completed_target_calls):
        seed = 20260812
        responses = adapter.replay_responses(case, seed=seed)
        backend = ExhaustEvaluatedModelAfter(
            responses,
            target_model=model,
            completed_target_calls=completed_target_calls,
        )
        return adapter.execute_case(
            case,
            backend=backend,
            run_id="budget-termination-test",
            seed=seed,
            model=model,
        )

    def test_sotopia_scores_partial_transcript(self):
        adapter = SotopiaAdapter()
        result = self.execute(
            adapter,
            self.fixtures["sotopia"][0][0],
            model="fixture",
            completed_target_calls=1,
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["terminal_reason"], "token_budget_exhausted")
        self.assertEqual(result.metadata["judge_status"], "available")
        self.assertFalse(result.metadata["natural_termination"])

    def test_coser_scores_partial_scene(self):
        adapter = CoserAdapter()
        result = self.execute(
            adapter,
            self.fixtures["coser"][0][0],
            model="fixture",
            completed_target_calls=1,
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["terminal_reason"], "token_budget_exhausted")
        self.assertEqual(result.metadata["judge_call_count"], 4)
        self.assertFalse(result.metadata["natural_termination"])

    def test_mirrorbench_scores_available_partial_proxy(self):
        adapter = MirrorBenchAdapter()
        result = self.execute(
            adapter,
            self.fixtures["mirrorbench"][0][0],
            model="candidate-user",
            completed_target_calls=1,
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["terminal_reason"], "token_budget_exhausted")
        self.assertFalse(result.metadata["natural_termination"])

    def test_userlm_scores_existing_dialogue(self):
        adapter = UserLMAdapter()
        result = self.execute(
            adapter,
            self.fixtures["userlm"][0][0],
            model="candidate-user",
            completed_target_calls=1,
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["terminal_reason"], "token_budget_exhausted")
        self.assertTrue(result.metadata["userlm"]["all_user_text"])
        self.assertFalse(result.metadata["natural_termination"])

    def test_tau_usi_preserves_environment_reward(self):
        feature_inputs = []

        def recording_feature_extractor(messages):
            feature_inputs.append(tuple(messages))
            return extract_behavior_features(messages)

        adapter = TauUSIAdapter(feature_extractor=recording_feature_extractor)
        result = self.execute(
            adapter,
            self.fixtures["tau_usi"][0][0],
            model="fixture",
            completed_target_calls=1,
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["terminal_reason"], "token_budget_exhausted")
        self.assertEqual(result.metadata["tau_usi"]["environment_reward"], 1.0)
        self.assertEqual(result.metadata["tau_usi"]["stop_compliance"], 0.0)
        self.assertNotIn("target_output_failure", result.metadata)
        expected_partial_user_trajectory = tuple(
            str(event.content)
            for event in result.trace
            if event.actor == "evaluated_user" and event.kind == "message"
        )
        self.assertIn(expected_partial_user_trajectory, feature_inputs)

    def test_agentsense_uses_completed_partial_outcome(self):
        adapter = AgentSenseAdapter()
        result = self.execute(
            adapter,
            self.fixtures["agentsense"][0][0],
            model="fixture",
            completed_target_calls=1,
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["terminal_reason"], "token_budget_exhausted")
        self.assertFalse(result.metadata["natural_termination"])
        self.assertEqual(
            result.prediction["information"][0]["parse_status"],
            "token_budget_exhausted",
        )


if __name__ == "__main__":
    unittest.main()
