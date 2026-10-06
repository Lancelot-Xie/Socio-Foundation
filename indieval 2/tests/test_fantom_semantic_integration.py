import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.fantom import FantomAdapter
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.errors import ConfigurationError
from sim_eval.integrations.fantom_semantic import FantomSentenceTransformerScorer


ROOT = Path(__file__).resolve().parents[1]


class FantomSemanticIntegrationTests(unittest.TestCase):
    def test_local_scorer_loads_the_configured_directory(self) -> None:
        fake_module = types.ModuleType("sentence_transformers")
        calls = []

        class FakeSentenceTransformer:
            def __init__(self, model_path, **kwargs):
                calls.append((model_path, kwargs))

            def encode(self, values, **kwargs):
                self.values = values
                self.encode_kwargs = kwargs
                return ([1.0, 0.0], [1.0, 0.0], [0.0, 1.0])

        fake_module.SentenceTransformer = FakeSentenceTransformer
        with TemporaryDirectory() as model_path:
            with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
                scorer = FantomSentenceTransformerScorer(
                    model_revision="pinned-test-revision",
                    model_path=model_path,
                    device="cpu",
                )
                evidence = scorer("prediction", "correct", "wrong")

        self.assertEqual(calls, [(model_path, {"device": "cpu"})])
        self.assertEqual(evidence["model_revision"], "pinned-test-revision")
        self.assertEqual(evidence["correct_similarity"], 1.0)
        self.assertEqual(evidence["wrong_similarity"], 0.0)
        self.assertEqual(evidence["batch"]["observed_case_batch_size"], 1)

    def test_local_scorer_batches_cases_and_preserves_tail_order(self) -> None:
        fake_module = types.ModuleType("sentence_transformers")
        encoded = []

        class FakeSentenceTransformer:
            def __init__(self, model_path, **kwargs):
                self.model_path = model_path

            def encode(self, values, **kwargs):
                encoded.append((list(values), dict(kwargs)))
                vectors = {
                    "prediction-1": [1.0, 0.0],
                    "correct-1": [1.0, 0.0],
                    "wrong-1": [0.0, 1.0],
                    "prediction-2": [0.0, 1.0],
                    "correct-2": [1.0, 0.0],
                    "wrong-2": [0.0, 1.0],
                }
                return tuple(vectors[value] for value in values)

        fake_module.SentenceTransformer = FakeSentenceTransformer
        with TemporaryDirectory() as model_path:
            with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
                scorer = FantomSentenceTransformerScorer(
                    model_revision="pinned-test-revision",
                    model_path=model_path,
                    device="cpu",
                    batch_size=4,
                )
                evidence = scorer.score_batch(
                    (
                        ("prediction-1", "correct-1", "wrong-1"),
                        ("prediction-2", "correct-2", "wrong-2"),
                    )
                )
                scalar_evidence = (
                    scorer("prediction-1", "correct-1", "wrong-1"),
                    scorer("prediction-2", "correct-2", "wrong-2"),
                )

        self.assertEqual(
            encoded[0],
            (
                [
                    "prediction-1", "correct-1", "wrong-1",
                    "prediction-2", "correct-2", "wrong-2",
                ],
                {"batch_size": 4},
            ),
        )
        self.assertEqual([item["correct_similarity"] for item in evidence], [1.0, 0.0])
        self.assertEqual([item["wrong_similarity"] for item in evidence], [0.0, 1.0])
        self.assertEqual(
            [item["batch"]["position"] for item in evidence],
            [0, 1],
        )
        self.assertTrue(
            all(item["batch"]["observed_case_batch_size"] == 2 for item in evidence)
        )
        self.assertEqual(
            [
                (item["correct_similarity"], item["wrong_similarity"])
                for item in evidence
            ],
            [
                (item["correct_similarity"], item["wrong_similarity"])
                for item in scalar_evidence
            ],
        )

    def test_local_scorer_rejects_a_missing_directory(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError,
            "FANToM semantic model directory does not exist",
        ):
            FantomSentenceTransformerScorer(
                model_revision="pinned-test-revision",
                model_path="/definitely/missing/all-roberta-large-v1",
                device="cpu",
            )

    def test_injected_semantic_scorer_supplies_official_evidence_contract(self) -> None:
        cases = load_fixture_suite(ROOT / "tests" / "fixtures")["fantom"][0]
        case = next(item for item in cases if item.input_data["question_family"] == "belief" and item.input_data["answer_format"] == "free_text")

        def scorer(prediction, correct, wrong):
            self.assertTrue(prediction and correct and wrong)
            return {
                "contract_revision": "fantom-belief-distance-evidence-v1",
                "model_id": "sentence-transformers/all-roberta-large-v1",
                "model_revision": "pinned-test-revision",
                "metric": "cosine_similarity",
                "correct_similarity": 0.9,
                "wrong_similarity": 0.1,
            }

        adapter = FantomAdapter(belief_semantic_scorer=scorer)
        responses = adapter.replay_responses(case, seed=1)
        response = responses[f"{case.case_id}:question"]
        if isinstance(response, dict):
            response = response["text"]
        result = adapter.execute_case(case, backend=ReplayBackend({f"{case.case_id}:question": response}), run_id="test", seed=1, model="candidate")
        metric = next(item for item in result.metrics if item.name == "fantom.belief_distance_accuracy")
        self.assertEqual(metric.value, 1)


if __name__ == "__main__":
    unittest.main()
