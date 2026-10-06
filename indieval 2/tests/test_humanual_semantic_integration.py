import sys
import types
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sim_eval.errors import ConfigurationError
from sim_eval.integrations.humanual_semantic import HumanualSentenceTransformerScorer


class HumanualSemanticIntegrationTests(unittest.TestCase):
    def test_local_scorer_batches_response_reference_pairs(self) -> None:
        fake_module = types.ModuleType("sentence_transformers")
        calls = []

        class FakeSentenceTransformer:
            def __init__(self, model_path, **kwargs):
                calls.append((model_path, kwargs))

            def encode(self, values, **kwargs):
                calls.append((list(values), dict(kwargs)))
                vectors = {
                    "prediction-1": [1.0, 0.0],
                    "reference-1": [1.0, 0.0],
                    "prediction-2": [0.0, 1.0],
                    "reference-2": [1.0, 0.0],
                }
                return tuple(vectors[value] for value in values)

        fake_module.SentenceTransformer = FakeSentenceTransformer
        with TemporaryDirectory() as model_path:
            with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
                scorer = HumanualSentenceTransformerScorer(
                    model_revision="pinned-test-revision",
                    model_path=model_path,
                    device="cpu",
                    batch_size=4,
                )
                evidence = scorer.score_batch(
                    (
                        ("prediction-1", "reference-1"),
                        ("prediction-2", "reference-2"),
                    )
                )

        self.assertEqual(calls[0], (model_path, {"device": "cpu"}))
        self.assertEqual(
            calls[1],
            (
                ["prediction-1", "reference-1", "prediction-2", "reference-2"],
                {"batch_size": 4},
            ),
        )
        self.assertEqual([item["similarity"] for item in evidence], [1.0, 0.0])
        self.assertEqual([item["batch"]["position"] for item in evidence], [0, 1])
        self.assertTrue(
            all(item["batch"]["observed_case_batch_size"] == 2 for item in evidence)
        )

    def test_local_scorer_rejects_a_missing_directory(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError,
            "HUMANUAL embedding model directory does not exist",
        ):
            HumanualSentenceTransformerScorer(
                model_revision="pinned-test-revision",
                model_path="/definitely/missing/all-roberta-large-v1",
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
