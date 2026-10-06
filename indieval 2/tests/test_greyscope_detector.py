import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sim_eval.errors import ConfigurationError, ValidationError
from sim_eval.integrations.greyscope import (
    GREYSCOPE_DEFAULT_MODEL_PATH,
    GREYSCOPE_DEFAULT_MODEL_REVISION,
    GREYSCOPE_MODEL_ID,
    GREYSCOPE_PREPROCESS_REVISION,
    GreyscopeDetectorConfig,
    GreyscopeLocalScorer,
    _validated_checkpoint_config,
    clean_greyscope_text,
    decode_greyscope_logits,
)


CALIBRATION = {
    "task": "editlens-ternary",
    "n_buckets": 4,
    "label_names": ["human", "AI-generated", "AI-edited"],
    "bucket_descriptions": ["none", "light", "moderate", "heavy"],
    "flip": False,
    "score_min": 0.02446465753018856,
    "score_max": 0.9999873638153076,
    "h_thresh": 0.16216216216216217,
    "ai_thresh": 0.994994994994995,
    "prompt_template": "Passage:\n{text}\nAnswer:",
    "lowercase": True,
    "max_length": 2048,
}


class GreyscopeDetectorTests(unittest.TestCase):
    def test_default_config_uses_cluster_model_path(self) -> None:
        config = GreyscopeDetectorConfig.from_mapping({"enabled": True})
        self.assertEqual(config.model_path, GREYSCOPE_DEFAULT_MODEL_PATH)
        self.assertEqual(config.model_id, GREYSCOPE_MODEL_ID)
        self.assertEqual(config.model_revision, GREYSCOPE_DEFAULT_MODEL_REVISION)
        self.assertEqual(
            config.identity()["preprocess_revision"], GREYSCOPE_PREPROCESS_REVISION
        )
        self.assertEqual(config.batch_size, 8)
        self.assertFalse(config.identity()["network_required_at_runtime"])

    def test_config_rejects_invalid_batch_size(self) -> None:
        with self.assertRaises(ConfigurationError):
            GreyscopeDetectorConfig.from_mapping(
                {"enabled": True, "batch_size": 0}
            )

    def test_config_rejects_unpinned_model_identity(self) -> None:
        with self.assertRaises(ConfigurationError):
            GreyscopeDetectorConfig.from_mapping(
                {"enabled": True, "model_id": "somewhere/another-detector"}
            )
        with self.assertRaises(ConfigurationError):
            GreyscopeDetectorConfig.from_mapping(
                {"enabled": True, "model_revision": "deadbeef"}
            )

    def test_preprocess_matches_pinned_editlens_contract(self) -> None:
        fake_emoji = SimpleNamespace(
            demojize=lambda value: value,
            replace_emoji=lambda value, replacement="": value,
        )
        with patch.dict("sys.modules", {"emoji": fake_emoji}):
            self.assertEqual(clean_greyscope_text("Сafe\u200b text"), "сafe\u200b text")
            self.assertEqual(
                clean_greyscope_text("reasoning</think>kept</think>discarded"),
                "kept",
            )

    def test_checkpoint_shape_is_pinned(self) -> None:
        valid = {
            "model_type": "qwen3_5_text",
            "architectures": ["Qwen3_5ForSequenceClassification"],
            "id2label": {str(index): f"LABEL_{index}" for index in range(4)},
            "dtype": "bfloat16",
        }
        self.assertEqual(_validated_checkpoint_config(valid), valid)
        with self.assertRaises(ConfigurationError):
            _validated_checkpoint_config({**valid, "id2label": {"0": "LABEL_0"}})

    def test_calibrated_decode_preserves_human_score_direction(self) -> None:
        human = decode_greyscope_logits([10.0, 0.0, 0.0, 0.0], CALIBRATION)
        ai = decode_greyscope_logits([0.0, 0.0, 0.0, 10.0], CALIBRATION)
        self.assertEqual(human["label"], "human")
        self.assertEqual(ai["label"], "AI-generated")
        self.assertGreater(human["human_likelihood"], 0.99)
        self.assertLess(ai["human_likelihood"], 0.01)
        self.assertAlmostEqual(
            human["human_likelihood"], 1.0 - human["ai_involvement"]
        )
        self.assertAlmostEqual(sum(human["bucket_probs"].values()), 1.0)

    def test_decode_rejects_wrong_logit_count(self) -> None:
        with self.assertRaises(ValidationError):
            decode_greyscope_logits([1.0, 2.0], CALIBRATION)

    def test_config_rejects_unknown_device(self) -> None:
        with self.assertRaises(ConfigurationError):
            GreyscopeDetectorConfig.from_mapping(
                {"enabled": True, "device": "gpu7"}
            )

    def test_config_accepts_indexed_cuda_device(self) -> None:
        config = GreyscopeDetectorConfig.from_mapping(
            {"enabled": True, "device": "cuda:7"}
        )
        self.assertEqual(config.device, "cuda:7")

    def test_batch_scorer_uses_one_padded_forward_and_preserves_order(self) -> None:
        class FakeEncoded(dict):
            def to(self, device):
                self.device = device
                return self

        class FakeTokenizer:
            def __call__(self, texts, *, return_tensors=None, **kwargs):
                values = list(texts)
                if return_tensors:
                    return FakeEncoded({"input_ids": values})
                return {"input_ids": [list(range(len(value.split()))) for value in values]}

        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def float(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class FakeModel:
            device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **encoded):
                self.calls.append(encoded)
                rows = [
                    [10.0, 0.0, 0.0, 0.0]
                    if "clearly human" in value
                    else [0.0, 0.0, 0.0, 10.0]
                    for value in encoded["input_ids"]
                ]
                return SimpleNamespace(
                    logits=FakeTensor(rows)
                )

        scorer = object.__new__(GreyscopeLocalScorer)
        scorer.config = GreyscopeDetectorConfig(
            enabled=True,
            model_path="/test/model",
            batch_size=4,
            device="cpu",
        )
        scorer.model_path = Path("/test/model")
        scorer.calibration = dict(CALIBRATION)
        scorer.calibration_sha256 = "test-calibration"
        scorer.tokenizer = FakeTokenizer()
        scorer.model = FakeModel()
        scorer.device = "cpu"
        scorer._lock = threading.Lock()
        scorer._torch = SimpleNamespace(inference_mode=nullcontext)

        with patch(
            "sim_eval.integrations.greyscope.clean_greyscope_text",
            side_effect=lambda value: value,
        ):
            evidence = scorer.score_batch(("A clearly human reply", "A generated reply"))

        self.assertEqual(len(scorer.model.calls), 1)
        self.assertEqual([item["label"] for item in evidence], ["human", "AI-generated"])
        self.assertEqual([item["batch"]["position"] for item in evidence], [0, 1])
        self.assertTrue(
            all(item["batch"]["observed_case_batch_size"] == 2 for item in evidence)
        )
        with patch(
            "sim_eval.integrations.greyscope.clean_greyscope_text",
            side_effect=lambda value: value,
        ):
            scalar_evidence = (
                scorer("A clearly human reply"),
                scorer("A generated reply"),
            )
        self.assertEqual(
            [item["human_likelihood"] for item in evidence],
            [item["human_likelihood"] for item in scalar_evidence],
        )


if __name__ == "__main__":
    unittest.main()
