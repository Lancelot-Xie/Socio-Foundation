from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.humanual import HumanualAdapter, STATE_NAMES, extract_response
from sim_eval.data.loaders import load_fixture_suite, load_import_spec, load_local_cases
from sim_eval.contracts import ResultStatus


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
HUMANUAL_MANIFEST = (
    ROOT.parent
    / "temp_data_qa"
    / "datasets"
    / "derived"
    / "eval_scale_variants_v1"
    / "humanual_100"
    / "import_manifest.json"
)


class HumanualAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases, _ = load_fixture_suite(FIXTURES)["humanual"]

    def test_official_permissive_response_parser(self) -> None:
        self.assertEqual(extract_response("<response>Hello</response>"), "Hello")
        self.assertEqual(extract_response("<think>private</think>\n<response>Hello"), "Hello")
        self.assertEqual(extract_response("Hello</response> trailing"), "Hello")
        self.assertEqual(extract_response("plain response"), "plain response")

    def test_official_context_role_mapping_and_generation_contract(self) -> None:
        adapter = HumanualAdapter()
        book = next(case for case in self.cases if case.input_data["domain"] == "book")
        chat = next(case for case in self.cases if case.input_data["domain"] == "chat")
        book_request = adapter.build_request(book, model="candidate", seed=7)
        chat_request = adapter.build_request(chat, model="candidate", seed=7)

        self.assertEqual(book_request.temperature, 0.0)
        self.assertEqual(book_request.max_tokens, 1024)
        self.assertEqual(book_request.stop, ())
        self.assertEqual([message.role for message in book_request.messages], ["system", "user"])
        self.assertEqual(
            [message.role for message in chat_request.messages],
            ["system", "user", "assistant"],
        )
        self.assertIsNone(chat_request.messages[1].name)
        self.assertTrue(chat_request.messages[1].content.startswith("HUMAN: "))
        self.assertTrue(chat_request.messages[2].content.startswith("ASSISTANT: "))
        self.assertIsNone(chat_request.messages[2].name)
        self.assertEqual(
            adapter.environment_identity_for_case(book),
            adapter.environment_identity_for_case(chat),
        )

    def test_replay_populates_response_and_six_state_metrics(self) -> None:
        adapter = HumanualAdapter()
        responses = {}
        for case in self.cases:
            responses.update(adapter.replay_responses(case, seed=3))
        backend = ReplayBackend(responses=responses)
        results = [
            adapter.execute_case(
                case,
                backend=backend,
                run_id="humanual-fixture",
                seed=3,
                model="offline-replay",
            )
            for case in self.cases
        ]
        self.assertTrue(all(result.status == ResultStatus.COMPLETED for result in results))
        self.assertEqual(STATE_NAMES, ("stance", "emotion", "belief", "value", "goal", "communication"))
        self.assertTrue(
            all(
                len([metric for metric in result.metrics if metric.name.startswith("humanual.state.")]) == 6
                for result in results
            )
        )
        aggregate = adapter.aggregate(results)
        self.assertAlmostEqual(aggregate["humanual.response_alignment"].value, 0.975)
        self.assertAlmostEqual(aggregate["humanual.state_alignment"].value, 0.9625)

    def test_project_pinned_embedding_cosine_is_batch_finalized(self) -> None:
        class FakeEmbeddingScorer:
            batch_size = 8

            @staticmethod
            def protocol_identity():
                return {
                    "model_id": "sentence-transformers/all-roberta-large-v1",
                    "model_revision": "fixture-revision",
                    "metric": "cosine_similarity",
                }

            @staticmethod
            def score_batch(items):
                return tuple(
                    {
                        "model_id": "sentence-transformers/all-roberta-large-v1",
                        "model_revision": "fixture-revision",
                        "metric": "cosine_similarity",
                        "similarity": 0.75 if position == 0 else 0.5,
                        "batch": {"position": position, "observed_case_batch_size": len(items)},
                    }
                    for position, _ in enumerate(items)
                )

        adapter = HumanualAdapter(
            embedding_scorer=FakeEmbeddingScorer(),
        )
        responses = {}
        for case in self.cases:
            responses.update(adapter.replay_responses(case, seed=3))
        backend = ReplayBackend(responses=responses)
        initial = tuple(
            adapter.execute_case(
                case,
                backend=backend,
                run_id="humanual-embedding-fixture",
                seed=3,
                model="offline-replay",
            )
            for case in self.cases
        )
        self.assertTrue(all(adapter.needs_embedding(case, result) for case, result in zip(self.cases, initial)))
        finalized = adapter.finalize_embedding_batch(
            tuple((case, result, 3) for case, result in zip(self.cases, initial))
        )
        values = [
            next(
                metric.value
                for metric in result.metrics
                if metric.name == "humanual.embedding_cosine_similarity"
            )
            for result in finalized
        ]
        self.assertEqual(values, [0.75, 0.5])
        self.assertAlmostEqual(
            adapter.aggregate(finalized)["humanual.embedding_cosine_similarity"].value,
            0.625,
        )

    def test_official_exact_and_empty_shortcuts_do_not_call_judges(self) -> None:
        adapter = HumanualAdapter()
        case = self.cases[0]
        for raw_response, expected, shortcut in (
            (case.gold, 1.0, "official_exact_ground_truth"),
            ("", 0.0, "official_empty_generation"),
        ):
            backend = ReplayBackend(responses={f"{case.case_id}:response": raw_response})
            result = adapter.execute_case(
                case,
                backend=backend,
                run_id=f"shortcut-{expected}",
                seed=5,
                model="offline-replay",
            )
            values = {metric.name: metric.value for metric in result.metrics}
            self.assertEqual(values["humanual.response_alignment"], expected)
            self.assertEqual(values["humanual.state_alignment"], expected)
            self.assertEqual(result.metadata["scoring_shortcut"], shortcut)




if __name__ == "__main__":
    unittest.main()
