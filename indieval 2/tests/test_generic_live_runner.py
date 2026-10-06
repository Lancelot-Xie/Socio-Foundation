import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.fantom import (
    FANTOM_SEMANTIC_CONTRACT,
    FANTOM_SEMANTIC_MODEL,
    FantomAdapter,
)
from sim_eval.benchmarks.humanual import HumanualAdapter
from sim_eval.benchmarks.mirrorbench import MirrorBenchAdapter
from sim_eval.data.loaders import load_import_spec, load_local_cases
from sim_eval.errors import ConfigurationError
from sim_eval.generic_runner import run_generic_import


ROOT = Path(__file__).resolve().parents[1]


class GenericLiveRunnerTests(unittest.TestCase):
    def test_fantom_runner_flushes_a_short_semantic_tail_as_one_batch(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            manifest_path = directory / "fantom_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "fantom",
                        "path": str(ROOT / "tests" / "fixtures" / "fantom.jsonl"),
                        "format": "jsonl",
                        "source_kind": "synthetic_fixture",
                        "source_revision": "fantom-batch-runner-test-v1",
                        "split": "fixture",
                    }
                ),
                encoding="utf-8",
            )
            config_path = directory / "fantom_runtime.yaml"
            config_path.write_text(
                """schema_version: "1.0"
benchmark_id: fantom
execution: {repetitions: 1, max_workers: 1}
roles:
  evaluated_model:
    backend: vllm
    base_url: http://127.0.0.1:8000/v1
    api_key_env: UNUSED_TEST_KEY
    require_api_key: false
    model: candidate-test-model
    model_revision: candidate-test-revision
    generation: {temperature: 0, max_tokens: 128}
routing: {default_role: evaluated_model}
prompts:
  question: {source: official_repository, revision: test-prompt-v1, locator: test-only}
resources:
  belief_semantic_backend:
    kind: sentence_transformers
    model: sentence-transformers/all-roberta-large-v1
    model_revision: pinned-test-revision
    model_path: /unused/test/model
    metric: cosine_similarity
    batch_size: 4
    device: cpu
""",
                encoding="utf-8",
            )
            cases, _ = load_local_cases(load_import_spec(manifest_path))
            replay_adapter = FantomAdapter()
            responses = {}
            semantic_case_ids = []
            for case in cases:
                replay = dict(replay_adapter.replay_responses(case, seed=20260819))
                if (
                    case.input_data.get("question_family") == "belief"
                    and case.input_data.get("answer_format") == "free_text"
                ):
                    semantic_case_ids.append(case.case_id)
                    key = f"{case.case_id}:question"
                    value = replay[key]
                    replay[key] = value["text"] if isinstance(value, dict) else value
                responses.update(replay)

            observed_batches = []

            class FakeBatchSemanticScorer:
                def __init__(self, *, model_revision, model_path, device, batch_size):
                    self.model_revision = model_revision
                    self.batch_size = batch_size

                def protocol_identity(self):
                    return {
                        "model_id": FANTOM_SEMANTIC_MODEL,
                        "model_revision": self.model_revision,
                        "contract_revision": FANTOM_SEMANTIC_CONTRACT,
                        "metric": "cosine_similarity",
                        "case_batch_size": self.batch_size,
                    }

                def score_batch(self, items):
                    observed_batches.append(tuple(items))
                    return tuple(
                        {
                            "contract_revision": FANTOM_SEMANTIC_CONTRACT,
                            "model_id": FANTOM_SEMANTIC_MODEL,
                            "model_revision": self.model_revision,
                            "metric": "cosine_similarity",
                            "correct_similarity": 0.9,
                            "wrong_similarity": 0.1,
                            "batch": {
                                "configured_case_batch_size": self.batch_size,
                                "observed_case_batch_size": len(items),
                                "position": position,
                            },
                        }
                        for position, _ in enumerate(items)
                    )

            with (
                patch(
                    "sim_eval.generic_runner.FantomSentenceTransformerScorer",
                    FakeBatchSemanticScorer,
                ),
                patch(
                    "sim_eval.generic_runner.build_role_routed_backend",
                    return_value=ReplayBackend(responses),
                ),
            ):
                summary = run_generic_import(
                    manifest_path=manifest_path,
                    runtime_config_path=config_path,
                    output_directory=directory / "artifacts",
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                )
                first_batches = tuple(observed_batches)
                observed_batches.clear()
                resumed = run_generic_import(
                    manifest_path=manifest_path,
                    runtime_config_path=config_path,
                    output_directory=directory / "artifacts",
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                )

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(resumed["run_id"], summary["run_id"])
            self.assertEqual(len(semantic_case_ids), 2)
            self.assertEqual(len(first_batches), 1)
            self.assertEqual(len(first_batches[0]), 2)
            self.assertEqual(observed_batches, [])
            records_path = directory / "artifacts" / summary["artifacts"]["records"]
            records = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
            ]
            semantic_records = [
                record for record in records if record["case_id"] in semantic_case_ids
            ]
            self.assertEqual(
                [
                    record["model_response"]["raw"]["fantom_belief_distance"]["batch"]["position"]
                    for record in semantic_records
                ],
                [0, 1],
            )

    def test_humanual_runner_batches_embedding_cosine_and_preserves_resume(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            manifest_path = directory / "humanual_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "humanual",
                        "path": str(ROOT / "tests" / "fixtures" / "humanual.jsonl"),
                        "format": "jsonl",
                        "source_kind": "synthetic_fixture",
                        "source_revision": "humanual-batch-runner-test-v1",
                        "split": "fixture",
                    }
                ),
                encoding="utf-8",
            )
            config_path = directory / "humanual_runtime.yaml"
            config_path.write_text(
                """schema_version: "1.0"
benchmark_id: humanual
execution: {repetitions: 1, max_workers: 2}
roles:
  evaluated_model:
    backend: vllm
    base_url: http://127.0.0.1:8000/v1
    api_key_env: UNUSED_TEST_KEY
    require_api_key: false
    model: candidate-test-model
    model_revision: candidate-test-revision
    generation: {temperature: 0.4, max_tokens: 1024}
  judge:
    backend: openai_compatible
    base_url: http://127.0.0.1:8001/v1
    api_key_env: UNUSED_JUDGE_KEY
    require_api_key: false
    model: fixture-humanual-judge
    model_revision: fixture-judge-revision
    generation: {temperature: 0, max_tokens: 4096}
routing: {default_role: evaluated_model}
prompts:
  response_generation: {source: official_repository, revision: humanual-test-prompt-v1, locator: test-only}
  alignment_judge: {source: official_repository, revision: humanual-test-judge-v1, locator: test-only}
resources:
  embedding_semantic_backend:
    kind: sentence_transformers
    model: sentence-transformers/all-roberta-large-v1
    model_revision: pinned-test-revision
    model_path: /unused/test/model
    metric: cosine_similarity
    batch_size: 8
    device: cpu
""",
                encoding="utf-8",
            )
            cases, _ = load_local_cases(load_import_spec(manifest_path))
            replay_adapter = HumanualAdapter()
            responses = {}
            for case in cases:
                responses.update(replay_adapter.replay_responses(case, seed=20260824))
            observed_batches = []

            class FakeHumanualEmbeddingScorer:
                def __init__(self, *, model_revision, model_path, device, batch_size):
                    self.model_revision = model_revision
                    self.batch_size = batch_size

                def protocol_identity(self):
                    return {
                        "model_id": "sentence-transformers/all-roberta-large-v1",
                        "model_revision": self.model_revision,
                        "contract_revision": "humanual-response-reference-cosine-project-pinned-v1",
                        "metric": "cosine_similarity",
                        "case_batch_size": self.batch_size,
                    }

                def score_batch(self, items):
                    observed_batches.append(tuple(items))
                    return tuple(
                        {
                            **self.protocol_identity(),
                            "similarity": 0.75 if position == 0 else 0.5,
                            "batch": {
                                "configured_case_batch_size": self.batch_size,
                                "observed_case_batch_size": len(items),
                                "position": position,
                            },
                        }
                        for position, _ in enumerate(items)
                    )

            with (
                patch(
                    "sim_eval.generic_runner.HumanualSentenceTransformerScorer",
                    FakeHumanualEmbeddingScorer,
                ),
                patch(
                    "sim_eval.generic_runner.build_role_routed_backend",
                    return_value=ReplayBackend(responses),
                ),
            ):
                summary = run_generic_import(
                    manifest_path=manifest_path,
                    runtime_config_path=config_path,
                    output_directory=directory / "artifacts",
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                )
                first_batches = tuple(observed_batches)
                observed_batches.clear()
                resumed = run_generic_import(
                    manifest_path=manifest_path,
                    runtime_config_path=config_path,
                    output_directory=directory / "artifacts",
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                )

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(resumed["run_id"], summary["run_id"])
            self.assertEqual(len(first_batches), 1)
            self.assertEqual(len(first_batches[0]), 2)
            self.assertEqual(observed_batches, [])
            self.assertAlmostEqual(
                summary["metrics"]["humanual.embedding_cosine_similarity"]["value"],
                0.625,
            )
            self.assertEqual(
                summary["metrics"]["humanual.embedding_cosine_similarity.availability_rate"]["value"],
                1.0,
            )

    @staticmethod
    def _mirror_runtime(*, assistant_model: str, judge_model: str, fallback_reason: str | None = None) -> str:
        fallback = f"\n    fallback_reason: {fallback_reason}" if fallback_reason else ""
        return f'''schema_version: "1.0"
benchmark_id: mirrorbench
execution: {{repetitions: 1, max_workers: 1, request_timeout_seconds: 30}}
roles:
  evaluated_user:
    backend: vllm
    base_url: http://127.0.0.1:8000/v1
    api_key_env: UNUSED_CANDIDATE_KEY
    require_api_key: false
    model: candidate-test-model
    model_revision: candidate-test-revision
    generation: {{temperature: 1.0, max_tokens: 512}}
  fixed_assistant:
    backend: openai_compatible
    base_url: http://127.0.0.1:8001/v1
    api_key_env: UNUSED_ASSISTANT_KEY
    require_api_key: false
    model: {assistant_model}
    model_revision: {assistant_model}-revision{fallback}
    generation: {{temperature: 0.0, max_tokens: 1024}}
  judge:
    backend: openai_compatible
    base_url: http://127.0.0.1:8002/v1
    api_key_env: UNUSED_JUDGE_KEY
    require_api_key: false
    model: {judge_model}
    model_revision: {judge_model}-revision{fallback}
    generation: {{temperature: 0.0, max_tokens: 1024}}
routing: {{default_role: evaluated_user}}
prompts:
  user_simulation: {{source: official_repository, revision: mirror-user-v1, locator: test-only}}
  fixed_assistant: {{source: official_repository, revision: mirror-assistant-v1, locator: test-only}}
  judge: {{source: official_repository, revision: mirror-judge-v1, locator: test-only}}
environment:
  max_user_turns: 8
  max_total_actions: 20
  generation_max_tokens: 2048
  request_timeout_seconds: 30
  max_retries: 1
scoring:
  tokenizer_policy: regex_word_v1
  tokenizer_model: regex_word_v1
  tokenizer_revision: mirror-regex-v1
  gteval_prompt_revision: mirrorbench-gteval-v1
  pi_prompt_revision: mirrorbench-pi-v1
  rnr_prompt_revision: mirrorbench-rnr-v1
  gteval_samples: 1
  pi_samples: 3
  rnr_samples: 2
  compute_controls: true
'''

    def test_lifechoices_real_runner_writes_checkpointed_metrics(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            record_path = directory / "lifechoices.jsonl"
            record = {
                "source_id": "live-runner-case",
                "group_source_id": "live-runner-group",
                "split": "test",
                "input": {
                    "character_name": "Sera Vale",
                    "character_id": "sera-vale",
                    "character_profile": "Sera protects shared records before trophies.",
                    "decision_context": "A flood reaches the archive.",
                    "scenario": "Only one room can be protected.",
                    "question": "What does Sera protect?",
                    "options": ["A. records", "B. trophies", "C. nothing", "D. an empty room"],
                    "book_id": "invented-live-runner-book",
                    "profile_method": "expert_description",
                    "entity_replacement_revision": "test-v1",
                },
                "gold": "A",
                "strata": {"book": "invented-live-runner-book", "context_condition": "expert_description"},
            }
            record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            manifest_path = directory / "import_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "lifechoices",
                        "path": record_path.name,
                        "format": "jsonl",
                        "source_kind": "synthetic_fixture",
                        "source_revision": "generic-live-runner-test-v1",
                        "split": "test",
                    }
                ),
                encoding="utf-8",
            )
            config_path = directory / "runtime.yaml"
            config_path.write_text(
                """schema_version: \"1.0\"
benchmark_id: lifechoices
execution: {repetitions: 1, max_workers: 1}
roles:
  evaluated_model:
    backend: vllm
    base_url: http://127.0.0.1:8000/v1
    api_key_env: UNUSED_TEST_KEY
    require_api_key: false
    model: candidate-test-model
    model_revision: candidate-test-revision
    generation: {temperature: 0, max_tokens: 32}
routing: {default_role: evaluated_model}
prompts:
  character_choice: {source: official_repository, revision: test-prompt-v1, locator: test-only}
""",
                encoding="utf-8",
            )
            cases, _ = load_local_cases(load_import_spec(manifest_path))
            responses = {f"{cases[0].case_id}:choice": '{"choice":"A"}'}
            with patch(
                "sim_eval.generic_runner.build_role_routed_backend",
                return_value=ReplayBackend(responses),
            ):
                summary = run_generic_import(
                    manifest_path=manifest_path,
                    runtime_config_path=config_path,
                    output_directory=directory / "artifacts",
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                )
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["completed_count"], 1)
            self.assertEqual(summary["failed_count"], 0)
            self.assertEqual(summary["metrics"]["lifechoices.accuracy"]["value"], 1.0)
            self.assertTrue((directory / "artifacts" / summary["artifacts"]["records"]).is_file())

            # A changed thinking mode is a changed experiment, even if generation
            # parameters and model identity are otherwise identical.
            config_path.write_text(config_path.read_text().replace(
                "    backend: vllm", "    backend: vllm\n    extra_body: {chat_template_kwargs: {enable_thinking: true}}"
            ))
            counter = Mock()
            counter.count_content.return_value = 1
            counter.count_prompt.return_value = 10
            counter.identity.return_value = {"kind": "fixture"}
            with patch(
                "sim_eval.generic_runner.build_role_routed_backend",
                return_value=ReplayBackend(responses),
            ), patch(
                "sim_eval.backends.episode_budget.get_shared_huggingface_token_counter",
                return_value=counter,
            ):
                changed = run_generic_import(
                    manifest_path=manifest_path,
                    runtime_config_path=config_path,
                    output_directory=directory / "artifacts",
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                )
                resumed = run_generic_import(
                    manifest_path=manifest_path,
                    runtime_config_path=config_path,
                    output_directory=directory / "artifacts",
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                )
            self.assertNotEqual(changed["run_id"], summary["run_id"])
            self.assertEqual(resumed["run_id"], changed["run_id"])
            self.assertEqual(changed["status"], "completed")
            self.assertTrue((directory / "artifacts" / summary["artifacts"]["records"]).is_file())

    def test_resume_allows_support_model_fallback_and_audits_both_attempts(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            manifest_path = directory / "mirror_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "mirrorbench",
                        "path": str(ROOT / "tests" / "fixtures" / "mirrorbench.jsonl"),
                        "format": "jsonl",
                        "source_kind": "synthetic_fixture",
                        "source_revision": "mirror-support-fallback-test-v1",
                        "split": "fixture",
                    }
                ),
                encoding="utf-8",
            )
            config_path = directory / "mirror_runtime.yaml"
            config_path.write_text(
                self._mirror_runtime(assistant_model="assistant-primary", judge_model="judge-primary"),
                encoding="utf-8",
            )
            cases, _ = load_local_cases(load_import_spec(manifest_path))
            replay_responses = {}
            adapter = MirrorBenchAdapter()
            for case in cases:
                replay_responses.update(adapter.replay_responses(case, seed=20260819))

            with patch(
                "sim_eval.generic_runner.build_role_routed_backend",
                return_value=ReplayBackend(replay_responses, errors={"*": "timeout"}),
            ):
                failed = run_generic_import(
                    manifest_path=manifest_path,
                    runtime_config_path=config_path,
                    output_directory=directory / "artifacts",
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                    limit=1,
                )
            self.assertEqual(failed["failed_count"], 1)

            config_path.write_text(
                self._mirror_runtime(
                    assistant_model="assistant-fallback",
                    judge_model="judge-fallback",
                ),
                encoding="utf-8",
            )
            with patch(
                "sim_eval.generic_runner.build_role_routed_backend",
                return_value=ReplayBackend(replay_responses),
            ):
                with self.assertRaisesRegex(ConfigurationError, "fallback_reason"):
                    run_generic_import(
                        manifest_path=manifest_path,
                        runtime_config_path=config_path,
                        output_directory=directory / "artifacts",
                        catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                        limit=1,
                    )

            config_path.write_text(
                self._mirror_runtime(
                    assistant_model="assistant-fallback",
                    judge_model="judge-fallback",
                    fallback_reason="primary_models_blocked_by_safety_policy",
                ),
                encoding="utf-8",
            )
            with patch(
                "sim_eval.generic_runner.build_role_routed_backend",
                return_value=ReplayBackend(replay_responses),
            ):
                resumed = run_generic_import(
                    manifest_path=manifest_path,
                    runtime_config_path=config_path,
                    output_directory=directory / "artifacts",
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                    limit=1,
                )

            self.assertEqual(resumed["run_id"], failed["run_id"])
            self.assertEqual(resumed["completed_count"], 1)
            self.assertEqual(resumed["failed_count"], 0)
            support = resumed["support_role_provenance"]
            self.assertTrue(support["mixed_models"])
            self.assertTrue(support["roles"]["fixed_assistant"]["fallback_recorded"])
            self.assertTrue(support["roles"]["judge"]["fallback_recorded"])

            records_path = directory / "artifacts" / resumed["artifacts"]["records"]
            records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["checkpoint_attempt"], 0)
            self.assertEqual(records[1]["checkpoint_attempt"], 1)
            self.assertEqual(
                records[0]["metadata"]["execution_provenance"]["support_roles"]["fixed_assistant"]["model"],
                "assistant-primary",
            )
            self.assertEqual(
                records[1]["metadata"]["execution_provenance"]["support_roles"]["fixed_assistant"]["model"],
                "assistant-fallback",
            )


if __name__ == "__main__":
    unittest.main()
