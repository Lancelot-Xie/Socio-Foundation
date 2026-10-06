import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_eval.backends.replay import ReplayBackend
from sim_eval.data.loaders import load_import_spec, load_local_cases
from sim_eval.external_runner import run_userlm_import


ROOT = Path(__file__).resolve().parents[1]


class UserLMLiveRunnerBatchingTests(unittest.TestCase):
    def test_runner_flushes_a_short_prism_detector_tail_as_one_batch(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            records_path = directory / "userlm_section3.jsonl"
            records = []
            for index in range(2):
                records.append(
                    {
                        "source_id": f"prism-batch-{index}",
                        "group_source_id": f"prism-batch-group-{index}",
                        "split": "fixture",
                        "input": {
                            "variant": "intrinsic_prism",
                            "execution_mode": "section3_single_user_turn",
                            "intent": "prepare for an exam",
                            "conversation_history": (
                                "<user>: How should I study?\n"
                                "<assistant>: Make a schedule.\n"
                            ),
                            "turn": index,
                            "is_last_turn": False,
                        },
                        "gold": None,
                        "strata": {
                            "variant": "intrinsic_prism",
                            "source_task": "prism",
                            "intent": "prepare_for_exam",
                            "required_information_pattern": "not_applicable",
                        },
                    }
                )
            records_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            manifest_path = directory / "import_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "userlm",
                        "path": records_path.name,
                        "format": "jsonl",
                        "source_kind": "synthetic_fixture",
                        "source_revision": "userlm-prism-batch-runner-test-v1",
                        "split": "fixture",
                    }
                ),
                encoding="utf-8",
            )
            config_path = directory / "runtime.yaml"
            role = """backend: vllm
    base_url: http://127.0.0.1:8000/v1
    api_key_env: UNUSED_TEST_KEY
    require_api_key: false
    model: test-model
    model_revision: test-model-revision
    generation: {temperature: 0.0, max_tokens: 256}"""
            config_path.write_text(
                f'''schema_version: "1.0"
benchmark_id: userlm
execution:
  max_workers: 2
  episode_max_output_tokens: 32768
roles:
  evaluated_user:
    {role}
  fixed_assistant:
    {role}
  intent_judge:
    {role}
  shard_judge:
    {role}
routing: {{default_role: evaluated_user}}
prompts:
  section3_user: {{source: official_paper, revision: test-prompt-v1, locator: test-only}}
resources:
  lic_repetitions: 1
  ai_text_detector:
    enabled: true
    backend: local_greyscope_seqcls
    model_id: yaoandy107/greyscope-qwen3.5-4b
    model_revision: bb25d4158a6795c6fce225156c0e330084ffb665
    model_path: /unused/test/greyscope
    batch_size: 4
    device: cpu
  code_execution:
    enabled: false
    backend: local_guarded
''',
                encoding="utf-8",
            )
            cases, _ = load_local_cases(load_import_spec(manifest_path))
            outputs = (
                "Please suggest a focused study routine.",
                "How can that routine stay manageable?",
            )
            responses = {
                f"{case.case_id}:section3:user_turn": output
                for case, output in zip(cases, outputs)
            }
            observed_batches = []
            active_lock = threading.Lock()
            active_requests = 0
            max_active_requests = 0
            rendezvous = threading.Barrier(2)

            class ConcurrentReplayBackend(ReplayBackend):
                def generate(self, request):
                    nonlocal active_requests, max_active_requests
                    tracked = request.metadata.get("actor") == "evaluated_user"
                    if tracked:
                        with active_lock:
                            active_requests += 1
                            max_active_requests = max(max_active_requests, active_requests)
                        try:
                            rendezvous.wait(timeout=2)
                            return super().generate(request)
                        finally:
                            with active_lock:
                                active_requests -= 1
                    return super().generate(request)

            class FakeBatchDetector:
                def __init__(self, config):
                    self.config = config

                def score_batch(self, texts):
                    observed_batches.append(tuple(texts))
                    return tuple(
                        {
                            "human_likelihood": value,
                            "model_revision": self.config.model_revision,
                            "batch": {
                                "configured_case_batch_size": self.config.batch_size,
                                "observed_case_batch_size": len(texts),
                                "position": position,
                            },
                        }
                        for position, value in enumerate((0.83, 0.29))
                    )

            with (
                patch(
                    "sim_eval.external_runner.GreyscopeLocalScorer",
                    FakeBatchDetector,
                ),
                patch(
                    "sim_eval.external_runner.build_role_routed_backend",
                    return_value=ConcurrentReplayBackend(responses),
                ),
            ):
                summary = run_userlm_import(
                    manifest_path=manifest_path,
                    runtime_config_path=config_path,
                    output_directory=directory / "artifacts",
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                )
                first_batches = tuple(observed_batches)
                observed_batches.clear()
                resumed = run_userlm_import(
                    manifest_path=manifest_path,
                    runtime_config_path=config_path,
                    output_directory=directory / "artifacts",
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                )

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["max_workers"], 2)
            self.assertEqual(max_active_requests, 2)
            self.assertEqual(resumed["run_id"], summary["run_id"])
            self.assertEqual(len(first_batches), 1)
            self.assertEqual(set(first_batches[0]), set(outputs))
            self.assertEqual(observed_batches, [])
            detector_metric = summary["metrics"][
                "userlm.intrinsic.ai_detector_human_likelihood"
            ]
            self.assertEqual(detector_metric["denominator"], 2)
            self.assertAlmostEqual(detector_metric["value"], 0.56)
            result_path = directory / "artifacts" / summary["artifacts"]["records"]
            result_records = [
                json.loads(line)
                for line in result_path.read_text(encoding="utf-8").splitlines()
            ]
            observed_scores = {}
            for record in result_records:
                detector = next(
                    metric
                    for metric in record["metrics"]
                    if metric["name"]
                    == "userlm.intrinsic.ai_detector_human_likelihood"
                )
                observed_scores[record["prediction"]["user_turn"]] = detector["value"]
            self.assertEqual(
                tuple(record["prediction"]["user_turn"] for record in result_records),
                first_batches[0],
            )
            self.assertEqual(
                observed_scores,
                {
                    first_batches[0][0]: 0.83,
                    first_batches[0][1]: 0.29,
                },
            )


if __name__ == "__main__":
    unittest.main()
