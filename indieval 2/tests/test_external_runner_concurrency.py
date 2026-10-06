import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from sim_eval.contracts import ModelResponse
from sim_eval.integrations.tau_usi_protocol import fixture_survey_options
from sim_eval.data.loaders import load_import_spec, load_local_cases
from sim_eval.external_runner import run_tau_usi_import
from sim_eval.interfaces import ModelBackend
from sim_eval.runtime_config import load_config_document


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT.parent / "temp_data_qa" / "datasets"
if not DATASETS.is_dir():
    DATASETS = ROOT.parent.parent / "datasets"


class DeterministicTauBackend(ModelBackend):
    name = "deterministic_tau"

    def __init__(self, *, require_first_turn_overlap: bool) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self._barrier = threading.Barrier(2) if require_first_turn_overlap else None

    def generate(self, request):
        first_user_turn = (
            request.metadata.get("actor") == "evaluated_user"
            and request.metadata.get("stage") != "post_interaction_survey"
            and str(request.request_id).endswith(":user:0")
        )
        if first_user_turn:
            with self._lock:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
            try:
                if self._barrier is not None:
                    self._barrier.wait(timeout=5)
            finally:
                with self._lock:
                    self._active -= 1

        if request.metadata.get("stage") == "post_interaction_survey":
            text = json.dumps(fixture_survey_options(
                {
                    "task_success": 0,
                    "efficiency": 0,
                    "question_amount": 0,
                    "answer_effort": 0,
                    "human_likeness": 0,
                    "interaction_flow": 0,
                    "overall": 0,
                    "reuse_intent": 0,
                }
            ))
        elif request.metadata.get("actor") == "fixed_assistant":
            text = "How can I help with that request?"
        elif str(request.request_id).endswith(":user:0"):
            text = "Hello, I need some help with my request."
        else:
            text = "###STOP###"
        return ModelResponse(
            text=text,
            finish_reason="stop",
            latency_ms=0.0,
            raw={"_sim_eval": {"protocol": "deterministic_test"}},
        )


class ExternalRunnerConcurrencyTests(unittest.TestCase):
    @staticmethod
    def _runtime_config(directory: Path, *, max_workers: int) -> Path:
        source_path = ROOT / "sim_eval/resources" / "tau_usi_runtime.example.yaml"
        _, raw = load_config_document(source_path)
        document = dict(raw)
        document["target_user"] = {
            **dict(document["target_user"]),
            "model": "deterministic-user",
            "model_revision": "deterministic-user-v1",
            "require_api_key": False,
        }
        document["fixed_assistant"] = {
            **dict(document["fixed_assistant"]),
            "model": "deterministic-assistant",
            "model_revision": "deterministic-assistant-v1",
            "require_api_key": False,
        }
        scoring = dict(document["scoring"])
        for key in ("annotation_local_path", "difficulty_local_path"):
            value = Path(str(scoring[key]))
            scoring[key] = str((source_path.parent / value).resolve())
        document["scoring"] = scoring
        document["limits"] = {**dict(document["limits"]), "max_workers": max_workers}
        destination = directory / f"tau_workers_{max_workers}.yaml"
        destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return destination

    @staticmethod
    def _semantic_records(output: Path, summary):
        records_path = output / summary["artifacts"]["records"]
        records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
        return [
            {
                "case_id": record["case_id"],
                "status": record["status"],
                "prediction": record["prediction"],
                "metrics": record["metrics"],
                "trace": record["trace"],
            }
            for record in records
        ]

    def test_tau_serial_and_parallel_runs_are_semantically_equivalent(self) -> None:
        manifest = DATASETS / "derived" / "tau_usi_eval_v1" / "import_manifest.json"
        if not manifest.is_file():
            self.skipTest("tau-USI derived import is not installed")
        cases, _ = load_local_cases(load_import_spec(manifest))
        selected_case_ids = tuple(case.case_id for case in cases[:2])

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            serial_output = directory / "serial"
            parallel_output = directory / "parallel"
            serial_backend = DeterministicTauBackend(require_first_turn_overlap=False)
            with patch(
                "sim_eval.external_runner._build_role_backend",
                return_value=serial_backend,
            ):
                serial = run_tau_usi_import(
                    manifest_path=manifest,
                    runtime_config_path=self._runtime_config(directory, max_workers=1),
                    output_directory=serial_output,
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                    selected_case_ids=selected_case_ids,
                )

            parallel_backend = DeterministicTauBackend(require_first_turn_overlap=True)
            with patch(
                "sim_eval.external_runner._build_role_backend",
                return_value=parallel_backend,
            ):
                parallel = run_tau_usi_import(
                    manifest_path=manifest,
                    runtime_config_path=self._runtime_config(directory, max_workers=2),
                    output_directory=parallel_output,
                    catalog_path=ROOT / "sim_eval/resources" / "benchmarks.json",
                    selected_case_ids=selected_case_ids,
                )

            self.assertEqual(serial["status"], "completed")
            self.assertEqual(parallel["status"], "completed")
            self.assertEqual(serial["metrics"], parallel["metrics"])
            self.assertEqual(
                self._semantic_records(serial_output, serial),
                self._semantic_records(parallel_output, parallel),
            )
            self.assertEqual(parallel_backend.max_active, 2)


if __name__ == "__main__":
    unittest.main()
