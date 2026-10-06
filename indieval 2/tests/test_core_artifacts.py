import json
import tempfile
import unittest
from pathlib import Path

from sim_eval.artifacts import CheckpointStore
from sim_eval.contracts import CaseResult, ErrorState, ResultStatus, RunIdentityInput, RunManifest
from sim_eval.errors import ArtifactError, DuplicateResultError, ResumeConflictError


def make_identity(prompt_revision="prompt-v1"):
    return RunIdentityInput(
        framework_version="0.1.0",
        benchmark_id="fantom",
        source_revision="source-rev",
        split="test",
        profile="default",
        sample_manifest_digest="sample-digest",
        seed=12,
        backend="replay",
        model="fixture",
        decoding={"temperature": 0},
        prompt_revision=prompt_revision,
        scorer_revision="scorer-v1",
        judge={},
        environment={},
        assistant_or_partner={},
    )


def make_manifest(prompt_revision="prompt-v1"):
    return RunManifest.create(
        make_identity(prompt_revision),
        catalog_revision="2026-08-12",
        result_label="synthetic_offline_smoke_not_a_benchmark_score",
        requested_case_count=2,
        selected_case_count=2,
        selected_group_count=1,
    )


class ArtifactTests(unittest.TestCase):
    def test_checkpoint_sanitizes_non_response_surrogates_and_remains_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            manifest = make_manifest()
            store.initialize(manifest)
            result = CaseResult(
                run_id=manifest.run_id,
                benchmark_id="fantom",
                case_id="case-1",
                group_id="conversation-1",
                repetition=0,
                status=ResultStatus.COMPLETED,
                prediction={
                    "normal": "中文 😀",
                    "paired": "\ud83d\ude00",
                    "unpaired": "bad:\udc50",
                },
            )
            store.append_result(result)
            raw = (Path(directory) / "records.jsonl").read_bytes()
            raw.decode("utf-8")
            row = list(store.iter_record_dicts())[0]
            self.assertEqual(row["prediction"]["normal"], "中文 😀")
            self.assertEqual(row["prediction"]["paired"], "😀")
            self.assertEqual(row["prediction"]["unpaired"], "bad:�")
            audit = row["metadata"]["artifact_unicode_sanitization"]
            self.assertEqual(audit["unpaired_surrogates_replaced"], 1)
            self.assertEqual(audit["surrogate_pairs_normalized"], 1)

            resumed = CheckpointStore(directory)
            resumed.initialize(manifest)
            self.assertEqual(resumed.completed_keys(), {("case-1", 0)})

    def test_checkpoint_preserves_success_and_structured_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            manifest = make_manifest()
            store.initialize(manifest)
            success = CaseResult(
                run_id=manifest.run_id,
                benchmark_id="fantom",
                case_id="case-1",
                group_id="conversation-1",
                repetition=0,
                status=ResultStatus.COMPLETED,
                prediction="A",
            )
            failure = CaseResult(
                run_id=manifest.run_id,
                benchmark_id="fantom",
                case_id="case-2",
                group_id="conversation-1",
                repetition=0,
                status=ResultStatus.FAILED,
                error=ErrorState(stage="parse", kind="invalid_answer", message="missing option"),
            )
            store.append_result(success)
            store.append_result(failure)
            self.assertEqual(store.completed_keys(), {("case-1", 0)})
            rows = list(store.iter_record_dicts())
            self.assertEqual(rows[1]["error"]["kind"], "invalid_answer")
            self.assertEqual([row["checkpoint_attempt"] for row in rows], [0, 0])
            with self.assertRaises(DuplicateResultError):
                store.append_result(success)

            resumed = CheckpointStore(directory)
            resumed.initialize(manifest)
            self.assertEqual(resumed.completed_keys(), store.completed_keys())

            retry = CaseResult(
                run_id=manifest.run_id,
                benchmark_id="fantom",
                case_id="case-2",
                group_id="conversation-1",
                repetition=0,
                status=ResultStatus.COMPLETED,
                prediction="B",
            )
            resumed.append_result(retry)
            self.assertEqual(resumed.completed_keys(), {("case-1", 0), ("case-2", 0)})
            rows = list(resumed.iter_record_dicts())
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[2]["checkpoint_attempt"], 1)
            latest = list(resumed.iter_latest_record_dicts())
            self.assertEqual(len(latest), 2)
            self.assertEqual(latest[1]["status"], "completed")
            self.assertEqual(latest[1]["prediction"], "B")
            with self.assertRaises(DuplicateResultError):
                resumed.append_result(failure)

    def test_resume_rejects_incompatible_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            CheckpointStore(directory).initialize(make_manifest("prompt-v1"))
            with self.assertRaisesRegex(ResumeConflictError, "incompatible"):
                CheckpointStore(directory).initialize(make_manifest("prompt-v2"))

    def test_incomplete_tail_is_ignored_but_invalid_middle_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            store = CheckpointStore(path)
            manifest = make_manifest()
            store.initialize(manifest)
            valid = {
                "case_id": "case-1",
                "repetition": 0,
                "run_id": manifest.run_id,
                "status": "completed",
            }
            (path / "records.jsonl").write_bytes((json.dumps(valid) + "\n{partial").encode("utf-8"))
            resumed = CheckpointStore(path)
            resumed.initialize(manifest)
            self.assertEqual(resumed.completed_keys(), {("case-1", 0)})

            (path / "records.jsonl").write_text("{bad}\n" + json.dumps(valid) + "\n", encoding="utf-8")
            with self.assertRaises(ArtifactError):
                list(CheckpointStore(path).iter_record_dicts())


if __name__ == "__main__":
    unittest.main()
