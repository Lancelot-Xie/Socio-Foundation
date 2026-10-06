from __future__ import annotations

import hashlib
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from sim_eval import cli
from sim_eval.catalog import REQUESTED_BENCHMARK_IDS
from sim_eval.delivery import verify_delivery
from sim_eval.errors import ConfigurationError
from sim_eval.reporting import NON_LEADERBOARD_LABEL, write_suite_report
from sim_eval.runner import run_fixture_suite


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "sim_eval/resources" / "benchmarks.json"
SAMPLING = ROOT / "sim_eval/resources" / "sampling_profiles.json"
FIXTURES = ROOT / "tests" / "fixtures"


def invoke_cli(arguments: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli.main(arguments)
    return code, stdout.getvalue(), stderr.getvalue()


def file_digest(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload.splitlines()), hashlib.sha256(payload).hexdigest()


class CliSurfaceTests(unittest.TestCase):
    def test_run_without_benchmarks_selects_all_thirteen(self) -> None:
        with patch("sim_eval.cli.run_fixture_suite", return_value={"status": "ok"}) as mocked:
            code, stdout, stderr = invoke_cli(
                [
                    "run",
                    "--profile",
                    "offline_smoke",
                    "--backend",
                    "replay",
                    "--output",
                    "artifacts/mock-all",
                    "--catalog",
                    str(CATALOG),
                    "--sampling",
                    str(SAMPLING),
                    "--fixtures",
                    str(FIXTURES),
                ]
            )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout), {"status": "ok"})
        self.assertEqual(mocked.call_args.kwargs["benchmark_ids"], sorted(REQUESTED_BENCHMARK_IDS))

    def test_list_exposes_exact_catalog_and_registered_adapters(self) -> None:
        code, stdout, stderr = invoke_cli(["list", "--catalog", str(CATALOG)])
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["benchmark_count"], 13)
        self.assertEqual({row["id"] for row in payload["benchmarks"]}, set(REQUESTED_BENCHMARK_IDS))
        self.assertTrue(all(row["adapter_registered"] for row in payload["benchmarks"]))

    def test_doctor_is_offline_and_optional_dependencies_are_nonfatal(self) -> None:
        code, stdout, stderr = invoke_cli(
            [
                "doctor",
                "--catalog",
                str(CATALOG),
                "--sampling",
                str(SAMPLING),
                "--fixtures",
                str(FIXTURES),
            ]
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["offline"])
        self.assertEqual(payload["network_calls"], 0)

    def test_static_delivery_verification_passes_before_artifact_generation(self) -> None:
        result = verify_delivery(ROOT, require_artifacts=False)
        self.assertEqual(result["status"], "passed", result)


class FullSuiteIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix=".integration-", dir=ROOT)
        cls.workspace = Path(cls.temporary.name)
        cls.run_directory = cls.workspace / "run"
        cls.report_path = cls.workspace / "reports" / "smoke.md"
        cls.summary = run_fixture_suite(
            benchmark_ids=sorted(REQUESTED_BENCHMARK_IDS),
            profile="offline_smoke",
            backend_name="replay",
            output_directory=cls.run_directory,
            fixture_directory=FIXTURES,
            catalog_path=CATALOG,
            sampling_path=SAMPLING,
            model="offline-replay",
        )
        write_suite_report(
            run_directory=cls.run_directory,
            output_path=cls.report_path,
            catalog_path=CATALOG,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_all_thirteen_adapters_emit_complete_non_leaderboard_artifacts(self) -> None:
        self.assertEqual(set(self.summary["benchmarks"]), set(REQUESTED_BENCHMARK_IDS))
        self.assertEqual(self.summary["benchmark_count"], 13)
        self.assertEqual(self.summary["result_label"], NON_LEADERBOARD_LABEL)
        self.assertTrue(self.summary["offline_only"])
        self.assertEqual(self.summary["totals"], {"result_count": 48, "completed_count": 48, "failed_count": 0})
        forbidden = {"suite_score", "overall_score", "average_score", "leaderboard_score"}
        self.assertFalse(forbidden & set(self.summary))
        for benchmark_id, entry in self.summary["benchmarks"].items():
            self.assertGreater(entry["result_count"], 0, benchmark_id)
            self.assertEqual(entry["failed_count"], 0, benchmark_id)
            self.assertEqual(entry["errors"], [], benchmark_id)
            for relative in entry["artifact_paths"].values():
                self.assertTrue((self.run_directory / relative).is_file(), relative)

    def test_report_labels_scope_preserves_native_metrics_and_has_working_links(self) -> None:
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn("NON-LEADERBOARD EVIDENCE", report)
        self.assertIn("Cross-suite raw average: **not computed**", report)
        for benchmark_id in REQUESTED_BENCHMARK_IDS:
            self.assertIn(f"`{benchmark_id}`", report)
        links = re.findall(r"\[[^]]+\]\(([^)]+)\)", report)
        self.assertGreaterEqual(len(links), 13 * 5)
        for link in links:
            self.assertTrue((self.report_path.parent / link).resolve().is_file(), link)

    def test_matching_resume_is_byte_stable_and_does_not_duplicate_records(self) -> None:
        records = sorted(self.run_directory.glob("*/*/records.jsonl"))
        before = {path.relative_to(self.run_directory): file_digest(path) for path in records}
        summary_before = (self.run_directory / "suite_summary.json").read_bytes()
        code, stdout, stderr = invoke_cli(
            [
                "resume",
                "--output",
                str(self.run_directory),
                "--catalog",
                str(CATALOG),
                "--sampling",
                str(SAMPLING),
                "--fixtures",
                str(FIXTURES),
            ]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["totals"]["result_count"], 48)
        after = {path.relative_to(self.run_directory): file_digest(path) for path in records}
        self.assertEqual(after, before)
        self.assertEqual((self.run_directory / "suite_summary.json").read_bytes(), summary_before)

    def test_incompatible_resume_model_or_selection_is_rejected_before_writes(self) -> None:
        before = (self.run_directory / "suite_summary.json").read_bytes()
        code, _, stderr = invoke_cli(
            [
                "resume",
                "--output",
                str(self.run_directory),
                "--model",
                "changed-model",
                "--catalog",
                str(CATALOG),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("refusing incompatible resume", stderr)
        code, _, stderr = invoke_cli(
            [
                "resume",
                "--output",
                str(self.run_directory),
                "--benchmarks",
                "fantom,social_r1",
                "--catalog",
                str(CATALOG),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("benchmarks:", stderr)
        self.assertEqual((self.run_directory / "suite_summary.json").read_bytes(), before)



if __name__ == "__main__":
    unittest.main()
