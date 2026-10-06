import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from sim_eval.catalog import REQUESTED_BENCHMARK_IDS, load_and_validate, load_catalog
from sim_eval.cli import main
from sim_eval.errors import ConfigurationError


ROOT = Path(__file__).resolve().parents[1]


class CatalogTests(unittest.TestCase):
    def test_checked_in_catalog_and_profiles_cover_requested_suite(self) -> None:
        catalog, profiles = load_and_validate(
            ROOT / "sim_eval/resources" / "benchmarks.json",
            ROOT / "sim_eval/resources" / "sampling_profiles.json",
        )
        self.assertEqual(set(catalog.benchmarks), set(REQUESTED_BENCHMARK_IDS))
        self.assertEqual(set(profiles.profiles), {"canonical", "default", "offline_smoke"})
        for profile in profiles.profiles.values():
            self.assertEqual(set(profile["benchmarks"]), set(REQUESTED_BENCHMARK_IDS))

    def test_catalog_rejects_primary_metric_not_in_metric_list(self) -> None:
        payload = json.loads((ROOT / "sim_eval/resources" / "benchmarks.json").read_text(encoding="utf-8"))
        payload["benchmarks"][0]["official_protocol"]["primary_metric"] = "fabricated"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "primary_metric"):
                load_catalog(path)

    def test_cli_validation_is_machine_readable(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "catalog",
                    "validate",
                    "--catalog",
                    str(ROOT / "sim_eval/resources" / "benchmarks.json"),
                    "--sampling",
                    str(ROOT / "sim_eval/resources" / "sampling_profiles.json"),
                ]
            )
        result = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["benchmark_count"], 13)


if __name__ == "__main__":
    unittest.main()
