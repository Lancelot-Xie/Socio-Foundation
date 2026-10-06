import json
import tempfile
import unittest
from pathlib import Path

from sim_eval.catalog import REQUESTED_BENCHMARK_IDS, load_catalog
from sim_eval.data.availability import acquisition_states
from sim_eval.data.ids import derive_case_id
from sim_eval.data.loaders import ImportSpec, load_fixture_suite, load_local_cases, normalize_record
from sim_eval.errors import ConfigurationError, ValidationError


ROOT = Path(__file__).resolve().parents[1]


class DataIngestionTests(unittest.TestCase):
    def test_original_fixture_suite_covers_all_catalog_ids_and_schemas(self) -> None:
        suite = load_fixture_suite(ROOT / "tests" / "fixtures")
        self.assertEqual(set(suite), set(REQUESTED_BENCHMARK_IDS))
        for benchmark_id, (cases, manifest) in suite.items():
            self.assertTrue(cases, benchmark_id)
            self.assertEqual(manifest.source_kind, "synthetic_fixture")
            self.assertEqual({case.benchmark_id for case in cases}, {benchmark_id})
            self.assertTrue(all(case.metadata["source_kind"] == "synthetic_fixture" for case in cases))

    def test_case_id_uses_source_identity_not_gold(self) -> None:
        spec = ImportSpec(
            benchmark_id="social_r1",
            path=Path("unused.jsonl"),
            format="jsonl",
            source_kind="synthetic_fixture",
            source_revision="rev-1",
            split="fixture",
        )
        base = {
            "source_id": "stable-question-id",
            "group_source_id": "stable-question-id",
            "split": "fixture",
            "input": {"question": "Synthetic?", "options": ["A", "B", "C", "D"], "atoms_dimension": "belief"},
            "strata": {"atoms_dimension": "belief"},
            "gold": "A",
        }
        changed = {**base, "gold": "B"}
        self.assertEqual(normalize_record(spec, base, 1).case_id, normalize_record(spec, changed, 1).case_id)
        self.assertEqual(
            derive_case_id("social_r1", "rev-1", "stable-question-id"),
            normalize_record(spec, base, 1).case_id,
        )

    def test_local_jsonl_import_checks_license_checksum_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "social.jsonl"
            record = {
                "source_id": "q1",
                "group_source_id": "q1",
                "split": "test",
                "input": {"question": "Choose.", "options": ["A", "B", "C", "D"], "atoms_dimension": "belief"},
                "gold": "A",
                "strata": {"atoms_dimension": "belief"},
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            unacknowledged = ImportSpec(
                benchmark_id="social_r1",
                path=path,
                format="jsonl",
                source_kind="official",
                source_revision="release-sha",
                split="test",
            )
            with self.assertRaisesRegex(ConfigurationError, "license_acknowledged"):
                load_local_cases(unacknowledged)
            acknowledged = ImportSpec(
                **{**unacknowledged.__dict__, "license_acknowledged": True}
            )
            cases, source = load_local_cases(acknowledged)
            self.assertEqual(len(cases), 1)
            self.assertNotIn(str(path), source.file_hashes)
            self.assertIn(path.name, source.file_hashes)
            bad_checksum = ImportSpec(
                **{**acknowledged.__dict__, "checksum_sha256": "0" * 64}
            )
            with self.assertRaisesRegex(ValidationError, "checksum mismatch"):
                load_local_cases(bad_checksum)

    def test_missing_normalized_field_fails_schema_probe(self) -> None:
        spec = ImportSpec(
            benchmark_id="social_r1",
            path=Path("unused"),
            format="jsonl",
            source_kind="synthetic_fixture",
            source_revision="rev",
            split="fixture",
        )
        record = {
            "source_id": "q",
            "input": {"question": "Missing options", "atoms_dimension": "belief"},
            "gold": "A",
            "strata": {"atoms_dimension": "belief"},
        }
        with self.assertRaisesRegex(ValidationError, "missing input fields"):
            normalize_record(spec, record, 1)

    def test_catalog_exposes_explicit_acquisition_state_for_every_id(self) -> None:
        catalog = load_catalog(ROOT / "sim_eval/resources" / "benchmarks.json")
        states = acquisition_states(catalog)
        self.assertEqual(set(states), set(REQUESTED_BENCHMARK_IDS))
        self.assertIn("gated", states["tau_usi"].state)
        self.assertIn("local", states["behaviorchain"].state)
        self.assertTrue(states["lifechoices"].instructions)


if __name__ == "__main__":
    unittest.main()
