"""Offline CLI operations for source probing and fixture sample generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import atomic_write_json, atomic_write_text
from ..catalog import load_and_validate, load_catalog
from ..errors import ConfigurationError
from ..json_utils import canonical_json, jsonable, sha256_digest
from .availability import acquisition_states
from .loaders import load_fixture_suite, load_import_spec, load_local_cases
from .sampling import DeterministicStratifiedSampler, resolve_sampling_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_output(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    else:
        output = output.resolve()
    return output


def source_status(catalog_path: str | Path) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    return {
        "catalog_revision": catalog.catalog_revision,
        "benchmarks": {benchmark_id: jsonable(state) for benchmark_id, state in acquisition_states(catalog).items()},
    }


def probe_import(manifest_path: str | Path) -> dict[str, Any]:
    spec = load_import_spec(manifest_path)
    cases, source_manifest = load_local_cases(spec)
    return {
        "status": "valid",
        "benchmark_id": spec.benchmark_id,
        "source_kind": spec.source_kind,
        "source_revision": spec.source_revision,
        "split": spec.split,
        "case_count": len(cases),
        "group_count": len({case.group_id for case in cases}),
        "source_manifest_digest": source_manifest.digest,
        "file_hashes": dict(source_manifest.file_hashes),
        "transformations": jsonable(source_manifest.transformations),
        "source_metadata": jsonable(source_manifest.metadata),
    }


def sample_fixture_suite(
    *,
    profile: str,
    fixture_directory: str | Path,
    output_directory: str | Path,
    catalog_path: str | Path,
    sampling_path: str | Path,
) -> dict[str, Any]:
    if profile != "offline_smoke":
        raise ConfigurationError(
            "synthetic fixtures may only use profile=offline_smoke; they cannot satisfy default or canonical targets"
        )
    catalog, profiles = load_and_validate(catalog_path, sampling_path)
    fixtures = load_fixture_suite(fixture_directory)
    if set(fixtures) != set(catalog.benchmarks):
        raise ConfigurationError("fixture suite and catalog benchmark IDs differ")
    output = _safe_output(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    suite_entries: dict[str, Any] = {}
    for benchmark_id in sorted(catalog.benchmarks):
        cases, source_manifest = fixtures[benchmark_id]
        plan = resolve_sampling_plan(profiles, profile, benchmark_id)
        selected, sample_manifest = DeterministicStratifiedSampler(plan, source_manifest).select(cases)
        benchmark_dir = output / benchmark_id
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        source_path = benchmark_dir / "source_manifest.json"
        sample_path = benchmark_dir / "sample_manifest.json"
        cases_path = benchmark_dir / "selected_cases.jsonl"
        atomic_write_json(source_path, source_manifest)
        atomic_write_json(sample_path, sample_manifest)
        atomic_write_text(cases_path, "".join(canonical_json(case) + "\n" for case in selected))
        suite_entries[benchmark_id] = {
            "source_manifest": str(source_path.relative_to(output)),
            "sample_manifest": str(sample_path.relative_to(output)),
            "selected_cases": str(cases_path.relative_to(output)),
            "source_manifest_digest": source_manifest.digest,
            "sample_manifest_digest": sample_manifest.digest,
            "selected_case_count": len(selected),
            "selected_group_count": len(sample_manifest.selected_group_ids),
            "result_label": sample_manifest.result_label,
        }
    suite_payload = {
        "schema_version": "1.0",
        "catalog_revision": catalog.catalog_revision,
        "profile": profile,
        "result_label": profiles.profiles[profile]["result_label"],
        "benchmark_count": len(suite_entries),
        "benchmarks": suite_entries,
    }
    suite_payload["fingerprint"] = sha256_digest(suite_payload)
    atomic_write_json(output / "sampling_manifest.json", suite_payload)
    return suite_payload
