"""Offline delivery-integrity checks for the standalone evaluation project."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import latest_checkpoint_record_dicts, load_json
from .backends import registered_backends
from .benchmarks import register_builtin_adapters
from .catalog import REQUESTED_BENCHMARK_IDS, load_and_validate
from .contracts import ResultStatus, case_result_from_dict
from .data.loaders import load_fixture_suite
from .errors import SimEvalError
from .registry import get_adapter, registered_adapters
from .reporting import NON_LEADERBOARD_LABEL


REQUIRED_BACKENDS = {
    "replay",
    "openai",
    "openai_compatible",
    "vllm",
    "huggingface",
    "hf",
}

REQUIRED_DOCUMENTS = ("README.md", "docs/architecture.md", "docs/protocols.md", "docs/data_and_runtime.md", "docs/anonymity_and_validation.md")


FORBIDDEN_IMPORT_ROOTS = {"verl", "agents", "recipe", "services"}


def _check(check_id: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"id": check_id, "status": "passed" if passed else "failed", "detail": detail}


def _read_records(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{index}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"record at {path}:{index} is not an object")
        records.append(value)
    return records


def _resolved_child(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path must be nonempty text")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes root: {relative}") from exc
    return path


def _static_import_scope(project_root: Path) -> tuple[bool, str]:
    violations: list[str] = []
    for base in (project_root / "sim_eval", project_root / "tools"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (OSError, SyntaxError) as exc:
                violations.append(f"{path.relative_to(project_root)}: unreadable/unparseable ({exc})")
                continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.append(node.module)
                for name in names:
                    if name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                        violations.append(f"{path.relative_to(project_root)} imports protected root {name!r}")
            protected_prefix = str(project_root.parent)
            if protected_prefix in source and str(project_root) not in source:
                violations.append(f"{path.relative_to(project_root)} embeds the outer repository path")
    return not violations, "; ".join(violations) if violations else "no protected outer imports or paths"


def _artifact_checks(project_root: Path, run_relative: str, report_relative: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    run_root = (project_root / run_relative).resolve()
    try:
        run_root.relative_to(project_root)
    except ValueError:
        return [_check("ARTIFACT-SCOPE", False, f"run path escapes project: {run_root}")]
    summary_path = run_root / "suite_summary.json"
    if not summary_path.is_file():
        return [_check("ARTIFACT-SUMMARY", False, f"missing {summary_path}")]
    try:
        summary = load_json(summary_path)
        if not isinstance(summary, Mapping):
            raise ValueError("suite summary is not an object")
        entries = summary.get("benchmarks")
        if not isinstance(entries, Mapping):
            raise ValueError("suite benchmarks is not an object")
    except (SimEvalError, ValueError) as exc:
        return [_check("ARTIFACT-SUMMARY", False, str(exc))]

    checks.append(
        _check(
            "ARTIFACT-COVERAGE",
            set(entries) == set(REQUESTED_BENCHMARK_IDS)
            and int(summary.get("benchmark_count", -1)) == len(REQUESTED_BENCHMARK_IDS),
            f"benchmark IDs={sorted(entries)}",
        )
    )
    checks.append(
        _check(
            "ARTIFACT-LABEL",
            summary.get("profile") == "offline_smoke"
            and summary.get("backend") == "replay"
            and summary.get("result_label") == NON_LEADERBOARD_LABEL
            and summary.get("offline_only") is True,
            f"profile={summary.get('profile')}, backend={summary.get('backend')}, label={summary.get('result_label')}",
        )
    )
    conflated = {"suite_score", "overall_score", "average_score", "leaderboard_score"} & set(summary)
    checks.append(
        _check(
            "ARTIFACT-NO-CONFLATED-SCORE",
            not conflated,
            "no cross-benchmark scalar" if not conflated else f"forbidden fields={sorted(conflated)}",
        )
    )

    total_records = 0
    failures = 0
    artifact_errors: list[str] = []
    for benchmark_id in sorted(REQUESTED_BENCHMARK_IDS):
        entry = entries.get(benchmark_id)
        if not isinstance(entry, Mapping):
            artifact_errors.append(f"{benchmark_id}: summary entry missing")
            continue
        paths = entry.get("artifact_paths")
        if not isinstance(paths, Mapping):
            artifact_errors.append(f"{benchmark_id}: artifact_paths missing")
            continue
        try:
            resolved = {
                name: _resolved_child(run_root, paths.get(name))
                for name in ("run_manifest", "source_manifest", "sample_manifest", "records", "metrics")
            }
            missing = [name for name, path in resolved.items() if not path.is_file()]
            if missing:
                raise ValueError(f"missing artifact files {missing}")
            manifest = load_json(resolved["run_manifest"])
            source = load_json(resolved["source_manifest"])
            sample = load_json(resolved["sample_manifest"])
            metrics = load_json(resolved["metrics"])
            raw_records = _read_records(resolved["records"])
            latest_records = latest_checkpoint_record_dicts(raw_records)
            records = [case_result_from_dict(value) for value in latest_records]
            run_id = entry.get("run_id")
            if not isinstance(manifest, Mapping) or manifest.get("run_id") != run_id:
                raise ValueError("run manifest ID mismatch")
            identity = manifest.get("identity")
            if not isinstance(identity, Mapping) or identity.get("benchmark_id") != benchmark_id:
                raise ValueError("run identity benchmark mismatch")
            if manifest.get("result_label") != NON_LEADERBOARD_LABEL:
                raise ValueError("run manifest lacks smoke-only label")
            if not isinstance(source, Mapping) or source.get("benchmark_id") != benchmark_id:
                raise ValueError("source manifest benchmark mismatch")
            if source.get("source_kind") != "synthetic_fixture":
                raise ValueError("source manifest is not synthetic_fixture")
            if not isinstance(sample, Mapping) or sample.get("profile") != "offline_smoke":
                raise ValueError("sample manifest profile mismatch")
            if sample.get("result_label") != NON_LEADERBOARD_LABEL:
                raise ValueError("sample manifest lacks smoke-only label")
            if not isinstance(metrics, Mapping) or metrics.get("run_id") != run_id or not metrics.get("metrics"):
                raise ValueError("metrics artifact is empty or has wrong run ID")
            if len(records) != int(entry.get("result_count", -1)) or not records:
                raise ValueError("record count is empty or disagrees with suite summary")
            if any(record.run_id != run_id or record.benchmark_id != benchmark_id for record in records):
                raise ValueError("record identity mismatch")
            completed = sum(record.status == ResultStatus.COMPLETED for record in records)
            failed = sum(record.status == ResultStatus.FAILED for record in records)
            if completed != int(entry.get("completed_count", -1)) or failed != int(entry.get("failed_count", -1)):
                raise ValueError("record statuses disagree with suite summary")
            summary_errors = entry.get("errors")
            if not isinstance(summary_errors, Sequence) or isinstance(summary_errors, (str, bytes)):
                raise ValueError("summary errors field is not an array")
            if len(summary_errors) != failed:
                raise ValueError("summary error count disagrees with failed records")
            total_records += len(records)
            failures += failed
        except (OSError, ValueError, SimEvalError, TypeError) as exc:
            artifact_errors.append(f"{benchmark_id}: {exc}")

    checks.append(
        _check(
            "ARTIFACT-PER-BENCHMARK",
            not artifact_errors,
            f"validated {total_records} typed records across {len(REQUESTED_BENCHMARK_IDS)} benchmarks"
            if not artifact_errors
            else "; ".join(artifact_errors),
        )
    )
    totals = summary.get("totals")
    totals_ok = (
        isinstance(totals, Mapping)
        and int(totals.get("result_count", -1)) == total_records
        and int(totals.get("failed_count", -1)) == failures
    )
    checks.append(_check("ARTIFACT-TOTALS", totals_ok, f"records={total_records}, failures={failures}"))

    report_path = (project_root / report_relative).resolve()
    try:
        report_path.relative_to(project_root)
        report = report_path.read_text(encoding="utf-8")
        report_ok = (
            "NON-LEADERBOARD EVIDENCE" in report
            and "Cross-suite raw average" in report
            and all(f"`{benchmark_id}`" in report for benchmark_id in REQUESTED_BENCHMARK_IDS)
        )
        detail = f"report={report_path.relative_to(project_root)}, chars={len(report)}"
    except (OSError, ValueError) as exc:
        report_ok = False
        detail = str(exc)
    checks.append(_check("ARTIFACT-REPORT", report_ok, detail))
    return checks


def verify_delivery(
    project_root: str | Path | None = None,
    *,
    require_artifacts: bool = True,
    run_relative: str = "artifacts/root_smoke",
    report_relative: str = "reports/root_smoke.md",
) -> dict[str, Any]:
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    checks: list[dict[str, Any]] = []

    missing_docs = [relative for relative in REQUIRED_DOCUMENTS if not (root / relative).is_file()]
    checks.append(
        _check(
            "STATIC-DOCS",
            not missing_docs,
            "all required user/protocol documents exist" if not missing_docs else f"missing={missing_docs}",
        )
    )
    try:
        readme = (root / "README.md").read_text(encoding="utf-8")
        required_phrases = (
            "sim_eval catalog validate",
            "sim_eval run",
            "sim_eval resume",
            "sim_eval report",
            "sim_eval doctor",
            "synthetic_offline_smoke_not_a_benchmark_score",
        )
        missing_phrases = [phrase for phrase in required_phrases if phrase not in readme]
        checks.append(
            _check(
                "STATIC-README-WORKFLOW",
                not missing_phrases,
                "install/run/resume/report/doctor workflow present"
                if not missing_phrases
                else f"missing README phrases={missing_phrases}",
            )
        )
    except OSError as exc:
        checks.append(_check("STATIC-README-WORKFLOW", False, str(exc)))

    try:
        catalog, profiles = load_and_validate(root / "sim_eval/resources/benchmarks.json", root / "sim_eval/resources/sampling_profiles.json")
        catalog_ok = set(catalog.benchmarks) == set(REQUESTED_BENCHMARK_IDS)
        profile_ok = set(profiles.profiles) == {"canonical", "default", "offline_smoke"}
        checks.append(
            _check(
                "STATIC-CATALOG",
                catalog_ok and profile_ok,
                f"{len(REQUESTED_BENCHMARK_IDS)} IDs and three required profiles",
            )
        )
    except SimEvalError as exc:
        catalog = None
        checks.append(_check("STATIC-CATALOG", False, str(exc)))

    register_builtin_adapters()
    adapter_names = set(registered_adapters())
    checks.append(
        _check(
            "STATIC-ADAPTER-REGISTRY",
            adapter_names == set(REQUESTED_BENCHMARK_IDS),
            f"registered={sorted(adapter_names)}",
        )
    )
    try:
        fixtures = load_fixture_suite(root / "tests/fixtures")
        adapter_errors: list[str] = []
        for benchmark_id in sorted(REQUESTED_BENCHMARK_IDS):
            adapter = get_adapter(benchmark_id)
            revisions = (getattr(adapter, "prompt_revision", ""), getattr(adapter, "scorer_revision", ""))
            if any(not isinstance(value, str) or not value or any(token in value.casefold() for token in ("todo", "placeholder")) for value in revisions):
                adapter_errors.append(f"{benchmark_id}: missing/placeholder revisions")
                continue
            if not all(callable(getattr(adapter, name, None)) for name in ("validate_case", "execute_case", "aggregate", "replay_responses")):
                adapter_errors.append(f"{benchmark_id}: incomplete execution surface")
                continue
            cases = fixtures.get(benchmark_id, ((), None))[0]
            if not cases:
                adapter_errors.append(f"{benchmark_id}: no fixture cases")
                continue
            try:
                for case in cases:
                    adapter.validate_case(case)
            except SimEvalError as exc:
                adapter_errors.append(f"{benchmark_id}: fixture validation failed ({exc})")
        checks.append(
            _check(
                "STATIC-NONPLACEHOLDER-ADAPTERS",
                not adapter_errors,
                "all adapters have revisions, execution methods, and valid original fixtures"
                if not adapter_errors
                else "; ".join(adapter_errors),
            )
        )
    except SimEvalError as exc:
        checks.append(_check("STATIC-NONPLACEHOLDER-ADAPTERS", False, str(exc)))

    backend_names = set(registered_backends())
    checks.append(
        _check(
            "STATIC-BACKENDS",
            REQUIRED_BACKENDS <= backend_names,
            f"registered={sorted(backend_names)}",
        )
    )
    scope_ok, scope_detail = _static_import_scope(root)
    checks.append(_check("STATIC-IMPORT-SCOPE", scope_ok, scope_detail))

    if catalog is not None:
        families = {spec.protocol_family for spec in catalog.benchmarks.values()}
        checks.append(
            _check(
                "STATIC-PROTOCOL-FAMILIES",
                len(families) >= 8 and all(spec.official_protocol.get("metrics") for spec in catalog.benchmarks.values()),
                f"protocol_families={sorted(families)}",
            )
        )

    if require_artifacts:
        checks.extend(_artifact_checks(root, run_relative, report_relative))
    else:
        checks.append(_check("ARTIFACT-CHECKS", True, "artifact checks intentionally deferred by caller"))

    failures = [item for item in checks if item["status"] == "failed"]
    return {
        "schema_version": "1.0",
        "status": "passed" if not failures else "failed",
        "project_root": str(root),
        "check_count": len(checks),
        "passed_count": len(checks) - len(failures),
        "failed_count": len(failures),
        "checks": checks,
    }


__all__ = ["REQUIRED_BACKENDS", "REQUIRED_DOCUMENTS", "verify_delivery"]
