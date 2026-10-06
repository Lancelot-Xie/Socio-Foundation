"""Shared offline execution loop used by family and root smoke verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .artifacts import CheckpointStore, atomic_write_json
from .backends import get_backend
from .benchmarks import register_builtin_adapters
from .catalog import load_and_validate
from .contracts import ResultStatus, RunIdentityInput, RunManifest, case_result_from_dict
from .data.commands import _safe_output
from .data.loaders import load_fixture_suite
from .data.sampling import DeterministicStratifiedSampler, resolve_sampling_plan
from .errors import ConfigurationError
from .json_utils import jsonable
from .registry import get_adapter


def _judge_identity(adapter: Any, cases: Sequence[Any]) -> Mapping[str, Any]:
    if not hasattr(adapter, "provenance_for_case"):
        return {}
    identities = []
    for case in cases:
        provenance = adapter.provenance_for_case(case)
        identities.append(provenance.to_dict() if provenance else {})
    if any(identity != identities[0] for identity in identities[1:]):
        raise ConfigurationError("selected cases use incompatible judge provenance in one run")
    return identities[0]


def _case_bound_identity(adapter: Any, cases: Sequence[Any], method_name: str) -> Mapping[str, Any]:
    method = getattr(adapter, method_name, None)
    if method is None:
        return {}
    identities = [dict(method(case)) for case in cases]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ConfigurationError(f"selected cases use incompatible {method_name} identities in one run")
    return identities[0]


def run_fixture_suite(
    *,
    benchmark_ids: Sequence[str],
    profile: str,
    backend_name: str,
    output_directory: str | Path,
    fixture_directory: str | Path,
    catalog_path: str | Path,
    sampling_path: str | Path,
    model: str = "offline-replay",
) -> dict[str, Any]:
    if profile != "offline_smoke":
        raise ConfigurationError("fixture-backed run requires profile=offline_smoke")
    if backend_name != "replay":
        raise ConfigurationError(
            "the fixture command is intentionally offline and requires backend=replay; real backend configuration is a separate run mode"
        )
    catalog, profiles = load_and_validate(catalog_path, sampling_path)
    fixtures = load_fixture_suite(fixture_directory)
    requested = tuple(dict.fromkeys(benchmark_ids))
    if not requested:
        raise ConfigurationError("--benchmarks must name at least one benchmark")
    unknown = set(requested) - set(catalog.benchmarks)
    if unknown:
        raise ConfigurationError(f"unknown benchmark IDs: {sorted(unknown)}")
    register_builtin_adapters()
    output = _safe_output(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for benchmark_id in requested:
        source_cases, source_manifest = fixtures[benchmark_id]
        plan = resolve_sampling_plan(profiles, profile, benchmark_id)
        selected, sample_manifest = DeterministicStratifiedSampler(plan, source_manifest).select(source_cases)
        adapter = get_adapter(benchmark_id)
        judge_identity = _judge_identity(adapter, selected)
        environment = getattr(adapter, "environment", None)
        environment_identity = _case_bound_identity(adapter, selected, "environment_identity_for_case")
        if not environment_identity and environment is not None:
            environment_identity = {
                "revision": getattr(environment, "environment_revision", None),
                "randomize_turn_order": getattr(environment, "randomize_turn_order", None),
                "allowed_actions": list(getattr(environment, "allowed_actions", ())),
            }
        assistant_or_partner = _case_bound_identity(
            adapter, selected, "assistant_or_partner_identity_for_case"
        )
        identity = RunIdentityInput(
            framework_version=__version__,
            benchmark_id=benchmark_id,
            source_revision=source_manifest.source_revision,
            split=source_manifest.split,
            profile=profile,
            sample_manifest_digest=sample_manifest.digest,
            seed=plan.seed,
            backend=backend_name,
            model=model,
            decoding={"adapter_controlled": True},
            prompt_revision=str(adapter.prompt_revision),
            scorer_revision=str(adapter.scorer_revision),
            judge=judge_identity,
            environment=environment_identity,
            assistant_or_partner=assistant_or_partner,
        )
        requested_count = plan.target if isinstance(plan.target, int) else None
        manifest = RunManifest.create(
            identity,
            catalog_revision=catalog.catalog_revision,
            result_label=sample_manifest.result_label,
            requested_case_count=requested_count,
            selected_case_count=len(selected),
            selected_group_count=len(sample_manifest.selected_group_ids),
            metadata={
                "source_manifest_digest": source_manifest.digest,
                "fixture_directory": str(Path(fixture_directory)),
                "offline_only": True,
            },
        )
        benchmark_output = output / benchmark_id / manifest.run_id
        store = CheckpointStore(benchmark_output)
        store.initialize(manifest)
        atomic_write_json(benchmark_output / "source_manifest.json", source_manifest)
        atomic_write_json(benchmark_output / "sample_manifest.json", sample_manifest)
        completed = store.completed_keys()
        for case in selected:
            seeds = sample_manifest.repetition_seeds.get(case.case_id) or [plan.seed]
            for repetition, rollout_seed in enumerate(seeds):
                if (case.case_id, repetition) in completed:
                    continue
                responses = adapter.replay_responses(case, seed=int(rollout_seed))
                backend = get_backend("replay", responses=responses)
                result = adapter.execute_case(
                    case,
                    backend=backend,
                    run_id=manifest.run_id,
                    seed=int(rollout_seed),
                    model=model,
                    repetition=repetition,
                )
                store.append_result(result)
        results = [case_result_from_dict(item) for item in store.iter_latest_record_dicts()]
        metrics = adapter.aggregate(results)
        metrics_path = store.write_metrics(metrics)
        completed_count = sum(result.status == ResultStatus.COMPLETED for result in results)
        failed_count = sum(result.status == ResultStatus.FAILED for result in results)
        errors = [
            {
                "case_id": result.case_id,
                "repetition": result.repetition,
                "stage": result.error.stage,
                "kind": result.error.kind,
                "message": result.error.message,
                "retryable": result.error.retryable,
            }
            for result in results
            if result.status == ResultStatus.FAILED and result.error is not None
        ]
        summaries[benchmark_id] = {
            "run_id": manifest.run_id,
            "result_label": manifest.result_label,
            "selected_case_count": len(selected),
            "selected_group_count": len(sample_manifest.selected_group_ids),
            "result_count": len(results),
            "completed_count": completed_count,
            "failed_count": failed_count,
            "errors": errors,
            "judge": judge_identity,
            "environment": environment_identity,
            "sample_manifest_digest": sample_manifest.digest,
            "metrics_path": str(metrics_path.relative_to(output)),
            "artifact_paths": {
                "run_manifest": str(store.manifest_path.relative_to(output)),
                "source_manifest": str((benchmark_output / "source_manifest.json").relative_to(output)),
                "sample_manifest": str((benchmark_output / "sample_manifest.json").relative_to(output)),
                "records": str(store.records_path.relative_to(output)),
                "metrics": str(metrics_path.relative_to(output)),
            },
            "metrics": jsonable(metrics),
        }
    total_results = sum(int(item["result_count"]) for item in summaries.values())
    total_completed = sum(int(item["completed_count"]) for item in summaries.values())
    total_failed = sum(int(item["failed_count"]) for item in summaries.values())
    suite = {
        "schema_version": "1.0",
        "framework_version": __version__,
        "catalog_revision": catalog.catalog_revision,
        "profile": profile,
        "backend": backend_name,
        "model": model,
        "result_label": profiles.profiles[profile]["result_label"],
        "status": "completed" if total_failed == 0 else "completed_with_failures",
        "offline_only": True,
        "requested_benchmark_ids": list(requested),
        "benchmark_count": len(summaries),
        "totals": {
            "result_count": total_results,
            "completed_count": total_completed,
            "failed_count": total_failed,
        },
        "benchmarks": summaries,
    }
    atomic_write_json(output / "suite_summary.json", suite)
    return suite
