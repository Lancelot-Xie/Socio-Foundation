"""Auditable sidecar recomputation for metrics that do not require new model calls."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import CheckpointStore, atomic_write_json, atomic_write_text, load_json
from .benchmarks.agentsense import AgentSenseAdapter
from .benchmarks.mirrorbench import MirrorBenchAdapter
from .contracts import CaseResult, MetricValue, case_result_from_dict
from .errors import ArtifactError, ConfigurationError
from .json_utils import jsonable


POSTHOC_METRIC_REVISION = "posthoc-derived-metrics-v2-20260826"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} must be a JSON object")
    return value


def _load_run(run_directory: str | Path) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], list[CaseResult]]:
    run_dir = Path(run_directory).expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    records_path = run_dir / "records.jsonl"
    for path in (manifest_path, metrics_path, records_path):
        if not path.is_file():
            raise ArtifactError(f"post-hoc recomputation requires {path}")
    manifest = _mapping(load_json(manifest_path), str(manifest_path))
    old_metrics_document = _mapping(load_json(metrics_path), str(metrics_path))
    store = CheckpointStore(run_dir)
    results = [case_result_from_dict(row) for row in store.iter_latest_record_dicts()]
    if not results:
        raise ArtifactError(f"records artifact contains no completed checkpoint rows: {records_path}")
    return run_dir, manifest, old_metrics_document, results


def _benchmark_id(manifest: Mapping[str, Any], results: Sequence[CaseResult]) -> str:
    identity = _mapping(manifest.get("identity"), "run_manifest.identity")
    benchmark_id = str(identity.get("benchmark_id") or "").strip()
    observed = {result.benchmark_id for result in results}
    if not benchmark_id or observed != {benchmark_id}:
        raise ArtifactError(
            f"run benchmark identity is inconsistent: manifest={benchmark_id!r}, records={sorted(observed)}"
        )
    return benchmark_id


def _selected_metrics(benchmark_id: str, aggregate: Mapping[str, MetricValue]) -> Mapping[str, MetricValue]:
    if benchmark_id == "agentsense":
        names = (
            "agentsense.profile_sensitivity_index.goal",
            "agentsense.profile_sensitivity_index.information",
        )
    elif benchmark_id == "mirrorbench":
        names = (
            "mirrorbench.judge.pi",
            "mirrorbench.judge.pi_deviation",
            "mirrorbench.lexical.mattr.proxy_raw",
            "mirrorbench.lexical.mattr.human_raw",
            "mirrorbench.lexical.mattr.z_score_mean",
            "mirrorbench.lexical.hdd.proxy_raw",
            "mirrorbench.lexical.hdd.human_raw",
            "mirrorbench.lexical.hdd.z_score_mean",
            "mirrorbench.lexical.yules_k.proxy_raw",
            "mirrorbench.lexical.yules_k.human_raw",
            "mirrorbench.lexical.yules_k.z_score_mean",
        )
    else:
        raise ConfigurationError(
            "post-hoc metric recomputation currently supports only agentsense and mirrorbench"
        )
    missing = [name for name in names if name not in aggregate]
    if missing:
        raise ArtifactError(f"recomputed aggregate is missing required metrics: {missing}")
    return {name: aggregate[name] for name in names}


def _render_comparison(
    *,
    benchmark_id: str,
    run_id: str,
    changes: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> str:
    lines = [
        "# Post-hoc metric correction",
        "",
        f"- Benchmark: `{benchmark_id}`",
        f"- Source run: `{run_id}`",
        f"- Revision: `{POSTHOC_METRIC_REVISION}`",
        f"- Output: `{output_directory}`",
        "- Model/Judge calls: `0`",
        "- Source checkpoints modified: `no`",
        "",
        "| Metric | Previous | Corrected | Direction |",
        "|---|---:|---:|---|",
    ]
    for change in changes:
        old = change.get("previous_value")
        new = change.get("corrected_value")
        old_text = "—" if old is None else f"{float(old):.10g}"
        new_text = "—" if new is None else f"{float(new):.10g}"
        lines.append(
            f"| `{change['name']}` | {old_text} | {new_text} | `{change['direction']}` |"
        )
    lines.extend(
        [
            "",
            "The corrected file merges these derived metrics into a copy of the original metrics map. "
            "The original `records.jsonl`, `metrics.json`, and manifests remain unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def recompute_posthoc_metrics(
    run_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
) -> Path:
    """Recompute corrected AgentSense/MirrorBench derived metrics into a sidecar."""

    run_dir, manifest, old_document, results = _load_run(run_directory)
    benchmark_id = _benchmark_id(manifest, results)
    if benchmark_id == "agentsense":
        aggregate = AgentSenseAdapter().aggregate(results)
    elif benchmark_id == "mirrorbench":
        aggregate = MirrorBenchAdapter().aggregate(results)
    else:
        raise ConfigurationError(
            f"post-hoc metric recomputation does not support benchmark {benchmark_id!r}"
        )
    corrected = _selected_metrics(benchmark_id, aggregate)
    old_metrics = dict(_mapping(old_document.get("metrics"), "metrics.json.metrics"))
    corrected_metrics = dict(old_metrics)
    corrected_metrics.update({name: jsonable(metric) for name, metric in corrected.items()})
    run_id = str(manifest.get("run_id") or old_document.get("run_id") or "")
    if not run_id:
        raise ArtifactError("source run lacks run_id")
    output_dir = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else run_dir / f"posthoc_{benchmark_id}_metrics_v2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    changes = []
    for name, metric in corrected.items():
        old = old_metrics.get(name)
        old = old if isinstance(old, Mapping) else {}
        changes.append(
            {
                "name": name,
                "previous_value": old.get("value"),
                "corrected_value": metric.value,
                "direction": metric.direction,
                "unit": metric.unit,
            }
        )

    source = {
        "schema_version": "1.0",
        "posthoc_revision": POSTHOC_METRIC_REVISION,
        "benchmark_id": benchmark_id,
        "source_run_id": run_id,
        "source_run_directory": str(run_dir),
        "source_artifacts": {
            "run_manifest.json": _sha256(run_dir / "run_manifest.json"),
            "metrics.json": _sha256(run_dir / "metrics.json"),
            "records.jsonl": _sha256(run_dir / "records.jsonl"),
        },
        "latest_record_count": len(results),
        "model_or_judge_calls": 0,
        "source_artifacts_modified": False,
        "corrected_metric_names": list(corrected),
    }
    atomic_write_json(output_dir / "recomputation_manifest.json", source)
    atomic_write_json(output_dir / "comparison.json", {"changes": changes})
    atomic_write_json(
        output_dir / "corrected_metrics.json",
        {
            "run_id": run_id,
            "source_run_id": run_id,
            "posthoc_revision": POSTHOC_METRIC_REVISION,
            "metrics": corrected_metrics,
        },
    )
    atomic_write_text(
        output_dir / "corrected_metric_summary.md",
        _render_comparison(
            benchmark_id=benchmark_id,
            run_id=run_id,
            changes=changes,
            output_directory=output_dir,
        ),
    )
    return output_dir


__all__ = ["POSTHOC_METRIC_REVISION", "recompute_posthoc_metrics"]
