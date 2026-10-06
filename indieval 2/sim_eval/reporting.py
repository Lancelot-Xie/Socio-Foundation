"""Validated human-readable reports over completed suite artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import atomic_write_text, load_json
from .catalog import BenchmarkCatalog, BenchmarkSpec, load_catalog
from .data.commands import _safe_output
from .errors import ArtifactError, ConfigurationError


NON_LEADERBOARD_LABEL = "synthetic_offline_smoke_not_a_benchmark_score"

REPORT_METRIC_PREFERENCES: Mapping[str, Sequence[str]] = {
    "fantom": ("fantom.item_correct", "fantom.all"),
    "social_r1": ("social_r1.accuracy", "social_r1.parse_failure_rate"),
    "lifechoices": ("lifechoices.accuracy", "lifechoices.parse_failure_rate"),
    "behaviorchain": ("behaviorchain.diagnostic.node_micro_score",),
    "alignx": ("alignx.alignment_accuracy", "alignx.alignment_score_availability_rate"),
    "humanllm": ("humanllm.top1_accuracy", "humanllm.parse_failure_rate"),
    "userlm": (
        "userlm.extrinsic.assistant_task_score",
        "userlm.extrinsic.intent_coverage",
        "userlm.intrinsic.role_adherence",
        "userlm.intrinsic.termination_f1",
    ),
    "tau_usi": (
        "tau_usi.usi",
        "tau_usi.d1_communication",
        "tau_usi.d2_information",
        "tau_usi.d3_clarification",
        "tau_usi.d4_error_reaction",
        "tau_usi.ece",
        "tau_usi.eval",
    ),
    "mirrorbench": (
        "mirrorbench.judge.gteval",
        "mirrorbench.judge.pi_deviation",
        "mirrorbench.judge.pi",
        "mirrorbench.judge.rnr",
        "mirrorbench.lexical.mattr.z_score_mean",
        "mirrorbench.lexical.yules_k.z_score_mean",
        "mirrorbench.lexical.hdd.z_score_mean",
    ),
    "coser": (
        "coser.scene.critic_average",
        "coser.scene.storyline_consistency",
        "coser.scene.anthropomorphism",
        "coser.scene.character_fidelity",
        "coser.scene.storyline_quality",
    ),
    "sotopia": (
        "sotopia.judge_score_availability_rate",
        "sotopia.evaluated_agent.believability",
        "sotopia.evaluated_agent.relationship",
        "sotopia.evaluated_agent.knowledge",
        "sotopia.evaluated_agent.secret",
        "sotopia.evaluated_agent.social_rules",
        "sotopia.evaluated_agent.financial_and_material_benefits",
        "sotopia.evaluated_agent.goal",
        "sotopia.evaluated_agent.normalized_dimension_mean",
    ),
    "agentsense": (
        "agentsense.episode.self_goal_completion",
        "agentsense.episode.other_goal_completion",
        "agentsense.episode.judge_average",
        "agentsense.episode.private_information_accuracy",
        "agentsense.profile_sensitivity_index.goal",
        "agentsense.profile_sensitivity_index.information",
    ),
}


def _safe_run_directory(path: str | Path) -> Path:
    directory = Path(path)
    resolved = (Path.cwd() / directory).resolve() if not directory.is_absolute() else directory.resolve()
    if not resolved.is_dir():
        raise ArtifactError(f"run directory does not exist: {resolved}")
    return resolved


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"suite summary field {field!r} must be an object")
    return value


def load_suite_summary(run_directory: str | Path) -> tuple[Path, Mapping[str, Any]]:
    root = _safe_run_directory(run_directory)
    summary = load_json(root / "suite_summary.json")
    if not isinstance(summary, Mapping):
        raise ArtifactError("suite_summary.json must contain an object")
    for field in ("schema_version", "profile", "backend", "result_label", "benchmark_count", "benchmarks"):
        if field not in summary:
            raise ArtifactError(f"suite_summary.json is missing {field!r}")
    benchmarks = _mapping(summary["benchmarks"], field="benchmarks")
    if int(summary["benchmark_count"]) != len(benchmarks):
        raise ArtifactError("suite benchmark_count does not match benchmark entries")
    return root, summary


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_value(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _metric_rows(benchmark_id: str, entry: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    metrics = _mapping(entry.get("metrics"), field=f"benchmarks.{benchmark_id}.metrics")
    preferred = [name for name in REPORT_METRIC_PREFERENCES.get(benchmark_id, ()) if name in metrics]
    official = [
        name
        for name, raw in metrics.items()
        if isinstance(raw, Mapping) and bool((raw.get("metadata") or {}).get("official_primary"))
    ]
    selected = list(dict.fromkeys((*preferred, *sorted(official))))
    if not selected:
        selected = sorted(metrics)[:5]
    return [(name, _mapping(metrics[name], field=f"metric {name}")) for name in selected]


def _artifact_link(
    run_root: Path,
    entry: Mapping[str, Any],
    key: str,
    *,
    link_base: Path,
) -> str | None:
    paths = entry.get("artifact_paths")
    candidate = paths.get(key) if isinstance(paths, Mapping) else None
    if candidate is None and key == "metrics":
        candidate = entry.get("metrics_path")
    if not isinstance(candidate, str) or not candidate:
        return None
    path = (run_root / candidate).resolve()
    try:
        relative = path.relative_to(run_root)
    except ValueError as exc:
        raise ArtifactError(f"artifact path escapes run directory: {candidate}") from exc
    if not path.is_file():
        raise ArtifactError(f"referenced artifact is missing: {path}")
    del relative
    return Path(os.path.relpath(path, start=link_base)).as_posix()


def _limitation(spec: BenchmarkSpec) -> str:
    status = str(spec.access.get("status") or "unresolved")
    constraint = str(spec.access.get("constraints") or "No additional constraint recorded.")
    unblock = spec.access.get("unblock")
    suffix = f" Unblock: {unblock}" if unblock else ""
    return f"Access state `{status}`. {constraint}{suffix}"


def render_suite_report(
    summary: Mapping[str, Any],
    *,
    run_root: Path,
    catalog: BenchmarkCatalog,
    link_base: Path | None = None,
) -> str:
    link_base = (link_base or run_root).resolve()
    entries = _mapping(summary["benchmarks"], field="benchmarks")
    missing = set(entries) - set(catalog.benchmarks)
    if missing:
        raise ArtifactError(f"suite contains benchmark IDs absent from catalog: {sorted(missing)}")
    label = str(summary["result_label"])
    lines = [
        "# Evaluation suite report",
        "",
        "> **NON-LEADERBOARD EVIDENCE:** This report contains synthetic fixtures and replayed outputs. "
        "It verifies software contracts only; it is not a real-model, official-test, or leaderboard result."
        if label == NON_LEADERBOARD_LABEL
        else f"> **Result label:** `{_escape(label)}`. Interpret every metric under that protocol label.",
        "",
        "## Run summary",
        "",
        f"- Profile: `{_escape(summary['profile'])}`",
        f"- Backend/model: `{_escape(summary['backend'])}` / `{_escape(summary.get('model', 'unspecified'))}`",
        f"- Result label: `{_escape(label)}`",
        f"- Catalog/framework revision: `{_escape(summary.get('catalog_revision', 'unknown'))}` / "
        f"`{_escape(summary.get('framework_version', 'unknown'))}`",
        f"- Benchmarks: {len(entries)}",
        "- Cross-suite raw average: **not computed**; benchmark-native scales are not commensurate.",
        "",
        "| Benchmark | Protocol family | Selected | Results | Completed | Failed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for benchmark_id in sorted(entries):
        entry = _mapping(entries[benchmark_id], field=f"benchmarks.{benchmark_id}")
        spec = catalog.get(benchmark_id)
        lines.append(
            f"| `{benchmark_id}` | `{_escape(spec.protocol_family)}` | "
            f"{int(entry.get('selected_case_count', 0))} | {int(entry.get('result_count', 0))} | "
            f"{int(entry.get('completed_count', 0))} | {int(entry.get('failed_count', 0))} |"
        )

    lines.extend(["", "## Benchmark-native results", ""])
    for benchmark_id in sorted(entries):
        entry = _mapping(entries[benchmark_id], field=f"benchmarks.{benchmark_id}")
        spec = catalog.get(benchmark_id)
        lines.extend(
            [
                f"### {spec.display_name} (`{benchmark_id}`)",
                "",
                f"Protocol: `{_escape(spec.protocol_family)}`; official unit: "
                f"`{_escape(spec.official_protocol.get('evaluation_unit', 'unresolved'))}`; "
                f"smoke sample: {int(entry.get('selected_case_count', 0))} cases in "
                f"{int(entry.get('selected_group_count', 0))} groups.",
                "",
                "| Metric | Value | Numerator | Denominator |",
                "|---|---:|---:|---:|",
            ]
        )
        metrics = _mapping(entry.get("metrics"), field=f"benchmarks.{benchmark_id}.metrics")
        for name, raw in _metric_rows(benchmark_id, entry):
            lines.append(
                f"| `{_escape(name)}` | {_format_value(raw.get('value'))} | "
                f"{_format_value(raw.get('numerator'))} | {_format_value(raw.get('denominator'))} |"
            )
        lines.append("")
        lines.append(
            f"Highlighted {len(_metric_rows(benchmark_id, entry))} of {len(metrics)} emitted metrics; "
            "the machine-readable metrics artifact remains authoritative."
        )
        lines.append("")
        errors = entry.get("errors") or []
        if not isinstance(errors, Sequence) or isinstance(errors, (str, bytes)):
            raise ArtifactError(f"benchmarks.{benchmark_id}.errors must be an array")
        if errors:
            lines.extend(
                [
                    f"Failures ({len(errors)}):",
                    "",
                    "| Case | Repetition | Stage | Kind | Retryable | Message |",
                    "|---|---:|---|---|---|---|",
                ]
            )
            for raw in errors:
                error = _mapping(raw, field=f"benchmarks.{benchmark_id}.errors[]")
                lines.append(
                    f"| `{_escape(error.get('case_id'))}` | {int(error.get('repetition', 0))} | "
                    f"`{_escape(error.get('stage'))}` | `{_escape(error.get('kind'))}` | "
                    f"{_format_value(error.get('retryable'))} | {_escape(error.get('message'))} |"
                )
        else:
            lines.append("Failures: none in this synthetic replay run.")
        lines.extend(["", f"Limitation: {_limitation(spec)}", ""])
        links = []
        for key in ("run_manifest", "source_manifest", "sample_manifest", "records", "metrics"):
            value = _artifact_link(run_root, entry, key, link_base=link_base)
            if value:
                links.append(f"[{key}]({value})")
        if links:
            lines.extend(["Artifacts: " + " · ".join(links), ""])

    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Synthetic fixture values test execution, parsing, scoring, aggregation, and reporting; they do not estimate model quality.",
            "- Unavailable metrics remain `unavailable`; they are not converted to zero or removed from an official deterministic denominator.",
            "- Judge-backed values in replay artifacts are recorded fixture payloads, not live judge calls.",
            "- Canonical/default results require the pinned official or authorized data, models, assistants/partners, judges, and repetition policy documented for each benchmark.",
            "- Raw metrics across benchmarks use different units and must not be averaged without a separately versioned normalization policy.",
            "",
        ]
    )
    return "\n".join(lines)


def write_suite_report(
    *,
    run_directory: str | Path,
    output_path: str | Path,
    catalog_path: str | Path,
) -> dict[str, Any]:
    run_root, summary = load_suite_summary(run_directory)
    catalog = load_catalog(catalog_path)
    output = _safe_output(output_path)
    report = render_suite_report(
        summary,
        run_root=run_root,
        catalog=catalog,
        link_base=output.parent,
    )
    atomic_write_text(output, report)
    return {
        "status": "written",
        "output": str(output),
        "run_directory": str(run_root),
        "benchmark_count": int(summary["benchmark_count"]),
        "result_label": str(summary["result_label"]),
        "failed_count": sum(
            int(_mapping(item, field="benchmark entry").get("failed_count", 0))
            for item in _mapping(summary["benchmarks"], field="benchmarks").values()
        ),
    }


def report_to_json(result: Mapping[str, Any]) -> str:
    return json.dumps(dict(result), ensure_ascii=False, indent=2)


__all__ = [
    "NON_LEADERBOARD_LABEL",
    "REPORT_METRIC_PREFERENCES",
    "load_suite_summary",
    "render_suite_report",
    "write_suite_report",
]
