"""Read-only, entry-aligned comparison reports for formal evaluation artifacts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import atomic_write_text
from .errors import ArtifactError
from .metric_markdown import (
    MetricCoverage,
    MetricEntry,
    _primary_metric_names,
    _record_metric_attributes,
    _record_metric_counts,
    load_metric_entries,
    metric_coverage,
)


@dataclass(frozen=True)
class _LoadedEntry:
    root: Path
    entry: MetricEntry


_ALIGNX_REFERENCE_ONLY_METRICS = {
    "alignx.alignment_accuracy",
    "alignx.alignment_score_availability_rate",
}

_HUMANUAL_DOMAINS = ("news", "book", "opinion", "politics", "chat", "email")
_ARTIFACT_DATE_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_TRAINING_STEP_RE = re.compile(r"(?:^|_)step(\d+)(?:_|$)")


def _safe_metric_token(value: str) -> str:
    token = "".join(character if character.isalnum() else "_" for character in value.casefold())
    return "_".join(part for part in token.split("_") if part) or "unspecified"


def _is_hidden_reference_only_metric(entry: MetricEntry, name: str) -> bool:
    return entry.benchmark_id == "alignx" and (
        name in _ALIGNX_REFERENCE_ONLY_METRICS
        or name.startswith("alignx.alignment_accuracy.variant.")
    )


def _with_alignx_direct_choice_slices(entry: MetricEntry) -> MetricEntry:
    """Derive direct-choice variant accuracies from immutable result records.

    AlignX's stored ``alignment_accuracy.variant.*`` metrics belong to the
    disabled reference-margin protocol.  The current formal protocol still
    has everything needed for direct-choice slices: each record stores its
    variant and one binary direct-choice score.  Parse/case failures count as
    incorrect, matching the adapter's global direct-choice aggregation.
    """

    if entry.benchmark_id != "alignx":
        return entry
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for record in entry.records:
        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        alignx = metadata.get("alignx")
        alignx = alignx if isinstance(alignx, Mapping) else {}
        variant = alignx.get("variant")
        if not isinstance(variant, str) or not variant.strip():
            continue
        variant = variant.strip()
        totals[variant] = totals.get(variant, 0) + 1
        raw_metrics = record.get("metrics")
        raw_metrics = (
            raw_metrics
            if isinstance(raw_metrics, Sequence) and not isinstance(raw_metrics, (str, bytes))
            else ()
        )
        value = next(
            (
                metric.get("value")
                for metric in raw_metrics
                if isinstance(metric, Mapping)
                and metric.get("name") == "alignx.direct_choice_accuracy"
            ),
            None,
        )
        correct[variant] = correct.get(variant, 0) + int(value == 1 or value is True)

    metrics = {
        name: metric
        for name, metric in entry.metrics.items()
        if not _is_hidden_reference_only_metric(entry, name)
    }
    for variant in sorted(totals):
        denominator = totals[variant]
        numerator = correct.get(variant, 0)
        name = f"alignx.direct_choice_accuracy.variant.{_safe_metric_token(variant)}"
        metrics[name] = {
            "name": name,
            "value": numerator / denominator if denominator else None,
            "unit": "proportion",
            "direction": "higher_is_better",
            "numerator": numerator,
            "denominator": denominator,
            "metadata": {
                "variant": variant,
                "posthoc_derived": True,
                "source_metric": "alignx.direct_choice_accuracy",
                "aggregation": "micro_with_parse_and_case_failures_incorrect",
            },
        }
    return replace(entry, metrics=metrics)


def _humanual_domain_for_entry(entry: MetricEntry) -> str | None:
    if entry.benchmark_id != "humanual":
        return None
    return _humanual_domain_from_entry_id(entry.entry_id)


def _humanual_domain_from_entry_id(entry_id: str) -> str | None:
    prefix = "humanual_"
    if not entry_id.startswith(prefix):
        return None
    domain = entry_id[len(prefix):].strip().casefold()
    return domain if domain in _HUMANUAL_DOMAINS else None


def _record_humanual_domain(record: Mapping[str, Any]) -> str | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    domain = metadata.get("domain")
    if not isinstance(domain, str):
        return None
    normalized = domain.strip().casefold()
    return normalized if normalized in _HUMANUAL_DOMAINS else None


def _aggregate_record_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    namespace: str,
    templates: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    """Rebuild arithmetic case aggregates from immutable result records."""

    values: dict[str, list[float]] = {}
    unavailable: dict[str, int] = {}
    attributes: dict[str, Mapping[str, Any]] = {}
    for record in records:
        raw_metrics = record.get("metrics")
        if not isinstance(raw_metrics, Sequence) or isinstance(raw_metrics, (str, bytes)):
            continue
        for raw in raw_metrics:
            if not isinstance(raw, Mapping):
                continue
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                continue
            attributes.setdefault(name, raw)
            numeric = _number(raw.get("value"))
            if numeric is None:
                unavailable[name] = unavailable.get(name, 0) + 1
            else:
                values.setdefault(name, []).append(numeric)

    metrics: dict[str, Mapping[str, Any]] = {}
    for name in sorted(set(values) | set(unavailable)):
        available = values.get(name, [])
        template = dict(templates.get(name, {}))
        source_attributes = attributes.get(name, {})
        metadata = template.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        metadata.update(
            {
                "unavailable_count": unavailable.get(name, 0),
                "aggregation": "arithmetic_mean_available",
                "posthoc_domain_slice": True,
            }
        )
        metrics[name] = {
            **template,
            "name": name,
            "value": sum(available) / len(available) if available else None,
            "unit": template.get("unit") or source_attributes.get("unit"),
            "direction": template.get("direction") or source_attributes.get("direction"),
            "numerator": sum(available) if available else None,
            "denominator": len(available),
            "metadata": metadata,
        }

    completed = sum(str(record.get("status")) == "completed" for record in records)
    failed = sum(str(record.get("status")) == "failed" for record in records)
    total = len(records)
    completion_name = f"{namespace}.case_completion_rate"
    failure_name = f"{namespace}.case_failure_count"
    metrics[completion_name] = {
        **dict(templates.get(completion_name, {})),
        "name": completion_name,
        "value": completed / total if total else None,
        "unit": "proportion",
        "direction": "higher_is_better",
        "numerator": completed,
        "denominator": total,
        "metadata": {"posthoc_domain_slice": True},
    }
    metrics[failure_name] = {
        **dict(templates.get(failure_name, {})),
        "name": failure_name,
        "value": failed,
        "unit": "cases",
        "direction": "lower_is_better",
        "numerator": failed,
        "denominator": total,
        "metadata": {"posthoc_domain_slice": True},
    }
    return metrics


def _humanual_domain_baseline(
    baseline: MetricEntry,
    *,
    domain: str,
    entry_id: str,
) -> MetricEntry:
    records = tuple(
        record for record in baseline.records if _record_humanual_domain(record) == domain
    )
    if not records:
        raise ArtifactError(
            f"baseline HUMANUAL entry has no immutable records for domain {domain!r}"
        )
    record_keys = _record_keys(replace(baseline, records=records))
    selected_case_count = len({case_id for case_id, _ in record_keys}) or len(records)
    completed = sum(str(record.get("status")) == "completed" for record in records)
    failed = sum(str(record.get("status")) == "failed" for record in records)
    return MetricEntry(
        entry_id=entry_id,
        benchmark_id="humanual",
        status=baseline.status,
        selected_case_count=selected_case_count,
        repetitions_per_case=baseline.repetitions_per_case,
        result_count=len(records),
        completed_count=completed,
        failed_count=failed,
        metrics=_aggregate_record_metrics(
            records,
            namespace="humanual",
            templates=baseline.metrics,
        ),
        records=records,
    )


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _format_value(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "true" if value else "false"
    number = _number(value)
    if number is not None:
        return f"{number:.8g}"
    return _escape(value)


def _format_delta(baseline: Any, candidate: Any) -> str:
    baseline_number = _number(baseline)
    candidate_number = _number(candidate)
    if baseline_number is None or candidate_number is None:
        return "—"
    delta = candidate_number - baseline_number
    if math.isclose(delta, 0.0, rel_tol=1e-12, abs_tol=1e-12):
        return "0"
    return f"{delta:+.8g}"


def _direction_text(direction: Any) -> str:
    symbols = {
        "higher_is_better": "↑",
        "lower_is_better": "↓",
        "closer_to_zero": "→0",
    }
    text = str(direction or "unspecified")
    return f"{symbols.get(text, '—')} {text}"


def _winner(
    baseline: Any,
    candidate: Any,
    direction: str | None,
    *,
    baseline_label: str,
    candidate_label: str,
) -> str:
    baseline_number = _number(baseline)
    candidate_number = _number(candidate)
    if baseline_number is None or candidate_number is None:
        return "—"
    if math.isclose(baseline_number, candidate_number, rel_tol=1e-12, abs_tol=1e-12):
        return "持平"
    if direction == "higher_is_better":
        return candidate_label if candidate_number > baseline_number else baseline_label
    if direction == "lower_is_better":
        return candidate_label if candidate_number < baseline_number else baseline_label
    if direction == "closer_to_zero":
        baseline_distance = abs(baseline_number)
        candidate_distance = abs(candidate_number)
        if math.isclose(baseline_distance, candidate_distance, rel_tol=1e-12, abs_tol=1e-12):
            return "持平"
        return candidate_label if candidate_distance < baseline_distance else baseline_label
    return "—"


def _winner_many(
    values: Sequence[Any],
    direction: str | None,
    *,
    labels: Sequence[str],
) -> str:
    if len(values) != len(labels):
        return "—"
    available = [
        (label, float(number))
        for label, value in zip(labels, values)
        if (number := _number(value)) is not None
    ]
    if len(available) < 2:
        return "—"
    available_labels = [label for label, _number_value in available]
    concrete = [number_value for _label, number_value in available]
    if direction == "higher_is_better":
        target = max(concrete)
        scores = concrete
    elif direction == "lower_is_better":
        target = min(concrete)
        scores = concrete
    elif direction == "closer_to_zero":
        distances = [abs(number) for number in concrete]
        target = min(distances)
        scores = distances
    else:
        return "—"
    winners = [
        label
        for label, score in zip(available_labels, scores)
        if math.isclose(score, target, rel_tol=1e-12, abs_tol=1e-12)
    ]
    return " / ".join(winners)


def _coverage_text(coverage: MetricCoverage | None) -> str:
    if coverage is None or coverage.applicable_count <= 0:
        return "—"
    return (
        f"{coverage.valid_count}/{coverage.applicable_count} "
        f"({coverage.valid_count / coverage.applicable_count:.2%})"
    )


def _metric_attributes(
    entry: MetricEntry,
    name: str,
    record_attributes: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, str | None]:
    metric = entry.metrics.get(name, {})
    fallback = record_attributes.get(name, {})
    unit = metric.get("unit") or fallback.get("unit")
    if unit is None and name.endswith(".normalized"):
        unit = "0_to_1"
    direction = metric.get("direction") or fallback.get("direction")
    return unit or "—", str(direction) if direction is not None else None


def _metric_coverage(entry: MetricEntry, name: str) -> MetricCoverage | None:
    metric = entry.metrics.get(name)
    if metric is None:
        return None
    return metric_coverage(
        name,
        metric,
        record_counts=_record_metric_counts(entry.records),
        total_records=entry.expected_record_count,
        completed_records=entry.completed_count,
    )


def _record_keys(entry: MetricEntry) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for record in entry.records:
        case_id = record.get("case_id")
        repetition = record.get("repetition", 0)
        if not isinstance(case_id, str) or not case_id:
            continue
        if isinstance(repetition, bool) or not isinstance(repetition, (int, float)):
            continue
        number = float(repetition)
        if number < 0 or not number.is_integer():
            continue
        keys.add((case_id, int(number)))
    return keys


def _population_comparison(baseline: MetricEntry, candidate: MetricEntry) -> tuple[str, bool]:
    baseline_keys = _record_keys(baseline)
    candidate_keys = _record_keys(candidate)
    if baseline_keys and candidate_keys:
        if baseline_keys == candidate_keys:
            return f"✅ exact record keys ({len(baseline_keys)})", True
        overlap = len(baseline_keys & candidate_keys)
        return (
            "⚠️ record-key mismatch "
            f"(baseline={len(baseline_keys)}, candidate={len(candidate_keys)}, overlap={overlap})",
            False,
        )
    if baseline.expected_record_count == candidate.expected_record_count:
        return f"ℹ️ counts only ({baseline.expected_record_count}); record keys unavailable", True
    return (
        "⚠️ record-count mismatch "
        f"(baseline={baseline.expected_record_count}, candidate={candidate.expected_record_count})",
        False,
    )


def _load_unique_candidates(candidate_paths: Sequence[str | Path]) -> Mapping[str, _LoadedEntry]:
    loaded: dict[str, _LoadedEntry] = {}
    if not candidate_paths:
        raise ArtifactError("at least one candidate result path is required")
    for candidate_path in candidate_paths:
        root, _summary_path, _summary, entries = load_metric_entries(candidate_path)
        for raw_entry in entries:
            entry = _with_alignx_direct_choice_slices(raw_entry)
            if entry.entry_id in loaded:
                raise ArtifactError(
                    f"candidate entry {entry.entry_id!r} appears in more than one result path: "
                    f"{loaded[entry.entry_id].root} and {root}"
                )
            loaded[entry.entry_id] = _LoadedEntry(root=root, entry=entry)
    return loaded


def _load_exact_entry_override(
    path: str | Path,
    *,
    entry_id: str,
) -> _LoadedEntry:
    root, _summary_path, _summary, entries = load_metric_entries(path)
    matches = [
        _with_alignx_direct_choice_slices(entry)
        for entry in entries
        if entry.entry_id == entry_id
    ]
    if len(matches) != 1:
        available = ", ".join(sorted(entry.entry_id for entry in entries)) or "<none>"
        raise ArtifactError(
            f"entry override for {entry_id!r} must contain exactly that entry; "
            f"artifact {root} contains: {available}"
        )
    return _LoadedEntry(root=root, entry=matches[0])


def _apply_candidate_entry_overrides(
    candidates: Mapping[str, _LoadedEntry],
    overrides: Mapping[str, str | Path] | None,
) -> Mapping[str, _LoadedEntry]:
    result = dict(candidates)
    for entry_id, path in (overrides or {}).items():
        if entry_id not in result:
            raise ArtifactError(
                f"candidate entry override {entry_id!r} is absent from the candidate scope"
            )
        replacement = _load_exact_entry_override(path, entry_id=entry_id)
        if replacement.entry.benchmark_id != result[entry_id].entry.benchmark_id:
            raise ArtifactError(
                f"candidate entry override {entry_id!r} changes benchmark identity from "
                f"{result[entry_id].entry.benchmark_id!r} to "
                f"{replacement.entry.benchmark_id!r}"
            )
        result[entry_id] = replacement
    return result


def _apply_reference_entry_overrides(
    entries: Sequence[MetricEntry],
    overrides: Mapping[str, str | Path] | None,
    *,
    reference_label: str,
) -> tuple[MetricEntry, ...]:
    result = list(entries)
    positions = {entry.entry_id: index for index, entry in enumerate(result)}
    for entry_id, path in (overrides or {}).items():
        if entry_id not in positions:
            raise ArtifactError(
                f"{reference_label} entry override {entry_id!r} is absent from the suite"
            )
        replacement = _load_exact_entry_override(path, entry_id=entry_id).entry
        original = result[positions[entry_id]]
        if replacement.benchmark_id != original.benchmark_id:
            raise ArtifactError(
                f"{reference_label} entry override {entry_id!r} changes benchmark "
                f"identity from {original.benchmark_id!r} to {replacement.benchmark_id!r}"
            )
        result[positions[entry_id]] = replacement
    return tuple(result)


def _entry_override_report_lines(
    label: str,
    overrides: Mapping[str, str | Path] | None,
) -> list[str]:
    return [
        f"- {_escape(label)} entry override `{_escape(entry_id)}`："
        f"`{_escape(Path(path).expanduser().resolve())}`"
        for entry_id, path in sorted((overrides or {}).items())
    ]


def _aligned_entry_map(
    *,
    reference_entries: Sequence[MetricEntry],
    candidates: Mapping[str, _LoadedEntry],
    reference_label: str,
) -> Mapping[str, MetricEntry]:
    """Align exact entries, slicing a full HUMANUAL run for domain LoRAs."""

    reference_by_id = {entry.entry_id: entry for entry in reference_entries}
    aligned: dict[str, MetricEntry] = {}
    missing: list[str] = []
    for entry_id, loaded in candidates.items():
        if entry_id in reference_by_id:
            aligned[entry_id] = reference_by_id[entry_id]
            continue
        domain = _humanual_domain_for_entry(loaded.entry)
        humanual_reference = reference_by_id.get("humanual")
        if domain is not None and humanual_reference is not None:
            aligned[entry_id] = _humanual_domain_baseline(
                humanual_reference,
                domain=domain,
                entry_id=entry_id,
            )
            continue
        missing.append(entry_id)
    if missing:
        raise ArtifactError(
            f"candidate entries are absent from {reference_label}: " + ", ".join(missing)
        )
    return aligned


def _ordered_candidate_ids(
    baseline_entries: Sequence[MetricEntry],
    candidates: Mapping[str, _LoadedEntry],
) -> tuple[str, ...]:
    ordered: list[str] = []
    for baseline_entry in baseline_entries:
        if baseline_entry.entry_id in candidates:
            ordered.append(baseline_entry.entry_id)
        if baseline_entry.entry_id == "humanual":
            ordered.extend(
                f"humanual_{domain}"
                for domain in _HUMANUAL_DOMAINS
                if f"humanual_{domain}" in candidates
            )
    return tuple(ordered)


def _ordered_three_way_ids(
    baseline_entries: Sequence[MetricEntry],
    candidates: Mapping[str, _LoadedEntry],
    third_entries: Sequence[MetricEntry],
) -> tuple[str, ...]:
    """Order the union while keeping HUMANUAL domain slices beside the full entry."""

    ordered: list[str] = []
    seen: set[str] = set()

    def add(entry_id: str) -> None:
        if entry_id not in seen:
            ordered.append(entry_id)
            seen.add(entry_id)

    humanual_domain_ids = tuple(
        f"humanual_{domain}"
        for domain in _HUMANUAL_DOMAINS
        if f"humanual_{domain}" in candidates
    )
    for entry in baseline_entries:
        add(entry.entry_id)
        if entry.entry_id == "humanual":
            for entry_id in humanual_domain_ids:
                add(entry_id)
    for entry_id in candidates:
        add(entry_id)
    for entry in third_entries:
        add(entry.entry_id)
    return tuple(ordered)


def _entry_or_humanual_slice(
    entries: Sequence[MetricEntry],
    *,
    entry_id: str,
) -> MetricEntry | None:
    by_id = {entry.entry_id: entry for entry in entries}
    if entry_id in by_id:
        return by_id[entry_id]
    domain = _humanual_domain_from_entry_id(entry_id)
    full = by_id.get("humanual")
    if domain is None or full is None:
        return None
    return _humanual_domain_baseline(full, domain=domain, entry_id=entry_id)


def _artifact_recency_rank(path: Path) -> tuple[date, int, int, str]:
    parsed_dates = []
    for raw in _ARTIFACT_DATE_RE.findall(path.name):
        try:
            parsed_dates.append(date.fromisoformat(raw))
        except ValueError:
            continue
    steps = [int(raw) for raw in _TRAINING_STEP_RE.findall(path.name)]
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError:
        modified_ns = 0
    return (
        max(parsed_dates, default=date.min),
        max(steps, default=-1),
        modified_ns,
        path.name,
    )


def discover_latest_candidate_paths(
    candidate_root: str | Path,
    *,
    glob_pattern: str = "*",
) -> tuple[Path, ...]:
    """Select the newest complete single-entry artifact for every entry ID.

    Evaluation date dominates, followed by training step, filesystem mtime,
    and name.  A failed newest artifact is rejected instead of silently
    comparing an older checkpoint.
    """

    root = Path(candidate_root).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactError(f"candidate root is not a directory: {root}")
    paths = tuple(sorted(path for path in root.glob(glob_pattern) if path.is_dir()))
    if not paths:
        raise ArtifactError(
            f"candidate root {root} has no directories matching {glob_pattern!r}"
        )

    grouped: dict[str, list[tuple[Path, MetricEntry]]] = {}
    for path in paths:
        _loaded_root, _summary_path, _summary, entries = load_metric_entries(path)
        if len(entries) != 1:
            raise ArtifactError(
                f"automatic candidate discovery requires single-entry artifacts: {path} "
                f"contains {len(entries)} entries"
            )
        entry = entries[0]
        grouped.setdefault(entry.entry_id, []).append((path, entry))

    selected: list[Path] = []
    for entry_id in sorted(grouped):
        path, entry = max(grouped[entry_id], key=lambda item: _artifact_recency_rank(item[0]))
        if entry.status.casefold() == "failed" or not entry.metrics:
            raise ArtifactError(
                f"newest candidate for entry {entry_id!r} is not usable: "
                f"{path} (status={entry.status!r}, metrics={len(entry.metrics)}); "
                "rerun this artifact instead of silently falling back to an older checkpoint"
            )
        selected.append(path)
    return tuple(selected)


def _comparison_row(
    *,
    name: str,
    baseline: MetricEntry,
    candidate: MetricEntry,
    baseline_label: str,
    candidate_label: str,
    primary: bool,
) -> str:
    baseline_metric = baseline.metrics.get(name)
    candidate_metric = candidate.metrics.get(name)
    baseline_value = baseline_metric.get("value") if baseline_metric is not None else None
    candidate_value = candidate_metric.get("value") if candidate_metric is not None else None
    baseline_attributes = _record_metric_attributes(baseline.records)
    candidate_attributes = _record_metric_attributes(candidate.records)
    baseline_unit, baseline_direction = _metric_attributes(
        baseline, name, baseline_attributes
    )
    candidate_unit, candidate_direction = _metric_attributes(
        candidate, name, candidate_attributes
    )
    unit = baseline_unit if baseline_unit == candidate_unit else f"{baseline_unit} / {candidate_unit}"
    direction = (
        baseline_direction
        if baseline_direction == candidate_direction
        else None
    )
    direction_display = (
        _direction_text(direction)
        if direction is not None
        else f"⚠️ {_direction_text(baseline_direction)} / {_direction_text(candidate_direction)}"
    )
    metric_label = f"**`{_escape(name)}`**" if primary else f"`{_escape(name)}`"
    baseline_display = "missing" if baseline_metric is None else _format_value(baseline_value)
    candidate_display = "missing" if candidate_metric is None else _format_value(candidate_value)
    winner = _winner(
        baseline_value,
        candidate_value,
        direction,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    return (
        f"| {metric_label} | {baseline_display} | {candidate_display} | "
        f"{_format_delta(baseline_value, candidate_value)} | `{_escape(unit)}` | "
        f"`{_escape(direction_display)}` | {_escape(winner)} | "
        f"{_coverage_text(_metric_coverage(baseline, name))} | "
        f"{_coverage_text(_metric_coverage(candidate, name))} |"
    )


def render_metric_comparison(
    *,
    baseline_path: str | Path,
    candidate_paths: Sequence[str | Path],
    baseline_label: str = "Baseline",
    candidate_label: str = "Candidate",
    baseline_entry_overrides: Mapping[str, str | Path] | None = None,
    candidate_entry_overrides: Mapping[str, str | Path] | None = None,
) -> str:
    """Render an exact-entry comparison without changing either input run."""

    baseline_root, _path, _summary, raw_baseline_entries = load_metric_entries(baseline_path)
    baseline_entries = _apply_reference_entry_overrides(
        tuple(_with_alignx_direct_choice_slices(entry) for entry in raw_baseline_entries),
        baseline_entry_overrides,
        reference_label="baseline",
    )
    baseline_by_id = {entry.entry_id: entry for entry in baseline_entries}
    candidates = _apply_candidate_entry_overrides(
        _load_unique_candidates(candidate_paths),
        candidate_entry_overrides,
    )
    comparison_baselines: dict[str, MetricEntry] = {}
    missing: list[str] = []
    for entry_id, loaded in candidates.items():
        if entry_id in baseline_by_id:
            comparison_baselines[entry_id] = baseline_by_id[entry_id]
            continue
        domain = _humanual_domain_for_entry(loaded.entry)
        humanual_baseline = baseline_by_id.get("humanual")
        if domain is not None and humanual_baseline is not None:
            comparison_baselines[entry_id] = _humanual_domain_baseline(
                humanual_baseline,
                domain=domain,
                entry_id=entry_id,
            )
            continue
        missing.append(entry_id)
    if missing:
        raise ArtifactError(
            "candidate entries are absent from the baseline suite: " + ", ".join(missing)
        )

    ordered_ids: list[str] = []
    for baseline_entry in baseline_entries:
        if baseline_entry.entry_id in candidates:
            ordered_ids.append(baseline_entry.entry_id)
        if baseline_entry.entry_id == "humanual":
            ordered_ids.extend(
                f"humanual_{domain}"
                for domain in _HUMANUAL_DOMAINS
                if f"humanual_{domain}" in candidates
            )
    lines = [
        f"# 评测指标对比：{_escape(candidate_label)} vs {_escape(baseline_label)}",
        "",
        f"- 基线结果：`{_escape(baseline_root)}`",
        f"- 候选结果数量：{len(candidate_paths)}；匹配 entry 数量：{len(ordered_ids)}",
        *_entry_override_report_lines("Baseline", baseline_entry_overrides),
        *_entry_override_report_lines("Candidate", candidate_entry_overrides),
        f"- 差值定义：`{_escape(candidate_label)} - {_escape(baseline_label)}`。",
        "- 只比较完全相同的 metric 名称；不计算跨 benchmark 总平均。",
        "- 更优列只对明确声明了方向且双方都有数值的 metric 判断。",
        "- AlignX 按当前 direct-choice 协议从 records 派生五个 variant accuracy；未运行的 reference-margin 指标不展示。",
        "",
        "## 数据与运行可比性",
        "",
        "| Entry | Candidate artifact | Benchmark | Baseline status | Candidate status | Baseline records | Candidate records | Population check |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    comparable = True
    for entry_id in ordered_ids:
        baseline = comparison_baselines[entry_id]
        loaded = candidates[entry_id]
        candidate = loaded.entry
        population, population_ok = _population_comparison(baseline, candidate)
        benchmark_ok = baseline.benchmark_id == candidate.benchmark_id
        comparable = comparable and population_ok and benchmark_ok
        benchmark = (
            baseline.benchmark_id
            if benchmark_ok
            else f"⚠️ {baseline.benchmark_id} / {candidate.benchmark_id}"
        )
        lines.append(
            f"| `{_escape(entry_id)}` | `{_escape(loaded.root.name)}` | `{_escape(benchmark)}` | "
            f"`{_escape(baseline.status)}` | `{_escape(candidate.status)}` | "
            f"{baseline.expected_record_count} | {candidate.expected_record_count} | {population} |"
        )

    lines.extend(
        [
            "",
            "## 主指标对比总览",
            "",
            "| Entry | Metric | Baseline | Candidate | Delta | Unit | Direction | 更优 | Baseline 有效/适用 | Candidate 有效/适用 |",
            "|---|---|---:|---:|---:|---|---|---|---:|---:|",
        ]
    )
    for entry_id in ordered_ids:
        baseline = comparison_baselines[entry_id]
        candidate = candidates[entry_id].entry
        primary_names = tuple(
            dict.fromkeys((*_primary_metric_names(candidate), *_primary_metric_names(baseline)))
        )
        for name in primary_names:
            row = _comparison_row(
                name=name,
                baseline=baseline,
                candidate=candidate,
                baseline_label=baseline_label,
                candidate_label=candidate_label,
                primary=True,
            )
            # Insert the entry column into the standard all-metric row.
            lines.append(f"| `{_escape(entry_id)}` |" + row[1:])

    lines.extend(["", "## 各 Benchmark 全部指标", ""])
    for entry_id in ordered_ids:
        baseline = comparison_baselines[entry_id]
        candidate = candidates[entry_id].entry
        primary_names = tuple(
            dict.fromkeys((*_primary_metric_names(candidate), *_primary_metric_names(baseline)))
        )
        all_names = tuple(
            dict.fromkeys(
                (*primary_names, *sorted(set(baseline.metrics) | set(candidate.metrics)))
            )
        )
        baseline_only = sorted(set(baseline.metrics) - set(candidate.metrics))
        candidate_only = sorted(set(candidate.metrics) - set(baseline.metrics))
        lines.extend(
            [
                f"### `{_escape(entry_id)}`",
                "",
                f"- Candidate artifact：`{_escape(candidates[entry_id].root)}`",
                f"- 共同 metrics：{len(set(baseline.metrics) & set(candidate.metrics))}；"
                f"仅 baseline：{len(baseline_only)}；仅 candidate：{len(candidate_only)}。",
                "",
                "<details open>",
                f"<summary>展开 {len(all_names)} 个指标</summary>",
                "",
                "| Metric | Baseline | Candidate | Delta | Unit | Direction | 更优 | Baseline 有效/适用 | Candidate 有效/适用 |",
                "|---|---:|---:|---:|---|---|---|---:|---:|",
            ]
        )
        primary_set = set(primary_names)
        for name in all_names:
            lines.append(
                _comparison_row(
                    name=name,
                    baseline=baseline,
                    candidate=candidate,
                    baseline_label=baseline_label,
                    candidate_label=candidate_label,
                    primary=name in primary_set,
                )
            )
        lines.extend(["", "</details>", ""])

    lines.extend(
        [
            "## 解释口径",
            "",
            "- `Delta` 始终是 Candidate 减 Baseline；对于 lower-is-better 指标，负差值通常更好。",
            "- `有效/适用` 只在该 metric 的真实适用单元内统计，不把结构性 N/A 当失败。",
            "- 主指标在完整表中以粗体标出；availability、parse failure、切片和诊断指标也保留，但不应与主指标等权解读。",
            "- 若 Population check 不是 exact record keys，应先核对数据版本，再把分数差异归因给模型。",
            "",
            (
                f"✅ {len(ordered_ids)} 组比较的 entry、benchmark 和结果 population "
                "通过当前可用检查。"
                if comparable
                else "⚠️ 至少一组比较存在 benchmark 或 population 差异；表格已生成，但不能直接视为严格配对实验。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _three_way_comparison_row(
    *,
    name: str,
    baseline: MetricEntry | None,
    candidate: MetricEntry | None,
    third: MetricEntry | None,
    baseline_label: str,
    candidate_label: str,
    third_label: str,
    primary: bool,
) -> str:
    entries = (baseline, candidate, third)
    raw_metrics = tuple(
        entry.metrics.get(name) if entry is not None else None
        for entry in entries
    )
    values = tuple(
        metric.get("value") if metric is not None else None
        for metric in raw_metrics
    )
    attributes = tuple(
        _metric_attributes(entry, name, _record_metric_attributes(entry.records))
        for entry, metric in zip(entries, raw_metrics)
        if entry is not None and metric is not None
    )
    units = tuple(unit for unit, _direction in attributes)
    unit = (
        units[0]
        if units and len(set(units)) == 1
        else " / ".join(str(value) for value in units) or "—"
    )
    directions = tuple(direction for _unit, direction in attributes)
    direction = (
        directions[0]
        if directions and len(set(directions)) == 1
        else None
    )
    direction_display = (
        _direction_text(direction)
        if directions and len(set(directions)) == 1
        else (
            "⚠️ " + " / ".join(_direction_text(value) for value in directions)
            if directions
            else _direction_text(None)
        )
    )
    metric_label = f"**`{_escape(name)}`**" if primary else f"`{_escape(name)}`"
    displays = tuple(
        "\\"
        if entry is None
        else ("missing" if metric is None else _format_value(value))
        for entry, metric, value in zip(entries, raw_metrics, values)
    )
    coverages = tuple(
        "\\" if entry is None else _coverage_text(_metric_coverage(entry, name))
        for entry in entries
    )
    winner = _winner_many(
        values,
        direction,
        labels=(baseline_label, candidate_label, third_label),
    )
    return (
        f"| {metric_label} | {displays[0]} | {displays[1]} | "
        f"{_format_delta(values[0], values[1])} | {displays[2]} | "
        f"{_format_delta(values[0], values[2])} | `{_escape(unit)}` | "
        f"`{_escape(direction_display)}` | {_escape(winner)} | "
        f"{coverages[0]} | {coverages[1]} | {coverages[2]} |"
    )


def render_three_way_metric_comparison(
    *,
    baseline_path: str | Path,
    candidate_paths: Sequence[str | Path],
    third_path: str | Path,
    baseline_label: str = "Baseline",
    candidate_label: str = "Candidate",
    third_label: str = "Unified-8B",
    baseline_entry_overrides: Mapping[str, str | Path] | None = None,
    candidate_entry_overrides: Mapping[str, str | Path] | None = None,
) -> str:
    """Render an entry-aligned three-system comparison over the entry union.

    Missing whole entries are displayed as ``\\``. A full HUMANUAL entry is
    sliced to match domain-specific candidate entries before scores or
    populations are compared.
    """

    baseline_root, _path, _summary, raw_baseline = load_metric_entries(baseline_path)
    baseline_entries = _apply_reference_entry_overrides(
        tuple(_with_alignx_direct_choice_slices(entry) for entry in raw_baseline),
        baseline_entry_overrides,
        reference_label="baseline",
    )
    candidates = _apply_candidate_entry_overrides(
        _load_unique_candidates(candidate_paths),
        candidate_entry_overrides,
    )
    third_root, _third_summary_path, _third_summary, raw_third = load_metric_entries(third_path)
    third_entries = tuple(_with_alignx_direct_choice_slices(entry) for entry in raw_third)
    ordered_ids = _ordered_three_way_ids(
        baseline_entries,
        candidates,
        third_entries,
    )
    baselines = {
        entry_id: _entry_or_humanual_slice(baseline_entries, entry_id=entry_id)
        for entry_id in ordered_ids
    }
    thirds = {
        entry_id: _entry_or_humanual_slice(third_entries, entry_id=entry_id)
        for entry_id in ordered_ids
    }
    missing_baseline = [entry_id for entry_id in ordered_ids if baselines[entry_id] is None]
    missing_candidate = [entry_id for entry_id in ordered_ids if entry_id not in candidates]
    missing_third = [entry_id for entry_id in ordered_ids if thirds[entry_id] is None]

    def missing_text(entry_ids: Sequence[str]) -> str:
        return ", ".join(f"`{_escape(entry_id)}`" for entry_id in entry_ids) or "无"

    lines = [
        (
            f"# 评测指标三方对比：{_escape(baseline_label)} vs "
            f"{_escape(candidate_label)} vs {_escape(third_label)}"
        ),
        "",
        f"- 基线结果：`{_escape(baseline_root)}`",
        f"- {_escape(candidate_label)} 结果数量：{len(candidate_paths)}；比较 entry 数量：{len(ordered_ids)}。",
        *_entry_override_report_lines("Baseline", baseline_entry_overrides),
        *_entry_override_report_lines("Candidate", candidate_entry_overrides),
        f"- {_escape(third_label)} 结果：`{_escape(third_root)}`",
        f"- `Δ {_escape(candidate_label)}` = {_escape(candidate_label)} - {_escape(baseline_label)}。",
        f"- `Δ {_escape(third_label)}` = {_escape(third_label)} - {_escape(baseline_label)}。",
        "- 比较范围为三方 entry 的并集；缺少整个 entry 的模型列以 `\\` 表示。",
        f"- {_escape(baseline_label)} 缺失 entry：{missing_text(missing_baseline)}。",
        f"- {_escape(candidate_label)} 缺失 entry：{missing_text(missing_candidate)}。",
        f"- {_escape(third_label)} 缺失 entry：{missing_text(missing_third)}。",
        "- 不计算跨 benchmark 总平均。",
        "- AlignX 从 immutable records 派生 direct-choice variant accuracy；未运行的 reference-margin 指标不展示。",
        "",
        "## 数据与运行可比性",
        "",
        (
            f"| Entry | {_escape(candidate_label)} artifact | {_escape(third_label)} artifact | "
            f"Benchmark | {_escape(baseline_label)} status | {_escape(candidate_label)} status | "
            f"{_escape(third_label)} status | Baseline↔Candidate population | Baseline↔Third population |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    comparable = True
    for entry_id in ordered_ids:
        baseline = baselines[entry_id]
        candidate_loaded = candidates.get(entry_id)
        candidate = candidate_loaded.entry if candidate_loaded is not None else None
        third = thirds[entry_id]
        candidate_population, candidate_population_ok = (
            _population_comparison(baseline, candidate)
            if baseline is not None and candidate is not None
            else ("\\", True)
        )
        third_population, third_population_ok = (
            _population_comparison(baseline, third)
            if baseline is not None and third is not None
            else ("\\", True)
        )
        present_entries = tuple(
            entry for entry in (baseline, candidate, third) if entry is not None
        )
        benchmark_ids = tuple(dict.fromkeys(entry.benchmark_id for entry in present_entries))
        benchmark_ok = len(benchmark_ids) <= 1
        comparable = (
            comparable
            and candidate_population_ok
            and third_population_ok
            and benchmark_ok
        )
        benchmark = (
            benchmark_ids[0]
            if len(benchmark_ids) == 1
            else (f"⚠️ {' / '.join(benchmark_ids)}" if benchmark_ids else "\\")
        )
        candidate_artifact = candidate_loaded.root.name if candidate_loaded is not None else "\\"
        third_artifact = third_root.name if third is not None else "\\"
        baseline_status = baseline.status if baseline is not None else "\\"
        candidate_status = candidate.status if candidate is not None else "\\"
        third_status = third.status if third is not None else "\\"
        lines.append(
            f"| `{_escape(entry_id)}` | {_escape(candidate_artifact)} | "
            f"{_escape(third_artifact)} | `{_escape(benchmark)}` | "
            f"{_escape(baseline_status)} | {_escape(candidate_status)} | "
            f"{_escape(third_status)} | {candidate_population} | {third_population} |"
        )

    lines.extend(
        [
            "",
            "## 主指标三方对比总览",
            "",
            (
                f"| Entry | Metric | {_escape(baseline_label)} | {_escape(candidate_label)} | "
                f"Δ {_escape(candidate_label)} | {_escape(third_label)} | "
                f"Δ {_escape(third_label)} | Unit | Direction | 最优 | "
                f"{_escape(baseline_label)} 有效/适用 | {_escape(candidate_label)} 有效/适用 | "
                f"{_escape(third_label)} 有效/适用 |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for entry_id in ordered_ids:
        baseline = baselines[entry_id]
        candidate_loaded = candidates.get(entry_id)
        candidate = candidate_loaded.entry if candidate_loaded is not None else None
        third = thirds[entry_id]
        present_entries = tuple(
            entry for entry in (candidate, baseline, third) if entry is not None
        )
        primary_names = tuple(
            dict.fromkeys(
                name
                for entry in present_entries
                for name in _primary_metric_names(entry)
            )
        )
        for name in primary_names:
            row = _three_way_comparison_row(
                name=name,
                baseline=baseline,
                candidate=candidate,
                third=third,
                baseline_label=baseline_label,
                candidate_label=candidate_label,
                third_label=third_label,
                primary=True,
            )
            lines.append(f"| `{_escape(entry_id)}` |" + row[1:])

    lines.extend(["", "## 各 Benchmark 全部指标", ""])
    for entry_id in ordered_ids:
        baseline = baselines[entry_id]
        candidate_loaded = candidates.get(entry_id)
        candidate = candidate_loaded.entry if candidate_loaded is not None else None
        third = thirds[entry_id]
        present_entries = tuple(
            entry for entry in (candidate, baseline, third) if entry is not None
        )
        primary_names = tuple(
            dict.fromkeys(
                name
                for entry in present_entries
                for name in _primary_metric_names(entry)
            )
        )
        all_names = tuple(
            dict.fromkeys(
                (
                    *primary_names,
                    *sorted(
                        set().union(*(set(entry.metrics) for entry in present_entries))
                    ),
                )
            )
        )
        candidate_artifact = (
            f"`{_escape(candidate_loaded.root)}`"
            if candidate_loaded is not None
            else "\\"
        )
        third_artifact = f"`{_escape(third_root)}`" if third is not None else "\\"
        lines.extend(
            [
                f"### `{_escape(entry_id)}`",
                "",
                f"- {_escape(candidate_label)} artifact：{candidate_artifact}",
                f"- {_escape(third_label)} artifact：{third_artifact}",
                "",
                "<details open>",
                f"<summary>展开 {len(all_names)} 个指标</summary>",
                "",
                (
                    f"| Metric | {_escape(baseline_label)} | {_escape(candidate_label)} | "
                    f"Δ {_escape(candidate_label)} | {_escape(third_label)} | "
                    f"Δ {_escape(third_label)} | Unit | Direction | 最优 | "
                    f"{_escape(baseline_label)} 有效/适用 | {_escape(candidate_label)} 有效/适用 | "
                    f"{_escape(third_label)} 有效/适用 |"
                ),
                "|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|",
            ]
        )
        primary_set = set(primary_names)
        for name in all_names:
            lines.append(
                _three_way_comparison_row(
                    name=name,
                    baseline=baseline,
                    candidate=candidate,
                    third=third,
                    baseline_label=baseline_label,
                    candidate_label=candidate_label,
                    third_label=third_label,
                    primary=name in primary_set,
                )
            )
        lines.extend(["", "</details>", ""])

    lines.extend(
        [
            "## 解释口径",
            "",
            f"- 两个 Delta 均以 {_escape(baseline_label)} 为基准；lower-is-better 指标的负差值通常更好。",
            "- `有效/适用` 只在该指标真实适用的单元内统计，不把结构性 N/A 当作失败。",
            "- `\\` 表示该模型没有评测整个 entry；`missing` 表示 entry 存在，但没有产出该 metric。",
            "- `最优` 在至少两方数值可用、方向一致且明确时判断；并列会同时列出。",
            "- 主指标以粗体标出；availability、parse failure、切片和诊断指标保留在展开表中。",
            "",
            (
                f"✅ {len(ordered_ids)} 组三方比较已按 entry 并集展示；重叠结果的 "
                "benchmark 和 population 通过当前可用检查。"
                if comparable
                else "⚠️ 至少一组三方比较存在 benchmark 或 population 差异；表格可读，但不能直接视为严格配对实验。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_metric_comparison(
    *,
    baseline_path: str | Path,
    candidate_paths: Sequence[str | Path],
    output_path: str | Path,
    baseline_label: str = "Baseline",
    candidate_label: str = "Candidate",
    baseline_entry_overrides: Mapping[str, str | Path] | None = None,
    candidate_entry_overrides: Mapping[str, str | Path] | None = None,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    atomic_write_text(
        output,
        render_metric_comparison(
            baseline_path=baseline_path,
            candidate_paths=candidate_paths,
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            baseline_entry_overrides=baseline_entry_overrides,
            candidate_entry_overrides=candidate_entry_overrides,
        ),
    )
    return output


def write_three_way_metric_comparison(
    *,
    baseline_path: str | Path,
    candidate_paths: Sequence[str | Path],
    third_path: str | Path,
    output_path: str | Path,
    baseline_label: str = "Baseline",
    candidate_label: str = "Candidate",
    third_label: str = "Unified-8B",
    baseline_entry_overrides: Mapping[str, str | Path] | None = None,
    candidate_entry_overrides: Mapping[str, str | Path] | None = None,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    atomic_write_text(
        output,
        render_three_way_metric_comparison(
            baseline_path=baseline_path,
            candidate_paths=candidate_paths,
            third_path=third_path,
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            third_label=third_label,
            baseline_entry_overrides=baseline_entry_overrides,
            candidate_entry_overrides=candidate_entry_overrides,
        ),
    )
    return output


__all__ = [
    "discover_latest_candidate_paths",
    "render_metric_comparison",
    "render_three_way_metric_comparison",
    "write_metric_comparison",
    "write_three_way_metric_comparison",
]
