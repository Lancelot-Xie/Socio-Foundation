"""Read-only Markdown summaries for smoke/formal evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import atomic_write_text, latest_checkpoint_record_dicts, load_json
from .errors import ArtifactError


SUMMARY_FILENAMES = ("suite_summary.json", "smoke_summary.json")


# The report deliberately uses an explicit, reviewable scorecard instead of
# guessing that every aggregate metric is equally important.  Exact entry IDs
# take priority because one benchmark adapter can expose distinct protocols
# (notably UserLM Section 3 and LiC).
# Kept in sync with capability_main_metrics_v1.json by reporting contract tests.
PRIMARY_METRICS: Mapping[str, tuple[str, ...]] = {
    "lifechoices": ("lifechoices.accuracy",),
    "alignx": ("alignx.direct_choice_accuracy",),
    "humanllm": ("humanllm.diagnostic.top5_accuracy",),
    "behaviorchain": (
        "behaviorchain.diagnostic.node_micro_score",
        "behaviorchain.prediction.cum_score",
    ),
    "coser": (
        "coser.scene.character_fidelity",
        "coser.scene.storyline_consistency",
        "coser.scene.storyline_quality",
        "coser.scene.anthropomorphism",
    ),
    "humanual": (
        "humanual.response_alignment",
        "humanual.state_alignment",
    ),
    "userlm_section3": (
        "userlm.intrinsic.intent_adherence",
        "userlm.intrinsic.termination_f1",
        "userlm.intrinsic.ai_detector_human_likelihood",
        "userlm.intrinsic.first_turn_diversity",
        "userlm.intrinsic.intent_decomposition_overlap",
        "userlm.intrinsic.role_adherence",
    ),
    "fantom": ("fantom.item_correct",),
    "social_r1": ("social_r1.accuracy",),
    "agentsense": (
        "agentsense.episode.private_information_accuracy",
        "agentsense.episode.judge_majority",
    ),
    "sotopia": (
        "sotopia.configuration_mean.goal",
        "sotopia.configuration_mean.relationship",
        "sotopia.configuration_mean.knowledge",
        "sotopia.configuration_mean.financial_and_material_benefits",
        "sotopia.configuration_mean.believability",
    ),
    "userlm_lic": (
        "userlm.lic.two_domain_macro.intent_coverage",
        "userlm.lic.two_domain_macro.assistant_task_score",
    ),
    "mirrorbench": (
        "mirrorbench.judge.gteval",
        "mirrorbench.lexical.mattr.z_score_mean",
    ),
    "tau_usi": (
        "tau_usi.usi",
        "tau_usi.eval",
        "tau_usi.outcome_alignment",
    ),
    "userlm_extrinsic": ("userlm.extrinsic.assistant_task_score",),
}


KEY_COMPONENT_METRICS: Mapping[str, tuple[str, ...]] = {
    "sotopia": (
        "sotopia.configuration_mean.believability",
        "sotopia.configuration_mean.relationship",
        "sotopia.configuration_mean.knowledge",
        "sotopia.configuration_mean.secret",
        "sotopia.configuration_mean.social_rules",
        "sotopia.configuration_mean.financial_and_material_benefits",
        "sotopia.configuration_mean.goal",
    ),
    "coser": (
        "coser.scene.anthropomorphism",
        "coser.scene.character_fidelity",
        "coser.scene.storyline_consistency",
        "coser.scene.storyline_quality",
        "coser.scene.bleu",
        "coser.scene.rouge_l",
    ),
    "tau_usi": (
        "tau_usi.eval",
        "tau_usi.usi_without_eval",
        "tau_usi.d1_communication",
        "tau_usi.d2_information",
        "tau_usi.d3_clarification",
        "tau_usi.d4_error_reaction",
        "tau_usi.outcome_alignment",
        "tau_usi.ece",
        "tau_usi.stop_compliance_rate",
    ),
    "lifechoices": ("lifechoices.book_macro_accuracy",),
    "mirrorbench": (
        "mirrorbench.judge.pi",
    ),
    "agentsense": (
        "agentsense.episode.self_goal_completion",
        "agentsense.episode.other_goal_completion",
        "agentsense.episode.judge_average",
        "agentsense.profile_sensitivity_index.goal",
        "agentsense.profile_sensitivity_index.information",
    ),
    "userlm_lic": (
        "userlm.lic.code.assistant_task_score",
        "userlm.lic.math.assistant_task_score",
        "userlm.lic.two_domain_macro.intent_coverage",
        "userlm.lic.two_domain_macro.repeat_required",
        "userlm.lic.code.skip_non_required",
        "userlm.lic.two_domain_macro.additional_demands",
        "userlm.lic.two_domain_macro.user_turn_count",
    ),
    "behaviorchain": (),
    "alignx": ("alignx.alignment_accuracy",),
    "humanllm": (
        "humanllm.supplemental.hit_at_5",
        "humanllm.supplemental.reciprocal_rank",
    ),
    "fantom": (),
    "humanual": (
        "humanual.state_alignment",
        "humanual.embedding_cosine_similarity",
    ),
    "userlm_section3": (
        "userlm.intrinsic.termination_precision",
        "userlm.intrinsic.termination_recall",
    ),
    "userlm_extrinsic": (
        "userlm.extrinsic.intent_coverage",
        "userlm.extrinsic.repeat_required",
        "userlm.extrinsic.skip_non_required",
        "userlm.extrinsic.additional_demands",
        "userlm.intrinsic.first_turn_diversity",
    ),
}


METRIC_NOTES: Mapping[str, str] = {
    "sotopia.configuration_mean.normalized_dimension_mean": "七维等权派生综合分；不是官方单一总分",
    "mirrorbench.judge.gteval": "MirrorBench 多主指标之一",
    "mirrorbench.judge.pi": "逐 episode proxy win rate 的原始均值；论文主表使用 PI-Deviation",
    "mirrorbench.judge.pi_deviation": "论文 PI-Deviation（Δw）= raw PI - 0.5；0 为人类/代理无偏好点",
    "mirrorbench.judge.rnr": "MirrorBench 多主指标之一",
    "mirrorbench.lexical.mattr.z_score_mean": "相对人类基线的 MATTR z-score；越接近 0 越好",
    "mirrorbench.lexical.hdd.z_score_mean": "相对人类基线的 HD-D z-score；越接近 0 越好",
    "mirrorbench.lexical.yules_k.z_score_mean": "相对人类基线的 Yule's K z-score；越接近 0 越好",
    "humanllm.diagnostic.top5_accuracy": "候选排序前五名中包含正确商品的比例；当前比较主表按约定展示",
    "agentsense.episode.judge_majority": "目标达成的三 Judge 多数票",
    "agentsense.episode.private_information_accuracy": "仅适用于含私有信息问题的 episode",
    "userlm.lic.two_domain_macro.assistant_task_score": "先按 task 聚合，再对 code/math 等权",
    "behaviorchain.diagnostic.node_micro_score": "节点级微平均准确率；不是按 persona chain 等权的 AvgScore / CumScore",
    "behaviorchain.prediction.avg_score": "完整预测链的节点准确率，再对 persona chain 等权平均",
    "behaviorchain.prediction.cum_score": "完整预测链的连续正确段累计分，再对 persona chain 等权平均",
    "alignx.direct_choice_accuracy": "当前选定的直接选择协议；非 reference-margin",
    "fantom.item_correct": "逐 item 正确率；当前性能看板按约定展示",
    "fantom.all": "官方 set-level ALL；一个 set 内要求的非自由文本 ToM 问题全部正确才得 1",
    "humanual.response_alignment": "HUMANUAL 官方主指标",
    "userlm.intrinsic.intent_decomposition_overlap": "越低越好",
    "userlm.intrinsic.termination_f1": "只在带真实终止标签的 PRISM cases 上计算",
}


@dataclass(frozen=True)
class MetricEntry:
    entry_id: str
    benchmark_id: str
    status: str
    selected_case_count: int
    repetitions_per_case: int
    result_count: int
    completed_count: int
    failed_count: int
    metrics: Mapping[str, Mapping[str, Any]]
    records: Sequence[Mapping[str, Any]]

    @property
    def expected_record_count(self) -> int:
        planned = self.selected_case_count * self.repetitions_per_case
        return max(planned, self.result_count, len(self.records))


@dataclass(frozen=True)
class MetricCoverage:
    applicable_count: int
    valid_count: int
    total_record_count: int
    basis: str


@dataclass(frozen=True)
class MetricOverrideAudit:
    entry_id: str
    operation: str
    source_path: Path
    source_sha256: str
    validation: str


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"cannot hash metric source {path}: {exc}") from exc
    return digest.hexdigest()


def _nonnegative_int(value: Any, label: str, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{label} must be a non-negative integer")
    numeric = float(value)
    if numeric < 0 or not numeric.is_integer():
        raise ArtifactError(f"{label} must be a non-negative integer")
    return int(numeric)


def _resolve_inside(root: Path, relative: str, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactError(f"{label} escapes the evaluation result directory: {relative}") from exc
    return candidate


def _locate_summary(input_path: str | Path) -> tuple[Path, Path, Mapping[str, Any]]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        summary_path = path
        root = path.parent
    elif path.is_dir():
        matches = [path / name for name in SUMMARY_FILENAMES if (path / name).is_file()]
        if not matches:
            raise ArtifactError(
                f"no evaluation summary found in {path}; expected one of {SUMMARY_FILENAMES}"
            )
        summary_path = matches[0]
        root = path
    else:
        raise ArtifactError(f"evaluation result path does not exist: {path}")
    summary = load_json(summary_path)
    return root, summary_path, _mapping(summary, str(summary_path))


def _read_latest_records(path: Path | None) -> tuple[Mapping[str, Any], ...]:
    if path is None:
        return ()
    if not path.is_file():
        raise ArtifactError(f"referenced records artifact is missing: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot read records artifact {path}: {exc}") from exc
    rows: list[Mapping[str, Any]] = []
    lines = raw.splitlines(keepends=True)
    for index, encoded in enumerate(lines, start=1):
        if not encoded.strip():
            continue
        complete = encoded.endswith((b"\n", b"\r"))
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == len(lines) and not complete:
                # Match checkpoint recovery: an interrupted final append is not
                # a completed record and is ignored.
                break
            raise ArtifactError(f"invalid records JSONL at {path}:{index}: {exc}") from exc
        rows.append(_mapping(value, f"{path}:{index}"))
    return tuple(latest_checkpoint_record_dicts(rows))


def _metric_mapping(value: Any, label: str) -> Mapping[str, Mapping[str, Any]]:
    raw = _mapping(value, label)
    return {
        str(name): _mapping(metric, f"{label}.{name}")
        for name, metric in raw.items()
    }


def _records_path(
    *,
    campaign_root: Path,
    root_entry: Mapping[str, Any],
    child_root: Path,
    child_summary: Mapping[str, Any],
) -> Path | None:
    direct = root_entry.get("records")
    if isinstance(direct, str) and direct:
        return _resolve_inside(campaign_root, direct, "entry records path")
    artifacts = child_summary.get("artifacts")
    if isinstance(artifacts, Mapping):
        relative = artifacts.get("records")
        if isinstance(relative, str) and relative:
            return _resolve_inside(child_root, relative, "child records path")
    paths = child_summary.get("artifact_paths")
    if isinstance(paths, Mapping):
        relative = paths.get("records")
        if isinstance(relative, str) and relative:
            return _resolve_inside(child_root, relative, "child records path")
    return None


def _entry_from_child(
    *,
    campaign_root: Path,
    entry_id: str,
    root_entry: Mapping[str, Any],
    child_root: Path,
    child_summary: Mapping[str, Any],
) -> MetricEntry:
    metrics = _metric_mapping(child_summary.get("metrics", {}), f"{entry_id}.metrics")
    records = _read_latest_records(
        _records_path(
            campaign_root=campaign_root,
            root_entry=root_entry,
            child_root=child_root,
            child_summary=child_summary,
        )
    )
    selected = _nonnegative_int(
        child_summary.get("selected_case_count", root_entry.get("selected_case_count")),
        f"{entry_id}.selected_case_count",
    )
    repetitions = _nonnegative_int(
        child_summary.get("repetitions_per_case", 1),
        f"{entry_id}.repetitions_per_case",
        default=1,
    )
    if repetitions == 0:
        raise ArtifactError(f"{entry_id}.repetitions_per_case must be positive")
    completed_count = _nonnegative_int(
        child_summary.get("completed_count", root_entry.get("completed_count")),
        f"{entry_id}.completed_count",
    )
    failed_count = _nonnegative_int(
        child_summary.get("failed_count", root_entry.get("failed_count")),
        f"{entry_id}.failed_count",
    )
    status = root_entry.get("status") or child_summary.get("status")
    if not status:
        status = "completed" if failed_count == 0 else "completed_with_failures"
    return MetricEntry(
        entry_id=entry_id,
        benchmark_id=str(
            child_summary.get("benchmark_id") or root_entry.get("benchmark_id") or entry_id
        ),
        status=str(status),
        selected_case_count=selected,
        repetitions_per_case=repetitions,
        result_count=_nonnegative_int(
            child_summary.get("result_count", len(records)),
            f"{entry_id}.result_count",
            default=len(records),
        ),
        completed_count=completed_count,
        failed_count=failed_count,
        metrics=metrics,
        records=records,
    )


def load_metric_entries(input_path: str | Path) -> tuple[Path, Path, Mapping[str, Any], tuple[MetricEntry, ...]]:
    """Load current formal/smoke, legacy suite, or one child summary."""

    root, summary_path, summary = _locate_summary(input_path)
    raw_entries = summary.get("entries")
    entries: list[MetricEntry] = []
    if isinstance(raw_entries, Sequence) and not isinstance(raw_entries, (str, bytes)):
        for index, raw in enumerate(raw_entries):
            root_entry = _mapping(raw, f"entries[{index}]")
            entry_id = str(root_entry.get("id") or root_entry.get("benchmark_id") or "").strip()
            if not entry_id:
                raise ArtifactError(f"entries[{index}] is missing id")
            relative = root_entry.get("child_summary")
            if not isinstance(relative, str) or not relative:
                raise ArtifactError(f"entry {entry_id!r} is missing child_summary")
            child_path = _resolve_inside(root, relative, f"entry {entry_id!r} child_summary")
            child_summary = _mapping(load_json(child_path), str(child_path))
            entries.append(
                _entry_from_child(
                    campaign_root=root,
                    entry_id=entry_id,
                    root_entry=root_entry,
                    child_root=child_path.parent,
                    child_summary=child_summary,
                )
            )
    elif isinstance(summary.get("benchmarks"), Mapping):
        # Legacy all-benchmark suite summary keeps child summaries inline.
        for entry_id, raw in summary["benchmarks"].items():
            child_summary = _mapping(raw, f"benchmarks.{entry_id}")
            entries.append(
                _entry_from_child(
                    campaign_root=root,
                    entry_id=str(entry_id),
                    root_entry=child_summary,
                    child_root=root,
                    child_summary=child_summary,
                )
            )
    elif "benchmark_id" in summary and "metrics" in summary:
        entry_id = str(summary.get("benchmark_id") or "benchmark")
        entries.append(
            _entry_from_child(
                campaign_root=root,
                entry_id=entry_id,
                root_entry=summary,
                child_root=root,
                child_summary=summary,
            )
        )
    else:
        raise ArtifactError(
            f"unrecognized evaluation summary shape: {summary_path}; expected entries, benchmarks, or benchmark_id+metrics"
        )
    return root, summary_path, summary, tuple(entries)


def _suite_identity(root: Path) -> Mapping[str, Any] | None:
    for filename in ("suite_plan.json", "smoke_plan.json"):
        path = root / filename
        if not path.is_file():
            continue
        plan = _mapping(load_json(path), str(path))
        model = plan.get("model")
        model = model if isinstance(model, Mapping) else {}
        return {
            "candidate_model": model.get("model"),
            "candidate_revision": model.get("model_revision"),
            "global_eval_model": plan.get("global_eval_model"),
            "seed": plan.get("seed"),
        }
    return None


def _validate_replacement_identity(
    *,
    base_root: Path,
    replacement_root: Path,
) -> str:
    base = _suite_identity(base_root)
    replacement_identity = _suite_identity(replacement_root)
    if base is None or replacement_identity is None:
        return "benchmark matched; suite candidate/support identity unavailable for one source"
    mismatches = {
        key: (base.get(key), replacement_identity.get(key))
        for key in ("candidate_model", "candidate_revision", "global_eval_model", "seed")
        if base.get(key) is not None
        and replacement_identity.get(key) is not None
        and base.get(key) != replacement_identity.get(key)
    }
    if mismatches:
        raise ArtifactError(
            "replacement entry belongs to a different suite identity: "
            + ", ".join(
                f"{key}={old!r}->{new!r}" for key, (old, new) in mismatches.items()
            )
        )
    return "benchmark, candidate model/revision, support preset, and seed matched"


def _corrected_metric_entry(
    entry: MetricEntry,
    source: str | Path,
) -> tuple[MetricEntry, MetricOverrideAudit]:
    source_path = Path(source).expanduser().resolve()
    if source_path.is_dir():
        source_path = source_path / "corrected_metrics.json"
    if not source_path.is_file():
        raise ArtifactError(f"corrected metrics source does not exist: {source_path}")
    document = _mapping(load_json(source_path), str(source_path))
    metrics = _metric_mapping(document.get("metrics"), f"{source_path}.metrics")
    manifest_path = source_path.parent / "recomputation_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError(
            "--corrected-metrics requires the AgentSense/MirrorBench post-hoc "
            f"recomputation_manifest.json beside {source_path}"
        )
    manifest = _mapping(load_json(manifest_path), str(manifest_path))
    if str(manifest.get("benchmark_id") or "") != entry.benchmark_id:
        raise ArtifactError(
            f"corrected metrics benchmark mismatch for {entry.entry_id}: "
            f"expected {entry.benchmark_id!r}, got {manifest.get('benchmark_id')!r}"
        )
    source_run_id = str(document.get("source_run_id") or manifest.get("source_run_id") or "")
    observed_run_ids = {
        str(record.get("run_id"))
        for record in entry.records
        if record.get("run_id") is not None
    }
    if source_run_id and observed_run_ids and observed_run_ids != {source_run_id}:
        raise ArtifactError(
            f"corrected metrics source_run_id {source_run_id!r} does not match "
            f"{entry.entry_id} records {sorted(observed_run_ids)}"
        )
    return (
        replace(entry, metrics=metrics),
        MetricOverrideAudit(
            entry_id=entry.entry_id,
            operation="corrected_metrics",
            source_path=source_path,
            source_sha256=_sha256(source_path),
            validation="benchmark and source run_id matched; records/status retained from base suite",
        ),
    )


def apply_metric_overrides(
    *,
    base_root: Path,
    entries: Sequence[MetricEntry],
    corrected_metrics: Mapping[str, str | Path] | None = None,
    replacement_entries: Mapping[str, str | Path] | None = None,
) -> tuple[tuple[MetricEntry, ...], tuple[MetricOverrideAudit, ...]]:
    """Apply explicit, non-mutating aggregate corrections and whole-entry replacements."""

    corrected = dict(corrected_metrics or {})
    replacements = dict(replacement_entries or {})
    overlap = sorted(set(corrected) & set(replacements))
    if overlap:
        raise ArtifactError(
            "one entry cannot use both --corrected-metrics and --replace-entry: "
            + ", ".join(overlap)
        )
    known = {entry.entry_id for entry in entries}
    unknown = sorted((set(corrected) | set(replacements)) - known)
    if unknown:
        raise ArtifactError("metric override names unknown base entries: " + ", ".join(unknown))

    effective: list[MetricEntry] = []
    audits: list[MetricOverrideAudit] = []
    for entry in entries:
        if entry.entry_id in replacements:
            replacement_root, replacement_summary, _summary, candidates = load_metric_entries(
                replacements[entry.entry_id]
            )
            exact = [candidate for candidate in candidates if candidate.entry_id == entry.entry_id]
            if not exact:
                exact = [
                    candidate
                    for candidate in candidates
                    if candidate.benchmark_id == entry.benchmark_id
                ]
            if len(exact) != 1:
                raise ArtifactError(
                    f"replacement source for {entry.entry_id!r} must resolve to exactly one "
                    f"matching entry; found {len(exact)}"
                )
            candidate = exact[0]
            if candidate.benchmark_id != entry.benchmark_id:
                raise ArtifactError(
                    f"replacement benchmark mismatch for {entry.entry_id}: "
                    f"{entry.benchmark_id!r} vs {candidate.benchmark_id!r}"
                )
            validation = _validate_replacement_identity(
                base_root=base_root,
                replacement_root=replacement_root,
            )
            effective.append(replace(candidate, entry_id=entry.entry_id))
            audits.append(
                MetricOverrideAudit(
                    entry_id=entry.entry_id,
                    operation="replace_entry",
                    source_path=replacement_summary,
                    source_sha256=_sha256(replacement_summary),
                    validation=validation + "; records/status/metrics replaced together",
                )
            )
            continue
        if entry.entry_id in corrected:
            corrected_entry, audit = _corrected_metric_entry(
                entry,
                corrected[entry.entry_id],
            )
            effective.append(corrected_entry)
            audits.append(audit)
            continue
        effective.append(entry)
    return tuple(effective), tuple(audits)


def _record_metric_counts(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, tuple[int, int]]:
    applicable: dict[str, int] = {}
    valid: dict[str, int] = {}
    for record in records:
        raw_metrics = record.get("metrics")
        if not isinstance(raw_metrics, Sequence) or isinstance(raw_metrics, (str, bytes)):
            continue
        present_in_record: set[str] = set()
        available_in_record: set[str] = set()
        for raw in raw_metrics:
            if not isinstance(raw, Mapping):
                continue
            name = raw.get("name")
            if isinstance(name, str) and name:
                if not _record_metric_is_applicable(record, name, raw):
                    continue
                present_in_record.add(name)
                if raw.get("value") is not None:
                    available_in_record.add(name)
        for name in present_in_record:
            applicable[name] = applicable.get(name, 0) + 1
        for name in available_in_record:
            valid[name] = valid.get(name, 0) + 1
    return {
        name: (applicable_count, valid.get(name, 0))
        for name, applicable_count in applicable.items()
    }


def _record_metric_is_applicable(
    record: Mapping[str, Any],
    name: str,
    metric: Mapping[str, Any],
) -> bool:
    """Return whether a per-case metric belongs to that case's protocol.

    Adapters often emit a stable metric schema with explicit null placeholders.
    A placeholder marked ``not_applicable`` must not be counted as an evaluator
    failure.  The UserLM variant fallback keeps compatibility with older result
    artifacts produced before those explicit markers were added.
    """

    metric_metadata = metric.get("metadata")
    metric_metadata = metric_metadata if isinstance(metric_metadata, Mapping) else {}
    availability = str(metric_metadata.get("availability") or "").casefold()
    if availability == "not_applicable" or availability.startswith("not_applicable_"):
        return False

    record_metadata = record.get("metadata")
    record_metadata = record_metadata if isinstance(record_metadata, Mapping) else {}
    if not name.startswith("userlm."):
        return True

    variant = str(record_metadata.get("variant") or "")
    if name.startswith("userlm.extrinsic."):
        if variant != "extrinsic_verifiable":
            return False
        if name == "userlm.extrinsic.skip_non_required":
            if metric.get("value") is None and metric_metadata.get(
                "not_applicable_when_all_shards_required"
            ):
                return False
            # The active LiC protocol defines all GSM8K shards as required.
            # Older capability-failure records wrote a synthetic zero before
            # this structural applicability rule was fixed in the scorer.
            if (
                str(record_metadata.get("source_task") or "").casefold() == "math"
                and isinstance(metric_metadata.get("target_output_failure"), Mapping)
            ):
                return False
        return True
    expected_variants = {
        "userlm.intrinsic.intent_decomposition_overlap": "intrinsic_prism",
        "userlm.intrinsic.ai_detector_human_likelihood": "intrinsic_prism",
        "userlm.intrinsic.role_adherence": "intrinsic_role_adherence",
        "userlm.intrinsic.intent_adherence": "intrinsic_intent_adherence",
    }
    expected = expected_variants.get(name)
    return expected is None or variant == expected


def _record_metric_attributes(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    attributes: dict[str, dict[str, Any]] = {}
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
            item = attributes.setdefault(name, {})
            if raw.get("unit") is not None:
                item.setdefault("unit", raw.get("unit"))
            if raw.get("direction") is not None:
                item.setdefault("direction", raw.get("direction"))
    return attributes


def _count_from_metadata(metadata: Mapping[str, Any], *, availability_metric: bool) -> int | None:
    candidates: list[tuple[int, str, int]] = []
    for key, raw in metadata.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        number = float(raw)
        if number < 0 or not number.is_integer() or not str(key).endswith("_count"):
            continue
        key_text = str(key)
        if availability_metric and (key_text.startswith("available_") or key_text.startswith("valid_")):
            priority = 0
        elif key_text.startswith("valid_"):
            priority = 1
        elif key_text.startswith("available_"):
            priority = 2
        elif key_text.startswith("completed_"):
            priority = 3
        elif key_text.startswith("eligible_"):
            priority = 4
        else:
            continue
        candidates.append((priority, key_text, int(number)))
    return min(candidates)[2] if candidates else None


def _numeric_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _metadata_count(metadata: Mapping[str, Any], key: str) -> int | None:
    return _numeric_count(metadata.get(key))


def metric_coverage(
    name: str,
    metric: Mapping[str, Any],
    *,
    record_counts: Mapping[str, tuple[int, int]],
    total_records: int,
    completed_records: int,
) -> MetricCoverage:
    """Separate metric applicability from validity.

    A sparse agent/category/slice metric can be perfectly valid over only the
    records to which it applies. Aggregate-only metrics retain their natural
    unit instead of being mislabeled as missing case records.
    """

    direct = record_counts.get(name)
    if direct is not None:
        applicable, valid = direct
        return MetricCoverage(
            min(applicable, total_records),
            min(valid, applicable, total_records),
            total_records,
            "record_metric",
        )
    lowered = name.casefold()
    if lowered.endswith("case_completion_rate") or lowered.endswith("case_failure_count"):
        return MetricCoverage(total_records, total_records, total_records, "record_population")

    metadata = metric.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    # Explicit aggregate-unit pairs take priority. These cover SOTOPIA
    # configuration means, BehaviorChain chains, and lexical episode metrics.
    count_pairs = (
        ("configuration_count", "available_configuration_count"),
        ("chain_count", "available_chain_count"),
        ("completed_episode_count", "available_episode_count"),
        ("episode_count", "valid_episode_count"),
        ("case_count", "available_case_count"),
        ("applicable_task_count", "available_task_count"),
        ("task_count", "available_task_count"),
        ("domain_count", "available_domain_count"),
        ("batch_count", "available_batch_count"),
    )
    for applicable_key, valid_key in count_pairs:
        applicable = _metadata_count(metadata, applicable_key)
        valid = _metadata_count(metadata, valid_key)
        if applicable is not None and valid is not None:
            return MetricCoverage(
                applicable,
                min(valid, applicable),
                total_records,
                "record_population" if applicable == total_records else "aggregate_units",
            )

    valid_metadata_count = _count_from_metadata(
        metadata,
        availability_metric="availability" in lowered,
    )
    excluded_or_unavailable = sum(
        count or 0
        for count in (
            _metadata_count(metadata, "unavailable_count"),
            _metadata_count(metadata, "excluded_episode_count"),
            _metadata_count(metadata, "incomplete_group_count"),
            _metadata_count(metadata, "evaluator_unavailable_group_count"),
        )
    )
    aggregate_valid = _numeric_count(metric.get("denominator"))
    if aggregate_valid is None:
        # Some official population metrics (for example UserLM termination
        # precision/recall/F1) keep the labeled population size in metadata
        # because their mathematical MetricValue denominator is not a simple
        # accuracy denominator.
        aggregate_valid = _metadata_count(metadata, "denominator")
    if aggregate_valid is None:
        aggregate_valid = valid_metadata_count
    if aggregate_valid is not None:
        valid = aggregate_valid if metric.get("value") is not None else 0
        applicable = aggregate_valid + excluded_or_unavailable
        if applicable == 0 and valid_metadata_count is not None:
            applicable = valid_metadata_count
        basis = "record_population" if applicable == total_records else "aggregate_units"
        return MetricCoverage(applicable, min(valid, applicable), total_records, basis)

    eligible = next(
        (
            count
            for count in (
                _metadata_count(metadata, "eligible_case_count"),
                _metadata_count(metadata, "eligible_episode_count"),
                _metadata_count(metadata, "eligible_template_count"),
            )
            if count is not None
        ),
        None,
    )
    if eligible is not None:
        valid = valid_metadata_count or 0
        if metric.get("value") is None:
            valid = 0
        return MetricCoverage(eligible, min(valid, eligible), total_records, "aggregate_units")

    # Population metrics such as tau-USI components have no per-record metric
    # or count because they are defined over the whole simulator population.
    applicable = total_records
    valid = min(completed_records, total_records) if metric.get("value") is not None else 0
    return MetricCoverage(applicable, valid, total_records, "record_population")


def metric_valid_count(
    name: str,
    metric: Mapping[str, Any],
    *,
    record_counts: Mapping[str, tuple[int, int]],
    total_records: int,
    completed_records: int,
) -> tuple[int, str]:
    """Backward-compatible projection of :func:`metric_coverage`."""

    coverage = metric_coverage(
        name,
        metric,
        record_counts=record_counts,
        total_records=total_records,
        completed_records=completed_records,
    )
    return coverage.valid_count, coverage.basis


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_value(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.8g}"
    return str(value)


def _format_direction(value: Any) -> str:
    text = str(value or "unspecified")
    symbols = {
        "higher_is_better": "↑ higher_is_better",
        "lower_is_better": "↓ lower_is_better",
        "closer_to_zero": "→0 closer_to_zero",
    }
    return symbols.get(text, text)


def _coverage(valid: int, total: int) -> str:
    if total <= 0:
        return "—"
    return f"{valid} / {total} ({valid / total:.2%})"


def _applicability(coverage: MetricCoverage) -> str:
    if coverage.basis == "aggregate_units":
        return f"{coverage.applicable_count} aggregate units"
    return _coverage(coverage.applicable_count, coverage.total_record_count)


def _validity(coverage: MetricCoverage) -> str:
    return _coverage(coverage.valid_count, coverage.applicable_count)


def _scorecard_key(entry: MetricEntry) -> str:
    if entry.entry_id in PRIMARY_METRICS or entry.entry_id in KEY_COMPONENT_METRICS:
        return entry.entry_id
    if entry.benchmark_id == "userlm":
        if any(name.startswith("userlm.lic.") for name in entry.metrics):
            return "userlm_lic"
        variant_count = entry.metrics.get("userlm.variant_count", {}).get("value")
        if isinstance(variant_count, (int, float)) and not isinstance(variant_count, bool) and variant_count > 1:
            return "userlm_section3"
        return "userlm_extrinsic"
    return entry.benchmark_id


def _is_health_metric(name: str) -> bool:
    lowered = name.casefold()
    return any(
        token in lowered
        for token in (
            "case_completion_rate",
            "case_failure_count",
            "parse_failure_rate",
            "availability_rate",
            "failure_count",
            "unavailable_count",
            "configuration_count",
            "complete_role_pair_rate",
            "chain_structure_complete_rate",
            "chain_score_availability_rate",
            "suite_complete",
        )
    )


def _primary_metric_names(entry: MetricEntry) -> tuple[str, ...]:
    configured = PRIMARY_METRICS.get(_scorecard_key(entry), ())
    if configured:
        return configured
    official = tuple(
        name
        for name, metric in sorted(entry.metrics.items())
        if isinstance(metric.get("metadata"), Mapping)
        and metric["metadata"].get("official_primary") is True
    )
    if official:
        return official
    # Unknown third-party adapters remain useful: expose their non-health
    # aggregates instead of silently producing an empty scorecard.
    return tuple(name for name in sorted(entry.metrics) if not _is_health_metric(name))


def _component_metric_names(entry: MetricEntry) -> tuple[str, ...]:
    primary = set(_primary_metric_names(entry))
    return tuple(
        name
        for name in KEY_COMPONENT_METRICS.get(_scorecard_key(entry), ())
        if name not in primary
    )


def _metric_render_fields(
    entry: MetricEntry,
    name: str,
    *,
    record_counts: Mapping[str, tuple[int, int]],
    record_attributes: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Any, Any, MetricCoverage]:
    metric = entry.metrics[name]
    source_attributes = record_attributes.get(name, {})
    unit = metric.get("unit") or source_attributes.get("unit")
    if unit is None and name.endswith(".normalized"):
        unit = "0_to_1"
    unit = unit or "—"
    direction = metric.get("direction") or source_attributes.get("direction")
    coverage = metric_coverage(
        name,
        metric,
        record_counts=record_counts,
        total_records=entry.expected_record_count,
        completed_records=entry.completed_count,
    )
    return metric, unit, direction, coverage


def _entry_metric_table(
    entry: MetricEntry,
    names: Sequence[str],
    *,
    include_notes: bool = False,
) -> list[str]:
    record_counts = _record_metric_counts(entry.records)
    record_attributes = _record_metric_attributes(entry.records)
    header = (
        "| Metric | Value | Unit | Direction | 适用范围 | 有效率（有效/适用） | 说明 |"
        if include_notes
        else "| Metric | Value | Unit | Direction | 适用范围 | 有效率（有效/适用） |"
    )
    separator = (
        "|---|---:|---|---|---:|---:|---|"
        if include_notes
        else "|---|---:|---|---|---:|---:|"
    )
    lines = [header, separator]
    rendered = 0
    for name in names:
        if name not in entry.metrics:
            continue
        metric, unit, direction, coverage = _metric_render_fields(
            entry,
            name,
            record_counts=record_counts,
            record_attributes=record_attributes,
        )
        row = (
            f"| `{_escape(name)}` | {_escape(_format_value(metric.get('value')))} | "
            f"`{_escape(unit)}` | `{_escape(_format_direction(direction))}` | "
            f"{_applicability(coverage)} | {_validity(coverage)}"
        )
        if include_notes:
            row += f" | {_escape(METRIC_NOTES.get(name, ''))}"
        lines.append(row + " |")
        rendered += 1
    if rendered == 0:
        suffix = " |" if include_notes else ""
        lines.append(f"| _No selected metrics available_ | unavailable | — | — | — | —{suffix} |")
    return lines


def _primary_overview_table(entries: Sequence[MetricEntry]) -> list[str]:
    """Group fixed headline metrics by F/S/U/T/N without score recomputation."""
    mapping = load_json(Path(__file__).resolve().parent / "resources/reporting/capability_main_metrics_v1.json")
    by_entry = {entry.entry_id: entry for entry in entries}
    dimensions = {d["id"]: d["name_zh"] for d in mapping["dimensions"]}
    lines = [
        "| 能力维度 | Benchmark entry | 主要指标 | Value | Unit | Direction | 有效率（有效/适用） | 运行状态 |",
        "|---|---|---|---:|---|---|---:|---|",
    ]
    cached = {entry.entry_id: (_record_metric_counts(entry.records), _record_metric_attributes(entry.records))
              for entry in entries}
    for row in mapping["metrics"]:
        entry = by_entry.get(row["entry"])
        if entry is None:
            continue
        name = row["metric"]
        dimension = f"{row['dimension']}. {dimensions[row['dimension']]}"
        status = f"{'✅' if entry.status == 'completed' and entry.failed_count == 0 else '⚠️'} `{_escape(entry.status)}`"
        prefix = f"| {dimension} | `{_escape(entry.entry_id)}` | `{_escape(name)}` |"
        if name not in entry.metrics:
            lines.append(f"{prefix} missing | `—` | `—` | — | {status} |")
            continue
        counts, attributes = cached[entry.entry_id]
        metric, unit, direction, coverage = _metric_render_fields(
            entry, name, record_counts=counts, record_attributes=attributes)
        lines.append(f"{prefix} {_escape(_format_value(metric.get('value')))} | `{_escape(unit)}` | "
                     f"`{_escape(_format_direction(direction))}` | {_validity(coverage)} | {status} |")
    return lines


def _automatic_findings(entries: Sequence[MetricEntry]) -> list[str]:
    findings: list[str] = []
    failed_entries = [entry.entry_id for entry in entries if entry.failed_count or entry.status != "completed"]
    if failed_entries:
        findings.append(
            "⚠️ 存在未完全完成的 entry：" + ", ".join(f"`{_escape(name)}`" for name in failed_entries) + "。"
        )
    else:
        findings.append("✅ 所有 benchmark entry 均完成，且没有 case 级基础设施失败。")

    for entry in entries:
        primary_names = _primary_metric_names(entry)
        counts = _record_metric_counts(entry.records)
        attributes = _record_metric_attributes(entry.records)
        for name in primary_names:
            metric = entry.metrics.get(name)
            if metric is None:
                findings.append(f"⚠️ `{entry.entry_id}` 缺少预期主指标 `{name}`。")
                continue
            _metric, _unit, _direction, coverage = _metric_render_fields(
                entry,
                name,
                record_counts=counts,
                record_attributes=attributes,
            )
            if metric.get("value") is None:
                findings.append(f"⚠️ `{entry.entry_id}` 主指标 `{name}` 当前不可用。")
            elif coverage.applicable_count and coverage.valid_count < coverage.applicable_count:
                findings.append(
                    f"⚠️ `{entry.entry_id}` 主指标 `{name}` 仅 "
                    f"{coverage.valid_count}/{coverage.applicable_count} 个适用单元有效。"
                )

        if entry.entry_id == "alignx":
            alignment = entry.metrics.get("alignx.alignment_accuracy", {})
            direct = entry.metrics.get("alignx.direct_choice_accuracy", {})
            if alignment.get("value") is None and direct.get("value") is not None:
                findings.append(
                    "ℹ️ `alignx` 按当前约定未运行 reference-margin；性能看板使用 "
                    "`direct_choice_accuracy`，不把 unavailable 当作运行故障。"
                )
        if entry.entry_id == "behaviorchain":
            node_micro = entry.metrics.get("behaviorchain.diagnostic.node_micro_score", {})
            structure = entry.metrics.get("behaviorchain.chain_structure_complete_rate", {}).get("value")
            if structure == 1:
                findings.append(
                    "ℹ️ `behaviorchain` 数据包含完整 persona chain；F 展示节点微平均，T 展示 "
                    "`CumScore`，`AvgScore` 保留在详细诊断中。"
                )
            elif node_micro.get("value") is not None:
                findings.append(
                    "ℹ️ `behaviorchain` 当前数据不支持完整 persona chain；性能看板按约定使用 "
                    "`node_micro_score`，不把它表述为官方 chain score。"
                )
        if entry.entry_id == "userlm_lic":
            detector = entry.metrics.get("userlm.intrinsic.ai_detector_human_likelihood", {})
            if detector.get("value") is not None:
                findings.append(
                    "⚠️ `userlm_lic` 产物含历史 AI-detector 数值，但该指标只适用于 PRISM；"
                    "报告不把它列入 LiC 性能，更新后的 scorer 也不会再由能力失败写入伪零分。"
                )
            skip_macro = entry.metrics.get("userlm.lic.two_domain_macro.skip_non_required", {})
            if skip_macro.get("value") is not None:
                findings.append(
                    "⚠️ 当前 LiC 旧产物的 two-domain `skip_non_required` 受历史 fallback 污染；"
                    "数学任务没有 non-required shard，因此只报告 code-domain 值。"
                )
        termination = entry.metrics.get("userlm.intrinsic.termination_f1")
        if termination is not None:
            if isinstance(termination, Mapping) and termination.get("value") == 0:
                metadata = termination.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                findings.append(
                    "ℹ️ `userlm_section3` termination F1=0 是有效能力结果，不是运行缺失："
                    f"TP={_format_value(metadata.get('tp'))}、FP={_format_value(metadata.get('fp'))}、"
                    f"FN={_format_value(metadata.get('fn'))}，适用 population="
                    f"{_format_value(metadata.get('denominator'))}。"
                )
    return findings


def _input_truncation_table(entries: Sequence[MetricEntry]) -> list[str]:
    rows = []
    for entry in entries:
        enabled = applied_cases = applied_requests = rejected = 0
        for record in entry.records:
            budget = (record.get("metadata") or {}).get("episode_output_token_budget") or {}
            if budget.get("input_truncation") != "protected_left_v1":
                continue
            enabled += 1
            events = budget.get("input_truncation_events") or ()
            applied = sum(event.get("status") == "applied" for event in events)
            applied_cases += int(applied > 0)
            applied_requests += applied
            rejected += sum(event.get("status") != "applied" for event in events)
        if enabled:
            rows.append(f"| `{_escape(entry.entry_id)}` | {enabled} | {applied_cases} | {applied_requests} | {rejected} |")
    if not rows:
        return []
    return ["### 输入裁剪", "",
            "仅统计启用 protected_left_v1 的已记录 case/repetition；不会将未裁剪题目排除出指标。"
            "无法裁剪的请求仍按上下文超限处理。逐请求长度和删除区域见 records.jsonl 的 episode_output_token_budget。", "",
            "| Benchmark entry | 启用策略的记录数 | 实际裁剪记录数 | 裁剪请求数 | 无法裁剪请求数 |",
            "|---|---:|---:|---:|---:|", *rows, ""]


def render_metric_markdown(
    *,
    input_path: str | Path,
    corrected_metrics: Mapping[str, str | Path] | None = None,
    replacement_entries: Mapping[str, str | Path] | None = None,
) -> str:
    root, summary_path, summary, entries = load_metric_entries(input_path)
    entries, override_audits = apply_metric_overrides(
        base_root=root,
        entries=entries,
        corrected_metrics=corrected_metrics,
        replacement_entries=replacement_entries,
    )
    effective_status = (
        "completed"
        if all(entry.status == "completed" and entry.failed_count == 0 for entry in entries)
        else "completed_with_failures"
    )
    lines = [
        "# 评测指标汇总",
        "",
        f"- 结果路径：`{_escape(root)}`",
        f"- 汇总文件：`{_escape(summary_path.name)}`",
        f"- 套件：`{_escape(summary.get('suite_id', summary.get('profile', 'unspecified')))}`",
        f"- 有效状态：`{_escape(effective_status)}`",
        f"- 基础汇总状态：`{_escape(summary.get('status', 'unknown'))}`",
        f"- Benchmark entry 数量：{len(entries)}",
        "- 跨 benchmark 总平均：不计算；不同指标的量纲不可直接混合。",
        "",
    ]
    if override_audits:
        lines.extend(
            [
                "## 修正与替换来源",
                "",
                "以下覆盖由命令行显式指定；基础 suite、checkpoint、metrics 和 records 均未改写。",
                "",
                "| Entry | 操作 | 来源 | SHA-256 | 校验 |",
                "|---|---|---|---|---|",
            ]
        )
        for audit in override_audits:
            lines.append(
                f"| `{_escape(audit.entry_id)}` | `{_escape(audit.operation)}` | "
                f"`{_escape(audit.source_path)}` | `{audit.source_sha256}` | "
                f"{_escape(audit.validation)} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "- 修正覆盖：未启用；全部 entry 直接读取基础 suite。",
                "",
            ]
        )
    lines.extend(
        [
            "## Benchmark Entry 概览",
            "",
            "| Entry | Benchmark | Status | 选中 cases | 结果记录 | 完成 | 失败 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for entry in entries:
        lines.append(
            f"| `{_escape(entry.entry_id)}` | `{_escape(entry.benchmark_id)}` | "
            f"`{_escape(entry.status)}` | {entry.selected_case_count} | "
            f"{entry.expected_record_count} | {entry.completed_count} | {entry.failed_count} |"
        )

    lines.extend(
        [
            "",
            "## 五维主指标总览",
            "",
            "主表按 F/S/U/T/N 五个能力维度展开；同一数据集可对应多行。不同指标量纲不同，"
            "这里只并列展示，不计算跨 benchmark 总平均。",
            "",
            *_primary_overview_table(entries),
        ]
    )

    lines.extend(["", "## 自动核查结论", ""])
    for finding in _automatic_findings(entries):
        lines.append(f"- {finding}")

    lines.extend(
        [
            "",
            "## 逐数据集指标明细",
            "",
            "这里只放用于比较待测模型的主指标。运行完成率、availability、persona/character/"
            "category 切片不进入主看板；多主指标 benchmark 不强行合成一个未经定义的总分。",
            "",
        ]
    )
    for entry in entries:
        lines.extend(
            [
                f"### `{_escape(entry.entry_id)}` (`{_escape(entry.benchmark_id)}`)",
                "",
            ]
        )
        lines.extend(
            _entry_metric_table(
                entry,
                _primary_metric_names(entry),
                include_notes=True,
            )
        )
        lines.append("")

    lines.extend(["## 关键组件指标", ""])
    for entry in entries:
        component_names = tuple(
            name for name in _component_metric_names(entry) if name in entry.metrics
        )
        if not component_names:
            continue
        lines.extend(
            [
                f"### `{_escape(entry.entry_id)}`",
                "",
                *_entry_metric_table(entry, component_names),
                "",
            ]
        )

    lines.extend(
        [
            "## 运行健康度",
            "",
            "这些指标用于判断评测是否完整、解析是否稳定，不作为模型总体能力分数。",
            "",
        ]
    )
    lines.extend(_input_truncation_table(entries))
    for entry in entries:
        health_names = tuple(name for name in sorted(entry.metrics) if _is_health_metric(name))
        if not health_names:
            continue
        lines.extend(
            [
                f"### `{_escape(entry.entry_id)}`",
                "",
                *_entry_metric_table(entry, health_names),
                "",
            ]
        )

    lines.extend(
        [
            "## 详细诊断附录",
            "",
            "以下是没有进入主看板、关键组件或运行健康度的细粒度指标。它们用于定位 persona、"
            "character、category、domain、feature 等局部差异，默认折叠，不建议逐项当作总体性能。",
            "",
        ]
    )
    for entry in entries:
        selected = set(_primary_metric_names(entry)) | set(_component_metric_names(entry))
        diagnostic_names = tuple(
            name
            for name in sorted(entry.metrics)
            if name not in selected and not _is_health_metric(name)
        )
        if not diagnostic_names:
            continue
        lines.extend(
            [
                "<details>",
                f"<summary><code>{_escape(entry.entry_id)}</code>：{len(diagnostic_names)} 个诊断指标</summary>",
                "",
                *_entry_metric_table(entry, diagnostic_names),
                "",
                "</details>",
                "",
            ]
        )

    lines.extend(
        [
            "## 有效数据口径",
            "",
            "- 分母是该 entry 计划执行的结果记录数：`selected cases × repetitions`；若产物记录更多，则采用实际结果记录数。",
            "- 优先逐条读取 `records.jsonl`，并且每个 `(case_id, repetition)` 只保留最新 checkpoint attempt；同名 metric 的非空值计为有效。",
            "- 值为 `null` 且元数据明确标记 `availability=not_applicable` 的稳定 schema 占位符不计入适用范围；UserLM 旧产物还会按 variant 恢复这一语义。",
            "- `适用范围` 表示该 metric 实际适用多少记录；例如某个 SOTOPIA persona 只出现在 12/100 个 episode 中，这不是失败。",
            "- `有效率` 只在适用数据内部计算。`12/12 (100%)` 表示该 persona 的 12 个适用 episode 都成功产生了该 metric。",
            "- 对只存在于聚合层的宏平均、set/chain、domain、batch 或 population metric，保留 adapter 的自然 aggregate unit，不再用它除以 case 总数制造误导性百分比。",
            "- `unavailable` 的 metric 有效数记为 0。这里不展示正确样本数、得分和或指标内部的计算分母。",
            "- SOTOPIA 的 `.normalized` 是把各维度原始区间线性映射到 `[0,1]`，用于跨维度比较；它不是额外 Judge，也不是另一批样本。",
            "- `.normalized` 使用固定 rubric 区间，因此同一指标在相同数据、prompt、Judge 和聚合协议下可以跨待测模型比较；它不消除更换 Judge 带来的偏差。",
            "",
        ]
    )
    return "\n".join(lines)


def write_metric_markdown(
    *,
    input_path: str | Path,
    output_path: str | Path | None = None,
    corrected_metrics: Mapping[str, str | Path] | None = None,
    replacement_entries: Mapping[str, str | Path] | None = None,
) -> Path:
    root, _summary_path, _summary = _locate_summary(input_path)
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else root / "metric_summary.md"
    )
    atomic_write_text(
        output,
        render_metric_markdown(
            input_path=input_path,
            corrected_metrics=corrected_metrics,
            replacement_entries=replacement_entries,
        ),
    )
    return output


__all__ = [
    "MetricEntry",
    "MetricCoverage",
    "MetricOverrideAudit",
    "apply_metric_overrides",
    "load_metric_entries",
    "metric_coverage",
    "metric_valid_count",
    "render_metric_markdown",
    "write_metric_markdown",
]
