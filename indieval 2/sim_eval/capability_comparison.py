"""Read-only grouping of stored evaluation scores by the user's five dimensions.

No inference, metric recomputation, implicit aliases, or composite scores.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .artifacts import atomic_write_text, load_json
from .errors import ArtifactError
from .metric_markdown import (
    MetricEntry, _entry_from_child, _locate_summary, _records_path,
    _resolve_inside, _record_metric_attributes, _record_metric_is_applicable,
    load_metric_entries, metric_coverage, SUMMARY_FILENAMES,
)

DEFAULT_MAPPING = Path(__file__).resolve().parent / "resources/reporting/capability_dimensions_v1.json"
DIRECTIONS = {"higher_is_better": "↑", "lower_is_better": "↓", "closer_to_zero": "→0", "descriptive": "描述性"}
IDENTITY_FIELDS = ("framework_version", "prompt_revision", "scorer_revision", "seed",
                   "source_revision", "split", "judge", "assistant_or_partner", "environment")


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_dimension_mapping(path: str | Path = DEFAULT_MAPPING) -> dict[str, Any]:
    value = load_json(Path(path))
    if not isinstance(value, dict) or value.get("aggregation") != "group_only_no_composite_score":
        raise ArtifactError("dimension mapping requires aggregation=group_only_no_composite_score")
    dimensions = value.get("dimensions", [])
    if [d.get("id") for d in dimensions] != list("FSUTN"):
        raise ArtifactError("dimension mapping must declare F, S, U, T, N in that order")
    seen = set()
    summary_groups: dict[tuple[str, str], int] = {}
    if not value.get("metrics"):
        raise ArtifactError("dimension mapping has no metrics")
    for row in value["metrics"]:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k].strip()
                                                for k in ("dimension", "entry", "metric", "label")):
            raise ArtifactError("each mapping row needs dimension, entry, metric, label")
        key = row["dimension"], row["entry"], row["metric"]
        if key[0] not in set("FSUTN") or key in seen:
            raise ArtifactError(f"unknown dimension or duplicate row: {key}")
        seen.add(key)
        dataset = row.get("dataset")
        if not isinstance(dataset, str) or not dataset.strip() or not isinstance(row.get("summary"), bool):
            raise ArtifactError("each mapping row needs dataset and a boolean summary selection")
        group = row["dimension"], dataset
        summary_groups[group] = summary_groups.get(group, 0) + int(row["summary"])
        if row["summary"] and not row.get("summary_reason"):
            raise ArtifactError("representative metrics require an explicit summary_reason")
    if any(count < 1 for count in summary_groups.values()):
        raise ArtifactError("summary must select at least one metric per dimension/dataset")
    return value


@dataclass
class Run:
    root: Path
    summary_path: Path
    summary: Mapping[str, Any]
    entries: dict[str, MetricEntry]
    plan: Mapping[str, Any]
    identities: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source_audit: dict[str, Any] = field(default_factory=dict)


def _load_run(path: str | Path, *, allow_missing_summary: bool = False) -> Run:
    directory = Path(path).expanduser().resolve()
    snapshot = (allow_missing_summary and directory.is_dir()
                and not any((directory / name).exists() for name in SUMMARY_FILENAMES))
    source_audit = {}
    saved_plan = None
    if snapshot:
        root = directory
        plan_path = next((root / name for name in ("suite_plan.json", "smoke_plan.json")
                          if (root / name).is_file()), None)
        if plan_path is None:
            raise ArtifactError(f"cannot compare child results without a suite/smoke plan: {root}")
        saved_plan = load_json(plan_path)
        planned = _map(saved_plan).get("entries")
        if not isinstance(planned, list) or not planned:
            raise ArtifactError(f"plan must contain nonempty entries: {plan_path}")
        discovered = []
        for raw in planned:
            entry_id = _map(raw).get("id")
            if not isinstance(entry_id, str) or not entry_id or Path(entry_id).name != entry_id:
                raise ArtifactError(f"invalid entry ID in {plan_path}")
            relative = f"entries/{entry_id}/suite_summary.json"
            _resolve_inside(root, relative, "planned child summary")
            discovered.append({**raw, "child_summary": relative})
        summary_path = plan_path  # Real source path; provenance below identifies it as a plan.
        summary = {"status": "incomplete_snapshot", "entries": discovered}
        source_audit = {"summary": None, "summary_sha256": None,
                        "source_kind": "planned_child_summaries_snapshot",
                        "plan": str(plan_path), "plan_content_sha256": _digest(saved_plan),
                        "child_summary_content_sha256": {}}
    else:
        root, summary_path, summary = _locate_summary(path)
    entries: dict[str, MetricEntry] = {}
    locations: dict[str, tuple[Path, Mapping[str, Any], Mapping[str, Any]]] = {}
    warnings = (["根汇总缺失：直接比较计划内已保存的子结果；这是读取时的快照，不代表整套评测已结束。"]
                if snapshot else [])
    raw_entries = summary.get("entries")
    if isinstance(raw_entries, list):
        # A failed benchmark can legitimately have no child summary. Keep its
        # slot missing instead of aborting the other thirteen comparisons.
        for raw in raw_entries:
            entry_id = str(raw.get("id") or raw.get("benchmark_id") or "")
            if not entry_id or entry_id in entries:
                raise ArtifactError(f"missing or duplicate entry ID in {summary_path}")
            relative = raw.get("child_summary")
            child_path = _resolve_inside(root, relative, "child summary") if relative else None
            if child_path is None or not child_path.is_file():
                warnings.append(f"{entry_id}: 缺少 child summary，保留缺失，不重算")
                repetitions = (_map(saved_plan.get("execution")).get("userlm_lic_repetitions", 1)
                               if snapshot and entry_id == "userlm_lic" and raw.get("benchmark_id") == "userlm"
                               else 1)
                if type(repetitions) is not int or repetitions < 1:
                    raise ArtifactError("invalid planned UserLM repetition count")
                entries[entry_id] = MetricEntry(entry_id, str(raw.get("benchmark_id") or entry_id),
                    str(raw.get("status") or "missing_summary"), int(raw.get("selected_case_count") or 0),
                    repetitions, 0, 0, 0, {}, ())
                continue
            child = load_json(child_path)
            if snapshot:
                if not isinstance(child, Mapping) or child.get("benchmark_id") != raw.get("benchmark_id"):
                    raise ArtifactError(f"child summary benchmark disagrees with plan: {child_path}")
                source_audit["child_summary_content_sha256"][entry_id] = _digest(child)
            entries[entry_id] = _entry_from_child(campaign_root=root, entry_id=entry_id,
                root_entry=raw, child_root=child_path.parent, child_summary=child)
            locations[entry_id] = (child_path.parent, raw, child)
    else:
        _, _, _, loaded = load_metric_entries(path)
        entries = {entry.entry_id: entry for entry in loaded}
        if len(entries) != len(loaded):
            raise ArtifactError("duplicate entry IDs")
        for entry in loaded:
            child = _map(summary.get("benchmarks")).get(entry.entry_id, summary)
            locations[entry.entry_id] = (root, child, child)
    identities = {}
    for entry_id, (child_root, raw, child) in locations.items():
        records_path = _records_path(campaign_root=root, root_entry=raw, child_root=child_root, child_summary=child)
        if records_path is not None:
            manifest_path = records_path.parent / "run_manifest.json"
            if manifest_path.is_file():
                identity = _map(load_json(manifest_path).get("identity"))
                identities[entry_id] = {k: identity[k] for k in IDENTITY_FIELDS if k in identity}
    plan = saved_plan or {}
    if saved_plan is None:
        for filename in ("suite_plan.json", "smoke_plan.json"):
            if (root / filename).is_file():
                plan = load_json(root / filename)
                break
    return Run(root, summary_path, summary, entries, plan, identities, warnings, source_audit)


def _key(record: Mapping[str, Any]) -> tuple[str, int]:
    return str(record.get("case_id")), int(record.get("repetition", 0))


def _record_index(entry: MetricEntry) -> dict[str, tuple[set, set]]:
    counts: dict[str, tuple[set, set]] = {}
    for record in entry.records:
        for metric in record.get("metrics", []):
            name = metric.get("name")
            if not name or not _record_metric_is_applicable(record, name, metric):
                continue
            applicable, valid = counts.setdefault(name, (set(), set()))
            applicable.add(_key(record))
            if _number(metric.get("value")) is not None:
                valid.add(_key(record))
    return counts


def _entry_counts(entry: MetricEntry | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {"status": entry.status, "selected_cases": entry.selected_case_count,
            "repetitions_per_case": entry.repetitions_per_case, "expected_records": entry.expected_record_count,
            "reported_records": entry.result_count, "observed_latest_records": len(entry.records),
            "completed": entry.completed_count, "failed": entry.failed_count}


def _entry_audit(entry_id: str, a: Run, b: Run) -> dict[str, Any]:
    ea, eb = a.entries.get(entry_id), b.entries.get(entry_id)
    notes = []
    ka = {_key(r) for r in ea.records} if ea else set()
    kb = {_key(r) for r in eb.records} if eb else set()
    if ea is None or eb is None:
        population = "missing_entry"
        notes.append("一方或双方缺少整个 entry")
    elif ka and kb:
        population = "same_record_keys" if ka == kb else "different_record_keys"
        if ka != kb:
            notes.append("结果样本键集合不同；未按交集重算")
    else:
        population = "unknown_record_keys"
        notes.append("缺少 records，不能仅凭数量确认同一批样本")
    if ea and eb and ea.benchmark_id != eb.benchmark_id:
        notes.append("benchmark_id 不一致")
    for label, entry in (("A", ea), ("B", eb)):
        if entry and entry.result_count < entry.expected_record_count:
            notes.append(f"{label} 记录未齐")
        if entry and entry.records and len(entry.records) != entry.result_count:
            notes.append(f"{label} summary 与最新 records 数量不同，汇总可能过期")
        if entry and entry.failed_count:
            notes.append(f"{label} 有 {entry.failed_count} 个失败记录")
    mismatches = []
    ia, ib = a.identities.get(entry_id, {}), b.identities.get(entry_id, {})
    for key in sorted(set(ia) & set(ib)):
        if ia[key] != ib[key]:
            mismatches.append("run_identity." + key)
    for key in ("framework_version", "seed", "global_eval_model"):
        if key in a.plan and key in b.plan and a.plan[key] != b.plan[key]:
            mismatches.append("plan." + key)
    pa = {e["id"]: e for e in a.plan.get("entries", [])}.get(entry_id, {})
    pb = {e["id"]: e for e in b.plan.get("entries", [])}.get(entry_id, {})
    for key in ("source_manifest_digest", "selected_case_ids"):
        if key in pa and key in pb:
            va, vb = pa[key], pb[key]
            if key == "selected_case_ids":
                va, vb = sorted(va), sorted(vb)
            if va != vb:
                mismatches.append("plan_entry." + key)
    if mismatches:
        notes.append("配置/来源差异: " + ", ".join(mismatches))
    if not ia or not ib:
        notes.append("run identity 信息不齐，prompt/Judge 等可比性未完全核验")
    return {"entry": entry_id, "baseline": _entry_counts(ea), "candidate": _entry_counts(eb),
            "population": population, "record_key_overlap": len(ka & kb),
            "identity_mismatches": mismatches, "notes": notes}


def _cell(entry: MetricEntry | None, name: str, index: dict, attributes: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"value": None, "status": "missing_entry", "unit": None,
                              "direction": None, "coverage": None, "reason": "缺少 entry"}
    if entry is None:
        return result
    metric = entry.metrics.get(name)
    if metric is None:
        result.update(status="missing_metric", reason="原汇总未输出此指标；不使用近似指标替代")
        return result
    metadata = _map(metric.get("metadata"))
    value = _number(metric.get("value"))
    not_applicable = metadata.get("availability") == "not_applicable"
    status = "available" if value is not None else "unavailable"
    if not_applicable:
        value, status = None, "not_applicable"
    elif metric.get("value") is not None and value is None:
        status = "invalid_value"
    elif metadata.get("availability") == "unavailable":
        value, status = None, "unavailable"
    coverage = metric_coverage(name, metric,
        record_counts={k: (len(v[0]), len(v[1])) for k, v in index.items()},
        total_records=entry.expected_record_count, completed_records=entry.completed_count)
    basis = coverage.basis
    natural_unit = metadata.get("independent_statistical_unit")
    coverage_data = {"valid": coverage.valid_count, "applicable": coverage.applicable_count}
    if "chain_count" in metadata:
        natural_unit = "chain"
    elif "configuration_count" in metadata:
        natural_unit = "configuration"
    elif metadata.get("aggregation") == "equal_weight_mean_of_code_and_math_task_means":
        natural_unit = "domain"
        # A missing required domain must not look like a complete one-domain macro.
        applicable, valid = len(metadata.get("required_domains", ["code", "math"])), len(metadata.get("available_domains", []))
        basis = "aggregate_units"
        coverage_data = {"valid": valid, "applicable": applicable}
    if name in {"fantom.all", "fantom.all_star"}:
        natural_unit, basis = "set", "aggregate_units"
        # The headline is an exact alias of this saved context/scenario metric.
        # Its denominator already includes incomplete/unavailable sets. Do not
        # add them twice or silently reinterpret a fail-closed score as zero.
        source_name = f"fantom.{metadata.get('context_condition')}.inaccessible.{name.rsplit('.', 1)[-1]}"
        source = entry.metrics.get(source_name, {})
        source_metadata = _map(source.get("metadata"))
        denominator = _number(source.get("denominator"))
        if denominator is not None and denominator >= 0 and denominator.is_integer():
            excluded = int(source_metadata.get("incomplete_group_count", 0)) + int(source_metadata.get("evaluator_unavailable_group_count", 0))
            coverage_data = {"valid": max(0, int(denominator) - excluded), "applicable": int(denominator)}
    if not_applicable:
        coverage_data = {"valid": 0, "applicable": 0}
    coverage_data.update(basis=basis, natural_unit=natural_unit,
                         expected_records=entry.expected_record_count)
    # Coverage describes supporting units; the explicit status says whether the
    # final aggregate exists. E.g. 1/2 available domains still yields no macro.
    result.update(value=value, status=status, unit=metric.get("unit") or attributes.get("unit"),
                  direction=metric.get("direction") or attributes.get("direction"),
                  coverage=coverage_data, reason=str(metadata.get("reason") or metadata.get("unavailable_reason") or ""),
                  aggregation=metadata.get("aggregation"))
    return result


def build_capability_comparison(*, baseline_path: str | Path, candidate_path: str | Path,
        baseline_label: str = "Baseline", candidate_label: str = "Candidate",
        mapping_path: str | Path = DEFAULT_MAPPING, allow_missing_summary: bool = False) -> dict[str, Any]:
    mapping = load_dimension_mapping(mapping_path)
    if not baseline_label.strip() or not candidate_label.strip() or baseline_label == candidate_label:
        raise ArtifactError("model labels must be nonempty and distinct")
    a, b = (_load_run(path, allow_missing_summary=allow_missing_summary)
            for path in (baseline_path, candidate_path))
    entry_ids = sorted(set(a.entries) | set(b.entries) | {r["entry"] for r in mapping["metrics"]})
    audits = [_entry_audit(e, a, b) for e in entry_ids]
    by_audit = {r["entry"]: r for r in audits}
    indexes = [{k: _record_index(v) for k, v in run.entries.items()} for run in (a, b)]
    attributes = [{k: _record_metric_attributes(v.records) for k, v in run.entries.items()} for run in (a, b)]
    rows = []
    for spec in mapping["metrics"]:
        entry_id, name = spec["entry"], spec["metric"]
        ca = _cell(a.entries.get(entry_id), name, indexes[0].get(entry_id, {}), attributes[0].get(entry_id, {}).get(name, {}))
        cb = _cell(b.entries.get(entry_id), name, indexes[1].get(entry_id, {}), attributes[1].get(entry_id, {}).get(name, {}))
        flags = list(by_audit[entry_id]["notes"])
        delta = None
        if ca["value"] is not None and cb["value"] is not None:
            if ca["unit"] != cb["unit"]:
                flags.append("单位不同，不计算差值")
            elif ca["direction"] != cb["direction"]:
                flags.append("方向不同，不计算差值")
            elif a.entries[entry_id].benchmark_id != b.entries[entry_id].benchmark_id:
                flags.append("benchmark 不同，不计算差值")
            else:
                delta = cb["value"] - ca["value"]
        for label, cell in (("A", ca), ("B", cb)):
            if cell["status"] != "available":
                flags.append(f"{label}: {cell['status']}")
            cov = cell["coverage"]
            if cov and cov["valid"] < cov["applicable"]:
                flags.append(f"{label} 指标覆盖不完整")
        ai = indexes[0].get(entry_id, {}).get(name)
        bi = indexes[1].get(entry_id, {}).get(name)
        if ai is not None and bi is not None and ai[1] != bi[1]:
            flags.append("指标有效样本集合不同（即使有效数相同也可能不同）")
        rows.append({**spec, "baseline": ca, "candidate": cb, "delta": delta, "flags": flags})
    dimensions = []
    for dimension in mapping["dimensions"]:
        members = [r for r in rows if r["dimension"] == dimension["id"]]
        dimensions.append({**dimension, "metric_count": len(members),
            "baseline_available": sum(r["baseline"]["value"] is not None for r in members),
            "candidate_available": sum(r["candidate"]["value"] is not None for r in members)})
    mapped = {(r["entry"], r["metric"]) for r in rows}
    unmapped = {entry_id: sorted((set(a.entries[entry_id].metrics) if entry_id in a.entries else set()) |
                                  (set(b.entries[entry_id].metrics) if entry_id in b.entries else set()))
                for entry_id in entry_ids}
    unmapped = {e: [n for n in names if (e, n) not in mapped] for e, names in unmapped.items()}
    return {"schema_version": "1.0", "mapping_revision": mapping["mapping_revision"],
            "summary_revision": mapping.get("summary_revision"),
            "mapping_sha256": _digest(mapping), "aggregation": mapping["aggregation"],
            "baseline": {"label": baseline_label, "root": str(a.root), "summary": str(a.summary_path),
                         "summary_sha256": hashlib.sha256(a.summary_path.read_bytes()).hexdigest(), **a.source_audit},
            "candidate": {"label": candidate_label, "root": str(b.root), "summary": str(b.summary_path),
                          "summary_sha256": hashlib.sha256(b.summary_path.read_bytes()).hexdigest(), **b.source_audit},
            "dimensions": dimensions, "entries": audits, "rows": rows,
            "summary_rows": [row for row in rows if row["summary"]], "unmapped_metrics": unmapped,
            "warnings": [*("A: " + w for w in a.warnings), *("B: " + w for w in b.warnings)]}


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("`", "'")


def _format(value: Any, digits: int) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if value else "0"


def _coverage(cell: Mapping[str, Any]) -> str:
    cov = cell.get("coverage")
    if cov is None:
        return "—"
    if cov["applicable"] == 0:
        return "N/A (0 适用)"
    unit = cov.get("natural_unit") or {"record_metric": "记录", "record_population": "记录", "aggregate_units": "聚合单位"}.get(cov["basis"], cov["basis"])
    return f"{cov['valid']}/{cov['applicable']} {unit}"


def render_capability_markdown(report: Mapping[str, Any], *, digits: int = 4) -> str:
    if not 1 <= digits <= 12:
        raise ValueError("digits must be between 1 and 12")
    a, b = report["baseline"], report["candidate"]
    lines = ["# 五个能力维度的评测对比", "", f"- A：{_escape(a['label'])}；`{_escape(a['root'])}`",
             f"- B：{_escape(b['label'])}；`{_escape(b['root'])}`",
             f"- 指标映射：`{report['mapping_revision']}`。差值均为 B − A，保留原单位。",
             "- 只分类展示已有聚合分；不重新评分、不调用模型、不生成维度平均或总排名。",
             "- 缺失不补零；0 是有效分。↑ 越高越好；↓ 越低越好；→0 越接近 0 越好（不能只看差值正负）。",
             "- 有效/适用计数保留原自然单位；两个模型的有效样本可能不同。差值不是交集配对重算，也不是显著性检验。",
             "- 运行中的子汇总可能落后于 records；缺根汇总模式仅展示已落盘结果，不代表全部完成。配置差异及缺失信息见后面的核查表。", "",
             "## 维度与指标可用情况", "", "这里的数量是可用指标数，不是能力分。", "",
             "| 维度 | 核心问题 | A 可用/配置指标 | B 可用/配置指标 |", "|---|---|---:|---:|"]
    for d in report["dimensions"]:
        lines.append(f"| {d['id']}. {_escape(d['name_zh'])} | {_escape(d['question'])} | {d['baseline_available']}/{d['metric_count']} | {d['candidate_available']}/{d['metric_count']} |")
    for d in report["dimensions"]:
        lines += ["", f"## {d['id']}. {d['name_en']} / {d['name_zh']}", "", d["question"], "",
                  "| 指标 | A | B | B − A | 单位 · 方向 | A 有效/适用 | B 有效/适用 | 状态/说明 |",
                  "|---|---:|---:|---:|---|---|---|---|"]
        for r in report["rows"]:
            if r["dimension"] != d["id"]:
                continue
            ca, cb = r["baseline"], r["candidate"]
            unit = ca["unit"] or cb["unit"] or "未标注"
            direction = ca["direction"] or cb["direction"]
            if ca["unit"] and cb["unit"] and ca["unit"] != cb["unit"]:
                unit = f"A:{ca['unit']}; B:{cb['unit']}"
            detail = "; ".join(dict.fromkeys([r.get("note", ""), *r["flags"], ca.get("reason", ""), cb.get("reason", "")]))
            lines.append(f"| {_escape(r['label'])}<br>`{r['metric']}` | {_format(ca['value'], digits)} | {_format(cb['value'], digits)} | {_format(r['delta'], digits)} | {_escape(unit)} · {DIRECTIONS.get(direction, '未知')} | {_coverage(ca)} | {_coverage(cb)} | {_escape(detail.strip('; '))} |")
    lines += ["", "## 样本、运行与可比性核查", "",
              "A/B 列为：最新记录数 / 计划记录数；完成数；失败数；运行状态。", "",
              "| Entry | A | B | 样本键核查 | 说明 |", "|---|---|---|---|---|"]
    for audit in report["entries"]:
        def counts(c):
            return "缺少 entry" if c is None else f"{c['observed_latest_records']}/{c['expected_records']}; 完成 {c['completed']}; 失败 {c['failed']}; {c['status']}"
        lines.append(f"| `{audit['entry']}` | {_escape(counts(audit['baseline']))} | {_escape(counts(audit['candidate']))} | {audit['population']}; 交集 {audit['record_key_overlap']} | {_escape('; '.join(audit['notes']) or '已检查字段未发现差异')} |")
    if report["warnings"]:
        lines += ["", "### 读取提示", "", *("- " + _escape(w) for w in report["warnings"])]
    lines += ["", "## 未纳入五维主表的原始指标", "",
              "这里列出未映射的指标名，便于发现新增指标；包括健康度、细分诊断和重复量纲，不自动猜测分类。完整名单也保存在 JSON 中。", ""]
    for entry, names in report["unmapped_metrics"].items():
        if names:
            lines += ["<details>", f"<summary>{entry}：{len(names)} 个未映射指标</summary>", "",
                      *(f"- `{_escape(n)}`" for n in names), "", "</details>", ""]
    return "\n".join(lines) + "\n"


def render_capability_summary_markdown(report: Mapping[str, Any], *, digits: int = 4) -> str:
    """One table, one preselected metric per dataset within each dimension."""
    if not 1 <= digits <= 12:
        raise ValueError("digits must be between 1 and 12")
    a, b = report["baseline"], report["candidate"]
    dimensions = {d["id"]: d["name_zh"] for d in report["dimensions"]}
    lines = ["# 五维能力对比：主指标大表", "",
             f"A：{_escape(a['label'])}；B：{_escape(b['label'])}。差值为 B − A，保留原单位。", "",
             f"按五个能力维度展开，保留各数据集选定的重要指标，共 {len(report['summary_rows'])} 行。选择版本：`{report['summary_revision']}`。",
             "缺失保持 —，不根据结果临时换指标，不计算维度总分。UserLM 的协议在数据集列中单独注明。", "",
             "| 能力 | 数据集 | 主指标 | A | B | B − A | 单位 · 方向 | A 有效/适用 | B 有效/适用 | 核查 |",
             "|---|---|---|---:|---:|---:|---|---|---|---|"]
    if any(run.get("source_kind") == "planned_child_summaries_snapshot" for run in (a, b)):
        lines[4:4] = ["**本表含缺少根汇总的运行：已直接读取子结果，未完成或缺失指标保留 —；不表示整套评测完成。**", ""]
    for row in report["summary_rows"]:
        ca, cb = row["baseline"], row["candidate"]
        unit = ca["unit"] or cb["unit"] or "未标注"
        if ca["unit"] and cb["unit"] and ca["unit"] != cb["unit"]:
            unit = f"A:{ca['unit']}; B:{cb['unit']}"
        direction = ca["direction"] or cb["direction"]
        dataset = row["dataset"]
        if row["entry"] in {"userlm_lic", "userlm_section3"}:
            dataset += " (LiC)" if row["entry"] == "userlm_lic" else " (Section 3)"
        label = row["label"].split(" · ", 1)[-1]
        check = "†" if row["flags"] else "—"
        lines.append(f"| {row['dimension']}. {_escape(dimensions[row['dimension']])} | {_escape(dataset)} | {_escape(label)} | {_format(ca['value'], digits)} | {_format(cb['value'], digits)} | {_format(row['delta'], digits)} | {_escape(unit)} · {DIRECTIONS.get(direction, '未知')} | {_coverage(ca)} | {_coverage(cb)} | {check} |")
    lines += ["", "† 表示该行有缺失、覆盖不完整、样本集合差异或实验身份未完全核验等提示；具体原因保留在精简 CSV、JSON 和[详细报告](capability_comparison.md)中。差值不是共同样本上的配对重算，也不是显著性结论。",
              "", "计数保留 chain/set/configuration/task/domain 等自然单位；有部分底层数据不代表整体分数可用。↑ 越高越好；↓ 越低越好；→0 越接近 0 越好。",
              "", "代表指标是阅读摘要，不覆盖该数据集在该能力下的所有方面。例如 UserLM Termination F1 仅覆盖带标签的终止任务，AI-human likelihood 是检测器代理指标。其余指标与选择依据见详细报告/JSON。", ""]
    return "\n".join(lines)


def _render_csv(report: Mapping[str, Any], rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    fields = ["dimension", "dataset", "entry", "metric", "label", "summary_reason", "baseline_label", "candidate_label", "baseline_value", "candidate_value", "delta",
              "baseline_unit", "candidate_unit", "baseline_direction", "candidate_direction", "baseline_status", "candidate_status",
              "baseline_valid", "baseline_applicable", "baseline_coverage_basis", "baseline_natural_unit",
              "candidate_valid", "candidate_applicable", "candidate_coverage_basis", "candidate_natural_unit", "note", "flags"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        flat = {k: row.get(k, "") for k in ("dimension", "dataset", "entry", "metric", "label", "summary_reason", "delta", "note")}
        flat["flags"] = "; ".join(row["flags"])
        for side in ("baseline", "candidate"):
            cell, cov = row[side], row[side]["coverage"] or {}
            flat[side + "_label"] = report[side]["label"]
            for key in ("value", "unit", "direction", "status"):
                flat[side + "_" + key] = cell[key]
            for key in ("valid", "applicable", "natural_unit"):
                flat[side + "_" + key] = cov.get(key)
            flat[side + "_coverage_basis"] = cov.get("basis")
        writer.writerow(flat)
    return "\ufeff" + stream.getvalue()


def write_capability_comparison(*, output_dir: str | Path, digits: int = 4, **kwargs) -> dict[str, Path]:
    report = build_capability_comparison(**kwargs)
    markdown = render_capability_markdown(report, digits=digits)
    summary = render_capability_summary_markdown(report, digits=digits)
    destination = Path(output_dir).expanduser().resolve()
    outputs = {extension: destination / ("capability_comparison." + extension) for extension in ("md", "csv", "json")}
    outputs.update(summary_md=destination / "capability_summary.md", summary_csv=destination / "capability_summary.csv")
    atomic_write_text(outputs["md"], markdown)
    atomic_write_text(outputs["csv"], _render_csv(report, report["rows"]))
    atomic_write_text(outputs["json"], json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_write_text(outputs["summary_md"], summary)
    atomic_write_text(outputs["summary_csv"], _render_csv(report, report["summary_rows"]))
    return outputs
