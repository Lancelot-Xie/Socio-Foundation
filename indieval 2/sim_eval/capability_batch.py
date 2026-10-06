"""Collect available headline metrics and execution counts from formal runs, offline."""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_text, load_json
from .capability_comparison import (
    DIRECTIONS, IDENTITY_FIELDS, Run, _cell, _digest, _escape, _format, _key, _record_index,
)
from .errors import ArtifactError, SimEvalError
from .json_utils import canonical_json
from .metric_markdown import _entry_from_child, _record_metric_attributes, _records_path, _resolve_inside

DEFAULT_MAPPING = Path(__file__).resolve().parent / "resources/reporting/capability_main_metrics_v1.json"
FULL_ENTRIES = frozenset((
    "sotopia", "coser", "tau_usi", "lifechoices", "mirrorbench", "agentsense",
    "userlm_lic", "social_r1", "behaviorchain", "alignx", "humanllm",
    "fantom", "humanual", "userlm_section3",
))


def load_mapping(path: str | Path, selection: str) -> dict:
    mapping = load_json(Path(path))
    if mapping.get("aggregation") != "group_only_no_composite_score":
        raise ArtifactError("mapping must group metrics without a composite score")
    if [d["id"] for d in mapping["dimensions"]] != list("FSUTN"):
        raise ArtifactError("mapping dimensions must be F/S/U/T/N")
    if selection not in {"by-type", "one-per-dataset"}:
        raise ArtifactError("unknown metric selection")
    rows = mapping["metrics"]
    seen = set()
    representatives = {}
    for row in rows:
        for field in ("dimension", "dataset", "entry", "metric", "label", "metric_type"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ArtifactError(f"mapping row requires {field}")
        if row["dimension"] not in set("FSUTN") or row["entry"] not in FULL_ENTRIES:
            raise ArtifactError("mapping contains an unknown dimension or entry")
        key = row["dimension"], row["dataset"], row["metric_type"]
        if key in seen:
            raise ArtifactError(f"duplicate metric type: {key}")
        seen.add(key)
        group = key[:2]
        representatives[group] = representatives.get(group, 0) + int(row.get("representative") is True)
    if any(n != 1 for n in representatives.values()):
        raise ArtifactError("mapping requires one fixed representative per dimension/dataset")
    return {**mapping, "metrics": rows if selection == "by-type" else [r for r in rows if r["representative"]]}


def _count(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _read_object(path: Path, warnings: list[str]) -> dict:
    if not path.is_file():
        return {}
    try:
        value = load_json(path)
        if not isinstance(value, dict):
            raise ArtifactError("JSON 根节点不是对象")
        return value
    except (SimEvalError, OSError, ValueError) as exc:
        warnings.append(f"{path.name}: {exc}")
        return {}


def _load_available_run(path: Path) -> tuple[Run, dict]:
    warnings = []
    summary_path = path / "suite_summary.json"
    summary = _read_object(summary_path, warnings)
    plan = _read_object(path / "suite_plan.json", warnings)
    # Also collect completed child summaries when a suite was interrupted before
    # writing its root summary. Never pick arbitrary old run directories.
    raw_entries = {r["id"]: r for r in summary.get("entries", []) if isinstance(r, dict) and r.get("id")}
    planned = {r["id"]: r for r in plan.get("entries", []) if isinstance(r, dict) and r.get("id")}
    discovered = {p.parent.name for p in (path / "entries").glob("*/suite_summary.json")}
    entry_ids = set(raw_entries) | set(planned) | discovered
    if not entry_ids:
        raise ArtifactError("没有可读取的 suite/子入口汇总")
    if not summary:
        warnings.append("缺少有效的根 suite_summary.json；读取现存子入口汇总")
    entries, identities, health = {}, {}, {}
    for key in sorted(entry_ids):
        raw, selected = raw_entries.get(key, {}), planned.get(key, {})
        issues = []
        try:
            child_path = _resolve_inside(path, raw.get("child_summary") or f"entries/{key}/suite_summary.json", "child summary")
            child = _read_object(child_path, issues)
        except (SimEvalError, TypeError, ValueError) as exc:
            child_path, child = path / "entries" / key / "suite_summary.json", {}
            issues.append(str(exc))
        selected_count = _count(child.get("selected_case_count", selected.get("selected_case_count", raw.get("selected_case_count"))))
        repeats = _count(child.get("repetitions_per_case"))
        if repeats is None:
            repeats = (plan.get("execution", {}).get("userlm_lic_repetitions", 10)
                       if key == "userlm_lic" else 1)
        expected = selected_count * repeats if selected_count is not None else None
        h = {"status": child.get("status", raw.get("status", "missing_summary")),
             "expected_count": expected, "result_count": _count(child.get("result_count")),
             "completed_count": _count(child.get("completed_count", raw.get("completed_count"))),
             "failed_count": _count(child.get("failed_count", raw.get("failed_count"))),
             "pending_judge_count": _count(child.get("pending_judge_count", raw.get("pending_judge_count"))),
             "summary_completed_count": _count(child.get("completed_count", raw.get("completed_count"))),
             "summary_failed_count": _count(child.get("failed_count", raw.get("failed_count"))),
             "records_available": False, "count_source": "summary", "missing_count": None,
             "child_summary": str(child_path), "issues": issues}
        if child:
            normalized = {**raw, "status": child.get("status", raw.get("status"))}
            records_path = None
            try:
                records_path = _records_path(campaign_root=path, root_entry=normalized,
                    child_root=child_path.parent, child_summary=child)
                entry = _entry_from_child(campaign_root=path, entry_id=key,
                    root_entry=normalized, child_root=child_path.parent, child_summary=child)
                h["records_available"] = records_path is not None and records_path.is_file()
            except (SimEvalError, OSError, ValueError, TypeError, KeyError) as exc:
                # Missing/corrupt records cannot erase readable aggregate metrics.
                issues.append(f"records/核查数据不可用，仅保留 summary 指标：{exc}")
                clean = {k: v for k, v in child.items() if k not in {"artifacts", "artifact_paths"}}
                clean_raw = {k: v for k, v in normalized.items() if k != "records"}
                try:
                    entry = _entry_from_child(campaign_root=path, entry_id=key,
                        root_entry=clean_raw, child_root=child_path.parent, child_summary=clean)
                except (SimEvalError, OSError, ValueError, TypeError, KeyError) as nested:
                    entry = None
                    issues.append(f"子入口汇总无法读取：{nested}")
            if entry is not None:
                entries[key] = entry
                if h["records_available"]:
                    observed = {_key(r) for r in entry.records}
                    h.update(count_source="latest_records", result_count=len(entry.records),
                             completed_count=sum(r.get("status") == "completed" for r in entry.records),
                             failed_count=sum(r.get("status") == "failed" for r in entry.records))
                    if (h["completed_count"] != h["summary_completed_count"]
                            or h["failed_count"] != h["summary_failed_count"]
                            or len(entry.records) != _count(child.get("result_count"))):
                        issues.append("summary 与最新 records 计数不一致；指标仍为保存的 summary 值")
                    case_ids = selected.get("selected_case_ids")
                    if case_ids is not None:
                        expected_keys = {(case_id, rep) for case_id in case_ids for rep in range(repeats)}
                        h["missing_count"] = len(expected_keys - observed)
                        if observed - expected_keys:
                            issues.append("records 存在计划外样本键")
                    if records_path is not None:
                        manifest = _read_object(records_path.parent / "run_manifest.json", issues)
                        identity = manifest.get("identity", {})
                        identities[key] = {k: identity[k] for k in IDENTITY_FIELDS if k in identity}
                else:
                    issues.append("缺少可核查的 records；完成/失败数采用 summary，覆盖核查受限")
        else:
            issues.append("没有有效子 summary，指标保留缺失")
        if h["missing_count"] is None and expected is not None and h["result_count"] is not None:
            h["missing_count"] = max(0, expected - h["result_count"])
        if h["status"] != "completed" or (h["failed_count"] or 0) > 0 or (h["pending_judge_count"] or 0) > 0 or (h["missing_count"] or 0) > 0:
            issues.append("入口存在失败、未完成记录或待补评分")
        health[key] = h
        warnings.extend(f"{key}: {issue}" for issue in issues)
    return Run(path, summary_path, summary, entries, plan, identities, warnings), health


def _totals(health: dict) -> dict:
    result = {}
    for field in ("expected_count", "result_count", "completed_count", "failed_count", "pending_judge_count", "missing_count"):
        values = [h[field] for h in health.values() if h[field] is not None]
        result[field] = sum(values) if values else None
        result[field + "_known_entries"] = len(values)
    return result


def _thinking(run) -> str:
    # Saved per-attempt provenance takes precedence over intended plan settings.
    observed = set()
    for entry in run.entries.values():
        for record in entry.records:
            role = record.get("metadata", {}).get("execution_provenance", {}).get("evaluated_role_identity")
            if isinstance(role, dict):
                flag = role.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking")
                observed.add("关闭" if flag is False else "开启" if flag is True else "未显式设置")
    if observed:
        return "/".join(sorted(observed)) + "（records）"
    if not run.plan.get("model"):
        return "未知（无配置证据）"
    flag = run.plan.get("model", {}).get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking")
    return ("关闭" if flag is False else "开启" if flag is True else "未显式设置") + "（仅 plan）"


def _collect_run(path: Path, specs: list[dict]) -> dict:
    run, health = _load_available_run(path)
    cells = []
    indexes = {k: _record_index(e) for k, e in run.entries.items()}
    attributes = {k: _record_metric_attributes(e.records) for k, e in run.entries.items()}
    for spec in specs:
        key, name = spec["entry"], spec["metric"]
        cell = _cell(run.entries.get(key), name, indexes.get(key, {}), attributes.get(key, {}).get(name, {}))
        flags = []
        if cell["status"] not in {"available", "not_applicable"}:
            flags.append(cell["status"])
        coverage = cell["coverage"]
        if coverage and coverage["valid"] < coverage["applicable"]:
            flags.append("指标覆盖不完整")
        if key in health:
            flags.extend(health[key]["issues"])
        cells.append({**cell, "flags": flags})
    if not any(c["value"] is not None for c in cells):
        raise ArtifactError("没有可用的所选主指标（不从 records 重算缺失聚合分）")
    entries = {}
    planned = {e["id"]: e for e in run.plan.get("entries", [])}
    for key in sorted(FULL_ENTRIES | set(run.entries)):
        entries[key] = {
            "sample_sha256": (_digest(sorted(_key(r) for r in run.entries[key].records))
                              if health.get(key, {}).get("records_available") else None),
            "source_manifest_digest": planned.get(key, {}).get("source_manifest_digest"),
            "protocol_identity": run.identities.get(key),
        }
    return {
        "directory": path.name, "root": str(run.root),
        "model": run.plan.get("model", {}).get("model", path.name),
        "model_revision": run.plan.get("model", {}).get("model_revision"),
        "thinking": _thinking(run), "support_model": run.plan.get("global_eval_model"),
        "summary_sha256": (hashlib.sha256(run.summary_path.read_bytes()).hexdigest() if run.summary_path.is_file() else None),
        "status": run.summary.get("status", "missing_root_summary"),
        "health": health, "totals": _totals(health),
        "missing_entries": sorted(FULL_ENTRIES - set(run.entries)),
        "plan_sha256": _digest(run.plan), "entries": entries, "cells": cells,
        "warnings": list(run.warnings),
    }


def build_batch_report(root: str | Path, *, mapping_path: str | Path = DEFAULT_MAPPING,
                       selection: str = "by-type") -> dict:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactError(f"formal root does not exist: {root}")
    mapping = load_mapping(mapping_path, selection)
    runs, skipped = [], []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        try:
            run = _collect_run(path, mapping["metrics"])
        except (SimEvalError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            skipped.append({"directory": path.name, "root": str(path), "reason": str(exc)})
            continue
        run["column"] = f"M{len(runs) + 1}"
        if runs:
            reference = runs[0]
            for entry_id in sorted(FULL_ENTRIES):
                for field in ("sample_sha256", "source_manifest_digest", "protocol_identity"):
                    value = run["entries"][entry_id][field]
                    previous = reference["entries"][entry_id][field]
                    if value is None or previous is None or value != previous:
                        run["warnings"].append(f"相对 M1：{entry_id} 的 {field} 不同或未核验")
            if run["thinking"] != reference["thinking"]:
                run["warnings"].append("相对 M1：thinking 设置或证据来源不同")
            if run["support_model"] != reference["support_model"]:
                run["warnings"].append("相对 M1：支撑模型预设不同")
        runs.append(run)
    return {"schema_version": "1.0", "input_root": str(root), "selection": selection,
            "aggregation": mapping["aggregation"], "mapping_revision": mapping["mapping_revision"],
            "mapping_sha256": _digest(mapping), "dimensions": mapping["dimensions"],
            "selection_notes": mapping.get("selection_notes", []), "metrics": mapping["metrics"],
            "runs": runs, "skipped": skipped}


def render_batch_markdown(report: dict, *, digits: int = 4) -> str:
    if not 1 <= digits <= 12:
        raise ValueError("digits must be between 1 and 12")
    runs = report["runs"]
    lines = ["# 全量评测：五维主指标汇总", "",
             f"纳入 {len(runs)} 个有可用主指标的 run；跳过 {len(report['skipped'])} 个目录。",
             "读取原始 JSON 聚合分，按能力维度分组，不计算跨量纲均分或模型总排名。",
             "`—` 表示缺失/不可用，`N/A` 表示不适用，0 保留为真实零分；† 表示指标缺失、覆盖不完整或入口状态需核查。", "",
             "| 列 | 模型 | Thinking | 结果目录 |", "|---|---|---|---|"]
    for run in runs:
        lines.append("| " + " | ".join(_escape(run[k]) for k in ("column", "model", "thinking", "directory")) + " |")
    if not runs:
        lines.extend(["", "没有可用主指标，详见跳过原因。"])
    lines.extend(["", "## 执行数量", "",
                  "按 case/repetition 计数（LiC 每个任务重复 10 次）；— 表示未知。数目是可读取入口的已知合计，待补 Judge 与完成数可能重叠。",
                  "| 列 | 原状态 | 计划 | 已记录 | 完成 | 失败 | 待补 Judge | 缺记录 | 缺汇总入口 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|"])
    for run in runs:
        counts = []
        for field in ("expected_count", "result_count", "completed_count", "failed_count", "pending_judge_count", "missing_count"):
            value = run["totals"][field]
            known = run["totals"][field + "_known_entries"]
            text = str(value) if value is not None else "—"
            if value is not None and known < len(run["health"]):
                text += f"（{known}/{len(run['health'])} 入口已知）"
            counts.append(text)
        lines.append("| " + " | ".join([run["column"], _escape(run["status"]), *counts,
                     _escape(", ".join(run["missing_entries"])) or "无"]) + " |")
    for dimension in report["dimensions"] if runs else []:
        lines.extend(["", f"## {dimension['id']}. {dimension['name_zh']}", "", dimension["question"], "",
                      "| 数据集 · 主指标 | 单位 · 方向 | " + " | ".join(r["column"] for r in runs) + " |",
                      "|---|---|" + "---:|" * len(runs)])
        for index, spec in enumerate(report["metrics"]):
            if spec["dimension"] != dimension["id"]:
                continue
            cells = [r["cells"][index] for r in runs]
            units = sorted({f"{c['unit'] or '?'} · {DIRECTIONS.get(c['direction'], '?')}" for c in cells})
            values = [("N/A" if c["status"] == "not_applicable" else _format(c["value"], digits))
                      + ("†" if c["flags"] else "") for c in cells]
            lines.append("| " + " | ".join([_escape(spec["label"]), _escape(" / ".join(units)), *values]) + " |")
    lines.extend(["", "## 指标选择", ""])
    lines.extend("- " + note for note in report["selection_notes"])
    lines.extend(["", "## 核查信息", ""])
    lines.append("模型差异按上表原样标注；未显式设置 thinking 不代表已关闭。CSV/JSON 保留精确指标键、数值、覆盖信息及来源。")
    for run in runs:
        lines.extend(f"- {run['column']}：{_escape(warning)}" for warning in run["warnings"])
        for spec, cell in zip(report["metrics"], run["cells"]):
            if cell["flags"]:
                lines.append(f"- {run['column']} / {spec['label']}：{', '.join(cell['flags'])}；{_escape(cell['reason'])}")
    lines.extend(["", "## 跳过的目录", "", "| 目录 | 原因 |", "|---|---|"])
    lines.extend(f"| {_escape(r['directory'])} | {_escape(r['reason'])} |" for r in report["skipped"])
    return "\n".join(lines) + "\n"


def write_batch_report(*, root: str | Path, output_dir: str | Path,
                       mapping_path: str | Path = DEFAULT_MAPPING,
                       selection: str = "by-type", digits: int = 4) -> tuple[dict, dict[str, Path]]:
    output = Path(output_dir).expanduser().resolve()
    source = Path(root).expanduser().resolve()
    if source == output or source in output.parents:
        raise ArtifactError("output-dir must be outside the formal input tree")
    report = build_batch_report(source, mapping_path=mapping_path, selection=selection)
    markdown = render_batch_markdown(report, digits=digits)
    wide = io.StringIO(newline="")
    writer = csv.writer(wide)
    writer.writerow(["dimension", "dataset", "metric_type", "entry", "metric", "label"] + [r["directory"] for r in report["runs"]])
    for index, spec in enumerate(report["metrics"]):
        writer.writerow([spec[k] for k in ("dimension", "dataset", "metric_type", "entry", "metric", "label")]
                        + [r["cells"][index]["value"] for r in report["runs"]])
    detail = io.StringIO(newline="")
    writer = csv.writer(detail)
    writer.writerow(["run", "model", "thinking", "dimension", "entry", "metric", "value", "status", "unit", "direction", "valid", "applicable", "natural_unit", "flags", "reason"])
    for run in report["runs"]:
        for spec, cell in zip(report["metrics"], run["cells"]):
            cov = cell["coverage"] or {}
            writer.writerow([run["directory"], run["model"], run["thinking"], spec["dimension"], spec["entry"], spec["metric"], cell["value"], cell["status"], cell["unit"], cell["direction"], cov.get("valid"), cov.get("applicable"), cov.get("natural_unit") or cov.get("basis"), "; ".join(cell["flags"]), cell["reason"]])
    skipped = io.StringIO(newline="")
    writer = csv.writer(skipped)
    writer.writerow(["directory", "reason"])
    writer.writerows((r["directory"], r["reason"]) for r in report["skipped"])
    status_csv = io.StringIO(newline="")
    writer = csv.writer(status_csv)
    fields = ("status", "expected_count", "result_count", "completed_count", "failed_count",
              "pending_judge_count", "missing_count", "count_source", "records_available",
              "summary_completed_count", "summary_failed_count")
    writer.writerow(["run", "entry", *fields, "issues"])
    for run in report["runs"]:
        for key, h in run["health"].items():
            writer.writerow([run["directory"], key, *(h[k] for k in fields), "; ".join(h["issues"])])
    contents = {"run_status.csv": "\ufeff" + status_csv.getvalue(), "capability_summary.md": markdown, "capability_summary.csv": "\ufeff" + wide.getvalue(),
                "capability_details.csv": "\ufeff" + detail.getvalue(), "capability_summary.json": canonical_json(report) + "\n",
                "skipped_runs.csv": "\ufeff" + skipped.getvalue()}
    paths = {}
    for name, content in contents.items():
        paths[name] = output / name
        atomic_write_text(paths[name], content)
    return report, paths
