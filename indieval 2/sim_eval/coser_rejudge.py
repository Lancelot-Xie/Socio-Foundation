"""Judge-only CoSER pilot using stored dialogues and the upstream critic prompt."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import CheckpointStore, atomic_write_json, atomic_write_text, load_json
from .backends.concurrency import EndpointLimiterRegistry
from .benchmarks.common import (
    DEFAULT_CONTRACT_RETRIES,
    generate_and_parse_with_contract_retries,
    json_schema_response_format,
    parse_json_object,
)
from .benchmarks.coser import (
    COSER_DIMENSIONS,
    LENGTH_CORRECTION_PER_ACTOR_TURN,
    OFFICIAL_COSER_CRITIC_PROMPT_REVISION,
    OFFICIAL_COSER_CRITIC_TEMPLATE as _OFFICIAL_CRITIC_TEMPLATE,
    CoserAdapter,
    _dialogue_text,
    _remove_inner_thoughts,
    build_official_coser_critic_prompt,
    coser_length_corrected_score,
)
from .contracts import (
    BenchmarkCase,
    ChatMessage,
    ModelRequest,
    ModelResponse,
    ResultStatus,
    case_result_from_dict,
)
from .data.loaders import load_import_spec, load_local_cases
from .environments.coser import ENVIRONMENT_ROLE, deterministic_token_count
from .errors import ArtifactError, ConfigurationError, ParseError, ValidationError
from .json_utils import canonical_json, jsonable, sha256_digest
from .runtime_config import (
    build_api_backend,
    load_benchmark_runtime_config,
    role_request_overrides,
)


COSER_REJUDGE_REVISION = "coser-upstream-critic-judge-only-v3-20260826"
OFFICIAL_PROMPT_REVISION = OFFICIAL_COSER_CRITIC_PROMPT_REVISION
DEFAULT_PILOT_LIMIT = 20
DEFAULT_PILOT_SEED = 20260826
_LEGACY_COSER_NORMALIZED_RECORD_SHA256 = "44e113a5f79dd6e61b5955c03781be64f96659a0a15c773d47903d089c36b722"
_CURRENT_COSER_NORMALIZED_RECORD_SHA256 = "418363ad52bed9d8b0a6ba3b25c66bd3814625ab4db5da1961557c7c3a63efb5"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _stable_rank(seed: int, case_id: str) -> str:
    return hashlib.sha256(
        f"{seed}:{case_id}:{COSER_REJUDGE_REVISION}".encode("utf-8")
    ).hexdigest()


def _source_manifest_compatibility(
    frozen: Mapping[str, Any],
    loaded: Any,
) -> str:
    """Allow only the audited CoSER context/topology upgrade for old rollouts."""

    if loaded.digest == sha256_digest(frozen):
        return "exact_source_manifest_match"
    frozen_hashes = frozen.get("file_hashes")
    loaded_hashes = loaded.file_hashes
    frozen_metadata = frozen.get("metadata")
    loaded_metadata = loaded.metadata
    stable_fields_match = (
        frozen.get("benchmark_id") == loaded.benchmark_id == "coser"
        and frozen.get("source_revision") == loaded.source_revision == "7cc80430f92532cda85df45015a4aca8ecc068d0"
        and frozen.get("source_kind") == loaded.source_kind == "official"
        and frozen.get("split") == loaded.split == "test"
        and frozen.get("resolved_population") == loaded.resolved_population == 200
        and isinstance(frozen_hashes, Mapping)
        and _LEGACY_COSER_NORMALIZED_RECORD_SHA256 in set(frozen_hashes.values())
        and _CURRENT_COSER_NORMALIZED_RECORD_SHA256 in set(loaded_hashes.values())
        and isinstance(frozen_metadata, Mapping)
        and frozen_metadata.get("source_sha256") == loaded_metadata.get("source_sha256")
        == "9b0abc0e43805a447bc7f6e1ac8cee7a7c7fc7c89a8013df5d757a794c8b0a2b"
    )
    if stable_fields_match:
        return "audited_legacy_rollout_to_gca_v2_context_upgrade"
    raise ArtifactError(
        "CoSER import source manifest does not match the source manifest frozen in the run; "
        "only the audited legacy eval_collection_v1 context upgrade is accepted"
    )


def _largest_remainder_quotas(counts: Mapping[str, int], limit: int) -> Mapping[str, int]:
    total = sum(counts.values())
    if total <= 0 or limit <= 0 or limit > total:
        raise ValidationError("CoSER pilot limit must be positive and no larger than available cases")
    exact = {key: limit * count / total for key, count in counts.items()}
    quotas = {key: min(counts[key], math.floor(value)) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (-(exact[key] - math.floor(exact[key])), key),
    )
    while remaining:
        advanced = False
        for key in order:
            if quotas[key] < counts[key]:
                quotas[key] += 1
                remaining -= 1
                advanced = True
                if not remaining:
                    break
        if not advanced:
            raise AssertionError("unable to allocate CoSER pilot quotas")
    return quotas


def select_coser_pilot_cases(
    cases: Sequence[BenchmarkCase],
    available_case_ids: Sequence[str],
    *,
    limit: int = DEFAULT_PILOT_LIMIT,
    seed: int = DEFAULT_PILOT_SEED,
) -> tuple[BenchmarkCase, ...]:
    """Select a deterministic proportional ID/OOD pilot from completed source records."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValidationError("CoSER pilot limit must be a positive integer")
    available = set(available_case_ids)
    candidates = [case for case in cases if case.case_id in available]
    if len(candidates) < limit:
        raise ValidationError(
            f"CoSER pilot requested {limit} cases but only {len(candidates)} completed source cases are available"
        )
    grouped: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in candidates:
        strata = case.metadata.get("strata")
        label = str(strata.get("in_domain_status") if isinstance(strata, Mapping) else "unknown")
        grouped[label].append(case)
    quotas = _largest_remainder_quotas({key: len(value) for key, value in grouped.items()}, limit)
    selected = []
    for label in sorted(grouped):
        ranked = sorted(grouped[label], key=lambda case: _stable_rank(seed, case.case_id))
        selected.extend(ranked[: quotas[label]])
    return tuple(sorted(selected, key=lambda case: _stable_rank(seed, case.case_id)))


def _clean_generated_dialogue(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    prediction = _mapping(record.get("prediction"), "CoSER record prediction")
    raw = prediction.get("public_dialogue")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise ArtifactError("CoSER record prediction.public_dialogue must be a nonempty array")
    cleaned = []
    for index, item in enumerate(raw):
        value = dict(_mapping(item, f"prediction.public_dialogue[{index}]"))
        value.pop("inner_thought", None)
        speaker = str(value.get("speaker") or value.get("character") or "")
        if not speaker:
            raise ArtifactError(f"prediction.public_dialogue[{index}] lacks speaker")
        for field in ("content", "message", "speech"):
            if isinstance(value.get(field), str) and speaker.casefold() != ENVIRONMENT_ROLE.casefold():
                value[field] = _remove_inner_thoughts(str(value[field]))
        cleaned.append(value)
    return cleaned


def _official_prompt(case: BenchmarkCase, dimension: str) -> str:
    return build_official_coser_critic_prompt(case, dimension)


def build_official_coser_judge_request(
    case: BenchmarkCase,
    record: Mapping[str, Any],
    *,
    dimension: str,
    judge_role: Mapping[str, Any],
    seed: int,
    max_context_tokens: int,
) -> ModelRequest:
    """Build the upstream system-prompt/plain-dialogue-user request with no goal/full-plot extras."""

    official_name = COSER_DIMENSIONS[dimension]
    generated = _clean_generated_dialogue(record)
    system_prompt = _official_prompt(case, dimension)
    dialogue = _dialogue_text(generated)
    used_tokens = deterministic_token_count(system_prompt) + deterministic_token_count(dialogue)
    if used_tokens > max_context_tokens:
        raise ValidationError(
            f"official CoSER critic context needs {used_tokens} accounting tokens, budget is {max_context_tokens}"
        )
    flaw_schema = {
        "type": "object",
        "properties": {
            "instance": {"type": "string", "minLength": 1},
            "type": {"type": "string", "minLength": 1},
            "severity": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["instance", "type", "severity"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            official_name: {
                "type": "object",
                "properties": {"flaws": {"type": "array", "items": flaw_schema}},
                "required": ["flaws"],
                "additionalProperties": False,
            }
        },
        "required": [official_name],
        "additionalProperties": False,
    }
    overrides = role_request_overrides(judge_role)
    return ModelRequest(
        request_id=f"coser-rejudge:{case.case_id}:{dimension}",
        messages=(
            ChatMessage("system", system_prompt, metadata={"visibility": "evaluator_only"}),
            ChatMessage("user", dialogue, metadata={"visibility": "evaluator_only"}),
        ),
        model=str(overrides.pop("model")),
        seed=seed,
        response_format=json_schema_response_format(f"coser_rejudge_{dimension}", schema),
        metadata={
            "route_role": "judge",
            "benchmark_id": "coser",
            "dimension": dimension,
            "prompt_revision": OFFICIAL_PROMPT_REVISION,
            "input_protocol": "upstream_system_prompt_and_plain_simulation_user_message",
            "explicit_character_goals_included": False,
            "full_structured_plot_included": False,
            "used_accounting_tokens": used_tokens,
        },
        **overrides,
    )


def parse_official_coser_judge_response(
    response: ModelResponse,
    *,
    dimension: str,
) -> Mapping[str, Any]:
    payload = parse_json_object(response, label=f"CoSER {dimension} rejudge response")
    official_name = COSER_DIMENSIONS[dimension]
    if set(payload) != {official_name}:
        raise ParseError(f"CoSER rejudge requires exactly the key {official_name!r}")
    body = payload[official_name]
    if not isinstance(body, Mapping) or set(body) != {"flaws"}:
        raise ParseError(f"CoSER rejudge {official_name} requires exactly the flaws field")
    flaws = body["flaws"]
    if isinstance(flaws, (str, bytes)) or not isinstance(flaws, Sequence):
        raise ParseError("CoSER rejudge flaws must be an array")
    normalized = []
    for index, raw in enumerate(flaws):
        if not isinstance(raw, Mapping):
            raise ParseError(f"CoSER rejudge flaw #{index} must be an object")
        flaw = raw
        if set(flaw) != {"instance", "type", "severity"}:
            raise ParseError("each CoSER rejudge flaw requires exactly instance/type/severity")
        instance = str(flaw["instance"]).strip()
        flaw_type = str(flaw["type"]).strip()
        severity = flaw["severity"]
        if not instance or not flaw_type or isinstance(severity, bool) or not isinstance(severity, int) or not 1 <= severity <= 5:
            raise ParseError("CoSER rejudge flaw values violate the official output contract")
        normalized.append({"instance": instance, "type": flaw_type, "severity": severity})
    return {official_name: {"flaws": normalized}}


def _old_score(record: Mapping[str, Any], dimension: str) -> float | None:
    raw = record.get("metrics")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return None
    name = f"coser.scene.{dimension}"
    for metric in raw:
        if isinstance(metric, Mapping) and metric.get("name") == name:
            value = metric.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _actor_rounds(record: Mapping[str, Any]) -> int:
    return sum(
        str(item.get("speaker") or item.get("character") or "").casefold()
        != ENVIRONMENT_ROLE.casefold()
        for item in _clean_generated_dialogue(record)
    )


def _request_seed(seed: int, case_id: str, dimension: str) -> int:
    digest = hashlib.sha256(f"{seed}:{case_id}:{dimension}:{COSER_REJUDGE_REVISION}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _read_checkpoint_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    rows = []
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise ArtifactError(f"cannot read CoSER rejudge checkpoint {path}: {exc}") from exc
    for index, encoded in enumerate(lines, start=1):
        if not encoded.strip():
            continue
        try:
            row = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"invalid CoSER rejudge checkpoint at {path}:{index}: {exc}") from exc
        rows.append(_mapping(row, f"{path}:{index}"))
    return rows


def _latest_rejudge_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    latest = {}
    attempts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        key = (str(row.get("case_id") or ""), str(row.get("dimension") or ""))
        if not key[0] or key[1] not in COSER_DIMENSIONS:
            raise ArtifactError(f"invalid CoSER rejudge checkpoint key: {key}")
        expected = attempts[key]
        if row.get("attempt") != expected:
            raise ArtifactError(f"invalid CoSER rejudge attempt for {key}: expected {expected}")
        if key in latest and latest[key].get("status") == "completed":
            raise ArtifactError(f"CoSER rejudge checkpoint appended after completion for {key}")
        latest[key] = row
        attempts[key] += 1
    return latest


def _append_checkpoint(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(row) + "\n"
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ArtifactError(f"cannot append CoSER rejudge checkpoint {path}: {exc}") from exc


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_sum = sum((x - left_mean) ** 2 for x in left)
    right_sum = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_sum * right_sum)
    return numerator / denominator if denominator else None


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = rank
        start = end
    return ranks


def summarize_coser_rejudge_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_case_ids: Sequence[str],
) -> Mapping[str, Any]:
    latest = _latest_rejudge_rows(rows)
    selected = set(selected_case_ids)
    unexpected = sorted(case_id for case_id, _ in latest if case_id not in selected)
    if unexpected:
        raise ArtifactError(
            f"CoSER rejudge checkpoint contains cases outside pilot manifest: {unexpected[:5]}"
        )
    completed = [row for row in latest.values() if row.get("status") == "completed"]
    dimension_summary = {}
    for dimension in COSER_DIMENSIONS:
        current = [row for row in completed if row["dimension"] == dimension]
        old = [float(row["old_score"]) for row in current if row.get("old_score") is not None]
        paired = [row for row in current if row.get("old_score") is not None]
        old_paired = [float(row["old_score"]) for row in paired]
        new_paired = [float(row["new_score"]) for row in paired]
        dimension_summary[dimension] = {
            "completed_count": len(current),
            "paired_count": len(paired),
            "old_mean": _mean(old),
            "new_mean": _mean([float(row["new_score"]) for row in current]),
            "paired_delta_mean": _mean(
                [float(row["new_score"]) - float(row["old_score"]) for row in paired]
            ),
            "old_saturation_100_rate": (
                sum(value == 100.0 for value in old_paired) / len(old_paired)
                if old_paired
                else None
            ),
            "new_saturation_100_rate": (
                sum(value == 100.0 for value in new_paired) / len(new_paired)
                if new_paired
                else None
            ),
            "mean_flaw_count": _mean([float(row["flaw_count"]) for row in current]),
            "pearson": _pearson(old_paired, new_paired),
            "spearman": _pearson(_ranks(old_paired), _ranks(new_paired)) if old_paired else None,
        }
    case_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        case_rows[str(row["case_id"])].append(row)
    paired_case_averages = []
    for case_id in selected_case_ids:
        current = case_rows.get(case_id, [])
        if len(current) != len(COSER_DIMENSIONS) or any(row.get("old_score") is None for row in current):
            continue
        old_average = statistics.mean(float(row["old_score"]) for row in current)
        new_average = statistics.mean(float(row["new_score"]) for row in current)
        paired_case_averages.append(
            {
                "case_id": case_id,
                "old_critic_average": old_average,
                "new_critic_average": new_average,
                "delta": new_average - old_average,
            }
        )
    expected = len(selected_case_ids) * len(COSER_DIMENSIONS)
    response_attempt_count = sum(
        int(row.get("response_attempt_count") or 0) for row in latest.values()
    )
    usage_rows = [
        response["usage"]
        for row in latest.values()
        for response in row.get("responses", [])
        if isinstance(response, Mapping) and isinstance(response.get("usage"), Mapping)
    ]

    def usage_total(field: str) -> int | None:
        values = [
            int(usage[field])
            for usage in usage_rows
            if isinstance(usage.get(field), int) and not isinstance(usage.get(field), bool)
        ]
        return sum(values) if values else None

    return {
        "status": "completed" if len(completed) == expected else "completed_with_failures",
        "selected_case_count": len(selected_case_ids),
        "expected_judge_unit_count": expected,
        "completed_judge_unit_count": len(completed),
        "failed_judge_unit_count": sum(row.get("status") == "failed" for row in latest.values()),
        "response_attempt_count": response_attempt_count,
        "usage": {
            "responses_with_usage": len(usage_rows),
            "prompt_tokens": usage_total("prompt_tokens"),
            "completion_tokens": usage_total("completion_tokens"),
            "total_tokens": usage_total("total_tokens"),
        },
        "dimensions": dimension_summary,
        "critic_average": {
            "paired_case_count": len(paired_case_averages),
            "old_mean": _mean([row["old_critic_average"] for row in paired_case_averages]),
            "new_mean": _mean([row["new_critic_average"] for row in paired_case_averages]),
            "paired_delta_mean": _mean([row["delta"] for row in paired_case_averages]),
            "pearson": _pearson(
                [row["old_critic_average"] for row in paired_case_averages],
                [row["new_critic_average"] for row in paired_case_averages],
            ),
            "spearman": (
                _pearson(
                    _ranks([row["old_critic_average"] for row in paired_case_averages]),
                    _ranks([row["new_critic_average"] for row in paired_case_averages]),
                )
                if paired_case_averages
                else None
            ),
        },
        "paired_cases": paired_case_averages,
    }


def _render_report(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    formal = manifest.get("mode") == "formal_full_correction"
    lines = [
        "# CoSER official-prompt formal correction" if formal else "# CoSER official-prompt rejudge pilot",
        "",
        f"- Source run: `{manifest['source_run_id']}`",
        f"- Selected scenes: `{summary['selected_case_count']}`",
        f"- Judge units: `{summary['completed_judge_unit_count']}/{summary['expected_judge_unit_count']}`",
        f"- Response attempts (including contract retries): `{summary['response_attempt_count']}`",
        f"- Judge model: `{manifest['judge']['model']}`",
        f"- Prompt revision: `{OFFICIAL_PROMPT_REVISION}`",
        "- Stored dialogue reused: `yes`",
        "- Actor/environment/NSP regeneration: `no`",
        "",
        "| Dimension | Old mean | Official-prompt mean | Δ | Old 100% | New 100% | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def value(raw: Any) -> str:
        return "—" if raw is None else f"{float(raw):.6f}"

    for dimension in COSER_DIMENSIONS:
        row = summary["dimensions"][dimension]
        lines.append(
            "| "
            + " | ".join(
                (
                    dimension,
                    value(row["old_mean"]),
                    value(row["new_mean"]),
                    value(row["paired_delta_mean"]),
                    value(row["old_saturation_100_rate"]),
                    value(row["new_saturation_100_rate"]),
                    value(row["pearson"]),
                    value(row["spearman"]),
                )
            )
            + " |"
        )
    critic = summary["critic_average"]
    lines.extend(
        [
            "",
            "## Four-dimension critic average",
            "",
            f"- Paired scenes: `{critic['paired_case_count']}`",
            f"- Old mean: `{value(critic['old_mean'])}`",
            f"- Official-prompt mean: `{value(critic['new_mean'])}`",
            f"- Mean paired delta: `{value(critic['paired_delta_mean'])}`",
            f"- Pearson/Spearman: `{value(critic['pearson'])}` / `{value(critic['spearman'])}`",
            "",
        ]
    )
    return "\n".join(lines)


def _default_import_manifest() -> Path:
    return Path("local_data/coser/import_manifest.json")


def _resolve_runtime_config(
    run_manifest: Mapping[str, Any],
    explicit: str | Path | None,
) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
    else:
        metadata = _mapping(run_manifest.get("metadata", {}), "run_manifest.metadata")
        raw = metadata.get("runtime_config")
        path = Path(str(raw)).expanduser().resolve() if raw else Path()
        if not raw or not path.is_file():
            path = Path(__file__).resolve().parent / "resources" / "protocols" / "coser.json"
    if not path.is_file():
        raise ArtifactError(
            f"CoSER runtime config is unavailable at {path}; pass --runtime-config explicitly"
        )
    return path


def _stored_judge_role(
    records: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, str]:
    """Recover the exact non-secret judge route frozen on live source records."""

    configurations: dict[str, Mapping[str, Any]] = {}
    missing = 0
    for record in records:
        metadata = record.get("metadata")
        execution = metadata.get("execution_provenance") if isinstance(metadata, Mapping) else None
        support_roles = execution.get("support_roles") if isinstance(execution, Mapping) else None
        judge = support_roles.get("judge") if isinstance(support_roles, Mapping) else None
        if not isinstance(judge, Mapping):
            missing += 1
            continue
        configurations[canonical_json(judge)] = judge
    if not configurations:
        return None, "runtime_config_fallback_no_record_execution_provenance"
    if missing:
        raise ArtifactError(
            "CoSER source records contain only partial judge execution provenance; "
            "refusing to guess a mixed source configuration"
        )
    if len(configurations) != 1:
        raise ArtifactError(
            "CoSER source records contain multiple configured judge routes; "
            "split the source run by judge configuration before rejudging"
        )
    return next(iter(configurations.values())), "records.metadata.execution_provenance.support_roles.judge"


def _record_judge_context_budget(
    records: Sequence[Mapping[str, Any]],
) -> int | None:
    values = set()
    for record in records:
        metadata = record.get("metadata")
        runtime = metadata.get("runtime_provenance") if isinstance(metadata, Mapping) else None
        value = runtime.get("max_judge_context_tokens") if isinstance(runtime, Mapping) else None
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ArtifactError("CoSER record max_judge_context_tokens is invalid")
            values.add(value)
    if len(values) > 1:
        raise ArtifactError("CoSER source records contain mixed judge context budgets")
    return next(iter(values)) if values else None


def _source_judge_model_identity(
    records: Sequence[Mapping[str, Any]],
    run_identity: Mapping[str, Any],
) -> tuple[str, str | None]:
    observed: set[tuple[str, str | None]] = set()
    for record in records:
        metadata = record.get("metadata")
        provenance = metadata.get("judge_provenance") if isinstance(metadata, Mapping) else None
        models = provenance.get("judge_models") if isinstance(provenance, Mapping) else None
        revisions = provenance.get("judge_revisions") if isinstance(provenance, Mapping) else None
        if (
            isinstance(models, Sequence)
            and not isinstance(models, (str, bytes))
            and len(models) == 1
        ):
            revision = (
                str(revisions[0])
                if isinstance(revisions, Sequence)
                and not isinstance(revisions, (str, bytes))
                and len(revisions) == 1
                else None
            )
            observed.add((str(models[0]), revision))
    if len(observed) > 1:
        raise ArtifactError("CoSER source records contain mixed judge model identities")
    if observed:
        return next(iter(observed))

    manifest_judge = run_identity.get("judge")
    manifest_judge = manifest_judge if isinstance(manifest_judge, Mapping) else {}
    models = manifest_judge.get("judge_models")
    revisions = manifest_judge.get("judge_revisions")
    if not (
        isinstance(models, Sequence)
        and not isinstance(models, (str, bytes))
        and len(models) == 1
    ):
        raise ArtifactError("CoSER source artifacts do not freeze exactly one judge model")
    revision = (
        str(revisions[0])
        if isinstance(revisions, Sequence)
        and not isinstance(revisions, (str, bytes))
        and len(revisions) == 1
        else None
    )
    return str(models[0]), revision


def validate_coser_rejudge_pilot(
    run_directory: str | Path,
    *,
    import_manifest: str | Path | None = None,
    runtime_config: str | Path | None = None,
    limit: int | None = DEFAULT_PILOT_LIMIT,
    seed: int = DEFAULT_PILOT_SEED,
) -> Mapping[str, Any]:
    run_dir = Path(run_directory).expanduser().resolve()
    required = tuple(run_dir / name for name in ("run_manifest.json", "source_manifest.json", "sample_manifest.json", "records.jsonl"))
    for path in required:
        if not path.is_file():
            raise ArtifactError(f"CoSER rejudge requires {path}")
    run_manifest = _mapping(load_json(run_dir / "run_manifest.json"), "run_manifest")
    identity = _mapping(run_manifest.get("identity"), "run_manifest.identity")
    if identity.get("benchmark_id") != "coser":
        raise ArtifactError("CoSER rejudge input run is not benchmark_id=coser")
    source_manifest_document = _mapping(load_json(run_dir / "source_manifest.json"), "source_manifest")
    sample_manifest = _mapping(load_json(run_dir / "sample_manifest.json"), "sample_manifest")
    store = CheckpointStore(run_dir)
    source_records = list(store.iter_latest_record_dicts())
    completed_rows = [
        row for row in source_records if row.get("status") == ResultStatus.COMPLETED.value
    ]
    completed_case_ids = [str(row["case_id"]) for row in completed_rows]
    duplicate_case_ids = sorted(
        case_id for case_id, count in Counter(completed_case_ids).items() if count > 1
    )
    if duplicate_case_ids:
        raise ArtifactError(
            "CoSER rejudge requires one completed repetition per case; duplicates include "
            f"{duplicate_case_ids[:5]}"
        )
    completed_records = dict(zip(completed_case_ids, completed_rows))
    if not str(run_manifest.get("run_id") or "").strip():
        raise ArtifactError("CoSER source run manifest lacks run_id")
    manifest_path = Path(import_manifest).expanduser().resolve() if import_manifest else _default_import_manifest()
    spec = load_import_spec(manifest_path)
    cases, loaded_source_manifest = load_local_cases(spec)
    source_manifest_compatibility = _source_manifest_compatibility(
        source_manifest_document, loaded_source_manifest
    )
    selected_case_ids = sample_manifest.get("selected_case_ids")
    if isinstance(selected_case_ids, (str, bytes)) or not isinstance(selected_case_ids, Sequence):
        raise ArtifactError("CoSER sample manifest lacks selected_case_ids")
    if set(completed_records) - set(str(value) for value in selected_case_ids):
        raise ArtifactError("CoSER records contain cases outside the frozen sample manifest")
    case_by_id = {case.case_id: case for case in cases}
    missing_cases = sorted(set(completed_records) - set(case_by_id))
    if missing_cases:
        raise ArtifactError(
            f"CoSER import manifest is missing completed source cases: {missing_cases[:5]}"
        )
    if limit is None:
        selected = tuple(
            case_by_id[str(case_id)]
            for case_id in selected_case_ids
            if str(case_id) in completed_records
        )
        if not selected:
            raise ArtifactError("CoSER source run has no completed records to rejudge")
        selection_algorithm = "all_completed_cases_in_frozen_sample_order_v1"
    else:
        selected = select_coser_pilot_cases(
            cases,
            tuple(completed_records),
            limit=limit,
            seed=seed,
        )
        selection_algorithm = "proportional_in_domain_status_largest_remainder_then_sha256_rank_v1"
    runtime_path = _resolve_runtime_config(run_manifest, runtime_config)
    metadata = _mapping(run_manifest.get("metadata", {}), "run_manifest.metadata")
    global_eval_model = metadata.get("global_eval_model")
    config = load_benchmark_runtime_config(
        runtime_path,
        global_eval_model=str(global_eval_model) if global_eval_model else None,
    )
    if config.get("benchmark_id") != "coser":
        raise ConfigurationError("resolved runtime config is not for CoSER")
    roles = _mapping(config.get("roles"), "runtime roles")
    configured_judge_role = _mapping(roles.get("judge"), "runtime roles.judge")
    stored_judge_role, judge_role_source = _stored_judge_role(completed_rows)
    judge_role = stored_judge_role or configured_judge_role
    expected_model, expected_revision = _source_judge_model_identity(
        completed_rows, identity
    )
    if str(judge_role.get("model")) != expected_model:
        raise ConfigurationError(
            "resolved rejudge model does not match the source run judge identity: "
            f"resolved={judge_role.get('model')!r}, source={expected_model!r}"
        )
    if expected_revision is not None and str(judge_role.get("model_revision")) != expected_revision:
        raise ConfigurationError(
            "resolved rejudge revision does not match the source run judge identity: "
            f"resolved={judge_role.get('model_revision')!r}, source={expected_revision!r}"
        )
    environment = _mapping(config.get("environment", {}), "runtime environment")
    record_context_budget = _record_judge_context_budget(completed_rows)
    max_context_tokens = record_context_budget or int(
        environment.get("max_judge_context_tokens", 131072)
    )
    request_accounting_tokens = []
    for case in selected:
        source_record = completed_records[case.case_id]
        for dimension in COSER_DIMENSIONS:
            request = build_official_coser_judge_request(
                case,
                source_record,
                dimension=dimension,
                judge_role=judge_role,
                seed=_request_seed(seed, case.case_id, dimension),
                max_context_tokens=max_context_tokens,
            )
            request_accounting_tokens.append(
                int(request.metadata["used_accounting_tokens"])
            )
    strata_counts = Counter(
        str((_mapping(case.metadata.get("strata", {}), "case strata")).get("in_domain_status") or "unknown")
        for case in selected
    )
    return {
        "run_directory": run_dir,
        "run_manifest": run_manifest,
        "source_latest_records": source_records,
        "source_records": completed_records,
        "cases": {case.case_id: case for case in selected},
        "selected_case_ids": [case.case_id for case in selected],
        "selection_strata_counts": dict(strata_counts),
        "selection_algorithm": selection_algorithm,
        "runtime_config": runtime_path,
        "import_manifest": manifest_path,
        "source_manifest_compatibility": source_manifest_compatibility,
        "judge_role": judge_role,
        "judge_role_source": judge_role_source,
        "max_context_tokens": max_context_tokens,
        "request_validation": {
            "validated_request_count": len(request_accounting_tokens),
            "minimum_accounting_tokens": min(request_accounting_tokens),
            "maximum_accounting_tokens": max(request_accounting_tokens),
        },
        "request_timeout_seconds": float(
            _mapping(config.get("execution", {}), "runtime execution").get(
                "request_timeout_seconds", 120.0
            )
        ),
        "global_eval_model": global_eval_model,
    }


def _corrected_coser_record(
    source_record: Mapping[str, Any],
    dimension_rows: Mapping[str, Mapping[str, Any]],
    *,
    source_run_id: str,
    judge: Mapping[str, Any],
    max_context_tokens: int,
) -> Mapping[str, Any]:
    """Replace only CoSER critic-derived fields while preserving the stored rollout."""

    if set(dimension_rows) != set(COSER_DIMENSIONS):
        raise ArtifactError(
            f"corrected CoSER record {source_record.get('case_id')!r} lacks four completed dimensions"
        )
    actor_rounds = {int(row["actor_rounds"]) for row in dimension_rows.values()}
    if len(actor_rounds) != 1:
        raise ArtifactError("CoSER rejudge rows disagree on actor-round count")
    rounds = next(iter(actor_rounds))
    provenance = {
        "judge_models": [judge.get("model")],
        "judge_revisions": [judge.get("model_revision")],
        "rubric_revision": OFFICIAL_PROMPT_REVISION,
        "calls_per_output": len(COSER_DIMENSIONS),
        "source": "official_prompt_rejudge_sidecar",
    }
    replacement_metrics: dict[str, Mapping[str, Any]] = {}
    new_values: list[float] = []
    old_values: dict[str, float | None] = {}
    for dimension, official_name in COSER_DIMENSIONS.items():
        row = dimension_rows[dimension]
        value = float(row["new_score"])
        flaws = list(row["flaws"])
        new_values.append(value)
        old_raw = row.get("old_score")
        old_values[dimension] = float(old_raw) if isinstance(old_raw, (int, float)) else None
        replacement_metrics[f"coser.scene.{dimension}"] = {
            "name": f"coser.scene.{dimension}",
            "value": value,
            "direction": "higher_is_better",
            "unit": "score_0_to_100",
            "numerator": None,
            "denominator": None,
            "uncertainty": {},
            "metadata": {
                "official_dimension_name": official_name,
                "flaws": flaws,
                "flaw_visibility": "evaluator_only",
                "actor_rounds": rounds,
                "length_correction_per_actor_turn": LENGTH_CORRECTION_PER_ACTOR_TURN,
                "formula": "clamp(100 - 5*sum(severity) + 1.5*actor_rounds, 0, 100)",
                "judge_provenance": provenance,
                "scope": "scene_official_gca",
                "output_protocol": "upstream_dimension_flaws_envelope",
                "critic_prompt_revision": OFFICIAL_PROMPT_REVISION,
            },
        }
    average = statistics.mean(new_values)
    replacement_metrics["coser.scene.critic_average"] = {
        "name": "coser.scene.critic_average",
        "value": average,
        "direction": "higher_is_better",
        "unit": "score_0_to_100",
        "numerator": sum(new_values),
        "denominator": len(new_values),
        "uncertainty": {},
        "metadata": {
            "scope": "scene_official_gca_average",
            "aggregation": "arithmetic_mean_four_dimensions",
            "missing_dimensions": [],
            "availability": "available",
            "critic_prompt_revision": OFFICIAL_PROMPT_REVISION,
        },
    }
    source_metrics = source_record.get("metrics")
    if isinstance(source_metrics, (str, bytes)) or not isinstance(source_metrics, Sequence):
        raise ArtifactError("CoSER source record metrics must be an array")
    corrected_metrics = []
    replaced = set()
    for raw in source_metrics:
        if not isinstance(raw, Mapping):
            raise ArtifactError("CoSER source record metric must be an object")
        name = str(raw.get("name") or "")
        if name in replacement_metrics:
            corrected_metrics.append(replacement_metrics[name])
            replaced.add(name)
        else:
            metric = dict(raw)
            if name.startswith("coser.character."):
                metadata = dict(metric.get("metadata") or {})
                metadata["judge_provenance"] = provenance
                metric["metadata"] = metadata
            corrected_metrics.append(metric)
    for name in replacement_metrics:
        if name not in replaced:
            corrected_metrics.append(replacement_metrics[name])

    corrected = dict(source_record)
    corrected["metrics"] = corrected_metrics
    metadata = dict(corrected.get("metadata") or {})
    audits = [
        dict(item)
        for item in metadata.get("context_audits", ())
        if isinstance(item, Mapping) and item.get("request_kind") != "judge"
    ]
    audits.extend(
        {
            "request_kind": "judge",
            "visibility": "evaluator_only",
            "dimension": dimension,
            "tokenizer_revision": "unicode-regex-token-v1",
            "budget_tokens": max_context_tokens,
            "used_tokens": int(dimension_rows[dimension]["request_accounting_tokens"]),
            "protected_context_truncated": False,
            "hidden_thoughts_included": False,
            "prompt_revision": OFFICIAL_PROMPT_REVISION,
        }
        for dimension in COSER_DIMENSIONS
    )
    metadata.update(
        {
            "prompt_revision": CoserAdapter.prompt_revision,
            "critic_prompt_revision": OFFICIAL_PROMPT_REVISION,
            "context_audits": audits,
            "judge_status": {dimension: "available" for dimension in COSER_DIMENSIONS},
            "judge_errors": {},
            "judge_call_count": len(COSER_DIMENSIONS),
            "judge_provenance": provenance,
            "coser_rejudge": {
                "revision": COSER_REJUDGE_REVISION,
                "source_run_id": source_run_id,
                "source_record_preserved": True,
                "rollout_regenerated": False,
                "old_scene_scores": old_values,
            },
        }
    )
    corrected["metadata"] = metadata
    return corrected


def _materialize_corrected_results(
    *,
    resolved: Mapping[str, Any],
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    judge: Mapping[str, Any],
) -> Mapping[str, str]:
    latest = _latest_rejudge_rows(rows)
    completed_by_case: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for (case_id, dimension), row in latest.items():
        if row.get("status") == "completed":
            completed_by_case[case_id][dimension] = row
    expected_case_ids = set(resolved["source_records"])
    if set(completed_by_case) != expected_case_ids or any(
        set(completed_by_case[case_id]) != set(COSER_DIMENSIONS)
        for case_id in expected_case_ids
    ):
        raise ArtifactError("formal CoSER correction requires four completed rejudge units for every completed source case")

    run_manifest = resolved["run_manifest"]
    source_run_id = str(run_manifest["run_id"])
    corrected_rows = []
    for source_record in resolved["source_latest_records"]:
        case_id = str(source_record.get("case_id") or "")
        if case_id in completed_by_case:
            corrected_rows.append(
                _corrected_coser_record(
                    source_record,
                    completed_by_case[case_id],
                    source_run_id=source_run_id,
                    judge=judge,
                    max_context_tokens=int(resolved["max_context_tokens"]),
                )
            )
        else:
            corrected_rows.append(dict(source_record))

    records_path = output_dir / "corrected_records.jsonl"
    atomic_write_text(
        records_path,
        "".join(canonical_json(row) + "\n" for row in corrected_rows),
    )
    corrected_results = [case_result_from_dict(row) for row in corrected_rows]
    aggregate = CoserAdapter().aggregate(corrected_results)
    metrics_path = output_dir / "corrected_metrics.json"
    atomic_write_json(
        metrics_path,
        {
            "run_id": source_run_id,
            "source_run_id": source_run_id,
            "correction_revision": COSER_REJUDGE_REVISION,
            "critic_prompt_revision": OFFICIAL_PROMPT_REVISION,
            "metrics": jsonable(aggregate),
        },
    )
    source_paths = {
        name: resolved["run_directory"] / name
        for name in ("run_manifest.json", "source_manifest.json", "sample_manifest.json", "records.jsonl", "metrics.json")
        if (resolved["run_directory"] / name).is_file()
    }
    correction_path = output_dir / "correction_manifest.json"
    atomic_write_json(
        correction_path,
        {
            "schema_version": "1.0",
            "revision": COSER_REJUDGE_REVISION,
            "source_run_id": source_run_id,
            "source_run_directory": str(resolved["run_directory"]),
            "source_artifacts": {name: _sha256(path) for name, path in source_paths.items()},
            "source_artifacts_modified": False,
            "rollout_regenerated": False,
            "completed_source_case_count": len(expected_case_ids),
            "corrected_record_count": len(corrected_rows),
            "judge": dict(judge),
            "critic_prompt_revision": OFFICIAL_PROMPT_REVISION,
            "corrected_artifacts": {
                "corrected_records.jsonl": _sha256(records_path),
                "corrected_metrics.json": _sha256(metrics_path),
            },
        },
    )
    return {
        "correction_manifest": correction_path.name,
        "corrected_records": records_path.name,
        "corrected_metrics": metrics_path.name,
    }


def run_coser_rejudge_pilot(
    run_directory: str | Path,
    *,
    import_manifest: str | Path | None = None,
    runtime_config: str | Path | None = None,
    output_directory: str | Path | None = None,
    limit: int | None = DEFAULT_PILOT_LIMIT,
    seed: int = DEFAULT_PILOT_SEED,
    max_workers: int = 4,
    validate_only: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Mapping[str, Any]:
    formal = limit is None
    resolved = validate_coser_rejudge_pilot(
        run_directory,
        import_manifest=import_manifest,
        runtime_config=runtime_config,
        limit=limit,
        seed=seed,
    )
    run_dir = resolved["run_directory"]
    output_dir = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else (
            run_dir / "rejudge_official_prompt_full"
            if formal
            else run_dir / f"rejudge_official_prompt_pilot_n{limit}_seed{seed}"
        )
    )
    run_manifest = resolved["run_manifest"]
    judge_role = resolved["judge_role"]
    selected_count = len(resolved["selected_case_ids"])
    pilot_manifest = {
        "schema_version": "1.0",
        "revision": COSER_REJUDGE_REVISION,
        "mode": "formal_full_correction" if formal else "pilot_comparison",
        "prompt_revision": OFFICIAL_PROMPT_REVISION,
        "source_run_id": str(run_manifest["run_id"]),
        "source_run_directory": str(run_dir),
        "source_artifacts": {
            name: _sha256(run_dir / name)
            for name in ("run_manifest.json", "source_manifest.json", "sample_manifest.json", "records.jsonl")
        },
        "import_manifest": str(resolved["import_manifest"]),
        "import_manifest_sha256": _sha256(resolved["import_manifest"]),
        "source_manifest_compatibility": resolved["source_manifest_compatibility"],
        "runtime_config": str(resolved["runtime_config"]),
        "runtime_config_sha256": _sha256(resolved["runtime_config"]),
        "global_eval_model": resolved["global_eval_model"],
        "selection": {
            "algorithm": resolved["selection_algorithm"],
            "seed": seed,
            "limit": limit,
            "selected_case_ids": resolved["selected_case_ids"],
            "strata_counts": resolved["selection_strata_counts"],
        },
        "judge": {
            "model": judge_role.get("model"),
            "model_revision": judge_role.get("model_revision"),
            "backend": judge_role.get("backend"),
            "profile": judge_role.get("profile"),
            "generation": judge_role.get("generation"),
            "structured_output": judge_role.get("structured_output"),
            "configuration_source": resolved["judge_role_source"],
            "configuration_digest": sha256_digest(judge_role),
        },
        "reuse": {
            "stored_public_dialogue": True,
            "actor_generation_calls": 0,
            "environment_generation_calls": 0,
            "next_speaker_calls": 0,
            "judge_calls_expected_before_retries": selected_count * len(COSER_DIMENSIONS),
        },
        "request_validation": resolved["request_validation"],
        "output_directory": str(output_dir),
    }
    if validate_only:
        key = "rejudge_manifest" if formal else "pilot_manifest"
        return {"status": "valid", "network_calls": 0, key: pilot_manifest}

    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0:
        raise ConfigurationError("CoSER rejudge max_workers must be a positive integer")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_filename = "rejudge_manifest.json" if formal else "pilot_manifest.json"
    manifest_path = output_dir / manifest_filename
    if manifest_path.exists():
        existing = _mapping(load_json(manifest_path), "existing pilot manifest")
        if canonical_json(existing) != canonical_json(pilot_manifest):
            raise ArtifactError(
                f"refusing to reuse incompatible CoSER rejudge output directory: {output_dir}"
            )
    else:
        atomic_write_json(manifest_path, pilot_manifest)

    checkpoint_path = output_dir / "judge_records.jsonl"
    existing_rows = _read_checkpoint_rows(checkpoint_path)
    latest = _latest_rejudge_rows(existing_rows)
    tasks = []
    for case_id in resolved["selected_case_ids"]:
        for dimension in COSER_DIMENSIONS:
            row = latest.get((case_id, dimension))
            if row is None or row.get("status") != "completed":
                tasks.append((case_id, dimension, 0 if row is None else int(row["attempt"]) + 1))

    limiter = EndpointLimiterRegistry()
    backend = (
        build_api_backend(
            judge_role,
            timeout=resolved["request_timeout_seconds"],
            limiter_registry=limiter,
        )
        if tasks
        else None
    )

    def execute(task: tuple[str, str, int]) -> Mapping[str, Any]:
        case_id, dimension, attempt = task
        case = resolved["cases"][case_id]
        source_record = resolved["source_records"][case_id]
        request = build_official_coser_judge_request(
            case,
            source_record,
            dimension=dimension,
            judge_role=judge_role,
            seed=_request_seed(seed, case_id, dimension),
            max_context_tokens=resolved["max_context_tokens"],
        )
        responses: list[ModelResponse] = []
        official_name = COSER_DIMENSIONS[dimension]
        contract = (
            f'Return only JSON shaped as {{"{official_name}":{{"flaws":[]}}}}; '
            "each flaw requires nonempty instance/type and integer severity 1-5."
        )
        try:
            if backend is None:
                raise AssertionError("CoSER rejudge backend is unavailable for a pending task")
            payload = generate_and_parse_with_contract_retries(
                backend=backend,
                request=request,
                parser=lambda response: parse_official_coser_judge_response(
                    response, dimension=dimension
                ),
                responses=responses,
                contract=contract,
                max_retries=DEFAULT_CONTRACT_RETRIES,
            )
            flaws = payload[official_name]["flaws"]
            rounds = _actor_rounds(source_record)
            new_score = coser_length_corrected_score(flaws, rounds)
            old_score = _old_score(source_record, dimension)
            return {
                "schema_version": "1.0",
                "case_id": case_id,
                "group_id": case.group_id,
                "dimension": dimension,
                "attempt": attempt,
                "status": "completed",
                "old_score": old_score,
                "new_score": new_score,
                "delta": new_score - old_score if old_score is not None else None,
                "actor_rounds": rounds,
                "flaw_count": len(flaws),
                "severity_sum": sum(flaw["severity"] for flaw in flaws),
                "flaws": flaws,
                "request_fingerprint": request.fingerprint,
                "request_accounting_tokens": request.metadata["used_accounting_tokens"],
                "response_attempt_count": len(responses),
                "responses": [
                    {
                        "text": response.text,
                        "finish_reason": response.finish_reason,
                        "latency_ms": response.latency_ms,
                        "usage": response.usage.__dict__ if response.usage is not None else None,
                        "response_id": response.response_id,
                    }
                    for response in responses
                ],
            }
        except Exception as exc:
            return {
                "schema_version": "1.0",
                "case_id": case_id,
                "group_id": case.group_id,
                "dimension": dimension,
                "attempt": attempt,
                "status": "failed",
                "error": {"kind": type(exc).__name__, "message": str(exc)},
                "request_fingerprint": request.fingerprint,
                "response_attempt_count": len(responses),
                "responses": [
                    {
                        "text": response.text,
                        "finish_reason": response.finish_reason,
                        "latency_ms": response.latency_ms,
                        "usage": response.usage.__dict__ if response.usage is not None else None,
                        "response_id": response.response_id,
                    }
                    for response in responses
                ],
            }

    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="coser-rejudge") as pool:
            for index, row in enumerate(pool.map(execute, tasks), start=1):
                _append_checkpoint(checkpoint_path, row)
                if progress is not None:
                    progress(
                        f"CoSER rejudge {index}/{len(tasks)} {row['case_id']} {row['dimension']} {row['status']}"
                    )
    final_rows = _read_checkpoint_rows(checkpoint_path)
    summary = dict(
        summarize_coser_rejudge_rows(
            final_rows,
            selected_case_ids=resolved["selected_case_ids"],
        )
    )
    summary.update(
        {
            "schema_version": "1.0",
            "revision": COSER_REJUDGE_REVISION,
            "source_run_id": run_manifest["run_id"],
            "judge": pilot_manifest["judge"],
            "endpoint_concurrency": limiter.snapshot(),
            "artifacts": {
                "rejudge_manifest" if formal else "pilot_manifest": manifest_filename,
                "judge_records": "judge_records.jsonl",
                "summary": "summary.json",
                "report": "report.md",
            },
            "output_directory": str(output_dir),
        }
    )
    if formal and summary["status"] == "completed":
        summary["artifacts"].update(
            _materialize_corrected_results(
                resolved=resolved,
                output_dir=output_dir,
                rows=final_rows,
                judge=pilot_manifest["judge"],
            )
        )
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_text(output_dir / "report.md", _render_report(summary, pilot_manifest))
    return summary


def run_coser_rejudge(
    run_directory: str | Path,
    *,
    import_manifest: str | Path | None = None,
    runtime_config: str | Path | None = None,
    output_directory: str | Path | None = None,
    seed: int = DEFAULT_PILOT_SEED,
    max_workers: int = 4,
    validate_only: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Mapping[str, Any]:
    """Rejudge every completed CoSER scene and emit non-destructive corrected artifacts."""

    return run_coser_rejudge_pilot(
        run_directory,
        import_manifest=import_manifest,
        runtime_config=runtime_config,
        output_directory=output_directory,
        limit=None,
        seed=seed,
        max_workers=max_workers,
        validate_only=validate_only,
        progress=progress,
    )


__all__ = [
    "COSER_REJUDGE_REVISION",
    "DEFAULT_PILOT_LIMIT",
    "DEFAULT_PILOT_SEED",
    "OFFICIAL_PROMPT_REVISION",
    "build_official_coser_judge_request",
    "parse_official_coser_judge_response",
    "run_coser_rejudge",
    "run_coser_rejudge_pilot",
    "select_coser_pilot_cases",
    "summarize_coser_rejudge_rows",
    "validate_coser_rejudge_pilot",
]
