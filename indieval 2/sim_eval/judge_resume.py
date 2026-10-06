"""Protocol-aware continuation of scoring on an immutable completed rollout.

The adapters still own their prompts, parsers and scoring formulas. This module
selects incomplete scoring work and merges its results without replacing the
original generation or an already available score.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping

from .contracts import CaseResult, ResultStatus
from .errors import ArtifactError, ResumeConflictError
from .json_utils import canonical_json, jsonable, sha256_digest


def _metric_values(result: CaseResult) -> dict[str, Any]:
    return {m.name: m.value for m in result.metrics}


def required_judge_metrics(adapter: Any, case: Any, result: CaseResult) -> tuple[str, ...]:
    """Return only metrics required by this case's scoring protocol."""
    if result.status != ResultStatus.COMPLETED or result.metadata.get("target_output_failure"):
        return ()
    bid = result.benchmark_id
    if bid == "sotopia":
        if adapter.provenance_for_case(case) is None:
            return ()
        from .benchmarks.sotopia import SOTOPIA_DIMENSIONS
        prefixes = [*(f"sotopia.agent.{a['id']}" for a in case.input_data["agents"]), "sotopia.diagnostic.mean_across_agents"]
        role_prefixes = []
        if case.input_data.get("evaluated_agent_id"):
            role_prefixes.append("sotopia.evaluated_agent")
            if case.input_data.get("partner_agent_id"):
                role_prefixes.append("sotopia.partner_agent")
        names = [f"{prefix}.{dimension}{suffix}" for prefix in prefixes + role_prefixes
                 for dimension in SOTOPIA_DIMENSIONS for suffix in ("", ".normalized")]
        return tuple(names + [f"{prefix}.normalized_dimension_mean" for prefix in role_prefixes])
    if bid == "coser":
        from .benchmarks.coser import COSER_DIMENSIONS
        if adapter.provenance_for_case(case) is None:
            return ()
        return tuple(f"coser.scene.{d}" for d in COSER_DIMENSIONS) + ("coser.scene.critic_average",)
    if bid == "humanual":
        from .benchmarks.humanual import STATE_NAMES
        if adapter.provenance_for_case(case) is None or result.metadata.get("scoring_shortcut"):
            return ()
        domain = str(case.input_data["domain"])
        suffixes = ("response_alignment", "state_alignment", *(f"state.{n}_alignment" for n in STATE_NAMES))
        return tuple(f"{prefix}.{s}" for prefix in ("humanual", f"humanual.domain.{domain}") for s in suffixes)
    if bid == "mirrorbench":
        from .benchmarks.mirrorbench import JUDGE_METRICS
        prov = adapter.provenance_for_case(case)
        if not prov.judge_model:
            return ()
        return tuple(f"mirrorbench.judge.{name}" + ("" if scope == "main" else f".{scope}_control")
                     for name in JUDGE_METRICS for scope in adapter._judge_scopes(name, prov))
    if bid == "userlm":
        variant = adapter.variant_for_case(case)
        prov = adapter.provenance_for_case(case)
        if variant == "intrinsic_intent_adherence" and prov.intent_judge_model:
            return ("userlm.intrinsic.intent_adherence",)
        if variant == "extrinsic_verifiable" and prov.shard_judge_model:
            from .benchmarks.userlm import _parse_shards
            names = ["intent_coverage", "repeat_required", "additional_demands"]
            if _parse_shards(case)[1]:
                names.append("skip_non_required")
            return tuple("userlm.extrinsic." + n for n in names)
    if bid == "behaviorchain":
        if adapter.task_mode(case) == "generation" and adapter.judge_provenance_for_case(case) is not None:
            return ("behaviorchain.node_score", "behaviorchain.generation.node_judge_score")
    if bid == "agentsense":
        prov = adapter.provenance_for_case(case)
        if prov is None or (result.metadata.get("judge_error") or {}).get("kind") == "token_budget_exhausted":
            return ()
        names = ("self_goal_completion", "other_goal_completion", "judge_average", "judge_majority",
                 *(f"judge.{j}" for j in prov.logical_judge_ids))
        state = adapter.reset_official_state(case, seed=0)
        return tuple(f"{prefix}.{name}" for prefix in ("agentsense.episode", *(f"agentsense.agent.{a}" for a in state.agents)) for name in names)
    return ()


def needs_judge_resume(adapter: Any, case: Any, result: CaseResult) -> bool:
    values = _metric_values(result)
    return any(values.get(name) is None for name in required_judge_metrics(adapter, case, result))


def restore_judge_state(state: Any, previous: CaseResult) -> None:
    """Restore evaluator-visible evidence; never execute a dialogue action."""
    state.terminal = True
    state.terminal_reason = previous.metadata.get("terminal_reason") or (previous.prediction or {}).get("terminal_reason")
    if hasattr(state, "transcript"):
        state.transcript = list(previous.trace)
        state.turn_count = int((previous.prediction or {}).get("turn_count", previous.metadata.get("turn_count", 0)))
    else:
        from .environments.dialogue import EVALUATED_USER_ROLE, FIXED_ASSISTANT_ROLE
        state.trace = list(previous.trace)
        state.user_turn_count = sum(e.actor == EVALUATED_USER_ROLE and e.kind == "message" for e in state.public_transcript)
        state.assistant_turn_count = sum(e.actor == FIXED_ASSISTANT_ROLE and e.kind == "message" for e in state.public_transcript)
        state.action_count = state.user_turn_count + state.assistant_turn_count


def mirror_saved_records(previous: CaseResult | None, name: str) -> Mapping[str, Any] | None:
    if previous is None:
        return None
    records = deepcopy(previous.metadata.get("mirrorbench", {}).get("judge_records", {}).get(name, {}))
    values = {m.name: m for m in previous.metrics}
    for scope in ("main", "hh", "pp"):
        key = f"mirrorbench.judge.{name}" + ("" if scope == "main" else f".{scope}_control")
        metric = values.get(key)
        if metric is not None and metric.value is not None:
            records[scope] = {**records.get(scope, {}), "status": "available", "score": metric.value,
                              "samples": metric.metadata.get("samples") or records.get(scope, {}).get("samples", [])}
    return records


def _judge_metric(name: str, bid: str) -> bool:
    if bid == "sotopia":
        return name.startswith("sotopia.")
    if bid == "humanual":
        return name.startswith("humanual.") and "embedding" not in name
    if bid == "coser":
        return name.startswith("coser.") and name not in {"coser.scene.bleu", "coser.scene.rouge_l"}
    if bid == "agentsense":
        return name.startswith("agentsense.") and "information" not in name
    if bid == "mirrorbench":
        return name.startswith("mirrorbench.judge.")
    if bid == "userlm":
        return name == "userlm.intrinsic.intent_adherence" or name in {
            "userlm.extrinsic." + n for n in ("intent_coverage", "repeat_required", "skip_non_required", "additional_demands")}
    return bid == "behaviorchain" and name in {"behaviorchain.node_score", "behaviorchain.generation.node_judge_score"}


def merge_judge_result(previous: CaseResult, scored: CaseResult, *, api_calls: Any = ()) -> CaseResult:
    if scored.status != ResultStatus.COMPLETED:
        raise ArtifactError("judge continuation must not replace a completed rollout with a failed rollout")
    bid = previous.benchmark_id
    metrics = {m.name: m for m in previous.metrics}
    changed = []
    for metric in scored.metrics:
        old = metrics.get(metric.name)
        if _judge_metric(metric.name, bid) and (old is None or old.value is None or metric.name.endswith(".availability_rate")):
            if old != metric:
                changed.append(metric.name)
            metrics[metric.name] = metric
    metadata = deepcopy(dict(previous.metadata))
    for key in ("judge_status", "judge_error", "judge_errors", "judge_records", "judge_payload", "goal_evaluation_cache"):
        if key in scored.metadata:
            metadata[key] = scored.metadata[key]
    for container, keys in {"mirrorbench": ("judge_records",), "behaviorchain": ("judge_record", "judge_error"), "userlm": ("shard_metrics",)}.items():
        if container in scored.metadata:
            nested = dict(metadata.get(container) or {})
            for key in keys:
                if key in scored.metadata[container]:
                    nested[key] = scored.metadata[container][key]
            metadata[container] = nested
    history = list(metadata.get("judge_resume_history") or [])
    history.append({"api_calls": list(api_calls), "token_usage": jsonable(scored.token_usage),
                    "latency_ms": scored.latency_ms, "updated_metrics": changed,
                    "request_audits": scored.metadata.get("request_audits", scored.metadata.get("context_audits", []))})
    metadata["judge_resume_history"] = history
    metadata["judge_resume"] = {"revision": "protocol-judge-resume-v1", "previous_digest": sha256_digest(previous.to_dict())}
    return replace(previous, metrics=tuple(metrics.values()), metadata=metadata)


def validate_judge_append(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Narrow exception to terminal checkpoint immutability, also used on read."""
    from .contracts import case_result_from_dict
    old = case_result_from_dict(previous)
    new = case_result_from_dict(current)
    marker = new.metadata.get("judge_resume") or {}
    if marker.get("revision") != "protocol-judge-resume-v1" or old.status != ResultStatus.COMPLETED or new.status != ResultStatus.COMPLETED or marker.get("previous_digest") != sha256_digest(old.to_dict()):
        raise ArtifactError("invalid judge continuation checkpoint link")
    for key in ("run_id", "benchmark_id", "case_id", "group_id", "repetition", "prediction", "trace", "model_response", "error", "token_usage", "latency_ms", "schema_version"):
        if canonical_json(getattr(old, key)) != canonical_json(getattr(new, key)):
            raise ArtifactError(f"judge continuation changed immutable rollout field {key}")
    updated = {m.name: m for m in new.metrics}
    for metric in old.metrics:
        replacement = updated.get(metric.name)
        mutable = _judge_metric(metric.name, old.benchmark_id) and (metric.value is None or metric.name.endswith(".availability_rate"))
        if replacement is None or (not mutable and canonical_json(metric) != canonical_json(replacement)):
            raise ArtifactError(f"judge continuation changed preserved metric {metric.name}")
    allowed_metadata = {"judge_status", "judge_error", "judge_errors", "judge_records", "judge_payload",
                        "goal_evaluation_cache", "judge_resume", "judge_resume_history", "judge_resume_episode_output_token_budget"}
    nested_keys = {"mirrorbench": {"judge_records"}, "behaviorchain": {"judge_record", "judge_error"}, "userlm": {"shard_metrics"}}
    for key in set(old.metadata) | set(new.metadata):
        if key in allowed_metadata:
            continue
        before, after = old.metadata.get(key), new.metadata.get(key)
        if key in nested_keys:
            before = {k: v for k, v in (before or {}).items() if k not in nested_keys[key]}
            after = {k: v for k, v in (after or {}).items() if k not in nested_keys[key]}
        if canonical_json(before) != canonical_json(after):
            raise ArtifactError(f"judge continuation changed preserved metadata {key}")
    old_history = list(old.metadata.get("judge_resume_history") or [])
    history = list(new.metadata.get("judge_resume_history") or [])
    if len(history) != len(old_history) + 1 or canonical_json(history[:-1]) != canonical_json(old_history):
        raise ArtifactError("judge continuation must append one audit entry")
    old_names = {m.name for m in old.metrics}
    if any(m.name not in old_names and not _judge_metric(m.name, old.benchmark_id) for m in new.metrics):
        raise ArtifactError("judge continuation added a non-judge metric")


def validate_judge_roles(previous: CaseResult, support_roles: Mapping[str, Any]) -> None:
    old_roles = previous.metadata.get("execution_provenance", {}).get("support_roles", {})
    # A support-model fallback for a new rollout must not silently mix judges
    # inside an already completed rollout. Operational concurrency may change.
    def scoring_identity(value: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in value.items() if k not in {"max_inflight", "fallback_reason"}}
    for role, old in old_roles.items():
        if role not in support_roles or canonical_json(scoring_identity(old)) != canonical_json(scoring_identity(support_roles[role])):
            raise ResumeConflictError(f"cannot resume judging with changed support role {role!r}")


def stored_judge_incomplete(record: Mapping[str, Any]) -> bool:
    """Select legacy suite children for the adapter's authoritative check.

    Old suite summaries called these children completed even when a scoring
    attempt explicitly failed. This hint never decides which metrics to judge.
    """
    if record.get("status") != "completed":
        return False
    metadata = record.get("metadata") or {}
    if metadata.get("target_output_failure") or (metadata.get("judge_error") or {}).get("kind") == "token_budget_exhausted":
        return False
    def unavailable(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return value == "unavailable"
        if value.get("status") == "unavailable":
            return True
        return any(unavailable(v) for v in value.values() if isinstance(v, (str, Mapping)))
    return bool(metadata.get("judge_error") or metadata.get("judge_errors")
                or unavailable(metadata.get("judge_status"))
                or unavailable(metadata.get("judge_records"))
                or unavailable(metadata.get("mirrorbench", {}).get("judge_records"))
                or metadata.get("behaviorchain", {}).get("judge_error"))
