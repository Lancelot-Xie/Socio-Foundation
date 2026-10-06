"""Separate stable evaluation semantics from mutable support-role routing."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .contracts import CaseResult
from .errors import ConfigurationError
from .json_utils import canonical_json, jsonable


_SUPPORT_MODEL_KEYS = {
    "model",
    "model_revision",
    "backend",
    "profile",
    "base_url",
    "base_url_env",
    "api_key_env",
    "require_api_key",
    "token_limit_field",
    "send_sampling_params",
    "max_inflight_requests",
    "extra_body",
    "structured_output",
    "fallback",
    "fallback_reason",
    "judge_model",
    "judge_revision",
    "judge_models",
    "judge_revisions",
    "fixed_assistant_model",
    "fixed_assistant_revision",
    "partner_model",
    "partner_revision",
    "environment_model",
    "environment_revision",
    "next_speaker_model",
    "next_speaker_revision",
    "intent_judge_model",
    "intent_judge_revision",
    "shard_judge_model",
    "shard_judge_revision",
}

_MUTABLE_OPERATION_KEYS = {"request_timeout_seconds", "max_retries"}


def _without_keys(value: Any, excluded: set[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_keys(item, excluded)
            for key, item in value.items()
            if str(key) not in excluded
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_without_keys(item, excluded) for item in value]
    return value


def support_protocol_identity(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep support-role protocol settings while excluding mutable model routing."""

    return _without_keys(value, _SUPPORT_MODEL_KEYS)


def environment_protocol_identity(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Exclude failure-recovery knobs while preserving environment semantics."""

    return _without_keys(value, _MUTABLE_OPERATION_KEYS)


def role_execution_identity(role: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the non-secret model routing applied to one execution attempt."""

    identity = {
        key: jsonable(role.get(key))
        for key in (
            "backend",
            "profile",
            "base_url",
            "base_url_env",
            "api_key_env",
            "require_api_key",
            "model",
            "model_revision",
            "model_adapter",
            "judge_id",
            "generation",
            "token_limit_field",
            "send_sampling_params",
            "max_inflight_requests",
            "extra_body",
            "structured_output",
            "token_accounting",
            "reasoning_token_allowance",
        )
        if key in role
    }
    reason = str(role.get("fallback_reason") or "").strip()
    if reason:
        identity["fallback_reason"] = reason
    fallback = role.get("fallback")
    if isinstance(fallback, Mapping):
        fallback_identity = role_execution_identity(fallback)
        fallback_identity["trigger"] = fallback.get("trigger")
        identity["fallback"] = fallback_identity
    return jsonable(identity)


def attach_execution_provenance(
    result: CaseResult,
    *,
    evaluated_role: str,
    evaluated_role_identity: Mapping[str, Any],
    support_roles: Mapping[str, Mapping[str, Any]],
    api_calls: Sequence[Mapping[str, Any]] = (),
) -> CaseResult:
    """Attach the resolved role configuration to a result before checkpointing."""

    metadata = dict(result.metadata)
    execution_provenance = {
        "schema_version": "1.0",
        "evaluated_role": evaluated_role,
        "evaluated_role_identity": jsonable(evaluated_role_identity),
        "support_roles": {
            str(name): jsonable(identity)
            for name, identity in sorted(support_roles.items())
        },
        "api_calls": [jsonable(call) for call in api_calls],
    }
    response = result.model_response
    if response is not None and isinstance(response.raw, Mapping):
        api_execution = response.raw.get("_sim_eval")
        if isinstance(api_execution, Mapping):
            execution_provenance["last_model_response_api"] = jsonable(api_execution)
    metadata["execution_provenance"] = execution_provenance
    return replace(result, metadata=metadata)


def summarize_support_role_provenance(results: Sequence[CaseResult]) -> dict[str, Any]:
    """Summarize support-model mixtures across all supplied execution attempts."""

    role_models: dict[str, dict[str, dict[str, Any]]] = {}
    role_configurations: dict[str, dict[str, dict[str, Any]]] = {}
    fallback_models: dict[str, dict[str, dict[str, Any]]] = {}
    safety_fallback_call_count = 0
    unattributed = 0
    for result in results:
        execution = result.metadata.get("execution_provenance")
        if not isinstance(execution, Mapping):
            unattributed += 1
            continue
        support_roles = execution.get("support_roles")
        if not isinstance(support_roles, Mapping):
            unattributed += 1
            continue
        for role_name, raw_identity in support_roles.items():
            if not isinstance(raw_identity, Mapping):
                continue
            role = str(role_name)
            identity = dict(raw_identity)
            model_identity = {
                "model": identity.get("model"),
                "model_revision": identity.get("model_revision"),
            }
            model_key = canonical_json(model_identity)
            model_entry = role_models.setdefault(role, {}).setdefault(
                model_key,
                {**model_identity, "attempt_count": 0},
            )
            model_entry["attempt_count"] += 1
            configuration_key = canonical_json(identity)
            configuration_entry = role_configurations.setdefault(role, {}).setdefault(
                configuration_key,
                {"identity": jsonable(identity), "attempt_count": 0},
            )
            configuration_entry["attempt_count"] += 1
        history = result.metadata.get("judge_resume_history")
        api_calls = history[-1].get("api_calls") if history else execution.get("api_calls")
        if isinstance(api_calls, Sequence) and not isinstance(api_calls, (str, bytes)):
            for call in api_calls:
                if not isinstance(call, Mapping):
                    continue
                fallback = call.get("safety_fallback")
                if not isinstance(fallback, Mapping) or not fallback.get("used"):
                    continue
                role = str(call.get("route_role") or "")
                identity = fallback.get("fallback")
                if not role or not isinstance(identity, Mapping):
                    continue
                safety_fallback_call_count += 1
                model_identity = {
                    "model": identity.get("model"),
                    "model_revision": identity.get("model_revision"),
                }
                key = canonical_json(model_identity)
                entry = fallback_models.setdefault(role, {}).setdefault(
                    key,
                    {**model_identity, "call_count": 0},
                )
                entry["call_count"] += 1

    roles = {}
    for role in sorted(set(role_models) | set(role_configurations)):
        models = list(role_models.get(role, {}).values())
        configurations = list(role_configurations.get(role, {}).values())
        effective_fallback_models = list(fallback_models.get(role, {}).values())
        distinct_model_keys = {
            canonical_json({"model": item.get("model"), "model_revision": item.get("model_revision")})
            for item in (*models, *effective_fallback_models)
        }
        roles[role] = {
            "mixed_models": len(distinct_model_keys) > 1,
            "fallback_recorded": any(
                bool(entry["identity"].get("fallback_reason")) for entry in configurations
            ) or bool(effective_fallback_models),
            "models": models,
            "effective_fallback_models": effective_fallback_models,
            "attempt_configurations": configurations,
        }
    return {
        "mixed_models": any(entry["mixed_models"] for entry in roles.values()),
        "roles": roles,
        "unattributed_attempt_count": unattributed,
        "safety_fallback_call_count": safety_fallback_call_count,
    }


def summarize_evaluation_warnings(results: Sequence[CaseResult]) -> list[Mapping[str, Any]]:
    """Promote non-fatal judge/scorer failures that otherwise only appear inside case records."""

    warnings: list[Mapping[str, Any]] = []
    seen: set[str] = set()

    def visit(result: CaseResult, value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            handled_nested_error = False
            if "judge" in ".".join(path).casefold():
                error = value.get("error") if value.get("status") == "unavailable" else None
                handled_nested_error = isinstance(error, Mapping)
                candidate = error if isinstance(error, Mapping) else value
                if isinstance(candidate.get("kind"), str) and isinstance(candidate.get("message"), str):
                    warning = {
                        "case_id": result.case_id,
                        "repetition": result.repetition,
                        "path": ".".join(path),
                        "kind": str(candidate["kind"]),
                        "message": str(candidate["message"]),
                    }
                    key = canonical_json(warning)
                    if key not in seen:
                        seen.add(key)
                        warnings.append(warning)
            for key, item in value.items():
                if key == "judge_resume_history":
                    continue  # Historical repair failures are audit, not current warnings.
                if handled_nested_error and key == "error":
                    continue
                visit(result, item, (*path, str(key)))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                visit(result, item, (*path, str(index)))

    for result in results:
        visit(result, result.metadata, ("metadata",))
    return warnings


def validate_support_role_transition(
    previous_attempts: Sequence[CaseResult],
    current_support_roles: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require an explicit reason when a resumed attempt changes a support model."""

    previous_by_role: dict[str, Mapping[str, Any]] = {}
    for result in previous_attempts:
        execution = result.metadata.get("execution_provenance")
        support_roles = execution.get("support_roles") if isinstance(execution, Mapping) else None
        if not isinstance(support_roles, Mapping):
            continue
        for role_name, identity in support_roles.items():
            if isinstance(identity, Mapping):
                previous_by_role[str(role_name)] = identity

    missing_reasons = []
    for role_name, current in current_support_roles.items():
        previous = previous_by_role.get(str(role_name))
        if previous is None:
            continue
        previous_model = (previous.get("model"), previous.get("model_revision"))
        current_model = (current.get("model"), current.get("model_revision"))
        if previous_model != current_model and not str(current.get("fallback_reason") or "").strip():
            missing_reasons.append(str(role_name))
    if missing_reasons:
        raise ConfigurationError(
            "support model changed during resume; add a nonempty fallback_reason for roles "
            f"{sorted(missing_reasons)}"
        )


__all__ = [
    "attach_execution_provenance",
    "environment_protocol_identity",
    "role_execution_identity",
    "summarize_evaluation_warnings",
    "summarize_support_role_provenance",
    "support_protocol_identity",
    "validate_support_role_transition",
]
