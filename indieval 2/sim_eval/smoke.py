"""Resource-aware smoke and formal-suite orchestration.

This module is deliberately additive: normal ``evaluate`` calls keep their
existing seeded selection and runtime configuration.  The smoke suite builds
explicit dependency-group/case selections and materializes non-secret runtime
overlays in its own artifact directory.
"""

from __future__ import annotations

import copy
import hashlib
import tempfile
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from . import __version__
from .artifacts import atomic_write_json, atomic_write_text, load_json
from .backends.concurrency import EndpointLimiterRegistry
from .data.commands import _safe_output
from .data.loaders import load_import_spec, load_local_cases
from .errors import ConfigurationError, SimEvalError
from .external_runner import run_tau_usi_import, run_userlm_import
from .generic_runner import GENERIC_LIVE_BENCHMARK_IDS, run_generic_import
from .generation_policy import apply_benchmark_generation_policy
from .integrations.token_counting import default_token_accounting_config
from .json_utils import canonical_json, sha256_digest
from .runtime_config import (
    EVALUATED_ROLE_BY_BENCHMARK,
    apply_global_eval_model,
    load_config_document,
    normalize_global_eval_model,
)


DEFAULT_SMOKE_CONFIG = Path(__file__).resolve().parent / "resources" / "example_suite.yaml"
DEFAULT_FORMAL_CONFIG = Path(__file__).resolve().parent / "resources" / "example_suite.yaml"
SMOKE_ARTIFACT_SCHEMA = "1.0"
SUITE_MODES = frozenset({"smoke", "formal"})

_RESUME_ROUTE_IDENTITY_KEYS = (
    "backend",
    "profile",
    "model",
    "model_revision",
    "generation",
    "extra_body",
    "structured_output",
    "judge_id",
    "policy_revision",
    "send_sampling_params",
    "token_limit_field",
    "token_accounting",
    "reasoning_token_allowance",
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"smoke config {label} must be an object")
    return value


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ConfigurationError(f"smoke config requires {label}")
    return text


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"smoke config {label} must be a positive integer")
    return value


def _resume_route_identity(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep model/protocol semantics while allowing endpoint/concurrency recovery changes."""

    identity = {
        key: copy.deepcopy(value[key])
        for key in _RESUME_ROUTE_IDENTITY_KEYS
        if key in value
    }
    fallback = value.get("fallback")
    if isinstance(fallback, Mapping):
        identity["fallback"] = {
            "trigger": fallback.get("trigger"),
            **_resume_route_identity(fallback),
        }
    return identity


def _resume_runtime_identity(document: Mapping[str, Any]) -> Mapping[str, Any]:
    routes: dict[str, Any] = {}
    raw_roles = document.get("roles")
    if isinstance(raw_roles, Mapping):
        for name, raw in raw_roles.items():
            if isinstance(raw, Mapping):
                routes[str(name)] = _resume_route_identity(raw)
    for name in ("target_user", "fixed_assistant"):
        raw = document.get(name)
        if isinstance(raw, Mapping):
            routes[name] = _resume_route_identity(raw)
    return {
        "benchmark_id": document.get("benchmark_id"),
        "routes": routes,
        "routing": copy.deepcopy(document.get("routing")),
        # These resources and scoring inputs directly affect reported metrics
        # and may not be changed inside a compatible failed-only resume.
        "resources": copy.deepcopy(document.get("resources")),
        "scoring": copy.deepcopy(document.get("scoring")),
    }


def _resume_plan_identity(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    entries = []
    for raw in plan.get("entries") or ():
        if not isinstance(raw, Mapping):
            raise ConfigurationError("stored suite plan contains an invalid entry")
        entries.append(
            {
                "id": raw.get("id"),
                "benchmark_id": raw.get("benchmark_id"),
                "source_manifest_digest": raw.get("source_manifest_digest"),
                "selected_group_ids": copy.deepcopy(raw.get("selected_group_ids")),
                "selected_case_ids": copy.deepcopy(raw.get("selected_case_ids")),
                "selected_group_count": raw.get("selected_group_count"),
                "selected_case_count": raw.get("selected_case_count"),
            }
        )
    raw_model = plan.get("model")
    if not isinstance(raw_model, Mapping):
        raise ConfigurationError("suite plan model identity is missing")
    return {
        "schema_version": plan.get("schema_version"),
        "framework_version": plan.get("framework_version"),
        "suite_mode": plan.get("suite_mode"),
        "suite_id": plan.get("suite_id"),
        "coverage_revision": plan.get("coverage_revision"),
        "seed": plan.get("seed"),
        "model": _resume_route_identity(raw_model),
        "global_eval_model": plan.get("global_eval_model"),
        "entries": entries,
    }


def _compatible_failed_resume_entries(
    *,
    output: Path,
    plan: Mapping[str, Any],
    runtime_documents: Mapping[str, Mapping[str, Any]],
    plan_name: str,
    summary_name: str,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    """Validate a user-selected output directory and return only failed entries.

    Model identities, metric resources, data, selection and seed remain hard
    constraints.  Environment/prompt implementation and operational limits may
    change so a repaired failure can be retried without rerunning completed
    cases that never exercised the repaired branch.
    """

    existing_plan = load_json(output / plan_name)
    existing_summary = load_json(output / summary_name)
    if not isinstance(existing_plan, Mapping) or not isinstance(existing_summary, Mapping):
        raise ConfigurationError("compatible suite resume requires valid stored plan and summary artifacts")
    if canonical_json(_resume_plan_identity(existing_plan)) != canonical_json(
        _resume_plan_identity(plan)
    ):
        raise ConfigurationError(
            "refusing incompatible suite resume: evaluated/helper model, data, sample selection, "
            "seed, framework version, or suite membership changed; use a new --output directory"
        )

    old_entries_raw = existing_summary.get("entries")
    if not isinstance(old_entries_raw, Sequence) or isinstance(old_entries_raw, (str, bytes)):
        raise ConfigurationError("stored suite summary lacks entry results")
    old_entries = {
        str(item.get("id")): item
        for item in old_entries_raw
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    current_entries = {
        str(item["id"]): item
        for item in plan.get("entries") or ()
        if isinstance(item, Mapping)
    }
    if set(old_entries) != set(current_entries):
        raise ConfigurationError("compatible suite resume requires the same benchmark entry IDs")
    failed_ids = {
        entry_id
        for entry_id, item in old_entries.items()
        if item.get("status") != "completed"
    }

    # Older summaries could say completed while protocol judges had failed.
    # Let the child adapter inspect those checkpoints before reusing the entry.
    from .artifacts import CheckpointStore
    from .judge_resume import stored_judge_incomplete
    for entry_id, item in old_entries.items():
        if entry_id in failed_ids:
            continue
        records = item.get("records")
        if isinstance(records, str) and (output / records).is_file():
            store = CheckpointStore((output / records).parent)
            if any(stored_judge_incomplete(record) for record in store.iter_latest_record_dicts()):
                failed_ids.add(entry_id)

    stored_runtime_root = output / "runtime_configs"
    # Model/generation/resource/scoring identity is checked for every entry,
    # including entries that will be reused.  A same-directory repair must not
    # silently mix completed results from a different candidate or support
    # model condition.
    for entry_id in sorted(current_entries):
        stored_runtime_path = stored_runtime_root / f"{entry_id}.yaml"
        if not stored_runtime_path.is_file():
            raise ConfigurationError(
                f"compatible suite resume lacks stored runtime config for {entry_id!r}"
            )
        _stored_runtime_path, stored_runtime = load_config_document(stored_runtime_path)
        current_runtime = runtime_documents.get(entry_id)
        if not isinstance(current_runtime, Mapping):
            raise ConfigurationError(
                f"compatible suite resume lacks current runtime config for {entry_id!r}"
            )
        if canonical_json(_resume_runtime_identity(stored_runtime)) != canonical_json(
            _resume_runtime_identity(current_runtime)
        ):
            raise ConfigurationError(
                f"refusing compatible resume for {entry_id!r}: evaluated/helper model, generation, "
                "metric resource, or scoring configuration changed"
            )

    repair_entries = [
        current_entries[str(item["id"])]
        for item in plan.get("entries") or ()
        if isinstance(item, Mapping) and str(item["id"]) in failed_ids
    ]
    return repair_entries, existing_plan, existing_summary


def _resolve_path(value: Any, *, config_path: Path, label: str) -> Path:
    path = Path(_required_text(value, label))
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _case_cost(case: Any) -> int:
    return len(canonical_json(case.input_data))


def _group_cases(cases: Sequence[Any]) -> Mapping[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for case in cases:
        grouped[case.group_id].append(case)
    return {
        group_id: tuple(sorted(items, key=lambda case: case.case_id))
        for group_id, items in grouped.items()
    }


def _strata(case: Any) -> Mapping[str, Any]:
    value = case.metadata.get("strata")
    return value if isinstance(value, Mapping) else {}


def _has_nonrequired_information_shard(case: Any) -> bool:
    raw = case.input_data.get(
        "information_shards",
        case.input_data.get("required_information"),
    )
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return False
    return any(
        isinstance(item, Mapping) and not bool(item.get("required", True))
        for item in raw
    )


def _complete_fantom_set_scenario(items: Sequence[Any]) -> str | None:
    contracts = [case.input_data.get("set_contract") for case in items]
    if not contracts or not all(isinstance(value, Mapping) for value in contracts):
        return None
    if len({canonical_json(value) for value in contracts}) != 1:
        return None
    contract = contracts[0]
    members = contract.get("members")
    if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
        return None
    expected = {
        str(member.get("id"))
        for member in members
        if isinstance(member, Mapping) and not bool(member.get("short_no_long_excluded"))
    }
    observed = {
        str(case.input_data.get("question_member_id"))
        for case in items
        if not bool(case.input_data.get("short_no_long_excluded"))
    }
    scenarios = {
        str(case.input_data.get("scenario"))
        for case in items
        if case.input_data.get("question_family") != "fact"
        and not bool(case.input_data.get("short_no_long_excluded"))
    }
    if expected != observed or len(scenarios) != 1:
        return None
    return next(iter(scenarios))


def _coverage_contract(
    entry_id: str,
    benchmark_id: str,
    grouped: Mapping[str, tuple[Any, ...]],
) -> tuple[dict[str, int], Mapping[str, frozenset[str]]]:
    tokens: dict[str, frozenset[str]] = {}

    if benchmark_id == "fantom":
        observed = {
            f"question:{_strata(case).get('question_family')}:{_strata(case).get('answer_format')}"
            for items in grouped.values()
            for case in items
        }
        complete_scenarios = {
            f"complete_set:{scenario}"
            for items in grouped.values()
            if (scenario := _complete_fantom_set_scenario(items)) is not None
        }
        for group_id, items in grouped.items():
            group_tokens = {
                f"question:{_strata(case).get('question_family')}:{_strata(case).get('answer_format')}"
                for case in items
            }
            if (scenario := _complete_fantom_set_scenario(items)) is not None:
                group_tokens.add(f"complete_set:{scenario}")
            tokens[group_id] = frozenset(group_tokens)
        return {token: 1 for token in sorted(observed | complete_scenarios)}, tokens

    if benchmark_id == "social_r1":
        observed = {
            f"option_count:{len(case.input_data.get('options') or ())}"
            for items in grouped.values()
            for case in items
        }
        for group_id, items in grouped.items():
            tokens[group_id] = frozenset(
                f"option_count:{len(case.input_data.get('options') or ())}" for case in items
            )
        return {token: 1 for token in sorted(observed)}, tokens

    if benchmark_id == "lifechoices":
        observed = {
            f"context:{_strata(case).get('context_condition')}"
            for items in grouped.values()
            for case in items
        }
        for group_id, items in grouped.items():
            tokens[group_id] = frozenset(
                f"context:{_strata(case).get('context_condition')}" for case in items
            )
        return {token: 1 for token in sorted(observed)}, tokens

    if benchmark_id == "behaviorchain":
        observed = {
            f"mode:{_strata(case).get('task_mode')}"
            for items in grouped.values()
            for case in items
        } | {
            f"key_status:{_strata(case).get('key_behavior_status')}"
            for items in grouped.values()
            for case in items
        }
        for group_id, items in grouped.items():
            tokens[group_id] = frozenset(
                {
                    *(f"mode:{_strata(case).get('task_mode')}" for case in items),
                    *(f"key_status:{_strata(case).get('key_behavior_status')}" for case in items),
                }
            )
        return {token: 1 for token in sorted(observed)}, tokens

    if benchmark_id == "alignx":
        observed = {
            f"variant:{case.input_data.get('variant')}"
            for items in grouped.values()
            for case in items
        }
        for group_id, items in grouped.items():
            tokens[group_id] = frozenset(
                f"variant:{case.input_data.get('variant')}" for case in items
            )
        return {token: 1 for token in sorted(observed)}, tokens

    if benchmark_id == "humanual":
        observed = {
            f"domain:{_strata(case).get('domain')}"
            for items in grouped.values()
            for case in items
        }
        requirements = {token: 1 for token in sorted(observed)}
        for group_id, items in grouped.items():
            tokens[group_id] = frozenset(
                f"domain:{_strata(case).get('domain')}" for case in items
            )
        return requirements, tokens

    if entry_id == "userlm_section3":
        requirements = {
            "variant:intrinsic_prism_group": 2,
            "variant:intrinsic_role_adherence": 1,
            "variant:intrinsic_intent_adherence": 1,
        }
        for group_id, items in grouped.items():
            variants = {str(case.input_data.get("variant")) for case in items}
            group_tokens: set[str] = set()
            if "intrinsic_prism" in variants and any(int(case.input_data.get("turn", -1)) == 0 for case in items):
                group_tokens.add("variant:intrinsic_prism_group")
            if "intrinsic_role_adherence" in variants:
                group_tokens.add("variant:intrinsic_role_adherence")
            if "intrinsic_intent_adherence" in variants:
                group_tokens.add("variant:intrinsic_intent_adherence")
            tokens[group_id] = frozenset(group_tokens)
        return requirements, tokens

    if entry_id == "userlm_lic":
        for group_id, items in grouped.items():
            group_tokens = {
                "task:math"
                for case in items
                if _strata(case).get("source_task") == "math"
            }
            if any(_has_nonrequired_information_shard(case) for case in items):
                group_tokens.add("metric:skip_non_required")
            tokens[group_id] = frozenset(group_tokens)
        return {"task:math": 1, "metric:skip_non_required": 1}, tokens

    if benchmark_id == "tau_usi":
        observed = {
            f"domain:{_strata(case).get('domain')}"
            for items in grouped.values()
            for case in items
        }
        for group_id, items in grouped.items():
            tokens[group_id] = frozenset(
                f"domain:{_strata(case).get('domain')}" for case in items
            )
        return {token: 1 for token in sorted(observed)}, tokens

    if benchmark_id == "coser":
        observed = {
            f"domain_status:{_strata(case).get('in_domain_status')}"
            for items in grouped.values()
            for case in items
        }
        for group_id, items in grouped.items():
            tokens[group_id] = frozenset(
                f"domain_status:{_strata(case).get('in_domain_status')}" for case in items
            )
        return {token: 1 for token in sorted(observed)}, tokens

    if benchmark_id == "agentsense":
        requirements = {"goal_metrics": 1, "paired_profiles": 1, "private_information_metrics": 1}
        for group_id, items in grouped.items():
            profile_ids = {str(case.metadata.get("profile_id") or case.case_id) for case in items}
            numeric_profiles = sum(bool(case.input_data.get("information_questions")) for case in items)
            group_tokens = {"goal_metrics"}
            if len(profile_ids) >= 2:
                group_tokens.add("paired_profiles")
            if numeric_profiles >= 2:
                group_tokens.add("private_information_metrics")
            tokens[group_id] = frozenset(group_tokens)
        return requirements, tokens

    for group_id in grouped:
        tokens[group_id] = frozenset({"metric_family:base"})
    return {"metric_family:base": 1}, tokens


def _select_multicover(
    grouped: Mapping[str, tuple[Any, ...]],
    requirements: Mapping[str, int],
    tokens_by_group: Mapping[str, frozenset[str]],
    *,
    seed: int,
    max_cases: int,
) -> tuple[tuple[str, ...], Mapping[str, int], tuple[str, ...]]:
    names = tuple(sorted(requirements))
    targets = tuple(int(requirements[name]) for name in names)
    zero = tuple(0 for _ in names)
    # state -> (score, selected_group_ids).  Score minimizes row count first,
    # then serialized input size and group count, with a seeded stable tie-break.
    states: dict[tuple[int, ...], tuple[tuple[Any, ...], tuple[str, ...]]] = {
        zero: ((0, 0, 0, ()), ())
    }
    ordered_groups = sorted(
        grouped,
        key=lambda group_id: hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest(),
    )
    for group_id in ordered_groups:
        items = grouped[group_id]
        group_cost = len(items)
        if group_cost > max_cases:
            continue
        input_cost = sum(_case_cost(case) for case in items)
        group_tokens = tokens_by_group.get(group_id, frozenset())
        tie = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
        updates = dict(states)
        for state, (score, chosen) in states.items():
            if int(score[0]) + group_cost > max_cases:
                continue
            next_state = tuple(
                min(target, count + int(name in group_tokens))
                for name, target, count in zip(names, targets, state)
            )
            next_chosen = (*chosen, group_id)
            next_score = (
                int(score[0]) + group_cost,
                int(score[1]) + input_cost,
                int(score[2]) + 1,
                (*score[3], tie),
            )
            previous = updates.get(next_state)
            if previous is None or next_score < previous[0]:
                updates[next_state] = (next_score, next_chosen)
        states = updates

    if targets in states:
        selected = states[targets][1]
        achieved_state = targets
    else:
        achieved_state, (_score, selected) = min(
            states.items(),
            key=lambda item: (
                -sum(
                    count / target if target else 1.0
                    for count, target in zip(item[0], targets)
                ),
                item[1][0],
            ),
        )
    achieved = {name: count for name, count in zip(names, achieved_state)}
    missing = tuple(
        name for name, count, target in zip(names, achieved_state, targets) if count < target
    )
    return tuple(selected), achieved, missing


def _fill_selection_to_case_cap(
    grouped: Mapping[str, tuple[Any, ...]],
    selected_group_ids: Sequence[str],
    *,
    seed: int,
    max_cases: int,
) -> tuple[str, ...]:
    """Deterministically add complete groups after coverage is already met."""

    selected = list(selected_group_ids)
    selected_set = set(selected)
    selected_case_count = sum(len(grouped[group_id]) for group_id in selected)
    remaining = sorted(
        (group_id for group_id in grouped if group_id not in selected_set),
        key=lambda group_id: hashlib.sha256(
            f"{seed}:{group_id}:coverage-fill-v1".encode("utf-8")
        ).hexdigest(),
    )
    for group_id in remaining:
        group_size = len(grouped[group_id])
        if selected_case_count + group_size > max_cases:
            continue
        selected.append(group_id)
        selected_case_count += group_size
        if selected_case_count == max_cases:
            break
    return tuple(selected)


def _normalized_suite_execution(execution: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(execution))
    max_active = _positive_int(
        execution.get("max_active_benchmarks", 1),
        "execution.max_active_benchmarks",
    )
    raw_limits = execution.get("benchmark_class_limits", {})
    if not isinstance(raw_limits, Mapping):
        raise ConfigurationError("execution.benchmark_class_limits must be an object")
    class_limits: dict[str, int] = {}
    for raw_name, raw_value in raw_limits.items():
        name = _required_text(raw_name, "execution.benchmark_class_limits key")
        limit = _positive_int(
            raw_value,
            f"execution.benchmark_class_limits.{name}",
        )
        if limit > max_active:
            raise ConfigurationError(
                f"execution.benchmark_class_limits.{name} cannot exceed "
                "execution.max_active_benchmarks"
            )
        class_limits[name] = limit
    result["max_active_benchmarks"] = max_active
    result["benchmark_class_limits"] = class_limits
    return result


def _deepseek_role(
    raw: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    judge_id: str | None = None,
    preserve_raw_max_tokens: bool = False,
    evaluated_role: bool = False,
) -> dict[str, Any]:
    result = {
        "backend": _required_text(model.get("backend"), "model.backend"),
        "profile": _required_text(model.get("profile"), "model.profile"),
        "model": _required_text(model.get("model"), "model.model"),
        "model_revision": _required_text(model.get("model_revision"), "model.model_revision"),
    }
    for key in (
        "base_url",
        "base_url_env",
        "api_key_env",
        "require_api_key",
        "token_limit_field",
        "send_sampling_params",
        "max_inflight_requests",
        "reasoning_token_allowance",
    ):
        if key in model:
            result[key] = copy.deepcopy(model[key])
    if evaluated_role and model.get("model_adapter") is not None:
        result["model_adapter"] = model["model_adapter"]
    if evaluated_role:
        result["token_accounting"] = copy.deepcopy(
            model.get("token_accounting") or default_token_accounting_config()
        )
    raw_generation = raw.get("generation")
    generation: dict[str, Any] = (
        copy.deepcopy(dict(raw_generation))
        if isinstance(raw_generation, Mapping)
        else {}
    )
    smoke_generation = model.get("generation")
    if isinstance(smoke_generation, Mapping):
        generation.update(copy.deepcopy(dict(smoke_generation)))
    if preserve_raw_max_tokens and isinstance(raw_generation, Mapping):
        raw_max_tokens = raw_generation.get("max_tokens")
        if raw_max_tokens is not None:
            generation["max_tokens"] = _positive_int(
                raw_max_tokens,
                "benchmark role generation.max_tokens",
            )
        else:
            # Some protocols (notably UserLM) declare different limits per
            # request.  Do not let the shared smoke-model profile erase those
            # adapter-level values by injecting its generic max_tokens.
            generation.pop("max_tokens", None)
    if generation:
        result["generation"] = generation
    for key in ("extra_body", "structured_output"):
        value = model.get(key)
        if isinstance(value, Mapping):
            result[key] = copy.deepcopy(dict(value))
    if judge_id is not None:
        result["judge_id"] = judge_id
    return result


def _smoke_max_workers(
    execution_options: Mapping[str, Any],
    *,
    benchmark_id: str,
) -> int:
    by_benchmark = execution_options.get("max_workers_by_benchmark", {})
    if not isinstance(by_benchmark, Mapping):
        raise ConfigurationError("execution.max_workers_by_benchmark must be an object")
    value = by_benchmark.get(benchmark_id, execution_options.get("max_workers"))
    return _positive_int(value, f"execution.max_workers[{benchmark_id}]")


def _absolute_tau_paths(document: dict[str, Any], *, base_path: Path) -> None:
    scoring = document.get("scoring")
    if not isinstance(scoring, Mapping):
        return
    resolved = dict(scoring)
    for key in ("annotation_local_path", "difficulty_local_path"):
        value = resolved.get(key)
        if value:
            path = Path(str(value))
            resolved[key] = str(path.resolve() if path.is_absolute() else (base_path.parent / path).resolve())
    document["scoring"] = resolved


def _smoke_runtime_document(
    base_path: Path,
    *,
    benchmark_id: str,
    suite_id: str,
    model: Mapping[str, Any],
    execution_options: Mapping[str, Any],
    global_eval_model: str | None = None,
) -> dict[str, Any]:
    _, raw = load_config_document(base_path)
    document = copy.deepcopy(dict(raw))
    local_device = _required_text(execution_options.get("local_device"), "execution.local_device")

    if benchmark_id == "tau_usi":
        if global_eval_model is None:
            document["target_user"] = _deepseek_role(
                _mapping(document.get("target_user"), "target_user"),
                model=model,
                preserve_raw_max_tokens=True,
                evaluated_role=True,
            )
            old_assistant = _mapping(document.get("fixed_assistant"), "fixed_assistant")
            assistant = _deepseek_role(
                old_assistant,
                model=model,
                preserve_raw_max_tokens=True,
            )
            if old_assistant.get("policy_revision"):
                assistant["policy_revision"] = old_assistant["policy_revision"]
            document["fixed_assistant"] = assistant
        else:
            document["target_user"] = _deepseek_role(
                _mapping(document.get("target_user"), "target_user"),
                model=model,
                preserve_raw_max_tokens=True,
                evaluated_role=True,
            )
            document = apply_global_eval_model(document, global_eval_model)
        limits = dict(_mapping(document.get("limits"), "limits"))
        limits["max_workers"] = _smoke_max_workers(
            execution_options,
            benchmark_id=benchmark_id,
        )
        document["limits"] = limits
        _absolute_tau_paths(document, base_path=base_path)
        return apply_benchmark_generation_policy(document)

    if global_eval_model is None:
        roles = _mapping(document.get("roles"), "roles")
        transformed = {}
        evaluated_role = EVALUATED_ROLE_BY_BENCHMARK.get(benchmark_id)
        for role_name, value in roles.items():
            logical_id = None
            if benchmark_id == "agentsense" and str(role_name).startswith("judge_"):
                logical_id = f"deepseek_{role_name}"
            transformed[str(role_name)] = _deepseek_role(
                _mapping(value, f"roles.{role_name}"),
                model=model,
                judge_id=logical_id,
                preserve_raw_max_tokens=str(role_name) == evaluated_role,
                evaluated_role=str(role_name) == evaluated_role,
            )
        document["roles"] = transformed
    else:
        evaluated_role = EVALUATED_ROLE_BY_BENCHMARK.get(benchmark_id)
        if evaluated_role is None:
            raise ConfigurationError(
                f"smoke global_eval_model does not support benchmark {benchmark_id!r}"
            )
        roles = dict(_mapping(document.get("roles"), "roles"))
        roles[evaluated_role] = _deepseek_role(
            _mapping(roles.get(evaluated_role), f"roles.{evaluated_role}"),
            model=model,
            preserve_raw_max_tokens=True,
            evaluated_role=True,
        )
        document["roles"] = roles
        document = apply_global_eval_model(document, global_eval_model)

    execution = dict(_mapping(document.get("execution", {}), "execution"))
    if execution:
        execution["max_workers"] = _smoke_max_workers(
            execution_options,
            benchmark_id=benchmark_id,
        )
        execution["profile"] = suite_id
        document["execution"] = execution

    resources = document.get("resources")
    if isinstance(resources, Mapping):
        resources = copy.deepcopy(dict(resources))
        for semantic_key in ("belief_semantic_backend", "embedding_semantic_backend"):
            semantic = resources.get(semantic_key)
            if isinstance(semantic, Mapping):
                semantic = dict(semantic)
                semantic["device"] = local_device
                resources[semantic_key] = semantic
        detector = resources.get("ai_text_detector")
        if isinstance(detector, Mapping):
            detector = dict(detector)
            detector["device"] = local_device
            resources["ai_text_detector"] = detector
        if benchmark_id == "userlm":
            resources["lic_repetitions"] = _positive_int(
                execution_options.get("userlm_lic_repetitions"),
                "execution.userlm_lic_repetitions",
            )
        document["resources"] = resources
    return apply_benchmark_generation_policy(document)


def _apply_entry_thinking_override(
    document: Mapping[str, Any], entry: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Override only this entry's evaluated vLLM role; preserve support roles."""
    if "enable_thinking" not in entry:
        return document
    enabled = entry["enable_thinking"]
    if not isinstance(enabled, bool):
        raise ConfigurationError("entries[].enable_thinking must be boolean")
    result = copy.deepcopy(dict(document))
    benchmark_id = str(result.get("benchmark_id"))
    role_name = EVALUATED_ROLE_BY_BENCHMARK.get(benchmark_id)
    if role_name is None:
        raise ConfigurationError(f"entry thinking override is unsupported for {benchmark_id!r}")
    roles = result if benchmark_id == "tau_usi" else result["roles"]
    role = roles[role_name]
    if role.get("profile", role.get("backend")) != "vllm":
        raise ConfigurationError("entries[].enable_thinking requires an evaluated vLLM role")
    extra = dict(role.get("extra_body") or {})
    extra["chat_template_kwargs"] = {
        **dict(extra.get("chat_template_kwargs") or {}), "enable_thinking": enabled,
    }
    if not enabled:
        # Remove inherited thinking-only fields when the suite model defaults on.
        extra.pop("thinking_token_budget", None)
        extra.pop("include_reasoning", None)
    role["extra_body"] = extra
    return apply_benchmark_generation_policy(result)


def _selected_entries(raw: Sequence[Any], selected: Sequence[str] | None) -> list[Mapping[str, Any]]:
    entries = [_mapping(value, "entries[]") for value in raw]
    ids = [_required_text(entry.get("id"), "entries[].id") for entry in entries]
    if len(ids) != len(set(ids)):
        raise ConfigurationError("smoke entry IDs must be unique")
    if selected is None:
        return entries
    requested = list(dict.fromkeys(str(value) for value in selected))
    unknown = sorted(set(requested) - set(ids))
    if unknown:
        raise ConfigurationError(f"unknown smoke entry IDs: {unknown}")
    by_id = dict(zip(ids, entries))
    return [by_id[entry_id] for entry_id in requested]


def _prepare_smoke(
    config_path: str | Path,
    *,
    selected_entries: Sequence[str] | None = None,
    global_eval_model: str | None = None,
) -> tuple[dict[str, Any], Mapping[str, Mapping[str, Any]]]:
    resolved_path, raw = load_config_document(config_path)
    if raw.get("schema_version") != "1.0":
        raise ConfigurationError("smoke config schema_version must be 1.0")
    suite_mode = str(raw.get("suite_mode") or "smoke").strip().casefold()
    if suite_mode not in SUITE_MODES:
        raise ConfigurationError(
            f"suite_mode must be one of {sorted(SUITE_MODES)}"
        )
    if global_eval_model is None:
        global_eval_model = raw.get("global_eval_model")
    global_eval_model = normalize_global_eval_model(global_eval_model)
    suite_id = _required_text(raw.get("suite_id"), "suite_id")
    coverage_revision = _required_text(raw.get("coverage_revision"), "coverage_revision")
    seed = _positive_int(raw.get("seed"), "seed")
    max_cases = None
    if suite_mode == "smoke":
        max_cases = _positive_int(raw.get("max_cases_per_entry"), "max_cases_per_entry")
        if max_cases > 9:
            raise ConfigurationError("smoke max_cases_per_entry must remain a single-digit number")
    model = _mapping(raw.get("model"), "model")
    execution = _normalized_suite_execution(
        _mapping(raw.get("execution"), "execution")
    )
    catalog = _resolve_path(raw.get("catalog"), config_path=resolved_path, label="catalog")
    raw_entries = raw.get("entries")
    if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, Sequence) or not raw_entries:
        raise ConfigurationError("smoke config entries must be a nonempty array")

    plan_entries = []
    runtime_documents: dict[str, Mapping[str, Any]] = {}
    for raw_entry in _selected_entries(raw_entries, selected_entries):
        entry_id = _required_text(raw_entry.get("id"), "entries[].id")
        benchmark_id = _required_text(raw_entry.get("benchmark_id"), f"entries.{entry_id}.benchmark_id")
        manifest_path = _resolve_path(
            raw_entry.get("manifest"), config_path=resolved_path, label=f"entries.{entry_id}.manifest"
        )
        runtime_path = _resolve_path(
            raw_entry.get("runtime_config"),
            config_path=resolved_path,
            label=f"entries.{entry_id}.runtime_config",
        )
        spec = load_import_spec(manifest_path)
        if spec.benchmark_id != benchmark_id:
            raise ConfigurationError(
                f"smoke entry {entry_id!r} benchmark_id does not match its import manifest"
            )
        cases, source_manifest = load_local_cases(spec)
        grouped = _group_cases(cases)
        requirements, tokens_by_group = _coverage_contract(entry_id, benchmark_id, grouped)
        if suite_mode == "formal":
            entry_max_cases = len(cases)
            selected_group_ids = tuple(sorted(grouped))
            achieved = {
                token: sum(
                    token in tokens_by_group.get(group_id, frozenset())
                    for group_id in selected_group_ids
                )
                for token in requirements
            }
            missing = tuple(
                token
                for token, required in requirements.items()
                if achieved.get(token, 0) < required
            )
            fill_to_case_cap = False
            selection_kind = "full_import_all_dependency_groups"
        else:
            assert max_cases is not None
            entry_max_cases = _positive_int(
                raw_entry.get("max_cases", max_cases),
                f"entries.{entry_id}.max_cases",
            )
            if entry_max_cases > 99:
                raise ConfigurationError(f"smoke entry {entry_id!r} max_cases must be at most 99")
            selected_group_ids, achieved, missing = _select_multicover(
                grouped,
                requirements,
                tokens_by_group,
                seed=seed,
                max_cases=entry_max_cases,
            )
            fill_to_case_cap = raw_entry.get("fill_to_case_cap", False)
            if not isinstance(fill_to_case_cap, bool):
                raise ConfigurationError(
                    f"smoke entry {entry_id!r} fill_to_case_cap must be boolean"
                )
            if fill_to_case_cap:
                selected_group_ids = _fill_selection_to_case_cap(
                    grouped,
                    selected_group_ids,
                    seed=seed,
                    max_cases=entry_max_cases,
                )
            selection_kind = (
                "explicit_dependency_groups"
                if benchmark_id in GENERIC_LIVE_BENCHMARK_IDS
                else "explicit_cases_from_complete_groups"
            ) + ("_then_seeded_fill_to_case_cap" if fill_to_case_cap else "")
        selected_cases = tuple(
            case
            for group_id in selected_group_ids
            for case in grouped[group_id]
        )
        if not selected_cases:
            raise ConfigurationError(f"smoke entry {entry_id!r} selected no cases")
        if len(selected_cases) > entry_max_cases:
            raise ConfigurationError(
                f"smoke entry {entry_id!r} selected {len(selected_cases)} cases, cap={entry_max_cases}"
            )
        runtime_document = _smoke_runtime_document(
            runtime_path,
            benchmark_id=benchmark_id,
            suite_id=suite_id,
            model=model,
            execution_options=execution,
            global_eval_model=global_eval_model,
        )
        runtime_document = _apply_entry_thinking_override(runtime_document, raw_entry)
        runtime_documents[entry_id] = runtime_document
        plan_entries.append(
            {
                "id": entry_id,
                "benchmark_id": benchmark_id,
                **({"evaluated_enable_thinking": raw_entry["enable_thinking"]}
                   if "enable_thinking" in raw_entry else {}),
                "manifest": str(manifest_path),
                "base_runtime_config": str(runtime_path),
                "source_manifest_digest": source_manifest.digest,
                "source_population": len(cases),
                "source_group_count": len(grouped),
                "selection_kind": selection_kind,
                "fill_to_case_cap": fill_to_case_cap,
                "concurrency_class": _required_text(
                    raw_entry.get("concurrency_class", "default"),
                    f"entries.{entry_id}.concurrency_class",
                ),
                "selected_group_ids": list(selected_group_ids),
                "selected_case_ids": [case.case_id for case in selected_cases],
                "selected_group_count": len(selected_group_ids),
                "selected_case_count": len(selected_cases),
                "case_cap": entry_max_cases,
                "coverage_requirements": dict(requirements),
                "coverage_achieved": dict(achieved),
                "uncovered_requirements": list(missing),
                "coverage_status": "complete" if not missing else "partial",
                "runtime_document_digest": sha256_digest(runtime_document),
            }
        )

    identity = {
        "schema_version": SMOKE_ARTIFACT_SCHEMA,
        "framework_version": __version__,
        "suite_mode": suite_mode,
        "suite_id": suite_id,
        "coverage_revision": coverage_revision,
        "seed": seed,
        "max_cases_per_entry": max_cases,
        "model": dict(model),
        "global_eval_model": global_eval_model,
        "evaluated_model_source": (
            f"{suite_mode}.model_applied_to_every_network_role"
            if global_eval_model is None
            else f"{suite_mode}.model_applied_to_evaluated_role_only"
        ),
        "execution": dict(execution),
        "catalog": str(catalog),
        "entries": plan_entries,
    }
    plan = {**identity, "plan_digest": sha256_digest(identity), "network_calls": 0}
    return plan, runtime_documents


def build_smoke_plan(
    config_path: str | Path = DEFAULT_SMOKE_CONFIG,
    *,
    selected_entries: Sequence[str] | None = None,
    global_eval_model: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic coverage plan without creating a backend."""

    plan, _ = _prepare_smoke(
        config_path,
        selected_entries=selected_entries,
        global_eval_model=global_eval_model,
    )
    return plan


def _write_runtime(path: Path, document: Mapping[str, Any]) -> None:
    atomic_write_text(path, yaml.safe_dump(dict(document), allow_unicode=True, sort_keys=False))


def _run_entry(
    entry: Mapping[str, Any],
    *,
    runtime_path: Path,
    output: Path,
    catalog_path: str,
    seed: int,
    validate_only: bool,
    limiter_registry: EndpointLimiterRegistry | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    common = {
        "manifest_path": str(entry["manifest"]),
        "runtime_config_path": runtime_path,
        "output_directory": output,
        "catalog_path": catalog_path,
        "seed": seed,
        "validate_only": validate_only,
        "limiter_registry": limiter_registry,
        "progress": progress,
    }
    benchmark_id = str(entry["benchmark_id"])
    full_import = entry.get("selection_kind") == "full_import_all_dependency_groups"
    if benchmark_id in GENERIC_LIVE_BENCHMARK_IDS:
        return run_generic_import(
            **common,
            selected_group_ids=(
                None
                if full_import
                else tuple(str(value) for value in entry["selected_group_ids"])
            ),
        )
    if benchmark_id == "tau_usi":
        return run_tau_usi_import(
            **common,
            selected_case_ids=(
                None
                if full_import
                else tuple(str(value) for value in entry["selected_case_ids"])
            ),
        )
    if benchmark_id == "userlm":
        return run_userlm_import(
            **common,
            selected_case_ids=(
                None
                if full_import
                else tuple(str(value) for value in entry["selected_case_ids"])
            ),
        )
    raise ConfigurationError(f"smoke runner does not support benchmark {benchmark_id!r}")


def _validate_plan(
    plan: Mapping[str, Any],
    runtime_documents: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    validations = []
    with tempfile.TemporaryDirectory(prefix="sim-eval-smoke-") as temporary:
        temp_root = Path(temporary)
        for entry in plan["entries"]:
            entry_id = str(entry["id"])
            runtime_path = temp_root / f"{entry_id}.yaml"
            _write_runtime(runtime_path, runtime_documents[entry_id])
            validations.append(
                {
                    "id": entry_id,
                    **_run_entry(
                        entry,
                        runtime_path=runtime_path,
                        output=temp_root / "unused" / entry_id,
                        catalog_path=str(plan["catalog"]),
                        seed=int(plan["seed"]),
                        validate_only=True,
                    ),
                }
            )
    return validations


def _metric_availability(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        return {"covered": [], "executed_but_unavailable": []}
    covered = []
    unavailable = []
    for name, raw in metrics.items():
        value = raw.get("value") if isinstance(raw, Mapping) else None
        (covered if value is not None else unavailable).append(str(name))
    return {
        "covered": sorted(covered),
        "executed_but_unavailable": sorted(unavailable),
    }


def _execute_smoke_entry(
    entry: Mapping[str, Any],
    *,
    runtime_path: Path,
    output: Path,
    entry_root: Path,
    catalog_path: str,
    seed: int,
    limiter_registry: EndpointLimiterRegistry,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    entry_id = str(entry["id"])
    started_at = time.perf_counter()
    try:
        child = _run_entry(
            entry,
            runtime_path=runtime_path,
            output=entry_root / entry_id,
            catalog_path=catalog_path,
            seed=seed,
            validate_only=False,
            limiter_registry=limiter_registry,
            progress=(
                (lambda message: progress(f"{entry_id} | {message}"))
                if progress is not None
                else None
            ),
        )
        status = str(child.get("status") or "unknown")
        child_artifacts = child.get("artifacts")
        if not isinstance(child_artifacts, Mapping):
            child_artifacts = child.get("artifact_paths")
        records_path = None
        if isinstance(child_artifacts, Mapping) and isinstance(
            child_artifacts.get("records"), str
        ):
            records_path = str(
                (Path("entries") / entry_id / str(child_artifacts["records"])).as_posix()
            )
        child_errors = child.get("errors")
        failure_preview = (
            [dict(item) for item in child_errors[:5] if isinstance(item, Mapping)]
            if isinstance(child_errors, Sequence)
            and not isinstance(child_errors, (str, bytes))
            else []
        )
        child_warnings = child.get("evaluation_warnings")
        evaluation_warning_preview = (
            [dict(item) for item in child_warnings[:5] if isinstance(item, Mapping)]
            if isinstance(child_warnings, Sequence)
            and not isinstance(child_warnings, (str, bytes))
            else []
        )
        result = {
            "id": entry_id,
            "benchmark_id": entry["benchmark_id"],
            "status": status,
            "selected_case_count": entry["selected_case_count"],
            "selected_group_count": entry["selected_group_count"],
            "coverage_status": entry["coverage_status"],
            "uncovered_requirements": entry["uncovered_requirements"],
            "concurrency_class": entry["concurrency_class"],
            "metric_availability": _metric_availability(child),
            "child_summary": str(
                (entry_root / entry_id / "suite_summary.json").relative_to(output)
            ),
            "run_id": child.get("run_id"),
            "completed_count": child.get("completed_count"),
            "failed_count": child.get("failed_count"),
            "pending_judge_count": child.get("pending_judge_count", 0),
            "judge_resume_count": child.get("judge_resume_count", 0),
            "failure_preview": failure_preview,
            "failure_count_reported": (
                len(child_errors)
                if isinstance(child_errors, Sequence)
                and not isinstance(child_errors, (str, bytes))
                else 0
            ),
            "records": records_path,
            "evaluation_warning_preview": evaluation_warning_preview,
            "evaluation_warning_count": (
                len(child_warnings)
                if isinstance(child_warnings, Sequence)
                and not isinstance(child_warnings, (str, bytes))
                else 0
            ),
            "endpoint_concurrency": child.get("endpoint_concurrency"),
        }
    except SimEvalError as exc:
        result = {
            "id": entry_id,
            "benchmark_id": entry["benchmark_id"],
            "status": "failed",
            "selected_case_count": entry["selected_case_count"],
            "selected_group_count": entry["selected_group_count"],
            "coverage_status": entry["coverage_status"],
            "uncovered_requirements": entry["uncovered_requirements"],
            "concurrency_class": entry["concurrency_class"],
            "error": {"kind": type(exc).__name__, "message": str(exc)},
        }
    result["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
    return result


def _report_entry_completion(
    result: Mapping[str, Any],
    *,
    index: int,
    total: int,
    progress: Callable[[str], None] | None,
) -> None:
    if progress is None:
        return
    entry_id = str(result["id"])
    preview = result.get("failure_preview")
    if isinstance(preview, Sequence) and preview and isinstance(preview[0], Mapping):
        first = preview[0]
        detail = " ".join(
            value
            for value in (
                str(first.get("stage") or "").strip(),
                str(first.get("kind") or "").strip(),
                str(first.get("message") or "").strip(),
            )
            if value
        )
        progress(f"[{index}/{total}] {entry_id}: {result['status']} | {detail[:300]}")
        return
    warnings = result.get("evaluation_warning_preview")
    if isinstance(warnings, Sequence) and warnings and isinstance(warnings[0], Mapping):
        first = warnings[0]
        detail = " ".join(
            value
            for value in (
                str(first.get("path") or "").strip(),
                str(first.get("kind") or "").strip(),
                str(first.get("message") or "").strip(),
            )
            if value
        )
        progress(
            f"[{index}/{total}] {entry_id}: {result['status']} "
            f"| evaluation warning: {detail[:280]}"
        )
        return
    progress(f"[{index}/{total}] {entry_id}: {result['status']}")


def _execute_smoke_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    execution: Mapping[str, Any],
    runtime_paths: Mapping[str, Path],
    output: Path,
    entry_root: Path,
    catalog_path: str,
    seed: int,
    progress: Callable[[str], None] | None,
) -> tuple[list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    """Resource-aware benchmark scheduler with suite-wide endpoint limits."""

    max_active = _positive_int(
        execution.get("max_active_benchmarks"),
        "execution.max_active_benchmarks",
    )
    class_limits = _mapping(
        execution.get("benchmark_class_limits", {}),
        "execution.benchmark_class_limits",
    )
    pending = list(enumerate(entries))
    active: dict[Future[dict[str, Any]], tuple[int, Mapping[str, Any], str]] = {}
    active_by_class: dict[str, int] = defaultdict(int)
    peak_by_class: dict[str, int] = defaultdict(int)
    peak_active = 0
    start_order: list[str] = []
    completion_order: list[str] = []
    ordered_results: list[dict[str, Any] | None] = [None] * len(entries)
    shared_registry = EndpointLimiterRegistry()
    suite_started_at = time.perf_counter()

    def class_limit(name: str) -> int:
        return int(class_limits.get(name, max_active))

    with ThreadPoolExecutor(
        max_workers=max_active,
        thread_name_prefix="sim-eval-smoke-entry",
    ) as pool:
        while pending or active:
            while len(active) < max_active:
                eligible_position = next(
                    (
                        position
                        for position, (_index, candidate) in enumerate(pending)
                        if active_by_class[str(candidate["concurrency_class"])]
                        < class_limit(str(candidate["concurrency_class"]))
                    ),
                    None,
                )
                if eligible_position is None:
                    break
                index, entry = pending.pop(eligible_position)
                entry_id = str(entry["id"])
                class_name = str(entry["concurrency_class"])
                if progress is not None:
                    progress(f"[{index + 1}/{len(entries)}] {entry_id}: starting")
                future = pool.submit(
                    _execute_smoke_entry,
                    entry,
                    runtime_path=runtime_paths[entry_id],
                    output=output,
                    entry_root=entry_root,
                    catalog_path=catalog_path,
                    seed=seed,
                    limiter_registry=shared_registry.fork_scope(),
                    progress=progress,
                )
                active[future] = (index, entry, class_name)
                active_by_class[class_name] += 1
                peak_by_class[class_name] = max(
                    peak_by_class[class_name], active_by_class[class_name]
                )
                peak_active = max(peak_active, len(active))
                start_order.append(entry_id)

            if not active:
                raise ConfigurationError(
                    "smoke benchmark scheduler has pending entries but none are eligible"
                )
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: active[item][0]):
                index, entry, class_name = active.pop(future)
                active_by_class[class_name] -= 1
                result = future.result()
                ordered_results[index] = result
                completion_order.append(str(entry["id"]))
                _report_entry_completion(
                    result,
                    index=index + 1,
                    total=len(entries),
                    progress=progress,
                )

    if any(result is None for result in ordered_results):
        raise AssertionError("smoke benchmark scheduler lost an entry result")
    results = [result for result in ordered_results if result is not None]
    scheduler = {
        "revision": "resource-aware-benchmark-scheduler-v1",
        "max_active_benchmarks": max_active,
        "benchmark_class_limits": dict(class_limits),
        "peak_active_benchmarks": peak_active,
        "peak_active_by_class": dict(sorted(peak_by_class.items())),
        "entry_start_order": start_order,
        "entry_completion_order": completion_order,
        "wall_time_seconds": round(time.perf_counter() - suite_started_at, 3),
    }
    return results, scheduler, shared_registry.snapshot()


def run_smoke_suite(
    *,
    config_path: str | Path = DEFAULT_SMOKE_CONFIG,
    output_directory: str | Path | None = None,
    plan_only: bool = False,
    selected_entries: Sequence[str] | None = None,
    global_eval_model: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Validate or resource-consciously execute the coverage smoke suite."""

    plan, runtime_documents = _prepare_smoke(
        config_path,
        selected_entries=selected_entries,
        global_eval_model=global_eval_model,
    )
    validations = _validate_plan(plan, runtime_documents)
    live_blockers = [
        {
            "id": validation.get("id"),
            "blockers": validation.get("live_readiness_blockers", []),
        }
        for validation in validations
        if validation.get("live_ready") is not True
    ]
    if plan_only:
        return {
            "schema_version": SMOKE_ARTIFACT_SCHEMA,
            "status": (
                "valid"
                if all(entry["coverage_status"] == "complete" for entry in plan["entries"])
                else "valid_with_partial_coverage"
            ),
            "plan_only": True,
            "plan": plan,
            "validations": validations,
            "live_ready": not live_blockers,
            "live_readiness_blockers": live_blockers,
            "network_calls": 0,
        }
    if plan.get("suite_mode") == "formal" and live_blockers:
        details = "; ".join(
            f"{item['id']}: {item['blockers']}" for item in live_blockers
        )
        raise ConfigurationError(
            "formal suite is not live-ready; resolve every local/model dependency "
            f"before execution: {details}"
        )
    if output_directory is None:
        raise ConfigurationError("suite execution requires --output")

    output = _safe_output(output_directory)
    artifact_prefix = "suite" if plan.get("suite_mode") == "formal" else "smoke"
    manifest_name = f"{artifact_prefix}_manifest.json"
    plan_name = f"{artifact_prefix}_plan.json"
    summary_name = f"{artifact_prefix}_summary.json"
    manifest_path = output / manifest_name
    if output.exists() and any(output.iterdir()) and not manifest_path.is_file():
        raise ConfigurationError(
            f"refusing to use nonempty suite output without {manifest_name}: {output}"
        )
    manifest = {
        "schema_version": SMOKE_ARTIFACT_SCHEMA,
        "suite_mode": plan.get("suite_mode"),
        "suite_id": plan["suite_id"],
        "plan_digest": plan["plan_digest"],
        "coverage_revision": plan["coverage_revision"],
    }
    compatible_resume = False
    existing_plan: Mapping[str, Any] | None = None
    existing_summary: Mapping[str, Any] | None = None
    execution_entries: Sequence[Mapping[str, Any]] = plan["entries"]
    if manifest_path.is_file():
        existing = load_json(manifest_path)
        if not isinstance(existing, Mapping):
            raise ConfigurationError("stored suite manifest must contain an object")
        if existing.get("plan_digest") != plan["plan_digest"]:
            execution_entries, existing_plan, existing_summary = _compatible_failed_resume_entries(
                output=output,
                plan=plan,
                runtime_documents=runtime_documents,
                plan_name=plan_name,
                summary_name=summary_name,
            )
            compatible_resume = True
            if progress is not None:
                progress(
                    "compatible failed-only resume: re-executing "
                    + (", ".join(str(item["id"]) for item in execution_entries) or "no entries")
                )
    else:
        atomic_write_json(manifest_path, manifest)
    if not compatible_resume:
        atomic_write_json(output / plan_name, plan)

    runtime_root = output / "runtime_configs"
    entry_root = output / "entries"
    runtime_paths = {}
    for entry in execution_entries:
        entry_id = str(entry["id"])
        runtime_path = runtime_root / f"{entry_id}.yaml"
        _write_runtime(runtime_path, runtime_documents[entry_id])
        runtime_paths[entry_id] = runtime_path

    executed_results, benchmark_concurrency, endpoint_concurrency = _execute_smoke_entries(
        execution_entries,
        execution=_mapping(plan["execution"], "plan.execution"),
        runtime_paths=runtime_paths,
        output=output,
        entry_root=entry_root,
        catalog_path=str(plan["catalog"]),
        seed=int(plan["seed"]),
        progress=progress,
    )

    resume_event = None
    active_plan_digest = plan["plan_digest"]
    results = executed_results
    if compatible_resume:
        assert existing_plan is not None and existing_summary is not None
        old_entries = {
            str(item["id"]): dict(item)
            for item in existing_summary.get("entries") or ()
            if isinstance(item, Mapping) and item.get("id") is not None
        }
        repaired = {str(item["id"]): item for item in executed_results}
        results = [
            repaired.get(str(entry["id"]), old_entries[str(entry["id"])])
            for entry in existing_plan.get("entries") or ()
            if isinstance(entry, Mapping)
        ]
        active_plan_digest = str(existing_plan["plan_digest"])
        resume_event = {
            "policy": "compatible_failed_entries_only_v1",
            "resumed_at": datetime.now(timezone.utc).isoformat(),
            "stored_plan_digest": active_plan_digest,
            "candidate_plan_digest": plan["plan_digest"],
            "reexecuted_entries": [str(item["id"]) for item in execution_entries],
            "reused_completed_entries": [
                str(item["id"])
                for item in existing_plan.get("entries") or ()
                if isinstance(item, Mapping) and str(item["id"]) not in repaired
            ],
            "hard_compatibility_checks": [
                "framework_version",
                "evaluated_and_support_model_identity",
                "generation_configuration",
                "metric_resources_and_scoring",
                "source_manifest_digest",
                "selected_case_and_group_ids",
                "seed",
            ],
        }

    failed = [result for result in results if result["status"] != "completed"]
    summary = {
        "schema_version": SMOKE_ARTIFACT_SCHEMA,
        "framework_version": __version__,
        "suite_id": plan["suite_id"],
        "suite_mode": plan.get("suite_mode"),
        "status": "completed" if not failed else "completed_with_failures",
        "diagnostic_only": plan.get("suite_mode") != "formal",
        "official_score_eligible": False,
        "result_scope": (
            "formal_full_import_research_evaluation"
            if plan.get("suite_mode") == "formal"
            else "diagnostic_metric_coverage_smoke"
        ),
        "global_eval_model": plan.get("global_eval_model"),
        "plan_digest": active_plan_digest,
        "benchmark_concurrency": benchmark_concurrency,
        "endpoint_concurrency": endpoint_concurrency,
        "entry_count": len(results),
        "failed_entry_count": len(failed),
        "entries": results,
        "artifacts": {
            "manifest": manifest_name,
            "plan": plan_name,
            "runtime_configs": "runtime_configs",
            "entries": "entries",
        },
    }
    if resume_event is not None:
        prior_history = existing_summary.get("resume_history") if existing_summary is not None else None
        summary["resume_history"] = [
            *(
                [dict(item) for item in prior_history if isinstance(item, Mapping)]
                if isinstance(prior_history, Sequence) and not isinstance(prior_history, (str, bytes))
                else []
            ),
            resume_event,
        ]
    atomic_write_json(output / summary_name, summary)
    return summary


__all__ = [
    "DEFAULT_SMOKE_CONFIG",
    "DEFAULT_FORMAL_CONFIG",
    "build_smoke_plan",
    "run_smoke_suite",
]
