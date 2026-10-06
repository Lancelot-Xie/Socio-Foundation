"""Checkpointed execution of normalized, non-fixture benchmark imports."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import __version__
from .thinking_budget import thinking_budget_identity
from .judge_resume import needs_judge_resume, merge_judge_result, validate_judge_roles
from .artifacts import CheckpointStore, atomic_write_json
from .backends.concurrency import EndpointLimiterRegistry
from .backends.episode_budget import (
    DEFAULT_EPISODE_MAX_OUTPUT_TOKENS,
    EpisodeOutputTokenBudgetBackend,
    attach_episode_token_budget,
    episode_budget_options_from_role,
    episode_token_scope,
    token_accounting_preflight,
)
from .backends.routed import AttemptScopedBackend, RoleRoutedBackend
from .benchmarks.tau_usi import TauRuntimeProvenance, TauScoringProvenance, TauUSIAdapter
from .benchmarks.userlm import UserLMAdapter, UserLMRuntimeProvenance, UserLMScoringProvenance
from .catalog import load_catalog
from .model_adaptation import adaptation_identity
from .contracts import (
    ResultStatus,
    RunIdentityInput,
    RunManifest,
    SampleManifest,
    case_result_from_dict,
)
from .data.commands import _safe_output
from .data.loaders import file_sha256, load_import_spec, load_local_cases
from .errors import ConfigurationError
from .execution_provenance import (
    attach_execution_provenance,
    environment_protocol_identity,
    role_execution_identity,
    summarize_evaluation_warnings,
    summarize_support_role_provenance,
    support_protocol_identity,
    validate_support_role_transition,
)
from .integrations.greyscope import (
    GREYSCOPE_CONTRACT_REVISION,
    GreyscopeDetectorConfig,
    GreyscopeLocalScorer,
)
from .integrations.tau_bench_local import TauBenchRepository, TauUSIOfficialReferenceStore
from .json_utils import jsonable
from .progress import AttemptProgress
from .runtime_config import (
    apply_evaluated_model,
    apply_evaluate_overrides,
    apply_global_eval_model,
    build_api_backend,
    build_role_routed_backend,
    episode_max_output_tokens,
    load_benchmark_runtime_config,
    load_config_document,
    load_evaluated_model_config,
    role_request_overrides,
    resolve_role_config,
)
from .verifiers.humaneval import HumanEvalExecutionConfig, build_humaneval_verifier


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"tau-USI runtime config {name} must be an object")
    return value


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ConfigurationError(f"tau-USI runtime config requires {name}")
    return text


def _positive_int(value: Any, name: str, default: int) -> int:
    result = default if value is None else value
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return result


def _resolve_path(value: Any, *, config_path: Path, name: str) -> Path:
    path = Path(_required_text(value, name))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _load_runtime_config(
    path: str | Path,
    *,
    global_eval_model: str | None = None,
    evaluated_model_config: str | Path | None = None,
    evaluate_overrides: Mapping[str, Any] | None = None,
) -> tuple[Path, Mapping[str, Any]]:
    config_path, raw = load_config_document(path)
    if raw.get("benchmark_id") != "tau_usi":
        raise ConfigurationError("runtime config must be an object with benchmark_id='tau_usi'")
    if raw.get("schema_version") != "1.2":
        raise ConfigurationError("tau-USI runtime config schema_version must be 1.2")
    evaluated_model = None
    if evaluated_model_config is not None:
        _, evaluated_model = load_evaluated_model_config(evaluated_model_config)
    config = apply_evaluated_model(
        apply_global_eval_model(raw, global_eval_model), evaluated_model,
    )
    return config_path, apply_evaluate_overrides(config, evaluate_overrides)


def _select(
    cases: Sequence[Any],
    *,
    seed: int,
    limit: int | None,
    selected_case_ids: Sequence[str] | None = None,
) -> list[Any]:
    if selected_case_ids is not None:
        if limit is not None:
            raise ConfigurationError("explicit selected_case_ids cannot be combined with --limit")
        requested = [str(case_id) for case_id in selected_case_ids]
        if not requested or len(requested) != len(set(requested)):
            raise ConfigurationError("selected_case_ids must be a nonempty unique sequence")
        by_id = {case.case_id: case for case in cases}
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ConfigurationError(f"selected_case_ids contains unknown cases: {unknown}")
        return [by_id[case_id] for case_id in requested]
    ranked = sorted(
        cases,
        key=lambda case: hashlib.sha256(f"{seed}:{case.case_id}".encode("utf-8")).hexdigest(),
    )
    if limit is None:
        return ranked
    if isinstance(limit, bool) or limit <= 0:
        raise ConfigurationError("--limit must be a positive integer")
    return ranked[:limit]


def _build_role_backend(
    role: Mapping[str, Any],
    *,
    timeout: float,
    limiter_registry: EndpointLimiterRegistry | None = None,
) -> Any:
    return build_api_backend(
        role,
        timeout=timeout,
        limiter_registry=limiter_registry,
    )


def run_tau_usi_import(
    *,
    manifest_path: str | Path,
    runtime_config_path: str | Path,
    output_directory: str | Path,
    catalog_path: str | Path,
    seed: int = 20260817,
    limit: int | None = None,
    validate_only: bool = False,
    selected_case_ids: Sequence[str] | None = None,
    global_eval_model: str | None = None,
    evaluated_model_config: str | Path | None = None,
    evaluate_overrides: Mapping[str, Any] | None = None,
    limiter_registry: EndpointLimiterRegistry | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    spec = load_import_spec(manifest_path)
    if spec.benchmark_id != "tau_usi":
        raise ConfigurationError("the tau-USI evaluate command requires a tau_usi import manifest")
    cases, source_manifest = load_local_cases(spec)
    if source_manifest.resolved_population != 165:
        raise ConfigurationError("official tau-USI execution requires a 165-task source population")
    config_path, config = _load_runtime_config(
        runtime_config_path,
        global_eval_model=global_eval_model,
        evaluated_model_config=evaluated_model_config,
        evaluate_overrides=evaluate_overrides,
    )
    target = resolve_role_config(
        _mapping(config.get("target_user"), "target_user"),
        label="target_user",
    )
    assistant = resolve_role_config(
        _mapping(config.get("fixed_assistant"), "fixed_assistant"),
        label="fixed_assistant",
    )
    environment = _mapping(config.get("environment"), "environment")
    limits = _mapping(config.get("limits"), "limits")
    scoring = _mapping(config.get("scoring"), "scoring")
    max_workers = _positive_int(limits.get("max_workers"), "limits.max_workers", 1)
    episode_token_limit = limits.get(
        "episode_max_output_tokens", DEFAULT_EPISODE_MAX_OUTPUT_TOKENS
    )
    if (
        isinstance(episode_token_limit, bool)
        or not isinstance(episode_token_limit, int)
        or episode_token_limit <= 0
    ):
        raise ConfigurationError("limits.episode_max_output_tokens must be a positive integer")
    token_accounting_status = token_accounting_preflight(target)

    backend_name = _required_text(target.get("backend"), "target_user.backend")
    assistant_backend_name = _required_text(assistant.get("backend"), "fixed_assistant.backend")
    _required_text(target.get("base_url"), "target_user.base_url")
    _required_text(target.get("model_revision"), "target_user.model_revision")
    _required_text(assistant.get("base_url"), "fixed_assistant.base_url")
    if "fallback_reason" in assistant:
        _required_text(assistant.get("fallback_reason"), "fixed_assistant.fallback_reason")
    if environment.get("runtime") != "local_tau_bench_v1":
        raise ConfigurationError("tau-USI runtime config environment.runtime must be local_tau_bench_v1")
    repository = TauBenchRepository()
    runtime_digest = repository.runtime_digest()
    declared_digest = _required_text(environment.get("task_source_revision"), "environment.task_source_revision")
    if declared_digest != runtime_digest:
        raise ConfigurationError(
            f"configured tau-bench task source {declared_digest} differs from local runtime {runtime_digest}"
        )
    source_digest = str(source_manifest.metadata.get("tau_bench_runtime_digest") or "")
    if source_digest != runtime_digest:
        raise ConfigurationError("import manifest and local tau-bench runtime digests differ")

    annotation_path = _resolve_path(
        scoring.get("annotation_local_path"), config_path=config_path, name="scoring.annotation_local_path"
    )
    annotation_revision = _required_text(scoring.get("annotation_revision"), "scoring.annotation_revision")
    if source_manifest.metadata.get("annotation_sha256") != annotation_revision:
        raise ConfigurationError("runtime scoring annotation revision differs from import manifest")
    reference_store = TauUSIOfficialReferenceStore(
        annotation_path,
        expected_sha256=annotation_revision,
    )
    difficulty_path = _resolve_path(
        scoring.get("difficulty_local_path"), config_path=config_path, name="scoring.difficulty_local_path"
    )
    difficulty_revision = _required_text(scoring.get("difficulty_revision"), "scoring.difficulty_revision")
    if not difficulty_path.is_file() or file_sha256(difficulty_path) != difficulty_revision:
        raise ConfigurationError("configured tau-USI difficulty file is missing or has a different SHA-256")
    if source_manifest.metadata.get("difficulty_sha256") != difficulty_revision:
        raise ConfigurationError("runtime scoring difficulty revision differs from import manifest")
    feature_revision = _required_text(
        scoring.get("feature_extractor_revision"), "scoring.feature_extractor_revision"
    )
    runtime_provenance = TauRuntimeProvenance(
        fixed_assistant_model=_required_text(assistant.get("model"), "fixed_assistant.model"),
        fixed_assistant_revision=_required_text(
            assistant.get("model_revision"), "fixed_assistant.model_revision"
        ),
        assistant_policy_revision=_required_text(
            assistant.get("policy_revision"), "fixed_assistant.policy_revision"
        ),
        environment_revision=_required_text(environment.get("revision"), "environment.revision"),
        tool_schema_revision=_required_text(
            environment.get("tool_schema_revision"), "environment.tool_schema_revision"
        ),
        max_user_turns=int(limits.get("max_user_turns", 60)),
        max_assistant_steps_per_user_turn=int(limits.get("max_assistant_steps_per_user_turn", 64)),
        request_timeout_seconds=float(limits.get("request_timeout_seconds", 120)),
        max_retries=int(limits.get("max_retries", 2)),
        survey_required=bool(limits.get("survey_required", True)),
        source=f"runtime_config:{config_path.name}",
    )
    scoring_provenance = TauScoringProvenance(
        annotation_revision=annotation_revision,
        feature_extractor_revision=feature_revision,
        survey_schema_revision=_required_text(
            scoring.get("survey_schema_revision"), "scoring.survey_schema_revision"
        ),
        difficulty_revision=difficulty_revision,
        human_batch_ids=tuple(str(value) for value in scoring.get("human_batch_ids") or ()),
        expected_task_count=int(scoring.get("expected_task_count", 0)),
        source="authorized_local_tau_usi_official_annotations",
        protocol_status="supplemental_compatible_scoring_with_privacy_minimized_local_reference_resolution",
    )
    for case in cases:
        declared = case.metadata.get("scoring_provenance")
        if not isinstance(declared, Mapping):
            raise ConfigurationError("tau-USI imported case lacks scoring provenance")
        for name, configured in (
            ("annotation_revision", scoring_provenance.annotation_revision),
            ("feature_extractor_revision", scoring_provenance.feature_extractor_revision),
            ("survey_schema_revision", scoring_provenance.survey_schema_revision),
            ("difficulty_revision", scoring_provenance.difficulty_revision),
            ("expected_task_count", scoring_provenance.expected_task_count),
        ):
            if declared.get(name) != configured:
                raise ConfigurationError(
                    f"tau-USI configured {name} differs from imported case provenance"
                )
        if tuple(declared.get("human_batch_ids") or ()) != scoring_provenance.human_batch_ids:
            raise ConfigurationError("tau-USI configured human batches differ from imported case provenance")
    adapter = TauUSIAdapter(
        runtime_provenance=runtime_provenance,
        scoring_provenance=scoring_provenance,
        feature_extractor_revision=feature_revision,
        human_reference_resolver=reference_store.references_for_case,
    )
    explicit_selection = selected_case_ids is not None
    selected = _select(
        cases,
        seed=seed,
        limit=limit,
        selected_case_ids=selected_case_ids,
    )
    for case in selected:
        adapter.validate_case(case)

    if validate_only:
        target_model_value = _required_text(target.get("model"), "target_user.model")
        assistant_model_value = runtime_provenance.fixed_assistant_model
        placeholder = any(
            value.startswith("replace-with-")
            for value in (target_model_value, assistant_model_value)
        )
        token_accounting_blockers = [
            f"evaluated_model_token_accounting:{reason}"
            for reason in token_accounting_status.get("blocking_reasons", ())
        ]
        return {
            "schema_version": "1.0",
            "status": "valid",
            "benchmark_id": "tau_usi",
            "validated_case_count": len(selected),
            "source_population": len(cases),
            "annotation_reference_count": len(selected) * 3,
            "annotation_sha256": reference_store.sha256,
            "tau_bench_runtime_digest": runtime_digest,
            "fixed_assistant_model": assistant_model_value,
            "target_user_model": target_model_value,
            "model_configuration_is_placeholder": placeholder,
            "episode_max_output_tokens": episode_token_limit,
            "max_workers": max_workers,
            "global_eval_model": config.get("global_eval_model"),
            "evaluated_model_override_applied": bool(
                config.get("evaluated_model_override_applied")
            ),
            "evaluated_model_token_accounting": token_accounting_status,
            "live_ready": not placeholder and not token_accounting_blockers,
            "live_readiness_blockers": (
                (["model_configuration_contains_placeholder"] if placeholder else [])
                + token_accounting_blockers
            ),
            "network_calls": 0,
        }

    episode_budget_options = episode_budget_options_from_role(target)
    profile = (
        f"coverage_smoke_{len(selected)}_cases"
        if explicit_selection
        else ("supplemental_compatible_165" if len(selected) == 165 else f"diagnostic_limit_{len(selected)}")
    )
    result_label = (
        "tau_usi_supplemental_compatible_165_task_score"
        if len(selected) == 165
        else "diagnostic_partial_run_not_a_tau_usi_score"
    )
    sample_manifest = SampleManifest(
        benchmark_id="tau_usi",
        source_revision=source_manifest.source_revision,
        split=source_manifest.split,
        profile=profile,
        result_label=result_label,
        algorithm=(
            "explicit_case_coverage_selection_v1"
            if explicit_selection
            else "sha256(seed:case_id)_ascending_then_optional_limit_v1"
        ),
        seed=seed,
        target=(
            {"selected_cases": len(selected), "explicit_coverage_selection": True}
            if explicit_selection
            else (165 if limit is None else limit)
        ),
        population_group_count=len({case.group_id for case in cases}),
        population_case_count=len(cases),
        selected_group_ids=tuple(case.group_id for case in selected),
        selected_case_ids=tuple(case.case_id for case in selected),
        strata=("domain", "difficulty_bin"),
        repetition_seeds={case.case_id: [seed] for case in selected},
        source_manifest_digest=source_manifest.digest,
        metadata={
            "partial_diagnostic": len(selected) != 165,
            "selection_mode": "explicit_coverage_cases" if explicit_selection else "seeded_case_limit",
        },
    )
    target_model = _required_text(target.get("model"), "target_user.model")
    if any(
        value.startswith("replace-with-")
        for value in (target_model, runtime_provenance.fixed_assistant_model)
    ):
        raise ConfigurationError(
            "replace the target and fixed-assistant model placeholders before a live tau-USI run"
        )
    identity = RunIdentityInput(
        framework_version=__version__,
        benchmark_id="tau_usi",
        source_revision=source_manifest.source_revision,
        split=source_manifest.split,
        profile=profile,
        sample_manifest_digest=sample_manifest.digest,
        seed=seed,
        backend="role_routed_api_v2",
        model=target_model,
        decoding={
            "adapter_controlled": True,
            **adaptation_identity(target),
            "target_model_revision": target.get("model_revision"),
            "target_generation": dict(_mapping(target.get("generation", {}), "target_user.generation")),
            "evaluated_chat_template_kwargs": dict(
                (target.get("extra_body") or {}).get("chat_template_kwargs") or {}
            ),
            **thinking_budget_identity(target),
            "episode_max_output_tokens": episode_token_limit,
            "episode_token_scope": episode_token_scope(target),
            "token_accounting": dict(target.get("token_accounting") or {}),
        },
        prompt_revision=adapter.prompt_revision,
        scorer_revision=adapter.scorer_revision,
        judge=scoring_provenance.to_dict(),
        environment=environment_protocol_identity(adapter.environment_identity_for_case(selected[0])),
        assistant_or_partner=support_protocol_identity(
            {
                **adapter.assistant_or_partner_identity_for_case(selected[0]),
                "backend": assistant_backend_name,
                "base_url": assistant.get("base_url"),
                "generation": dict(
                    _mapping(assistant.get("generation", {}), "fixed_assistant.generation")
                ),
            }
        ),
    )
    catalog = load_catalog(catalog_path)
    run_manifest = RunManifest.create(
        identity,
        catalog_revision=catalog.catalog_revision,
        result_label=result_label,
        requested_case_count=165 if limit is None else limit,
        selected_case_count=len(selected),
        selected_group_count=len({case.group_id for case in selected}),
        metadata={
            "source_manifest_digest": source_manifest.digest,
            "runtime_config": str(config_path),
            "annotation_sha256": reference_store.sha256,
            "tau_bench_runtime_digest": runtime_digest,
            "raw_human_references_visible_to_models": False,
            "episode_max_output_tokens": episode_token_limit,
            "max_workers": max_workers,
            "global_eval_model": config.get("global_eval_model"),
        },
    )
    output = _safe_output(output_directory)
    run_output = output / "tau_usi" / run_manifest.run_id
    store = CheckpointStore(run_output)
    store.initialize(run_manifest)
    atomic_write_json(run_output / "source_manifest.json", source_manifest)
    atomic_write_json(run_output / "sample_manifest.json", sample_manifest)

    limiter_registry = limiter_registry or EndpointLimiterRegistry()
    backend = RoleRoutedBackend(
        evaluated_backend=_build_role_backend(
            target,
            timeout=runtime_provenance.request_timeout_seconds,
            limiter_registry=limiter_registry,
        ),
        fixed_assistant_backend=_build_role_backend(
            assistant,
            timeout=runtime_provenance.request_timeout_seconds,
            limiter_registry=limiter_registry,
        ),
        evaluated_request_overrides=role_request_overrides(target),
        fixed_assistant_request_overrides=role_request_overrides(assistant),
    )
    evaluated_role_identity = role_execution_identity(target)
    support_role_identities = {"fixed_assistant": role_execution_identity(assistant)}
    completed = store.completed_keys()
    attempt_progress = AttemptProgress(
        "tau_usi",
        len(selected),
        len(completed),
        progress,
    )
    pending_cases = [case for case in selected if (case.case_id, 0) not in completed]
    if pending_cases:
        validate_support_role_transition(
            [case_result_from_dict(value) for value in store.iter_record_dicts()],
            support_role_identities,
        )
    def execute(case: Any) -> Any:
        attempt_backend = AttemptScopedBackend(backend, default_role="evaluated_user")
        budgeted_backend = EpisodeOutputTokenBudgetBackend(
            attempt_backend,
            max_output_tokens=episode_token_limit,
            evaluated_role="evaluated_user",
            evaluated_model=target_model,
            support_roles=tuple(support_role_identities),
            **episode_budget_options,
        )
        result = adapter.execute_case(
            case,
            backend=budgeted_backend,
            run_id=run_manifest.run_id,
            seed=seed,
            model=target_model,
            repetition=0,
        )
        result = attach_episode_token_budget(result, budgeted_backend)
        return attach_execution_provenance(
            result,
            evaluated_role="evaluated_user",
            evaluated_role_identity=evaluated_role_identity,
            support_roles=support_role_identities,
            api_calls=attempt_backend.events(),
        )

    if max_workers == 1:
        for case in pending_cases:
            store.append_result(execute(case))
            attempt_progress.advance()
    else:
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sim-eval-tau-usi",
        ) as pool:
            # Preserve selected-case checkpoint order while requests overlap.
            for result in pool.map(execute, pending_cases):
                store.append_result(result)
                attempt_progress.advance()

    results = [case_result_from_dict(value) for value in store.iter_latest_record_dicts()]
    attempt_results = [case_result_from_dict(value) for value in store.iter_record_dicts()]
    metrics = adapter.aggregate(results)
    metrics_path = store.write_metrics(metrics)
    completed_count = sum(result.status == ResultStatus.COMPLETED for result in results)
    failed_count = sum(result.status == ResultStatus.FAILED for result in results)
    errors = [
        {
            "case_id": result.case_id,
            "stage": result.error.stage,
            "kind": result.error.kind,
            "message": result.error.message,
            "retryable": result.error.retryable,
        }
        for result in results
        if result.status == ResultStatus.FAILED and result.error is not None
    ]
    summary = {
        "schema_version": "1.0",
        "framework_version": __version__,
        "status": "completed" if failed_count == 0 else "completed_with_failures",
        "benchmark_id": "tau_usi",
        "run_id": run_manifest.run_id,
        "result_label": result_label,
        "profile": profile,
        "selected_case_count": len(selected),
        "max_workers": max_workers,
        "result_count": len(results),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "errors": errors,
        "evaluation_warnings": summarize_evaluation_warnings(results),
        "global_eval_model": config.get("global_eval_model"),
        "endpoint_concurrency": limiter_registry.snapshot(),
        "support_role_provenance": summarize_support_role_provenance(attempt_results),
        "metrics": jsonable(metrics),
        "artifacts": {
            "run_manifest": str(store.manifest_path.relative_to(output)),
            "source_manifest": str((run_output / "source_manifest.json").relative_to(output)),
            "sample_manifest": str((run_output / "sample_manifest.json").relative_to(output)),
            "records": str(store.records_path.relative_to(output)),
            "metrics": str(metrics_path.relative_to(output)),
        },
    }
    atomic_write_json(output / "suite_summary.json", summary)
    return summary


def run_userlm_import(
    *,
    manifest_path: str | Path,
    runtime_config_path: str | Path,
    output_directory: str | Path,
    catalog_path: str | Path,
    seed: int = 20260818,
    limit: int | None = None,
    validate_only: bool = False,
    selected_case_ids: Sequence[str] | None = None,
    global_eval_model: str | None = None,
    evaluated_model_config: str | Path | None = None,
    evaluate_overrides: Mapping[str, Any] | None = None,
    limiter_registry: EndpointLimiterRegistry | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run a normalized UserLM Section 3 or LiC import with four routed roles."""

    spec = load_import_spec(manifest_path)
    if spec.benchmark_id != "userlm":
        raise ConfigurationError("the UserLM evaluate command requires a userlm import manifest")
    cases, source_manifest = load_local_cases(spec)
    config = load_benchmark_runtime_config(
        runtime_config_path,
        global_eval_model=global_eval_model,
        evaluated_model_config=evaluated_model_config,
        evaluate_overrides=evaluate_overrides,
    )
    if config.get("benchmark_id") != "userlm":
        raise ConfigurationError("UserLM execution requires configs/runtime/userlm.yaml shape")
    episode_token_limit = episode_max_output_tokens(config)
    execution = _mapping(config.get("execution", {}), "execution")
    max_workers = _positive_int(
        execution.get("max_workers"), "execution.max_workers", 1
    )
    roles = _mapping(config.get("roles"), "roles")
    required_roles = {"evaluated_user", "fixed_assistant", "intent_judge", "shard_judge"}
    if set(roles) != required_roles:
        raise ConfigurationError(f"UserLM runtime roles must be exactly {sorted(required_roles)}")
    user = _mapping(roles["evaluated_user"], "roles.evaluated_user")
    token_accounting_status = token_accounting_preflight(user)
    assistant = _mapping(roles["fixed_assistant"], "roles.fixed_assistant")
    intent_judge = _mapping(roles["intent_judge"], "roles.intent_judge")
    shard_judge = _mapping(roles["shard_judge"], "roles.shard_judge")
    resources = _mapping(config.get("resources", {}), "resources")
    detector_config = GreyscopeDetectorConfig.from_mapping(
        _mapping(resources.get("ai_text_detector", {}), "resources.ai_text_detector")
    )
    code_execution = HumanEvalExecutionConfig.from_mapping(
        _mapping(resources.get("code_execution", {}), "resources.code_execution")
    )
    runtime = UserLMRuntimeProvenance(
        fixed_assistant_model=_required_text(assistant.get("model"), "roles.fixed_assistant.model"),
        fixed_assistant_revision=_required_text(assistant.get("model_revision"), "roles.fixed_assistant.model_revision"),
        assistant_policy_revision="userlm-lic-official-assistant-prompts-v1",
        guardrail_revision="userlm-lic-guardrails-v2-no-first-token-filter",
        verifier_revision=code_execution.verifier_revision,
        max_user_turns=20,
        max_total_actions=42,
        request_timeout_seconds=180,
        max_retries=2,
        user_temperature=float(_mapping(user.get("generation", {}), "evaluated_user.generation").get("temperature", 0.0)),
        assistant_temperature=float(_mapping(assistant.get("generation", {}), "fixed_assistant.generation").get("temperature", 0.0)),
        apply_extrinsic_guardrails=True,
        deepseek_lic_first_turn=user.get("profile") == "deepseek",
        model_adapter=user.get("model_adapter"),
        source=f"runtime_config:{Path(runtime_config_path).name}",
    )
    scoring = UserLMScoringProvenance(
        shard_judge_model=_required_text(shard_judge.get("model"), "roles.shard_judge.model"),
        shard_judge_revision=_required_text(shard_judge.get("model_revision"), "roles.shard_judge.model_revision"),
        shard_prompt_revision="required-shard-v2-vllm-compatible-schema-local-uniqueness-check",
        intent_judge_model=_required_text(intent_judge.get("model"), "roles.intent_judge.model"),
        intent_judge_revision=_required_text(intent_judge.get("model_revision"), "roles.intent_judge.model_revision"),
        intent_prompt_revision="userlm-paper-figure10-transcribed-v1",
        lemmatizer_revision="userlm-paper-unigram-tokenizer-v1",
        ai_detector_model=(detector_config.model_id if detector_config.enabled else None),
        ai_detector_revision=(detector_config.model_revision if detector_config.enabled else None),
        ai_detector_contract_revision=(
            GREYSCOPE_CONTRACT_REVISION if detector_config.enabled else None
        ),
        replayed=False,
        source=f"runtime_config:{Path(runtime_config_path).name}",
    )
    adapter = UserLMAdapter(runtime_provenance=runtime, scoring_provenance=scoring)
    explicit_selection = selected_case_ids is not None
    selected = _select(
        cases,
        seed=seed,
        limit=limit,
        selected_case_ids=selected_case_ids,
    )
    for case in selected:
        adapter.validate_case(case)
    variants = sorted({adapter.variant_for_case(case) for case in selected})
    is_lic = variants == ["extrinsic_verifiable"]
    repetitions = int(resources.get("lic_repetitions", 10)) if is_lic else 1
    if repetitions <= 0:
        raise ConfigurationError("UserLM repetition count must be positive")
    user_model = _required_text(user.get("model"), "roles.evaluated_user.model")
    placeholder = any(
        str(role.get("model") or "").startswith("replace-with-")
        for role in roles.values()
        if isinstance(role, Mapping)
    )
    if validate_only:
        detector_preflight = detector_config.preflight()
        code_execution_preflight = code_execution.preflight()
        humaneval_code_case_count = sum(
            isinstance(case.input_data.get("assistant_task"), Mapping)
            and case.input_data["assistant_task"].get("kind") == "code"
            for case in selected
        )
        detector_required = "intrinsic_prism" in variants
        live_blockers = []
        if placeholder:
            live_blockers.append("model_configuration_contains_placeholder")
        if not token_accounting_status.get("ready_for_local_load", False):
            live_blockers.extend(
                f"evaluated_model_token_accounting:{reason}"
                for reason in token_accounting_status.get("blocking_reasons", ())
            )
        if detector_required and detector_config.enabled and not detector_preflight.get("ready_for_local_load"):
            live_blockers.extend(
                f"ai_text_detector:{reason}"
                for reason in detector_preflight.get("blocking_reasons", ())
            )
        if detector_required and not detector_config.enabled:
            live_blockers.append("ai_text_detector_disabled")
        if humaneval_code_case_count and not code_execution.enabled:
            live_blockers.append("humaneval_code_execution_disabled")
        if (
            humaneval_code_case_count
            and code_execution.enabled
            and not code_execution_preflight.get("ready")
        ):
            live_blockers.extend(
                f"humaneval_code_execution:{reason}"
                for reason in code_execution_preflight.get("blocking_reasons", ())
            )
        return {
            "schema_version": "1.0",
            "status": "valid",
            "benchmark_id": "userlm",
            "validated_case_count": len(selected),
            "source_population": len(cases),
            "variants": variants,
            "repetitions_per_case": repetitions,
            "max_workers": max_workers,
            "role_count": len(roles),
            "model_configuration_is_placeholder": placeholder,
            "code_execution": code_execution.identity(),
            "code_execution_preflight": code_execution_preflight,
            "ai_text_detector": detector_preflight,
            "evaluated_model_token_accounting": token_accounting_status,
            "ai_text_detector_required_for_selected_cases": detector_required,
            "episode_max_output_tokens": episode_token_limit,
            "global_eval_model": config.get("global_eval_model"),
            "humaneval_code_case_count": humaneval_code_case_count,
            "code_execution_security_boundary": (
                "linux_kernel_hardened_fail_closed"
                if code_execution.backend == "local_hardened_linux"
                else (
                    "local_guarded_is_not_a_security_sandbox; run formal LiC code evaluation "
                    "inside an isolated container or dedicated node"
                )
            ),
            "evaluated_model_override_applied": bool(
                config.get("evaluated_model_override_applied")
            ),
            "live_ready": not live_blockers,
            "live_readiness_blockers": live_blockers,
            "network_calls": 0,
        }
    if placeholder:
        raise ConfigurationError("replace every UserLM role model placeholder before a live run")
    episode_budget_options = episode_budget_options_from_role(user)

    protocol_variant = str(source_manifest.metadata.get("protocol_variant") or "").strip()
    declared_claim_boundary = str(source_manifest.metadata.get("claim_boundary") or "").strip()
    profile = (
        f"coverage_smoke_{len(selected)}_cases"
        if explicit_selection
        else (
            protocol_variant
            if protocol_variant and len(selected) == len(cases)
            else f"diagnostic_limit_{len(selected)}"
        )
    )
    result_label = (
        "diagnostic_coverage_smoke_not_an_official_score"
        if explicit_selection
        else (
            protocol_variant
            if protocol_variant and len(selected) == len(cases)
            else f"diagnostic_limit_{len(selected)}"
        )
    )
    repetition_seeds = {
        case.case_id: [seed + repetition for repetition in range(repetitions)]
        for case in selected
    }
    sample_manifest = SampleManifest(
        benchmark_id="userlm",
        source_revision=source_manifest.source_revision,
        split=source_manifest.split,
        profile=profile,
        result_label=result_label,
        algorithm=(
            "explicit_case_coverage_selection_v1"
            if explicit_selection
            else "sha256(seed:case_id)_ascending_then_optional_limit_v1"
        ),
        seed=seed,
        target=len(selected),
        population_group_count=len({case.group_id for case in cases}),
        population_case_count=len(cases),
        selected_group_ids=tuple(case.group_id for case in selected),
        selected_case_ids=tuple(case.case_id for case in selected),
        strata=("source_task", "intent", "required_information_pattern"),
        repetition_seeds=repetition_seeds,
        source_manifest_digest=source_manifest.digest,
        metadata={
            "variants": variants,
            "repetitions_per_case": repetitions,
            "partial_diagnostic": len(selected) != len(cases),
            "selection_mode": "explicit_coverage_cases" if explicit_selection else "seeded_case_limit",
            "protocol_variant": protocol_variant or None,
            "claim_boundary": declared_claim_boundary or None,
        },
    )
    role_identities = {
        str(name): role_execution_identity(_mapping(role, f"roles.{name}"))
        for name, role in roles.items()
    }
    evaluated_role_identity = role_identities["evaluated_user"]
    support_role_identities = {
        name: identity
        for name, identity in role_identities.items()
        if name != "evaluated_user"
    }
    identity = RunIdentityInput(
        framework_version=__version__,
        benchmark_id="userlm",
        source_revision=source_manifest.source_revision,
        split=source_manifest.split,
        profile=profile,
        sample_manifest_digest=sample_manifest.digest,
        seed=seed,
        backend="named_role_routed_api_v2",
        model=user_model,
        decoding={
            "adapter_controlled": True,
            **adaptation_identity(evaluated_role_identity),
            "evaluated_model_revision": evaluated_role_identity.get("model_revision"),
            "evaluated_generation": evaluated_role_identity.get("generation", {}),
            "evaluated_chat_template_kwargs": dict(
                (user.get("extra_body") or {}).get("chat_template_kwargs") or {}
            ),
            **thinking_budget_identity(user),
            "prompt_provenance": dict(_mapping(config.get("prompts"), "prompts")),
            "episode_max_output_tokens": episode_token_limit,
            "episode_token_scope": episode_token_scope(user),
            "token_accounting": dict(user.get("token_accounting") or {}),
        },
        prompt_revision=adapter.prompt_revision_for_case(selected[0]),
        scorer_revision=adapter.scorer_revision,
        judge={
            "scoring_protocol": support_protocol_identity(scoring.to_dict()),
            "configured_judge_roles": {
                name: support_protocol_identity(identity)
                for name, identity in support_role_identities.items()
                if "judge" in name
            },
        },
        environment={
            "section3": "single_next_user_turn",
            "lic": "role_isolated_dialogue_v1",
            "code_execution": code_execution.identity(),
            "ai_text_detector": detector_config.protocol_identity(),
        },
        assistant_or_partner={
            "protocol": support_protocol_identity(
                adapter.assistant_or_partner_identity_for_case(selected[0])
            ),
            "configured_role": support_protocol_identity(
                support_role_identities["fixed_assistant"]
            ),
        },
    )
    catalog = load_catalog(catalog_path)
    run_manifest = RunManifest.create(
        identity,
        catalog_revision=catalog.catalog_revision,
        result_label=result_label,
        requested_case_count=len(selected),
        selected_case_count=len(selected),
        selected_group_count=len({case.group_id for case in selected}),
        metadata={
            "source_manifest_digest": source_manifest.digest,
            "runtime_config": str(Path(runtime_config_path).resolve()),
            "code_execution": code_execution.identity(),
            "ai_text_detector": detector_config.identity(),
            "episode_max_output_tokens": episode_token_limit,
            "max_workers": max_workers,
            "global_eval_model": config.get("global_eval_model"),
        },
    )
    output = _safe_output(output_directory)
    code_task_verifier = build_humaneval_verifier(code_execution, output_root=output)
    ai_text_detector = (
        GreyscopeLocalScorer(detector_config)
        if detector_config.enabled and "intrinsic_prism" in variants
        else None
    )
    adapter = UserLMAdapter(
        runtime_provenance=runtime,
        scoring_provenance=scoring,
        code_task_verifier=code_task_verifier,
        ai_text_detector=ai_text_detector,
        defer_ai_text_detection=ai_text_detector is not None,
    )
    run_output = output / "userlm" / run_manifest.run_id
    store = CheckpointStore(run_output)
    store.initialize(run_manifest)
    atomic_write_json(run_output / "source_manifest.json", source_manifest)
    atomic_write_json(run_output / "sample_manifest.json", sample_manifest)
    limiter_registry = limiter_registry or EndpointLimiterRegistry()
    backend = build_role_routed_backend(
        config,
        timeout=runtime.request_timeout_seconds,
        limiter_registry=limiter_registry,
    )
    completed = store.completed_keys()
    latest_results = {r.checkpoint_key: r for r in (case_result_from_dict(v) for v in store.iter_latest_record_dicts())}
    judge_pending = {
        (case.case_id, repetition): latest_results[(case.case_id, repetition)]
        for case in selected
        for repetition in range(repetitions)
        if (case.case_id, repetition) in completed
        and needs_judge_resume(adapter, case, latest_results[(case.case_id, repetition)])
    }
    for previous in judge_pending.values():
        validate_judge_roles(previous, support_role_identities)
    attempt_progress = AttemptProgress(
        "userlm",
        len(selected) * repetitions,
        len(completed) - len(judge_pending),
        progress,
    )
    pending_tasks_by_case = [
        (
            case,
            tuple(
                (case, repetition, rollout_seed)
                for repetition, rollout_seed in enumerate(repetition_seeds[case.case_id])
                if (case.case_id, repetition) not in completed or (case.case_id, repetition) in judge_pending
            ),
        )
        for case in selected
    ]
    pending_tasks_by_case = [
        (case, tasks) for case, tasks in pending_tasks_by_case if tasks
    ]
    pending_tasks = [task for _, tasks in pending_tasks_by_case for task in tasks]
    if pending_tasks:
        validate_support_role_transition(
            [case_result_from_dict(value) for value in store.iter_record_dicts()],
            support_role_identities,
        )
    detector_buffer: list[tuple[Any, Any]] = []

    def flush_detector_buffer() -> None:
        if not detector_buffer:
            return
        for finalized in adapter.finalize_ai_text_detector_batch(detector_buffer):
            store.append_result(finalized)
        detector_buffer.clear()

    def execute(task: tuple[Any, int, int]) -> tuple[Any, Any]:
        case, repetition, rollout_seed = task
        attempt_backend = AttemptScopedBackend(backend, default_role="evaluated_user")
        budgeted_backend = EpisodeOutputTokenBudgetBackend(
            attempt_backend,
            max_output_tokens=episode_token_limit,
            evaluated_role="evaluated_user",
            evaluated_model=user_model,
            support_roles=tuple(support_role_identities),
            **episode_budget_options,
        )
        previous = judge_pending.get((case.case_id, repetition))
        if previous is not None:
            budget_summary = previous.metadata.get("judge_resume_episode_output_token_budget", previous.metadata.get("episode_output_token_budget", {}))
            for field in ("used_output_tokens", "evaluated_request_count", "provider_usage_request_count", "tokenizer_usage_request_count", "fallback_usage_request_count", "budget_overrun_tokens"):
                setattr(budgeted_backend, field, int(budget_summary.get(field, 0)))
            scored = adapter.execute_case(
                case, backend=budgeted_backend, run_id=run_manifest.run_id,
                seed=rollout_seed, model=user_model, repetition=repetition,
                previous_result=previous,
            )
            result = merge_judge_result(previous, scored, api_calls=attempt_backend.events())
            if budgeted_backend.evaluated_request_count != int(budget_summary.get("evaluated_request_count", 0)):
                from dataclasses import replace
                result = replace(result, metadata={**result.metadata, "judge_resume_episode_output_token_budget": dict(budgeted_backend.usage_summary())})
            return case, result
        result = adapter.execute_case(
            case,
            backend=budgeted_backend,
            run_id=run_manifest.run_id,
            seed=rollout_seed,
            model=user_model,
            repetition=repetition,
        )
        result = attach_episode_token_budget(result, budgeted_backend)
        result = attach_execution_provenance(
            result,
            evaluated_role="evaluated_user",
            evaluated_role_identity=evaluated_role_identity,
            support_roles=support_role_identities,
            api_calls=attempt_backend.events(),
        )
        return case, result

    def execute_case_tasks(
        item: tuple[Any, Sequence[tuple[Any, int, int]]],
    ) -> tuple[tuple[Any, Any], ...]:
        # Repetitions of one logical LiC case stay serial.  This prevents
        # same-key verifier/cache races while different cases overlap.
        _, tasks = item
        return tuple(execute(task) for task in tasks)

    def accept(case: Any, result: Any) -> None:
        if result.checkpoint_key in judge_pending:
            store.append_judge_result(result)
            return
        if ai_text_detector is not None and adapter.needs_ai_text_detection(case, result):
            detector_buffer.append((case, result))
            if len(detector_buffer) >= adapter.ai_text_detector_batch_size:
                flush_detector_buffer()
        else:
            store.append_result(result)

    if max_workers == 1:
        completed_groups = map(execute_case_tasks, pending_tasks_by_case)
        for group in completed_groups:
            for case, result in group:
                accept(case, result)
                attempt_progress.advance()
    else:
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sim-eval-userlm",
        ) as pool:
            # Ordered map keeps repetition order, detector batch composition,
            # and checkpoint order equal to the serial implementation.
            for group in pool.map(execute_case_tasks, pending_tasks_by_case):
                for case, result in group:
                    accept(case, result)
                    attempt_progress.advance()
    flush_detector_buffer()
    results = [case_result_from_dict(value) for value in store.iter_latest_record_dicts()]
    attempt_results = [case_result_from_dict(value) for value in store.iter_record_dicts()]
    metrics = adapter.aggregate(results)
    metrics_path = store.write_metrics(metrics)
    completed_count = sum(result.status == ResultStatus.COMPLETED for result in results)
    failed_count = sum(result.status == ResultStatus.FAILED for result in results)
    errors = [
        {
            "case_id": result.case_id,
            "repetition": result.repetition,
            "stage": result.error.stage,
            "kind": result.error.kind,
            "message": result.error.message,
            "retryable": result.error.retryable,
        }
        for result in results
        if result.status == ResultStatus.FAILED and result.error is not None
    ]
    selected_by_id = {case.case_id: case for case in selected}
    pending_judge_count = sum(needs_judge_resume(adapter, selected_by_id[r.case_id], r) for r in results)
    summary = {
        "schema_version": "1.0",
        "framework_version": __version__,
        "status": "completed" if failed_count == 0 and pending_judge_count == 0 else "completed_with_failures",
        "pending_judge_count": pending_judge_count,
        "judge_resume_count": len(judge_pending),
        "benchmark_id": "userlm",
        "run_id": run_manifest.run_id,
        "result_label": result_label,
        "profile": profile,
        "selected_case_count": len(selected),
        "repetitions_per_case": repetitions,
        "max_workers": max_workers,
        "result_count": len(results),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "errors": errors,
        "evaluation_warnings": summarize_evaluation_warnings(results),
        "code_execution": code_execution.identity(),
        "ai_text_detector": detector_config.identity(),
        "global_eval_model": config.get("global_eval_model"),
        "endpoint_concurrency": limiter_registry.snapshot(),
        "support_role_provenance": summarize_support_role_provenance(attempt_results),
        "metrics": jsonable(metrics),
        "artifacts": {
            "run_manifest": str(store.manifest_path.relative_to(output)),
            "source_manifest": str((run_output / "source_manifest.json").relative_to(output)),
            "sample_manifest": str((run_output / "sample_manifest.json").relative_to(output)),
            "records": str(store.records_path.relative_to(output)),
            "metrics": str(metrics_path.relative_to(output)),
        },
    }
    atomic_write_json(output / "suite_summary.json", summary)
    return summary


__all__ = ["run_tau_usi_import", "run_userlm_import"]
