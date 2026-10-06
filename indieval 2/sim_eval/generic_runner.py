"""Checkpointed live execution for the ten normalized benchmark imports."""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import __version__
from .thinking_budget import thinking_budget_identity
from .judge_resume import needs_judge_resume, merge_judge_result, validate_judge_roles
from .artifacts import CheckpointStore, atomic_write_json
from .backends.concurrency import EndpointLimiterRegistry
from .backends.episode_budget import (
    EpisodeOutputTokenBudgetBackend,
    attach_episode_token_budget,
    episode_budget_options_from_role,
    episode_token_scope,
    token_accounting_preflight,
)
from .backends.routed import AttemptScopedBackend
from .benchmarks import register_builtin_adapters
from .benchmarks.agentsense import AgentSenseAdapter
from .benchmarks.behaviorchain import BehaviorChainAdapter, BehaviorChainJudgeProvenance
from .benchmarks.coser import COSER_DIMENSIONS, CoserAdapter, CoserRuntimeProvenance
from .benchmarks.fantom import FantomAdapter
from .benchmarks.humanual import HumanualAdapter
from .benchmarks.mirrorbench import MirrorBenchAdapter, MirrorRuntimeProvenance, MirrorScoringProvenance
from .benchmarks.sotopia import SotopiaAdapter
from .catalog import load_catalog
from .model_adaptation import adaptation_identity
from .contracts import ResultStatus, RunIdentityInput, RunManifest, SampleManifest, case_result_from_dict
from .data.commands import _safe_output
from .data.loaders import HUMANUAL_DOMAINS, HUMANUAL_OFFICIAL_COLLECTION_FORMAT, load_import_spec, load_local_cases
from .environments.social import JudgeProvenance
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
from .integrations.fantom_semantic import (
    FANTOM_SEMANTIC_DEFAULT_MODEL_PATH,
    FantomSentenceTransformerScorer,
)
from .integrations.humanual_semantic import (
    HUMANUAL_EMBEDDING_DEFAULT_MODEL_PATH,
    HumanualSentenceTransformerScorer,
)
from .integrations.local_model_readiness import sentence_transformer_preflight
from .json_utils import jsonable, sha256_digest
from .progress import AttemptProgress
from .registry import get_adapter
from .runtime_config import (
    build_role_routed_backend,
    episode_max_output_tokens,
    load_benchmark_runtime_config,
)


from .extensions.supplemental import BENCHMARK_IDS as SUPPLEMENTAL_IDS, EVALUATED_ROLES as SUPPLEMENTAL_EVALUATED_ROLES


GENERIC_LIVE_BENCHMARK_IDS = SUPPLEMENTAL_IDS | frozenset(
    {
        "fantom",
        "social_r1",
        "lifechoices",
        "behaviorchain",
        "alignx",
        "humanllm",
        "humanual",
        "mirrorbench",
        "coser",
        "sotopia",
        "agentsense",
    }
)

EVALUATED_ROLES = {
    **SUPPLEMENTAL_EVALUATED_ROLES,
    "fantom": "evaluated_model",
    "social_r1": "evaluated_model",
    "lifechoices": "evaluated_model",
    "behaviorchain": "evaluated_model",
    "alignx": "evaluated_model",
    "humanllm": "evaluated_model",
    "humanual": "evaluated_model",
    "mirrorbench": "evaluated_user",
    "coser": "evaluated_actor",
    "sotopia": "evaluated_agent",
    "agentsense": "evaluated_actor",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"generic live runtime config {name} must be an object")
    return value


def _role(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    roles = _mapping(config.get("roles"), "roles")
    if name not in roles:
        raise ConfigurationError(f"runtime config requires role {name!r}")
    return _mapping(roles[name], f"roles.{name}")


def _generation(role: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(role.get("generation", {}), "role.generation")


def _prompt_revision(config: Mapping[str, Any], name: str, fallback: str) -> str:
    prompts = _mapping(config.get("prompts"), "prompts")
    raw = prompts.get(name)
    if raw is None:
        return fallback
    prompt = _mapping(raw, f"prompts.{name}")
    return str(prompt.get("revision") or fallback)


def _positive_int(value: Any, name: str, default: int) -> int:
    result = default if value is None else value
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return result


def _positive_float(value: Any, name: str, default: float) -> float:
    result = default if value is None else value
    if isinstance(result, bool) or not isinstance(result, (int, float)) or float(result) <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return float(result)


def _rollout_seed(seed: int, case_id: str, repetition: int) -> int:
    digest = hashlib.sha256(f"{seed}:{case_id}:{repetition}:generic-live-v1".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _select_groups(
    cases: Sequence[Any],
    *,
    seed: int,
    limit: int | None,
    selected_group_ids: Sequence[str] | None = None,
) -> tuple[list[Any], tuple[str, ...]]:
    available = {case.group_id for case in cases}
    if selected_group_ids is not None:
        if limit is not None:
            raise ConfigurationError("explicit selected_group_ids cannot be combined with --limit")
        group_ids = [str(group_id) for group_id in selected_group_ids]
        if not group_ids or len(group_ids) != len(set(group_ids)):
            raise ConfigurationError("selected_group_ids must be a nonempty unique sequence")
        unknown = sorted(set(group_ids) - available)
        if unknown:
            raise ConfigurationError(f"selected_group_ids contains unknown groups: {unknown}")
    else:
        group_ids = sorted(
            available,
            key=lambda group_id: hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest(),
        )
        if limit is not None:
            if isinstance(limit, bool) or limit <= 0:
                raise ConfigurationError("--limit must be a positive integer")
            group_ids = group_ids[:limit]
    rank = {group_id: index for index, group_id in enumerate(group_ids)}
    selected = [case for case in cases if case.group_id in rank]
    selected.sort(
        key=lambda case: (
            rank[case.group_id],
            hashlib.sha256(f"{seed}:{case.case_id}".encode("utf-8")).hexdigest(),
        )
    )
    return selected, tuple(group_ids)


def _social_judge(
    config: Mapping[str, Any],
    *,
    role_names: Sequence[str],
    rubric_revision: str,
    calls_per_output: int,
) -> JudgeProvenance:
    roles = [_role(config, role_name) for role_name in role_names]
    explicit_ids = tuple(str(role.get("judge_id") or "").strip() for role in roles)
    if any(explicit_ids) and not all(explicit_ids):
        raise ConfigurationError("social judge roles must either all define judge_id or all omit it")
    return JudgeProvenance(
        judge_models=tuple(str(role["model"]) for role in roles),
        judge_revisions=tuple(str(role["model_revision"]) for role in roles),
        rubric_revision=rubric_revision,
        calls_per_output=calls_per_output,
        source=f"runtime_config:{Path(str(config['_config_path'])).name}",
        judge_ids=explicit_ids if all(explicit_ids) else (),
    )


def _build_adapter(
    benchmark_id: str,
    cases: Sequence[Any],
    config: Mapping[str, Any],
    *,
    validate_only: bool,
) -> tuple[Any, tuple[str, ...], Mapping[str, str], Mapping[str, Any]]:
    """Return adapter, required roles, derived model routes, and resource status."""

    if benchmark_id in SUPPLEMENTAL_IDS:
        from .extensions.supplemental import build_adapter
        return build_adapter(benchmark_id, config)

    evaluated_role = EVALUATED_ROLES[benchmark_id]
    required_roles: list[str] = [evaluated_role]
    model_routes: dict[str, str] = {}
    resource_status: dict[str, Any] = {}

    if benchmark_id == "fantom":
        needs_semantic = any(
            case.input_data.get("question_family") == "belief"
            and case.input_data.get("answer_format") == "free_text"
            for case in cases
        )
        resource_status["belief_semantic_backend_required"] = needs_semantic
        scorer = None
        if needs_semantic:
            resources = _mapping(config.get("resources", {}), "resources")
            semantic = _mapping(resources.get("belief_semantic_backend", {}), "resources.belief_semantic_backend")
            revision = str(semantic.get("model_revision") or "").strip()
            batch_size = _positive_int(
                semantic.get("batch_size"),
                "resources.belief_semantic_backend.batch_size",
                32,
            )
            resource_status["belief_semantic_case_batch_size"] = batch_size
            model_path = str(
                semantic.get("model_path") or FANTOM_SEMANTIC_DEFAULT_MODEL_PATH
            ).strip()
            resource_status["belief_semantic_backend"] = sentence_transformer_preflight(
                model_path,
                model_revision=revision,
            )
            if not revision or revision.startswith(("replace-with-", "pin-on-")):
                if not validate_only:
                    raise ConfigurationError(
                        "FANToM free-text belief scoring requires a pinned resources.belief_semantic_backend.model_revision"
                    )
                resource_status["belief_semantic_model_is_placeholder"] = True
            elif not validate_only:
                scorer = FantomSentenceTransformerScorer(
                    model_revision=revision,
                    model_path=model_path,
                    device=str(semantic.get("device") or "").strip() or None,
                    batch_size=batch_size,
                )
        return FantomAdapter(
            belief_semantic_scorer=scorer,
            defer_belief_semantic=scorer is not None,
        ), tuple(required_roles), model_routes, resource_status

    if benchmark_id == "behaviorchain":
        judge = None
        if any(BehaviorChainAdapter.task_mode(case) == "generation" for case in cases):
            required_roles.append("generation_judge")
            role = _role(config, "generation_judge")
            judge = BehaviorChainJudgeProvenance(
                model=str(role["model"]),
                model_revision=str(role["model_revision"]),
                rubric_revision=_prompt_revision(config, "generation_judge", "behaviorchain-generation-evaluator-v1"),
                prompt_revision=_prompt_revision(config, "generation_judge", "behaviorchain-generation-evaluator-v1"),
                source=f"runtime_config:{Path(str(config['_config_path'])).name}",
                replayed=False,
            )
        return BehaviorChainAdapter(judge_provenance=judge), tuple(required_roles), model_routes, resource_status

    if benchmark_id == "humanual":
        required_roles.append("judge")
        judge = _social_judge(
            config,
            role_names=("judge",),
            rubric_revision=_prompt_revision(
                config,
                "alignment_judge",
                "humanlm-official-response-and-state-alignment-v1",
            ),
            calls_per_output=2,
        )
        resources = _mapping(config.get("resources", {}), "resources")
        semantic = _mapping(
            resources.get("embedding_semantic_backend", {}),
            "resources.embedding_semantic_backend",
        )
        revision = str(semantic.get("model_revision") or "").strip()
        batch_size = _positive_int(
            semantic.get("batch_size"),
            "resources.embedding_semantic_backend.batch_size",
            32,
        )
        resource_status["embedding_semantic_backend_required"] = True
        resource_status["embedding_semantic_case_batch_size"] = batch_size
        model_path = str(
            semantic.get("model_path") or HUMANUAL_EMBEDDING_DEFAULT_MODEL_PATH
        ).strip()
        resource_status["embedding_semantic_backend"] = sentence_transformer_preflight(
            model_path,
            model_revision=revision,
        )
        embedding_scorer = None
        if not revision or revision.startswith(("replace-with-", "pin-on-")):
            if not validate_only:
                raise ConfigurationError(
                    "HUMANUAL embedding cosine requires a pinned "
                    "resources.embedding_semantic_backend.model_revision"
                )
            resource_status["embedding_semantic_model_is_placeholder"] = True
        elif not validate_only:
            embedding_scorer = HumanualSentenceTransformerScorer(
                model_revision=revision,
                model_path=model_path,
                device=str(semantic.get("device") or "").strip() or None,
                batch_size=batch_size,
            )
        return HumanualAdapter(
            judge_provenance=judge,
            embedding_scorer=embedding_scorer,
        ), tuple(required_roles), model_routes, resource_status

    if benchmark_id == "mirrorbench":
        required_roles.extend(("fixed_assistant", "judge"))
        user = _role(config, "evaluated_user")
        assistant = _role(config, "fixed_assistant")
        judge = _role(config, "judge")
        user_generation = _generation(user)
        assistant_generation = _generation(assistant)
        judge_generation = _generation(judge)
        environment = _mapping(config.get("environment", {}), "environment")
        scoring = _mapping(config.get("scoring", {}), "scoring")
        runtime = MirrorRuntimeProvenance(
            fixed_assistant_model=str(assistant["model"]),
            fixed_assistant_revision=str(assistant["model_revision"]),
            assistant_policy_revision=_prompt_revision(config, "fixed_assistant", _prompt_revision(config, "user_simulation", "mirrorbench-assistant-policy-v1")),
            max_user_turns=_positive_int(environment.get("max_user_turns"), "environment.max_user_turns", 12),
            max_total_actions=_positive_int(environment.get("max_total_actions"), "environment.max_total_actions", 32),
            request_timeout_seconds=_positive_float(environment.get("request_timeout_seconds"), "environment.request_timeout_seconds", 600.0),
            max_retries=int(environment.get("max_retries", 2)),
            user_temperature=float(user_generation.get("temperature", 0.0)),
            assistant_temperature=float(assistant_generation.get("temperature", 0.0)),
            generation_max_tokens=_positive_int(environment.get("generation_max_tokens"), "environment.generation_max_tokens", 2048),
            source=f"runtime_config:{Path(str(config['_config_path'])).name}",
        )
        provenance = MirrorScoringProvenance(
            tokenizer_policy=str(scoring.get("tokenizer_policy") or "regex_word_v1"),
            tokenizer_model=str(scoring.get("tokenizer_model") or "regex_word_v1"),
            tokenizer_revision=str(scoring.get("tokenizer_revision") or "mirrorbench-regex-word-v1"),
            judge_model=str(judge["model"]),
            judge_revision=str(judge["model_revision"]),
            judge_temperature=float(judge_generation.get("temperature", 0.0)),
            judge_max_tokens=_positive_int(judge_generation.get("max_tokens"), "roles.judge.generation.max_tokens", 1024),
            gteval_prompt_revision=str(
                scoring.get("gteval_prompt_revision") or "mirrorbench-upstream-gteval-v1.0-json-schema"
            ),
            pi_prompt_revision=str(
                scoring.get("pi_prompt_revision") or "mirrorbench-upstream-pairwise-v1.1-json-schema"
            ),
            rnr_prompt_revision=str(
                scoring.get("rnr_prompt_revision") or "mirrorbench-upstream-rnr-v1.1-json-schema"
            ),
            gteval_samples=_positive_int(scoring.get("gteval_samples"), "scoring.gteval_samples", 1),
            pi_samples=_positive_int(scoring.get("pi_samples"), "scoring.pi_samples", 3),
            rnr_samples=_positive_int(scoring.get("rnr_samples"), "scoring.rnr_samples", 2),
            compute_controls=bool(scoring.get("compute_controls", True)),
            replayed=False,
            source=f"runtime_config:{Path(str(config['_config_path'])).name}",
        )
        return MirrorBenchAdapter(runtime_provenance=runtime, scoring_provenance=provenance), tuple(required_roles), model_routes, resource_status

    if benchmark_id == "coser":
        required_roles.extend(("environment_actor", "next_speaker", "judge"))
        environment_actor = _role(config, "environment_actor")
        next_speaker = _role(config, "next_speaker")
        environment = _mapping(config.get("environment", {}), "environment")
        runtime = CoserRuntimeProvenance(
            environment_model=str(environment_actor["model"]),
            environment_revision=str(environment_actor["model_revision"]),
            next_speaker_model=str(next_speaker["model"]),
            next_speaker_revision=str(next_speaker["model_revision"]),
            retrieval_mode=str(environment.get("retrieval_mode") or "none"),
            retrieval_k=int(environment.get("retrieval_k", 0)),
            hidden_thought_policy=str(
                environment.get("hidden_thought_policy")
                or "role_private_removed_from_other_roles_and_critic"
            ),
            max_judge_context_tokens=_positive_int(
                environment.get("max_judge_context_tokens"),
                "environment.max_judge_context_tokens",
                131072,
            ),
            source=f"runtime_config:{Path(str(config['_config_path'])).name}",
        )
        judge = _social_judge(
            config,
            role_names=("judge",),
            rubric_revision=_prompt_revision(config, "critic", "coser-gca-critic-v1"),
            calls_per_output=len(COSER_DIMENSIONS),
        )
        return CoserAdapter(judge_provenance=judge, runtime_provenance=runtime), tuple(required_roles), model_routes, resource_status

    if benchmark_id == "sotopia":
        required_roles.extend(("partner_agent", "judge"))
        judge = _social_judge(
            config,
            role_names=("judge",),
            rubric_revision=_prompt_revision(config, "evaluator", "sotopia-evaluator-v1"),
            calls_per_output=1,
        )
        return SotopiaAdapter(judge_provenance=judge), tuple(required_roles), model_routes, resource_status

    if benchmark_id == "agentsense":
        scoring = _mapping(config.get("scoring", {}), "scoring")
        execution = _mapping(config.get("execution", {}), "execution")
        judge_max_workers = _positive_int(
            execution.get("judge_max_workers"),
            "execution.judge_max_workers",
            1,
        )
        if judge_max_workers > 3:
            raise ConfigurationError("AgentSense execution.judge_max_workers cannot exceed 3")
        raw_judge_roles = scoring.get("judge_roles", ("judge_1", "judge_2", "judge_3"))
        if isinstance(raw_judge_roles, (str, bytes)) or not isinstance(raw_judge_roles, Sequence):
            raise ConfigurationError("AgentSense scoring.judge_roles must be an array")
        judge_roles = tuple(str(value) for value in raw_judge_roles)
        if len(judge_roles) != 3 or len(set(judge_roles)) != 3:
            raise ConfigurationError("AgentSense live evaluation requires exactly three distinct judge roles")
        required_roles.extend(judge_roles)
        judge = _social_judge(
            config,
            role_names=judge_roles,
            rubric_revision=_prompt_revision(config, "judge", "agentsense-evaluator-v1"),
            calls_per_output=3,
        )
        judge_role_by_id: dict[str, str] = {}
        if judge.judge_ids:
            invalid_ids = [
                judge_id
                for judge_id in judge.judge_ids
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", judge_id) is None
            ]
            if invalid_ids:
                raise ConfigurationError(
                    f"AgentSense judge_id values must be stable identifier slugs: {invalid_ids}"
                )
            judge_role_by_id = dict(zip(judge.judge_ids, judge_roles))
        else:
            if len(set(judge.judge_models)) != 3:
                raise ConfigurationError(
                    "AgentSense requires three distinct judge model IDs unless every judge role defines a distinct judge_id"
                )
            for role_name, model in zip(judge_roles, judge.judge_models):
                model_routes[model] = role_name
        return (
            AgentSenseAdapter(
                judge_provenance=judge,
                judge_role_by_id=judge_role_by_id,
                judge_max_workers=judge_max_workers,
            ),
            tuple(required_roles),
            model_routes,
            resource_status,
        )

    register_builtin_adapters()
    return get_adapter(benchmark_id), tuple(required_roles), model_routes, resource_status


def _effective_config(config: Mapping[str, Any], derived_model_routes: Mapping[str, str]) -> Mapping[str, Any]:
    routing = dict(_mapping(config.get("routing", {}), "routing"))
    declared = dict(_mapping(routing.get("model_routes", {}), "routing.model_routes"))
    for model, role_name in derived_model_routes.items():
        existing = declared.get(model)
        if existing is not None and existing != role_name:
            raise ConfigurationError(
                f"model {model!r} is routed to {existing!r}, but protocol provenance requires {role_name!r}"
            )
        declared[model] = role_name
    routing["model_routes"] = declared
    return {**dict(config), "routing": routing}


def _role_identities(config: Mapping[str, Any]) -> Mapping[str, Any]:
    result = {}
    for name, raw in _mapping(config.get("roles"), "roles").items():
        role = _mapping(raw, f"roles.{name}")
        result[str(name)] = role_execution_identity(role)
    return result


def _case_identity_digest(
    adapter: Any,
    cases: Sequence[Any],
    method_name: str,
    *,
    projection: Any | None = None,
) -> Mapping[str, Any]:
    method = getattr(adapter, method_name, None)
    if method is None:
        return {}
    values = [dict(method(case)) for case in cases]
    if projection is not None:
        values = [dict(projection(value)) for value in values]
    return {
        "case_bound": True,
        "identity_count": len(values),
        "identities_digest": sha256_digest(values),
    }


def _result_label(source_kind: str) -> str:
    return {
        "official": "official_source_protocol_variant",
        "official_derived_evaluation_positions": "official_source_protocol_variant",
        "authorized_local": "authorized_local_noncanonical",
        "local_compatibility": "local_compatibility",
        "synthetic_fixture": "synthetic_fixture_only",
    }.get(source_kind, "noncanonical_local_import")


def run_generic_import(
    *,
    manifest_path: str | Path,
    runtime_config_path: str | Path,
    output_directory: str | Path,
    catalog_path: str | Path,
    seed: int = 20260819,
    limit: int | None = None,
    validate_only: bool = False,
    selected_group_ids: Sequence[str] | None = None,
    global_eval_model: str | None = None,
    evaluated_model_config: str | Path | None = None,
    evaluate_overrides: Mapping[str, Any] | None = None,
    humanual_domain: str | None = None,
    limiter_registry: EndpointLimiterRegistry | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Validate or execute one non-UserLM/non-tau normalized live import."""

    spec = load_import_spec(manifest_path)
    if spec.benchmark_id not in GENERIC_LIVE_BENCHMARK_IDS:
        raise ConfigurationError(
            f"generic live runner does not support {spec.benchmark_id!r}; supported={sorted(GENERIC_LIVE_BENCHMARK_IDS)}"
        )
    if humanual_domain is not None:
        if spec.benchmark_id != "humanual" or spec.format != HUMANUAL_OFFICIAL_COLLECTION_FORMAT:
            raise ConfigurationError("--humanual-domain requires a HUMANUAL official collection manifest")
        if humanual_domain not in HUMANUAL_DOMAINS:
            raise ConfigurationError(f"unsupported HUMANUAL domain: {humanual_domain!r}")
        spec = replace(spec, humanual_domains=(humanual_domain,))
    cases, source_manifest = load_local_cases(spec)
    if humanual_domain is not None:
        source_manifest = replace(source_manifest, metadata={
            **dict(source_manifest.metadata), "selected_records": len(cases),
        })
    config = load_benchmark_runtime_config(
        runtime_config_path,
        global_eval_model=global_eval_model,
        evaluated_model_config=evaluated_model_config,
        evaluate_overrides=evaluate_overrides,
    )
    if config.get("benchmark_id") != spec.benchmark_id:
        raise ConfigurationError(
            f"runtime config benchmark_id={config.get('benchmark_id')!r} does not match import {spec.benchmark_id!r}"
        )
    catalog = load_catalog(catalog_path)
    catalog.get(spec.benchmark_id)
    explicit_selection = selected_group_ids is not None
    selected, resolved_group_ids = _select_groups(
        cases,
        seed=seed,
        limit=limit,
        selected_group_ids=selected_group_ids,
    )
    adapter, required_roles, derived_routes, resource_status = _build_adapter(
        spec.benchmark_id, selected, config, validate_only=validate_only
    )
    effective_config = _effective_config(config, derived_routes)
    for case in selected:
        adapter.validate_case(case)

    roles = _mapping(effective_config.get("roles"), "roles")
    missing_roles = sorted(set(required_roles) - set(roles))
    if missing_roles:
        raise ConfigurationError(f"runtime config lacks required live roles: {missing_roles}")
    placeholder_roles = sorted(
        role_name
        for role_name in required_roles
        if str(_mapping(roles[role_name], f"roles.{role_name}").get("model") or "").startswith("replace-with-")
    )
    execution = _mapping(effective_config.get("execution", {}), "execution")
    repetitions = _positive_int(execution.get("repetitions"), "execution.repetitions", 1)
    max_workers = _positive_int(execution.get("max_workers"), "execution.max_workers", 1)
    judge_max_workers = int(getattr(adapter, "judge_max_workers", 1))
    timeout = _positive_float(execution.get("request_timeout_seconds"), "execution.request_timeout_seconds", 120.0)
    episode_token_limit = episode_max_output_tokens(effective_config)
    evaluated_role = EVALUATED_ROLES[spec.benchmark_id]
    evaluated_role_config = _role(effective_config, evaluated_role)
    evaluated_model = str(evaluated_role_config["model"])
    token_accounting_status = token_accounting_preflight(evaluated_role_config)
    resource_status = {
        **dict(resource_status),
        "evaluated_model_token_accounting": token_accounting_status,
    }
    resource_blockers: list[str] = []
    for resource_name in ("belief_semantic_backend", "embedding_semantic_backend"):
        status = resource_status.get(resource_name)
        if isinstance(status, Mapping) and not status.get("ready_for_local_load", False):
            resource_blockers.extend(
                f"{resource_name}:{reason}"
                for reason in status.get("blocking_reasons", ())
            )
    if not token_accounting_status.get("ready_for_local_load", False):
        resource_blockers.extend(
            f"evaluated_model_token_accounting:{reason}"
            for reason in token_accounting_status.get("blocking_reasons", ())
        )
    live_blockers = [
        *(f"placeholder_role:{role_name}" for role_name in placeholder_roles),
        *resource_blockers,
    ]

    if validate_only:
        return {
            "schema_version": "1.0",
            "status": "valid",
            "benchmark_id": spec.benchmark_id,
            "validated_case_count": len(selected),
            "validated_group_count": len(resolved_group_ids),
            "source_population": len(cases),
            "source_group_count": len({case.group_id for case in cases}),
            "limit_unit": "dependency_groups",
            "required_roles": list(required_roles),
            "role_count": len(roles),
            "placeholder_roles": placeholder_roles,
            "model_configuration_is_placeholder": bool(placeholder_roles),
            "repetitions_per_case": repetitions,
            "max_workers": max_workers,
            "judge_max_workers": judge_max_workers,
            "episode_max_output_tokens": episode_token_limit,
            "resources": resource_status,
            "global_eval_model": config.get("global_eval_model"),
            "evaluated_model_override_applied": bool(
                config.get("evaluated_model_override_applied")
            ),
            "live_ready": not live_blockers,
            "live_readiness_blockers": live_blockers,
            "network_calls": 0,
        }
    if placeholder_roles:
        raise ConfigurationError(
            f"replace model placeholders for required roles before a live run: {placeholder_roles}"
        )
    episode_budget_options = episode_budget_options_from_role(evaluated_role_config)

    result_label = (
        "diagnostic_coverage_smoke_not_an_official_score"
        if explicit_selection
        else _result_label(source_manifest.source_kind)
    )
    if spec.benchmark_id in SUPPLEMENTAL_IDS and not explicit_selection and source_manifest.source_kind != "synthetic_fixture":
        result_label = "supplemental_compat_supplementary_not_source_author_canonical"
    profile = (
        f"coverage_smoke_{len(resolved_group_ids)}_groups_{len(selected)}_cases"
        if explicit_selection
        else (
            str(execution.get("profile") or f"live_full_{len(selected)}")
            if limit is None
            else f"diagnostic_{len(resolved_group_ids)}_groups_{len(selected)}_cases"
        )
    )
    repetition_seeds = {
        case.case_id: [_rollout_seed(seed, case.case_id, repetition) for repetition in range(repetitions)]
        for case in selected
    }
    sample_manifest = SampleManifest(
        benchmark_id=spec.benchmark_id,
        source_revision=source_manifest.source_revision,
        split=source_manifest.split,
        profile=profile,
        result_label=result_label,
        algorithm=(
            "explicit_dependency_group_coverage_selection_then_sha256_case_order_v1"
            if explicit_selection
            else "sha256(seed:group_id)_dependency_group_selection_then_sha256_case_order_v1"
        ),
        seed=seed,
        target={
            "group_limit": limit,
            "selected_cases": len(selected),
            "explicit_coverage_selection": explicit_selection,
        },
        population_group_count=len({case.group_id for case in cases}),
        population_case_count=len(cases),
        selected_group_ids=resolved_group_ids,
        selected_case_ids=tuple(case.case_id for case in selected),
        strata=tuple(str(value) for value in execution.get("strata", ())),
        repetition_seeds=repetition_seeds,
        source_manifest_digest=source_manifest.digest,
        metadata={
            "limit_unit": "dependency_groups",
            "repetitions_per_case": repetitions,
            "partial_diagnostic": limit is not None or explicit_selection,
            "selection_mode": "explicit_coverage_groups" if explicit_selection else "seeded_group_limit",
        },
    )
    provenance = adapter.provenance_for_case(selected[0]) if hasattr(adapter, "provenance_for_case") else None
    judge_identity = provenance.to_dict() if provenance is not None else {}
    role_identities = _role_identities(effective_config)
    evaluated_role_identity = role_identities[evaluated_role]
    support_role_identities = {
        role_name: role_identities[role_name]
        for role_name in required_roles
        if role_name != evaluated_role
    }
    environment_identity = {
        "runtime_config": environment_protocol_identity(
            _mapping(effective_config.get("environment", {}), "environment")
        ),
        **_case_identity_digest(
            adapter,
            selected,
            "environment_identity_for_case",
            projection=environment_protocol_identity,
        ),
    }
    assistant_identity = {
        "configured_support_roles": {
            name: support_protocol_identity(value)
            for name, value in support_role_identities.items()
        },
        **_case_identity_digest(
            adapter,
            selected,
            "assistant_or_partner_identity_for_case",
            projection=support_protocol_identity,
        ),
    }
    identity = RunIdentityInput(
        framework_version=__version__,
        benchmark_id=spec.benchmark_id,
        source_revision=source_manifest.source_revision,
        split=source_manifest.split,
        profile=profile,
        sample_manifest_digest=sample_manifest.digest,
        seed=seed,
        backend="named_role_routed_api_v2",
        model=evaluated_model,
        decoding={
            "adapter_controlled": True,
            **adaptation_identity(evaluated_role_identity),
            "evaluated_model_revision": evaluated_role_identity.get("model_revision"),
            "evaluated_generation": evaluated_role_identity.get("generation", {}),
            "evaluated_chat_template_kwargs": dict(
                (evaluated_role_config.get("extra_body") or {}).get("chat_template_kwargs") or {}
            ),
            **thinking_budget_identity(evaluated_role_config),
            "prompt_provenance": dict(_mapping(effective_config.get("prompts"), "prompts")),
            "repetitions": repetitions,
            "episode_max_output_tokens": episode_token_limit,
            "episode_token_scope": episode_token_scope(evaluated_role_config),
            "token_accounting": dict(evaluated_role_config.get("token_accounting") or {}),
        },
        prompt_revision=str(adapter.prompt_revision),
        scorer_revision=str(adapter.scorer_revision),
        judge={
            **support_protocol_identity(judge_identity),
            **(
                {
                    "belief_semantic_backend": dict(
                        adapter.belief_semantic_protocol_identity()
                    )
                }
                if isinstance(adapter, FantomAdapter)
                and adapter.belief_semantic_protocol_identity()
                else {}
            ),
            **(
                {
                    "embedding_semantic_backend": dict(
                        adapter.embedding_protocol_identity()
                    )
                }
                if isinstance(adapter, HumanualAdapter)
                and adapter.embedding_protocol_identity()
                else {}
            ),
        },
        environment=environment_identity,
        assistant_or_partner=assistant_identity,
    )
    run_manifest = RunManifest.create(
        identity,
        catalog_revision=catalog.catalog_revision,
        result_label=result_label,
        requested_case_count=len(selected),
        selected_case_count=len(selected),
        selected_group_count=len(resolved_group_ids),
        metadata={
            "source_manifest_digest": source_manifest.digest,
            "runtime_config": str(Path(runtime_config_path).resolve()),
            "limit_unit": "dependency_groups",
            "max_workers": max_workers,
            "judge_max_workers": judge_max_workers,
            "episode_max_output_tokens": episode_token_limit,
            "global_eval_model": config.get("global_eval_model"),
        },
    )
    output = _safe_output(output_directory)
    run_output = output / spec.benchmark_id / run_manifest.run_id
    store = CheckpointStore(run_output)
    store.initialize(run_manifest)
    atomic_write_json(run_output / "source_manifest.json", source_manifest)
    atomic_write_json(run_output / "sample_manifest.json", sample_manifest)
    limiter_registry = limiter_registry or EndpointLimiterRegistry()
    backend = build_role_routed_backend(
        effective_config,
        timeout=timeout,
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
        spec.benchmark_id,
        len(selected) * repetitions,
        len(completed) - len(judge_pending),
        progress,
    )
    tasks = [
        (case, repetition, rollout_seed)
        for case in selected
        for repetition, rollout_seed in enumerate(repetition_seeds[case.case_id])
        if (case.case_id, repetition) not in completed or (case.case_id, repetition) in judge_pending
    ]
    if tasks:
        validate_support_role_transition(
            [case_result_from_dict(value) for value in store.iter_record_dicts()],
            support_role_identities,
        )

    def execute(task: tuple[Any, int, int]) -> Any:
        case, repetition, rollout_seed = task
        attempt_backend = AttemptScopedBackend(backend, default_role=evaluated_role)
        budgeted_backend = EpisodeOutputTokenBudgetBackend(
            attempt_backend,
            max_output_tokens=episode_token_limit,
            evaluated_role=evaluated_role,
            evaluated_model=evaluated_model,
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
                seed=rollout_seed, model=evaluated_model, repetition=repetition,
                previous_result=previous,
            )
            result = merge_judge_result(previous, scored, api_calls=attempt_backend.events())
            if budgeted_backend.evaluated_request_count != int(budget_summary.get("evaluated_request_count", 0)):
                result = replace(result, metadata={**result.metadata, "judge_resume_episode_output_token_budget": dict(budgeted_backend.usage_summary())})
            return result
        result = adapter.execute_case(
            case,
            backend=budgeted_backend,
            run_id=run_manifest.run_id,
            seed=rollout_seed,
            model=evaluated_model,
            repetition=repetition,
        )
        result = attach_episode_token_budget(result, budgeted_backend)
        return attach_execution_provenance(
            result,
            evaluated_role=evaluated_role,
            evaluated_role_identity=evaluated_role_identity,
            support_roles=support_role_identities,
            api_calls=attempt_backend.events(),
        )

    semantic_buffer: list[tuple[Any, Any, int]] = []

    def flush_semantic_buffer() -> None:
        if not semantic_buffer:
            return
        if isinstance(adapter, FantomAdapter):
            finalized_results = adapter.finalize_belief_semantic_batch(semantic_buffer)
        elif isinstance(adapter, HumanualAdapter):
            finalized_results = adapter.finalize_embedding_batch(semantic_buffer)
        else:
            raise AssertionError("semantic buffer requires a semantic-scoring adapter")
        for finalized in finalized_results:
            store.append_result(finalized)
        semantic_buffer.clear()

    def accept_result(task: tuple[Any, int, int], result: Any) -> None:
        case, _, rollout_seed = task
        if result.checkpoint_key in judge_pending:
            store.append_judge_result(result)
            return
        if isinstance(adapter, FantomAdapter) and adapter.needs_belief_semantic(case, result):
            semantic_buffer.append((case, result, rollout_seed))
            if len(semantic_buffer) >= adapter.belief_semantic_batch_size:
                flush_semantic_buffer()
            return
        if isinstance(adapter, HumanualAdapter) and adapter.needs_embedding(case, result):
            semantic_buffer.append((case, result, rollout_seed))
            if len(semantic_buffer) >= adapter.embedding_batch_size:
                flush_semantic_buffer()
            return
        store.append_result(result)

    if max_workers == 1:
        for task in tasks:
            accept_result(task, execute(task))
            attempt_progress.advance()
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"sim-eval-{spec.benchmark_id}") as pool:
            # ``map`` executes concurrently but yields in task order.  This
            # keeps checkpoint order and local semantic batch composition
            # identical to the serial path.
            for task, result in zip(tasks, pool.map(execute, tasks)):
                accept_result(task, result)
                attempt_progress.advance()
    flush_semantic_buffer()

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
        "benchmark_id": spec.benchmark_id,
        "run_id": run_manifest.run_id,
        "result_label": result_label,
        "profile": profile,
        "selected_case_count": len(selected),
        "selected_group_count": len(resolved_group_ids),
        "repetitions_per_case": repetitions,
        "max_workers": max_workers,
        "judge_max_workers": judge_max_workers,
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


__all__ = ["GENERIC_LIVE_BENCHMARK_IDS", "run_generic_import"]
