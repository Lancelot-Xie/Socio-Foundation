"""Per-benchmark API role configuration, routing, and provenance checks."""

from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from .backends import get_backend
from .backends.concurrency import ConcurrencyLimitedBackend, EndpointLimiterRegistry
from .backends.episode_budget import DEFAULT_EPISODE_MAX_OUTPUT_TOKENS
from .backends.fallback import SafetyFallbackBackend
from .backends.openai_compatible import CHAT_PROFILES, STRUCTURED_MODES
from .backends.routed import NamedRoleRoutedBackend
from .errors import ConfigurationError
from .generation_policy import apply_benchmark_generation_policy
from .thinking_budget import apply_thinking_defaults
from .extensions.supplemental import EVALUATED_ROLES as SUPPLEMENTAL_EVALUATED_ROLES
from .integrations.token_counting import default_token_accounting_config
from .input_truncation import validate_input_truncation
from .model_adaptation import validate_model_adapter, USERLM_NATIVE


SUPPORTED_ROLE_BACKENDS = {
    "openai",
    "openai_compatible",
    "chat_completions",
    "openai_responses",
    "vllm",
}
PROMPT_SOURCES = {"official_paper", "official_repository", "supplemental_compat", "indieval_reconstruction"}
API_PROFILE_PATH = Path(__file__).resolve().parent / "resources" / "api_profiles.yaml"
GLOBAL_EVAL_MODEL_PATH = (
    Path(__file__).resolve().parent / "resources" / "global_eval_models.yaml"
)
GLOBAL_EVAL_MODEL_CHOICES = ("Qwen", "Deepseek", "GPT", "DeepseekRelay")
EVALUATED_ROLE_BY_BENCHMARK = {
    **SUPPLEMENTAL_EVALUATED_ROLES,
    "agentsense": "evaluated_actor",
    "alignx": "evaluated_model",
    "behaviorchain": "evaluated_model",
    "coser": "evaluated_actor",
    "fantom": "evaluated_model",
    "humanllm": "evaluated_model",
    "humanual": "evaluated_model",
    "lifechoices": "evaluated_model",
    "mirrorbench": "evaluated_user",
    "social_r1": "evaluated_model",
    "sotopia": "evaluated_agent",
    "tau_usi": "target_user",
    "userlm": "evaluated_user",
}
_EVALUATED_MODEL_ROUTING_FIELDS = {
    "model_adapter",
    "backend",
    "profile",
    "base_url",
    "base_url_env",
    "api_key_env",
    "require_api_key",
    "model",
    "model_revision",
    "max_inflight_requests",
    "token_limit_field",
    "send_sampling_params",
    "extra_body",
    "token_accounting",
    "reasoning_token_allowance",
}
_GLOBAL_SUPPORT_ROLE_PROTOCOL_FIELDS = (
    "generation",
    "structured_output",
    "judge_id",
    "policy_revision",
)


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"runtime config {label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ConfigurationError(f"runtime config requires {label}")
    return result


def load_config_document(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    """Load one JSON or YAML configuration document without applying a schema."""

    config_path = Path(path).resolve()
    try:
        raw = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigurationError(f"runtime config not found: {config_path}") from exc
    try:
        if config_path.suffix.casefold() == ".json":
            value = json.loads(raw)
        elif config_path.suffix.casefold() in {".yaml", ".yml"}:
            value = yaml.safe_load(raw)
        else:
            raise ConfigurationError("runtime config extension must be .json, .yaml, or .yml")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"invalid runtime config document: {exc}") from exc
    return config_path, _object(value, "root")


@lru_cache(maxsize=1)
def load_api_profiles() -> Mapping[str, Mapping[str, Any]]:
    _, document = load_config_document(API_PROFILE_PATH)
    if document.get("schema_version") != "1.0":
        raise ConfigurationError("API profile schema_version must be 1.0")
    profiles = _object(document.get("profiles"), "API profiles")
    return {str(name): dict(_object(value, f"profiles.{name}")) for name, value in profiles.items()}


@lru_cache(maxsize=1)
def load_global_eval_model_presets() -> Mapping[str, Mapping[str, Any]]:
    """Load the non-secret support-model presets exposed by the CLI."""

    _, document = load_config_document(GLOBAL_EVAL_MODEL_PATH)
    if document.get("schema_version") != "1.0":
        raise ConfigurationError("global eval model schema_version must be 1.0")
    presets = _object(document.get("presets"), "global eval model presets")
    observed = {str(name) for name in presets}
    expected = set(GLOBAL_EVAL_MODEL_CHOICES)
    if observed != expected:
        raise ConfigurationError(
            f"global eval model presets must be exactly {sorted(expected)}; got {sorted(observed)}"
        )
    return {
        str(name): dict(_object(value, f"global eval model presets.{name}"))
        for name, value in presets.items()
    }


def normalize_global_eval_model(value: str | None) -> str | None:
    """Return the canonical preset name, accepting case-insensitive API callers."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    aliases = {name.casefold(): name for name in GLOBAL_EVAL_MODEL_CHOICES}
    normalized = aliases.get(text.casefold())
    if normalized is None:
        raise ConfigurationError(
            f"global_eval_model must be one of {list(GLOBAL_EVAL_MODEL_CHOICES)}"
        )
    return normalized


def _global_support_role(
    raw_role: Mapping[str, Any],
    *,
    preset: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace provider/model routing while retaining role protocol settings."""

    result = copy.deepcopy(dict(preset))
    for field in _GLOBAL_SUPPORT_ROLE_PROTOCOL_FIELDS:
        if field in raw_role:
            result[field] = copy.deepcopy(raw_role[field])
    fallback = result.get("fallback")
    if isinstance(fallback, Mapping):
        resolved_fallback = copy.deepcopy(dict(fallback))
        for field in ("generation", "structured_output"):
            if field in raw_role:
                resolved_fallback[field] = copy.deepcopy(raw_role[field])
        result["fallback"] = resolved_fallback
    return result


def apply_global_eval_model(
    document: Mapping[str, Any],
    global_eval_model: str | None,
) -> dict[str, Any]:
    """Override every support role without modifying the evaluated model role.

    The override is applied to the raw runtime document before API-profile
    resolution so provider-specific defaults cannot leak from the original role.
    Qwen and Deepseek remain single-model routes. Relay-backed GPT presets
    carry a safety-only DeepSeek fallback, and the same role-level
    generation/output contract is applied to both primary and fallback
    requests.
    """

    selected = normalize_global_eval_model(global_eval_model)
    result = copy.deepcopy(dict(document))
    if selected is None:
        return result
    preset = load_global_eval_model_presets()[selected]
    benchmark_id = _text(result.get("benchmark_id"), "benchmark_id")
    if benchmark_id == "tau_usi":
        assistant = _object(result.get("fixed_assistant"), "fixed_assistant")
        result["fixed_assistant"] = _global_support_role(assistant, preset=preset)
    else:
        evaluated_role = EVALUATED_ROLE_BY_BENCHMARK.get(benchmark_id)
        if evaluated_role is None:
            raise ConfigurationError(
                f"global_eval_model does not support benchmark {benchmark_id!r}"
            )
        roles = _object(result.get("roles"), "roles")
        if evaluated_role not in roles:
            raise ConfigurationError(
                f"runtime config for {benchmark_id!r} lacks evaluated role {evaluated_role!r}"
            )
        result["roles"] = {
            str(role_name): (
                copy.deepcopy(dict(_object(raw_role, f"roles.{role_name}")))
                if str(role_name) == evaluated_role
                else _global_support_role(
                    _object(raw_role, f"roles.{role_name}"),
                    preset=preset,
                )
            )
            for role_name, raw_role in roles.items()
        }
    result["global_eval_model"] = selected
    return result


def load_evaluated_model_config(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    """Load a reusable evaluated-model routing override.

    Generation and structured-output protocol fields deliberately remain in
    each benchmark runtime YAML.  This file only selects the endpoint/model,
    so one target checkpoint can be evaluated across all benchmarks without
    editing twelve protocol documents.
    """

    config_path, document = load_config_document(path)
    if document.get("schema_version") != "1.0":
        raise ConfigurationError("evaluated model config schema_version must be 1.0")
    raw = _object(document.get("evaluated_model"), "evaluated_model")
    unknown = sorted(set(raw) - _EVALUATED_MODEL_ROUTING_FIELDS)
    if unknown:
        raise ConfigurationError(
            "evaluated_model contains protocol fields or unknown keys: " + ", ".join(unknown)
        )
    for required in ("backend", "model", "model_revision"):
        _text(raw.get(required), f"evaluated_model.{required}")
    return config_path, copy.deepcopy(dict(raw))


def apply_evaluated_model(
    document: Mapping[str, Any],
    evaluated_model: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Replace only evaluated-role routing while preserving benchmark protocol."""

    result = copy.deepcopy(dict(document))
    if evaluated_model is None:
        return result
    benchmark_id = _text(result.get("benchmark_id"), "benchmark_id")
    role_name = EVALUATED_ROLE_BY_BENCHMARK.get(benchmark_id)
    if role_name is None:
        raise ConfigurationError(
            f"evaluated model override does not support benchmark {benchmark_id!r}"
        )
    override = dict(_object(evaluated_model, "evaluated_model"))
    override.setdefault("token_accounting", default_token_accounting_config())
    if benchmark_id == "tau_usi":
        original = dict(_object(result.get(role_name), role_name))
        for field in _EVALUATED_MODEL_ROUTING_FIELDS:
            if field in override:
                original[field] = copy.deepcopy(override[field])
        if "model_adapter" not in override:
            original.pop("model_adapter", None)
        result[role_name] = original
    else:
        roles = dict(_object(result.get("roles"), "roles"))
        original = dict(_object(roles.get(role_name), f"roles.{role_name}"))
        for field in _EVALUATED_MODEL_ROUTING_FIELDS:
            if field in override:
                original[field] = copy.deepcopy(override[field])
        if "model_adapter" not in override:
            original.pop("model_adapter", None)
        roles[role_name] = original
        result["roles"] = roles
    result["evaluated_model_override_applied"] = True
    return result


def _inferred_profile(backend: str) -> str:
    return {
        "vllm": "vllm",
        "openai_responses": "openai",
        "openai": "openai_compatible",  # backward-compatible Chat Completions alias
        "openai_compatible": "openai_compatible",
        "chat_completions": "openai_compatible",
    }.get(backend, "openai_compatible")


def _validate_generation(generation: Mapping[str, Any], label: str) -> None:
    if "temperature" in generation and not isinstance(generation["temperature"], (int, float)):
        raise ConfigurationError(f"{label}.temperature must be numeric")
    if "top_p" in generation and (
        isinstance(generation["top_p"], bool)
        or not isinstance(generation["top_p"], (int, float))
        or not 0 <= float(generation["top_p"]) <= 1
    ):
        raise ConfigurationError(f"{label}.top_p must be numeric in [0,1]")
    if "max_tokens" in generation and (
        isinstance(generation["max_tokens"], bool)
        or not isinstance(generation["max_tokens"], int)
        or generation["max_tokens"] <= 0
    ):
        raise ConfigurationError(f"{label}.max_tokens must be a positive integer")
    if "reasoning_effort" in generation and generation["reasoning_effort"] is not None:
        _text(generation["reasoning_effort"], f"{label}.reasoning_effort")


def resolve_role_config(
    raw: Mapping[str, Any],
    *,
    label: str = "role",
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Merge one role with non-secret API profile defaults and validate it."""

    role = dict(_object(raw, label))
    adapter = validate_model_adapter(role.get("model_adapter"))
    if adapter == USERLM_NATIVE and role.get("backend") != "vllm":
        raise ConfigurationError("userlm_native_v1 requires vllm with the native UserLM chat template")
    token_accounting = role.get("token_accounting")
    if token_accounting is not None:
        accounting = _object(token_accounting, f"{label}.token_accounting")
        validate_input_truncation(accounting.get("input_truncation"))
        unknown_accounting = sorted(
            set(accounting)
            - {
                "kind",
                "model_path",
                "model_id",
                "model_revision",
                "local_files_only",
                "use_fast",
                "model_context_tokens",
                "min_generation_tokens",
                "output_token_source",
                "input_truncation",
            }
        )
        if unknown_accounting:
            raise ConfigurationError(
                f"{label}.token_accounting contains unknown fields: {unknown_accounting}"
            )
        for required in ("model_path", "model_id", "model_revision"):
            _text(accounting.get(required), f"{label}.token_accounting.{required}")
        if str(accounting.get("kind") or "huggingface") != "huggingface":
            raise ConfigurationError(f"{label}.token_accounting.kind must be huggingface")
        if accounting.get("output_token_source", "provider") not in ("provider", "local_content"):
            raise ConfigurationError(f"{label}.token_accounting.output_token_source must be provider or local_content")
        for field in ("local_files_only", "use_fast"):
            if field in accounting and not isinstance(accounting[field], bool):
                raise ConfigurationError(f"{label}.token_accounting.{field} must be boolean")
        for field in ("model_context_tokens", "min_generation_tokens"):
            if field in accounting and (
                isinstance(accounting[field], bool)
                or not isinstance(accounting[field], int)
                or accounting[field] <= 0
            ):
                raise ConfigurationError(
                    f"{label}.token_accounting.{field} must be a positive integer"
                )
    declared_backend = str(role.get("backend") or "").strip()
    declared_profile = str(role.get("profile") or "").strip()
    profile_name = declared_profile or _inferred_profile(declared_backend)
    profiles = load_api_profiles()
    defaults = profiles.get(profile_name)
    if defaults is None:
        raise ConfigurationError(f"{label}.profile is unsupported: {profile_name!r}")
    resolved = copy.deepcopy(dict(defaults))
    for key, value in role.items():
        if key in {"generation", "structured_output", "extra_body"} and isinstance(value, Mapping):
            merged = dict(resolved.get(key) or {})
            merged.update(copy.deepcopy(dict(value)))
            resolved[key] = merged
        else:
            resolved[key] = copy.deepcopy(value)
    resolved["profile"] = profile_name
    base_url_env = str(resolved.get("base_url_env") or "").strip()
    if base_url_env:
        resolved["base_url_env"] = base_url_env
        configured_url = os.getenv(base_url_env)
        if configured_url:
            resolved["base_url"] = configured_url.strip()

    backend = _text(resolved.get("backend"), f"{label}.backend")
    if backend not in SUPPORTED_ROLE_BACKENDS:
        raise ConfigurationError(
            f"{label}.backend must be chat_completions/openai_responses/openai/openai_compatible/vllm"
        )
    if backend == "openai_responses" and profile_name != "openai":
        raise ConfigurationError(f"{label} OpenAI Responses backend requires profile=openai")
    if backend == "vllm" and profile_name != "vllm":
        raise ConfigurationError(f"{label} vllm backend requires profile=vllm")
    if backend != "openai_responses" and profile_name not in CHAT_PROFILES:
        raise ConfigurationError(f"{label}.profile is not a Chat Completions profile")
    base_url = _text(resolved.get("base_url"), f"{label}.base_url")
    if not base_url.startswith(("http://", "https://")):
        raise ConfigurationError(f"{label}.base_url must be HTTP(S)")
    _text(resolved.get("model"), f"{label}.model")
    _text(resolved.get("model_revision"), f"{label}.model_revision")
    if "judge_id" in resolved:
        _text(resolved.get("judge_id"), f"{label}.judge_id")
    _text(resolved.get("api_key_env"), f"{label}.api_key_env")
    if "fallback_reason" in resolved:
        _text(resolved.get("fallback_reason"), f"{label}.fallback_reason")
    if "require_api_key" in resolved and not isinstance(resolved["require_api_key"], bool):
        raise ConfigurationError(f"{label}.require_api_key must be boolean")
    if "send_sampling_params" in resolved and not isinstance(resolved["send_sampling_params"], bool):
        raise ConfigurationError(f"{label}.send_sampling_params must be boolean")
    max_inflight = resolved.get("max_inflight_requests")
    if (
        isinstance(max_inflight, bool)
        or not isinstance(max_inflight, int)
        or max_inflight <= 0
    ):
        raise ConfigurationError(f"{label}.max_inflight_requests must be a positive integer")
    token_field = resolved.get("token_limit_field")
    if token_field is not None and token_field not in {"max_tokens", "max_completion_tokens"}:
        raise ConfigurationError(f"{label}.token_limit_field is unsupported")
    _object(resolved.get("extra_body", {}), f"{label}.extra_body")
    resolved = apply_thinking_defaults(resolved)
    generation = _object(resolved.get("generation", {}), f"{label}.generation")
    _validate_generation(generation, f"{label}.generation")
    structured = _object(resolved.get("structured_output", {}), f"{label}.structured_output")
    mode = str(structured.get("mode") or "auto")
    if mode not in STRUCTURED_MODES:
        raise ConfigurationError(f"{label}.structured_output.mode is unsupported")
    fallback_order = structured.get("fallback_order")
    if fallback_order is not None and (
        not isinstance(fallback_order, list)
        or not fallback_order
        or any(str(value) not in STRUCTURED_MODES - {"auto"} for value in fallback_order)
    ):
        raise ConfigurationError(f"{label}.structured_output.fallback_order is invalid")

    fallback = resolved.get("fallback")
    if fallback is not None:
        if not allow_fallback:
            raise ConfigurationError(f"{label} cannot define a nested fallback")
        fallback_raw = dict(_object(fallback, f"{label}.fallback"))
        trigger = _text(fallback_raw.pop("trigger", None), f"{label}.fallback.trigger")
        if trigger != "safety_filter":
            raise ConfigurationError(f"{label}.fallback.trigger must be exactly safety_filter")
        reason = _text(
            fallback_raw.get("fallback_reason"),
            f"{label}.fallback.fallback_reason",
        )
        resolved_fallback = resolve_role_config(
            fallback_raw,
            label=f"{label}.fallback",
            allow_fallback=False,
        )
        resolved_fallback["trigger"] = trigger
        resolved_fallback["fallback_reason"] = reason
        resolved["fallback"] = resolved_fallback
    return resolved


def apply_evaluate_overrides(
    config: Mapping[str, Any], overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply direct CLI routing/concurrency options without changing scoring."""
    result = copy.deepcopy(dict(config))
    if not overrides:
        return result
    unknown = set(overrides) - {"model", "model_revision", "base_url", "max_workers"}
    if unknown:
        raise ConfigurationError(f"unsupported evaluate overrides: {sorted(unknown)}")
    target = {
        key: _text(overrides[key], f"evaluate.{key}")
        for key in ("model", "model_revision", "base_url") if key in overrides
    }
    if "model" in target:
        # A newly served weight must not inherit the old checkpoint's revision.
        target.setdefault("model_revision", target["model"])
    if target:
        role_name = EVALUATED_ROLE_BY_BENCHMARK[str(result["benchmark_id"])]
        original = (result[role_name] if result["benchmark_id"] == "tau_usi"
                    else result["roles"][role_name])
        target.setdefault("token_accounting", original.get("token_accounting") or default_token_accounting_config())
        result = apply_evaluated_model(result, target)
    if "max_workers" in overrides:
        workers = overrides["max_workers"]
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ConfigurationError("evaluate.max_workers must be a positive integer")
        section = "limits" if result["benchmark_id"] == "tau_usi" else "execution"
        result[section] = {**dict(result.get(section) or {}), "max_workers": workers}
    return result


def load_benchmark_runtime_config(
    path: str | Path,
    *,
    global_eval_model: str | None = None,
    evaluated_model_config: str | Path | None = None,
    evaluate_overrides: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    config_path, raw_config = load_config_document(path)
    evaluated_config_path: Path | None = None
    evaluated_model: Mapping[str, Any] | None = None
    if evaluated_model_config is not None:
        evaluated_config_path, evaluated_model = load_evaluated_model_config(
            evaluated_model_config
        )
    config = apply_evaluated_model(
        apply_global_eval_model(raw_config, global_eval_model),
        evaluated_model,
    )
    config = apply_evaluate_overrides(config, evaluate_overrides)
    config = apply_benchmark_generation_policy(config)
    if config.get("schema_version") != "1.0":
        raise ConfigurationError("benchmark runtime config schema_version must be 1.0")
    _text(config.get("benchmark_id"), "benchmark_id")
    roles = _object(config.get("roles"), "roles")
    if not roles:
        raise ConfigurationError("runtime config requires at least one role")
    resolved_roles = {}
    for role_name, raw in roles.items():
        _text(role_name, "role name")
        role = resolve_role_config(_object(raw, f"roles.{role_name}"), label=f"roles.{role_name}")
        if (
            role.get("fallback") is not None
            and "judge" not in str(role_name).casefold()
            and config.get("global_eval_model") != "GPT"
        ):
            raise ConfigurationError(
                f"roles.{role_name}.fallback is forbidden: automatic safety fallback is judge-only"
            )
        resolved_roles[str(role_name)] = role
    prompts = _object(config.get("prompts"), "prompts")
    if not prompts:
        raise ConfigurationError("runtime config requires prompt provenance")
    for prompt_name, raw in prompts.items():
        prompt = _object(raw, f"prompts.{prompt_name}")
        source = _text(prompt.get("source"), f"prompts.{prompt_name}.source")
        if source not in PROMPT_SOURCES:
            raise ConfigurationError(f"prompts.{prompt_name}.source is unsupported")
        _text(prompt.get("revision"), f"prompts.{prompt_name}.revision")
        _text(prompt.get("locator"), f"prompts.{prompt_name}.locator")
    routing = _object(config.get("routing", {}), "routing")
    default_role = routing.get("default_role")
    if default_role is not None and default_role not in resolved_roles:
        raise ConfigurationError(f"routing.default_role {default_role!r} is not configured")
    model_routes = _object(routing.get("model_routes", {}), "routing.model_routes")
    for model, role_name in model_routes.items():
        _text(model, "routing.model_routes model")
        if role_name not in resolved_roles:
            raise ConfigurationError(
                f"routing.model_routes maps model {model!r} to unconfigured role {role_name!r}"
            )
    episode_max_output_tokens(config)
    return {
        **dict(config),
        "roles": resolved_roles,
        "_config_path": str(config_path),
        "_api_profile_path": str(API_PROFILE_PATH),
        "_global_eval_model_path": str(GLOBAL_EVAL_MODEL_PATH),
        "_evaluated_model_config_path": (
            str(evaluated_config_path) if evaluated_config_path is not None else None
        ),
    }


def episode_max_output_tokens(config: Mapping[str, Any]) -> int:
    """Resolve the evaluated-model cumulative output budget for one episode."""

    execution = _object(config.get("execution", {}), "execution")
    value = execution.get("episode_max_output_tokens", DEFAULT_EPISODE_MAX_OUTPUT_TOKENS)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError("execution.episode_max_output_tokens must be a positive integer")
    return value


def role_request_overrides(role: Mapping[str, Any]) -> dict[str, Any]:
    """Return the model and generation fields that a resolved role pins per request."""

    generation = _object(role.get("generation", {}), "role.generation")
    return {
        "model": str(role["model"]),
        **({"temperature": float(generation["temperature"])} if "temperature" in generation else {}),
        **({"top_p": float(generation["top_p"])} if "top_p" in generation else {}),
        **({"max_tokens": int(generation["max_tokens"])} if "max_tokens" in generation else {}),
        **(
            {"reasoning_effort": str(generation["reasoning_effort"])}
            if generation.get("reasoning_effort") is not None
            else {}
        ),
    }


def _backend_identity(role: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(role.get(key))
        for key in (
            "backend",
            "profile",
            "base_url",
            "base_url_env",
            "model",
            "model_revision",
            "token_limit_field",
            "send_sampling_params",
            "max_inflight_requests",
            "structured_output",
            "token_accounting",
            "reasoning_token_allowance",
        )
        if key in role
    }


def _plain_api_backend(
    role: Mapping[str, Any],
    *,
    timeout: float,
    limiter_registry: EndpointLimiterRegistry,
) -> Any:
    backend_name = str(role["backend"])
    backend = get_backend(
        backend_name,
        base_url=str(role["base_url"]),
        api_key_env=str(role["api_key_env"]),
        require_api_key=bool(role.get("require_api_key", backend_name != "vllm")),
        timeout=timeout,
        profile=str(role.get("profile") or _inferred_profile(backend_name)),
        token_limit_field=role.get("token_limit_field"),
        send_sampling_params=role.get("send_sampling_params"),
        extra_body=_object(role.get("extra_body", {}), "role.extra_body"),
        structured_output=_object(role.get("structured_output", {}), "role.structured_output"),
        **({"reasoning_token_allowance": role["reasoning_token_allowance"]}
           if role.get("reasoning_token_allowance") is not None else {}),
    )
    limiter = limiter_registry.get(
        base_url=str(role["base_url"]),
        model=str(role["model"]),
        max_inflight_requests=int(role["max_inflight_requests"]),
    )
    return ConcurrencyLimitedBackend(backend, limiter=limiter)


def build_api_backend(
    role: Mapping[str, Any],
    *,
    timeout: float = 120.0,
    limiter_registry: EndpointLimiterRegistry | None = None,
) -> Any:
    """Build one resolved API role, including its optional safety-only fallback."""

    resolved = resolve_role_config(role)
    registry = limiter_registry or EndpointLimiterRegistry()
    primary = _plain_api_backend(
        resolved,
        timeout=timeout,
        limiter_registry=registry,
    )
    fallback = resolved.get("fallback")
    if not isinstance(fallback, Mapping):
        return primary
    fallback_backend = _plain_api_backend(
        fallback,
        timeout=timeout,
        limiter_registry=registry,
    )
    return SafetyFallbackBackend(
        primary=primary,
        fallback=fallback_backend,
        fallback_request_overrides=role_request_overrides(fallback),
        primary_identity=_backend_identity(resolved),
        fallback_identity=_backend_identity(fallback),
        fallback_reason=str(fallback["fallback_reason"]),
    )


def build_role_routed_backend(
    config: Mapping[str, Any],
    *,
    timeout: float = 120.0,
    limiter_registry: EndpointLimiterRegistry | None = None,
) -> NamedRoleRoutedBackend:
    roles = _object(config.get("roles"), "roles")
    registry = limiter_registry or EndpointLimiterRegistry()
    backends = {}
    request_overrides = {}
    for role_name, raw in roles.items():
        role = resolve_role_config(_object(raw, f"roles.{role_name}"), label=f"roles.{role_name}")
        backends[str(role_name)] = build_api_backend(
            role,
            timeout=timeout,
            limiter_registry=registry,
        )
        request_overrides[str(role_name)] = role_request_overrides(role)
    routing = _object(config.get("routing", {}), "routing")
    return NamedRoleRoutedBackend(
        roles=backends,
        default_role=routing.get("default_role"),
        model_routes=_object(routing.get("model_routes", {}), "routing.model_routes"),
        request_overrides=request_overrides,
    )


__all__ = [
    "API_PROFILE_PATH",
    "EVALUATED_ROLE_BY_BENCHMARK",
    "GLOBAL_EVAL_MODEL_CHOICES",
    "GLOBAL_EVAL_MODEL_PATH",
    "apply_global_eval_model",
    "build_api_backend",
    "build_role_routed_backend",
    "load_api_profiles",
    "load_benchmark_runtime_config",
    "load_config_document",
    "load_global_eval_model_presets",
    "normalize_global_eval_model",
    "role_request_overrides",
    "resolve_role_config",
]
