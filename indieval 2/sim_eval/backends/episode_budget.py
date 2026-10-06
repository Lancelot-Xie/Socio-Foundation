"""Evaluated-model budgets with opt-in local content accounting."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import CaseResult, ModelRequest, ModelResponse
from ..errors import EpisodeTokenBudgetExceeded
from ..errors import ConfigurationError
from ..integrations.token_counting import (
    TokenCounter,
    default_token_accounting_config,
    get_shared_huggingface_token_counter,
)
from ..interfaces import ModelBackend
from ..model_adaptation import (adaptation_identity, adapt_evaluated_request,
                                annotate_adapted_response, validate_model_adapter)
from ..input_truncation import truncate_material, validate_input_truncation
from ..thinking_budget import thinking_token_budget as role_thinking_token_budget


DEFAULT_EPISODE_MAX_OUTPUT_TOKENS = 32_768
DEFAULT_MODEL_CONTEXT_TOKENS = 32_768
DEFAULT_MIN_GENERATION_TOKENS = 10
EPISODE_BUDGET_REVISION = "evaluated-model-output-and-context-token-budget-v2"
CONTENT_EPISODE_BUDGET_REVISION = "evaluated-content-and-context-token-budget-v3"
LOCAL_CONTENT_BUDGET_REVISION = "evaluated-local-content-and-context-token-budget-v1"
_GENERATION_ALLOWANCE_METADATA_KEY = "_sim_eval_generation_token_allowance"


def episode_token_scope(role: Mapping[str, Any]) -> str:
    accounting = role.get("token_accounting") or {}
    if role_thinking_token_budget(role) is not None or accounting.get("output_token_source") == "local_content":
        return "evaluated_model_content_only"
    return "evaluated_model_outputs_only"


class EpisodeOutputTokenBudgetBackend(ModelBackend):
    """Clamp evaluated-model requests to the remaining episode output budget.

    Fixed assistants, environment actors, and judges are intentionally not
    charged.  This mirrors Supplemental's target-rollout response budget rather
    than treating every support-model call as part of the evaluated policy.
    """

    name = "episode_output_token_budget"

    def __init__(
        self,
        backend: ModelBackend,
        *,
        max_output_tokens: int = DEFAULT_EPISODE_MAX_OUTPUT_TOKENS,
        evaluated_role: str,
        evaluated_model: str,
        support_roles: Sequence[str] = (),
        token_counter: TokenCounter | None = None,
        model_context_tokens: int | None = None,
        min_generation_tokens: int = DEFAULT_MIN_GENERATION_TOKENS,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        thinking_token_budget: int | None = None,
        reasoning_token_allowance: int | None = None,
        output_token_source: str = "provider",
        input_truncation: str | None = None,
        model_adapter: str | None = None,
    ) -> None:
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            raise ValueError("episode max_output_tokens must be a positive integer")
        if not evaluated_role or not evaluated_model:
            raise ValueError("episode token budget requires evaluated_role and evaluated_model")
        if model_context_tokens is not None and (
            isinstance(model_context_tokens, bool)
            or not isinstance(model_context_tokens, int)
            or model_context_tokens <= 0
        ):
            raise ValueError("model context tokens must be a positive integer or null")
        if (
            isinstance(min_generation_tokens, bool)
            or not isinstance(min_generation_tokens, int)
            or min_generation_tokens <= 0
        ):
            raise ValueError("minimum generation tokens must be a positive integer")
        if model_context_tokens is not None and token_counter is None:
            raise ValueError("model context guarding requires a token counter")
        self.backend = backend
        self.model_adapter = validate_model_adapter(model_adapter)
        self.max_output_tokens = max_output_tokens
        self.evaluated_role = evaluated_role
        self.evaluated_model = evaluated_model
        self.support_roles = frozenset(str(role) for role in support_roles)
        self.token_counter = token_counter
        self.model_context_tokens = model_context_tokens
        self.min_generation_tokens = min_generation_tokens
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.input_truncation = validate_input_truncation(input_truncation)
        if self.input_truncation and model_context_tokens is None:
            raise ConfigurationError("input truncation requires a model context limit and tokenizer")
        self.input_truncation_events: list[dict[str, Any]] = []
        if reasoning_token_allowance is not None:
            if thinking_token_budget is not None:
                raise ConfigurationError("cannot combine relay and vLLM output allowances")
            thinking_token_budget = reasoning_token_allowance
        self.reasoning_token_allowance = reasoning_token_allowance
        self.thinking_token_budget = thinking_token_budget
        if output_token_source not in ("provider", "local_content"):
            raise ConfigurationError("output_token_source must be provider or local_content")
        self.output_token_source = output_token_source
        if thinking_token_budget is not None and (
            isinstance(thinking_token_budget, bool)
            or not isinstance(thinking_token_budget, int)
            or thinking_token_budget < 0
        ):
            raise ValueError("thinking_token_budget must be a non-negative integer or null")
        self.content_only = thinking_token_budget is not None or output_token_source == "local_content"
        if self.content_only and not callable(getattr(token_counter, "count_content", None)):
            raise ConfigurationError("content-only episode budgets require a content tokenizer")
        self.used_output_tokens = 0
        self.evaluated_request_count = 0
        self.provider_usage_request_count = 0
        self.tokenizer_usage_request_count = 0
        self.fallback_usage_request_count = 0
        self.budget_overrun_tokens = 0
        self.exhaustion: Mapping[str, Any] | None = None
        self.last_prompt_tokens: int | None = None
        self.last_context_remaining_tokens: int | None = None

    @property
    def remaining_output_tokens(self) -> int:
        return max(0, self.max_output_tokens - self.used_output_tokens)

    def _is_evaluated_request(self, request: ModelRequest) -> bool:
        route_role = request.metadata.get("route_role")
        if isinstance(route_role, str) and route_role:
            return route_role == self.evaluated_role
        actor = request.metadata.get("actor")
        if actor == self.evaluated_role:
            return True
        if isinstance(actor, str) and actor in self.support_roles:
            return False
        return request.model == self.evaluated_model

    def _response_output_tokens(
        self,
        response: ModelResponse,
        *,
        reserved_tokens: int,
    ) -> tuple[int, str]:
        if self.content_only:
            assert self.token_counter is not None
            # Provider completion usage includes hidden reasoning. This budget
            # is for the answer only, regardless of whether usage is available.
            return self.token_counter.count_content(response), "local_content_tokenizer"
        usage = response.usage
        if usage is not None and usage.completion_tokens is not None:
            return usage.completion_tokens, "provider_completion_tokens"
        if (
            usage is not None
            and usage.total_tokens is not None
            and usage.prompt_tokens is not None
        ):
            return max(0, usage.total_tokens - usage.prompt_tokens), "provider_total_minus_prompt"
        if self.token_counter is not None:
            return self.token_counter.count_response(response), "local_tokenizer"
        # Reserving the complete request allowance is deliberately
        # conservative, but it keeps the cumulative cap hard even when an API
        # omits usage and its tokenizer is unavailable locally.
        return reserved_tokens, "requested_max_tokens_reservation"

    def _raise_exhausted(
        self,
        *,
        scope: str,
        episode_remaining: int,
        prompt_tokens: int | None = None,
        context_remaining: int | None = None,
    ) -> None:
        details = {
            "scope": scope,
            "episode_remaining_tokens": episode_remaining,
            "context_remaining_tokens": context_remaining,
            "prompt_tokens": prompt_tokens,
            "minimum_generation_tokens": self.min_generation_tokens,
        }
        self.exhaustion = details
        if scope == "model_context":
            message = (
                "evaluated model has fewer than "
                f"{self.min_generation_tokens} safe output tokens left in its context window"
            )
        else:
            message = (
                "evaluated model has fewer than "
                f"{self.min_generation_tokens} tokens left in its episode output budget"
            )
        raise EpisodeTokenBudgetExceeded(
            message,
            scope=scope,
            episode_remaining_tokens=episode_remaining,
            context_remaining_tokens=context_remaining,
            prompt_tokens=prompt_tokens,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not self._is_evaluated_request(request):
            return self.backend.generate(request)

        request = adapt_evaluated_request(request, self.model_adapter)
        remaining = self.remaining_output_tokens
        if remaining < self.min_generation_tokens:
            self._raise_exhausted(scope="episode_output", episode_remaining=remaining)
        prompt_tokens: int | None = None
        context_remaining: int | None = None
        # The episode allowance is for content, whereas context and API caps
        # apply to reasoning + content. Do not subtract reasoning from content.
        generation_allowance = remaining + (self.thinking_token_budget or 0)
        if self.model_context_tokens is not None:
            assert self.token_counter is not None
            prompt_tokens = self.token_counter.count_prompt(
                request,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            context_remaining = max(0, self.model_context_tokens - prompt_tokens)
            if context_remaining < self.min_generation_tokens and self.input_truncation:
                answer_reserve = min(remaining, request.max_tokens or remaining)
                output_reserve = answer_reserve + (self.thinking_token_budget or 0)
                request, event = truncate_material(
                    request,
                    count_prompt=lambda candidate: self.token_counter.count_prompt(
                        candidate, chat_template_kwargs=self.chat_template_kwargs),
                    prompt_tokens=prompt_tokens,
                    context_tokens=self.model_context_tokens,
                    reserved_output_tokens=output_reserve,
                )
                self.input_truncation_events.append(event)
                if event["status"] == "applied":
                    prompt_tokens = event["prompt_tokens_after"]
                    context_remaining = max(0, self.model_context_tokens - prompt_tokens)
            self.last_prompt_tokens = prompt_tokens
            self.last_context_remaining_tokens = context_remaining
            if context_remaining < self.min_generation_tokens:
                self._raise_exhausted(
                    scope="model_context",
                    episode_remaining=remaining,
                    prompt_tokens=prompt_tokens,
                    context_remaining=context_remaining,
                )
            generation_allowance = min(generation_allowance, context_remaining)
        request_limit = request.max_tokens
        total_request_limit = (
            None if request_limit is None
            else request_limit + (self.thinking_token_budget or 0)
        )
        effective_limit = (
            generation_allowance
            if total_request_limit is None
            else min(total_request_limit, generation_allowance)
        )
        revision = self._budget_revision()
        answer_limit = effective_limit
        if self.thinking_token_budget is not None:
            answer_limit = min(remaining, effective_limit)
            if request_limit is not None:
                answer_limit = min(answer_limit, request_limit)
        metadata = {
            **dict(request.metadata),
            _GENERATION_ALLOWANCE_METADATA_KEY: effective_limit,
            "episode_output_token_budget_revision": revision,
        }
        budgeted_request = replace(
            request,
            # Keep the answer cap until after role overrides; the API backend
            # adds thinking once, then clamps to the total allowance metadata.
            max_tokens=answer_limit,
            metadata=metadata,
        )
        response = self.backend.generate(budgeted_request)
        consumed, source = self._response_output_tokens(
            response,
            reserved_tokens=effective_limit,
        )
        self.evaluated_request_count += 1
        if source == "requested_max_tokens_reservation":
            self.fallback_usage_request_count += 1
        elif source in {"local_tokenizer", "local_content_tokenizer"}:
            self.tokenizer_usage_request_count += 1
        else:
            self.provider_usage_request_count += 1
        self.used_output_tokens += consumed
        if self.used_output_tokens > self.max_output_tokens:
            # Thinking can end early, leaving some of its allowance for content.
            # Keep the complete returned answer; stop before the next generation
            # instead of truncating the answer or corrupting structured output.
            self.budget_overrun_tokens += self.used_output_tokens - self.max_output_tokens
        return annotate_adapted_response(response, self.model_adapter)

    def _budget_revision(self) -> str:
        if self.output_token_source == "local_content":
            return LOCAL_CONTENT_BUDGET_REVISION
        return CONTENT_EPISODE_BUDGET_REVISION if self.content_only else EPISODE_BUDGET_REVISION

    def identity(self) -> Mapping[str, Any]:
        return {
            **({"model_adapter": self.model_adapter} if self.model_adapter else {}),
            "revision": self._budget_revision(),
            "scope": ("evaluated_model_content_only" if self.content_only
                      else "evaluated_model_outputs_only"),
            **({"output_token_source": "local_content"} if self.output_token_source == "local_content" else {}),
            "max_output_tokens": self.max_output_tokens,
            "single_request_limit_is_also_applied": True,
            "minimum_generation_tokens": self.min_generation_tokens,
            "model_context_tokens": self.model_context_tokens,
            "prompt_tokens_are_not_charged_to_episode_output_budget": True,
            "token_counter": (
                dict(self.token_counter.identity()) if self.token_counter is not None else None
            ),
            "chat_template_kwargs": dict(self.chat_template_kwargs),
            **({"input_truncation": self.input_truncation} if self.input_truncation else {}),
            **({"reasoning_token_allowance": self.reasoning_token_allowance,
                "reasoning_mode": "provider_default"}
               if self.reasoning_token_allowance is not None else
               {"thinking_token_budget": self.thinking_token_budget,
                "include_reasoning": False}
               if self.thinking_token_budget is not None else {}),
        }

    def usage_summary(self) -> Mapping[str, Any]:
        return {
            **dict(self.identity()),
            "used_output_tokens": self.used_output_tokens,
            "remaining_output_tokens": self.remaining_output_tokens,
            "evaluated_request_count": self.evaluated_request_count,
            "provider_usage_request_count": self.provider_usage_request_count,
            "tokenizer_usage_request_count": self.tokenizer_usage_request_count,
            "fallback_usage_request_count": self.fallback_usage_request_count,
            "budget_overrun_tokens": self.budget_overrun_tokens,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_context_remaining_tokens": self.last_context_remaining_tokens,
            "exhausted": self.exhaustion is not None,
            "exhaustion": dict(self.exhaustion) if self.exhaustion is not None else None,
            **({"input_truncation_events": list(self.input_truncation_events)}
               if self.input_truncation else {}),
        }


def attach_episode_token_budget(
    result: CaseResult,
    budget: EpisodeOutputTokenBudgetBackend,
) -> CaseResult:
    return replace(
        result,
        metadata={
            **dict(result.metadata),
            "episode_output_token_budget": dict(budget.usage_summary()),
        },
    )


def remaining_budget_from_metadata(metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get(_GENERATION_ALLOWANCE_METADATA_KEY)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _token_accounting_settings(role: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Validate and normalize evaluated-role token-accounting settings."""

    raw = role.get("token_accounting")
    if raw is None and role_thinking_token_budget(role) is not None:
        raw = default_token_accounting_config()
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConfigurationError("evaluated role token_accounting must be an object")
    kind = str(raw.get("kind") or "huggingface").strip()
    if kind != "huggingface":
        raise ConfigurationError("token_accounting.kind must be huggingface")
    output_source = raw.get("output_token_source", "provider")
    truncation = validate_input_truncation(raw.get("input_truncation"))
    if output_source not in ("provider", "local_content"):
        raise ConfigurationError("token_accounting.output_token_source must be provider or local_content")
    model_path = str(raw.get("model_path") or "").strip()
    model_id = str(raw.get("model_id") or "").strip()
    model_revision = str(raw.get("model_revision") or "").strip()
    if not model_path or not model_id or not model_revision:
        raise ConfigurationError(
            "token_accounting requires model_path, model_id, and model_revision"
        )
    local_files_only = raw.get("local_files_only", True)
    use_fast = raw.get("use_fast", True)
    if not isinstance(local_files_only, bool) or not isinstance(use_fast, bool):
        raise ConfigurationError(
            "token_accounting local_files_only and use_fast must be boolean"
        )
    context_tokens = raw.get("model_context_tokens", DEFAULT_MODEL_CONTEXT_TOKENS)
    minimum = raw.get("min_generation_tokens", DEFAULT_MIN_GENERATION_TOKENS)
    for value, label in (
        (context_tokens, "token_accounting.model_context_tokens"),
        (minimum, "token_accounting.min_generation_tokens"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"{label} must be a positive integer")
    extra_body = role.get("extra_body")
    chat_template_kwargs: Mapping[str, Any] = {}
    if isinstance(extra_body, Mapping):
        candidate = extra_body.get("chat_template_kwargs")
        if candidate is not None:
            if not isinstance(candidate, Mapping):
                raise ConfigurationError(
                    "evaluated role extra_body.chat_template_kwargs must be an object"
                )
            chat_template_kwargs = candidate
    return {
        "kind": kind,
        "model_path": model_path,
        "model_id": model_id,
        "model_revision": model_revision,
        "local_files_only": local_files_only,
        "use_fast": use_fast,
        "model_context_tokens": context_tokens,
        "min_generation_tokens": minimum,
        "chat_template_kwargs": dict(chat_template_kwargs),
        **({"output_token_source": output_source} if "output_token_source" in raw else {}),
        **({"input_truncation": truncation} if truncation else {}),
    }


def token_accounting_preflight(role: Mapping[str, Any]) -> Mapping[str, Any]:
    """Check live tokenizer prerequisites without importing or loading it."""

    settings = _token_accounting_settings(role)
    if settings is None:
        return {
            "enabled": False,
            "ready_for_local_load": True,
            "blocking_reasons": [],
        }
    blockers: list[str] = []
    if importlib.util.find_spec("transformers") is None:
        blockers.append("transformers_not_installed")
    if settings["local_files_only"] and not Path(str(settings["model_path"])).is_dir():
        blockers.append(f"model_path_not_found:{settings['model_path']}")
    return {
        "enabled": True,
        "ready_for_local_load": not blockers,
        "blocking_reasons": blockers,
        **dict(settings),
        "device": "cpu",
        "model_weights_loaded": False,
        "tokenizer_loaded_during_preflight": False,
    }


def episode_budget_options_from_role(role: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load the shared tokenizer and resolve live episode-budget options."""

    thinking_budget = role_thinking_token_budget(role)
    thinking_options = (
        {("reasoning_token_allowance" if role.get("profile") == "relay" else "thinking_token_budget"):
         thinking_budget} if thinking_budget is not None else {}
    )
    thinking_options.update(adaptation_identity(role))
    settings = _token_accounting_settings(role)
    if settings is None:
        return thinking_options
    counter = get_shared_huggingface_token_counter(
        model_path=str(settings["model_path"]),
        model_id=str(settings["model_id"]),
        model_revision=str(settings["model_revision"]),
        local_files_only=bool(settings["local_files_only"]),
        use_fast=bool(settings["use_fast"]),
    )
    return {
        **thinking_options,
        "token_counter": counter,
        "model_context_tokens": int(settings["model_context_tokens"]),
        "min_generation_tokens": int(settings["min_generation_tokens"]),
        "chat_template_kwargs": dict(settings["chat_template_kwargs"]),
        **({"output_token_source": "local_content"}
           if settings.get("output_token_source") == "local_content" else {}),
        **({"input_truncation": settings["input_truncation"]}
           if settings.get("input_truncation") else {}),
    }


__all__ = [
    "DEFAULT_EPISODE_MAX_OUTPUT_TOKENS",
    "DEFAULT_MIN_GENERATION_TOKENS",
    "DEFAULT_MODEL_CONTEXT_TOKENS",
    "EPISODE_BUDGET_REVISION",
    "EpisodeOutputTokenBudgetBackend",
    "attach_episode_token_budget",
    "episode_budget_options_from_role",
    "episode_token_scope",
    "remaining_budget_from_metadata",
    "token_accounting_preflight",
]
