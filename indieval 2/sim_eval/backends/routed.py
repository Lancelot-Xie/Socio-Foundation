"""Route one interactive episode to independently configured role backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import threading
from typing import Any, Sequence

from ..contracts import ModelRequest, ModelResponse
from ..errors import ConfigurationError
from ..interfaces import ModelBackend
from .episode_budget import remaining_budget_from_metadata


def _respect_episode_budget(request: ModelRequest) -> ModelRequest:
    remaining = remaining_budget_from_metadata(request.metadata)
    if remaining is None:
        return request
    effective = remaining if request.max_tokens is None else min(request.max_tokens, remaining)
    return replace(request, max_tokens=effective)


class _AttemptTraceMixin:
    def _initialize_attempt_trace(self) -> None:
        self._attempt_trace = threading.local()

    def begin_attempt(self) -> None:
        self._attempt_trace.events = []

    def end_attempt(self) -> tuple[Mapping[str, Any], ...]:
        events = tuple(getattr(self._attempt_trace, "events", ()))
        self._attempt_trace.events = []
        return events

    def _record_response(self, role: str, request: ModelRequest, response: ModelResponse) -> None:
        events = getattr(self._attempt_trace, "events", None)
        if events is None or not isinstance(response.raw, Mapping):
            return
        api = response.raw.get("_sim_eval")
        if not isinstance(api, Mapping):
            return
        events.append(
            {
                "route_role": role,
                "request_id": request.request_id,
                **dict(api),
            }
        )


class RoleRoutedBackend(_AttemptTraceMixin, ModelBackend):
    """Use a distinct backend for the evaluated model and fixed partner."""

    name = "role_routed"

    def __init__(
        self,
        *,
        evaluated_backend: ModelBackend,
        fixed_assistant_backend: ModelBackend,
        evaluated_request_overrides: Mapping[str, Any] | None = None,
        fixed_assistant_request_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self._initialize_attempt_trace()
        self.evaluated_backend = evaluated_backend
        self.fixed_assistant_backend = fixed_assistant_backend
        self.evaluated_request_overrides = dict(evaluated_request_overrides or {})
        self.fixed_assistant_request_overrides = dict(fixed_assistant_request_overrides or {})

    def route_role_for_request(self, request: ModelRequest) -> str:
        actor = request.metadata.get("actor")
        if actor in {"fixed_assistant", "evaluated_user"}:
            return str(actor)
        raise ConfigurationError(f"role-routed backend received unknown actor {actor!r}")

    def generate(self, request: ModelRequest) -> ModelResponse:
        actor = self.route_role_for_request(request)
        if actor == "fixed_assistant":
            routed_request = (
                replace(request, **self.fixed_assistant_request_overrides)
                if self.fixed_assistant_request_overrides
                else request
            )
            routed_request = _respect_episode_budget(routed_request)
            response = self.fixed_assistant_backend.generate(routed_request)
            self._record_response("fixed_assistant", routed_request, response)
            return response
        if actor == "evaluated_user":
            routed_request = (
                replace(request, **self.evaluated_request_overrides)
                if self.evaluated_request_overrides
                else request
            )
            routed_request = _respect_episode_budget(routed_request)
            response = self.evaluated_backend.generate(routed_request)
            self._record_response("evaluated_user", routed_request, response)
            return response
        raise AssertionError("unreachable role-routed actor")


class NamedRoleRoutedBackend(_AttemptTraceMixin, ModelBackend):
    """Route by explicit role, pinned judge model, configured actor, or default."""

    name = "named_role_routed"

    def __init__(
        self,
        *,
        roles: Mapping[str, ModelBackend],
        default_role: str | None = None,
        model_routes: Mapping[str, str] | None = None,
        request_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._initialize_attempt_trace()
        self.roles = dict(roles)
        self.default_role = default_role
        self.model_routes = dict(model_routes or {})
        self.request_overrides = {
            str(role): dict(values) for role, values in (request_overrides or {}).items()
        }
        if not self.roles:
            raise ConfigurationError("named role router requires at least one role backend")
        if default_role is not None and default_role not in self.roles:
            raise ConfigurationError(f"unknown default role {default_role!r}")
        unknown_routes = set(self.model_routes.values()) - set(self.roles)
        if unknown_routes:
            raise ConfigurationError(f"model routes reference unknown roles: {sorted(unknown_routes)}")
        unknown_overrides = set(self.request_overrides) - set(self.roles)
        if unknown_overrides:
            raise ConfigurationError(f"request overrides reference unknown roles: {sorted(unknown_overrides)}")

    def route_role_for_request(self, request: ModelRequest) -> str:
        requested_role = request.metadata.get("route_role")
        actor = requested_role if requested_role in self.roles else None
        if actor is None:
            actor = self.model_routes.get(request.model)
        if actor is None:
            requested_actor = request.metadata.get("actor")
            actor = requested_actor if requested_actor in self.roles else self.default_role
        if actor not in self.roles:
            raise ConfigurationError(
                f"no backend configured for request actor {actor!r}; available roles={sorted(self.roles)}"
            )
        return str(actor)

    def generate(self, request: ModelRequest) -> ModelResponse:
        role_name = self.route_role_for_request(request)
        overrides = dict(self.request_overrides.get(role_name, {}))
        if request.metadata.get("preserve_request_sampling"):
            for field in ("temperature", "top_p", "reasoning_effort"):
                overrides.pop(field, None)
        routed_request = replace(request, **overrides) if overrides else request
        routed_request = _respect_episode_budget(routed_request)
        response = self.roles[role_name].generate(routed_request)
        self._record_response(role_name, routed_request, response)
        return response


class AttemptScopedBackend(ModelBackend):
    """Collect one episode's API provenance across all of its worker threads.

    The role routers' legacy ``begin_attempt``/``end_attempt`` API is
    thread-local, which is correct for case-level parallelism but cannot see a
    bounded nested fan-out such as AgentSense's three independent judges.  One
    instance of this wrapper is therefore created per episode.  It delegates to
    the shared, thread-safe role router and keeps only that episode's successful
    API response metadata behind a lock.
    """

    name = "attempt_scoped"

    def __init__(self, backend: ModelBackend, *, default_role: str | None = None) -> None:
        route = getattr(backend, "route_role_for_request", None)
        self.backend = backend
        self._default_role = str(default_role or "").strip() or None
        self._route_role_for_request = route if callable(route) else self._metadata_role
        self._events: list[Mapping[str, Any]] = []
        self._lock = threading.Lock()

    def _metadata_role(self, request: ModelRequest) -> str:
        role = request.metadata.get("route_role") or request.metadata.get("actor")
        if (not isinstance(role, str) or not role) and self._default_role is not None:
            return self._default_role
        if not isinstance(role, str) or not role:
            raise ConfigurationError(
                "attempt-scoped provenance cannot infer the request role"
            )
        return role

    def generate(self, request: ModelRequest) -> ModelResponse:
        role = str(self._route_role_for_request(request))
        response = self.backend.generate(request)
        if isinstance(response.raw, Mapping):
            api = response.raw.get("_sim_eval")
            if isinstance(api, Mapping):
                event = {
                    "route_role": role,
                    "request_id": request.request_id,
                    **dict(api),
                }
                with self._lock:
                    self._events.append(event)
        return response

    def events(self) -> Sequence[Mapping[str, Any]]:
        with self._lock:
            return tuple(self._events)


__all__ = ["AttemptScopedBackend", "NamedRoleRoutedBackend", "RoleRoutedBackend"]
