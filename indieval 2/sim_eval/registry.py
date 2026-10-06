"""Explicit adapter registry; registration has no dataset or network side effects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .errors import ConfigurationError


AdapterFactory = Callable[..., Any]
_ADAPTERS: dict[str, AdapterFactory] = {}


def register_adapter(benchmark_id: str, factory: AdapterFactory, *, replace: bool = False) -> None:
    if benchmark_id in _ADAPTERS and not replace:
        raise ConfigurationError(f"adapter already registered: {benchmark_id}")
    _ADAPTERS[benchmark_id] = factory


def adapter(benchmark_id: str) -> Callable[[AdapterFactory], AdapterFactory]:
    def decorator(factory: AdapterFactory) -> AdapterFactory:
        register_adapter(benchmark_id, factory)
        return factory

    return decorator


def get_adapter(benchmark_id: str, **kwargs: Any) -> Any:
    try:
        factory = _ADAPTERS[benchmark_id]
    except KeyError as exc:
        raise ConfigurationError(f"no adapter registered for {benchmark_id!r}") from exc
    return factory(**kwargs)


def registered_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))
