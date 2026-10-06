"""Lazy model-backend registry."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ..errors import ConfigurationError


BackendFactory = Callable[..., Any]
_CUSTOM: dict[str, BackendFactory] = {}
_BUILTINS = {
    "replay": "sim_eval.backends.replay:ReplayBackend",
    "openai": "sim_eval.backends.openai_compatible:OpenAICompatibleBackend",
    "chat_completions": "sim_eval.backends.openai_compatible:OpenAICompatibleBackend",
    "openai_compatible": "sim_eval.backends.openai_compatible:OpenAICompatibleBackend",
    "openai_responses": "sim_eval.backends.openai_responses:OpenAIResponsesBackend",
    "vllm": "sim_eval.backends.openai_compatible:VLLMBackend",
    "huggingface": "sim_eval.backends.huggingface:HuggingFaceBackend",
    "hf": "sim_eval.backends.huggingface:HuggingFaceBackend",
}


def _resolve(path: str) -> BackendFactory:
    module_name, attribute = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def register_backend(name: str, factory: BackendFactory, *, replace: bool = False) -> None:
    if (name in _CUSTOM or name in _BUILTINS) and not replace:
        raise ConfigurationError(f"backend already registered: {name}")
    _CUSTOM[name] = factory


def get_backend(name: str, **kwargs: Any) -> Any:
    factory = _CUSTOM.get(name)
    if factory is None:
        path = _BUILTINS.get(name)
        if path is None:
            choices = ", ".join(registered_backends())
            raise ConfigurationError(f"unknown backend {name!r}; choose one of: {choices}")
        factory = _resolve(path)
    return factory(**kwargs)


def registered_backends() -> tuple[str, ...]:
    return tuple(sorted(set(_BUILTINS) | set(_CUSTOM)))


__all__ = ["get_backend", "register_backend", "registered_backends"]
