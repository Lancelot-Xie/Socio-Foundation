"""Lazy optional local Hugging Face text-generation backend."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..contracts import ModelRequest, ModelResponse
from ..errors import BackendError, OptionalDependencyError
from ..interfaces import ModelBackend


class HuggingFaceBackend(ModelBackend):
    name = "huggingface"

    def __init__(
        self,
        *,
        model: str | None = None,
        device_map: str = "auto",
        pipeline_factory: Callable[..., Any] | None = None,
        pipeline_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.device_map = device_map
        self._pipeline_factory = pipeline_factory
        self.pipeline_kwargs = dict(pipeline_kwargs or {})
        self._pipeline: Any = None

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def _load(self, model: str) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        factory = self._pipeline_factory
        if factory is None:
            try:
                from transformers import pipeline as factory  # type: ignore
            except ImportError as exc:
                raise OptionalDependencyError(
                    "Hugging Face backend requires optional dependencies. "
                    "Install this project with: pip install '.[huggingface]'"
                ) from exc
        try:
            self._pipeline = factory(
                "text-generation",
                model=model,
                device_map=self.device_map,
                **self.pipeline_kwargs,
            )
        except OptionalDependencyError:
            raise
        except Exception as exc:
            raise BackendError(f"failed to initialize Hugging Face pipeline for {model!r}: {exc}") from exc
        return self._pipeline

    @staticmethod
    def _prompt(request: ModelRequest) -> str:
        if any(message.tool_calls or message.tool_call_id for message in request.messages):
            raise BackendError("plain Hugging Face backend cannot serialize native tool history; use a tool-capable API backend")
        return "\n".join(f"{message.role}: {message.content}" for message in request.messages) + "\nassistant:"

    @staticmethod
    def _extract_text(output: Any, prompt: str) -> str:
        try:
            value = output[0]["generated_text"]
        except (IndexError, KeyError, TypeError) as exc:
            raise BackendError("Hugging Face pipeline returned an unsupported response shape") from exc
        if isinstance(value, list):
            for message in reversed(value):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    return str(message.get("content", ""))
            return str(value)
        text = str(value)
        return text[len(prompt):] if text.startswith(prompt) else text

    def generate(self, request: ModelRequest) -> ModelResponse:
        model = self.model or request.model
        pipeline = self._load(model)
        prompt = self._prompt(request)
        kwargs: dict[str, Any] = {"return_full_text": True}
        if request.max_tokens is not None:
            kwargs["max_new_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
            kwargs["do_sample"] = request.temperature > 0
        started = time.perf_counter()
        try:
            output = pipeline(prompt, **kwargs)
        except Exception as exc:
            raise BackendError(f"Hugging Face generation failed: {type(exc).__name__}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ModelResponse(
            text=self._extract_text(output, prompt),
            finish_reason="generated",
            latency_ms=latency_ms,
            raw={"backend": "huggingface", "model": model},
        )
