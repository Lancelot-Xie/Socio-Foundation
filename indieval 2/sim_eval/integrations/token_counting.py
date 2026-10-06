"""Shared CPU-only Hugging Face token counting for evaluated-model budgets."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Mapping, Protocol, Sequence

from ..contracts import ModelRequest, ModelResponse
from ..errors import ConfigurationError, OptionalDependencyError


DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TOKENIZER_MODEL_PATH = (
    "models/Qwen3-0.6B"
)
DEFAULT_TOKENIZER_REVISION = "qwen3-0.6b-shared-tokenizer-v1"
TOKEN_COUNTER_REVISION = "huggingface-chat-template-token-counter-v2"


def default_token_accounting_config() -> dict[str, Any]:
    """Return a fresh copy of the framework's evaluated-model default."""

    return {
        "kind": "huggingface",
        "model_path": DEFAULT_TOKENIZER_MODEL_PATH,
        "model_id": DEFAULT_TOKENIZER_MODEL,
        "model_revision": DEFAULT_TOKENIZER_REVISION,
        "local_files_only": True,
        "use_fast": True,
        "model_context_tokens": 32_768,
        "min_generation_tokens": 10,
    }


class TokenCounter(Protocol):
    """Read-only token counter shared by concurrent episode workers."""

    def count_prompt(
        self,
        request: ModelRequest,
        *,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> int:
        ...

    def count_response(self, response: ModelResponse) -> int:
        ...

    def count_content(self, response: ModelResponse) -> int:
        """Count only the extracted answer, excluding usage and reasoning fields."""
        ...

    def identity(self) -> Mapping[str, Any]:
        ...


def _message_payload(request: ModelRequest) -> list[dict[str, Any]]:
    return [message.to_chat_dict() for message in request.messages]


def _reasoning_text(response: ModelResponse) -> str:
    """Return provider-separated reasoning text when it is present."""

    raw = response.raw
    try:
        choices = raw.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            return ""
        first = choices[0]
        if not isinstance(first, Mapping):
            return ""
        message = first.get("message")
        if not isinstance(message, Mapping):
            return ""
        reasoning = message.get("reasoning_content")
        return reasoning if isinstance(reasoning, str) else ""
    except (IndexError, TypeError):
        return ""


class HuggingFaceChatTokenCounter:
    """Load tokenizer assets only; no model weights or CUDA device are used."""

    def __init__(
        self,
        *,
        model_path: str,
        model_id: str,
        model_revision: str,
        local_files_only: bool = True,
        use_fast: bool = True,
    ) -> None:
        normalized_path = str(model_path).strip()
        normalized_id = str(model_id).strip()
        normalized_revision = str(model_revision).strip()
        if not normalized_path or not normalized_id or not normalized_revision:
            raise ConfigurationError(
                "token accounting requires tokenizer model_path, model_id, and model_revision"
            )
        if local_files_only and not Path(normalized_path).is_dir():
            raise ConfigurationError(
                f"token accounting tokenizer directory does not exist: {normalized_path}"
            )
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise OptionalDependencyError(
                "token accounting requires transformers; install .[formal]"
            ) from exc

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                normalized_path,
                local_files_only=local_files_only,
                use_fast=use_fast,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise ConfigurationError(
                f"failed to load token accounting tokenizer from {normalized_path}: {exc}"
            ) from exc
        if not callable(getattr(self.tokenizer, "apply_chat_template", None)):
            raise ConfigurationError(
                "token accounting tokenizer must provide apply_chat_template"
            )
        if not getattr(self.tokenizer, "chat_template", None):
            raise ConfigurationError(
                "token accounting tokenizer must declare a chat_template"
            )
        self.model_path = normalized_path
        self.model_id = normalized_id
        self.model_revision = normalized_revision
        self.local_files_only = bool(local_files_only)
        self.use_fast = bool(use_fast)
        # A single instance is shared by all case workers. Serialize access to
        # tokenizer/template internals so correctness does not depend on a
        # particular transformers/tokenizers release being thread-safe.
        self._count_lock = threading.RLock()

    @staticmethod
    def _token_count(value: Any, *, label: str) -> int:
        # transformers documents both token-ID sequences and dict-like
        # BatchEncoding objects as valid apply_chat_template return types.
        # Transformers 5 may default to the latter even when tokenize=True.
        if isinstance(value, Mapping):
            if "input_ids" not in value:
                raise ConfigurationError(f"{label} result is missing input_ids")
            value = value["input_ids"]
        elif hasattr(value, "input_ids"):
            value = value.input_ids
        elif hasattr(value, "ids"):
            value = value.ids
        if hasattr(value, "tolist"):
            value = value.tolist()
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and value
            and isinstance(value[0], Sequence)
            and not isinstance(value[0], (str, bytes))
        ):
            if len(value) != 1:
                raise ConfigurationError(f"{label} unexpectedly returned a token batch")
            value = value[0]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ConfigurationError(f"{label} did not return token IDs")
        return len(value)

    def count_prompt(
        self,
        request: ModelRequest,
        *,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> int:
        kwargs = dict(chat_template_kwargs or {})
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            **kwargs,
            # Keep the counter independent of the transformers-version
            # default. _token_count still accepts BatchEncoding defensively.
            "return_dict": False,
        }
        if request.tools:
            template_kwargs["tools"] = list(request.tools)
        try:
            with self._count_lock:
                token_ids = self.tokenizer.apply_chat_template(
                    _message_payload(request),
                    **template_kwargs,
                )
        except Exception as exc:
            raise ConfigurationError(
                f"failed to count request tokens with the configured chat template: {exc}"
            ) from exc
        return self._token_count(token_ids, label="tokenizer.apply_chat_template")

    def _count_text(self, text: str) -> int:
        if not text:
            return 0
        try:
            with self._count_lock:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:
            raise ConfigurationError(
                f"failed to count generated text with the configured tokenizer: {exc}"
            ) from exc
        return self._token_count(token_ids, label="tokenizer.encode")

    def count_response(self, response: ModelResponse) -> int:
        # Count separate fields independently so their boundary cannot merge two
        # tokens. Provider usage remains authoritative whenever it is present.
        return self._count_text(_reasoning_text(response)) + self._count_text(response.text)

    def count_content(self, response: ModelResponse) -> int:
        return self._count_text(response.text)

    def identity(self) -> Mapping[str, Any]:
        return {
            "kind": "huggingface_auto_tokenizer",
            "counter_revision": TOKEN_COUNTER_REVISION,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_path": self.model_path,
            "local_files_only": self.local_files_only,
            "use_fast": self.use_fast,
            "device": "cpu",
            "model_weights_loaded": False,
        }


def get_shared_huggingface_token_counter(
    *,
    model_path: str,
    model_id: str,
    model_revision: str,
    local_files_only: bool = True,
    use_fast: bool = True,
) -> HuggingFaceChatTokenCounter:
    """Return one immutable tokenizer instance per process/configuration."""

    key = (
        str(model_path),
        str(model_id),
        str(model_revision),
        bool(local_files_only),
        bool(use_fast),
    )
    with _SHARED_COUNTER_LOCK:
        counter = _SHARED_COUNTERS.get(key)
        if counter is None:
            counter = HuggingFaceChatTokenCounter(
                model_path=model_path,
                model_id=model_id,
                model_revision=model_revision,
                local_files_only=local_files_only,
                use_fast=use_fast,
            )
            _SHARED_COUNTERS[key] = counter
        return counter


_SHARED_COUNTER_LOCK = threading.Lock()
_SHARED_COUNTERS: dict[
    tuple[str, str, str, bool, bool], HuggingFaceChatTokenCounter
] = {}


__all__ = [
    "DEFAULT_TOKENIZER_MODEL",
    "DEFAULT_TOKENIZER_MODEL_PATH",
    "DEFAULT_TOKENIZER_REVISION",
    "HuggingFaceChatTokenCounter",
    "TOKEN_COUNTER_REVISION",
    "TokenCounter",
    "default_token_accounting_config",
    "get_shared_huggingface_token_counter",
]
