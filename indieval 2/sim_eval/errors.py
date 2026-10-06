"""Structured exception hierarchy for the framework."""


class SimEvalError(Exception):
    """Base class for expected framework errors."""


class ConfigurationError(SimEvalError):
    """Configuration is invalid or internally inconsistent."""


class ValidationError(SimEvalError):
    """An input artifact does not satisfy its declared contract."""


class AvailabilityError(SimEvalError):
    """Required official or authorized data is not locally available."""


class OptionalDependencyError(SimEvalError):
    """An explicitly requested optional integration is not installed."""


class BackendError(SimEvalError):
    """A model backend failed before returning a valid response."""


class EpisodeTokenBudgetExhausted(SimEvalError):
    """Normal control flow: the evaluated policy has no safe generation room."""

    def __init__(
        self,
        message: str,
        *,
        scope: str,
        episode_remaining_tokens: int,
        context_remaining_tokens: int | None = None,
        prompt_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.scope = scope
        self.episode_remaining_tokens = episode_remaining_tokens
        self.context_remaining_tokens = context_remaining_tokens
        self.prompt_tokens = prompt_tokens

    def details(self) -> dict[str, int | str | None]:
        return {
            "scope": self.scope,
            "episode_remaining_tokens": self.episode_remaining_tokens,
            "context_remaining_tokens": self.context_remaining_tokens,
            "prompt_tokens": self.prompt_tokens,
        }


class EpisodeTokenBudgetExceeded(EpisodeTokenBudgetExhausted):
    """Backward-compatible exception name for a normally exhausted budget."""


class BackendHTTPError(BackendError):
    """An HTTP model endpoint returned a non-success response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.error_type = error_type


class BackendSafetyError(BackendHTTPError):
    """A provider explicitly blocked a request or response for safety."""


class BackendStructuredOutputError(BackendError):
    """A response did not satisfy the requested JSON transport contract."""


class BackendTokenLimitError(BackendError):
    """A model response was truncated by its per-request output-token limit."""


class BackendTimeoutError(BackendError):
    """A backend exceeded its request timeout."""


class InjectedBackendError(BackendError):
    """An offline replay test intentionally injected a failure."""


class ParseError(SimEvalError):
    """A model response could not be parsed for a benchmark."""


class ContextLimitError(SimEvalError):
    """Protected prompt context cannot fit within the declared context budget."""


class ArtifactError(SimEvalError):
    """A run artifact could not be safely read or written."""


class ResumeConflictError(ArtifactError):
    """Existing artifacts belong to an incompatible run identity."""


class DuplicateResultError(ArtifactError):
    """A case/repetition result was already checkpointed."""
