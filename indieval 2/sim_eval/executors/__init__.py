"""Pluggable code-execution backends used only by explicit verifiers."""

from .base import CodeExecutionRequest, CodeExecutionResult, CodeExecutor
from .local_guarded import LocalGuardedExecutor, LocalGuardedPolicy
from .local_hardened_linux import (
    LocalHardenedLinuxExecutor,
    LocalHardenedLinuxPolicy,
    linux_hardening_preflight,
)

__all__ = [
    "CodeExecutionRequest",
    "CodeExecutionResult",
    "CodeExecutor",
    "LocalGuardedExecutor",
    "LocalGuardedPolicy",
    "LocalHardenedLinuxExecutor",
    "LocalHardenedLinuxPolicy",
    "linux_hardening_preflight",
]
