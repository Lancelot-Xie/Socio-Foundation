"""Typed boundary between task verification and untrusted-code execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CodeExecutionRequest:
    request_id: str
    program: str


@dataclass(frozen=True)
class CodeExecutionResult:
    status: str
    passed: bool | None
    reason: str
    runtime_ms: float
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "reason": self.reason,
            "runtime_ms": self.runtime_ms,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "metadata": dict(self.metadata),
        }


class CodeExecutor(ABC):
    name: str
    revision: str

    @abstractmethod
    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        raise NotImplementedError

    @abstractmethod
    def identity(self) -> Mapping[str, Any]:
        raise NotImplementedError

