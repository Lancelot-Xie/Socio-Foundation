"""Best-effort local subprocess executor; explicitly not a security sandbox."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import ConfigurationError
from .base import CodeExecutionRequest, CodeExecutionResult, CodeExecutor


@dataclass(frozen=True)
class LocalGuardedPolicy:
    timeout_seconds: float = 5.0
    cpu_seconds: int = 5
    memory_mb: int = 1024
    max_processes: int = 8
    max_open_files: int = 64
    max_output_bytes: int = 65536
    max_code_bytes: int = 262144
    sanitize_env: bool = True
    temp_workdir: bool = True
    python_socket_guard: bool = True
    revision: str = "indieval-local-guarded-v1"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ConfigurationError("local_guarded timeout_seconds must be positive")
        for name in (
            "cpu_seconds",
            "memory_mb",
            "max_processes",
            "max_open_files",
            "max_output_bytes",
            "max_code_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"local_guarded {name} must be a positive integer")
        if not self.sanitize_env or not self.temp_workdir or not self.python_socket_guard:
            raise ConfigurationError(
                "local_guarded requires sanitized env, temporary cwd, and the Python socket guard"
            )
        if not self.revision:
            raise ConfigurationError("local_guarded revision cannot be empty")

    def identity(self) -> dict[str, Any]:
        return {
            "backend": "local_guarded",
            "revision": self.revision,
            "security_level": "best_effort_local_subprocess_not_security_sandbox",
            "timeout_seconds": self.timeout_seconds,
            "cpu_seconds": self.cpu_seconds,
            "memory_mb": self.memory_mb,
            "max_processes": self.max_processes,
            "max_open_files": self.max_open_files,
            "max_output_bytes": self.max_output_bytes,
            "max_code_bytes": self.max_code_bytes,
            "sanitize_env": self.sanitize_env,
            "temp_workdir": self.temp_workdir,
            "python_socket_guard": self.python_socket_guard,
            "network_namespace_isolation": False,
            "filesystem_namespace_isolation": False,
            "darwin_memory_rlimit_may_be_unavailable_and_is_reported_per_result": True,
        }


class LocalGuardedExecutor(CodeExecutor):
    name = "local_guarded"

    def __init__(self, policy: LocalGuardedPolicy, *, python_executable: str | None = None) -> None:
        if os.name != "posix":
            raise ConfigurationError("local_guarded currently requires a POSIX execution host")
        self.policy = policy
        self.revision = policy.revision
        self.python_executable = str(Path(python_executable or sys.executable).resolve())
        if not Path(self.python_executable).is_file():
            raise ConfigurationError(f"local_guarded Python executable not found: {self.python_executable}")
        self.worker_path = Path(__file__).with_name("_local_guarded_worker.py").resolve()

    def identity(self) -> Mapping[str, Any]:
        return {**self.policy.identity(), "python_executable": self.python_executable}

    @staticmethod
    def _read_bounded(path: Path, limit: int) -> tuple[str, bool]:
        try:
            raw = path.read_bytes()
        except OSError:
            return "", False
        truncated = len(raw) > limit
        return raw[:limit].decode("utf-8", errors="replace"), truncated

    def _environment(self, temporary: Path) -> dict[str, str]:
        return {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "OMP_NUM_THREADS": "1",
            "TMPDIR": str(temporary),
        }

    def _security_request(self, temporary: Path) -> dict[str, Any]:
        del temporary
        return {"mode": "local_guarded"}

    def _prepare_temporary(
        self,
        temporary: Path,
        request_path: Path,
        security: Mapping[str, Any],
    ) -> None:
        del temporary, request_path, security

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        encoded = request.program.encode("utf-8")
        if len(encoded) > self.policy.max_code_bytes:
            return CodeExecutionResult(
                status="completed",
                passed=False,
                reason="program_too_large",
                runtime_ms=0.0,
                metadata={"program_bytes": len(encoded), "limit_bytes": self.policy.max_code_bytes},
            )
        started = time.perf_counter()
        prefix = "indieval-humaneval-"
        with tempfile.TemporaryDirectory(prefix=prefix) as dirname:
            temporary = Path(dirname).resolve()
            token = uuid.uuid4().hex
            request_path = temporary / f".{token}.request.json"
            result_path = temporary / f".{token}.result.json"
            stdout_path = temporary / f".{token}.stdout"
            stderr_path = temporary / f".{token}.stderr"
            security = self._security_request(temporary)
            request_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "request_id": request.request_id,
                        "program": request.program,
                        "limits": {
                            "cpu_seconds": self.policy.cpu_seconds,
                            "memory_bytes": self.policy.memory_mb * 1024 * 1024,
                            "max_processes": self.policy.max_processes,
                            "max_open_files": self.policy.max_open_files,
                            "max_output_bytes": self.policy.max_output_bytes,
                        },
                        "security": security,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            try:
                self._prepare_temporary(temporary, request_path, security)
            except (OSError, ConfigurationError) as exc:
                return CodeExecutionResult(
                    status="executor_error",
                    passed=None,
                    reason="worker_security_preparation_error",
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                    metadata={
                        "executor_identity": dict(self.identity()),
                        "detail": f"{type(exc).__name__}: {exc}",
                    },
                )
            command = [
                self.python_executable,
                "-I",
                "-S",
                str(self.worker_path),
                str(request_path),
                str(result_path),
            ]
            timed_out = False
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=temporary,
                        env=self._environment(temporary),
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        start_new_session=True,
                        close_fds=True,
                    )
                except OSError as exc:
                    return CodeExecutionResult(
                        status="executor_error",
                        passed=None,
                        reason="worker_start_error",
                        runtime_ms=(time.perf_counter() - started) * 1000.0,
                        metadata={"executor_identity": dict(self.identity()), "detail": f"{type(exc).__name__}: {exc}"},
                    )
                try:
                    exit_code = process.wait(timeout=self.policy.timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    exit_code = process.wait()
            runtime_ms = (time.perf_counter() - started) * 1000.0
            stdout, stdout_truncated = self._read_bounded(stdout_path, self.policy.max_output_bytes)
            stderr, stderr_truncated = self._read_bounded(stderr_path, self.policy.max_output_bytes)
            common = {
                "executor_identity": dict(self.identity()),
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
            if timed_out:
                return CodeExecutionResult(
                    status="timed_out",
                    passed=False,
                    reason="timeout",
                    runtime_ms=runtime_ms,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    metadata=common,
                )
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                worker_initialized_but_terminated = exit_code != 70
                return CodeExecutionResult(
                    status="completed" if worker_initialized_but_terminated else "executor_error",
                    passed=False if worker_initialized_but_terminated else None,
                    reason=(
                        "worker_terminated_during_candidate_execution"
                        if worker_initialized_but_terminated
                        else "missing_or_invalid_worker_result"
                    ),
                    runtime_ms=runtime_ms,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    metadata={**common, "detail": f"{type(exc).__name__}: {exc}"},
                )
            status = str(payload.get("status") or "executor_error")
            passed = payload.get("passed")
            if passed is not None and not isinstance(passed, bool):
                status, passed = "executor_error", None
            return CodeExecutionResult(
                status=status,
                passed=passed,
                reason=str(payload.get("reason") or "unknown"),
                runtime_ms=float(payload.get("runtime_ms", runtime_ms)),
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                metadata={
                    **common,
                    "detail": payload.get("detail"),
                    "applied_limits": payload.get("applied_limits", {}),
                    "security_evidence": payload.get("security_evidence", {}),
                },
            )


__all__ = ["LocalGuardedExecutor", "LocalGuardedPolicy"]
