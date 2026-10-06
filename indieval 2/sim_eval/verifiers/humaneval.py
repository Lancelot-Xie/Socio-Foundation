"""HumanEval functional verifier with pluggable child-process execution."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..contracts import BenchmarkCase
from ..errors import ConfigurationError, ValidationError
from ..executors import (
    CodeExecutionRequest,
    CodeExecutor,
    LocalGuardedExecutor,
    LocalGuardedPolicy,
    LocalHardenedLinuxExecutor,
    LocalHardenedLinuxPolicy,
    linux_hardening_preflight,
)
from ..json_utils import sha256_digest
from .cache import VerificationCache


_FENCE = re.compile(r"```(?P<language>[A-Za-z0-9_+-]*)\s*\n(?P<code>.*?)```", re.DOTALL)


def _positive_int(value: Any, name: str, default: int) -> int:
    result = default if value is None else value
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise ConfigurationError(f"HumanEval {name} must be a positive integer")
    return result


def _positive_float(value: Any, name: str, default: float) -> float:
    result = default if value is None else value
    if isinstance(result, bool) or not isinstance(result, (int, float)) or float(result) <= 0:
        raise ConfigurationError(f"HumanEval {name} must be positive")
    return float(result)


@dataclass(frozen=True)
class HumanEvalExecutionConfig:
    enabled: bool
    backend: str
    acknowledge_not_secure: bool
    verifier_revision: str
    extraction_revision: str
    cache_enabled: bool
    cache_store_completion: bool
    cache_subdirectory: str
    python_executable: str
    local_policy: LocalGuardedPolicy | LocalHardenedLinuxPolicy

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HumanEvalExecutionConfig":
        enabled = bool(raw.get("enabled", False))
        backend = str(raw.get("backend") or "local_guarded")
        if backend not in {"local_guarded", "local_hardened_linux"}:
            raise ConfigurationError(
                f"unsupported HumanEval code_execution backend {backend!r}"
            )
        acknowledge = bool(raw.get("acknowledge_local_guarded_is_not_security_sandbox", False))
        if enabled and backend == "local_guarded" and not acknowledge:
            raise ConfigurationError(
                "enabling local_guarded requires acknowledge_local_guarded_is_not_security_sandbox=true"
            )
        network = str(raw.get("network") or ("unsupported" if backend == "local_guarded" else "seccomp_denied"))
        expected_network = "unsupported" if backend == "local_guarded" else "seccomp_denied"
        if network != expected_network:
            raise ConfigurationError(
                f"{backend} network must be declared {expected_network}"
            )
        sanitize_env = bool(raw.get("sanitize_env", True))
        temp_workdir = bool(raw.get("temp_workdir", True))
        socket_guard = bool(raw.get("python_socket_guard", True))
        verifier_revision = str(raw.get("verifier_revision") or "indieval-humaneval-verifier-v1")
        extraction_revision = str(raw.get("extraction_revision") or "indieval-python-completion-extraction-v1")
        cache_subdirectory = str(raw.get("cache_subdirectory") or "verification_cache")
        if not cache_subdirectory or Path(cache_subdirectory).name != cache_subdirectory:
            raise ConfigurationError("HumanEval cache_subdirectory must be one plain directory name")
        python_executable = str(raw.get("python_executable") or "current")
        policy_class = LocalGuardedPolicy if backend == "local_guarded" else LocalHardenedLinuxPolicy
        policy_kwargs = dict(
            timeout_seconds=_positive_float(raw.get("timeout_seconds"), "timeout_seconds", 5.0),
            cpu_seconds=_positive_int(raw.get("cpu_seconds"), "cpu_seconds", 5),
            memory_mb=_positive_int(raw.get("memory_mb"), "memory_mb", 1024),
            max_processes=_positive_int(raw.get("max_processes"), "max_processes", 8),
            max_open_files=_positive_int(raw.get("max_open_files"), "max_open_files", 64),
            max_output_bytes=_positive_int(raw.get("max_output_bytes"), "max_output_bytes", 65536),
            max_code_bytes=_positive_int(raw.get("max_code_bytes"), "max_code_bytes", 262144),
            sanitize_env=sanitize_env,
            temp_workdir=temp_workdir,
            python_socket_guard=socket_guard,
            revision=str(
                raw.get("executor_revision")
                or (
                    "indieval-local-guarded-v1"
                    if backend == "local_guarded"
                    else "indieval-local-hardened-linux-v1"
                )
            ),
        )
        if backend == "local_hardened_linux":
            policy_kwargs.update(
                sandbox_uid=_positive_int(raw.get("sandbox_uid"), "sandbox_uid", 65534),
                sandbox_gid=_positive_int(raw.get("sandbox_gid"), "sandbox_gid", 65534),
                require_non_root=bool(raw.get("require_non_root", True)),
                require_landlock=bool(raw.get("require_landlock", True)),
                require_seccomp=bool(raw.get("require_seccomp", True)),
            )
        policy = policy_class(**policy_kwargs)
        if not verifier_revision or not extraction_revision:
            raise ConfigurationError("HumanEval verifier/extraction revisions cannot be empty")
        return cls(
            enabled=enabled,
            backend=backend,
            acknowledge_not_secure=acknowledge,
            verifier_revision=verifier_revision,
            extraction_revision=extraction_revision,
            cache_enabled=bool(raw.get("cache_enabled", True)),
            cache_store_completion=bool(raw.get("cache_store_completion", True)),
            cache_subdirectory=cache_subdirectory,
            python_executable=python_executable,
            local_policy=policy,
        )

    def identity(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "verifier_revision": self.verifier_revision,
            "extraction_revision": self.extraction_revision,
            "cache_enabled": self.cache_enabled,
            "cache_store_completion": self.cache_store_completion,
            "cache_subdirectory": self.cache_subdirectory,
            "acknowledge_local_guarded_is_not_security_sandbox": self.acknowledge_not_secure,
            "executor": self.local_policy.identity(),
        }

    def preflight(self) -> dict[str, Any]:
        if not self.enabled:
            return {"backend": self.backend, "ready": False, "blocking_reasons": ["disabled"]}
        if self.backend == "local_hardened_linux":
            return linux_hardening_preflight()
        return {
            "backend": "local_guarded",
            "ready": True,
            "blocking_reasons": [],
            "warning": "best_effort_local_subprocess_not_security_sandbox",
        }

    def build_executor(self) -> CodeExecutor:
        executable = None if self.python_executable == "current" else self.python_executable
        if self.backend == "local_hardened_linux":
            if not isinstance(self.local_policy, LocalHardenedLinuxPolicy):
                raise ConfigurationError("local_hardened_linux policy type mismatch")
            return LocalHardenedLinuxExecutor(self.local_policy, python_executable=executable)
        return LocalGuardedExecutor(self.local_policy, python_executable=executable)


def extract_python_completion(text: str, *, entry_point: str) -> tuple[str, str]:
    """Return deterministic candidate code and whether it is full code or a suffix."""

    raw = str(text or "").strip("\r\n")
    if not raw.strip():
        raise ValidationError("HumanEval assistant completion is empty")
    fenced = list(_FENCE.finditer(raw))
    if fenced:
        python_blocks = [
            match.group("code").strip("\r\n")
            for match in fenced
            if match.group("language").casefold() in {"", "py", "python", "python3"}
        ]
        blocks = python_blocks or [match.group("code").strip("\r\n") for match in fenced]
        with_entry = [block for block in blocks if re.search(rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", block)]
        raw = (with_entry or blocks)[0]
    full_definition = bool(
        re.search(rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", raw)
    )
    return raw.rstrip() + "\n", "full_definition" if full_definition else "official_suffix"


class HumanEvalVerifier:
    name = "humaneval"

    def __init__(
        self,
        *,
        executor: CodeExecutor,
        verifier_revision: str,
        extraction_revision: str,
        cache: VerificationCache | None,
    ) -> None:
        self.executor = executor
        self.verifier_revision = verifier_revision
        self.extraction_revision = extraction_revision
        self.cache = cache

    @staticmethod
    def _task_payload(case: BenchmarkCase) -> tuple[str, Mapping[str, Any], str]:
        task = case.input_data.get("assistant_task")
        if not isinstance(task, Mapping) or task.get("kind") != "code":
            raise ValidationError("HumanEval verifier requires assistant_task.kind=code")
        payload = task.get("payload")
        if not isinstance(payload, Mapping) or payload.get("source") != "humaneval":
            raise ValidationError("HumanEval code task requires a source=humaneval payload")
        source_record = case.metadata.get("source_record")
        source = source_record.get("source") if isinstance(source_record, Mapping) else None
        task_id = str(source.get("task_id") or "") if isinstance(source, Mapping) else ""
        if not task_id:
            task_id = str(case.metadata.get("source_group_id") or case.group_id)
        metadata = payload.get("metadata")
        entry_point = str(metadata.get("func_name") or "") if isinstance(metadata, Mapping) else ""
        if not entry_point or not entry_point.isidentifier():
            raise ValidationError("HumanEval payload requires a valid metadata.func_name entry point")
        return task_id, payload, entry_point

    def verify_case(self, case: BenchmarkCase, assistant_completion: str) -> tuple[float | None, Mapping[str, Any]]:
        task_id, payload, entry_point = self._task_payload(case)
        prompt = payload.get("prompt")
        tests = payload.get("test")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(tests, str) or not tests.strip():
            raise ValidationError("HumanEval payload requires nonempty prompt and test code")
        completion, completion_mode = extract_python_completion(assistant_completion, entry_point=entry_point)
        completion_sha256 = hashlib.sha256(completion.encode("utf-8")).hexdigest()
        material_identity = {
            "prompt": prompt,
            "test": tests,
            "entry_point": entry_point,
        }
        material_sha256 = sha256_digest(material_identity)
        cache_identity = {
            "schema_version": "1.0",
            "task_id": task_id,
            "completion_sha256": completion_sha256,
            "task_material_sha256": material_sha256,
            "verifier_revision": self.verifier_revision,
            "extraction_revision": self.extraction_revision,
            "executor_identity": dict(self.executor.identity()),
        }
        if self.cache is not None:
            cached = self.cache.load(cache_identity)
            if cached is not None:
                return (
                    float(bool(cached["passed"])) if cached.get("passed") is not None else None,
                    {**dict(cached), "cache_hit": True},
                )
        solution = completion if completion_mode == "full_definition" else prompt.rstrip() + "\n" + completion
        program = solution.rstrip() + "\n\n" + tests.rstrip() + f"\n\ncheck({entry_point})\n"
        execution = self.executor.execute(
            CodeExecutionRequest(
                request_id=f"{task_id}:{completion_sha256[:16]}",
                program=program,
            )
        )
        completed = execution.status in {"completed", "timed_out"}
        verification = {
            "status": "completed" if completed else "unavailable",
            "passed": execution.passed if completed else None,
            "verifier_passed": execution.passed if completed else None,
            "reason": execution.reason,
            "task_id": task_id,
            "entry_point": entry_point,
            "completion_sha256": completion_sha256,
            "completion_mode": completion_mode,
            "task_material_sha256": material_sha256,
            "verifier": self.name,
            "verifier_revision": self.verifier_revision,
            "extraction_revision": self.extraction_revision,
            "executor": dict(self.executor.identity()),
            "runtime_ms": execution.runtime_ms,
            "exit_code": execution.exit_code,
            "executor_reason": execution.reason,
            "executor_detail": execution.metadata.get("detail"),
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "cache_key": VerificationCache.key_for(cache_identity),
            "cache_hit": False,
        }
        if self.cache is not None and completed:
            self.cache.store(cache_identity, verification, completion=completion)
        score = float(bool(execution.passed)) if completed and execution.passed is not None else None
        return score, verification


def build_humaneval_verifier(
    config: HumanEvalExecutionConfig,
    *,
    output_root: str | Path,
) -> HumanEvalVerifier | None:
    if not config.enabled:
        return None
    executor = config.build_executor()
    cache = (
        VerificationCache(
            Path(output_root) / config.cache_subdirectory,
            store_completion=config.cache_store_completion,
        )
        if config.cache_enabled
        else None
    )
    return HumanEvalVerifier(
        executor=executor,
        verifier_revision=config.verifier_revision,
        extraction_revision=config.extraction_revision,
        cache=cache,
    )


__all__ = [
    "HumanEvalExecutionConfig",
    "HumanEvalVerifier",
    "build_humaneval_verifier",
    "extract_python_completion",
]
