"""Fail-closed Linux kernel hardening for local HumanEval execution.

This backend deliberately has no portable fallback.  It combines a dedicated
unprivileged identity (when the parent is root), Landlock filesystem rules,
``no_new_privs``, and a seccomp-BPF syscall deny list.  It is still intended
for benchmark-sized Python snippets, not arbitrary production multi-tenant
workloads.
"""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import ConfigurationError
from .local_guarded import LocalGuardedExecutor, LocalGuardedPolicy


_SUPPORTED_MACHINES = {"x86_64", "amd64", "aarch64", "arm64"}
_LANDLOCK_CREATE_RULESET = 444
_LANDLOCK_CREATE_RULESET_VERSION = 1


def linux_hardening_preflight() -> dict[str, Any]:
    """Inspect availability without changing the current process."""

    blockers: list[str] = []
    machine = platform.machine().casefold()
    linux_host = platform.system() == "Linux"
    supported_machine = machine in _SUPPORTED_MACHINES
    if not linux_host:
        blockers.append("host_is_not_linux")
    if not supported_machine:
        blockers.append(f"unsupported_machine:{machine or 'unknown'}")
    landlock_abi: int | None = None
    if not blockers:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        result = libc.syscall(
            ctypes.c_long(_LANDLOCK_CREATE_RULESET),
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
        )
        if result < 1:
            error_number = ctypes.get_errno()
            blockers.append(f"landlock_unavailable:errno_{error_number}")
        else:
            landlock_abi = int(result)
        actions = Path("/proc/sys/kernel/seccomp/actions_avail")
        try:
            available_actions = actions.read_text(encoding="utf-8").split()
        except OSError as exc:
            blockers.append(f"seccomp_status_unavailable:{type(exc).__name__}")
            available_actions = []
        if "allow" not in available_actions or not ({"kill_process", "kill_thread"} & set(available_actions)):
            blockers.append("seccomp_filter_actions_unavailable")
    return {
        "backend": "local_hardened_linux",
        "platform": platform.system(),
        "machine": machine,
        "effective_uid": os.geteuid() if hasattr(os, "geteuid") else None,
        "landlock_abi": landlock_abi,
        "seccomp_filter_available": (
            linux_host
            and supported_machine
            and not any(reason.startswith("seccomp") for reason in blockers)
        ),
        "ready": not blockers,
        "blocking_reasons": blockers,
    }


@dataclass(frozen=True)
class LocalHardenedLinuxPolicy(LocalGuardedPolicy):
    sandbox_uid: int = 65534
    sandbox_gid: int = 65534
    require_non_root: bool = True
    require_landlock: bool = True
    require_seccomp: bool = True
    revision: str = "indieval-local-hardened-linux-v1"

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("sandbox_uid", "sandbox_gid"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"local_hardened_linux {name} must be a positive integer")
        if not self.require_non_root or not self.require_landlock or not self.require_seccomp:
            raise ConfigurationError(
                "local_hardened_linux requires non-root execution, Landlock, and seccomp"
            )

    def identity(self) -> dict[str, Any]:
        base = super().identity()
        base.pop("darwin_memory_rlimit_may_be_unavailable_and_is_reported_per_result", None)
        return {
            **base,
            "backend": "local_hardened_linux",
            "security_level": "linux_kernel_hardened_subprocess",
            "sandbox_uid": self.sandbox_uid,
            "sandbox_gid": self.sandbox_gid,
            "require_non_root": self.require_non_root,
            "require_landlock": self.require_landlock,
            "require_seccomp": self.require_seccomp,
            "no_new_privs": True,
            "filesystem_isolation": "landlock_allowlist",
            "network_isolation": "seccomp_syscall_deny",
            "network_namespace_isolation": False,
            "filesystem_namespace_isolation": False,
            "fail_closed": True,
        }


class LocalHardenedLinuxExecutor(LocalGuardedExecutor):
    name = "local_hardened_linux"

    def __init__(
        self,
        policy: LocalHardenedLinuxPolicy,
        *,
        python_executable: str | None = None,
    ) -> None:
        preflight = linux_hardening_preflight()
        if not preflight["ready"]:
            reasons = ", ".join(preflight["blocking_reasons"])
            raise ConfigurationError(f"local_hardened_linux is unavailable: {reasons}")
        super().__init__(policy, python_executable=python_executable)
        self.policy = policy
        self.preflight = preflight

    def identity(self) -> Mapping[str, Any]:
        return {
            **super().identity(),
            "host_preflight": self.preflight,
        }

    def _security_request(self, temporary: Path) -> dict[str, Any]:
        parent_uid = os.geteuid()
        parent_gid = os.getegid()
        sandbox_uid = self.policy.sandbox_uid if parent_uid == 0 else parent_uid
        sandbox_gid = self.policy.sandbox_gid if parent_uid == 0 else parent_gid
        return {
            "mode": "local_hardened_linux",
            "sandbox_uid": sandbox_uid,
            "sandbox_gid": sandbox_gid,
            "drop_privileges": parent_uid == 0,
            "require_non_root": self.policy.require_non_root,
            "require_landlock": self.policy.require_landlock,
            "require_seccomp": self.policy.require_seccomp,
            "temporary_root": str(temporary),
        }

    def _prepare_temporary(
        self,
        temporary: Path,
        request_path: Path,
        security: Mapping[str, Any],
    ) -> None:
        if not security.get("drop_privileges"):
            if os.geteuid() == 0:
                raise ConfigurationError("local_hardened_linux refused to execute candidate as root")
            return
        uid = int(security["sandbox_uid"])
        gid = int(security["sandbox_gid"])
        os.chown(temporary, uid, gid)
        os.chmod(temporary, 0o700)
        os.chown(request_path, uid, gid)
        os.chmod(request_path, 0o400)


__all__ = [
    "LocalHardenedLinuxExecutor",
    "LocalHardenedLinuxPolicy",
    "linux_hardening_preflight",
]
