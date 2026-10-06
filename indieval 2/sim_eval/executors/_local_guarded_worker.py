"""Private child-process worker for LocalGuardedExecutor.

This module intentionally uses only the Python standard library.  Its guards
reduce accidental damage but do not provide an OS security boundary.
"""

from __future__ import annotations

import builtins
import ctypes
import importlib
import json
import os
import pathlib
import platform
import resource
import shutil
import socket
import subprocess
import sys
import sysconfig
import time
import traceback
from typing import Any


_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2
_AUDIT_ARCH_X86_64 = 0xC000003E
_AUDIT_ARCH_AARCH64 = 0xC00000B7
_LANDLOCK_CREATE_RULESET = 444
_LANDLOCK_ADD_RULE = 445
_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1

_SAFE_IMPORTS = frozenset(
    {
        "__future__",
        "abc",
        "array",
        "base64",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "io",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "os",
        "pathlib",
        "random",
        "re",
        "statistics",
        "string",
        "sys",
        "time",
        "typing",
        "unicodedata",
    }
)
_BLOCKED_STDLIB_IMPORTS = frozenset(
    {
        "_ctypes",
        "_multiprocessing",
        "_posixsubprocess",
        "_socket",
        "_ssl",
        "asyncio",
        "concurrent",
        "ctypes",
        "ftplib",
        "http",
        "imaplib",
        "importlib",
        "multiprocessing",
        "pkgutil",
        "poplib",
        "pydoc",
        "pty",
        "runpy",
        "site",
        "smtplib",
        "socket",
        "ssl",
        "subprocess",
        "telnetlib",
        "urllib",
        "webbrowser",
        "xmlrpc",
    }
)


def _disable(*_args: Any, **_kwargs: Any) -> Any:
    raise PermissionError("operation disabled by indieval local_guarded executor")


def _within(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], root: pathlib.Path) -> bool:
    try:
        pathlib.Path(path).resolve().relative_to(root)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _apply_resource_limits(limits: dict[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {"applied": {}, "unavailable": {}}

    def set_limit(name: str, soft: int, hard: int | None = None, *, required: bool = True) -> None:
        constant = getattr(resource, name, None)
        if constant is None:
            if required:
                raise RuntimeError(f"required resource limit {name} is unavailable")
            applied["unavailable"][name] = "platform_constant_unavailable"
            return
        value = (soft, soft if hard is None else hard)
        try:
            resource.setrlimit(constant, value)
            applied["applied"][name] = list(value)
        except (OSError, ValueError) as exc:
            if required:
                raise
            applied["unavailable"][name] = f"{type(exc).__name__}: {exc}"

    cpu_seconds = int(limits["cpu_seconds"])
    memory_bytes = int(limits["memory_bytes"])
    max_processes = int(limits["max_processes"])
    max_output_bytes = int(limits["max_output_bytes"])
    set_limit("RLIMIT_CPU", cpu_seconds, cpu_seconds + 1)
    memory_limit_required = sys.platform != "darwin"
    set_limit("RLIMIT_AS", memory_bytes, required=memory_limit_required)
    if hasattr(resource, "RLIMIT_DATA"):
        set_limit("RLIMIT_DATA", memory_bytes, required=memory_limit_required)
    if sys.platform != "darwin" and hasattr(resource, "RLIMIT_STACK"):
        set_limit("RLIMIT_STACK", memory_bytes)
    set_limit("RLIMIT_NPROC", max_processes)
    set_limit("RLIMIT_NOFILE", int(limits.get("max_open_files", 64)))
    set_limit("RLIMIT_FSIZE", max(max_output_bytes * 2, 1024 * 1024))
    set_limit("RLIMIT_CORE", 0)
    return applied


def _install_reliability_guards(root: pathlib.Path, *, restrict_reads: bool = False) -> None:
    original_open = builtins.open
    original_os_open = os.open

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        writes = any(flag in mode for flag in ("w", "a", "x", "+"))
        if restrict_reads and not writes and not _within(file, root):
            raise PermissionError("reads outside the temporary work directory are disabled")
        if writes and not _within(file, root):
            raise PermissionError("writes outside the temporary work directory are disabled")
        return original_open(file, mode, *args, **kwargs)

    def guarded_os_open(file: Any, flags: int, *args: Any, **kwargs: Any) -> Any:
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if restrict_reads and not flags & write_flags and not _within(file, root):
            raise PermissionError("reads outside the temporary work directory are disabled")
        if flags & write_flags and not _within(file, root):
            raise PermissionError("writes outside the temporary work directory are disabled")
        return original_os_open(file, flags, *args, **kwargs)

    builtins.open = guarded_open
    builtins.exit = _disable
    builtins.quit = _disable
    builtins.help = _disable
    builtins.input = _disable
    os.open = guarded_os_open
    for name in (
        "system",
        "popen",
        "kill",
        "killpg",
        "fork",
        "forkpty",
        "remove",
        "removedirs",
        "rmdir",
        "rename",
        "renames",
        "replace",
        "unlink",
        "truncate",
        "chmod",
        "chown",
        "fchmod",
        "fchown",
        "lchmod",
        "lchown",
        "chroot",
        "setuid",
        "setgid",
    ):
        if hasattr(os, name):
            setattr(os, name, _disable)
    for name in dir(os):
        if name.startswith(("exec", "spawn")):
            setattr(os, name, _disable)
    for name in ("Popen", "run", "call", "check_call", "check_output"):
        if hasattr(subprocess, name):
            setattr(subprocess, name, _disable)
    for name in ("rmtree", "move", "chown"):
        if hasattr(shutil, name):
            setattr(shutil, name, _disable)
    for name in ("socket", "socketpair", "create_connection"):
        if hasattr(socket, name):
            setattr(socket, name, _disable)
    os.environ["OMP_NUM_THREADS"] = "1"


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(_SockFilter))]


def _libc() -> ctypes.CDLL:
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    library.prctl.restype = ctypes.c_int
    library.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    return library


def _set_no_new_privs(library: ctypes.CDLL) -> None:
    if library.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "prctl(PR_SET_NO_NEW_PRIVS) failed")


def _drop_privileges(security: dict[str, Any]) -> dict[str, Any]:
    requested_uid = int(security["sandbox_uid"])
    requested_gid = int(security["sandbox_gid"])
    before_uid, before_gid = os.geteuid(), os.getegid()
    if bool(security.get("drop_privileges")):
        if before_uid != 0:
            raise RuntimeError("privilege drop was requested but worker is not root")
        os.setgroups([])
        os.setgid(requested_gid)
        os.setuid(requested_uid)
    after_uid, after_gid = os.geteuid(), os.getegid()
    if bool(security.get("require_non_root", True)) and after_uid == 0:
        raise RuntimeError("local_hardened_linux refuses to execute candidate as root")
    effective_capabilities = 0
    try:
        for line in pathlib.Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("CapEff:"):
                effective_capabilities = int(line.split(":", 1)[1].strip(), 16)
                break
    except OSError as exc:
        raise RuntimeError(f"cannot verify effective Linux capabilities: {exc}") from exc
    if effective_capabilities:
        raise RuntimeError(
            f"worker retained Linux effective capabilities: 0x{effective_capabilities:x}"
        )
    return {
        "uid_before": before_uid,
        "gid_before": before_gid,
        "uid_after": after_uid,
        "gid_after": after_gid,
        "privileges_dropped": before_uid == 0 and after_uid != 0,
        "supplementary_groups_after": list(os.getgroups()),
        "effective_capabilities_after": "0x0",
    }


def _preload_safe_modules() -> None:
    for module_name in sorted(_SAFE_IMPORTS):
        importlib.import_module(module_name)


def _landlock_handled_access(abi: int) -> int:
    highest_bit = 12
    if abi >= 2:
        highest_bit = 13
    if abi >= 3:
        highest_bit = 14
    if abi >= 5:
        highest_bit = 15
    return (1 << (highest_bit + 1)) - 1


def _install_landlock(library: ctypes.CDLL, temporary_root: pathlib.Path) -> dict[str, Any]:
    abi = library.syscall(
        ctypes.c_long(_LANDLOCK_CREATE_RULESET),
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    if abi < 1:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "Landlock ABI query failed")
    handled = _landlock_handled_access(int(abi))
    ruleset_attr = _LandlockRulesetAttr(handled_access_fs=handled)
    ruleset_fd = library.syscall(
        ctypes.c_long(_LANDLOCK_CREATE_RULESET),
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        ctypes.c_uint(0),
    )
    if ruleset_fd < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "Landlock ruleset creation failed")

    read_access = (1 << 2) | (1 << 3)
    temporary_access = handled & ~(1 << 0)
    allowed_roots: list[tuple[pathlib.Path, int]] = [(temporary_root, temporary_access)]
    for key in ("stdlib", "platstdlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            path = pathlib.Path(value).resolve()
            if path.exists() and all(existing != path for existing, _ in allowed_roots):
                allowed_roots.append((path, read_access))
    added_paths: list[str] = []
    try:
        for path, access in allowed_roots:
            parent_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                path_attr = _LandlockPathBeneathAttr(
                    allowed_access=access & handled,
                    parent_fd=parent_fd,
                )
                result = library.syscall(
                    ctypes.c_long(_LANDLOCK_ADD_RULE),
                    ctypes.c_int(ruleset_fd),
                    ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
                    ctypes.byref(path_attr),
                    ctypes.c_uint(0),
                )
                if result != 0:
                    error_number = ctypes.get_errno()
                    raise OSError(error_number, f"Landlock rule failed for {path}")
                added_paths.append(str(path))
            finally:
                os.close(parent_fd)
        result = library.syscall(
            ctypes.c_long(_LANDLOCK_RESTRICT_SELF),
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint(0),
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, "Landlock restriction failed")
    finally:
        os.close(ruleset_fd)
    return {
        "enabled": True,
        "abi": int(abi),
        "allowed_roots": added_paths,
        "temporary_execute_allowed": False,
    }


def _seccomp_instruction(code: int, jt: int, jf: int, value: int) -> _SockFilter:
    return _SockFilter(code=code, jt=jt, jf=jf, k=value)


def _seccomp_platform_policy() -> tuple[int, set[int], int, str]:
    machine = platform.machine().casefold()
    common = {
        425,
        426,
        427,
        428,
        429,
        430,
        431,
        432,
        433,
        434,
        435,
        438,
        442,
        443,
    }
    if machine in {"x86_64", "amd64"}:
        denied = {
            *range(41, 60),
            62,
            101,
            105,
            106,
            113,
            114,
            117,
            119,
            122,
            123,
            155,
            156,
            161,
            165,
            166,
            167,
            168,
            169,
            172,
            173,
            175,
            176,
            246,
            248,
            249,
            250,
            272,
            288,
            298,
            299,
            304,
            307,
            308,
            310,
            311,
            313,
            321,
            322,
            323,
            424,
            *common,
        }
        return _AUDIT_ARCH_X86_64, denied, 41, "x86_64"
    if machine in {"aarch64", "arm64"}:
        denied = {
            39,
            40,
            41,
            51,
            97,
            104,
            105,
            106,
            117,
            129,
            130,
            131,
            142,
            143,
            144,
            145,
            146,
            147,
            149,
            151,
            152,
            *range(198, 221),
            220,
            221,
            224,
            225,
            241,
            242,
            243,
            265,
            268,
            269,
            270,
            271,
            273,
            280,
            281,
            282,
            *common,
        }
        return _AUDIT_ARCH_AARCH64, denied, 198, "aarch64"
    raise RuntimeError(f"unsupported seccomp architecture: {machine or 'unknown'}")


def _install_seccomp(library: ctypes.CDLL) -> dict[str, Any]:
    audit_arch, denied, socket_syscall, architecture = _seccomp_platform_policy()
    # Network, new-process/exec, privilege, kernel/module, namespace, tracing,
    # io_uring, cross-process-memory, and modern mount API entry points.
    denied_syscalls = sorted(denied)
    bpf_ld_w_abs = 0x20
    bpf_jmp_jeq_k = 0x15
    bpf_ret_k = 0x06
    seccomp_ret_kill_process = 0x80000000
    seccomp_ret_errno_eperm = 0x00050000 | 1
    seccomp_ret_allow = 0x7FFF0000
    instructions = [
        _seccomp_instruction(bpf_ld_w_abs, 0, 0, 4),
        _seccomp_instruction(bpf_jmp_jeq_k, 1, 0, audit_arch),
        _seccomp_instruction(bpf_ret_k, 0, 0, seccomp_ret_kill_process),
        _seccomp_instruction(bpf_ld_w_abs, 0, 0, 0),
    ]
    for syscall_number in denied_syscalls:
        instructions.append(_seccomp_instruction(bpf_jmp_jeq_k, 0, 1, syscall_number))
        instructions.append(_seccomp_instruction(bpf_ret_k, 0, 0, seccomp_ret_errno_eperm))
    instructions.append(_seccomp_instruction(bpf_ret_k, 0, 0, seccomp_ret_allow))
    instruction_array = (_SockFilter * len(instructions))(*instructions)
    program = _SockFprog(length=len(instructions), filters=instruction_array)
    pointer = ctypes.cast(ctypes.byref(program), ctypes.c_void_p).value
    if pointer is None or library.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, pointer, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "prctl(PR_SET_SECCOMP) failed")
    return {
        "enabled": True,
        "policy_revision": f"indieval-seccomp-{architecture}-v1",
        "architecture": architecture,
        "socket_syscall": socket_syscall,
        "denied_syscall_count": len(denied_syscalls),
        "default_action": "allow",
        "deny_action": "errno_eperm",
    }


def _install_import_guard() -> None:
    original_import = builtins.__import__
    standard_library = frozenset(getattr(sys, "stdlib_module_names", ()))
    if not standard_library:
        raise RuntimeError("Python runtime does not expose sys.stdlib_module_names")

    def guarded_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        top_level = name.split(".", 1)[0]
        if level:
            origin = str((globals or {}).get("__name__") or "").split(".", 1)[0]
            if origin in standard_library and origin not in _BLOCKED_STDLIB_IMPORTS:
                return original_import(name, globals, locals, fromlist, level)
            raise ImportError(f"relative import {name!r} is not allowed by local_hardened_linux")
        if top_level not in standard_library or top_level in _BLOCKED_STDLIB_IMPORTS:
            raise ImportError(f"module {name!r} is not allowed by local_hardened_linux")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import


def _install_linux_hardening(
    security: dict[str, Any],
    temporary_root: pathlib.Path,
) -> dict[str, Any]:
    if sys.platform != "linux" or platform.machine().casefold() not in {
        "x86_64",
        "amd64",
        "aarch64",
        "arm64",
    }:
        raise RuntimeError("local_hardened_linux requires a supported Linux architecture")
    identity = _drop_privileges(security)
    _preload_safe_modules()
    library = _libc()
    _set_no_new_privs(library)
    landlock = _install_landlock(library, temporary_root)
    seccomp = _install_seccomp(library)
    try:
        with builtins.open("/etc/passwd", "rb") as handle:
            handle.read(1)
    except PermissionError:
        filesystem_probe = "blocked"
    else:
        raise RuntimeError("Landlock self-test unexpectedly read /etc/passwd")
    ctypes.set_errno(0)
    socket_result = library.syscall(
        ctypes.c_long(int(seccomp["socket_syscall"])),
        ctypes.c_int(2),
        ctypes.c_int(1),
        ctypes.c_int(0),
    )
    socket_errno = ctypes.get_errno()
    if socket_result != -1 or socket_errno != 1:
        if socket_result >= 0:
            os.close(int(socket_result))
        raise RuntimeError(
            f"seccomp self-test unexpectedly returned result={socket_result}, errno={socket_errno}"
        )
    return {
        **identity,
        "no_new_privs": True,
        "landlock": landlock,
        "seccomp": seccomp,
        "kernel_self_tests": {
            "read_etc_passwd": filesystem_probe,
            "socket_syscall": "blocked_with_eperm",
        },
        "import_policy": {
            "scope": "python_standard_library_except_blocked_modules",
            "blocked_modules": sorted(_BLOCKED_STDLIB_IMPORTS),
            "preloaded_common_modules": sorted(_SAFE_IMPORTS),
        },
    }


def main() -> int:
    if len(sys.argv) != 3:
        return 64
    request_path = pathlib.Path(sys.argv[1]).resolve()
    result_path = pathlib.Path(sys.argv[2]).resolve()
    original_open = builtins.open
    json_dumps = json.dumps
    started = time.perf_counter()
    try:
        with original_open(request_path, "r", encoding="utf-8") as handle:
            request = json.load(handle)
        root = request_path.parent.resolve()
        program = request["program"]
        if not isinstance(program, str):
            raise TypeError("program must be text")
        os.umask(0o077)
        applied_limits = _apply_resource_limits(dict(request["limits"]))
        security = dict(request.get("security") or {"mode": "local_guarded"})
        security_evidence: dict[str, Any] = {"mode": str(security.get("mode"))}
        if security.get("mode") == "local_hardened_linux":
            security_evidence.update(_install_linux_hardening(security, root))
            _install_reliability_guards(root, restrict_reads=True)
            _install_import_guard()
        elif security.get("mode") == "local_guarded":
            _install_reliability_guards(root)
        else:
            raise RuntimeError(f"unsupported worker security mode: {security.get('mode')!r}")
        sys.argv = ["candidate.py"]
        namespace: dict[str, Any] = {"__name__": "__main__", "__file__": str(root / "candidate.py")}
        try:
            compiled = compile(program, str(root / "candidate.py"), "exec")
            exec(compiled, namespace, namespace)
            result = {
                "status": "completed",
                "passed": True,
                "reason": "passed",
                "applied_limits": applied_limits,
                "security_evidence": security_evidence,
            }
        except AssertionError as exc:
            result = {
                "status": "completed",
                "passed": False,
                "reason": "failed_test",
                "detail": str(exc)[:2048],
                "applied_limits": applied_limits,
                "security_evidence": security_evidence,
            }
        except SyntaxError as exc:
            result = {
                "status": "completed",
                "passed": False,
                "reason": "compilation_error",
                "detail": f"{type(exc).__name__}: {exc}"[:2048],
                "applied_limits": applied_limits,
                "security_evidence": security_evidence,
            }
        except BaseException as exc:
            result = {
                "status": "completed",
                "passed": False,
                "reason": "runtime_error",
                "detail": f"{type(exc).__name__}: {exc}"[:2048],
                "applied_limits": applied_limits,
                "security_evidence": security_evidence,
            }
    except BaseException as exc:
        result = {
            "status": "executor_error",
            "passed": None,
            "reason": "worker_initialization_error",
            "detail": f"{type(exc).__name__}: {exc}"[:2048],
            "traceback": traceback.format_exc(limit=3)[-4096:],
        }
    result["runtime_ms"] = (time.perf_counter() - started) * 1000.0
    try:
        payload = json_dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with original_open(result_path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        return 70
    return 0 if result["status"] == "completed" else 70


if __name__ == "__main__":
    raise SystemExit(main())
