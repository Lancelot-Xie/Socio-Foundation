"""Load an optional gitignored local environment file without dependencies."""

from __future__ import annotations

import os
from pathlib import Path

from .errors import ConfigurationError


DEFAULT_LOCAL_ENV = Path(__file__).resolve().parents[1] / ".env.local"
DEFAULT_CLUSTER_RESOURCE_BASE = Path(
    "local_data/resources"
)


def load_local_env(path: str | Path | None = None) -> Path | None:
    """Load KEY=VALUE entries without overriding an existing process environment."""

    configured = path or os.getenv("SIM_EVAL_ENV_FILE")
    env_path = Path(configured).expanduser().resolve() if configured else DEFAULT_LOCAL_ENV
    if not env_path.is_file():
        return None
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"cannot read local environment file: {env_path}") from exc
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigurationError(
                f"invalid local environment entry at {env_path}:{line_number}"
            )
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "A").isalnum() or name[0].isdigit():
            raise ConfigurationError(
                f"invalid environment variable name at {env_path}:{line_number}"
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)
    return env_path


def configure_resource_cache_defaults() -> dict[str, str]:
    """Set cluster cache defaults while preserving every explicit override.

    The fixed cluster root is used only when it exists.  A different machine can
    opt in with ``SIM_EVAL_RESOURCE_BASE`` without inheriting a nonexistent
    a machine-specific path.
    """

    configured_base = os.getenv("SIM_EVAL_RESOURCE_BASE")
    resource_base = (
        Path(configured_base).expanduser().resolve()
        if configured_base
        else DEFAULT_CLUSTER_RESOURCE_BASE
    )
    if not configured_base and not resource_base.is_dir():
        return {}

    defaults = {
        "TIKTOKEN_CACHE_DIR": str(resource_base / ".cache" / "tiktoken"),
        "NLTK_DATA": str(resource_base / ".cache" / "nltk_data"),
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    return {name: os.environ[name] for name in defaults}


__all__ = [
    "DEFAULT_CLUSTER_RESOURCE_BASE",
    "DEFAULT_LOCAL_ENV",
    "configure_resource_cache_defaults",
    "load_local_env",
]
