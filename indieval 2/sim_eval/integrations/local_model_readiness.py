"""No-load readiness checks for locally pinned evaluation models."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def sentence_transformer_preflight(
    model_path: str,
    *,
    model_revision: str,
) -> dict[str, Any]:
    """Check package/files without importing torch or allocating accelerator memory."""

    root = Path(str(model_path).strip())
    required_files = {
        name: (root / name).is_file()
        for name in ("config.json", "modules.json")
    }
    dependency_available = importlib.util.find_spec("sentence_transformers") is not None
    revision_pinned = bool(model_revision) and not model_revision.startswith(
        ("replace-with-", "pin-on-")
    )
    ready = (
        root.is_dir()
        and all(required_files.values())
        and dependency_available
        and revision_pinned
    )
    blockers = []
    if not revision_pinned:
        blockers.append("model_revision_not_pinned")
    if not root.is_dir():
        blockers.append("model_path_missing")
    elif not all(required_files.values()):
        blockers.append("sentence_transformer_files_incomplete")
    if not dependency_available:
        blockers.append("sentence_transformers_not_installed")
    return {
        "model_path": str(root),
        "model_revision": model_revision,
        "model_path_exists": root.is_dir(),
        "required_files": required_files,
        "dependency": "sentence-transformers",
        "dependency_available": dependency_available,
        "ready_for_local_load": ready,
        "blocking_reasons": blockers,
        "network_calls": 0,
    }


__all__ = ["sentence_transformer_preflight"]
