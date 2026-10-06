"""Task-level dataset manifest with multiple checkpoint versions per task."""

import copy
import os
from pathlib import Path

import yaml

from .upstream import canonical_task


def checkpoint_candidates(expert):
    if "checkpoints" not in expert:
        return [{"id": canonical_task(expert["task"]), **{k: expert[k] for k in ("adapter", "model") if k in expert}}]
    active = [{k: c[k] for k in ("id", "adapter", "model") if k in c}
              for c in expert["checkpoints"] if c.get("enabled", True)]
    if not active:
        raise ValueError(f"No enabled checkpoints for {expert['task']}")
    return active


def load_manifest(path, tasks):
    path = Path(path).resolve()
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != "simulation-task-experts-v1":
        raise ValueError("Expected simulation-task-experts-v1 manifest")
    root = os.environ.get("OPD_EXPERT_ROOT", manifest["checkpoint_root"])
    root = os.path.expanduser(os.path.expandvars(root))
    if "$" in root:
        raise ValueError("Unresolved OPD_EXPERT_ROOT/checkpoint_root")
    root = (path.parent / root).resolve()
    inventory, identifiers = {}, set()
    for original in manifest["tasks"]:
        expert = copy.deepcopy(original)
        task = canonical_task(expert["task"])
        if task in inventory:
            raise ValueError(f"Duplicate task in manifest: {task}")
        expert["task"] = task
        for candidate in expert["checkpoints"]:
            name = candidate.get("id")
            if not name or name in identifiers:
                raise ValueError("Each checkpoint needs a globally unique id")
            identifiers.add(name)
            if type(candidate.get("enabled", True)) is not bool:
                raise ValueError(f"{name}.enabled must be boolean")
            candidate["adapter"] = str((root / candidate["adapter"]).resolve())
        inventory[task] = expert
    selected = [canonical_task(t) for t in tasks]
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(inventory):
        raise ValueError("tasks must select unique registered tasks from expert_manifest")
    return [copy.deepcopy(inventory[t]) for t in selected], {
        "path": str(path), "checkpoint_root": str(root), "tasks": list(inventory.values()),
        "checkpoint_count": len(identifiers), "selected_tasks": selected,
        "note": "Paths supplied by the user; remote existence/base provenance must be verified by doctor on the training host."}
