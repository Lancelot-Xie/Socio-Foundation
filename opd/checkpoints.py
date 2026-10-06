"""Commit markers and recoverable archives for interrupted local stages."""

import json
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def archive_incomplete(path):
    path = Path(path)
    destination = path.with_name(path.name + f".incomplete.{time.time_ns()}")
    path.rename(destination)
    print(f"[recovery] Preserved incomplete output: {destination}", flush=True)
    return destination


def complete_checkpoint(path):
    path = Path(path)
    try:
        progress = json.loads((path / "progress.json").read_text())
        return (progress["step"] > 0 and progress["world_size"] > 0
                and (path / "model/opd_metadata.json").is_file()
                and (path / "state").is_dir() and any((path / "state").iterdir())
                and all((path / f"sampler_rank_{rank}.json").is_file()
                        for rank in range(progress["world_size"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def latest_checkpoint(output):
    # Scan committed directories: latest.json itself may have been interrupted,
    # or may still refer to the path on the previous server.
    candidates = [p for p in Path(output).glob("step_*")
                  if p.name.removeprefix("step_").isdigit() and complete_checkpoint(p)]
    return max(candidates, key=lambda p: int(p.name[5:])) if candidates else None
