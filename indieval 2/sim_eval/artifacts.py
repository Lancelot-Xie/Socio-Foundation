"""Atomic manifests and append-only per-case checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import CaseResult, ResultStatus, RunManifest
from .errors import ArtifactError, DuplicateResultError, ResumeConflictError
from .json_utils import (
    UNICODE_SANITIZATION_REVISION,
    canonical_json,
    jsonable,
    sanitize_unicode_scalars,
)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write a UTF-8 file via fsync + same-directory atomic replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(value) + "\n")


def load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON artifact {source}: {exc}") from exc


def _checkpoint_record_key(item: Mapping[str, Any]) -> tuple[str, int]:
    try:
        return str(item["case_id"]), int(item["repetition"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("checkpoint record lacks a valid case_id/repetition key") from exc


def _checkpoint_record_status(item: Mapping[str, Any]) -> ResultStatus:
    try:
        return ResultStatus(item["status"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("checkpoint record lacks a valid status") from exc


def latest_checkpoint_record_dicts(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse an append-only attempt log to the latest result for each key.

    Failed attempts may be retried. A completed result accepts only a validated
    judge continuation that preserves its rollout and available scores. Skipped
    results remain terminal. Earlier attempts stay in the log for audit.
    """

    latest: dict[tuple[str, int], dict[str, Any]] = {}
    attempt_counts: dict[tuple[str, int], int] = {}
    for item in records:
        key = _checkpoint_record_key(item)
        _checkpoint_record_status(item)
        attempt = attempt_counts.get(key, 0)
        explicit_attempt = item.get("checkpoint_attempt")
        if explicit_attempt is not None and (
            isinstance(explicit_attempt, bool)
            or not isinstance(explicit_attempt, int)
            or explicit_attempt != attempt
        ):
            raise ArtifactError(
                "checkpoint attempt sequence is invalid for "
                f"case={key[0]!r}, repetition={key[1]}: expected {attempt}, got {explicit_attempt!r}"
            )
        previous = latest.get(key)
        if previous is not None and _checkpoint_record_status(previous) != ResultStatus.FAILED:
            from .judge_resume import validate_judge_append
            validate_judge_append(previous, item)
        latest[key] = dict(item)
        attempt_counts[key] = attempt + 1
    return list(latest.values())


class CheckpointStore:
    """Single-writer, append-only attempt store with strict resume identity."""

    manifest_filename = "run_manifest.json"
    records_filename = "records.jsonl"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.manifest_path = self.directory / self.manifest_filename
        self.records_path = self.directory / self.records_filename
        self._run_id: str | None = None
        self._completed: set[tuple[str, int]] | None = None
        self._attempt_counts: dict[tuple[str, int], int] | None = None
        self._latest: dict[tuple[str, int], dict[str, Any]] = {}

    def initialize(self, manifest: RunManifest) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = manifest.to_dict()
        if self.manifest_path.exists():
            existing = load_json(self.manifest_path)
            existing_id = existing.get("run_id") if isinstance(existing, dict) else None
            if existing_id != manifest.run_id:
                raise ResumeConflictError(
                    "refusing to resume into an incompatible run directory: "
                    f"existing run_id={existing_id!r}, requested run_id={manifest.run_id!r}"
                )
            existing_identity = existing.get("identity") if isinstance(existing, dict) else None
            if canonical_json(existing_identity) != canonical_json(payload["identity"]):
                raise ResumeConflictError("run_id collision or mutated identity payload detected")
        else:
            atomic_write_json(self.manifest_path, payload)
        self._run_id = manifest.run_id
        self._repair_incomplete_tail()
        records = list(self.iter_record_dicts())
        latest = latest_checkpoint_record_dicts(records)
        self._latest = {_checkpoint_record_key(item): item for item in latest}
        self._attempt_counts = {}
        for item in records:
            key = _checkpoint_record_key(item)
            self._attempt_counts[key] = self._attempt_counts.get(key, 0) + 1
        self._completed = {
            _checkpoint_record_key(item)
            for item in latest
            if _checkpoint_record_status(item) != ResultStatus.FAILED
        }

    def _repair_incomplete_tail(self) -> None:
        if not self.records_path.exists():
            return
        try:
            with self.records_path.open("rb") as handle:
                if handle.seek(0, os.SEEK_END) == 0:
                    return
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) in (b"\n", b"\r"):
                    return
                handle.seek(0)
                raw = handle.read()
        except OSError as exc:
            raise ArtifactError(f"cannot read checkpoint records: {exc}") from exc
        boundary = max(raw.rfind(b"\n"), raw.rfind(b"\r")) + 1
        try:
            json.loads(raw[boundary:].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            with self.records_path.open("r+b") as handle:
                handle.truncate(boundary)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with self.records_path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _assert_initialized(self) -> None:
        if self._run_id is None or self._completed is None or self._attempt_counts is None:
            raise ArtifactError("CheckpointStore.initialize(manifest) must be called before use")

    def append_result(self, result: CaseResult) -> None:
        self._append_result(result, judging=False)

    def append_judge_result(self, result: CaseResult) -> None:
        self._append_result(result, judging=True)

    def _append_result(self, result: CaseResult, *, judging: bool) -> None:
        self._assert_initialized()
        if result.run_id != self._run_id:
            raise ResumeConflictError(
                f"result run_id {result.run_id!r} does not match checkpoint run_id {self._run_id!r}"
            )
        key = result.checkpoint_key
        assert self._completed is not None
        assert self._attempt_counts is not None
        if judging:
            from .judge_resume import validate_judge_append
            if key not in self._latest:
                raise ArtifactError("judge continuation requires an existing checkpoint")
            validate_judge_append(self._latest[key], result.to_dict())
        elif key in self._completed:
            raise DuplicateResultError(f"result already checkpointed for case={key[0]!r}, repetition={key[1]}")
        record, unicode_stats = sanitize_unicode_scalars(result.to_dict())
        if unicode_stats["unpaired_surrogates_replaced"] or unicode_stats["surrogate_pairs_normalized"]:
            metadata = dict(record.get("metadata") or {})
            metadata["artifact_unicode_sanitization"] = {
                "revision": UNICODE_SANITIZATION_REVISION,
                **unicode_stats,
            }
            record["metadata"] = metadata
        record["checkpoint_attempt"] = self._attempt_counts.get(key, 0)
        payload = canonical_json(record) + "\n"
        try:
            with self.records_path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, UnicodeError) as exc:
            raise ArtifactError(f"failed to append checkpoint record: {exc}") from exc
        self._latest[key] = record
        self._attempt_counts[key] = record["checkpoint_attempt"] + 1
        if result.status != ResultStatus.FAILED:
            self._completed.add(key)

    def iter_record_dicts(self) -> Iterable[dict[str, Any]]:
        if not self.records_path.exists():
            return
        try:
            raw = self.records_path.read_bytes()
        except OSError as exc:
            raise ArtifactError(f"cannot read checkpoint records: {exc}") from exc
        lines = raw.splitlines(keepends=True)
        for index, encoded in enumerate(lines, start=1):
            complete = encoded.endswith((b"\n", b"\r"))
            if not encoded.strip():
                continue
            try:
                item = json.loads(encoded.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if index == len(lines) and not complete:
                    break
                raise ArtifactError(f"invalid checkpoint JSONL at line {index}: {exc}") from exc
            if not isinstance(item, dict):
                raise ArtifactError(f"checkpoint JSONL line {index} is not an object")
            yield item

    def completed_keys(self) -> frozenset[tuple[str, int]]:
        """Return terminal keys; latest failed keys are intentionally absent."""

        self._assert_initialized()
        assert self._completed is not None
        return frozenset(self._completed)

    def iter_latest_record_dicts(self) -> Iterable[dict[str, Any]]:
        """Yield only the latest attempt for each logical case/repetition key."""

        yield from latest_checkpoint_record_dicts(self.iter_record_dicts())

    def write_metrics(self, metrics: Mapping[str, Any]) -> Path:
        self._assert_initialized()
        path = self.directory / "metrics.json"
        atomic_write_json(path, {"run_id": self._run_id, "metrics": jsonable(metrics)})
        return path

    def write_report(self, markdown: str) -> Path:
        self._assert_initialized()
        path = self.directory / "report.md"
        atomic_write_text(path, markdown)
        return path
