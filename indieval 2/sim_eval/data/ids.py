"""Stable identity functions that cannot inspect benchmark gold values."""

from __future__ import annotations

import hashlib

from ..errors import ValidationError


def _identity_digest(namespace: str, benchmark_id: str, source_revision: str, source_id: str) -> str:
    if not all(isinstance(value, str) and value for value in (namespace, benchmark_id, source_revision, source_id)):
        raise ValidationError("stable identity inputs must be non-empty strings")
    payload = "\0".join(("our_eval/identity/v1", namespace, benchmark_id, source_revision, source_id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_case_id(benchmark_id: str, source_revision: str, source_id: str) -> str:
    """Derive an opaque case ID only from upstream identity—not row content or gold."""

    return f"{benchmark_id}:case:{_identity_digest('case', benchmark_id, source_revision, source_id)[:24]}"


def derive_group_id(benchmark_id: str, source_revision: str, source_group_id: str) -> str:
    return f"{benchmark_id}:group:{_identity_digest('group', benchmark_id, source_revision, source_group_id)[:24]}"


def derive_rollout_seed(run_seed: int, case_id: str, repetition_index: int) -> int:
    if repetition_index < 0:
        raise ValidationError("repetition_index cannot be negative")
    payload = f"{run_seed}\0{case_id}\0{repetition_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)

