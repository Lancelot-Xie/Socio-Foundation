"""Translate audited catalog access metadata into explicit acquisition states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..catalog import BenchmarkCatalog


@dataclass(frozen=True)
class AcquisitionState:
    benchmark_id: str
    state: str
    accepted_formats: tuple[str, ...]
    source_status: str
    instructions: str
    canonical_possible: bool


def _state_for(status: str) -> tuple[str, bool]:
    lowered = status.lower()
    if "gated" in lowered:
        return "gated_requires_user_action", True
    if "not_publicly_released" in lowered or "unreleased" in lowered or "unresolved" in lowered:
        return "unavailable_or_authorized_local_required", False
    if "withheld" in lowered:
        return "authorized_local_required", False
    if "partial" in lowered or "omitted" in lowered:
        return "partial_additional_local_context_required", True
    return "ready_for_local_import", True


def acquisition_states(catalog: BenchmarkCatalog) -> Mapping[str, AcquisitionState]:
    result: dict[str, AcquisitionState] = {}
    for benchmark_id, spec in catalog.benchmarks.items():
        status = str(spec.access.get("status", "unresolved"))
        state, canonical_possible = _state_for(status)
        unblock = spec.access.get("unblock") or spec.access.get("constraints") or "Provide a validated local source manifest."
        result[benchmark_id] = AcquisitionState(
            benchmark_id=benchmark_id,
            state=state,
            accepted_formats=("jsonl", "json", "csv", "parquet"),
            source_status=status,
            instructions=str(unblock),
            canonical_possible=canonical_possible,
        )
    return result

