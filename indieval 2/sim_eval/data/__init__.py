"""Manifest-driven local data ingestion and deterministic sampling."""

from .availability import AcquisitionState, acquisition_states
from .ids import derive_case_id, derive_group_id, derive_rollout_seed
from .loaders import ImportSpec, load_fixture_suite, load_import_spec, load_local_cases
from .sampling import DeterministicStratifiedSampler, resolve_sampling_plan, validate_no_group_leakage
from .schemas import SCHEMAS, probe_case

__all__ = [
    "AcquisitionState",
    "DeterministicStratifiedSampler",
    "ImportSpec",
    "SCHEMAS",
    "acquisition_states",
    "derive_case_id",
    "derive_group_id",
    "derive_rollout_seed",
    "load_fixture_suite",
    "load_import_spec",
    "load_local_cases",
    "probe_case",
    "resolve_sampling_plan",
    "validate_no_group_leakage",
]

