"""Group-aware deterministic stratified selection and sample manifests."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..catalog import SamplingProfiles
from ..contracts import BenchmarkCase, SampleManifest, SourceManifest
from ..errors import ConfigurationError, ValidationError
from ..interfaces import Sampler
from .ids import derive_rollout_seed


@dataclass(frozen=True)
class SamplingPlan:
    benchmark_id: str
    profile: str
    result_label: str
    seed: int
    strategy: str
    target: Any
    unit: str
    strata: tuple[str, ...]
    repetitions: Any
    raw: Mapping[str, Any]


def resolve_sampling_plan(config: SamplingProfiles, profile: str, benchmark_id: str) -> SamplingPlan:
    try:
        profile_data = config.profiles[profile]
    except KeyError as exc:
        raise ConfigurationError(f"unknown sampling profile {profile!r}") from exc
    target = config.get(profile, benchmark_id)
    strata = target.get("strata")
    if not strata and profile != "default":
        strata = config.get("default", benchmark_id).get("strata")
    return SamplingPlan(
        benchmark_id=benchmark_id,
        profile=profile,
        result_label=str(profile_data["result_label"]),
        seed=int(profile_data["seed"]),
        strategy=str(target["strategy"]),
        target=target["target"],
        unit=str(target["unit"]),
        strata=tuple(strata or ()),
        repetitions=target.get("repetitions", 1),
        raw=target,
    )


def validate_no_group_leakage(partitions: Mapping[str, Sequence[BenchmarkCase]]) -> None:
    owners: dict[tuple[str, str], str] = {}
    for boundary, cases in partitions.items():
        for case in cases:
            key = (case.benchmark_id, case.group_id)
            previous = owners.setdefault(key, boundary)
            if previous != boundary:
                raise ValidationError(
                    f"group leakage: {case.benchmark_id}/{case.group_id} appears in both {previous!r} and {boundary!r}"
                )


def _rank(seed: int, benchmark_id: str, source_revision: str, profile: str, group_id: str) -> str:
    payload = "\0".join(
        ("our_eval/sampling/v1", benchmark_id, source_revision, profile, str(seed), group_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stratum_key(case: BenchmarkCase, fields: Sequence[str]) -> str:
    values = case.metadata.get("strata")
    if not isinstance(values, Mapping):
        raise ValidationError(f"case {case.case_id} has no strata mapping")
    missing = [field for field in fields if field not in values]
    if missing:
        raise ValidationError(f"case {case.case_id} lacks configured strata {missing}")
    return json.dumps({field: values[field] for field in fields}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _allocate(capacities: Mapping[str, int], target: int) -> dict[str, int]:
    if target < 0:
        raise ValidationError("numeric sampling target cannot be negative")
    total = sum(capacities.values())
    if target >= total:
        return dict(capacities)
    keys = sorted(capacities)
    quotas = {key: 0 for key in keys}
    remaining = target
    if target >= len(keys):
        for key in keys:
            quotas[key] = 1
            remaining -= 1
    residual_capacity = {key: capacities[key] - quotas[key] for key in keys}
    residual_total = sum(residual_capacity.values())
    if not remaining or not residual_total:
        return quotas
    exact = {key: remaining * residual_capacity[key] / residual_total for key in keys}
    for key in keys:
        addition = min(residual_capacity[key], math.floor(exact[key]))
        quotas[key] += addition
    seats = target - sum(quotas.values())
    order = sorted(keys, key=lambda key: (-(exact[key] - math.floor(exact[key])), key))
    while seats:
        progressed = False
        for key in order:
            if quotas[key] < capacities[key]:
                quotas[key] += 1
                seats -= 1
                progressed = True
                if seats == 0:
                    break
        if not progressed:
            raise ValidationError("unable to allocate requested sampling target")
    return quotas


class DeterministicStratifiedSampler(Sampler):
    def __init__(self, plan: SamplingPlan, source_manifest: SourceManifest) -> None:
        self.plan = plan
        self.source_manifest = source_manifest

    def select(self, cases: Sequence[BenchmarkCase]) -> tuple[Sequence[BenchmarkCase], SampleManifest]:
        if not cases:
            raise ValidationError("cannot sample an empty population")
        case_ids = [case.case_id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValidationError("duplicate case_id detected before sampling")
        for case in cases:
            if case.benchmark_id != self.plan.benchmark_id:
                raise ValidationError("sample population mixes benchmark IDs")
            if case.source_revision != self.source_manifest.source_revision:
                raise ValidationError("case/source manifest revision mismatch")
            if case.split != self.source_manifest.split:
                raise ValidationError("case/source manifest split mismatch")

        groups: dict[str, list[BenchmarkCase]] = defaultdict(list)
        for case in cases:
            groups[case.group_id].append(case)
        by_stratum: dict[str, list[str]] = defaultdict(list)
        for group_id, members in groups.items():
            keys = {_stratum_key(member, self.plan.strata) for member in members}
            if len(keys) != 1:
                raise ValidationError(f"group {group_id} crosses configured strata: {sorted(keys)}")
            by_stratum[next(iter(keys))].append(group_id)

        numeric_target = isinstance(self.plan.target, int) and not isinstance(self.plan.target, bool)
        all_strategy = self.plan.strategy in {
            "all",
            "all_groups",
            "all_manifest_entries",
            "official_stratified_all",
            "fixture_all",
            "fixture_all_groups",
        }
        if self.source_manifest.source_kind == "synthetic_fixture" and self.plan.profile != "offline_smoke":
            raise ValidationError("synthetic fixtures can only be sampled with offline_smoke")
        if self.source_manifest.source_kind == "local_compatibility" and self.plan.profile in {"default", "canonical"}:
            raise ValidationError(
                "local_compatibility data cannot silently satisfy default/canonical profiles; "
                "use an explicitly non-canonical run configuration"
            )
        target_available_units = len(groups) if (
            "group" in self.plan.strategy
            or self.plan.unit in {"conversation", "persona_chain", "task", "source_conversation", "scenario_template"}
        ) else len(cases)
        target_shortfall = bool(numeric_target and target_available_units < int(self.plan.target))
        fallback_reason = self.plan.raw.get("fallback")
        if target_shortfall and self.plan.profile in {"default", "canonical"} and not fallback_reason:
            raise ValidationError(
                f"{self.plan.benchmark_id}/{self.plan.profile} requires {self.plan.target} {self.plan.unit} units, "
                f"but only {target_available_units} are available; refusing an undeclared undersized fallback"
            )
        if all_strategy or not numeric_target:
            target_groups = len(groups)
        else:
            target_groups = min(int(self.plan.target), len(groups))
        quotas = _allocate({key: len(value) for key, value in by_stratum.items()}, target_groups)
        selected_group_ids: list[str] = []
        for stratum, group_ids in sorted(by_stratum.items()):
            ranked = sorted(
                group_ids,
                key=lambda group_id: (
                    _rank(
                        self.plan.seed,
                        self.plan.benchmark_id,
                        self.source_manifest.source_revision,
                        self.plan.profile,
                        group_id,
                    ),
                    group_id,
                ),
            )
            selected_group_ids.extend(ranked[: quotas[stratum]])
        selected_group_set = set(selected_group_ids)
        selected = sorted(
            (case for case in cases if case.group_id in selected_group_set),
            key=lambda case: (case.group_id, case.case_id),
        )
        sorted_group_ids = sorted(selected_group_set)
        repetitions = self.plan.repetitions if isinstance(self.plan.repetitions, int) else 1
        repetition_seeds = {
            case.case_id: [derive_rollout_seed(self.plan.seed, case.case_id, repetition) for repetition in range(repetitions)]
            for case in selected
        }
        population_strata = {key: len(value) for key, value in sorted(by_stratum.items())}
        selected_strata = Counter(_stratum_key(case, self.plan.strata) for case in selected)
        exclusions = []
        if target_shortfall:
            exclusions.append(
                {
                    "kind": "population_shortfall",
                    "requested": self.plan.target,
                    "available_units": target_available_units,
                    "reason": fallback_reason,
                }
            )
        manifest = SampleManifest(
            benchmark_id=self.plan.benchmark_id,
            source_revision=self.source_manifest.source_revision,
            split=self.source_manifest.split,
            profile=self.plan.profile,
            result_label=self.plan.result_label,
            algorithm="stable_hash_rank_v1",
            seed=self.plan.seed,
            target=self.plan.target,
            population_group_count=len(groups),
            population_case_count=len(cases),
            selected_group_ids=sorted_group_ids,
            selected_case_ids=[case.case_id for case in selected],
            strata=self.plan.strata,
            quotas=quotas,
            repetition_seeds=repetition_seeds,
            exclusions=tuple(exclusions),
            source_manifest_digest=self.source_manifest.digest,
            metadata={
                "source_kind": self.source_manifest.source_kind,
                "unit": self.plan.unit,
                "requested_strategy": self.plan.strategy,
                "population_exhausted": target_groups == len(groups),
                "population_strata_group_counts": population_strata,
                "selected_strata_case_counts": dict(sorted(selected_strata.items())),
                "fully_exhausted_strata": sorted(
                    key for key, capacity in population_strata.items() if quotas.get(key) == capacity
                ),
                "selected_group_count": len(sorted_group_ids),
                "selected_case_count": len(selected),
                "canonical_population": self.source_manifest.metadata.get("canonical_population"),
                "profile_target_metadata": dict(self.plan.raw),
            },
        )
        return selected, manifest
