"""Load and validate the benchmark catalog and sampling profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError


REQUESTED_BENCHMARK_IDS = frozenset(
    {
        "fantom",
        "social_r1",
        "lifechoices",
        "behaviorchain",
        "alignx",
        "humanllm",
        "humanual",
        "userlm",
        "tau_usi",
        "mirrorbench",
        "coser",
        "sotopia",
        "agentsense",
    }
)
REQUIRED_PROFILES = frozenset({"canonical", "default", "offline_smoke"})


@dataclass(frozen=True)
class BenchmarkSpec:
    id: str
    display_name: str
    protocol_family: str
    source: Mapping[str, Any]
    access: Mapping[str, Any]
    official_protocol: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class BenchmarkCatalog:
    schema_version: str
    catalog_revision: str
    benchmarks: Mapping[str, BenchmarkSpec]
    raw: Mapping[str, Any]

    def get(self, benchmark_id: str) -> BenchmarkSpec:
        try:
            return self.benchmarks[benchmark_id]
        except KeyError as exc:
            choices = ", ".join(sorted(self.benchmarks))
            raise ConfigurationError(f"unknown benchmark {benchmark_id!r}; choose one of: {choices}") from exc


@dataclass(frozen=True)
class SamplingProfiles:
    schema_version: str
    catalog_revision: str
    profiles: Mapping[str, Mapping[str, Any]]
    raw: Mapping[str, Any]

    def get(self, profile: str, benchmark_id: str) -> Mapping[str, Any]:
        try:
            return self.profiles[profile]["benchmarks"][benchmark_id]
        except KeyError as exc:
            raise ConfigurationError(f"sampling target is missing for profile={profile!r}, benchmark={benchmark_id!r}") from exc


def _read_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"configuration file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"top-level value in {source} must be an object")
    return value


def load_catalog(path: str | Path, *, strict_requested_ids: bool = True) -> BenchmarkCatalog:
    raw = _read_json(path)
    for field in ("schema_version", "catalog_revision", "benchmarks"):
        if field not in raw:
            raise ConfigurationError(f"catalog is missing top-level field {field!r}")
    entries = raw["benchmarks"]
    if not isinstance(entries, list):
        raise ConfigurationError("catalog.benchmarks must be an array")
    by_id: dict[str, BenchmarkSpec] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigurationError(f"catalog benchmark #{index} must be an object")
        for field in ("id", "display_name", "protocol_family", "source", "access", "official_protocol"):
            if field not in entry:
                raise ConfigurationError(f"catalog benchmark #{index} is missing {field!r}")
        benchmark_id = entry["id"]
        if not isinstance(benchmark_id, str) or not benchmark_id:
            raise ConfigurationError(f"catalog benchmark #{index} has an invalid id")
        if benchmark_id in by_id:
            raise ConfigurationError(f"duplicate benchmark id: {benchmark_id}")
        protocol = entry["official_protocol"]
        if not isinstance(protocol, dict) or not protocol.get("metrics") or not protocol.get("primary_metric"):
            raise ConfigurationError(f"benchmark {benchmark_id} requires protocol metrics and primary_metric")
        if protocol["primary_metric"] not in protocol["metrics"]:
            raise ConfigurationError(f"benchmark {benchmark_id} primary_metric is absent from metrics")
        if not isinstance(entry["source"], dict) or "revision_status" not in entry["source"]:
            raise ConfigurationError(f"benchmark {benchmark_id} requires source.revision_status")
        if not isinstance(entry["access"], dict) or "status" not in entry["access"]:
            raise ConfigurationError(f"benchmark {benchmark_id} requires access.status")
        by_id[benchmark_id] = BenchmarkSpec(
            id=benchmark_id,
            display_name=entry["display_name"],
            protocol_family=entry["protocol_family"],
            source=entry["source"],
            access=entry["access"],
            official_protocol=protocol,
            raw=entry,
        )
    actual = frozenset(by_id)
    expected_ids = REQUESTED_BENCHMARK_IDS
    if raw.get("extension_suite") == "supplemental_v1":
        from .extensions.supplemental import BENCHMARK_IDS
        expected_ids = BENCHMARK_IDS
    if strict_requested_ids and actual != expected_ids:
        raise ConfigurationError(
            "catalog benchmark IDs differ from the requested suite; "
            f"missing={sorted(expected_ids - actual)}, extra={sorted(actual - expected_ids)}"
        )
    return BenchmarkCatalog(
        schema_version=str(raw["schema_version"]),
        catalog_revision=str(raw["catalog_revision"]),
        benchmarks=by_id,
        raw=raw,
    )


def load_sampling_profiles(path: str | Path, *, expected_ids: frozenset[str] = REQUESTED_BENCHMARK_IDS) -> SamplingProfiles:
    raw = _read_json(path)
    for field in ("schema_version", "catalog_revision", "profiles"):
        if field not in raw:
            raise ConfigurationError(f"sampling config is missing top-level field {field!r}")
    profiles = raw["profiles"]
    if not isinstance(profiles, dict):
        raise ConfigurationError("sampling profiles must be an object")
    missing_profiles = REQUIRED_PROFILES - frozenset(profiles)
    if missing_profiles:
        raise ConfigurationError(f"sampling profiles missing required names: {sorted(missing_profiles)}")
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ConfigurationError(f"sampling profile {profile_name!r} must be an object")
        if not isinstance(profile.get("seed"), int):
            raise ConfigurationError(f"sampling profile {profile_name!r} requires an integer seed")
        targets = profile.get("benchmarks")
        if not isinstance(targets, dict):
            raise ConfigurationError(f"sampling profile {profile_name!r} requires a benchmarks object")
        actual = frozenset(targets)
        if actual != expected_ids:
            raise ConfigurationError(
                f"sampling profile {profile_name!r} coverage mismatch; "
                f"missing={sorted(expected_ids - actual)}, extra={sorted(actual - expected_ids)}"
            )
        for benchmark_id, target in targets.items():
            if not isinstance(target, dict) or "strategy" not in target or "target" not in target:
                raise ConfigurationError(f"sampling target {profile_name}/{benchmark_id} requires strategy and target")
    return SamplingProfiles(
        schema_version=str(raw["schema_version"]),
        catalog_revision=str(raw["catalog_revision"]),
        profiles=profiles,
        raw=raw,
    )


def load_and_validate(catalog_path: str | Path, sampling_path: str | Path) -> tuple[BenchmarkCatalog, SamplingProfiles]:
    catalog = load_catalog(catalog_path)
    sampling = load_sampling_profiles(sampling_path, expected_ids=frozenset(catalog.benchmarks))
    if catalog.catalog_revision != sampling.catalog_revision:
        raise ConfigurationError(
            f"catalog revision {catalog.catalog_revision!r} does not match sampling revision {sampling.catalog_revision!r}"
        )
    return catalog, sampling
