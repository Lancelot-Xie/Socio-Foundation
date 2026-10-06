"""Shared, observable concurrency limits for API endpoints."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import replace
from typing import Any

from ..contracts import ModelRequest, ModelResponse
from ..errors import ConfigurationError
from ..interfaces import ModelBackend


ENDPOINT_CONCURRENCY_REVISION = "endpoint-model-shared-capacity-scoped-stats-v2"


def _rounded(value: float) -> float:
    return round(float(value), 3)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class _RequestMeasurements:
    """Thread-safe counters for either a shared capacity or one child scope."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._queued = 0
        self._active = 0
        self._peak_queued = 0
        self._peak_active = 0
        self._queue_wait_ms: list[float] = []
        self._service_time_ms: list[float] = []
        self._first_submitted_at: float | None = None
        self._last_finished_at: float | None = None

    def submission_started(self, *, queued_at: float) -> None:
        with self._lock:
            self._submitted += 1
            self._queued += 1
            self._peak_queued = max(self._peak_queued, self._queued)
            if self._first_submitted_at is None:
                self._first_submitted_at = queued_at

    def submission_cancelled(self) -> None:
        with self._lock:
            self._queued -= 1

    def request_acquired(self, *, wait_ms: float) -> int:
        with self._lock:
            self._queued -= 1
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)
            self._queue_wait_ms.append(wait_ms)
            return self._active

    def request_finished(self, *, acquired_at: float, succeeded: bool) -> None:
        finished_at = time.perf_counter()
        with self._lock:
            self._active -= 1
            if succeeded:
                self._completed += 1
            else:
                self._failed += 1
            self._service_time_ms.append((finished_at - acquired_at) * 1000.0)
            self._last_finished_at = finished_at

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            waits = list(self._queue_wait_ms)
            service = list(self._service_time_ms)
            first = self._first_submitted_at
            last = self._last_finished_at
            submitted = self._submitted
            completed = self._completed
            failed = self._failed
            queued = self._queued
            active = self._active
            peak_queued = self._peak_queued
            peak_active = self._peak_active
        window = max(0.0, (last - first)) if first is not None and last is not None else 0.0
        finished = completed + failed
        return {
            "requests_submitted": submitted,
            "requests_completed": completed,
            "requests_failed": failed,
            "currently_queued": queued,
            "currently_inflight": active,
            "peak_queued_requests": peak_queued,
            "peak_inflight_requests": peak_active,
            "queue_wait_ms": {
                "mean": _rounded(sum(waits) / len(waits)) if waits else 0.0,
                "p50": _rounded(_percentile(waits, 0.50)),
                "p95": _rounded(_percentile(waits, 0.95)),
                "max": _rounded(max(waits)) if waits else 0.0,
            },
            "service_time_ms": {
                "mean": _rounded(sum(service) / len(service)) if service else 0.0,
                "p50": _rounded(_percentile(service, 0.50)),
                "p95": _rounded(_percentile(service, 0.95)),
                "max": _rounded(max(service)) if service else 0.0,
            },
            "measurement_window_seconds": _rounded(window),
            "finished_requests_per_second": _rounded(finished / window) if window > 0 else 0.0,
        }


class _EndpointCapacity:
    """One physical semaphore and aggregate counters for an endpoint/model pair."""

    def __init__(self, *, base_url: str, model: str, max_inflight_requests: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_inflight_requests = max_inflight_requests
        material = f"{self.base_url}\n{self.model}".encode("utf-8")
        self.endpoint_id = hashlib.sha256(material).hexdigest()[:16]
        self._semaphore = threading.BoundedSemaphore(max_inflight_requests)
        self._measurements = _RequestMeasurements()

    def acquire(self) -> tuple[float, float, int]:
        queued_at = time.perf_counter()
        self._measurements.submission_started(queued_at=queued_at)
        try:
            self._semaphore.acquire()
        except BaseException:
            self._measurements.submission_cancelled()
            raise
        acquired_at = time.perf_counter()
        wait_ms = (acquired_at - queued_at) * 1000.0
        inflight_at_dispatch = self._measurements.request_acquired(wait_ms=wait_ms)
        return acquired_at, wait_ms, inflight_at_dispatch

    def release(self, *, acquired_at: float, succeeded: bool) -> None:
        self._measurements.request_finished(acquired_at=acquired_at, succeeded=succeeded)
        self._semaphore.release()

    def snapshot(self) -> dict[str, Any]:
        return {
            "revision": ENDPOINT_CONCURRENCY_REVISION,
            "endpoint_id": self.endpoint_id,
            "base_url": self.base_url,
            "model": self.model,
            "max_inflight_requests": self.max_inflight_requests,
            **self._measurements.snapshot(),
        }


class _EndpointCapacityPool:
    """Resolve shared capacities and reject conflicting limits suite-wide."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._capacities: dict[tuple[str, str], _EndpointCapacity] = {}

    def get(
        self,
        *,
        base_url: str,
        model: str,
        max_inflight_requests: int,
    ) -> _EndpointCapacity:
        key = (base_url.rstrip("/"), model)
        with self._lock:
            existing = self._capacities.get(key)
            if existing is not None:
                if existing.max_inflight_requests != max_inflight_requests:
                    raise ConfigurationError(
                        "roles sharing the same endpoint/model must use the same "
                        f"max_inflight_requests: endpoint={key[0]!r}, model={model!r}, "
                        f"observed={existing.max_inflight_requests} and {max_inflight_requests}"
                    )
                return existing
            capacity = _EndpointCapacity(
                base_url=key[0],
                model=model,
                max_inflight_requests=max_inflight_requests,
            )
            self._capacities[key] = capacity
            return capacity

    def capacities(self) -> tuple[_EndpointCapacity, ...]:
        with self._lock:
            return tuple(self._capacities.values())


class EndpointLimiter:
    """Measure one runner scope while consuming a suite-shared capacity."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        max_inflight_requests: int,
        _capacity: _EndpointCapacity | None = None,
    ) -> None:
        if (
            isinstance(max_inflight_requests, bool)
            or not isinstance(max_inflight_requests, int)
            or max_inflight_requests <= 0
        ):
            raise ConfigurationError("max_inflight_requests must be a positive integer")
        self._capacity = _capacity or _EndpointCapacity(
            base_url=base_url,
            model=model,
            max_inflight_requests=max_inflight_requests,
        )
        if (
            self._capacity.base_url != base_url.rstrip("/")
            or self._capacity.model != model
            or self._capacity.max_inflight_requests != max_inflight_requests
        ):
            raise ConfigurationError("endpoint limiter scope does not match its shared capacity")
        self.base_url = self._capacity.base_url
        self.model = self._capacity.model
        self.max_inflight_requests = self._capacity.max_inflight_requests
        self.endpoint_id = self._capacity.endpoint_id
        self._measurements = _RequestMeasurements()

    def acquire(self) -> tuple[float, float, int]:
        queued_at = time.perf_counter()
        self._measurements.submission_started(queued_at=queued_at)
        try:
            acquired_at, wait_ms, shared_inflight = self._capacity.acquire()
        except BaseException:
            self._measurements.submission_cancelled()
            raise
        self._measurements.request_acquired(wait_ms=wait_ms)
        return acquired_at, wait_ms, shared_inflight

    def release(self, *, acquired_at: float, succeeded: bool) -> None:
        self._measurements.request_finished(acquired_at=acquired_at, succeeded=succeeded)
        self._capacity.release(acquired_at=acquired_at, succeeded=succeeded)

    def snapshot(self) -> dict[str, Any]:
        return {
            "revision": ENDPOINT_CONCURRENCY_REVISION,
            "endpoint_id": self.endpoint_id,
            "base_url": self.base_url,
            "model": self.model,
            "max_inflight_requests": self.max_inflight_requests,
            **self._measurements.snapshot(),
        }


class EndpointLimiterRegistry:
    """Share capacities globally while retaining exact per-runner statistics."""

    def __init__(
        self,
        *,
        _capacity_pool: _EndpointCapacityPool | None = None,
        _aggregate_capacity_snapshot: bool = True,
    ) -> None:
        self._capacity_pool = _capacity_pool or _EndpointCapacityPool()
        self._aggregate_capacity_snapshot = _aggregate_capacity_snapshot
        self._lock = threading.Lock()
        self._limiters: dict[tuple[str, str], EndpointLimiter] = {}

    def fork_scope(self) -> "EndpointLimiterRegistry":
        """Return a child registry with local stats and the same physical limits."""

        return EndpointLimiterRegistry(
            _capacity_pool=self._capacity_pool,
            _aggregate_capacity_snapshot=False,
        )

    def get(
        self,
        *,
        base_url: str,
        model: str,
        max_inflight_requests: int,
    ) -> EndpointLimiter:
        key = (base_url.rstrip("/"), model)
        with self._lock:
            existing = self._limiters.get(key)
            if existing is not None:
                if existing.max_inflight_requests != max_inflight_requests:
                    raise ConfigurationError(
                        "roles sharing the same endpoint/model must use the same "
                        f"max_inflight_requests: endpoint={key[0]!r}, model={model!r}, "
                        f"observed={existing.max_inflight_requests} and {max_inflight_requests}"
                    )
                return existing
            capacity = self._capacity_pool.get(
                base_url=key[0],
                model=model,
                max_inflight_requests=max_inflight_requests,
            )
            limiter = EndpointLimiter(
                base_url=key[0],
                model=model,
                max_inflight_requests=max_inflight_requests,
                _capacity=capacity,
            )
            self._limiters[key] = limiter
            return limiter

    def snapshot(self) -> dict[str, Any]:
        if self._aggregate_capacity_snapshot:
            sources: tuple[Any, ...] = self._capacity_pool.capacities()
            scope = "shared_capacity"
        else:
            with self._lock:
                sources = tuple(self._limiters.values())
            scope = "runner_scope"
        endpoints = sorted(
            (source.snapshot() for source in sources),
            key=lambda item: (str(item["base_url"]), str(item["model"])),
        )
        return {
            "revision": ENDPOINT_CONCURRENCY_REVISION,
            "scope": scope,
            "endpoint_count": len(endpoints),
            "requests_submitted": sum(int(item["requests_submitted"]) for item in endpoints),
            "requests_completed": sum(int(item["requests_completed"]) for item in endpoints),
            "requests_failed": sum(int(item["requests_failed"]) for item in endpoints),
            "endpoints": endpoints,
        }


class ConcurrencyLimitedBackend(ModelBackend):
    """Apply an endpoint limiter without changing request or response semantics."""

    name = "endpoint_concurrency_limited"

    def __init__(self, backend: ModelBackend, *, limiter: EndpointLimiter) -> None:
        self.backend = backend
        self.limiter = limiter

    def generate(self, request: ModelRequest) -> ModelResponse:
        acquired_at, queue_wait_ms, inflight_at_dispatch = self.limiter.acquire()
        succeeded = False
        try:
            response = self.backend.generate(request)
            raw = dict(response.raw)
            sim_eval = dict(raw.get("_sim_eval") or {})
            sim_eval["endpoint_concurrency"] = {
                "revision": ENDPOINT_CONCURRENCY_REVISION,
                "endpoint_id": self.limiter.endpoint_id,
                "max_inflight_requests": self.limiter.max_inflight_requests,
                "queue_wait_ms": _rounded(queue_wait_ms),
                "inflight_at_dispatch": inflight_at_dispatch,
            }
            raw["_sim_eval"] = sim_eval
            succeeded = True
            return replace(response, raw=raw)
        finally:
            self.limiter.release(acquired_at=acquired_at, succeeded=succeeded)


__all__ = [
    "ConcurrencyLimitedBackend",
    "ENDPOINT_CONCURRENCY_REVISION",
    "EndpointLimiter",
    "EndpointLimiterRegistry",
]
