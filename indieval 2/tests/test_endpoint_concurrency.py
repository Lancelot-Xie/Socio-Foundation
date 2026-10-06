import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from sim_eval.backends.concurrency import (
    ConcurrencyLimitedBackend,
    EndpointLimiterRegistry,
)
from sim_eval.contracts import ChatMessage, ModelRequest, ModelResponse
from sim_eval.errors import ConfigurationError
from sim_eval.interfaces import ModelBackend


class SlowBackend(ModelBackend):
    name = "slow_fixture"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.peak_active = 0

    def generate(self, request):
        with self._lock:
            self._active += 1
            self.peak_active = max(self.peak_active, self._active)
        try:
            time.sleep(0.04)
            return ModelResponse(
                text=request.request_id,
                finish_reason="stop",
                raw={"_sim_eval": {"protocol": "fixture"}},
            )
        finally:
            with self._lock:
                self._active -= 1


def request(index: int) -> ModelRequest:
    return ModelRequest(
        request_id=f"request-{index}",
        messages=(ChatMessage("user", "hello"),),
        model="shared-model",
        max_tokens=8,
    )


class EndpointConcurrencyTests(unittest.TestCase):
    def test_registry_shares_endpoint_model_limiters_and_rejects_conflicts(self) -> None:
        registry = EndpointLimiterRegistry()
        first = registry.get(
            base_url="http://127.0.0.1:6666/v1/",
            model="shared-model",
            max_inflight_requests=2,
        )
        second = registry.get(
            base_url="http://127.0.0.1:6666/v1",
            model="shared-model",
            max_inflight_requests=2,
        )
        self.assertIs(first, second)
        with self.assertRaisesRegex(ConfigurationError, "same max_inflight_requests"):
            registry.get(
                base_url="http://127.0.0.1:6666/v1",
                model="shared-model",
                max_inflight_requests=3,
            )

    def test_limiter_caps_requests_and_reports_queue_and_throughput(self) -> None:
        registry = EndpointLimiterRegistry()
        limiter = registry.get(
            base_url="http://127.0.0.1:6666/v1",
            model="shared-model",
            max_inflight_requests=2,
        )
        delegate = SlowBackend()
        backend = ConcurrencyLimitedBackend(delegate, limiter=limiter)
        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(backend.generate, (request(index) for index in range(6))))

        snapshot = registry.snapshot()
        endpoint = snapshot["endpoints"][0]
        self.assertEqual(delegate.peak_active, 2)
        self.assertEqual(endpoint["peak_inflight_requests"], 2)
        self.assertGreaterEqual(endpoint["peak_queued_requests"], 4)
        self.assertEqual(endpoint["requests_completed"], 6)
        self.assertEqual(endpoint["requests_failed"], 0)
        self.assertGreater(endpoint["queue_wait_ms"]["max"], 20)
        self.assertGreater(endpoint["finished_requests_per_second"], 0)
        self.assertTrue(
            all(
                response.raw["_sim_eval"]["endpoint_concurrency"]["max_inflight_requests"]
                == 2
                for response in responses
            )
        )
        self.assertTrue(
            any(
                response.raw["_sim_eval"]["endpoint_concurrency"]["queue_wait_ms"]
                > 20
                for response in responses
            )
        )

    def test_queue_wait_is_reported_separately_from_model_latency(self) -> None:
        class FixedLatencyBackend(ModelBackend):
            name = "fixed_latency"

            def generate(self, model_request):
                time.sleep(0.04)
                return ModelResponse(text="ok", latency_ms=7.0, raw={})

        registry = EndpointLimiterRegistry()
        backend = ConcurrencyLimitedBackend(
            FixedLatencyBackend(),
            limiter=registry.get(
                base_url="http://127.0.0.1:6666/v1",
                model="shared-model",
                max_inflight_requests=1,
            ),
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(backend.generate, (request(0), request(1))))

        self.assertEqual([response.latency_ms for response in responses], [7.0, 7.0])
        self.assertGreater(
            responses[1].raw["_sim_eval"]["endpoint_concurrency"]["queue_wait_ms"],
            20,
        )

    def test_failed_request_releases_capacity_for_following_requests(self) -> None:
        class FailOnceBackend(ModelBackend):
            name = "fail_once"

            def __init__(self) -> None:
                self.calls = 0

            def generate(self, model_request):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("injected failure")
                return ModelResponse(text="recovered", raw={})

        registry = EndpointLimiterRegistry()
        backend = ConcurrencyLimitedBackend(
            FailOnceBackend(),
            limiter=registry.get(
                base_url="http://127.0.0.1:6666/v1",
                model="shared-model",
                max_inflight_requests=1,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            backend.generate(request(0))
        self.assertEqual(backend.generate(request(1)).text, "recovered")
        endpoint = registry.snapshot()["endpoints"][0]
        self.assertEqual(endpoint["requests_failed"], 1)
        self.assertEqual(endpoint["requests_completed"], 1)
        self.assertEqual(endpoint["currently_inflight"], 0)

    def test_forked_scopes_share_capacity_but_keep_exact_child_counts(self) -> None:
        suite_registry = EndpointLimiterRegistry()
        first_scope = suite_registry.fork_scope()
        second_scope = suite_registry.fork_scope()
        first_limiter = first_scope.get(
            base_url="http://127.0.0.1:6666/v1",
            model="shared-model",
            max_inflight_requests=2,
        )
        second_limiter = second_scope.get(
            base_url="http://127.0.0.1:6666/v1/",
            model="shared-model",
            max_inflight_requests=2,
        )
        self.assertIsNot(first_limiter, second_limiter)
        delegate = SlowBackend()
        first_backend = ConcurrencyLimitedBackend(delegate, limiter=first_limiter)
        second_backend = ConcurrencyLimitedBackend(delegate, limiter=second_limiter)
        work = [
            (first_backend if index % 2 == 0 else second_backend, request(index))
            for index in range(8)
        ]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda item: item[0].generate(item[1]), work))

        self.assertEqual(delegate.peak_active, 2)
        self.assertEqual(first_scope.snapshot()["requests_completed"], 4)
        self.assertEqual(second_scope.snapshot()["requests_completed"], 4)
        suite = suite_registry.snapshot()
        self.assertEqual(suite["scope"], "shared_capacity")
        self.assertEqual(suite["requests_completed"], 8)
        self.assertEqual(suite["endpoints"][0]["peak_inflight_requests"], 2)

    def test_forked_scopes_reject_conflicting_suite_wide_limits(self) -> None:
        suite_registry = EndpointLimiterRegistry()
        suite_registry.fork_scope().get(
            base_url="http://127.0.0.1:6666/v1",
            model="shared-model",
            max_inflight_requests=2,
        )
        with self.assertRaisesRegex(ConfigurationError, "same max_inflight_requests"):
            suite_registry.fork_scope().get(
                base_url="http://127.0.0.1:6666/v1",
                model="shared-model",
                max_inflight_requests=3,
            )


if __name__ == "__main__":
    unittest.main()
