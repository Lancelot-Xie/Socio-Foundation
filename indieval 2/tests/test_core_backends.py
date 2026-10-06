import unittest

from sim_eval.backends import get_backend, registered_backends
from sim_eval.backends.huggingface import HuggingFaceBackend
from sim_eval.backends.openai_compatible import OpenAICompatibleBackend, VLLMBackend
from sim_eval.contracts import ChatMessage, ModelRequest
from sim_eval.errors import BackendTimeoutError, InjectedBackendError


def request(request_id="request-1"):
    return ModelRequest(
        request_id=request_id,
        messages=[ChatMessage("system", "Be concise."), ChatMessage("user", "Choose A or B")],
        model="fixture-model",
        temperature=0,
        max_tokens=8,
        seed=17,
    )


class BackendTests(unittest.TestCase):
    def test_replay_is_deterministic_and_missing_keys_fail(self) -> None:
        backend = get_backend("replay", responses={"request-1": {"text": "A", "usage": {"total_tokens": 3}}})
        self.assertEqual(backend.generate(request()).text, "A")
        self.assertEqual(backend.generate(request()).usage.total_tokens, 3)
        with self.assertRaisesRegex(InjectedBackendError, "missing"):
            backend.generate(request("unknown"))

    def test_replay_injects_timeout_and_error(self) -> None:
        timeout = get_backend("replay", errors={"request-1": "timeout"})
        with self.assertRaises(BackendTimeoutError):
            timeout.generate(request())
        failure = get_backend("replay", errors={"request-1": {"kind": "rate_limit", "message": "fixture limit"}})
        with self.assertRaisesRegex(InjectedBackendError, "fixture limit"):
            failure.generate(request())

    def test_openai_compatible_payload_and_injected_transport(self) -> None:
        seen = {}

        def transport(payload, headers, timeout):
            seen.update({"payload": payload, "headers": headers, "timeout": timeout})
            return {
                "id": "response-1",
                "choices": [{"message": {"content": "B"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }

        backend = OpenAICompatibleBackend(api_key="fixture-key", transport=transport, timeout=9)
        response = backend.generate(request())
        self.assertEqual(response.text, "B")
        self.assertEqual(response.usage.total_tokens, 11)
        self.assertEqual(seen["payload"]["seed"], 17)
        self.assertEqual(seen["timeout"], 9)
        self.assertTrue(seen["headers"]["Authorization"].startswith("Bearer "))

    def test_vllm_does_not_require_api_key(self) -> None:
        backend = VLLMBackend(
            transport=lambda payload, headers, timeout: {
                "choices": [{"message": {"content": "local"}, "finish_reason": "stop"}]
            }
        )
        self.assertEqual(backend.generate(request()).text, "local")

    def test_huggingface_is_lazy_and_factory_is_injectable(self) -> None:
        calls = []

        def factory(task, **kwargs):
            calls.append((task, kwargs))
            return lambda prompt, **generation: [{"generated_text": prompt + "answer"}]

        backend = HuggingFaceBackend(pipeline_factory=factory)
        self.assertFalse(backend.loaded)
        self.assertEqual(calls, [])
        response = backend.generate(request())
        self.assertTrue(backend.loaded)
        self.assertEqual(response.text, "answer")
        self.assertEqual(calls[0][0], "text-generation")

    def test_builtin_registry_is_lazy_and_complete(self) -> None:
        names = set(registered_backends())
        self.assertTrue({"replay", "openai_compatible", "vllm", "huggingface"} <= names)
        self.assertTrue(get_backend("replay", default_response="ok"))


if __name__ == "__main__":
    unittest.main()

