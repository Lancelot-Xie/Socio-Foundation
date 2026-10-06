import unittest
from dataclasses import replace
from io import BytesIO
from urllib.error import HTTPError

from sim_eval.backends.fallback import SafetyFallbackBackend
from sim_eval.backends.openai_compatible import OpenAICompatibleBackend, VLLMBackend, _http_error
from sim_eval.backends.openai_responses import OpenAIResponsesBackend
from sim_eval.contracts import ChatMessage, ModelRequest
from sim_eval.errors import BackendError, BackendHTTPError, BackendSafetyError


def request(**overrides):
    base = ModelRequest(
        messages=(ChatMessage("user", "Return JSON."),),
        model="primary-model",
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
        seed=17,
        response_format={"type": "json_object"},
    )
    return replace(base, **overrides)


def chat_result(text="{}", *, finish_reason="stop"):
    return {
        "id": "chat-1",
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


class APIBackendTests(unittest.TestCase):
    def test_http_400_requires_an_explicit_safety_marker(self) -> None:
        safety = HTTPError(
            "https://example.test/chat/completions",
            400,
            "Bad Request",
            {},
            BytesIO(
                b'{"error":{"code":"content_filter","type":"invalid_request_error",'
                b'"message":"blocked by the safety policy"}}'
            ),
        )
        self.assertIsInstance(_http_error(safety, provider="fixture"), BackendSafetyError)

        ordinary = HTTPError(
            "https://example.test/chat/completions",
            400,
            "Bad Request",
            {},
            BytesIO(
                b'{"error":{"code":"unknown_parameter","type":"invalid_request_error",'
                b'"message":"Unknown parameter: chat_template_kwargs"}}'
            ),
        )
        self.assertNotIsInstance(_http_error(ordinary, provider="fixture"), BackendSafetyError)

    def test_vllm_auto_uses_structured_outputs_first(self) -> None:
        seen = []
        backend = VLLMBackend(
            transport=lambda payload, headers, timeout: seen.append(dict(payload)) or chat_result()
        )
        response = backend.generate(request())
        self.assertIn("structured_outputs", seen[0])
        self.assertEqual(response.raw["_sim_eval"]["structured_output_mode"], "structured_outputs")

    def test_auto_degrades_only_when_structured_parameter_is_rejected(self) -> None:
        seen = []

        def transport(payload, headers, timeout):
            seen.append(dict(payload))
            if "structured_outputs" in payload:
                raise BackendHTTPError(
                    "HTTP 400: unknown parameter structured_outputs",
                    status_code=400,
                    error_code="unknown_parameter",
                )
            return chat_result()

        response = VLLMBackend(transport=transport).generate(request())
        self.assertEqual(len(seen), 2)
        self.assertEqual(response.raw["_sim_eval"]["structured_output_mode"], "json_schema")

    def test_ordinary_400_does_not_degrade_or_fallback(self) -> None:
        calls = []

        def transport(payload, headers, timeout):
            calls.append(payload)
            raise BackendHTTPError(
                "HTTP 400: malformed prompt",
                status_code=400,
                error_code="invalid_request",
            )

        with self.assertRaises(BackendHTTPError):
            VLLMBackend(transport=transport).generate(request())
        self.assertEqual(len(calls), 1)

    def test_relay_maps_token_limit_and_omits_sampling(self) -> None:
        backend = OpenAICompatibleBackend(
            api_key="fixture",
            profile="relay",
            structured_output={"mode": "none"},
        )
        payload = backend.build_payload(request(), structured_mode="none")
        self.assertEqual(payload["max_completion_tokens"], 64)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertNotIn("seed", payload)

    def test_token_limit_finish_preserves_partial_output_for_scoring(self) -> None:
        backend = OpenAICompatibleBackend(
            api_key="fixture",
            structured_output={"mode": "none"},
            transport=lambda payload, headers, timeout: chat_result(
                '{"answer":"partial', finish_reason="length"
            ),
        )
        response = backend.generate(request())
        self.assertEqual(response.text, '{"answer":"partial')
        self.assertEqual(response.finish_reason, "length")
        self.assertTrue(response.raw["_sim_eval"]["output_truncated"])
        self.assertEqual(response.raw["_sim_eval"]["truncation_reason"], "length")

    def test_empty_structured_response_reports_reasoning_diagnostics(self) -> None:
        result = chat_result("")
        result["choices"][0]["message"]["reasoning_content"] = "hidden reasoning"
        backend = OpenAICompatibleBackend(
            api_key="fixture",
            profile="deepseek",
            structured_output={"mode": "none"},
            transport=lambda payload, headers, timeout: result,
        )
        with self.assertRaisesRegex(BackendError, "reasoning_content_chars=16"):
            backend.generate(request())

    def test_tau_fixed_assistant_disables_parallel_tool_calls(self) -> None:
        backend = OpenAICompatibleBackend(
            api_key="fixture",
            profile="deepseek",
            structured_output={"mode": "none"},
        )
        tau_request = request(
            response_format=None,
            tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ),
            metadata={"benchmark_id": "tau_usi", "actor": "fixed_assistant"},
        )
        payload = backend.build_payload(tau_request, structured_mode="none")
        self.assertIs(payload["parallel_tool_calls"], False)

        responses_backend = OpenAIResponsesBackend(api_key="fixture")
        responses_payload = responses_backend.build_payload(tau_request, structured_mode="none")
        self.assertIs(responses_payload["parallel_tool_calls"], False)

    def test_safety_wrapper_switches_model_only_for_safety_error(self) -> None:
        primary = OpenAICompatibleBackend(
            api_key="fixture",
            transport=lambda payload, headers, timeout: chat_result(finish_reason="content_filter"),
        )
        fallback_payloads = []
        fallback = OpenAICompatibleBackend(
            api_key="fixture",
            profile="deepseek",
            structured_output={"mode": "none"},
            transport=lambda payload, headers, timeout: fallback_payloads.append(dict(payload)) or chat_result(),
        )
        backend = SafetyFallbackBackend(
            primary=primary,
            fallback=fallback,
            fallback_request_overrides={"model": "example-support-model"},
            primary_identity={"model": "primary-model"},
            fallback_identity={"model": "example-support-model"},
            fallback_reason="primary_judge_explicitly_blocked_by_safety_filter",
        )
        response = backend.generate(request())
        self.assertEqual(fallback_payloads[0]["model"], "example-support-model")
        self.assertTrue(response.raw["_sim_eval"]["safety_fallback"]["used"])

        ordinary_failure = SafetyFallbackBackend(
            primary=OpenAICompatibleBackend(
                api_key="fixture",
                transport=lambda payload, headers, timeout: (_ for _ in ()).throw(
                    BackendHTTPError("ordinary 400", status_code=400)
                ),
            ),
            fallback=fallback,
            fallback_request_overrides={"model": "example-support-model"},
            primary_identity={"model": "primary-model"},
            fallback_identity={"model": "example-support-model"},
            fallback_reason="primary_judge_explicitly_blocked_by_safety_filter",
        )
        before = len(fallback_payloads)
        with self.assertRaises(BackendHTTPError):
            ordinary_failure.generate(request())
        self.assertEqual(len(fallback_payloads), before)

    def test_explicit_safety_exception_is_the_only_exception_caught(self) -> None:
        fallback_calls = []
        fallback = OpenAICompatibleBackend(
            api_key="fixture",
            structured_output={"mode": "none"},
            transport=lambda payload, headers, timeout: fallback_calls.append(payload) or chat_result(),
        )
        primary = OpenAICompatibleBackend(
            api_key="fixture",
            transport=lambda payload, headers, timeout: (_ for _ in ()).throw(
                BackendSafetyError(
                    "blocked by safety",
                    status_code=400,
                    error_code="content_filter",
                )
            ),
        )
        wrapped = SafetyFallbackBackend(
            primary=primary,
            fallback=fallback,
            fallback_request_overrides={"model": "fallback-model"},
            primary_identity={"model": "primary-model"},
            fallback_identity={"model": "fallback-model"},
            fallback_reason="safety",
        )
        wrapped.generate(request())
        self.assertEqual(len(fallback_calls), 1)

    def test_native_responses_payload_and_response_mapping(self) -> None:
        seen = []

        def transport(payload, headers, timeout):
            seen.append(dict(payload))
            return {
                "id": "resp-1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"ok":true}'}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            }

        backend = OpenAIResponsesBackend(api_key="fixture", transport=transport)
        response = backend.generate(request())
        self.assertEqual(seen[0]["max_output_tokens"], 64)
        self.assertEqual(seen[0]["text"]["format"]["type"], "json_object")
        self.assertFalse(seen[0]["store"])
        self.assertNotIn("temperature", seen[0])
        self.assertEqual(response.text, '{"ok":true}')
        self.assertEqual(response.raw["_sim_eval"]["protocol"], "openai_responses")

    def test_responses_token_limit_preserves_partial_output_for_scoring(self) -> None:
        result = {
            "id": "resp-partial-1",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"ok":'}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 64, "total_tokens": 67},
        }
        backend = OpenAIResponsesBackend(
            api_key="fixture", transport=lambda payload, headers, timeout: result
        )
        response = backend.generate(request())
        self.assertEqual(response.text, '{"ok":')
        self.assertEqual(response.finish_reason, "max_output_tokens")
        self.assertTrue(response.raw["_sim_eval"]["output_truncated"])
        self.assertEqual(
            response.raw["_sim_eval"]["truncation_reason"], "max_output_tokens"
        )

    def test_responses_native_tool_call_stays_out_of_message_text(self) -> None:
        result = {
            "id": "resp-tool-1",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "name": "lookup_order",
                    "arguments": '{"order_id":"ORD-1"}',
                }
            ],
        }
        backend = OpenAIResponsesBackend(
            api_key="fixture", transport=lambda payload, headers, timeout: result
        )
        tau_request = request(
            response_format=None,
            tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_order",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ),
            metadata={"benchmark_id": "tau_usi", "actor": "fixed_assistant"},
        )
        response = backend.generate(tau_request)
        self.assertEqual(response.text, "")
        self.assertEqual(
            response.raw["_sim_eval"]["native_tool_call"],
            {"name": "lookup_order", "arguments": {"order_id": "ORD-1"}},
        )


if __name__ == "__main__":
    unittest.main()
