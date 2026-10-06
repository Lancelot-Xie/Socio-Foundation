import unittest
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sim_eval.backends.openai_compatible import OpenAICompatibleBackend
from sim_eval.backends.replay import ReplayBackend
from sim_eval.backends.routed import AttemptScopedBackend, RoleRoutedBackend
from sim_eval.contracts import ChatMessage, ModelRequest, ModelResponse
from sim_eval.interfaces import ModelBackend
from sim_eval.integrations.tau_bench_local import (
    TauBenchLocalSession,
    TauBenchRepository,
    TauUSIOfficialReferenceStore,
)


ROOT = Path(__file__).resolve().parents[2]


class CaptureBackend(ModelBackend):
    name = "capture"

    def __init__(self, barrier=None):
        self.requests = []
        self.barrier = barrier

    def generate(self, request):
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        self.requests.append(request)
        return ModelResponse(
            text="ok",
            raw={"_sim_eval": {"protocol": "test"}},
        )


class TauUSILocalRuntimeTests(unittest.TestCase):
    def test_attempt_scoped_provenance_collects_cross_thread_role_calls(self):
        barrier = threading.Barrier(2)
        routed = RoleRoutedBackend(
            evaluated_backend=CaptureBackend(barrier),
            fixed_assistant_backend=CaptureBackend(barrier),
        )
        scoped = AttemptScopedBackend(routed)
        requests = (
            ModelRequest(
                messages=(ChatMessage("user", "user"),),
                model="candidate",
                request_id="candidate-call",
                metadata={"actor": "evaluated_user"},
            ),
            ModelRequest(
                messages=(ChatMessage("user", "assistant"),),
                model="assistant",
                request_id="assistant-call",
                metadata={"actor": "fixed_assistant"},
            ),
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            tuple(pool.map(scoped.generate, requests))
        events = {event["request_id"]: event for event in scoped.events()}
        self.assertEqual(events["candidate-call"]["route_role"], "evaluated_user")
        self.assertEqual(events["assistant-call"]["route_role"], "fixed_assistant")
        self.assertTrue(all(event["protocol"] == "test" for event in events.values()))

    def test_role_routed_backend_applies_role_generation_overrides(self):
        candidate = CaptureBackend()
        assistant = CaptureBackend()
        routed = RoleRoutedBackend(
            evaluated_backend=candidate,
            fixed_assistant_backend=assistant,
            evaluated_request_overrides={"model": "served-user", "max_tokens": 4096},
            fixed_assistant_request_overrides={"model": "served-assistant", "max_tokens": 4096},
        )
        base = dict(messages=(ChatMessage("user", "test"),), model="adapter-default")
        routed.generate(
            ModelRequest(
                **base,
                temperature=0.7,
                max_tokens=8,
                metadata={"actor": "evaluated_user"},
            )
        )
        routed.generate(
            ModelRequest(
                **base,
                temperature=0.0,
                max_tokens=16,
                metadata={"actor": "fixed_assistant"},
            )
        )
        self.assertEqual(candidate.requests[0].model, "served-user")
        self.assertEqual(candidate.requests[0].temperature, 0.7)
        self.assertEqual(candidate.requests[0].max_tokens, 4096)
        self.assertEqual(assistant.requests[0].model, "served-assistant")
        self.assertEqual(assistant.requests[0].temperature, 0.0)
        self.assertEqual(assistant.requests[0].max_tokens, 4096)

    def test_role_routed_backend_keeps_candidate_and_assistant_endpoints_separate(self):
        candidate = ReplayBackend(default_response="###STOP###")
        assistant = ReplayBackend(default_response="hello")
        routed = RoleRoutedBackend(
            evaluated_backend=candidate,
            fixed_assistant_backend=assistant,
        )
        base = dict(messages=(ChatMessage("user", "test"),), model="model")
        self.assertEqual(
            routed.generate(ModelRequest(**base, metadata={"actor": "evaluated_user"})).text,
            "###STOP###",
        )
        self.assertEqual(
            routed.generate(ModelRequest(**base, metadata={"actor": "fixed_assistant"})).text,
            "hello",
        )


    def test_openai_native_tool_call_is_normalized_for_fixed_assistant(self):
        def transport(payload, headers, timeout):
            return {
                "id": "native-tool-smoke",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "get_user_details",
                                        "arguments": '{"user_id":"mia_li_3668"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            }

        backend = OpenAICompatibleBackend(
            base_url="http://127.0.0.1:8000/v1",
            require_api_key=False,
            transport=transport,
        )
        response = backend.generate(
            ModelRequest(
                messages=(ChatMessage("user", "test"),),
                model="fixed-assistant",
                metadata={"benchmark_id": "tau_usi", "actor": "fixed_assistant"},
            )
        )
        self.assertEqual(response.text, "")
        self.assertEqual(
            response.raw["_sim_eval"]["native_tool_call"],
            {"name": "get_user_details", "arguments": {"user_id": "mia_li_3668"}},
        )

    def test_official_survey_uses_semantic_question_amount_mapping(self):
        path = ROOT / "temp_data_qa" / "datasets" / "official" / "tau_usi" / "tau_bench_tasks_unified.json"
        if not path.is_file():
            self.skipTest("authorized local tau-USI annotation is not installed")
        store = TauUSIOfficialReferenceStore(path)
        reference = store.references_for_ids(("airline_0",))[0]
        self.assertEqual(reference["survey"]["question_amount"], 1)
        self.assertEqual(set(reference["survey"]), {
            "task_success",
            "efficiency",
            "question_amount",
            "answer_effort",
            "human_likeness",
            "interaction_flow",
            "overall",
            "reuse_intent",
        })


if __name__ == "__main__":
    unittest.main()
