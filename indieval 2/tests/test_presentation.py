import json
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.backends.openai_compatible import OpenAICompatibleBackend
from sim_eval.backends.openai_responses import OpenAIResponsesBackend
from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.mirrorbench import MirrorBenchAdapter
from sim_eval.benchmarks.tau_usi import TauUSIAdapter
from sim_eval.benchmarks.userlm import UserLMAdapter
from sim_eval.contracts import BenchmarkCase, ChatMessage, ModelRequest
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.environments.dialogue import (
    DialogueEnvironment, DialogueSpec, DialogueAction, EVALUATED_USER_ROLE as U,
    FIXED_ASSISTANT_ROLE as A,
)
from sim_eval.errors import ValidationError
from sim_eval.integrations.token_counting import _message_payload
from sim_eval.presentation import material_text

ROOT = Path(__file__).resolve().parents[1]


class CapturingReplay(ReplayBackend):
    def __init__(self, responses):
        super().__init__(responses)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


class PresentationTests(unittest.TestCase):
    def test_native_tool_history_rejects_broken_pairing(self):
        call = ChatMessage("assistant", "", tool_calls=({
            "id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        },))
        result = ChatMessage("tool", '{"ok":true}', tool_call_id="call_1")
        ModelRequest(messages=(call, result), model="fixture")
        invalid_histories = (
            (call,), (result,),
            (call, ChatMessage("user", "next turn"), result),
            (call, ChatMessage("tool", "wrong result", tool_call_id="call_2")),
            (call, result, result),
            (call, result, call, result),
        )
        for messages in invalid_histories:
            with self.subTest(messages=messages), self.assertRaises(ValidationError):
                ModelRequest(messages=messages, model="fixture")
        with self.assertRaises(ValidationError):
            replace(call, role="user")
        with self.assertRaises(ValidationError):
            replace(result, role="assistant")

    def test_dialogue_empty_opening_consecutive_actor_and_refusal(self):
        env = DialogueEnvironment()
        case = BenchmarkCase(benchmark_id="mirrorbench", case_id="synthetic", group_id="g",
                             split="test", source_revision="fixture", input_data={})
        state = env.reset_with_spec(case, DialogueSpec(
            user_context={"goal": "user goal"}, assistant_context={"policy": "assistant policy"},
            scheduled_actors=(U, U, A),
        ), seed=17)
        self.assertEqual([m.role for m in env.observation(state, actor=U)], ["system"])
        with self.assertRaises(ValidationError):
            env.observation(state, actor=A)
        env.apply(state, actor=U, action=DialogueAction("message", "first part"))
        self.assertEqual(state.next_actor, U)
        env.apply(state, actor=U, action=DialogueAction("message", "second part"))
        messages = env.observation(state, actor=A)
        self.assertEqual([m.role for m in messages], ["system", "user", "user"])
        self.assertEqual([m.content for m in messages[1:]], ["first part", "second part"])
        env.apply(state, actor=A, action=DialogueAction("refuse", "Cannot help"))
        self.assertTrue(state.terminal)
        self.assertFalse(state.protocol_complete)
        with self.assertRaises(ValidationError):
            env.observation(state, actor=U)

    def test_tau_every_history_matches_visible_trace_and_stable_tool_ids(self):
        case = load_fixture_suite(ROOT / "tests/fixtures")["tau_usi"][0][0]
        adapter = TauUSIAdapter()
        backend = CapturingReplay(adapter.replay_responses(case, seed=17))
        result = adapter.execute_case(case, backend=backend, run_id="presentation-only", seed=17, model="candidate")
        events = {event.turn: event for event in result.trace}
        saw_two_calls = False
        for request in backend.requests:
            actor = request.metadata.get("actor")
            if request.metadata.get("stage") == "post_interaction_survey" or actor not in {U, A}:
                continue
            calls = 0
            previous_turn = -1
            for message in request.messages[1:]:
                event = events[message.metadata["turn"]]
                self.assertGreater(event.turn, previous_turn)
                previous_turn = event.turn
                self.assertIn(actor, event.visible_to)
                self.assertFalse(message.tool_calls)
                if event.kind == "tool_feedback":
                    self.assertEqual(actor, A)
                    self.assertEqual(message.role, "system")
                    calls += 1
                else:
                    self.assertEqual(message.role, "assistant" if event.actor == actor else "user")
                self.assertEqual(message.content, event.content)
            saw_two_calls |= calls == 2
        self.assertTrue(saw_two_calls)

    def test_nested_material_keeps_values_order_and_empty_fields(self):
        self.assertEqual(material_text({"memory": [" first\nsecond ", {"empty": [], "false": False}],
                                        "unknown": None}),
            "memory:\nItem 1:\n first\nsecond \n\nItem 2:\nempty:\n[]\nfalse:\nfalse\nunknown:\nnull")

    def test_dialogue_perspective_private_context_and_stop(self):
        env = DialogueEnvironment()
        case = BenchmarkCase(benchmark_id="userlm", case_id="synthetic", group_id="g", split="test", source_revision="fixture", input_data={})
        state = env.reset_with_spec(case, DialogueSpec(
            user_context={"goal": "PRIVATE USER"}, assistant_context={"policy": "PRIVATE ASSISTANT"},
            initial_messages=({"actor": A, "message": "Greeting"},),
        ), seed=17)
        self.assertEqual(state.next_actor, U)
        first = env.observation(state, actor=U)
        self.assertEqual([m.role for m in first], ["system", "user"])
        self.assertEqual(first[-1].content, "Greeting")
        env.apply(state, actor=U, action=DialogueAction("message", "My question"))
        peer = env.observation(state, actor=A)
        self.assertEqual([m.role for m in peer], ["system", "assistant", "user"])
        self.assertEqual([m.content for m in peer[1:]], ["Greeting", "My question"])
        self.assertNotIn("PRIVATE USER", "\n".join(m.content for m in peer))
        env.apply(state, actor=A, action=DialogueAction("message", "My answer"))
        own = env.observation(state, actor=U, self_action_envelope=True)
        self.assertEqual([m.role for m in own], ["system", "user", "assistant", "user"])
        self.assertEqual(json.loads(own[2].content), {"action": "message", "message": "My question"})
        self.assertEqual(own[-1].content, "My answer")
        self.assertNotIn("PRIVATE ASSISTANT", "\n".join(m.content for m in own))
        self.assertEqual(len([m for m in own if m.role == "system"]), 1)
        env.apply(state, actor=U, action=DialogueAction("end", ""))
        self.assertIsNone(state.next_actor)
        with self.assertRaises(ValidationError):
            env.observation(state, actor=U)
        with self.assertRaises(ValidationError):
            env.apply(state, actor=A, action=DialogueAction("message", "extra"))

    def test_mirrorbench_and_lic_replay_preserve_private_context_after_system_override(self):
        fixtures = load_fixture_suite(ROOT / "tests/fixtures")
        for benchmark, adapter in (("mirrorbench", MirrorBenchAdapter()), ("userlm", UserLMAdapter())):
            for case in fixtures[benchmark][0]:
                if benchmark == "userlm" and adapter.execution_mode_for_case(case) == "section3_single_user_turn":
                    continue
                backend = CapturingReplay(adapter.replay_responses(case, seed=17))
                result = adapter.execute_case(case, backend=backend, run_id="presentation-only", seed=17, model="candidate")
                self.assertEqual(result.status.value, "completed")
                actors = [r for r in backend.requests if r.metadata.get("actor") in {U, A}]
                self.assertTrue(any(len(r.messages) > 2 for r in actors))
                for request in actors:
                    self.assertEqual(request.messages[0].role, "system")
                    self.assertEqual(sum(m.role == "system" for m in request.messages), 1)
                    if benchmark == "userlm" and request.metadata.get("actor") == A:
                        self.assertNotIn("# Your Private Context", request.messages[0].content)
                        self.assertEqual(request.messages[0].content, adapter._assistant_system_prompt(case))
                    else:
                        self.assertIn("# Your Private Context", request.messages[0].content)
                    self.assertNotIn('"public_transcript"', request.messages[0].content)

    def test_tau_text_tool_history_on_wire_and_token_counter(self):
        case = load_fixture_suite(ROOT / "tests/fixtures")["tau_usi"][0][0]
        adapter = TauUSIAdapter()
        backend = CapturingReplay(adapter.replay_responses(case, seed=17))
        result = adapter.execute_case(case, backend=backend, run_id="presentation-only", seed=17, model="candidate")
        self.assertEqual(result.status.value, "completed")
        assistant = [r for r in backend.requests if r.metadata.get("actor") == A]
        request = next(r for r in assistant if any(m.metadata.get("source") == "tau_tool_feedback" for m in r.messages))
        payload = OpenAICompatibleBackend(base_url="http://fixture/v1").build_payload(request)
        self.assertEqual(payload["messages"], _message_payload(request))
        self.assertNotIn("tools", payload)
        self.assertNotIn("parallel_tool_calls", payload)
        self.assertEqual(payload["messages"][-1]["role"], "system")
        self.assertIn("<function=lookup_order>", payload["messages"][-2]["content"])
        self.assertTrue(all("tool_calls" not in message for message in payload["messages"]))
        for user in (r for r in backend.requests if r.metadata.get("actor") == U):
            self.assertFalse(any(m.tool_calls or m.role == "tool" for m in user.messages))
            self.assertNotIn("'eligible': True", "\n".join(m.content for m in user.messages))
        self.assertEqual(result.metadata["terminal_reason"], "user_stop")


if __name__ == "__main__":
    unittest.main()
