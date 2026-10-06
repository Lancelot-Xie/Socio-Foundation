import unittest
from pathlib import Path

from sim_eval.benchmarks.tau_usi import TauUSIAdapter
from sim_eval.contracts import ModelResponse
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.environments.tau_usi import (
    ASSISTANT_ROLE,
    INITIAL_ASSISTANT_MESSAGE,
    TAU_USER_STOP_TOKEN,
    USER_ROLE,
    TauAssistantAction,
    TauToolCall,
    parse_assistant_response,
    parse_user_action,
)
from sim_eval.errors import ParseError, ValidationError


ROOT = Path(__file__).resolve().parents[1]


def assistant_message(text):
    return parse_assistant_response(ModelResponse(text=text))


def assistant_tool(name, arguments, *, text=""):
    return parse_assistant_response(
        ModelResponse(
            text=text,
            raw={"_sim_eval": {"native_tool_call": {"name": name, "arguments": arguments}}},
        )
    )


class TauUSIEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = load_fixture_suite(ROOT / "tests" / "fixtures")["tau_usi"][0][0]
        cls.adapter = TauUSIAdapter()

    def test_user_assistant_and_environment_views_are_isolated(self) -> None:
        state = self.adapter.environment.reset(self.case, seed=7)
        user_text = "\n".join(message.content for message in self.adapter.environment.observation(state, actor=USER_ROLE))
        self.assertIn("# Your Goal", user_text)
        self.assertIn(INITIAL_ASSISTANT_MESSAGE, user_text)
        self.assertNotIn("your_known_state", user_text)
        self.assertNotIn("replacement_status", user_text)
        self.assertNotIn("fixture_h1", user_text)
        self.assertNotIn("EX-FIXTURE-91", user_text)
        state.initial_user_state = {**state.initial_user_state, "internal_user_id": "hidden-db-key"}
        user_text = "\n".join(
            message.content for message in self.adapter.environment.observation(state, actor=USER_ROLE)
        )
        self.assertNotIn("hidden-db-key", user_text)
        self.assertEqual(state.public_transcript[0].actor, ASSISTANT_ROLE)
        self.assertEqual(state.public_transcript[0].content, INITIAL_ASSISTANT_MESSAGE)

        first = self.case.metadata["replay"]["steps"][0]["output"]
        self.adapter.environment.apply(
            state, actor=USER_ROLE, action=parse_user_action(first["message"])
        )
        assistant_text = "\n".join(
            message.content for message in self.adapter.environment.observation(state, actor=ASSISTANT_ROLE)
        )
        self.assertIn(self.case.input_data["assistant_policy"], assistant_text)
        self.assertIn("ORD-RIVER-7319", assistant_text)
        self.assertNotIn("replacement_status", assistant_text)
        self.assertNotIn("fixture_h1", assistant_text)
        self.assertNotIn('"eligible":true', assistant_text)
        self.assertNotIn('"tools":', assistant_text)

        tool_output = self.case.metadata["replay"]["steps"][1]["output"]["tool_call"]
        self.adapter.environment.apply(
            state,
            actor=ASSISTANT_ROLE,
            action=assistant_tool(tool_output["name"], tool_output["arguments"]),
        )
        assistant_after_tool = "\n".join(
            message.content for message in self.adapter.environment.observation(state, actor=ASSISTANT_ROLE)
        )
        self.assertIn("'eligible': True", assistant_after_tool)
        self.assertEqual(state.environment_state["replacement_status"], "not_started")

        message_output = self.case.metadata["replay"]["steps"][2]["output"]["message"]
        self.adapter.environment.apply(
            state, actor=ASSISTANT_ROLE, action=assistant_message(message_output)
        )
        user_after_tool = "\n".join(
            message.content for message in self.adapter.environment.observation(state, actor=USER_ROLE)
        )
        self.assertNotIn("'eligible': True", user_after_tool)
        self.assertIn("Should I replace it", user_after_tool)

    def test_tool_calls_and_observations_are_typed_and_assistant_only(self) -> None:
        state = self.adapter.environment.reset(self.case, seed=11)
        self.adapter.environment.apply(
            state,
            actor=USER_ROLE,
            action=parse_user_action(self.case.metadata["replay"]["steps"][0]["output"]["message"]),
        )
        raw_tool = self.case.metadata["replay"]["steps"][1]["output"]["tool_call"]
        action = assistant_tool(raw_tool["name"], raw_tool["arguments"])
        transition = self.adapter.environment.apply(state, actor=ASSISTANT_ROLE, action=action)
        self.assertEqual([event.kind for event in transition.events], ["assistant_text", "tool_call", "tool_observation", "tool_feedback"])
        self.assertTrue(all(tuple(event.visible_to) == (ASSISTANT_ROLE,) for event in transition.events))
        self.assertEqual(state.next_actor, ASSISTANT_ROLE)

    def test_user_parser_is_strict(self) -> None:
        self.assertEqual(parse_user_action("Please help with my order.").action, "send")
        self.assertEqual(parse_user_action(TAU_USER_STOP_TOKEN).action, "stop")
        self.assertEqual(parse_user_action(f"Thanks. {TAU_USER_STOP_TOKEN}").action, "stop")
        self.assertEqual(
            parse_user_action("<think>private reasoning</think> Please help.").message,
            "Please help.",
        )
        for invalid in (
            "",
            "   ",
            "<think>private reasoning only</think>",
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(parse_user_action(invalid).action, "empty")

    def test_legacy_native_tool_metadata_remains_supported(self) -> None:
        valid = assistant_tool("lookup_order", {"order_id": "x"}, text="this content is ignored")
        self.assertEqual(valid.tool_call.name, "lookup_order")
        self.assertEqual(valid.message, "")
        authored_json = assistant_message(
            '{"message":"looking it up","tool_call":{"name":"lookup_order","arguments":{"order_id":"x"}},"done":false}'
        )
        self.assertIsNone(authored_json.tool_call)
        self.assertIn('"done":false', authored_json.message)
        for invalid in (
            ModelResponse(text="", raw={"_sim_eval": {"native_tool_call": []}}),
            ModelResponse(
                text="",
                raw={"_sim_eval": {"native_tool_call": {"name": "x", "arguments": []}}},
            ),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ParseError):
                parse_assistant_response(invalid)

    def test_unknown_and_missing_argument_tools_become_assistant_observations(self) -> None:
        state = self.adapter.environment.reset(self.case, seed=13)
        self.adapter.environment.apply(
            state, actor=USER_ROLE, action=parse_user_action("help")
        )
        unknown = self.adapter.environment.apply(
            state,
            actor=ASSISTANT_ROLE,
            action=TauAssistantAction("", TauToolCall("delete_everything", {})),
        )
        self.assertFalse(unknown.terminal)
        self.assertEqual(state.next_actor, ASSISTANT_ROLE)
        self.assertEqual(unknown.events[-1].content, "Unknown action delete_everything")
        self.assertEqual(unknown.events[-2].metadata["tool_error_kind"], "unknown_action")

        missing = self.adapter.environment.apply(
            state,
            actor=ASSISTANT_ROLE,
            action=TauAssistantAction("", TauToolCall("lookup_order", {})),
        )
        self.assertFalse(missing.terminal)
        self.assertEqual(state.next_actor, ASSISTANT_ROLE)
        self.assertIn("missing required arguments", missing.events[-1].content)
        self.assertEqual(
            missing.events[-2].metadata["tool_error_kind"],
            "missing_required_arguments",
        )

        with self.assertRaisesRegex(ValidationError, "replay_configuration_error"):
            self.adapter.environment.apply(
                state,
                actor=ASSISTANT_ROLE,
                action=TauAssistantAction(
                    "", TauToolCall("lookup_order", {"order_id": "ORD-NOT-AUTHORIZED"})
                ),
            )


if __name__ == "__main__":
    unittest.main()
