import json
import unittest

from sim_eval.contracts import BenchmarkCase
from sim_eval.environments.dialogue import (
    DialogueAction,
    DialogueEnvironment,
    DialogueSpec,
    EVALUATED_USER_ROLE,
    FIXED_ASSISTANT_ROLE,
    parse_dialogue_action,
)
from sim_eval.errors import ParseError, ValidationError


def make_case():
    return BenchmarkCase(
        benchmark_id="userlm",
        case_id="case",
        group_id="group",
        split="fixture",
        source_revision="fixture-v1",
        input_data={},
    )


class DialogueEnvironmentTests(unittest.TestCase):
    def test_private_contexts_are_role_isolated(self) -> None:
        environment = DialogueEnvironment()
        state = environment.reset_with_spec(
            make_case(),
            DialogueSpec(
                user_context={"secret_intent": "find the copper key"},
                assistant_context={"secret_policy": "ask one question"},
                scheduled_actors=(EVALUATED_USER_ROLE, FIXED_ASSISTANT_ROLE),
            ),
            seed=7,
        )
        user_text = "\n".join(message.content for message in environment.observation(state, actor=EVALUATED_USER_ROLE))
        self.assertIn("find the copper key", user_text)
        self.assertNotIn("ask one question", user_text)
        environment.apply(state, actor=EVALUATED_USER_ROLE, action=DialogueAction("message", "Could you help?"))
        assistant_text = "\n".join(message.content for message in environment.observation(state, actor=FIXED_ASSISTANT_ROLE))
        self.assertIn("ask one question", assistant_text)
        self.assertNotIn("find the copper key", assistant_text)
        self.assertIn("Could you help?", assistant_text)

    def test_schedule_terminates_only_after_all_roles(self) -> None:
        environment = DialogueEnvironment()
        state = environment.reset_with_spec(
            make_case(),
            DialogueSpec(
                user_context={},
                assistant_context={},
                scheduled_actors=(EVALUATED_USER_ROLE, FIXED_ASSISTANT_ROLE, EVALUATED_USER_ROLE),
            ),
            seed=1,
        )
        environment.apply(state, actor=EVALUATED_USER_ROLE, action=DialogueAction("message", "one"))
        environment.apply(state, actor=FIXED_ASSISTANT_ROLE, action=DialogueAction("message", "two"))
        transition = environment.apply(state, actor=EVALUATED_USER_ROLE, action=DialogueAction("message", "three"))
        self.assertTrue(transition.terminal)
        self.assertTrue(state.protocol_complete)
        self.assertEqual(state.terminal_reason, "scheduled_complete")
        self.assertEqual([event.actor for event in state.public_transcript], [EVALUATED_USER_ROLE, FIXED_ASSISTANT_ROLE, EVALUATED_USER_ROLE])

    def test_native_end_token_strict_json_empty_and_refusal(self) -> None:
        self.assertEqual(parse_dialogue_action("<|endconversation|>").action, "end")
        self.assertEqual(parse_dialogue_action(json.dumps({"action": "message", "message": "hello"})).message, "hello")
        with self.assertRaises(ParseError):
            parse_dialogue_action(json.dumps({"action": "message", "message": ""}))
        with self.assertRaises(ParseError):
            parse_dialogue_action(json.dumps({"action": "message", "message": "x", "extra": 1}))

        environment = DialogueEnvironment()
        state = environment.reset_with_spec(
            make_case(),
            DialogueSpec(user_context={}, assistant_context={}),
            seed=1,
        )
        transition = environment.apply(
            state,
            actor=EVALUATED_USER_ROLE,
            action=DialogueAction("refuse", "I cannot continue."),
        )
        self.assertTrue(transition.terminal)
        self.assertFalse(state.protocol_complete)
        self.assertEqual(state.terminal_reason, "evaluated_user_refusal")

    def test_limits_and_out_of_turn_actions_fail_closed(self) -> None:
        environment = DialogueEnvironment()
        state = environment.reset_with_spec(
            make_case(),
            DialogueSpec(user_context={}, assistant_context={}, max_user_turns=1),
            seed=1,
        )
        with self.assertRaises(ValidationError):
            environment.apply(state, actor=FIXED_ASSISTANT_ROLE, action=DialogueAction("message", "wrong"))
        environment.apply(state, actor=EVALUATED_USER_ROLE, action=DialogueAction("message", "first"))
        environment.apply(state, actor=FIXED_ASSISTANT_ROLE, action=DialogueAction("message", "reply"))
        with self.assertRaises(ValidationError):
            environment.apply(state, actor=EVALUATED_USER_ROLE, action=DialogueAction("message", "second"))


if __name__ == "__main__":
    unittest.main()
