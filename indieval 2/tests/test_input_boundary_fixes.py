import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.agentsense import AgentSenseAdapter
from sim_eval.benchmarks.alignx import AlignXAdapter
from sim_eval.benchmarks.coser import CoserAdapter
from sim_eval.benchmarks.humanual import HumanualAdapter
from sim_eval.benchmarks.lic_prompts import CODE_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT
from sim_eval.benchmarks.mirrorbench import MirrorBenchAdapter
from sim_eval.benchmarks.userlm import UserLMAdapter
from sim_eval.contracts import TraceEvent
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.environments.coser import CoserRoleOutput, deterministic_token_count
from sim_eval.environments.dialogue import DialogueAction, EVALUATED_USER_ROLE as U, FIXED_ASSISTANT_ROLE as A
from sim_eval.errors import EpisodeTokenBudgetExceeded, ValidationError


ROOT = Path(__file__).resolve().parents[1]


class InputBoundaryFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_fixture_suite(ROOT / "tests/fixtures")

    def case(self, benchmark, index=0):
        return self.fixtures[benchmark][0][index]

    def test_alignx_removes_only_nested_vectors_without_mutating_source(self):
        case = self.case("alignx", 1)
        history = [{"prompt": "POST", "chosen": "LIKE", "rejected": "DISLIKE",
                    "Preference Direction": [934857.25] * 90,
                    "nested": {"preference_direction": [934857.25], "comment": "KEEP"}}]
        case = replace(case, input_data={**case.input_data, "pairwise_feedback": history})
        original = copy.deepcopy(case.input_data)
        request = AlignXAdapter().build_request(case, model="audit", seed=17)
        text = "\n".join(m.content for m in request.messages)
        context = request.metadata["candidate_scoring"]["context"]
        self.assertNotIn("934857.25", text + json.dumps(context))
        self.assertNotIn("Preference Direction", text)
        cleaned = context["user_evidence"]["content"]
        self.assertEqual(cleaned, [{"prompt": "POST", "chosen": "LIKE", "rejected": "DISLIKE",
                                    "nested": {"comment": "KEEP"}}])
        self.assertEqual(case.input_data, original)

    def test_alignx_history16_keeps_all_entries(self):
        case = self.case("alignx", 1)
        rows = [{"pair": {"prompt": f"POST_{i}", "Preference Direction": [i] * 90},
                 "ugc": {"comment": f"COMMENT_{i}", "Preference Direction": [i] * 90}}
                for i in range(16)]
        case = replace(case, input_data={**case.input_data, "variant": "Reddit_history16", "history16": rows})
        evidence = AlignXAdapter()._conditioning(case)["content"]
        self.assertEqual(len(evidence), 16)
        for i, row in enumerate(evidence):
            self.assertEqual(row, {"pair": {"prompt": f"POST_{i}"}, "ugc": {"comment": f"COMMENT_{i}"}})

    def test_static_budget_failure_is_a_record_and_does_not_abort_next_case(self):
        class NoRoom:
            def generate(self, request):
                raise EpisodeTokenBudgetExceeded("no room", scope="model_context",
                                                episode_remaining_tokens=4096,
                                                context_remaining_tokens=0, prompt_tokens=40000)
        adapter = AlignXAdapter()
        case = self.case("alignx")
        failed = adapter.execute_case(case, backend=NoRoom(), run_id="audit", model="audit", seed=17)
        self.assertEqual(failed.status.value, "failed")
        self.assertEqual(failed.error.kind, "token_budget_exhausted")
        self.assertFalse(failed.error.retryable)
        self.assertEqual(failed.metadata["token_budget"]["prompt_tokens"], 40000)
        self.assertFalse(failed.metrics)
        completed = adapter.execute_case(case, backend=ReplayBackend(adapter.replay_responses(case, seed=17)),
                                         run_id="audit", model="audit", seed=17)
        self.assertEqual(completed.status.value, "completed")

    def test_mirror_reference_occurs_once_and_stays_hidden_from_user(self):
        adapter = MirrorBenchAdapter()
        case = self.case("mirrorbench")
        case = replace(case, input_data={**case.input_data,
            "reference_conversation": [{"role": "user", "content": "REFERENCE_USER_SENTINEL"},
                                       {"role": "assistant", "content": "REFERENCE_ASSISTANT_SENTINEL"}],
            "assistant_config": {"style": "ASSISTANT_STYLE_SENTINEL"}})
        state = adapter.environment.reset_with_spec(case, adapter.dialogue_spec_for_case(case), seed=17)
        user = adapter.build_user_request(case, state, model="audit", seed=17)
        self.assertNotIn("REFERENCE_", "\n".join(m.content for m in user.messages))
        adapter.environment.apply(state, actor=U, action=DialogueAction("message", "REAL_USER_TURN"))
        assistant = adapter.build_assistant_request(case, state, seed=17)
        text = "\n".join(m.content for m in assistant.messages)
        self.assertEqual(text.count("REFERENCE_USER_SENTINEL"), 1)
        self.assertEqual(text.count("REFERENCE_ASSISTANT_SENTINEL"), 1)
        self.assertIn("ASSISTANT_STYLE_SENTINEL", text)
        self.assertEqual(assistant.messages[-1].content, "REAL_USER_TURN")

    def test_lic_assistant_sees_only_official_policy_and_actual_history(self):
        adapter = UserLMAdapter()
        for index, policy in [(0, MATH_SYSTEM_PROMPT), (1, CODE_SYSTEM_PROMPT)]:
            with self.subTest(index=index):
                case = self.case("userlm", index)
                task = {**case.input_data["assistant_task"], "payload": {
                    "question": "HIDDEN_FULL_QUESTION", "answer": "HIDDEN_GOLD",
                    "test": "HIDDEN_TESTS", "prompt": "HIDDEN_CODE_PROMPT"}}
                case = replace(case, input_data={**case.input_data, "assistant_task": task})
                snapshot = copy.deepcopy(case.input_data)
                spec = adapter.dialogue_spec_for_case(case)
                self.assertNotIn("HIDDEN_", json.dumps(spec.assistant_context))
                state = adapter.environment.reset_with_spec(case, spec, seed=17)
                adapter.environment.apply(state, actor=U, action=DialogueAction("message", "USER_DISCLOSURE"))
                request = adapter.build_assistant_request(case, state, seed=17)
                self.assertEqual([(m.role, m.content) for m in request.messages],
                                 [("system", policy), ("user", "USER_DISCLOSURE")])
                adapter.environment.apply(state, actor=A, action=DialogueAction("message", "ACTUAL_ASSISTANT_REPLY"))
                user = adapter.build_user_request(case, state, model="audit", seed=17)
                self.assertEqual(user.messages[-1].role, "user")
                self.assertEqual(user.messages[-1].content, "ACTUAL_ASSISTANT_REPLY")
                self.assertNotIn("HIDDEN_", "\n".join(m.content for m in user.messages))
                self.assertEqual(case.input_data, snapshot)  # verifier source retained

    def test_lic_user_and_judge_have_exact_shard_id_text_mapping(self):
        adapter = UserLMAdapter()
        case = self.case("userlm")
        shards = [{"id": "shard_1", "text": "FIRST_FACT", "required": True},
                  {"id": "shard_2", "text": "OPTIONAL_FACT", "required": False}]
        case = replace(case, input_data={**case.input_data, "information_shards": shards})
        spec = adapter.dialogue_spec_for_case(case)
        self.assertEqual(spec.user_context["information_shards"], shards)
        state = adapter.environment.reset_with_spec(case, spec, seed=17)
        adapter.environment.apply(state, actor=U, action=DialogueAction("message", "FIRST_FACT"))
        judge = adapter.build_shard_judge_request(case, state, adapter.provenance_for_case(case), seed=17)
        payload = json.loads(judge.messages[-1].content)
        self.assertEqual(payload["information_shards"], shards)
        self.assertEqual(payload["required_shards"], ["shard_1"])
        self.assertEqual(payload["nonrequired_shards"], ["shard_2"])

    def test_lic_missing_opaque_shard_text_fails_closed(self):
        case = self.case("userlm")
        case = replace(case, input_data={**case.input_data,
                                       "information_shards": [{"id": "shard_1", "required": True}]})
        with self.assertRaisesRegex(ValidationError, "requires its text"):
            UserLMAdapter().dialogue_spec_for_case(case)

    def test_agentsense_speakers_are_visible_without_api_names(self):
        adapter = AgentSenseAdapter()
        state = adapter.reset_official_state(self.case("agentsense"), seed=17)
        actor = state.next_actor
        others = ["Alice", "Bob"]
        state.transcript = [TraceEvent(turn=i, actor=name, kind="speak", content=f"LINE_{i}",
                                      visible_to=(actor,), metadata={"visibility": "public_transcript"})
                            for i, name in enumerate(others)]
        messages = adapter._actor_chat_messages(state, actor)
        self.assertEqual([m.content for m in messages[1:]], ["Alice: LINE_0", "Bob: LINE_1"])
        self.assertTrue(all("name" not in m.to_chat_dict() for m in messages))

    def test_humanual_names_are_lossless_in_content_without_changing_judge_context(self):
        adapter = HumanualAdapter()
        case = next(c for c in self.fixtures["humanual"][0] if c.input_data["domain"] == "book")
        case = replace(case, input_data={**case.input_data, "target_user_id": "target",
            "prompt": [{"role": "张 三 / reader", "content": "OTHER_TURN"},
                       {"role": "target", "content": "OWN_TURN"}]})
        request = adapter.build_request(case, model="audit", seed=17)
        self.assertEqual([m.content for m in request.messages[1:]], ["张 三 / reader: OTHER_TURN", "HUMAN: OWN_TURN"])
        self.assertTrue(all("name" not in m.to_chat_dict() for m in request.messages))
        self.assertEqual(adapter._context(case), "**User 张 三 / reader**: OTHER_TURN\n**User HUMAN**: OWN_TURN")

    def test_coser_private_thought_returns_only_to_its_author_and_counts_toward_budget(self):
        env = CoserAdapter().environment
        state = env.reset(self.case("coser"), seed=17)
        actor = state.next_actor
        env.apply(state, actor=actor, action=CoserRoleOutput(speech="PUBLIC_SENTINEL", inner_thought="PRIVATE_SENTINEL"))
        for observer in state.speaking_roles:
            state.current_speaker = observer
            selection = env.assemble_context(state, actor=observer)
            text = "\n".join(m.content for m in selection.messages)
            self.assertIn("PUBLIC_SENTINEL", text)
            self.assertEqual("PRIVATE_SENTINEL" in text, observer == actor)
            self.assertEqual(selection.provenance["included_own_private_thought_turns"], [0] if observer == actor else [])
            self.assertEqual(selection.provenance["used_tokens"],
                             sum(deterministic_token_count(m.content) for m in selection.messages))
        self.assertNotIn("PRIVATE_SENTINEL", str(state.public_transcript))
        state.current_speaker = actor
        state.transcript[-1] = replace(state.transcript[-1], content="large_private " * (state.max_context_tokens * 2))
        selection = env.assemble_context(state, actor=actor)
        self.assertNotIn(0, selection.provenance["included_transcript_turns"])
        self.assertFalse(selection.provenance["included_own_private_thought_turns"])
        self.assertLessEqual(selection.provenance["used_tokens"], state.max_context_tokens)


if __name__ == "__main__":
    unittest.main()
