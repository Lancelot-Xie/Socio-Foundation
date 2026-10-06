"""Regression coverage for opt-in evaluated-model transport adapters."""
import json
from dataclasses import replace
from pathlib import Path
import unittest

from sim_eval.backends.episode_budget import EpisodeOutputTokenBudgetBackend, episode_budget_options_from_role
from sim_eval.backends.openai_compatible import VLLMBackend
from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.choice import DisplayedChoice, parse_strict_choice
from sim_eval.benchmarks.social_r1 import SocialR1Adapter
from sim_eval.benchmarks.userlm import UserLMAdapter, END_CONVERSATION
from sim_eval.contracts import ChatMessage, ModelRequest, ModelResponse, TokenUsage
from sim_eval.data.loaders import load_fixture_suite, load_import_spec, load_local_cases
from sim_eval.errors import ConfigurationError, ParseError
from sim_eval.model_adaptation import (
    COSER_FORMAT, USERLM_NATIVE, adapt_evaluated_request, adapted_choice_text,
    annotate_adapted_response, adaptation_identity,
)
from sim_eval.runtime_config import apply_evaluated_model, EVALUATED_ROLE_BY_BENCHMARK, load_config_document
from sim_eval.smoke import _deepseek_role, _prepare_smoke

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT.parent


class CaptureBackend:
    def __init__(self, text="hello"):
        self.requests = []
        self.text = text

    def generate(self, request):
        self.requests.append(request)
        return ModelResponse(text=self.text, usage=TokenUsage(completion_tokens=2))


class CaptureCounter:
    def __init__(self):
        self.requests = []

    def count_prompt(self, request, **kwargs):
        self.requests.append(request)
        return 20


class ModelAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_fixture_suite(ROOT / "tests/fixtures")["userlm"][0]
        cls.choices = tuple(DisplayedChoice(label, label, text, i) for i, (label, text) in enumerate(
            [("A", "No"), ("B", "Yes"), ("C", "Maybe")]))

    def test_absent_adapter_preserves_request_response_and_identity(self):
        request = ModelRequest(messages=(ChatMessage("user", "hello"),), model="ordinary-model")
        response = ModelResponse(text="B")
        self.assertIs(adapt_evaluated_request(request, None), request)
        self.assertIs(annotate_adapted_response(response, None), response)
        self.assertEqual(adaptation_identity({}), {})
        self.assertEqual(adapted_choice_text(response, self.choices), "B")
        self.assertEqual(episode_budget_options_from_role({}), {})

    def test_native_roles_flipped_before_token_count_and_support_untouched(self):
        backend, counter = CaptureBackend(), CaptureCounter()
        wrapped = EpisodeOutputTokenBudgetBackend(
            backend, evaluated_role="evaluated_user", evaluated_model="same-model",
            support_roles=("fixed_assistant",), model_adapter=USERLM_NATIVE,
            token_counter=counter, model_context_tokens=500,
        )
        request = ModelRequest(
            messages=(ChatMessage("system", "intent"), ChatMessage("assistant", "my earlier turn"),
                      ChatMessage("user", "assistant reply")), model="same-model", max_tokens=30,
            metadata={"actor": "evaluated_user"},
        )
        response = wrapped.generate(request)
        self.assertEqual([m.role for m in backend.requests[0].messages], ["system", "user", "assistant"])
        self.assertEqual(counter.requests[0].messages, backend.requests[0].messages)
        self.assertEqual([m.content for m in request.messages], [m.content for m in backend.requests[0].messages])
        self.assertEqual([m.role for m in request.messages], ["system", "assistant", "user"])
        self.assertEqual(response.raw["_sim_eval"]["model_adapter"], USERLM_NATIVE)
        support = replace(request, metadata={"actor": "fixed_assistant"})
        wrapped.generate(support)
        self.assertIs(backend.requests[-1], support)
        self.assertEqual(len(counter.requests), 1)
        self.assertEqual(wrapped.evaluated_request_count, 1)

    def test_coser_does_not_flip_roles_and_transform_is_idempotent(self):
        request = ModelRequest(messages=(ChatMessage("user", "hello"),), model="coser")
        new = adapt_evaluated_request(request, COSER_FORMAT)
        self.assertEqual(new.messages, request.messages)
        self.assertIs(adapt_evaluated_request(new, COSER_FORMAT), new)
        native = adapt_evaluated_request(request, USERLM_NATIVE)
        self.assertIs(adapt_evaluated_request(native, USERLM_NATIVE), native)

    def test_choice_compatibility_requires_explicit_adapter_and_matching_text(self):
        for adapter in (USERLM_NATIVE, COSER_FORMAT):
            for text in ("B", "b.", "B)", "B. Yes", "b.  yes"):
                response = annotate_adapted_response(ModelResponse(text=text), adapter)
                normalized = adapted_choice_text(response, self.choices)
                self.assertEqual(parse_strict_choice(normalized, self.choices, allow_answer_tag=True).display_id, "B")
                self.assertEqual(response.text, text)
        for text in ("B. No", "B or C", "I think B", "B. Yes because it is right", "<answer>B", "D"):
            response = annotate_adapted_response(ModelResponse(text=text), COSER_FORMAT)
            self.assertEqual(adapted_choice_text(response, self.choices), text)
        with self.assertRaises(ParseError):
            parse_strict_choice(adapted_choice_text(ModelResponse(text="B. Yes"), self.choices), self.choices)

    def test_ranked_answers_never_fabricate_missing_choices(self):
        choices = tuple(DisplayedChoice(c, c, c, i) for i, c in enumerate("ABCDEFGHIJKLMNOPQRST"))
        valid = annotate_adapted_response(ModelResponse(text="C, A, T, E, B"), COSER_FORMAT)
        self.assertEqual(adapted_choice_text(valid, choices, ranking=True), "<answer>C,A,T,E,B</answer>")
        for text in ("A", "A,B,C", "A,A,B,C,D", "A,B,C,D,Z", "X1,X2,X3,X4,X5"):
            response = annotate_adapted_response(ModelResponse(text=text), USERLM_NATIVE)
            self.assertEqual(adapted_choice_text(response, choices, ranking=True), text)


    def make_userlm(self, profile):
        case = self.cases[0]
        base = UserLMAdapter()
        runtime = replace(base.runtime_for_case(case), model_adapter=profile)
        adapter = UserLMAdapter(runtime_provenance=runtime)
        return case, adapter, adapter.runtime_for_case(case)

    def test_native_lic_uses_strict_json_and_three_to_sixty_words(self):
        case, adapter, runtime = self.make_userlm(USERLM_NATIVE)
        state = adapter.environment.reset_with_spec(case, adapter.dialogue_spec_for_case(case), seed=1)
        first = adapter.build_user_request(case, state, model="userlm_8b", seed=1)
        payload = VLLMBackend().build_payload(first, structured_mode="structured_outputs")
        schema = payload["structured_outputs"]["json"]
        self.assertEqual(set(schema["properties"]), {"action", "message"})
        self.assertEqual(set(schema["required"]), {"action", "message"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["action"]["enum"], ["message", "end", "refuse"])
        self.assertNotIn('"action":"message|end|refuse"', first.messages[0].content)
        self.assertIn('"action":"message"', first.messages[0].content)
        self.assertIn("3–60 words", first.messages[0].content)
        self.assertNotIn("3–25", first.messages[0].content)
        self.assertEqual([m.role for m in first.messages], ["system"])
        self.assertEqual(adapter.environment_identity_for_case(case)["guardrail_word_range"], [3, 60])
        self.assertEqual(runtime.max_guardrail_regenerations, 8)
        from sim_eval.environments.dialogue import DialogueAction
        for count in (2, 3, 25, 26, 60, 61):
            action = DialogueAction("message", " ".join(["word"] * count))
            expected = None if 3 <= count <= 60 else "word_count_out_of_range"
            self.assertEqual(adapter._guardrail_rejection(case, state, action, runtime), expected)
        adapter.environment.apply(state, actor="evaluated_user", action=DialogueAction("message", "Help me with this task"))
        adapter.environment.apply(state, actor="fixed_assistant", action=DialogueAction("message", "Which details are known?"))
        second = adapt_evaluated_request(adapter.build_user_request(case, state, model="userlm_8b", seed=1), USERLM_NATIVE)
        self.assertEqual([m.role for m in second.messages[1:]], ["user", "assistant"])
        self.assertEqual(json.loads(second.messages[1].content), {"action":"message", "message":"Help me with this task"})
        self.assertEqual(second.messages[2].content, "Which details are known?")
        self.assertEqual(second.response_format, first.response_format)
        valid = '{"action":"message","message":"Need help with this"}'
        self.assertEqual(adapter._parse_user_action(case, valid).message, "Need help with this")
        for invalid in (
            "Need help with this", '{"action":"message|end|refuse","message":"hello there friend"}',
            '{"action":"message","message":"hello there friend","extra":1}',
            '{"action":"message"}',
        ):
            with self.assertRaises(ParseError):
                adapter._parse_user_action(case, invalid)
        end = adapter._parse_user_action(case, END_CONVERSATION)
        self.assertEqual(adapter._guardrail_rejection(case, state, end, runtime), "termination_token_prohibited")

    def test_coser_lic_exact_schema_and_valid_example_reach_vllm_payload(self):
        case, adapter, _ = self.make_userlm(COSER_FORMAT)
        request = adapter.build_request(case, model="coser_8b", seed=1)
        self.assertNotIn('"action":"message|end|refuse"', request.messages[0].content)
        self.assertEqual(request.messages[-1].role, "user")
        payload = VLLMBackend().build_payload(request, structured_mode="structured_outputs")
        schema = payload["structured_outputs"]["json"]
        self.assertEqual(schema["properties"]["action"]["enum"], ["message", "end", "refuse"])
        self.assertEqual(set(schema["required"]), {"action", "message"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(adapter._parse_user_action(case, '{"action":"message","message":"Please help with this"}').action, "message")
        with self.assertRaises(ParseError):
            adapter._parse_user_action(case, '{"action":"message|end|refuse","message":"hello"}')

    def test_native_complete_episode_uses_same_judges_and_scores(self):
        case, adapter, _ = self.make_userlm(USERLM_NATIVE)
        legacy = UserLMAdapter()
        original = dict(legacy.replay_responses(case, seed=13))
        old_result = legacy.execute_case(case, backend=ReplayBackend(original), run_id="old", seed=13, model="model")
        backend = EpisodeOutputTokenBudgetBackend(ReplayBackend(original), evaluated_role="evaluated_user",
                    evaluated_model="model", support_roles=("fixed_assistant", "intent_judge", "shard_judge"),
                    model_adapter=USERLM_NATIVE)
        new_result = adapter.execute_case(case, backend=backend, run_id="new", seed=13, model="model")
        self.assertNotIn("target_output_failure", new_result.metadata)
        self.assertEqual({m.name: m.value for m in old_result.metrics}, {m.name: m.value for m in new_result.metrics})
        self.assertEqual(old_result.prediction, new_result.prediction)

    def test_lic_json_changes_are_isolated_from_other_profiles_and_variants(self):
        case, native, _ = self.make_userlm(USERLM_NATIVE)
        for profile in (None, COSER_FORMAT):
            _, adapter, runtime = self.make_userlm(profile)
            self.assertEqual(runtime.guardrail_max_words, 25)
            request = adapter.build_request(case, model="other", seed=1)
            self.assertIn("3–25 words", request.messages[0].content)
            self.assertNotIn("userlm-lic-json-3-60-v1", adapter.prompt_revision_for_case(case))
        for mode in ("interactive_dialogue", "section3_single_user_turn"):
            other = replace(case, input_data={**case.input_data,
                "variant":"intrinsic_prism", "execution_mode":mode, "conversation_history":""})
            self.assertEqual(native.runtime_for_case(other).guardrail_max_words, 25)
            request = native.build_request(other, model="userlm", seed=1)
            self.assertIsNone(request.response_format)
            self.assertNotIn("3–60", request.messages[0].content)
            self.assertNotIn("userlm-lic-json-3-60-v1", native.prompt_revision_for_case(other))

    def test_native_lic_regeneration_keeps_json_schema_and_sixty_word_limit(self):
        case, adapter, _ = self.make_userlm(USERLM_NATIVE)
        replay = ReplayBackend(UserLMAdapter().replay_responses(case, seed=13))
        captured = []

        class OneLongTurn:
            def generate(self, request):
                captured.append(request)
                if len(captured) == 1:
                    return ModelResponse(text=json.dumps({"action":"message", "message":" ".join(["word"] * 61)}))
                return replay.generate(replace(request, request_id=request.request_id.split(":guardrail_regen:")[0]))

        backend = EpisodeOutputTokenBudgetBackend(OneLongTurn(), evaluated_role="evaluated_user",
            evaluated_model="model", support_roles=("fixed_assistant", "intent_judge", "shard_judge"),
            model_adapter=USERLM_NATIVE)
        result = adapter.execute_case(case, backend=backend, run_id="test", seed=13, model="model")
        self.assertNotIn("target_output_failure", result.metadata)
        retry = captured[1]
        self.assertIn(":guardrail_regen:1", retry.request_id)
        self.assertEqual(retry.response_format, captured[0].response_format)
        self.assertIn("3–60 words", retry.messages[-1].content)
        self.assertIn('"action":"message"', retry.messages[-1].content)
        self.assertNotIn("without JSON", retry.messages[-1].content)
        self.assertIn("required 3–60", retry.messages[-1].content)
        self.assertEqual([m.role for m in retry.messages], ["system", "user", "assistant"])

    def test_only_evaluated_role_receives_model_adapter(self):
        model = {"backend":"vllm", "profile":"vllm", "model":"userlm_8b", "model_revision":"pinned", "model_adapter":USERLM_NATIVE}
        support = _deepseek_role({}, model=model)
        evaluated = _deepseek_role({}, model=model, evaluated_role=True)
        self.assertNotIn("model_adapter", support)
        self.assertEqual(evaluated["model_adapter"], USERLM_NATIVE)
        document = {"benchmark_id":"social_r1", "roles":{"evaluated_model":evaluated, "judge":support}}
        replacement = apply_evaluated_model(document, {"model":"other", "model_revision":"other"})
        self.assertNotIn("model_adapter", replacement["roles"]["evaluated_model"])
        self.assertEqual(replacement["roles"]["judge"], support)

    def test_profiles_have_separate_prompt_and_runtime_identity(self):
        case = self.cases[0]
        old = UserLMAdapter()
        original_revision = old.prompt_revision_for_case(case)
        self.assertNotIn("model_adapter", old.runtime_for_case(case).to_dict())
        for profile in (USERLM_NATIVE, COSER_FORMAT):
            _, adapter, runtime = self.make_userlm(profile)
            self.assertNotEqual(adapter.prompt_revision_for_case(case), original_revision)
            self.assertEqual(runtime.to_dict()["model_adapter"], profile)
            self.assertEqual(adaptation_identity({"model_adapter":profile}), {"model_adapter":profile})
        with self.assertRaises(ConfigurationError):
            adaptation_identity({"model_adapter":"typo"})



if __name__ == "__main__":
    unittest.main()
