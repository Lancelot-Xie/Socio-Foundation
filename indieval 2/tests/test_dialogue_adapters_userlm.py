import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.userlm import (
    UserLMAdapter,
    intent_decomposition_overlap,
    role_adherence_score,
)
from sim_eval.contracts import ResultStatus
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.environments.dialogue import EVALUATED_USER_ROLE, FIXED_ASSISTANT_ROLE
from sim_eval.errors import BackendStructuredOutputError


ROOT = Path(__file__).resolve().parents[1]


def metric(result, name):
    return next(item for item in result.metrics if item.name == name)


class CapturingReplayBackend(ReplayBackend):
    def __init__(self, responses):
        super().__init__(responses)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


class StructuredUserOutputFailureBackend(ReplayBackend):
    def generate(self, request):
        if request.metadata.get("actor") == EVALUATED_USER_ROLE:
            raise BackendStructuredOutputError("candidate emitted invalid JSON")
        return super().generate(request)


class FixedCodeVerifier:
    def __init__(self):
        self.completions = []

    def verify_case(self, case, assistant_completion):
        self.completions.append(assistant_completion)
        return 0.0, {
            "status": "completed",
            "passed": False,
            "reason": "failed_test",
            "completion_sha256": "test-completion-hash",
            "verifier_revision": "test-live-verifier-v1",
        }


class ContentAwareCodeVerifier:
    def __init__(self):
        self.completions = []

    def verify_case(self, case, assistant_completion):
        del case
        self.completions.append(assistant_completion)
        passed = "return value[::-1]" in assistant_completion
        index = len(self.completions) - 1
        return float(passed), {
            "status": "completed",
            "passed": passed,
            "verifier_passed": passed,
            "reason": "passed" if passed else "failed_test",
            "completion_sha256": f"test-completion-{index}",
            "verifier_revision": "test-content-aware-verifier-v1",
        }


class UnavailableThenFailCodeVerifier:
    def __init__(self):
        self.calls = 0

    def verify_case(self, case, assistant_completion):
        del case, assistant_completion
        index = self.calls
        self.calls += 1
        if index == 0:
            return None, {
                "status": "unavailable",
                "passed": None,
                "reason": "executor_unavailable",
                "completion_sha256": "test-unavailable",
            }
        return 0.0, {
            "status": "completed",
            "passed": False,
            "reason": "failed_test",
            "completion_sha256": "test-failed",
        }


class UserLMAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_fixture_suite(ROOT / "tests" / "fixtures")["userlm"][0]

    @staticmethod
    def execute(case, adapter=None, backend=None):
        adapter = adapter or UserLMAdapter()
        responses = adapter.replay_responses(case, seed=20260812)
        backend = backend or ReplayBackend(responses)
        return adapter.execute_case(
            case,
            backend=backend,
            run_id="userlm-test-run",
            seed=20260812,
            model="candidate-user",
        )

    def test_extrinsic_replay_scores_shards_and_verifier(self) -> None:
        self.assertEqual(
            UserLMAdapter().prompt_revision_for_case(self.cases[0]),
            UserLMAdapter.lic_prompt_revision,
        )
        first = self.execute(self.cases[0])
        second = self.execute(self.cases[1])
        for result in (first, second):
            self.assertEqual(result.status.value, "completed")
            self.assertEqual(result.metadata["evaluated_model_role"], EVALUATED_USER_ROLE)
            self.assertEqual(result.metadata["fixed_assistant_identity"]["role"], FIXED_ASSISTANT_ROLE)
            self.assertEqual(metric(result, "userlm.extrinsic.intent_coverage").value, 1.0)
            self.assertEqual(metric(result, "userlm.extrinsic.skip_non_required").value, True)
            self.assertEqual(metric(result, "userlm.extrinsic.assistant_task_score").value, 1.0)
        self.assertFalse(metric(first, "userlm.extrinsic.repeat_required").value)
        self.assertTrue(metric(second, "userlm.extrinsic.repeat_required").value)

    def test_arithmetic_verifier_accepts_thousands_separators_in_gold_and_response(self) -> None:
        base = self.cases[0]
        case = replace(base, gold={"task_answer": "114,200"})
        adapter = UserLMAdapter()
        responses = dict(adapter.replay_responses(case, seed=20260812))
        for request_id in tuple(responses):
            if ":assistant:" in request_id:
                responses[request_id] = "The exact total is $114,200."
        result = self.execute(case, adapter=adapter, backend=ReplayBackend(responses))
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertEqual(metric(result, "userlm.extrinsic.assistant_task_score").value, 1.0)

    def test_arithmetic_verifier_matches_comma_gold_to_plain_response(self) -> None:
        base = self.cases[0]
        case = replace(base, gold={"task_answer": "10,800"})
        adapter = UserLMAdapter()
        responses = dict(adapter.replay_responses(case, seed=20260812))
        for request_id in tuple(responses):
            if ":assistant:" in request_id:
                responses[request_id] = "FINAL: 10800"
        result = self.execute(case, adapter=adapter, backend=ReplayBackend(responses))
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertEqual(metric(result, "userlm.extrinsic.assistant_task_score").value, 1.0)

    def test_injected_code_verifier_overrides_trusted_fixture_cache(self) -> None:
        code_case = self.cases[1]
        verifier = FixedCodeVerifier()
        adapter = UserLMAdapter(code_task_verifier=verifier)
        result = self.execute(code_case, adapter=adapter)
        task_metric = metric(result, "userlm.extrinsic.assistant_task_score")
        self.assertEqual(task_metric.value, 0.0)
        self.assertEqual(task_metric.metadata["reason"], "failed_test")
        self.assertEqual(result.metadata["task_verification"]["completion_sha256"], "test-completion-hash")
        self.assertEqual(len(verifier.completions), 2)
        self.assertEqual(task_metric.metadata["evaluated_message_count"], 2)
        self.assertEqual(task_metric.metadata["passing_message_indices"], [])
        self.assertEqual(task_metric.metadata["selected_message_index"], 1)

    def test_code_assistant_score_passes_when_an_earlier_message_passes(self) -> None:
        code_case = self.cases[1]
        verifier = ContentAwareCodeVerifier()
        adapter = UserLMAdapter(code_task_verifier=verifier)
        responses = dict(adapter.replay_responses(code_case, seed=20260812))
        assistant_keys = sorted(key for key in responses if ":assistant:" in key)
        self.assertEqual(len(assistant_keys), 2)
        responses[assistant_keys[0]] = {
            **responses[assistant_keys[0]],
            "text": "def reverse_text(value: str) -> str:\n    return value[::-1]",
        }
        responses[assistant_keys[1]] = {
            **responses[assistant_keys[1]],
            "text": "That implementation should solve it.",
        }
        result = self.execute(
            code_case,
            adapter=adapter,
            backend=ReplayBackend(responses),
        )
        task_metric = metric(result, "userlm.extrinsic.assistant_task_score")
        self.assertEqual(task_metric.value, 1.0)
        self.assertEqual(len(verifier.completions), 2)
        self.assertEqual(task_metric.metadata["aggregation"], "one_if_any_assistant_message_passes")
        self.assertEqual(task_metric.metadata["passing_message_indices"], [0])
        self.assertEqual(task_metric.metadata["selected_message_index"], 0)
        self.assertEqual(
            [item["score"] for item in task_metric.metadata["per_message_results"]],
            [1.0, 0.0],
        )

    def test_code_assistant_score_is_unavailable_when_no_turn_passes_and_one_cannot_be_checked(self) -> None:
        code_case = self.cases[1]
        verifier = UnavailableThenFailCodeVerifier()
        adapter = UserLMAdapter(code_task_verifier=verifier)
        result = self.execute(code_case, adapter=adapter)
        task_metric = metric(result, "userlm.extrinsic.assistant_task_score")
        self.assertIsNone(task_metric.value)
        self.assertEqual(verifier.calls, 2)
        self.assertEqual(task_metric.metadata["availability"], "unavailable")
        self.assertEqual(task_metric.metadata["selected_message_index"], 0)
        self.assertEqual(
            [item["score"] for item in task_metric.metadata["per_message_results"]],
            [None, 0.0],
        )

    def test_candidate_user_never_sees_gold_or_fixed_assistant_private_policy(self) -> None:
        case = self.cases[0]
        adapter = UserLMAdapter()
        responses = adapter.replay_responses(case, seed=20260812)
        backend = CapturingReplayBackend(responses)
        result = adapter.execute_case(
            case,
            backend=backend,
            run_id="run",
            seed=20260812,
            model="candidate-user",
        )
        self.assertEqual(result.status.value, "completed")
        user_requests = [request for request in backend.requests if request.metadata.get("actor") == EVALUATED_USER_ROLE]
        assistant_requests = [request for request in backend.requests if request.metadata.get("actor") == FIXED_ASSISTANT_ROLE]
        self.assertTrue(user_requests and assistant_requests)
        user_text = "\n".join(message.content for request in user_requests for message in request.messages)
        assistant_text = "\n".join(message.content for request in assistant_requests for message in request.messages)
        self.assertNotIn('"task_answer":15', user_text)
        self.assertNotIn("Help the user complete", user_text)
        self.assertNotIn("Obtain the sum", assistant_text)
        self.assertTrue(all(request.model == "candidate-user" for request in user_requests))
        self.assertTrue(all(request.max_tokens == 512 for request in user_requests))
        self.assertTrue(all(request.model == "fixture-fixed-gpt4o-analogue" for request in assistant_requests))
        for request in (*user_requests, *assistant_requests):
            self.assertEqual(request.messages[0].role, "system")
            self.assertEqual(sum(m.role == "system" for m in request.messages), 1)
        self.assertIn("Act as a realistic human user", user_requests[0].messages[0].content)
        self.assertIn('"action":"message|end|refuse"', user_requests[0].messages[0].content)
        self.assertNotIn('"action":"message|end|refuse"', assistant_requests[0].messages[0].content)
        self.assertEqual(assistant_requests[0].messages[0].content, adapter._assistant_system_prompt(case))
        self.assertTrue(all(request.response_format is None for request in assistant_requests))
        shard_request = next(
            request for request in backend.requests if request.metadata.get("actor") == "shard_judge"
        )
        self.assertNotIn("uniqueItems", json.dumps(shard_request.response_format))

    def test_fixed_assistant_accepts_plain_multiline_code_without_json_envelope(self) -> None:
        case = self.cases[0]
        adapter = UserLMAdapter()
        responses = dict(adapter.replay_responses(case, seed=20260812))
        first_assistant = f"{case.case_id}:assistant:0"
        responses[first_assistant] = "Here is one way:\n```python\nprint(7 + 8)\n```"
        result = adapter.execute_case(
            case,
            backend=ReplayBackend(responses),
            run_id="run",
            seed=20260812,
            model="candidate-user",
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        assistant_event = next(event for event in result.trace if event.actor == FIXED_ASSISTANT_ROLE)
        self.assertIn("```python", assistant_event.content)

    def test_deepseek_lic_instruction_is_private_and_first_turn_only(self) -> None:
        case = self.cases[0]
        original = UserLMAdapter()
        runtime = replace(original.runtime_for_case(case), deepseek_lic_first_turn=True)
        adapted = UserLMAdapter(runtime_provenance=runtime)
        captured = []
        results = []
        for adapter in (original, adapted):
            backend = CapturingReplayBackend(adapter.replay_responses(case, seed=20260812))
            results.append(self.execute(case, adapter=adapter, backend=backend))
            captured.append(backend.requests)
        before, after = captured
        self.assertEqual([m.role for m in before[0].messages], ["system"])
        self.assertEqual([m.role for m in after[0].messages], ["system", "user"])
        self.assertEqual(after[0].messages[0], before[0].messages[0])
        self.assertIn("Generate the first user message now", after[0].messages[1].content)
        self.assertEqual(after[0].messages[1].metadata["visibility"], "transport_instruction")
        self.assertEqual(replace(after[0], messages=before[0].messages), before[0])
        # The transport instruction must not leak into later turns, judges,
        # the fixed assistant, public transcript, turn counts, or scores.
        self.assertEqual(after[1:], before[1:])
        self.assertEqual(results[1].trace, results[0].trace)
        self.assertEqual(results[1].metrics, results[0].metrics)
        self.assertEqual(results[1].prediction, results[0].prediction)
        self.assertNotEqual(adapted.prompt_revision_for_case(case), original.prompt_revision_for_case(case))
        self.assertNotIn("first_turn_request_revision", original.environment_identity_for_case(case))
        self.assertEqual(adapted.environment_identity_for_case(case)["first_turn_request_revision"],
                         "deepseek-lic-explicit-user-v1")

    def test_deepseek_first_turn_does_not_change_section3(self) -> None:
        base = self.cases[0]
        case = replace(base, input_data={
            "variant": "intrinsic_prism", "execution_mode": "section3_single_user_turn",
            "intent": "prepare for an exam", "conversation_history": "", "turn": 0,
        })
        original = UserLMAdapter()
        adapted = UserLMAdapter(runtime_provenance=replace(
            original.runtime_for_case(case), deepseek_lic_first_turn=True,
        ))
        self.assertEqual(original.build_request(case, model="candidate", seed=7),
                         adapted.build_request(case, model="candidate", seed=7))
        self.assertEqual(original.prompt_revision_for_case(case), adapted.prompt_revision_for_case(case))
        self.assertEqual(original.environment_identity_for_case(case), adapted.environment_identity_for_case(case))

    def test_malformed_shard_judge_is_unavailable_not_zero(self) -> None:
        case = self.cases[0]
        adapter = UserLMAdapter()
        responses = dict(adapter.replay_responses(case, seed=20260812))
        responses[f"{case.case_id}:judge:shards"] = "not-json"
        responses[f"{case.case_id}:judge:shards:judge_contract_retry:1"] = "not-json"
        responses[f"{case.case_id}:judge:shards:judge_contract_retry:2"] = "not-json"
        backend = CapturingReplayBackend(responses)
        result = adapter.execute_case(
            case,
            backend=backend,
            run_id="run",
            seed=20260812,
            model="candidate-user",
        )
        self.assertEqual(result.status.value, "completed")
        self.assertIsNone(metric(result, "userlm.extrinsic.intent_coverage").value)
        self.assertEqual(result.metadata["judge_records"]["shards"]["status"], "unavailable")
        self.assertEqual(result.metadata["judge_errors"]["shards"]["kind"], "ParseError")
        self.assertEqual(metric(result, "userlm.extrinsic.assistant_task_score").value, 1.0)

    def test_extrinsic_accepts_previously_filtered_first_words_without_retry(self) -> None:
        case = self.cases[0]
        adapter = UserLMAdapter()
        for opening in ("I", "i", "You", "you", "Here", "here"):
            with self.subTest(opening=opening):
                responses = dict(adapter.replay_responses(case, seed=20260812))
                messages = (
                    f"{opening} need help adding seven.",
                    f"{opening} also have the number eight.",
                )
                for turn, message in enumerate(messages):
                    responses[f"{case.case_id}:user:{turn}"] = json.dumps(
                        {"action": "message", "message": message}
                    )
                backend = CapturingReplayBackend(responses)
                result = adapter.execute_case(
                    case, backend=backend, run_id="run", seed=20260812, model="candidate-user",
                )
                self.assertEqual(result.status.value, "completed")
                self.assertNotIn("target_output_failure", result.metadata)
                self.assertEqual(metric(result, "userlm.extrinsic.assistant_task_score").value, 1.0)
                user_events = [event.content for event in result.trace if event.actor == EVALUATED_USER_ROLE]
                self.assertEqual(user_events, list(messages))
                self.assertFalse(any(
                    item["status"] == "guardrail_rejected" for item in result.metadata["request_audits"]
                ))
                self.assertFalse(any("guardrail_regen" in request.request_id for request in backend.requests))
                first_text = "\n".join(message.content for message in backend.requests[0].messages)
                self.assertNotIn("must not begin", first_text)
                self.assertIn("3–25 words", first_text)

    def test_extrinsic_word_count_retry_reports_observed_limit(self) -> None:
        case = self.cases[0]
        adapter = UserLMAdapter()
        responses = dict(adapter.replay_responses(case, seed=20260812))
        first_user = f"{case.case_id}:user:0"
        valid = responses[first_user]
        responses[first_user] = json.dumps(
            {"action": "message", "message": " ".join(f"word{index}" for index in range(26))}
        )
        responses[f"{first_user}:guardrail_regen:1"] = valid
        backend = CapturingReplayBackend(responses)
        result = adapter.execute_case(
            case,
            backend=backend,
            run_id="run",
            seed=20260812,
            model="candidate-user",
        )
        self.assertEqual(result.status.value, "completed")
        retry = next(request for request in backend.requests if request.request_id.endswith(":guardrail_regen:1"))
        retry_text = "\n".join(message.content for message in retry.messages)
        self.assertIn("received 26 words", retry_text)
        rejection = next(
            item for item in result.metadata["request_audits"] if item["status"] == "guardrail_rejected"
        )
        self.assertEqual(rejection["observed_word_count"], 26)

    def test_unconfigured_shard_judge_is_explicitly_unavailable(self) -> None:
        case = self.cases[0]
        base = UserLMAdapter()
        scoring = replace(
            base.provenance_for_case(case),
            shard_judge_model=None,
            shard_judge_revision=None,
        )
        adapter = UserLMAdapter(scoring_provenance=scoring)
        result = self.execute(case, adapter=adapter)
        self.assertEqual(result.status.value, "completed")
        self.assertIsNone(metric(result, "userlm.extrinsic.intent_coverage").value)
        self.assertEqual(metric(result, "userlm.extrinsic.intent_coverage").metadata["availability"], "unavailable")

    def test_empty_turn_refusal_and_max_turn_are_zero_score_capability_outcomes(self) -> None:
        case = self.cases[0]
        adapter = UserLMAdapter()
        baseline = dict(adapter.replay_responses(case, seed=20260812))
        first_user = f"{case.case_id}:user:0"

        empty = dict(baseline)
        empty[first_user] = json.dumps({"action": "message", "message": ""})
        empty_result = adapter.execute_case(case, backend=ReplayBackend(empty), run_id="run", seed=20260812, model="candidate")
        self.assertEqual(empty_result.status.value, "completed")
        self.assertIsNone(empty_result.error)
        self.assertEqual(empty_result.metadata["target_output_failure"]["kind"], "empty_turn")
        self.assertEqual(metric(empty_result, "userlm.extrinsic.intent_coverage").value, 0.0)

        refusal = dict(baseline)
        refusal[first_user] = json.dumps({"action": "refuse", "message": "I will not role-play this user."})
        refusal_result = adapter.execute_case(case, backend=ReplayBackend(refusal), run_id="run", seed=20260812, model="candidate")
        self.assertEqual(refusal_result.status.value, "completed")
        self.assertIsNone(refusal_result.error)
        self.assertEqual(refusal_result.metadata["target_output_failure"]["kind"], "refusal")
        self.assertEqual(refusal_result.prediction["terminal_reason"], "evaluated_user_refusal")

        constrained = UserLMAdapter(runtime_provenance=replace(adapter.runtime_for_case(case), max_user_turns=1))
        max_result = constrained.execute_case(case, backend=ReplayBackend(baseline), run_id="run", seed=20260812, model="candidate")
        self.assertEqual(max_result.status.value, "completed")
        self.assertIsNone(max_result.error)
        self.assertEqual(max_result.metadata["target_output_failure"]["stage"], "user_turn_limit")

    def test_backend_wrapped_user_json_violation_is_zero_not_failed(self) -> None:
        case = self.cases[0]
        adapter = UserLMAdapter()
        backend = StructuredUserOutputFailureBackend(adapter.replay_responses(case, seed=20260812))
        result = self.execute(case, adapter=adapter, backend=backend)
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["target_output_failure"]["kind"], "invalid_structured_output")
        self.assertEqual(metric(result, "userlm.extrinsic.intent_coverage").value, 0.0)

    def test_capability_failure_does_not_score_structurally_inapplicable_lic_metrics(self) -> None:
        base = self.cases[0]
        all_required = [
            {**dict(shard), "required": True}
            for shard in base.input_data["information_shards"]
        ]
        case = replace(
            base,
            input_data={**dict(base.input_data), "information_shards": all_required},
            metadata={
                **dict(base.metadata),
                "strata": {**dict(base.metadata["strata"]), "source_task": "math"},
            },
        )
        adapter = UserLMAdapter()
        completed = self.execute(case, adapter=adapter)
        completed_skip = metric(completed, "userlm.extrinsic.skip_non_required")
        completed_detector = metric(
            completed, "userlm.intrinsic.ai_detector_human_likelihood"
        )
        self.assertIsNone(completed_skip.value)
        self.assertEqual(completed_skip.metadata["availability"], "not_applicable")
        self.assertIsNone(completed_detector.value)
        self.assertEqual(completed_detector.metadata["availability"], "not_applicable")

        backend = StructuredUserOutputFailureBackend(
            adapter.replay_responses(case, seed=20260812)
        )

        result = self.execute(case, adapter=adapter, backend=backend)

        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertEqual(metric(result, "userlm.extrinsic.intent_coverage").value, 0.0)
        skip = metric(result, "userlm.extrinsic.skip_non_required")
        detector = metric(result, "userlm.intrinsic.ai_detector_human_likelihood")
        self.assertIsNone(skip.value)
        self.assertEqual(skip.metadata["availability"], "not_applicable")
        self.assertIsNone(detector.value)
        self.assertEqual(detector.metadata["availability"], "not_applicable")
        aggregate = adapter.aggregate([result])
        math_skip = aggregate["userlm.lic.math.skip_non_required"]
        self.assertIsNone(math_skip.value)
        self.assertEqual(math_skip.metadata["applicable_task_count"], 0)
        self.assertEqual(math_skip.metadata["not_applicable_task_count"], 1)
        self.assertIsNone(aggregate["userlm.lic.two_domain_macro.skip_non_required"].value)

    def test_intent_adherence_variant_uses_versioned_binary_judge(self) -> None:
        base = self.cases[0]
        metadata = dict(base.metadata)
        metadata["replay"] = {
            **metadata["replay"],
            "steps": [
                {"actor": EVALUATED_USER_ROLE, "output": {"action": "message", "message": "Who wrote the blue comet song?"}},
                {"actor": FIXED_ASSISTANT_ROLE, "output": {"action": "message", "message": "I am unsure; shall we discuss telescopes instead?"}},
                {"actor": EVALUATED_USER_ROLE, "output": {"action": "message", "message": "No, I still want the songwriter's name."}},
                {"actor": FIXED_ASSISTANT_ROLE, "output": {"action": "message", "message": "Let me focus on that question."}},
            ],
            "judges": {"intent_adherence": "REFUSED"},
        }
        case = replace(
            base,
            input_data={**base.input_data, "variant": "intrinsic_intent_adherence", "target_user_turns": 2},
            metadata=metadata,
        )
        result = self.execute(case)
        self.assertEqual(result.status.value, "completed")
        adherence = metric(result, "userlm.intrinsic.intent_adherence")
        self.assertEqual(adherence.value, 1.0)
        self.assertEqual(adherence.metadata["judge_revision"], "fixture-intent-judge-v1")
        self.assertTrue(adherence.metadata["replayed"])

    def test_prism_variant_reports_decomposition_and_termination_f1(self) -> None:
        base = self.cases[0]
        metadata = dict(base.metadata)
        metadata["evaluation"] = {"real_user_ended": True}
        metadata["replay"] = {
            **metadata["replay"],
            "steps": [
                {"actor": EVALUATED_USER_ROLE, "output": {"action": "message", "message": "Can we work through a small addition?"}},
                {"actor": FIXED_ASSISTANT_ROLE, "output": {"action": "message", "message": "Yes, tell me the numbers."}},
                {"actor": EVALUATED_USER_ROLE, "output": {"action": "end", "message": ""}},
            ],
            "judges": {},
        }
        case = replace(
            base,
            input_data={**base.input_data, "variant": "intrinsic_prism", "target_user_turns": 4},
            metadata=metadata,
        )
        result = self.execute(case)
        self.assertEqual(result.status.value, "completed")
        self.assertIsNotNone(metric(result, "userlm.intrinsic.intent_decomposition_overlap").value)
        aggregate = UserLMAdapter().aggregate([result])
        decomposition = aggregate["userlm.intrinsic.intent_decomposition_overlap"]
        self.assertEqual(decomposition.direction, "lower_is_better")
        self.assertEqual(decomposition.metadata["intent_scope"], "conversation_global_intent")
        self.assertEqual(aggregate["userlm.intrinsic.termination_precision"].value, 1.0)
        self.assertEqual(aggregate["userlm.intrinsic.termination_recall"].value, 1.0)
        self.assertEqual(aggregate["userlm.intrinsic.termination_f1"].value, 1.0)

    def test_role_and_decomposition_metric_rules(self) -> None:
        score, mentions = role_adherence_score("I think the amber answer fits.", ["amber", "blue", "green", "red"])
        self.assertEqual((score, mentions), (0.0, 1))
        score, mentions = role_adherence_score("amber blue green red were all listed", ["amber", "blue", "green", "red"])
        self.assertEqual((score, mentions), (1.0, 4))
        self.assertAlmostEqual(intent_decomposition_overlap("request a red bicycle", ["I would like a red bicycle"]), 2 / 3)

    def test_section3_prism_is_one_next_user_turn_not_a_dialogue_rollout(self) -> None:
        base = self.cases[0]
        case = replace(
            base,
            input_data={
                "variant": "intrinsic_prism",
                "execution_mode": "section3_single_user_turn",
                "intent": "prepare for an exam while maintaining concentration",
                "conversation_history": "<user>: How should I study?\n<assistant>: Make a schedule.\n",
                "turn": 1,
                "is_last_turn": False,
            },
        )
        adapter = UserLMAdapter()
        self.assertEqual(adapter.prompt_revision_for_case(case), adapter.prompt_revision)
        request = adapter.build_request(case, model="candidate-user", seed=7)
        self.assertEqual(request.max_tokens, 200)
        self.assertEqual(request.metadata["prompt_source"], "UserLM Appendix Figure 9")
        self.assertEqual(request.metadata["actor"], EVALUATED_USER_ROLE)
        result = adapter.execute_case(
            case,
            backend=ReplayBackend({request.request_id: "Could you suggest ways to stay focused?"}),
            run_id="section3-test",
            seed=7,
            model="candidate-user",
        )
        self.assertEqual(result.status.value, "completed")
        self.assertEqual(result.prediction["user_turn"], "Could you suggest ways to stay focused?")
        self.assertEqual(result.metadata["execution_mode"], "section3_single_user_turn")
        self.assertIsNotNone(metric(result, "userlm.intrinsic.intent_decomposition_overlap").value)

    def test_section3_prism_uses_injected_local_ai_detector(self) -> None:
        base = self.cases[0]
        case = replace(
            base,
            input_data={
                "variant": "intrinsic_prism",
                "execution_mode": "section3_single_user_turn",
                "intent": "prepare for an exam",
                "conversation_history": "<user>: How should I study?\n<assistant>: Make a schedule.\n",
                "turn": 1,
                "is_last_turn": False,
            },
        )
        base_adapter = UserLMAdapter()
        scoring = replace(
            base_adapter.provenance_for_case(case),
            ai_detector_model="yaoandy107/greyscope-qwen3.5-4b",
            ai_detector_revision="bb25d4158a6795c6fce225156c0e330084ffb665",
            ai_detector_contract_revision="greyscope-v1-calibrated-seqcls-v1",
        )
        seen = []

        def detector(text):
            seen.append(text)
            return {
                "human_likelihood": 0.82,
                "ai_involvement": 0.18,
                "label": "AI-edited",
                "model_revision": "bb25d4158a6795c6fce225156c0e330084ffb665",
            }

        adapter = UserLMAdapter(
            scoring_provenance=scoring,
            ai_text_detector=detector,
        )
        request = adapter.build_request(case, model="candidate-user", seed=7)
        output = "Could you suggest ways to stay focused?"
        result = adapter.execute_case(
            case,
            backend=ReplayBackend({request.request_id: output}),
            run_id="section3-detector-test",
            seed=7,
            model="candidate-user",
        )
        detector_metric = metric(
            result, "userlm.intrinsic.ai_detector_human_likelihood"
        )
        self.assertEqual(seen, [output])
        self.assertEqual(detector_metric.value, 0.82)
        self.assertEqual(detector_metric.metadata["label"], "AI-edited")
        self.assertEqual(result.metadata["ai_detector"]["source"], "live_local_detector")

    def test_section3_prism_batches_detector_and_preserves_case_order(self) -> None:
        base = self.cases[0]
        first_case = replace(
            base,
            case_id=f"{base.case_id}-batch-1",
            input_data={
                "variant": "intrinsic_prism",
                "execution_mode": "section3_single_user_turn",
                "intent": "prepare for an exam",
                "conversation_history": "<user>: How should I study?\n<assistant>: Make a schedule.\n",
                "turn": 1,
                "is_last_turn": False,
            },
        )
        second_case = replace(
            first_case,
            case_id=f"{base.case_id}-batch-2",
            input_data={**dict(first_case.input_data), "turn": 2},
        )
        scoring = replace(
            UserLMAdapter().provenance_for_case(first_case),
            ai_detector_model="yaoandy107/greyscope-qwen3.5-4b",
            ai_detector_revision="bb25d4158a6795c6fce225156c0e330084ffb665",
            ai_detector_contract_revision="greyscope-v1-calibrated-seqcls-v1",
        )
        seen_batches = []

        class BatchDetector:
            config = SimpleNamespace(batch_size=4)

            def __call__(self, text):
                raise AssertionError("deferred detector must not score one item at a time")

            def score_batch(self, texts):
                seen_batches.append(tuple(texts))
                return (
                    {"human_likelihood": 0.81, "label": "human", "batch": {"position": 0}},
                    {"human_likelihood": 0.27, "label": "AI-edited", "batch": {"position": 1}},
                )

        adapter = UserLMAdapter(
            scoring_provenance=scoring,
            ai_text_detector=BatchDetector(),
            defer_ai_text_detection=True,
        )
        outputs = ("Please suggest a focused study routine.", "How can that routine stay manageable?")
        pending = []
        for case, output in zip((first_case, second_case), outputs):
            request = adapter.build_request(case, model="candidate-user", seed=7)
            result = adapter.execute_case(
                case,
                backend=ReplayBackend({request.request_id: output}),
                run_id="section3-detector-batch-test",
                seed=7,
                model="candidate-user",
            )
            self.assertIsNone(
                metric(result, "userlm.intrinsic.ai_detector_human_likelihood").value
            )
            pending.append((case, result))

        finalized = adapter.finalize_ai_text_detector_batch(pending)
        self.assertEqual(seen_batches, [outputs])
        self.assertEqual(
            [metric(result, "userlm.intrinsic.ai_detector_human_likelihood").value for result in finalized],
            [0.81, 0.27],
        )
        self.assertEqual(
            [result.metadata["ai_detector"]["batch"]["position"] for result in finalized],
            [0, 1],
        )

    def test_role_adherence_dialogue_variant(self) -> None:
        base = self.cases[0]
        metadata = dict(base.metadata)
        metadata["replay"] = {
            **metadata["replay"],
            "steps": [
                {"actor": EVALUATED_USER_ROLE, "output": {"action": "message", "message": "Which option fits the riddle?"}},
                {"actor": FIXED_ASSISTANT_ROLE, "output": {"action": "message", "message": "I am unsure; can you tell me?"}},
                {"actor": EVALUATED_USER_ROLE, "output": {"action": "message", "message": "Please make your own best guess."}},
                {"actor": FIXED_ASSISTANT_ROLE, "output": {"action": "message", "message": "I will reason it through."}},
            ],
            "judges": {},
        }
        case = replace(
            base,
            input_data={
                **base.input_data,
                "variant": "intrinsic_role_adherence",
                "choices": ["amber", "blue", "green", "red"],
                "target_user_turns": 2,
            },
            metadata=metadata,
        )
        result = self.execute(case)
        self.assertEqual(metric(result, "userlm.intrinsic.role_adherence").value, 1.0)

    def test_group_distribution_metrics_require_repetitions(self) -> None:
        first = self.execute(self.cases[0])
        second = replace(self.execute(self.cases[0]), repetition=1)
        aggregate = UserLMAdapter().aggregate([first, second])
        self.assertEqual(aggregate["userlm.extrinsic.turn_variance"].value, 0.0)
        self.assertEqual(aggregate["userlm.extrinsic.unigram_difference"].value, 0.0)
        singleton = UserLMAdapter().aggregate([first])
        self.assertIsNone(singleton["userlm.extrinsic.turn_variance"].value)
        self.assertIsNone(singleton["userlm.extrinsic.unigram_difference"].value)

    def test_lic_aggregates_repetitions_by_task_then_reports_domain_macro(self) -> None:
        math = replace(
            self.execute(self.cases[0]),
            metadata={**self.execute(self.cases[0]).metadata, "source_task": "math"},
        )
        code = replace(
            self.execute(self.cases[1]),
            metadata={**self.execute(self.cases[1]).metadata, "source_task": "code"},
        )
        results = [
            math,
            replace(math, repetition=1),
            code,
            replace(code, repetition=1),
        ]
        aggregate = UserLMAdapter().aggregate(results)
        self.assertEqual(aggregate["userlm.lic.math.task_count"].value, 1)
        self.assertEqual(aggregate["userlm.lic.code.task_count"].value, 1)
        self.assertEqual(aggregate["userlm.lic.math.assistant_task_score"].denominator, 1)
        self.assertEqual(aggregate["userlm.lic.code.assistant_task_score"].denominator, 1)
        macro = aggregate["userlm.lic.two_domain_macro.assistant_task_score"]
        self.assertEqual(macro.value, 1.0)
        self.assertEqual(macro.denominator, 2)
        self.assertEqual(
            macro.metadata["aggregation"],
            "equal_weight_mean_of_code_and_math_task_means",
        )


if __name__ == "__main__":
    unittest.main()
