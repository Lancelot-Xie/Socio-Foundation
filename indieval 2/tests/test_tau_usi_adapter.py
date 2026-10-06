import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.artifacts import CheckpointStore
from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.tau_usi import TauUSIAdapter
from sim_eval.contracts import RunIdentityInput, RunManifest
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.environments.tau_usi import ASSISTANT_ROLE, USER_ROLE
from sim_eval.errors import BackendStructuredOutputError, BackendTimeoutError, ResumeConflictError
from sim_eval.execution_provenance import support_protocol_identity


ROOT = Path(__file__).resolve().parents[1]


class CapturingReplayBackend(ReplayBackend):
    def __init__(self, responses, **kwargs):
        super().__init__(responses, **kwargs)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


class FailOnceReplayBackend(ReplayBackend):
    def __init__(self, responses, fail_key):
        super().__init__(responses)
        self.fail_key = fail_key
        self.calls = {}

    def generate(self, request):
        key = self.request_key(request)
        self.calls[key] = self.calls.get(key, 0) + 1
        if key == self.fail_key and self.calls[key] == 1:
            raise BackendTimeoutError("one synthetic transient timeout")
        return super().generate(request)


class StructuredSurveyFailureBackend(ReplayBackend):
    def generate(self, request):
        if request.metadata.get("stage") == "post_interaction_survey":
            raise BackendStructuredOutputError("candidate emitted invalid survey JSON")
        return super().generate(request)


class TauUSIAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_fixture_suite(ROOT / "tests" / "fixtures")["tau_usi"][0]

    @staticmethod
    def execute(case, adapter=None, backend=None, model="candidate-user"):
        adapter = adapter or TauUSIAdapter()
        responses = adapter.replay_responses(case, seed=20260812)
        backend = backend or ReplayBackend(responses)
        return adapter.execute_case(
            case,
            backend=backend,
            run_id="tau-adapter-run",
            seed=20260812,
            model=model,
        )

    def test_retail_and_airline_execute_with_fixed_assistant_and_tools(self) -> None:
        for case in self.cases:
            with self.subTest(domain=case.input_data["domain"]):
                result = self.execute(case)
                self.assertEqual(result.status.value, "completed")
                self.assertTrue(result.metadata["episode_complete"])
                self.assertEqual(result.metadata["evaluated_model_role"], USER_ROLE)
                self.assertEqual(result.metadata["fixed_assistant_identity"]["role"], ASSISTANT_ROLE)
                self.assertEqual(result.prediction["tool_call_count"], 2)
                self.assertEqual(result.prediction["user_turn_count"], 2)
                self.assertEqual(result.prediction["terminal_reason"], "user_stop")
                self.assertEqual(
                    [event.kind for event in result.trace].count("tool_observation"), 2
                )
                self.assertEqual(result.metadata["tau_usi"]["environment_reward"], 1.0)
                self.assertEqual(set(result.metadata["tau_usi"]["human_references"]), {"fixture_h1", "fixture_h2", "fixture_h3"})

    def test_model_requests_never_receive_human_references_or_difficulty(self) -> None:
        case = self.cases[0]
        adapter = TauUSIAdapter()
        responses = adapter.replay_responses(case, seed=20260812)
        backend = CapturingReplayBackend(responses)
        result = self.execute(case, adapter=adapter, backend=backend)
        self.assertEqual(result.status.value, "completed")
        combined = "\n".join(message.content for request in backend.requests for message in request.messages)
        self.assertNotIn("fixture_h1", combined)
        self.assertNotIn("held-fixed synthetic simulator pool", combined)
        self.assertNotIn("I need the green mug in ORD-RIVER-7319 changed to blue", combined)
        user_requests = [request for request in backend.requests if request.metadata.get("actor") == USER_ROLE]
        interaction_user_requests = [
            request
            for request in user_requests
            if request.metadata.get("stage") != "post_interaction_survey"
        ]
        assistant_requests = [
            request for request in backend.requests if request.metadata.get("actor") == ASSISTANT_ROLE
        ]
        self.assertTrue(user_requests and assistant_requests)
        self.assertTrue(all(request.model == "candidate-user" for request in user_requests))
        self.assertTrue(
            all(request.model == "fixture-fixed-service-agent" for request in assistant_requests)
        )
        for request in (*user_requests, *assistant_requests):
            self.assertEqual(request.messages[0].role, "system")
            if request.metadata["actor"] == USER_ROLE:
                self.assertEqual(sum(message.role == "system" for message in request.messages), 1)
        self.assertTrue(any(len(request.messages) > 2 for request in interaction_user_requests))
        self.assertTrue(any(message.metadata.get("source") == "tau_tool_feedback" for request in assistant_requests for message in request.messages))
        self.assertIn("###STOP###", interaction_user_requests[0].messages[0].content)
        self.assertIn("raw user utterance", interaction_user_requests[0].messages[0].content)
        self.assertTrue(all(request.response_format is None for request in interaction_user_requests))
        self.assertTrue(all(request.max_tokens == 256 for request in user_requests))
        self.assertIn("Once the goal is satisfied", interaction_user_requests[0].messages[0].content)
        self.assertIn("<function=", assistant_requests[0].messages[0].content)
        self.assertIn("Only call one function at a time", assistant_requests[0].messages[0].content)
        self.assertTrue(all(request.response_format is None for request in assistant_requests))
        self.assertTrue(all(not request.tools for request in assistant_requests))
        self.assertTrue(all(request.max_tokens == 512 for request in assistant_requests))

    def test_empty_user_reply_ends_dialogue_and_preserves_reward(self) -> None:
        case = self.cases[0]
        adapter = TauUSIAdapter()
        responses = dict(adapter.replay_responses(case, seed=20260812))
        first_user = next(key for key in responses if ":user:0" in key)
        responses[first_user] = {"text": "   ", "usage": {"total_tokens": 1}}
        result = self.execute(case, adapter=adapter, backend=ReplayBackend(responses))
        self.assertEqual(result.status.value, "completed")
        self.assertIsNone(result.error)
        self.assertEqual(result.prediction["terminal_reason"], "empty_user_response")
        self.assertNotIn("target_output_failure", result.metadata)
        self.assertTrue(result.metadata["episode_complete"])
        self.assertEqual(
            next(item for item in result.metrics if item.name == "tau_usi.environment_reward").value,
            1.0,
        )

    def test_survey_backend_failure_is_not_a_received_missing_answer(self) -> None:
        case = self.cases[0]
        adapter = TauUSIAdapter()
        responses = adapter.replay_responses(case, seed=20260812)
        result = self.execute(
            case,
            adapter=adapter,
            backend=StructuredSurveyFailureBackend(responses),
        )
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.error.stage, "survey_backend")

    def test_invalid_assistant_tool_observation_allows_correction(self) -> None:
        case = self.cases[0]
        adapter = TauUSIAdapter()
        responses = dict(adapter.replay_responses(case, seed=20260812))
        first_assistant = next(key for key in responses if ":assistant:" in key)
        original_tool = responses[first_assistant]
        original_message_key = first_assistant.replace("step0", "step1")
        original_message = responses.pop(original_message_key)
        responses[first_assistant] = {
            "text": "",
            "raw": {
                "_sim_eval": {
                    "native_tool_call": {"name": "think", "arguments": {}}
                }
            },
            "usage": {"total_tokens": 1},
        }
        responses[first_assistant.replace("step0", "step1")] = original_tool
        responses[first_assistant.replace("step0", "step2")] = original_message
        result = self.execute(case, adapter=adapter, backend=ReplayBackend(responses))
        self.assertEqual(result.status.value, "completed")
        self.assertIsNone(result.error)
        error_observation = next(
            event
            for event in result.trace
            if event.kind == "tool_observation"
            and event.metadata.get("tool_error_kind") == "unknown_action"
        )
        self.assertEqual(error_observation.content, "Unknown action think")
        self.assertEqual(result.metadata["tau_usi"]["environment_reward"], 1.0)

    def test_user_backend_failure_retries_then_fails(self) -> None:
        case = self.cases[0]
        adapter = TauUSIAdapter()
        responses = adapter.replay_responses(case, seed=20260812)
        first_user = next(key for key in responses if ":user:0" in key)
        result = self.execute(
            case,
            adapter=adapter,
            backend=ReplayBackend(responses, errors={first_user: "timeout"}),
        )
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.error.stage, "user_backend")
        self.assertEqual(len(result.metadata["request_audits"]), 2)
        self.assertEqual(result.metadata["request_audits"][-1]["status"], "failed")

    def test_fixed_assistant_backend_failure_is_not_attributed_to_user(self) -> None:
        case = self.cases[0]
        adapter = TauUSIAdapter()
        responses = adapter.replay_responses(case, seed=20260812)
        first_assistant = next(key for key in responses if ":assistant:" in key)
        result = self.execute(
            case,
            adapter=adapter,
            backend=ReplayBackend(responses, errors={first_assistant: "timeout"}),
        )
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.error.stage, "assistant_backend")
        self.assertEqual(result.trace[-1].content["stage"], "assistant_backend")

    def test_latency_timeout_is_enforced_and_retried(self) -> None:
        case = self.cases[0]
        adapter = TauUSIAdapter()
        responses = dict(adapter.replay_responses(case, seed=20260812))
        first_user = next(key for key in responses if ":user:0" in key)
        timed = dict(responses[first_user])
        timed["latency_ms"] = 31_000
        responses[first_user] = timed
        result = self.execute(case, adapter=adapter, backend=ReplayBackend(responses))
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.error.stage, "user_backend")
        self.assertEqual(result.error.kind, "BackendTimeoutError")
        self.assertEqual(len(result.metadata["request_audits"]), 2)

    def test_transient_failure_recovers_within_retry_budget(self) -> None:
        case = self.cases[0]
        adapter = TauUSIAdapter()
        responses = adapter.replay_responses(case, seed=20260812)
        first_user = next(key for key in responses if ":user:0" in key)
        backend = FailOnceReplayBackend(responses, first_user)
        result = self.execute(case, adapter=adapter, backend=backend)
        self.assertEqual(result.status.value, "completed")
        self.assertEqual(backend.calls[first_user], 2)
        first_audits = [item for item in result.metadata["request_audits"] if item["request_id"] == first_user]
        self.assertEqual([item["status"] for item in first_audits], ["retry", "completed"])

    def test_user_turn_limit_is_scored(self) -> None:
        case = self.cases[0]
        baseline = TauUSIAdapter()
        runtime = baseline.runtime_for_case(case)
        limited_user = TauUSIAdapter(
            runtime_provenance=replace(runtime, max_user_turns=1)
        )
        user_result = self.execute(case, adapter=limited_user)
        self.assertEqual(user_result.status.value, "completed")
        self.assertIsNone(user_result.error)
        self.assertTrue(user_result.metadata["episode_complete"])
        self.assertEqual(user_result.prediction["terminal_reason"], "user_turn_limit")
        self.assertEqual(
            user_result.metadata["capability_termination"]["kind"],
            "stop_instruction_not_followed",
        )
        self.assertEqual(user_result.metadata["tau_usi"]["environment_reward"], 1.0)
        self.assertEqual(user_result.metadata["tau_usi"]["stop_compliance"], 0.0)
        self.assertEqual(user_result.metadata["tau_usi"]["strict_episode_success"], 0.0)
        self.assertEqual(user_result.metrics[0].value, 1.0)
        self.assertEqual(user_result.metrics[1].name, "tau_usi.stop_compliance")
        self.assertEqual(user_result.metrics[1].value, 0.0)
        self.assertEqual(
            user_result.metrics[2].name,
            "tau_usi.diagnostic.strict_episode_success",
        )
        self.assertEqual(user_result.metrics[2].value, 0.0)
        self.assertEqual(user_result.trace[-1].kind, "capability_limit")

    def test_scored_turn_limit_does_not_invalidate_distribution_metrics(self) -> None:
        baseline = TauUSIAdapter()
        limited = TauUSIAdapter(
            runtime_provenance=replace(
                baseline.runtime_for_case(self.cases[0]),
                max_user_turns=1,
            )
        )
        first = self.execute(self.cases[0], adapter=limited)
        second = self.execute(self.cases[1], adapter=limited)
        metrics = limited.aggregate([first, second])
        self.assertTrue(metrics["tau_usi.suite_complete"].value)
        self.assertEqual(metrics["tau_usi.case_failure_count"].value, 0)
        self.assertIsNotNone(metrics["tau_usi.usi"].value)
        self.assertEqual(metrics["tau_usi.stop_compliance_rate"].value, 0.0)
        self.assertEqual(
            metrics["tau_usi.diagnostic.strict_episode_success_rate"].value,
            0.0,
        )

    def test_accounted_tokens_are_diagnostic_not_an_episode_limit(self) -> None:
        case = self.cases[0]
        adapter = TauUSIAdapter()
        responses = {
            key: {**value, "usage": {"total_tokens": 100_000}}
            for key, value in adapter.replay_responses(case, seed=20260812).items()
        }
        result = self.execute(case, adapter=adapter, backend=ReplayBackend(responses))
        self.assertEqual(result.status.value, "completed")
        self.assertGreater(result.metadata["accounted_tokens"], 65_536)

    def test_invalid_target_output_remains_in_complete_suite_as_zero_capability_outcome(self) -> None:
        adapter = TauUSIAdapter()
        good = self.execute(self.cases[1], adapter=adapter)
        responses = dict(adapter.replay_responses(self.cases[0], seed=20260812))
        first_user = next(key for key in responses if ":user:0" in key)
        responses[first_user] = "   "
        failed = self.execute(self.cases[0], adapter=adapter, backend=ReplayBackend(responses))
        metrics = adapter.aggregate([failed, good])
        self.assertTrue(metrics["tau_usi.suite_complete"].value)
        self.assertIsNotNone(metrics["tau_usi.usi"].value)
        self.assertEqual(metrics["tau_usi.case_failure_count"].value, 0)

    def test_resume_identity_allows_assistant_model_fallback_but_rejects_policy_change(self) -> None:
        case = self.cases[0]
        adapter = TauUSIAdapter()
        scoring = adapter.provenance_for_case(case)
        runtime = adapter.runtime_for_case(case)

        def identity(assistant_identity):
            return RunIdentityInput(
                framework_version="test",
                benchmark_id="tau_usi",
                source_revision=case.source_revision,
                split=case.split,
                profile="offline_smoke",
                sample_manifest_digest="sample",
                seed=20260812,
                backend="replay",
                model="candidate-user",
                decoding={"adapter_controlled": True},
                prompt_revision=adapter.prompt_revision,
                scorer_revision=adapter.scorer_revision,
                judge=scoring.to_dict(),
                environment=adapter.environment_identity_for_case(case),
                assistant_or_partner=assistant_identity,
            )

        first_identity = identity(
            support_protocol_identity(adapter.assistant_or_partner_identity_for_case(case))
        )
        changed_runtime = replace(runtime, fixed_assistant_revision="fixture-agent-v2")
        changed = TauUSIAdapter(runtime_provenance=changed_runtime)
        second_identity = identity(
            support_protocol_identity(changed.assistant_or_partner_identity_for_case(case))
        )
        changed_policy_runtime = replace(runtime, assistant_policy_revision="fixture-policy-v2")
        changed_policy = TauUSIAdapter(runtime_provenance=changed_policy_runtime)
        third_identity = identity(
            support_protocol_identity(changed_policy.assistant_or_partner_identity_for_case(case))
        )
        first_manifest = RunManifest.create(
            first_identity,
            catalog_revision="test",
            result_label="fixture",
            requested_case_count=2,
            selected_case_count=2,
            selected_group_count=2,
        )
        second_manifest = RunManifest.create(
            second_identity,
            catalog_revision="test",
            result_label="fixture",
            requested_case_count=2,
            selected_case_count=2,
            selected_group_count=2,
        )
        third_manifest = RunManifest.create(
            third_identity,
            catalog_revision="test",
            result_label="fixture",
            requested_case_count=2,
            selected_case_count=2,
            selected_group_count=2,
        )
        self.assertEqual(first_manifest.run_id, second_manifest.run_id)
        self.assertNotEqual(first_manifest.run_id, third_manifest.run_id)
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            store.initialize(first_manifest)
            store.initialize(second_manifest)
            with self.assertRaises(ResumeConflictError):
                store.initialize(third_manifest)


if __name__ == "__main__":
    unittest.main()
