import unittest

from sim_eval.contracts import (
    BenchmarkCase,
    CaseResult,
    ChatMessage,
    ErrorState,
    MetricValue,
    ModelRequest,
    ModelResponse,
    ResultStatus,
    RunIdentityInput,
    RunManifest,
)
from sim_eval.errors import ValidationError


def identity(**changes):
    values = {
        "framework_version": "0.1.0",
        "benchmark_id": "fantom",
        "source_revision": "abc123",
        "split": "test",
        "profile": "default",
        "sample_manifest_digest": "sample-digest",
        "seed": 7,
        "backend": "replay",
        "model": "fixture-model",
        "decoding": {"temperature": 0},
        "prompt_revision": "prompt-v1",
        "scorer_revision": "score-v1",
        "judge": {},
        "environment": {},
        "assistant_or_partner": {},
    }
    values.update(changes)
    return RunIdentityInput(**values)


class ContractTests(unittest.TestCase):
    def test_model_response_normalizes_only_invalid_unicode_surrogates(self) -> None:
        response = ModelResponse(
            text="normal 中文 😀 | pair:\ud83d\ude00 | lone:\udc50",
            raw={"nested": {"value": "bad:\udddf"}},
        )
        self.assertEqual(response.text, "normal 中文 😀 | pair:😀 | lone:�")
        self.assertEqual(response.raw["nested"]["value"], "bad:�")
        audit = response.raw["_sim_eval"]["unicode_sanitization"]
        self.assertEqual(audit["unpaired_surrogates_replaced"], 2)
        self.assertEqual(audit["surrogate_pairs_normalized"], 1)
        response.text.encode("utf-8")

    def test_model_request_fingerprint_is_order_stable(self) -> None:
        left = ModelRequest(
            messages=[ChatMessage("user", "choose")],
            model="fixture",
            metadata={"b": 2, "a": 1},
        )
        right = ModelRequest(
            messages=[ChatMessage("user", "choose")],
            model="fixture",
            metadata={"a": 1, "b": 2},
        )
        self.assertEqual(left.fingerprint, right.fingerprint)

    def test_model_request_rejects_noncanonical_system_messages(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must precede"):
            ModelRequest(
                messages=[ChatMessage("user", "payload"), ChatMessage("system", "late instruction")],
                model="fixture",
            )
        with self.assertRaisesRegex(ValidationError, "at most one"):
            ModelRequest(
                messages=[
                    ChatMessage("system", "identity"),
                    ChatMessage("system", "output contract"),
                    ChatMessage("user", "payload"),
                ],
                model="fixture",
            )

    def test_resume_identity_changes_for_protocol_relevant_fields(self) -> None:
        base = identity()
        self.assertEqual(base.run_id, identity().run_id)
        for changed in (
            identity(profile="canonical"),
            identity(model="different"),
            identity(prompt_revision="prompt-v2"),
            identity(scorer_revision="score-v2"),
            identity(judge={"rubric_revision": "judge-rubric-v2"}),
            identity(assistant_or_partner={"policy_revision": "fixed-assistant-policy-v2"}),
        ):
            self.assertNotEqual(base.run_id, changed.run_id)

    def test_run_manifest_exposes_required_identity_and_counts(self) -> None:
        manifest = RunManifest.create(
            identity(),
            catalog_revision="2026-08-12",
            result_label="indieval_research_sample",
            requested_case_count=457,
            selected_case_count=457,
            selected_group_count=4,
        )
        payload = manifest.to_dict()
        self.assertEqual(payload["identity"]["benchmark_id"], "fantom")
        self.assertEqual(payload["identity"]["prompt_revision"], "prompt-v1")
        self.assertEqual(payload["selected_case_count"], 457)

    def test_static_and_interactive_case_shapes_share_neutral_payload(self) -> None:
        case = BenchmarkCase(
            benchmark_id="tau_usi",
            case_id="task-1",
            group_id="task-1",
            split="test",
            source_revision="rev",
            input_data={"initial_state": {}, "user_goal": "change flight"},
            metadata={"max_turns": 12, "private_state_keys": ["user_goal"]},
        )
        self.assertEqual(case.input_data["user_goal"], "change flight")

    def test_failure_result_requires_structured_error(self) -> None:
        with self.assertRaises(ValidationError):
            CaseResult(
                run_id="run",
                benchmark_id="fantom",
                case_id="c1",
                group_id="g1",
                repetition=0,
                status=ResultStatus.FAILED,
            )
        result = CaseResult(
            run_id="run",
            benchmark_id="fantom",
            case_id="c1",
            group_id="g1",
            repetition=0,
            status=ResultStatus.FAILED,
            error=ErrorState(stage="backend", kind="timeout", message="timed out", retryable=True),
            metrics=[MetricValue("exact_match", 0, numerator=0, denominator=1)],
        )
        self.assertEqual(result.error.kind, "timeout")


if __name__ == "__main__":
    unittest.main()
