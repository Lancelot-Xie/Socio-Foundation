import unittest
from dataclasses import replace

from sim_eval.contracts import CaseResult, ResultStatus
from sim_eval.execution_provenance import (
    attach_execution_provenance,
    environment_protocol_identity,
    role_execution_identity,
    summarize_evaluation_warnings,
    summarize_support_role_provenance,
    support_protocol_identity,
)


def result(case_id: str) -> CaseResult:
    return CaseResult(
        run_id="run",
        benchmark_id="mirrorbench",
        case_id=case_id,
        group_id=case_id,
        repetition=0,
        status=ResultStatus.COMPLETED,
    )


class ExecutionProvenanceTests(unittest.TestCase):
    def test_support_model_routing_is_mutable_but_protocol_is_not(self) -> None:
        primary = {
            "model": "assistant-primary",
            "model_revision": "primary-rev",
            "backend": "openai_compatible",
            "base_url": "https://primary.example/v1",
            "policy_revision": "assistant-policy-v1",
            "temperature": 0.0,
        }
        fallback = {
            **primary,
            "model": "assistant-fallback",
            "model_revision": "fallback-rev",
            "backend": "vllm",
            "base_url": "http://fallback.internal/v1",
            "fallback_reason": "primary model rejected the request",
        }
        self.assertEqual(support_protocol_identity(primary), support_protocol_identity(fallback))
        self.assertNotEqual(
            support_protocol_identity(primary),
            support_protocol_identity({**fallback, "policy_revision": "assistant-policy-v2"}),
        )
        self.assertNotEqual(
            support_protocol_identity(primary),
            support_protocol_identity({**fallback, "temperature": 0.5}),
        )

    def test_timeout_and_retry_are_recovery_knobs_not_environment_identity(self) -> None:
        primary = {"revision": "env-v1", "max_turns": 12, "request_timeout_seconds": 30, "max_retries": 1}
        recovery = {**primary, "request_timeout_seconds": 180, "max_retries": 5}
        self.assertEqual(environment_protocol_identity(primary), environment_protocol_identity(recovery))
        self.assertNotEqual(
            environment_protocol_identity(primary),
            environment_protocol_identity({**recovery, "max_turns": 20}),
        )

    def test_attempt_records_and_summary_expose_mixed_support_models(self) -> None:
        primary = role_execution_identity(
            {
                "backend": "openai_compatible",
                "base_url": "https://primary.example/v1",
                "model": "assistant-primary",
                "model_revision": "primary-rev",
                "generation": {"temperature": 0.0},
            }
        )
        fallback = role_execution_identity(
            {
                "backend": "vllm",
                "base_url": "http://fallback.internal/v1",
                "model": "assistant-fallback",
                "model_revision": "fallback-rev",
                "generation": {"temperature": 0.0},
                "fallback_reason": "primary safety filter blocked the case",
            }
        )
        results = [
            attach_execution_provenance(
                result("case-1"),
                evaluated_role="evaluated_user",
                evaluated_role_identity={"model": "candidate", "model_revision": "candidate-rev"},
                support_roles={"fixed_assistant": primary},
            ),
            attach_execution_provenance(
                result("case-2"),
                evaluated_role="evaluated_user",
                evaluated_role_identity={"model": "candidate", "model_revision": "candidate-rev"},
                support_roles={"fixed_assistant": fallback},
            ),
        ]
        summary = summarize_support_role_provenance(results)
        self.assertTrue(summary["mixed_models"])
        self.assertTrue(summary["roles"]["fixed_assistant"]["fallback_recorded"])
        self.assertEqual(len(summary["roles"]["fixed_assistant"]["models"]), 2)
        self.assertEqual(
            results[1].metadata["execution_provenance"]["support_roles"]["fixed_assistant"]["fallback_reason"],
            "primary safety filter blocked the case",
        )

    def test_safety_fallback_api_call_marks_effective_model_mix(self) -> None:
        primary = role_execution_identity(
            {
                "backend": "chat_completions",
                "profile": "relay",
                "base_url": "https://relay.example/v1",
                "model": "judge-primary",
                "model_revision": "primary-rev",
                "generation": {"temperature": 0.0},
            }
        )
        recorded = attach_execution_provenance(
            result("case-safety"),
            evaluated_role="evaluated_user",
            evaluated_role_identity={"model": "candidate", "model_revision": "candidate-rev"},
            support_roles={"judge": primary},
            api_calls=(
                {
                    "route_role": "judge",
                    "protocol": "chat_completions",
                    "safety_fallback": {
                        "used": True,
                        "fallback": {
                            "model": "example-support-model",
                            "model_revision": "example-support-model",
                        },
                    },
                },
            ),
        )
        summary = summarize_support_role_provenance([recorded])
        self.assertEqual(summary["safety_fallback_call_count"], 1)
        self.assertTrue(summary["mixed_models"])
        self.assertEqual(
            summary["roles"]["judge"]["effective_fallback_models"][0]["model"],
            "example-support-model",
        )

    def test_nonfatal_nested_judge_error_is_promoted(self) -> None:
        recorded = replace(
            result("case-warning"),
            metadata={
                "mirrorbench": {
                    "judge_records": {
                        "gteval": {
                            "main": {
                                "status": "unavailable",
                                "error": {"kind": "ParseError", "message": "score was outside [0,1]"},
                            }
                        }
                    }
                }
            },
        )
        warnings = summarize_evaluation_warnings([recorded])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["case_id"], "case-warning")
        self.assertEqual(warnings[0]["kind"], "ParseError")
        self.assertIn("judge_records", warnings[0]["path"])


if __name__ == "__main__":
    unittest.main()
