import json
import math
import statistics
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.mirrorbench import (
    MirrorBenchAdapter,
    hdd,
    mattr,
    mirror_tokenize,
    parse_gteval,
    parse_pi,
    parse_rnr,
    yules_k,
)
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.environments.dialogue import EVALUATED_USER_ROLE, FIXED_ASSISTANT_ROLE
from sim_eval.errors import ParseError


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


class MirrorBenchAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_fixture_suite(ROOT / "tests" / "fixtures")["mirrorbench"][0]

    @staticmethod
    def execute(case, adapter=None, backend=None):
        adapter = adapter or MirrorBenchAdapter()
        responses = adapter.replay_responses(case, seed=20260812)
        backend = backend or ReplayBackend(responses)
        return adapter.execute_case(
            case,
            backend=backend,
            run_id="mirror-test-run",
            seed=20260812,
            model="candidate-user",
        )

    def test_lexical_metric_golden_values(self) -> None:
        self.assertAlmostEqual(mattr([1, 1, 2], window=2), 0.75)
        self.assertAlmostEqual(hdd(["a", "a", "b"], sample_size=2), 5 / 6)
        self.assertAlmostEqual(yules_k([1, 1, 2]), 10_000 * 2 / 9)
        self.assertEqual(mattr([], window=50), 0.0)
        self.assertEqual(hdd([], sample_size=42), 0.0)
        self.assertEqual(yules_k([]), 0.0)

    def test_tiktoken_policy_encodes_candidate_special_literals_as_ordinary_text(self) -> None:
        class FakeEncoding:
            def __init__(self) -> None:
                self.seen = []

            def encode_ordinary(self, text):
                self.seen.append(text)
                return [11, 22, 33]

            def encode(self, text):
                raise AssertionError("the guarded special-token path must not be used")

        encoding = FakeEncoding()
        module = SimpleNamespace(encoding_for_model=lambda _model: encoding)
        with patch(
            "sim_eval.benchmarks.mirrorbench.importlib.import_module",
            return_value=module,
        ):
            tokens = mirror_tokenize(
                "visible <|endoftext|> text",
                policy="tiktoken_model",
                model="gpt-4o",
            )

        self.assertEqual(tokens, [11, 22, 33])
        self.assertEqual(encoding.seen, ["visible <|endoftext|> text"])

    def test_replay_preserves_distinct_lexical_and_judge_metrics(self) -> None:
        result = self.execute(self.cases[0])
        self.assertEqual(result.status.value, "completed")
        self.assertTrue(result.metadata["episode_complete"])
        self.assertEqual(result.metadata["evaluated_model_role"], EVALUATED_USER_ROLE)
        self.assertIsNotNone(metric(result, "mirrorbench.lexical.mattr.proxy_raw").value)
        self.assertIsNotNone(metric(result, "mirrorbench.lexical.hdd.proxy_raw").value)
        self.assertIsNotNone(metric(result, "mirrorbench.lexical.yules_k.proxy_raw").value)
        self.assertEqual(metric(result, "mirrorbench.judge.gteval").value, 0.82)
        self.assertIsNotNone(metric(result, "mirrorbench.judge.pi").value)
        self.assertEqual(metric(result, "mirrorbench.judge.rnr").value, 1.0)
        self.assertEqual(metric(result, "mirrorbench.judge.gteval.hh_control").value, 1.0)
        self.assertEqual(metric(result, "mirrorbench.judge.pi.hh_control").value, 0.5)
        self.assertEqual(len(result.metadata["mirrorbench"]["judge_records"]["pi"]["main"]["samples"]), 3)
        self.assertTrue(all(sample["order"] in {"PH", "HP"} for sample in result.metadata["mirrorbench"]["judge_records"]["pi"]["main"]["samples"]))

    def test_reference_is_hidden_from_user_but_visible_to_fixed_assistant(self) -> None:
        case = self.cases[0]
        adapter = MirrorBenchAdapter()
        responses = adapter.replay_responses(case, seed=20260812)
        backend = CapturingReplayBackend(responses)
        result = adapter.execute_case(case, backend=backend, run_id="run", seed=20260812, model="candidate")
        self.assertEqual(result.status.value, "completed")
        user_requests = [request for request in backend.requests if request.metadata.get("actor") == EVALUATED_USER_ROLE]
        assistant_requests = [request for request in backend.requests if request.metadata.get("actor") == FIXED_ASSISTANT_ROLE]
        reference_line = "I need somewhere peaceful to read this afternoon."
        user_text = "\n".join(message.content for request in user_requests for message in request.messages)
        assistant_text = "\n".join(message.content for request in assistant_requests for message in request.messages)
        self.assertNotIn(reference_line, user_text)
        self.assertIn(reference_line, assistant_text)
        self.assertTrue(all(request.model == "candidate" for request in user_requests))
        self.assertTrue(all(request.model == "fixture-fixed-gpt4o-analogue" for request in assistant_requests))
        for request in (*user_requests, *assistant_requests):
            self.assertEqual(request.messages[0].role, "system")
            self.assertEqual(sum(m.role == "system" for m in request.messages), 1)
            self.assertIsNone(request.response_format)
        self.assertNotIn('"action":"message|refuse"', user_requests[0].messages[0].content)
        self.assertNotIn('"action":"message|refuse"', assistant_requests[0].messages[0].content)
        self.assertIn("simulating a real human user", user_requests[0].messages[0].content)
        self.assertIn("Match the length, tone, and specificity", user_requests[0].messages[0].content)
        self.assertIn("Do not include any prefixes like 'User:'", user_requests[0].messages[0].content)
        self.assertIn("real conversation for context", assistant_requests[0].messages[0].content)
        self.assertFalse(result.metadata["reference_visible_to_evaluated_user"])

    def test_malformed_one_judge_is_unavailable_without_erasing_other_metrics(self) -> None:
        case = self.cases[0]
        adapter = MirrorBenchAdapter()
        responses = dict(adapter.replay_responses(case, seed=20260812))
        responses[f"{case.case_id}:judge:gteval:main:0"] = json.dumps({"score": 0.9})
        responses[f"{case.case_id}:judge:gteval:main:0:judge_contract_retry:1"] = json.dumps({"score": 0.9})
        responses[f"{case.case_id}:judge:gteval:main:0:judge_contract_retry:2"] = json.dumps({"score": 0.9})
        result = adapter.execute_case(case, backend=ReplayBackend(responses), run_id="run", seed=20260812, model="candidate")
        self.assertEqual(result.status.value, "completed")
        self.assertIsNone(metric(result, "mirrorbench.judge.gteval").value)
        self.assertEqual(metric(result, "mirrorbench.judge.gteval").metadata["error"]["kind"], "ParseError")
        self.assertIsNotNone(metric(result, "mirrorbench.judge.pi").value)
        self.assertIsNotNone(metric(result, "mirrorbench.judge.rnr").value)
        self.assertIsNotNone(metric(result, "mirrorbench.lexical.mattr.proxy_raw").value)

    def test_absent_judge_configuration_yields_nullable_metrics(self) -> None:
        case = self.cases[0]
        base = MirrorBenchAdapter()
        no_judge = replace(base.provenance_for_case(case), judge_model=None, judge_revision=None)
        adapter = MirrorBenchAdapter(scoring_provenance=no_judge)
        result = self.execute(case, adapter=adapter)
        self.assertEqual(result.status.value, "completed")
        for name in ("gteval", "pi", "rnr"):
            item = metric(result, f"mirrorbench.judge.{name}")
            self.assertIsNone(item.value)
            self.assertEqual(item.metadata["availability"], "unavailable")
        self.assertIsNotNone(metric(result, "mirrorbench.lexical.mattr.proxy_raw").value)

    def test_target_capability_errors_score_zero_but_backend_timeout_fails(self) -> None:
        case = self.cases[0]
        adapter = MirrorBenchAdapter()
        baseline = dict(adapter.replay_responses(case, seed=20260812))
        first_user = f"{case.case_id}:user:0"

        empty = dict(baseline)
        empty[first_user] = "   "
        result = adapter.execute_case(case, backend=ReplayBackend(empty), run_id="run", seed=20260812, model="candidate")
        self.assertEqual(result.status.value, "completed")
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["target_output_failure"]["kind"], "empty_turn")
        self.assertEqual(metric(result, "mirrorbench.judge.gteval").value, 0.0)

        timeout = dict(baseline)
        timeout[first_user] = {"text": baseline[first_user]["text"], "latency_ms": 31_000}
        result = adapter.execute_case(case, backend=ReplayBackend(timeout), run_id="run", seed=20260812, model="candidate")
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.error.stage, "user_backend")
        self.assertEqual(result.error.kind, "BackendTimeoutError")

        constrained = MirrorBenchAdapter(runtime_provenance=replace(adapter.runtime_for_case(case), max_user_turns=1))
        result = constrained.execute_case(case, backend=ReplayBackend(baseline), run_id="run", seed=20260812, model="candidate")
        self.assertEqual(result.status.value, "completed")
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["target_output_failure"]["stage"], "user_turn_limit")

    def test_judge_response_schemas_reject_malformed_values(self) -> None:
        self.assertEqual(parse_gteval('{"reasoning":"ok","score":0.25}')["score"], 0.25)
        self.assertEqual(parse_pi('{"reasoning":"same","verdict":"Tie"}')["verdict"], "TIE")
        self.assertEqual(parse_rnr('{"reasoning":"natural","verdict":"yes"}')["score"], 1.0)
        for malformed, parser in (
            ('{"reasoning":"","score":0.5}', parse_gteval),
            ('{"reasoning":"x","verdict":"C"}', parse_pi),
            ('{"reasoning":"x","verdict":"MAYBE"}', parse_rnr),
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ParseError):
                    parser(malformed)

    def test_ci_retains_student_t_finite_sample_correction(self) -> None:
        # Independent reference quantiles: integer-df Student-t CDF integrated
        # after u=atan(t/sqrt(df)), then inverted by bisection at p=.975.
        references = {32: 2.039513446396, 50: 2.009575237129,
                      200: 1.971956544252, 795: 1.962956213849}
        for count, critical in references.items():
            with self.subTest(count=count):
                values = list(range(count))
                mean, stdev, half_width = MirrorBenchAdapter._mean_stdev_ci(values)
                self.assertEqual(mean, statistics.mean(values))
                self.assertEqual(stdev, statistics.stdev(values))
                inferred = half_width * math.sqrt(count) / stdev
                self.assertAlmostEqual(inferred, critical, delta=3e-8)
                self.assertGreater(inferred, 1.96)

    def test_ci_empty_singleton_constant_and_small_samples(self) -> None:
        calculate = MirrorBenchAdapter._mean_stdev_ci
        self.assertEqual(calculate([]), (None, None, None))
        self.assertEqual(calculate([0.4]), (0.4, None, None))
        self.assertEqual(calculate([0.4, 0.4]), (0.4, 0.0, 0.0))
        self.assertAlmostEqual(calculate([0.0, 2.0])[2], 12.706)
        self.assertEqual(calculate([0.0, 2.0])[0], 1.0)

    def test_singleton_keeps_scores_but_does_not_claim_zero_uncertainty(self) -> None:
        result = self.execute(self.cases[0])
        aggregate = MirrorBenchAdapter().aggregate([result])
        pi = aggregate["mirrorbench.judge.pi"]
        self.assertEqual(pi.value, metric(result, "mirrorbench.judge.pi").value)
        self.assertEqual(aggregate["mirrorbench.judge.pi_deviation"].value, pi.value - 0.5)
        for name in ("mirrorbench.judge.pi", "mirrorbench.judge.pi_deviation",
                     *(f"mirrorbench.lexical.{item}.z_score_mean" for item in ("mattr", "hdd", "yules_k"))):
            self.assertIsNone(aggregate[name].uncertainty["ci95_half_width"])
            self.assertIsNone(aggregate[name].uncertainty["sample_stdev"])
            self.assertEqual(aggregate[name].metadata["confidence_interval_availability"], "unavailable_insufficient_samples")

    def test_human_anchored_z_aggregation_records_baseline_and_ci(self) -> None:
        results = [self.execute(case) for case in self.cases]
        aggregate = MirrorBenchAdapter().aggregate(results)
        for name in ("mattr", "hdd", "yules_k"):
            item = aggregate[f"mirrorbench.lexical.{name}.z_score_mean"]
            self.assertIsNotNone(item.value)
            self.assertEqual(item.direction, "closer_to_zero")
            self.assertEqual(item.metadata["valid_episode_count"], 2)
            self.assertIn("ci95_half_width", item.uncertainty)
            self.assertEqual(
                aggregate[f"mirrorbench.lexical.{name}.proxy_raw"].direction,
                "descriptive",
            )
            self.assertEqual(
                aggregate[f"mirrorbench.lexical.{name}.human_raw"].direction,
                "descriptive",
            )
        raw_pi = aggregate["mirrorbench.judge.pi"]
        pi_deviation = aggregate["mirrorbench.judge.pi_deviation"]
        self.assertAlmostEqual(pi_deviation.value, raw_pi.value - 0.5)
        self.assertEqual(pi_deviation.direction, "higher_is_better")
        self.assertEqual(pi_deviation.metadata["neutral_reference"], 0.0)
        self.assertEqual(
            pi_deviation.uncertainty["ci95_half_width"],
            raw_pi.uncertainty["ci95_half_width"],
        )
        self.assertEqual(aggregate["mirrorbench.judge.gteval.availability_rate"].value, 1.0)


if __name__ == "__main__":
    unittest.main()
