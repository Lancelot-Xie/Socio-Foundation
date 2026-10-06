import json
import threading
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.agentsense import AgentSenseAdapter, AgentSenseScorer, parse_information_choice
from sim_eval.benchmarks.sotopia import SotopiaAdapter, SotopiaScorer
from sim_eval.contracts import CaseResult, MetricValue, ResultStatus
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.environments.social import JudgeProvenance
from sim_eval.errors import ConfigurationError, ParseError


ROOT = Path(__file__).resolve().parents[1]


def metric(result, name):
    return next(item for item in result.metrics if item.name == name)


class SocialAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixtures = load_fixture_suite(ROOT / "tests" / "fixtures")
        cls.sotopia_case = fixtures["sotopia"][0][0]
        cls.agentsense_case = fixtures["agentsense"][0][0]

    def test_sotopia_replay_executes_all_dimensions_with_provenance(self) -> None:
        adapter = SotopiaAdapter()
        agent_ids = [str(item["id"]) for item in self.sotopia_case.input_data["agents"]]
        case = replace(
            self.sotopia_case,
            input_data={
                **self.sotopia_case.input_data,
                "evaluated_agent_id": agent_ids[0],
                "evaluated_role_position": "agent1",
                "partner_agent_id": agent_ids[1],
            },
        )
        seed = 11
        backend = ReplayBackend(responses=adapter.replay_responses(case, seed=seed))
        result = adapter.execute_case(
            case,
            backend=backend,
            run_id="run",
            seed=seed,
            model="fixture",
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertEqual(result.metadata["judge_status"], "available")
        self.assertEqual(metric(result, "sotopia.evaluated_agent.goal").value, 9)
        self.assertEqual(metric(result, "sotopia.evaluated_agent.secret.normalized").value, 1)
        self.assertEqual(metric(result, "sotopia.partner_agent.goal").value, 9)
        self.assertEqual(result.metadata["judge_provenance"]["rubric_revision"], "sotopia-dimensions-iclr2024-v1")

    def test_sotopia_formal_metric_selects_evaluated_agent_not_agent_mean(self) -> None:
        adapter = SotopiaAdapter()
        provenance = adapter.provenance_for_case(self.sotopia_case)
        self.assertIsNotNone(provenance)
        payload = json.loads(json.dumps(self.sotopia_case.metadata["replay"]["judge"]))
        payload["scores"]["a"]["goal"]["score"] = 10
        payload["scores"]["b"]["goal"]["score"] = 0
        metrics = SotopiaScorer().score_payload(
            payload,
            agent_ids=("a", "b"),
            provenance=provenance,
            evaluated_agent_id="b",
            partner_agent_id="a",
        )
        by_name = {item.name: item for item in metrics}
        self.assertEqual(by_name["sotopia.evaluated_agent.goal"].value, 0)
        self.assertEqual(by_name["sotopia.partner_agent.goal"].value, 10)
        self.assertEqual(by_name["sotopia.diagnostic.mean_across_agents.goal"].value, 5)
        self.assertTrue(by_name["sotopia.evaluated_agent.goal"].metadata["official_primary"])
        self.assertFalse(by_name["sotopia.diagnostic.mean_across_agents.goal"].metadata["official_primary"])

    def test_judge_failure_makes_metrics_unavailable_not_fabricated(self) -> None:
        adapter = SotopiaAdapter()
        seed = 12
        responses = dict(adapter.replay_responses(self.sotopia_case, seed=seed))
        responses[f"{self.sotopia_case.case_id}:judge"] = "not-json"
        result = adapter.execute_case(
            self.sotopia_case,
            backend=ReplayBackend(responses=responses),
            run_id="run",
            seed=seed,
            model="fixture",
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertEqual(result.metadata["judge_status"], "unavailable")
        self.assertTrue(all(item.value is None for item in result.metrics))
        self.assertTrue(result.metadata["judge_error"])

    def test_sotopia_aggregate_exposes_judge_score_availability(self) -> None:
        adapter = SotopiaAdapter()
        agent_ids = [str(item["id"]) for item in self.sotopia_case.input_data["agents"]]
        case = replace(
            self.sotopia_case,
            input_data={
                **self.sotopia_case.input_data,
                "evaluated_agent_id": agent_ids[0],
                "evaluated_role_position": "agent1",
                "partner_agent_id": agent_ids[1],
            },
        )
        good_seed = 11
        good = adapter.execute_case(
            case,
            backend=ReplayBackend(responses=adapter.replay_responses(case, seed=good_seed)),
            run_id="run",
            seed=good_seed,
            model="fixture",
        )
        bad_seed = 12
        bad_responses = dict(adapter.replay_responses(case, seed=bad_seed))
        bad_responses[f"{case.case_id}:judge"] = "not-json"
        bad = adapter.execute_case(
            case,
            backend=ReplayBackend(responses=bad_responses),
            run_id="run",
            seed=bad_seed,
            model="fixture",
        )

        aggregated = adapter.aggregate((good, bad))
        availability = aggregated["sotopia.judge_score_availability_rate"]
        self.assertEqual(availability.value, 0.5)
        self.assertEqual(availability.numerator, 1)
        self.assertEqual(availability.denominator, 2)
        self.assertEqual(aggregated["sotopia.judge_score_unavailable_count"].value, 1)
        self.assertEqual(aggregated["sotopia.judge_failure_count"].value, 1)
        goal = aggregated["sotopia.evaluated_agent.goal"]
        self.assertEqual(goal.denominator, 1)
        self.assertEqual(goal.metadata["unavailable_count"], 1)
        self.assertEqual(aggregated["sotopia.configuration_count"].value, 1)
        self.assertEqual(aggregated["sotopia.complete_role_pair_rate"].value, 0.0)
        configuration_goal = aggregated["sotopia.configuration_mean.goal"]
        self.assertIsNone(configuration_goal.value)
        self.assertEqual(configuration_goal.denominator, 0)
        self.assertEqual(
            configuration_goal.metadata["complete_value_pair_count"],
            0,
        )

    def test_missing_judge_configuration_is_explicitly_unavailable(self) -> None:
        adapter = SotopiaAdapter()
        replay = dict(self.sotopia_case.metadata["replay"])
        replay.pop("judge_provenance")
        case = replace(self.sotopia_case, metadata={**self.sotopia_case.metadata, "replay": replay})
        seed = 3
        result = adapter.execute_case(
            case,
            backend=ReplayBackend(responses=adapter.replay_responses(case, seed=seed)),
            run_id="run",
            seed=seed,
            model="fixture",
        )
        self.assertEqual(result.metadata["judge_status"], "unavailable")
        self.assertIn("judge configuration", result.metrics[0].metadata["reason"])

    def test_agentsense_active_goals_and_passive_information_are_separate(self) -> None:
        adapter = AgentSenseAdapter()
        seed = 19
        result = adapter.execute_case(
            self.agentsense_case,
            backend=ReplayBackend(responses=adapter.replay_responses(self.agentsense_case, seed=seed)),
            run_id="run",
            seed=seed,
            model="fixture",
        )
        self.assertEqual(metric(result, "agentsense.episode.self_goal_completion").value, 1)
        self.assertEqual(metric(result, "agentsense.episode.judge_average").value, 1)
        self.assertEqual(metric(result, "agentsense.episode.private_information_accuracy").value, 1)
        self.assertEqual(result.prediction["information"][0]["parse_status"], "parsed")
        self.assertEqual(result.metadata["judge_call_count"], 3)

    def test_agentsense_empty_actor_output_is_terminal_zero_score_capability_outcome(self) -> None:
        adapter = AgentSenseAdapter()
        seed = 19
        responses = dict(adapter.replay_responses(self.agentsense_case, seed=seed))
        first_actor = next(key for key in responses if ":actor:" in key)
        responses[first_actor] = "   "
        result = adapter.execute_case(
            self.agentsense_case,
            backend=ReplayBackend(responses=responses),
            run_id="run",
            seed=seed,
            model="fixture",
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["target_output_failure"]["stage"], "action_parse")
        self.assertEqual(metric(result, "agentsense.episode.judge_average").value, 0)
        self.assertEqual(metric(result, "agentsense.episode.private_information_accuracy").value, 0)

    def test_agentsense_official_goal_questions_use_per_interview_topology(self) -> None:
        serial_adapter = AgentSenseAdapter(judge_max_workers=1)
        adapter = AgentSenseAdapter(judge_max_workers=3)
        seed = 19
        questions = {
            "a": [
                {
                    "self": [{"obj": "a", "question": "Have you achieved goal A? Answer Yes or No."}],
                    "others": [{"obj": "b", "question": "Has a achieved goal A? Answer Yes or No."}],
                    "judge": [{"obj": "judge", "question": "Has a achieved goal A? Answer Yes or No."}],
                }
            ],
            "b": [
                {
                    "self": [{"obj": "b", "question": "Have you achieved goal B? Answer Yes or No."}],
                    "others": [{"obj": "a", "question": "Has b achieved goal B? Answer Yes or No."}],
                    "judge": [{"obj": "judge", "question": "Has b achieved goal B? Answer Yes or No."}],
                }
            ],
        }
        case = replace(
            self.agentsense_case,
            metadata={**self.agentsense_case.metadata, "goal_evaluation_questions": questions},
        )
        responses = dict(adapter.replay_responses(case, seed=seed))
        provenance = adapter.provenance_for_case(case)
        self.assertIsNotNone(provenance)
        answer = {"reasoning": "The transcript directly supports completion.", "answer": "Yes"}
        for target, other in (("a", "b"), ("b", "a")):
            responses[f"{case.case_id}:goal:{target}:0:self:0:{target}"] = json.dumps(answer)
            responses[f"{case.case_id}:goal:{target}:0:others:0:{other}"] = json.dumps(answer)
            for judge_id in provenance.logical_judge_ids:
                responses[f"{case.case_id}:goal:{target}:0:judge:0:{judge_id}"] = json.dumps(answer)
        serial_result = serial_adapter.execute_case(
            case,
            backend=ReplayBackend(responses=responses),
            run_id="run",
            seed=seed,
            model="fixture",
        )
        active_lock = threading.Lock()
        active_judges = 0
        max_active_judges = 0
        rendezvous = threading.Barrier(3)

        test_case = self

        class ConcurrentJudgeReplay(ReplayBackend):
            def generate(self, request):
                nonlocal active_judges, max_active_judges
                tracked = ":judge:" in str(request.request_id)
                test_case.assertEqual(request.temperature, 0.8 if tracked else 0.0)
                if tracked:
                    with active_lock:
                        active_judges += 1
                        max_active_judges = max(max_active_judges, active_judges)
                    try:
                        rendezvous.wait(timeout=2)
                        return super().generate(request)
                    finally:
                        with active_lock:
                            active_judges -= 1
                return super().generate(request)

        result = adapter.execute_case(
            case,
            backend=ConcurrentJudgeReplay(responses=responses),
            run_id="run",
            seed=seed,
            model="fixture",
        )
        self.assertEqual(result.metadata["judge_status"], "available")
        self.assertEqual(max_active_judges, 3)
        self.assertEqual(result.to_dict(), serial_result.to_dict())
        self.assertEqual(result.metadata["judge_call_count"], 6)
        self.assertEqual(
            result.metadata["goal_evaluation_call_counts"],
            {"self": 2, "others": 2, "judge": 6},
        )
        self.assertEqual(result.metadata["goal_evaluation_topology"], "official_per_goal_self_others_three_judges")
        self.assertEqual(metric(result, "agentsense.episode.judge_average").value, 1)

    def test_agentsense_three_agent_episode_executes_end_to_end(self) -> None:
        adapter = AgentSenseAdapter()
        agent_ids = ("a", "b", "c")
        judge_models = ("fixture-gpt4o", "fixture-qwen72b", "fixture-llama70b")
        goal_evaluations = {
            agent_id: [
                {
                    "goal_id": f"goal-{agent_id}",
                    "self": "Yes",
                    "others": ["Yes" for _ in agent_ids if _ != agent_id],
                    "judges": {judge: "Yes" for judge in judge_models},
                }
            ]
            for agent_id in agent_ids
        }
        replay = {
            "actions": [
                {"action_type": "speak", "content": "first contribution"},
                {"action_type": "speak", "content": "second contribution"},
                {"action_type": "speak", "content": "third contribution"},
                {"action_type": "speak", "content": "done"},
            ],
            "information_responses": ["A"],
            "judge_provenance": {
                "judge_models": list(judge_models),
                "judge_revisions": ["synthetic-v1"] * 3,
                "rubric_revision": "agentsense-goal-yes-no-v1",
                "source": "synthetic_fixture",
            },
            "judge": {"goal_evaluations": goal_evaluations},
        }
        case = replace(
            self.agentsense_case,
            input_data={
                **self.agentsense_case.input_data,
                "agents": [{"id": agent_id, "role": agent_id.upper()} for agent_id in agent_ids],
                "profiles": {agent_id: f"profile-{agent_id}" for agent_id in agent_ids},
                "goals": {agent_id: f"goal-{agent_id}" for agent_id in agent_ids},
                "private_information": {
                    agent_id: f"unique-secret-{agent_id}" for agent_id in agent_ids
                },
                "information_questions": [
                    {
                        "id": "info-c",
                        "agent_id": "c",
                        "question": "What is your private fact?",
                        "options": ["unique-secret-c", "not-c"],
                    }
                ],
                "max_turns": 5,
            },
            gold={"information_answers": {"info-c": 0}},
            metadata={**self.agentsense_case.metadata, "replay": replay, "profile_id": "three-agent"},
        )
        seed = 29
        result = adapter.execute_case(
            case,
            backend=ReplayBackend(responses=adapter.replay_responses(case, seed=seed)),
            run_id="run",
            seed=seed,
            model="fixture",
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertEqual(result.metadata["judge_status"], "available")
        self.assertEqual(result.metadata["judge_call_count"], 3)
        self.assertEqual(metric(result, "agentsense.episode.judge_average").value, 1)
        self.assertEqual(metric(result, "agentsense.episode.private_information_accuracy").value, 1)
        public_actors = [event.actor for event in result.trace if event.metadata.get("visibility") == "public_transcript"]
        self.assertTrue(all(left != right for left, right in zip(public_actors, public_actors[1:])))
        identity = adapter.environment_identity_for_case(case)
        self.assertEqual(identity["min_agent_count"], 2)
        self.assertEqual(identity["max_agent_count"], 5)

    def test_information_choice_parser_rejects_duplicates_and_no_answer(self) -> None:
        self.assertEqual(parse_information_choice("(A)", 4), 0)
        self.assertEqual(parse_information_choice("B.", 4), 1)
        for malformed in ("", "A or B", "answer A", "(E)"):
            with self.assertRaises(ParseError):
                parse_information_choice(malformed, 4)

    def test_agentsense_judge_requires_exact_provenance_keys(self) -> None:
        scorer = AgentSenseScorer()
        provenance = JudgeProvenance(
            judge_models=("j1", "j2", "j3"),
            judge_revisions=("r", "r", "r"),
            rubric_revision="rubric",
        )
        payload = {
            "goal_evaluations": {
                "a": [{"self": "Yes", "others": ["Yes"], "judges": {"j1": "Yes", "j2": "Yes"}}],
                "b": [{"self": "Yes", "others": ["Yes"], "judges": {"j1": "Yes", "j2": "Yes"}}],
            }
        }
        with self.assertRaisesRegex(ParseError, "judge keys"):
            scorer.score_goals(payload, agent_ids=("a", "b"), provenance=provenance)

    def test_agentsense_logical_judge_ids_can_share_one_physical_model(self) -> None:
        judge_ids = ("deepseek_judge_1", "deepseek_judge_2", "deepseek_judge_3")
        provenance = JudgeProvenance(
            judge_models=("example-support-model",) * 3,
            judge_revisions=("example-support-model",) * 3,
            judge_ids=judge_ids,
            rubric_revision="rubric",
        )
        adapter = AgentSenseAdapter(
            judge_provenance=provenance,
            judge_role_by_id=dict(zip(judge_ids, ("judge_1", "judge_2", "judge_3"))),
        )
        state = adapter.environment.reset(self.agentsense_case, seed=7)
        requests = [
            adapter.build_judge_request(
                self.agentsense_case,
                state,
                provenance,
                judge_model=model,
                judge_id=judge_id,
                judge_index=index,
                seed=7,
            )
            for index, (judge_id, model) in enumerate(zip(judge_ids, provenance.judge_models))
        ]
        self.assertEqual({request.model for request in requests}, {"example-support-model"})
        self.assertEqual(
            [request.metadata["route_role"] for request in requests],
            ["judge_1", "judge_2", "judge_3"],
        )
        self.assertEqual([request.metadata["judge_id"] for request in requests], list(judge_ids))

        agent_ids = tuple(state.agents)
        payload = {
            "goal_evaluations": {
                agent_id: [
                    {
                        "self": "Yes",
                        "others": ["Yes"],
                        "judges": {
                            judge_ids[0]: "Yes",
                            judge_ids[1]: "No",
                            judge_ids[2]: "Yes",
                        },
                    }
                ]
                for agent_id in agent_ids
            }
        }
        metrics = AgentSenseScorer().score_goals(
            payload,
            agent_ids=agent_ids,
            provenance=provenance,
        )
        by_name = {item.name: item for item in metrics}
        self.assertAlmostEqual(by_name["agentsense.episode.judge_average"].value, 2 / 3)
        self.assertEqual(by_name["agentsense.episode.judge_majority"].value, 1.0)
        self.assertEqual(provenance.to_dict()["judge_ids"], list(judge_ids))

    def test_judge_provenance_rejects_duplicate_or_misaligned_logical_ids(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "must match"):
            JudgeProvenance(
                judge_models=("same",) * 3,
                judge_revisions=("r",) * 3,
                judge_ids=("j1", "j2"),
                rubric_revision="rubric",
            )
        with self.assertRaisesRegex(ConfigurationError, "distinct"):
            JudgeProvenance(
                judge_models=("same",) * 3,
                judge_revisions=("r",) * 3,
                judge_ids=("j1", "j1", "j3"),
                rubric_revision="rubric",
            )

    def test_profile_sensitivity_uses_sample_std_across_profiles_per_judge(self) -> None:
        adapter = AgentSenseAdapter()
        results = []
        for index, value in enumerate((0.0, 1.0)):
            judge_values = {
                "j1": value,
                "j2": 1.0 - value,
                "j3": value,
            }
            results.append(
                CaseResult(
                    run_id="run",
                    benchmark_id="agentsense",
                    case_id=f"case-{index}",
                    group_id="same-template",
                    repetition=0,
                    status=ResultStatus.COMPLETED,
                    metrics=[
                        MetricValue(f"agentsense.episode.judge.{judge}", judge_value)
                        for judge, judge_value in judge_values.items()
                    ],
                    metadata={"profile_id": f"profile-{index}"},
                )
            )
        metrics = adapter.aggregate(results)
        goal = metrics["agentsense.profile_sensitivity_index.goal"]
        self.assertAlmostEqual(goal.value, 70.71067811865476)
        self.assertEqual(goal.direction, "lower_is_better")
        self.assertEqual(goal.metadata["standard_deviation_ddof"], 1)
        self.assertEqual(goal.metadata["eligible_template_count"], 1)
        self.assertEqual(set(goal.metadata["per_judge_mean_sample_standard_deviation"]), {"j1", "j2", "j3"})

    def test_information_profile_sensitivity_uses_sample_not_population_std(self) -> None:
        adapter = AgentSenseAdapter()
        results = [
            CaseResult(
                run_id="run",
                benchmark_id="agentsense",
                case_id=f"case-{index}",
                group_id="same-template",
                repetition=0,
                status=ResultStatus.COMPLETED,
                metrics=[
                    MetricValue("agentsense.episode.private_information_accuracy", value)
                ],
                metadata={"profile_id": f"profile-{index}"},
            )
            for index, value in enumerate((0.0, 1.0))
        ]
        information = adapter.aggregate(results)[
            "agentsense.profile_sensitivity_index.information"
        ]
        self.assertAlmostEqual(information.value, 70.71067811865476)
        self.assertEqual(information.direction, "lower_is_better")
        self.assertEqual(information.metadata["standard_deviation_ddof"], 1)

    def test_judge_reasoning_is_evaluator_only(self) -> None:
        adapter = SotopiaAdapter()
        seed = 21
        result = adapter.execute_case(
            self.sotopia_case,
            backend=ReplayBackend(responses=adapter.replay_responses(self.sotopia_case, seed=seed)),
            run_id="run",
            seed=seed,
            model="fixture",
        )
        item = metric(result, "sotopia.agent.a.believability")
        self.assertEqual(item.metadata["reasoning_visibility"], "evaluator_only")


if __name__ == "__main__":
    unittest.main()
