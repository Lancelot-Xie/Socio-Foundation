import unittest
from dataclasses import replace
from pathlib import Path
from typing import Optional

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.agentsense import AgentSenseAdapter
from sim_eval.benchmarks.sotopia import SOTOPIA_DIMENSIONS, SotopiaAdapter, normalize_sotopia_dimension
from sim_eval.contracts import ResultStatus
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.environments.social import RoleIsolatedSocialEnvironment, SocialAction, parse_social_action
from sim_eval.errors import BackendTokenLimitError, ParseError, ValidationError


ROOT = Path(__file__).resolve().parents[1]


class SocialEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixtures = load_fixture_suite(ROOT / "tests" / "fixtures")
        cls.sotopia_case = fixtures["sotopia"][0][0]
        cls.agentsense_case = fixtures["agentsense"][0][0]

    def test_each_role_observation_excludes_other_private_information(self) -> None:
        adapter = SotopiaAdapter()
        state = adapter.environment.reset(self.sotopia_case, seed=4)
        secrets = self.sotopia_case.input_data["private_information"]
        for agent_id in state.agents:
            state.next_actor_index = state.turn_order.index(agent_id)
            text = "\n".join(message.content for message in adapter.environment.observation(state, actor=agent_id))
            self.assertIn(secrets[agent_id], text)
            other = next(value for key, value in secrets.items() if key != agent_id)
            self.assertNotIn(other, text)

    def _agentsense_case_with_agent_count(
        self, count: int, *, max_turns: Optional[int] = None
    ):
        agent_ids = [chr(ord("a") + index) for index in range(count)]
        return replace(
            self.agentsense_case,
            input_data={
                **self.agentsense_case.input_data,
                "agents": [{"id": agent_id, "role": agent_id.upper()} for agent_id in agent_ids],
                "profiles": {agent_id: f"profile-{agent_id}" for agent_id in agent_ids},
                "goals": {agent_id: f"goal-{agent_id}" for agent_id in agent_ids},
                "private_information": {
                    agent_id: f"unique-secret-{agent_id}" for agent_id in agent_ids
                },
                "information_questions": [],
                "max_turns": max_turns if max_turns is not None else count * 2,
            },
            gold={"information_answers": {}},
        )

    def test_agent_count_policy_is_explicit_and_benchmark_scoped(self) -> None:
        three_agent_case = self._agentsense_case_with_agent_count(3)
        default_environment = RoleIsolatedSocialEnvironment(
            randomize_turn_order=False,
            default_max_turns=3,
        )
        with self.assertRaisesRegex(ValidationError, "exactly 2 agents"):
            default_environment.reset(three_agent_case, seed=1)

        adapter = AgentSenseAdapter()
        state = adapter.environment.reset(three_agent_case, seed=1)
        self.assertEqual(len(state.agents), 3)
        self.assertEqual(adapter.environment.min_agent_count, 2)
        self.assertEqual(adapter.environment.max_agent_count, 5)

        six_agent_case = self._agentsense_case_with_agent_count(6)
        with self.assertRaisesRegex(ValidationError, "between two and five agents"):
            adapter.validate_case(six_agent_case)
        with self.assertRaisesRegex(ValidationError, "between 2 and 5 agents"):
            adapter.environment.reset(six_agent_case, seed=1)

    def test_sotopia_position_contract_is_explicit(self) -> None:
        adapter = SotopiaAdapter()
        agents = self.sotopia_case.input_data["agents"]
        positioned = replace(
            self.sotopia_case,
            input_data={
                **self.sotopia_case.input_data,
                "evaluated_agent_id": agents[0]["id"],
                "evaluated_role_position": "agent1",
                "partner_agent_id": agents[1]["id"],
            },
        )
        adapter.validate_case(positioned)
        identity = adapter.assistant_or_partner_identity_for_case(positioned)
        self.assertTrue(identity["position_designated"])
        self.assertEqual(identity["evaluated_agent_id"], agents[0]["id"])
        self.assertEqual(identity["partner_agent_id"], agents[1]["id"])

        incomplete = replace(
            positioned,
            input_data={
                key: value
                for key, value in positioned.input_data.items()
                if key != "partner_agent_id"
            },
        )
        with self.assertRaisesRegex(ValidationError, "require evaluated_agent_id"):
            adapter.validate_case(incomplete)

    def test_multi_agent_private_information_is_role_isolated(self) -> None:
        adapter = AgentSenseAdapter()
        case = self._agentsense_case_with_agent_count(5)
        state = adapter.environment.reset(case, seed=7)
        secrets = case.input_data["private_information"]
        for agent_id in state.agents:
            state.next_actor_index = state.turn_order.index(agent_id)
            text = "\n".join(
                message.content for message in adapter.environment.observation(state, actor=agent_id)
            )
            self.assertIn(secrets[agent_id], text)
            for other_id, secret in secrets.items():
                if other_id != agent_id:
                    self.assertNotIn(secret, text)

    def test_agentsense_scheduler_is_seeded_random_each_turn_without_immediate_repeat(self) -> None:
        adapter = AgentSenseAdapter()
        case = self._agentsense_case_with_agent_count(5, max_turns=10)
        schedules = []
        for _ in range(2):
            state = adapter.environment.reset(case, seed=23)
            actors = []
            while not state.terminal:
                actor = state.next_actor
                actors.append(actor)
                adapter.environment.apply(
                    state,
                    actor=actor,
                    action=SocialAction("speak", f"turn-{len(actors)}"),
                )
            schedules.append(actors)
            self.assertEqual(len(actors), 10)
            self.assertTrue(all(left != right for left, right in zip(actors, actors[1:])))
            self.assertEqual(state.terminal_reason, "max_turns")
        self.assertEqual(schedules[0], schedules[1])
        self.assertNotEqual(set(schedules[0][:5]), set(state.agents))
        self.assertEqual(adapter.environment.identity()["turn_scheduler"], "random_no_repeat")
        self.assertFalse(adapter.environment.identity()["allow_repeat_speaker"])

    def test_invalid_and_ambiguous_actions_are_rejected(self) -> None:
        with self.assertRaises(ParseError):
            parse_social_action("hello", ("speak", "leave"))
        with self.assertRaises(ParseError):
            parse_social_action('{"action_type":"speak","content":""}', ("speak", "leave"))
        with self.assertRaises(ParseError):
            parse_social_action('{"action_type":"teleport","content":"x"}', ("speak", "leave"))

    def test_early_leave_and_max_turn_termination_are_distinct(self) -> None:
        environment = RoleIsolatedSocialEnvironment(randomize_turn_order=False, default_max_turns=3)
        state = environment.reset(self.sotopia_case, seed=1)
        actor = state.next_actor
        transition = environment.apply(state, actor=actor, action=SocialAction("leave", "done"))
        self.assertTrue(transition.terminal)
        self.assertEqual(transition.terminal_reason, "agent_left")

        one_turn_case = replace(
            self.sotopia_case,
            input_data={**self.sotopia_case.input_data, "max_turns": 1},
        )
        state = environment.reset(one_turn_case, seed=1)
        transition = environment.apply(
            state,
            actor=state.next_actor,
            action=SocialAction("speak", "one turn only"),
        )
        self.assertTrue(transition.terminal)
        self.assertEqual(transition.terminal_reason, "max_turns")

    def test_sotopia_terminates_after_more_than_two_consecutive_none_actions(self) -> None:
        adapter = SotopiaAdapter()
        state = adapter.environment.reset(self.sotopia_case, seed=1)
        for expected_count in (1, 2):
            transition = adapter.environment.apply(
                state,
                actor=state.next_actor,
                action=SocialAction("none"),
            )
            self.assertFalse(transition.terminal)
            self.assertEqual(transition.metadata["stale_turn_count"], expected_count)
        transition = adapter.environment.apply(
            state,
            actor=state.next_actor,
            action=SocialAction("none"),
        )
        self.assertTrue(transition.terminal)
        self.assertEqual(transition.terminal_reason, "stale_turns")
        self.assertEqual(transition.metadata["stale_turn_count"], 3)

    def test_out_of_turn_action_is_invalid(self) -> None:
        environment = RoleIsolatedSocialEnvironment(randomize_turn_order=False, default_max_turns=3)
        state = environment.reset(self.sotopia_case, seed=1)
        wrong = next(agent for agent in state.agents if agent != state.next_actor)
        with self.assertRaises(ValidationError):
            environment.apply(state, actor=wrong, action=SocialAction("speak", "wrong turn"))

    def test_actor_backend_failure_is_structured_and_incomplete(self) -> None:
        adapter = AgentSenseAdapter()
        state = adapter.reset_official_state(self.agentsense_case, seed=9)
        request = adapter.build_actor_request(
            self.agentsense_case,
            state,
            state.next_actor,
            model="fixture",
            seed=9,
        )
        backend = ReplayBackend(errors={request.request_id: "timeout"})
        result = adapter.execute_case(
            self.agentsense_case,
            backend=backend,
            run_id="run",
            seed=9,
            model="fixture",
        )
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertEqual(result.error.stage, "actor_backend")
        self.assertFalse(result.metadata["episode_complete"])
        self.assertEqual(result.trace[-1].kind, "actor_failure")

    def test_malformed_actor_output_becomes_audited_none_action(self) -> None:
        adapter = SotopiaAdapter()
        state = adapter.environment.reset(self.sotopia_case, seed=2)
        request = adapter.build_actor_request(self.sotopia_case, state, state.next_actor, model="fixture", seed=2)
        backend = ReplayBackend(
            responses={request.request_id: "not-json"},
            default_response="not-json",
        )
        result = adapter.execute_case(
            self.sotopia_case,
            backend=backend,
            run_id="run",
            seed=2,
            model="fixture",
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertEqual(result.prediction["terminal_reason"], "stale_turns")
        self.assertEqual([event.kind for event in result.trace[:3]], ["none", "none", "none"])
        self.assertEqual(len(result.metadata["actor_output_fallbacks"]), 3)
        self.assertTrue(
            all(item["kind"] == "invalid_action" for item in result.metadata["actor_output_fallbacks"])
        )

    def test_sotopia_actor_token_limit_becomes_none_but_network_failure_does_not(self) -> None:
        adapter = SotopiaAdapter()
        state = adapter.environment.reset(self.sotopia_case, seed=2)
        request = adapter.build_actor_request(
            self.sotopia_case, state, state.next_actor, model="fixture", seed=2
        )

        class TokenLimitThenInvalidBackend(ReplayBackend):
            def generate(self, model_request):
                if model_request.request_id == request.request_id:
                    raise BackendTokenLimitError("synthetic actor truncation")
                return super().generate(model_request)

        truncated = adapter.execute_case(
            self.sotopia_case,
            backend=TokenLimitThenInvalidBackend(default_response="not-json"),
            run_id="run",
            seed=2,
            model="fixture",
        )
        self.assertEqual(truncated.status, ResultStatus.COMPLETED)
        self.assertEqual(truncated.trace[0].kind, "none")
        self.assertEqual(
            truncated.metadata["actor_output_fallbacks"][0]["exception_type"],
            "BackendTokenLimitError",
        )

        failed = adapter.execute_case(
            self.sotopia_case,
            backend=ReplayBackend(errors={request.request_id: "timeout"}),
            run_id="run",
            seed=2,
            model="fixture",
        )
        self.assertEqual(failed.status, ResultStatus.FAILED)
        self.assertEqual(failed.error.stage, "actor_backend")

    def test_sotopia_actor_contract_retry_can_recover(self) -> None:
        adapter = SotopiaAdapter()
        responses = dict(adapter.replay_responses(self.sotopia_case, seed=2))
        first_request = adapter.build_request(self.sotopia_case, model="fixture", seed=2)
        valid = responses[first_request.request_id]
        responses[first_request.request_id] = '{"action_type":null,"content":"bad"}'
        responses[f"{first_request.request_id}:contract_retry:1"] = valid
        result = adapter.execute_case(
            self.sotopia_case,
            backend=ReplayBackend(responses),
            run_id="run",
            seed=2,
            model="fixture",
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)

    def test_agentsense_actor_uses_official_free_form_protocol(self) -> None:
        adapter = AgentSenseAdapter()
        state = adapter.reset_official_state(self.agentsense_case, seed=9)
        request = adapter.build_actor_request(
            self.agentsense_case,
            state,
            state.next_actor,
            model="fixture",
            seed=9,
        )
        self.assertIsNone(request.response_format)
        self.assertEqual(adapter.environment.allowed_actions, ("speak",))
        self.assertIn("hiding your secrets as much as possible", request.messages[0].content)
        self.assertIn("Please answer in a paragraph", request.messages[0].content)
        self.assertEqual(state.transcript[0].content, "Hi, there!")

    def test_agentsense_information_interview_keeps_full_chat_history(self) -> None:
        adapter = AgentSenseAdapter()
        state = adapter.reset_official_state(self.agentsense_case, seed=9)
        actor = state.next_actor
        adapter.environment.apply(
            state,
            actor=actor,
            action=SocialAction("speak", "A public detail from the interaction."),
        )
        question = self.agentsense_case.input_data["information_questions"][0]
        request = adapter.build_information_request(
            self.agentsense_case,
            state,
            question,
            index=0,
            model="fixture",
            seed=9,
        )
        combined = "\n".join(message.content for message in request.messages)
        self.assertIn("Hi, there!", combined)
        self.assertIn("A public detail from the interaction.", combined)
        self.assertIn("Please answer the question and only output your choice.", combined)
        self.assertIn("hiding your secrets as much as possible", request.messages[0].content)

    def test_sotopia_actor_prompt_contains_official_behavior_constraints(self) -> None:
        adapter = SotopiaAdapter()
        state = adapter.environment.reset(self.sotopia_case, seed=2)
        request = adapter.build_actor_request(
            self.sotopia_case,
            state,
            state.next_actor,
            model="fixture",
            seed=2,
        )
        system = request.messages[0].content
        self.assertIn("try your best to achieve", system)
        self.assertIn("align with their character traits", system)
        self.assertIn("naturalness and realism is essential", system)
        self.assertIn('You can "leave" this conversation if', system)
        self.assertNotIn("\nRelationship:", system)

    def test_sotopia_normalization_boundaries_and_validation(self) -> None:
        for name, (low, high) in SOTOPIA_DIMENSIONS.items():
            self.assertEqual(normalize_sotopia_dimension(name, low), 0.0)
            self.assertEqual(normalize_sotopia_dimension(name, high), 1.0)
            with self.assertRaises(ValidationError):
                normalize_sotopia_dimension(name, high + 1)


if __name__ == "__main__":
    unittest.main()
