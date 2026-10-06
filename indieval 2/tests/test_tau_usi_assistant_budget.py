import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.tau_usi import (
    ASSISTANT_STEP_LIMIT_POLICY, TauUSIAdapter, normalize_survey,
)
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.contracts import ModelResponse
from sim_eval.environments.tau_usi import ASSISTANT_ROLE
from sim_eval.errors import (
    BackendStructuredOutputError, BackendTimeoutError,
    EpisodeTokenBudgetExhausted, ValidationError,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260812


class CapturingBackend(ReplayBackend):
    def __init__(self, responses, survey_error=None, survey_text=None):
        super().__init__(responses)
        self.requests = []
        self.survey_error = survey_error
        self.survey_text = survey_text

    def generate(self, request):
        self.requests.append(request)
        if request.metadata.get('stage') == 'post_interaction_survey' and self.survey_error:
            raise self.survey_error
        if request.metadata.get('stage') == 'post_interaction_survey' and self.survey_text is not None:
            return ModelResponse(text=self.survey_text)
        return super().generate(request)


class TauAssistantBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_fixture_suite(ROOT / 'tests/fixtures')['tau_usi'][0]

    def adapter(self, limit=1):
        base = TauUSIAdapter()
        return TauUSIAdapter(runtime_provenance=replace(
            base.runtime_for_case(self.cases[0]), max_assistant_steps_per_user_turn=limit))

    def execute(self, case, adapter, backend=None):
        backend = backend or ReplayBackend(adapter.replay_responses(case, seed=SEED))
        return adapter.execute_case(case, backend=backend, run_id='budget-test', seed=SEED, model='user')

    def test_pending_tools_yield_actual_last_text_and_continue_to_user_stop(self):
        case, adapter = self.cases[0], self.adapter()
        responses = adapter.replay_responses(case, seed=SEED)
        backend = CapturingBackend(responses)
        result = self.execute(case, adapter, backend)
        self.assertEqual(result.status.value, 'completed')
        self.assertEqual(result.prediction['terminal_reason'], 'user_stop')
        self.assertNotIn('budget_termination', result.metadata)
        self.assertEqual(result.prediction['tool_call_count'], 2)
        self.assertEqual(result.prediction['user_turn_count'], 2)
        handoffs = [event for event in result.trace if event.kind == 'assistant_handoff']
        self.assertEqual(len(handoffs), 2)
        for event in handoffs:
            self.assertEqual(event.content, result.trace[event.metadata['source_turn']].content)
        user_requests = [request for request in backend.requests if request.metadata.get('actor') == 'evaluated_user']
        self.assertIn('<function=lookup_order>', user_requests[1].messages[-1].content)
        self.assertNotIn('"eligible":true', '\n'.join(m.content for m in user_requests[1].messages))
        self.assertEqual(result.metadata['tau_usi']['simulator_survey'], normalize_survey(case.metadata['replay']['survey']))
        self.assertEqual(sum(r.metadata.get('actor') == ASSISTANT_ROLE for r in backend.requests), 2)
        survey = backend.requests[-1]
        self.assertEqual(survey.metadata['stage'], 'post_interaction_survey')
        self.assertEqual(survey.messages[-2].content, '###STOP###')
        self.assertIn('Options:', survey.messages[-1].content)

    def test_budget_outcomes_participate_in_full_distribution_and_expose_rate(self):
        adapter = self.adapter()
        results = [self.execute(case, adapter) for case in self.cases]
        metrics = adapter.aggregate(results)
        self.assertTrue(metrics['tau_usi.suite_complete'].value)
        self.assertEqual(metrics['tau_usi.case_failure_count'].value, 0)
        self.assertIsNotNone(metrics['tau_usi.usi'].value)
        self.assertEqual(metrics['tau_usi.assistant_step_limit_rate'].value, 1)
        mixed = adapter.aggregate([results[0], self.execute(self.cases[1], TauUSIAdapter())])
        self.assertEqual(mixed['tau_usi.assistant_step_limit_rate'].value, .5)

    def test_message_on_last_allowed_step_hands_back_and_resets_budget(self):
        adapter = self.adapter(limit=2)
        for case in self.cases:
            baseline = self.execute(case, TauUSIAdapter())
            result = self.execute(case, adapter)
            self.assertEqual(result.prediction, baseline.prediction)
            self.assertEqual(result.metrics, baseline.metrics)
            self.assertEqual(result.trace, baseline.trace)
            self.assertNotIn('budget_termination', result.metadata)
            self.assertEqual(result.prediction['user_turn_count'], 2)
            self.assertEqual(adapter.aggregate([result])['tau_usi.assistant_step_limit_rate'].value, 0)

    def test_post_limit_survey_backend_failures_are_not_imputed(self):
        adapter = self.adapter()
        good = self.execute(self.cases[1], adapter)
        for error in (BackendStructuredOutputError('bad survey'), BackendTimeoutError('survey offline')):
            backend = CapturingBackend(adapter.replay_responses(self.cases[0], seed=SEED), survey_error=error)
            result = self.execute(self.cases[0], adapter, backend)
            self.assertEqual(result.status.value, 'failed')
            self.assertEqual(result.error.stage, 'survey_backend')
            self.assertIsNone(adapter.aggregate([result, good])['tau_usi.usi'].value)

    def test_post_limit_budget_without_survey_keeps_five_component_score(self):
        adapter = self.adapter()
        backend = CapturingBackend(adapter.replay_responses(self.cases[0], seed=SEED), survey_error=
            EpisodeTokenBudgetExhausted('survey budget exhausted', scope='episode', episode_remaining_tokens=0))
        result = self.execute(self.cases[0], adapter, backend)
        self.assertIsNone(result.metadata['tau_usi']['simulator_survey'])
        metrics = adapter.aggregate([result, self.execute(self.cases[1], adapter)])
        self.assertIsNone(metrics['tau_usi.usi'].value)
        self.assertIsNotNone(metrics['tau_usi.usi_without_eval'].value)

    def test_received_invalid_survey_is_imputed_only_in_aggregate(self):
        adapter = self.adapter()
        backend = CapturingBackend(adapter.replay_responses(self.cases[0], seed=SEED), survey_text='not json')
        result = self.execute(self.cases[0], adapter, backend)
        self.assertEqual(result.metadata['tau_usi']['simulator_survey'], {})
        self.assertEqual(len(result.metadata['tau_usi']['survey_missing_fields']), 8)
        metrics = adapter.aggregate([result, self.execute(self.cases[1], adapter)])
        self.assertIsNotNone(metrics['tau_usi.usi'].value)
        self.assertEqual(set(metrics['tau_usi.eval'].metadata['imputed_field_count_per_batch'].values()), {8})
        self.assertEqual(result.metadata['tau_usi']['simulator_survey'], {})

    def test_final_environment_reward_is_read_after_actual_tool_state_change(self):
        adapter, case = self.adapter(), self.cases[0]
        responses = adapter.replay_responses(case, seed=SEED)
        original_reset = adapter.environment.reset
        for reward in (0.0, 1.0):
            class Session:
                calls = 0
                reward_calls = 0

                def execute_tool(self, name, arguments):
                    self.calls += 1
                    return 'actual observation', False

                def calculate_reward(self):
                    assert self.calls == 2
                    self.reward_calls += 1
                    return reward

            session = Session()

            def reset(*args, **kwargs):
                state = original_reset(*args, **kwargs)
                if kwargs.get('seed') != 0:
                    state.local_runtime = session
                return state

            with patch.object(adapter.environment, 'reset', side_effect=reset):
                result = self.execute(case, adapter, ReplayBackend(responses))
            self.assertEqual(session.reward_calls, 1)
            self.assertEqual(result.metadata['tau_usi']['environment_reward'], reward)
            self.assertEqual(result.metrics[0].value, reward)

    def test_missing_or_failed_reward_keeps_only_supported_components(self):
        adapter, case = self.adapter(), self.cases[0]
        good = self.execute(self.cases[1], adapter)
        original_finalize = adapter.environment.finalize_reward
        responses = adapter.replay_responses(case, seed=SEED)
        for error in (None, ValidationError('reward unavailable')):
            def finalize(state, value):
                if not state.terminal:
                    return original_finalize(state, value)
                if error is not None:
                    raise error
                return None

            with patch.object(adapter.environment, 'finalize_reward', side_effect=finalize):
                if error is not None:
                    with self.assertRaises(ValidationError):
                        self.execute(case, adapter, ReplayBackend(responses))
                    continue
                result = self.execute(case, adapter, ReplayBackend(responses))
            self.assertIsNone(result.metadata['tau_usi']['environment_reward'])
            metrics = adapter.aggregate([result, good])
            for key in ('ece', 'outcome_alignment', 'usi', 'usi_without_eval'):
                self.assertIsNone(metrics[f'tau_usi.{key}'].value)
            self.assertIsNotNone(metrics['tau_usi.d1_communication'].value)
            self.assertIsNotNone(metrics['tau_usi.eval'].value)

    def test_rollout_infrastructure_failure_still_fails(self):
        adapter, case = self.adapter(), self.cases[0]
        responses = adapter.replay_responses(case, seed=SEED)
        first_assistant = next(k for k in responses if ':assistant:' in k)
        result = self.execute(case, adapter, ReplayBackend(responses, errors={first_assistant: 'timeout'}))
        self.assertEqual(result.status.value, 'failed')
        self.assertEqual(result.error.stage, 'assistant_backend')
        self.assertEqual(adapter.aggregate([result])['tau_usi.assistant_step_limit_rate'].value, 0)

    def test_tool_termination_on_last_step_is_not_budget_truncation(self):
        adapter, case = self.adapter(), self.cases[0]
        responses = adapter.replay_responses(case, seed=SEED)
        original_reset = adapter.environment.reset

        class Session:
            def record_assistant_message(self, message):
                pass

            def execute_tool(self, name, arguments):
                return 'transferred to human', True

            def calculate_reward(self):
                return 1.0

        def reset(*args, **kwargs):
            state = original_reset(*args, **kwargs)
            if kwargs.get('seed') != 0:
                state.local_runtime = Session()
            return state

        with patch.object(adapter.environment, 'reset', side_effect=reset):
            result = self.execute(case, adapter, ReplayBackend(responses))
        self.assertEqual(result.prediction['terminal_reason'], 'user_stop')
        self.assertNotIn('budget_termination', result.metadata)
        self.assertEqual(adapter.aggregate([result])['tau_usi.assistant_step_limit_rate'].value, 1)

    def test_failed_feature_extraction_does_not_disappear_from_limit_rate(self):
        adapter, case = self.adapter(), self.cases[0]
        original_extract = adapter._feature_extractor
        original_terminate = adapter.environment.handoff_assistant_step_limit
        truncated = False

        def terminate(*args, **kwargs):
            nonlocal truncated
            transition = original_terminate(*args, **kwargs)
            truncated = True
            return transition

        def extract(messages):
            if truncated:
                raise ValidationError('extractor unavailable')
            return original_extract(messages)

        with patch.object(adapter, '_feature_extractor', side_effect=extract), \
                patch.object(adapter.environment, 'handoff_assistant_step_limit', side_effect=terminate):
            result = self.execute(case, adapter)
        self.assertEqual(result.status.value, 'failed')
        self.assertEqual(result.error.stage, 'behavior_features')
        metrics = adapter.aggregate([result, self.execute(self.cases[1], TauUSIAdapter())])
        self.assertEqual(metrics['tau_usi.assistant_step_limit_rate'].value, .5)
        self.assertIsNone(metrics['tau_usi.usi'].value)

    def test_new_policy_is_part_of_runtime_and_run_identity(self):
        adapter, case = self.adapter(), self.cases[0]
        self.assertEqual(adapter.runtime_for_case(case).to_dict()['assistant_step_limit_policy'], ASSISTANT_STEP_LIMIT_POLICY)
        self.assertEqual(adapter.environment_identity_for_case(case)['assistant_step_limit_policy'], ASSISTANT_STEP_LIMIT_POLICY)
        self.assertIn('v5-text-options-rng42', adapter.scorer_revision)


if __name__ == '__main__':
    unittest.main()
