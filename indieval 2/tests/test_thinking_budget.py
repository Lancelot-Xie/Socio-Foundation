import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_eval.backends.episode_budget import EpisodeOutputTokenBudgetBackend, episode_budget_options_from_role
from sim_eval.backends.openai_compatible import OpenAICompatibleBackend
from sim_eval.backends.routed import RoleRoutedBackend
from sim_eval.contracts import ChatMessage, ModelRequest
from sim_eval.errors import ConfigurationError, EpisodeTokenBudgetExceeded
from sim_eval.runtime_config import (
    build_api_backend, build_role_routed_backend, load_benchmark_runtime_config,
    resolve_role_config, role_request_overrides, EVALUATED_ROLE_BY_BENCHMARK,
)
from sim_eval.smoke import _prepare_smoke
from sim_eval.thinking_budget import apply_thinking_defaults, thinking_budget_identity

ROOT = Path(__file__).resolve().parents[1]


class FakeCounter:
    def __init__(self, content_tokens=1):
        self.content_tokens = content_tokens

    def count_content(self, response):
        return self.content_tokens

    def count_prompt(self, request, **kwargs):
        return 100

    def count_response(self, response):
        return 1

    def identity(self):
        return {'kind': 'test'}


class ThinkingBudgetTests(unittest.TestCase):
    def setUp(self):
        # Live thinking evaluations load the shared tokenizer. Offline tests use
        # a deterministic content counter instead of cluster tokenizer files.
        self.counter_patch = patch(
            'sim_eval.backends.episode_budget.get_shared_huggingface_token_counter',
            return_value=FakeCounter())
        self.counter_patch.start()
        self.addCleanup(self.counter_patch.stop)

    def role(self, enabled=True, budget=None, cap=1024):
        extra = {'chat_template_kwargs': {'enable_thinking': enabled}}
        if budget is not None:
            extra['thinking_token_budget'] = budget
        return resolve_role_config({
            'backend': 'vllm', 'model': 'target', 'model_revision': 'test',
            'extra_body': extra, 'generation': {'temperature': 0, 'max_tokens': cap},
        })

    def request(self, cap=1024, role='evaluated_model'):
        return ModelRequest(model='target', messages=[ChatMessage('user', 'test')],
                            max_tokens=cap, metadata={'route_role': role, 'actor': role})

    def backend_factory(self, payloads, usage=7):
        def factory(name, **kwargs):
            def transport(payload, headers, timeout):
                payloads.append(copy.deepcopy(payload))
                result = {'choices': [{'message': {'content': 'answer', 'reasoning': 'hidden'},
                                       'finish_reason': 'stop'}]}
                if usage is not None:
                    result['usage'] = {'completion_tokens': usage, 'prompt_tokens': 100,
                                       'total_tokens': 100 + usage}
                return result
            kwargs['require_api_key'] = False
            return OpenAICompatibleBackend(**kwargs, transport=transport)
        return factory

    def router(self, role, payloads, usage=7):
        with patch('sim_eval.runtime_config.get_backend', side_effect=self.backend_factory(payloads, usage)):
            return build_role_routed_backend({'roles': {'evaluated_model': role},
                                              'routing': {'default_role': 'evaluated_model'}})

    def test_all_answer_caps_add_once_and_content_remains_answer(self):
        for cap in (200, 256, 512, 1024, 2048):
            with self.subTest(cap=cap):
                role = self.role(cap=cap)
                payloads = []
                router = self.router(role, payloads)
                budget = EpisodeOutputTokenBudgetBackend(
                    router, evaluated_role='evaluated_model', evaluated_model='target',
                    **episode_budget_options_from_role(role))
                request = self.request(cap)
                for _ in range(2):
                    response = budget.generate(request)
                    self.assertEqual(response.text, 'answer')
                    self.assertEqual(payloads[-1]['max_tokens'], cap + 2048)
                    self.assertEqual(payloads[-1]['thinking_token_budget'], 2048)
                    self.assertIs(payloads[-1]['include_reasoning'], False)
                self.assertEqual(request.max_tokens, cap)
                self.assertEqual(budget.used_output_tokens, 2)

    def test_no_thinking_payload_and_identity_are_unchanged(self):
        for enabled in (False, None):
            role = self.role(enabled=enabled)
            payloads = []
            self.router(role, payloads).generate(self.request())
            self.assertEqual(payloads[0]['max_tokens'], 1024)
            self.assertNotIn('thinking_token_budget', payloads[0])
            self.assertNotIn('include_reasoning', payloads[0])
            self.assertEqual(thinking_budget_identity(role), {})
            self.assertEqual(episode_budget_options_from_role(role), {})

    def test_non_vllm_profiles_do_not_get_the_budget_policy(self):
        for profile in ('deepseek', 'relay', 'openai_compatible'):
            backend = OpenAICompatibleBackend(
                profile=profile, extra_body={'chat_template_kwargs': {'enable_thinking': True}})
            payload = backend.build_payload(self.request())
            self.assertEqual(payload['max_tokens' if profile != 'relay' else 'max_completion_tokens'], 1024)
            self.assertNotIn('thinking_token_budget', payload)
            self.assertNotIn('include_reasoning', payload)

    def test_budget_override_validation_and_identity(self):
        for value in (0, 512, 4096):
            role = self.role(budget=value)
            payloads = []
            self.router(role, payloads).generate(self.request())
            self.assertEqual(payloads[0]['max_tokens'], 1024 + value)
        self.assertNotEqual(thinking_budget_identity(self.role(budget=512)),
                            thinking_budget_identity(self.role(budget=2048)))
        for value in (-1, True, 0.8, '2048'):
            with self.assertRaises(ConfigurationError):
                self.role(budget=value)
        with self.assertRaises(ConfigurationError):
            apply_thinking_defaults({'profile': 'vllm', 'extra_body': {
                'chat_template_kwargs': {'enable_thinking': True}, 'max_tokens': 9999}})

    def test_total_cap_adds_thinking_to_content_balance_but_respects_context(self):
        role = self.role()
        for context, episode, expected in ((10000, 500, 2548), (1600, 8000, 1500)):
            payloads = []
            router = self.router(role, payloads, usage=expected)
            options = dict(episode_budget_options_from_role(role))
            options['model_context_tokens'] = context
            budget = EpisodeOutputTokenBudgetBackend(
                router, evaluated_role='evaluated_model', evaluated_model='target',
                max_output_tokens=episode, **options)
            budget.generate(self.request())
            self.assertEqual(payloads[0]['max_tokens'], expected)
            self.assertEqual(budget.used_output_tokens, 1)
            self.assertEqual(budget.remaining_output_tokens, episode - 1)

    def test_hidden_reasoning_without_usage_still_charges_only_content(self):
        role = self.role()
        payloads = []
        router = self.router(role, payloads, usage=None)
        budget = EpisodeOutputTokenBudgetBackend(
            router, evaluated_role='evaluated_model', evaluated_model='target',
            **episode_budget_options_from_role(role))
        budget.generate(self.request())
        self.assertEqual(budget.used_output_tokens, 1)
        self.assertEqual(budget.fallback_usage_request_count, 0)
        self.assertEqual(budget.tokenizer_usage_request_count, 1)
        self.assertEqual(budget.identity()['scope'], 'evaluated_model_content_only')

    def test_ten_token_check_uses_content_remaining_not_total_usage(self):
        payloads = []
        role = self.role()
        options = dict(episode_budget_options_from_role(role))
        options['token_counter'] = FakeCounter(content_tokens=300)
        budget = EpisodeOutputTokenBudgetBackend(
            self.router(role, payloads, usage=1800), evaluated_role='evaluated_model',
            evaluated_model='target', max_output_tokens=609, **options)
        budget.generate(self.request())
        self.assertEqual(budget.remaining_output_tokens, 309)
        budget.generate(self.request())
        self.assertEqual(budget.remaining_output_tokens, 9)
        self.assertEqual([p['max_tokens'] for p in payloads], [2657, 2357])
        with self.assertRaises(EpisodeTokenBudgetExceeded) as caught:
            budget.generate(self.request())
        self.assertEqual(budget.exhaustion['scope'], 'episode_output')
        self.assertEqual(len(payloads), 2)
        self.assertEqual(budget.used_output_tokens, 600)

    def test_context_check_and_exact_ten_token_boundary(self):
        role = self.role()
        for context, episode, expected_calls in ((109, 500, 0), (110, 10, 1)):
            payloads = []
            options = dict(episode_budget_options_from_role(role))
            options['model_context_tokens'] = context
            budget = EpisodeOutputTokenBudgetBackend(
                self.router(role, payloads), evaluated_role='evaluated_model',
                evaluated_model='target', max_output_tokens=episode, **options)
            if expected_calls:
                budget.generate(self.request())
                self.assertEqual(payloads[0]['max_tokens'], 10)
            else:
                with self.assertRaises(EpisodeTokenBudgetExceeded):
                    budget.generate(self.request())
                self.assertEqual(budget.exhaustion['scope'], 'model_context')
            self.assertEqual(len(payloads), expected_calls)

    def test_answer_overrun_is_recorded_without_truncating_returned_content(self):
        payloads = []
        options = dict(episode_budget_options_from_role(self.role()))
        options['token_counter'] = FakeCounter(content_tokens=120)
        budget = EpisodeOutputTokenBudgetBackend(
            self.router(self.role(), payloads), evaluated_role='evaluated_model',
            evaluated_model='target', max_output_tokens=100, **options)
        response = budget.generate(self.request())
        self.assertEqual(response.text, 'answer')
        self.assertEqual(budget.budget_overrun_tokens, 20)
        with self.assertRaises(EpisodeTokenBudgetExceeded):
            budget.generate(self.request())

    def test_thinking_does_not_silently_fall_back_to_total_usage_without_tokenizer(self):
        with self.assertRaises(ConfigurationError):
            EpisodeOutputTokenBudgetBackend(
                self.router(self.role(), []), evaluated_role='evaluated_model',
                evaluated_model='target', thinking_token_budget=2048)

    def test_tau_router_adds_only_to_evaluated_role(self):
        payloads = []
        target = self.role(cap=256)
        assistant = self.role(enabled=False, cap=512)
        with patch('sim_eval.runtime_config.get_backend', side_effect=self.backend_factory(payloads)):
            router = RoleRoutedBackend(
                evaluated_backend=build_api_backend(target), fixed_assistant_backend=build_api_backend(assistant),
                evaluated_request_overrides=role_request_overrides(target),
                fixed_assistant_request_overrides=role_request_overrides(assistant))
        budget = EpisodeOutputTokenBudgetBackend(
            router, evaluated_role='evaluated_user', evaluated_model='target',
            support_roles=('fixed_assistant',), **episode_budget_options_from_role(target))
        budget.generate(self.request(256, 'evaluated_user'))
        budget.generate(self.request(512, 'fixed_assistant'))
        self.assertEqual([p['max_tokens'] for p in payloads], [2304, 512])
        self.assertNotIn('thinking_token_budget', payloads[1])
        self.assertEqual(budget.evaluated_request_count, 1)

    def test_userlm_adapter_caps_survive_with_no_role_cap(self):
        config = load_benchmark_runtime_config(ROOT / 'sim_eval/resources/protocols/userlm.json')
        config['roles']['evaluated_user']['extra_body']['chat_template_kwargs']['enable_thinking'] = True
        payloads = []
        with patch('sim_eval.runtime_config.get_backend', side_effect=self.backend_factory(payloads)):
            router = build_role_routed_backend(config)
        role = resolve_role_config(config['roles']['evaluated_user'])
        budget = EpisodeOutputTokenBudgetBackend(
            router, evaluated_role='evaluated_user', evaluated_model=role['model'],
            **episode_budget_options_from_role(role))
        for cap in (200, 512):
            budget.generate(self.request(cap, 'evaluated_user'))
        self.assertEqual([p['max_tokens'] for p in payloads], [2248, 2560])

