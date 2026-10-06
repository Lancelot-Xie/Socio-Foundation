import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_eval.backends.episode_budget import EpisodeOutputTokenBudgetBackend, episode_budget_options_from_role
from sim_eval.backends.openai_compatible import OpenAICompatibleBackend
from sim_eval.contracts import ChatMessage, ModelRequest
from sim_eval.errors import ConfigurationError, EpisodeTokenBudgetExceeded
from sim_eval.runtime_config import EVALUATED_ROLE_BY_BENCHMARK, build_role_routed_backend, resolve_role_config
from sim_eval.smoke import _prepare_smoke, _resume_route_identity
from sim_eval.thinking_budget import thinking_budget_identity

ROOT = Path(__file__).resolve().parents[1]


class Counter:
    def count_prompt(self, request, **kwargs):
        return 100

    def count_content(self, response):
        return 6

    def identity(self):
        return {'kind': 'fake'}


class RelayReasoningAllowanceTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch('sim_eval.backends.episode_budget.get_shared_huggingface_token_counter',
                           return_value=Counter())
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def router(self, role, payloads):
        def factory(name, **kwargs):
            kwargs['require_api_key'] = False
            def transport(payload, headers, timeout):
                payloads.append(copy.deepcopy(payload))
                return {'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}],
                        'usage': {'completion_tokens': 999999}}
            return OpenAICompatibleBackend(**kwargs, transport=transport)
        with patch('sim_eval.runtime_config.get_backend', side_effect=factory):
            return build_role_routed_backend({'roles': {'candidate': role}, 'routing': {'default_role': 'candidate'}})

    def request(self, cap):
        return ModelRequest(model='gpt-5.5', messages=(ChatMessage('user', 'test'),), max_tokens=cap,
                            metadata={'route_role': 'candidate'})

    def role(self):
        return resolve_role_config({'backend': 'chat_completions', 'profile': 'relay', 'model': 'gpt-5.5',
                                    'model_revision': 'test', 'reasoning_token_allowance': 2048})


    def test_context_and_remaining_content_budget_clamp_total_allowance(self):
        role = self.role()
        for context, expected in ((10000, 2060), (300, 200)):
            with self.subTest(context=context):
                options = dict(episode_budget_options_from_role(role))
                options['model_context_tokens'] = context
                payloads = []
                budget = EpisodeOutputTokenBudgetBackend(self.router(role, payloads),
                    evaluated_role='candidate', evaluated_model='gpt-5.5', max_output_tokens=12, **options)
                budget.generate(self.request(1024))
                self.assertEqual(payloads[0]['max_completion_tokens'], expected)
                self.assertEqual(budget.remaining_output_tokens, 6)
                with self.assertRaises(EpisodeTokenBudgetExceeded):
                    budget.generate(self.request(1024))
                self.assertEqual(len(payloads), 1)

    def test_allowance_changes_run_and_resume_identity_and_rejects_invalid_settings(self):
        role = self.role()
        original = copy.deepcopy(role)
        original.pop('reasoning_token_allowance')
        self.assertNotEqual(thinking_budget_identity(role), thinking_budget_identity(original))
        self.assertNotEqual(_resume_route_identity(role), _resume_route_identity(original))
        for value in (-1, True, '2048'):
            with self.assertRaises(ConfigurationError):
                resolve_role_config({**role, 'reasoning_token_allowance': value})
        with self.assertRaises(ConfigurationError):
            resolve_role_config({**role, 'backend': 'vllm', 'profile': 'vllm'})
        with self.assertRaises(ConfigurationError):
            resolve_role_config({**role, 'extra_body': {'max_completion_tokens': 100}})


if __name__ == '__main__':
    unittest.main()
