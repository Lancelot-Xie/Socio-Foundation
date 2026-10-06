"""Relay usage must not alter an explicitly selected local-content budget."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_eval.backends.episode_budget import (
    EpisodeOutputTokenBudgetBackend, episode_budget_options_from_role,
    episode_token_scope, token_accounting_preflight,
)
from sim_eval.backends.openai_compatible import OpenAICompatibleBackend
from sim_eval.contracts import ChatMessage, ModelRequest, ModelResponse, TokenUsage
from sim_eval.errors import ConfigurationError, EpisodeTokenBudgetExceeded
from sim_eval.integrations.token_counting import default_token_accounting_config
from sim_eval.runtime_config import EVALUATED_ROLE_BY_BENCHMARK, resolve_role_config, role_request_overrides
from sim_eval.smoke import _prepare_smoke

ROOT = Path(__file__).resolve().parents[1]


class Counter:
    def count_content(self, response):
        return len(response.text.split())

    def count_prompt(self, request, **kwargs):
        return 20

    def count_response(self, response):
        raise AssertionError('Local content accounting must not include hidden reasoning')

    def identity(self):
        return {'kind': 'test-local-counter'}


class Backend:
    def __init__(self, usage):
        self.requests = []
        self.response = ModelResponse(text='one two three four five six', usage=usage,
                                     raw={'reasoning_content': 'hidden reasoning'})

    def generate(self, request):
        self.requests.append(request)
        return self.response


class LocalContentBudgetTests(unittest.TestCase):
    def test_wrong_or_missing_usage_is_ignored_and_response_is_preserved(self):
        for usage in (None, TokenUsage(completion_tokens=0), TokenUsage(completion_tokens=999999),
                      TokenUsage(prompt_tokens=4409, total_tokens=999999)):
            with self.subTest(usage=usage):
                backend = Backend(usage)
                budget = EpisodeOutputTokenBudgetBackend(
                    backend, evaluated_role='evaluated_user', evaluated_model='gpt-5.5',
                    max_output_tokens=20, token_counter=Counter(), model_context_tokens=35,
                    output_token_source='local_content')
                req = ModelRequest(messages=(ChatMessage('user', 'test'),), model='gpt-5.5', max_tokens=100)
                self.assertIs(budget.generate(req), backend.response)
                self.assertEqual(backend.requests[0].max_tokens, 15)
                self.assertEqual(budget.used_output_tokens, 6)
                budget.generate(req)
                self.assertEqual(backend.requests[1].max_tokens, 14)
                self.assertEqual(budget.remaining_output_tokens, 8)
                with self.assertRaises(EpisodeTokenBudgetExceeded):
                    budget.generate(req)
                self.assertEqual(len(backend.requests), 2)
                self.assertEqual(budget.provider_usage_request_count, 0)
                self.assertEqual(budget.tokenizer_usage_request_count, 2)
                self.assertEqual(budget.identity()['scope'], 'evaluated_model_content_only')

    def test_default_behavior_and_identity_remain_provider_based(self):
        backend = Backend(TokenUsage(completion_tokens=13))
        budget = EpisodeOutputTokenBudgetBackend(
            backend, evaluated_role='evaluated_user', evaluated_model='gpt-5.5', token_counter=Counter())
        budget.generate(ModelRequest(messages=(ChatMessage('user', 'test'),), model='gpt-5.5', max_tokens=100))
        self.assertEqual(budget.used_output_tokens, 13)
        self.assertNotIn('output_token_source', budget.identity())
        self.assertEqual(budget.identity()['revision'], 'evaluated-model-output-and-context-token-budget-v2')
        self.assertEqual(episode_token_scope({}), 'evaluated_model_outputs_only')

    def test_support_calls_do_not_consume_content_budget(self):
        backend = Backend(TokenUsage(completion_tokens=999999))
        budget = EpisodeOutputTokenBudgetBackend(
            backend, evaluated_role='evaluated_user', evaluated_model='gpt-5.5',
            token_counter=Counter(), output_token_source='local_content')
        budget.generate(ModelRequest(messages=(ChatMessage('user', 'test'),), model='gpt-5.5', metadata={'route_role': 'judge'}))
        self.assertEqual(budget.used_output_tokens, 0)

    def test_configuration_validation_and_runtime_options(self):
        role = {'backend': 'chat_completions', 'profile': 'relay', 'model': 'gpt-5.5',
                'model_revision': 'test', 'token_accounting': default_token_accounting_config()}
        role['token_accounting']['output_token_source'] = 'local_content'
        resolved = resolve_role_config(role)
        self.assertEqual(episode_token_scope(resolved), 'evaluated_model_content_only')
        self.assertEqual(token_accounting_preflight(resolved)['output_token_source'], 'local_content')
        with patch('sim_eval.backends.episode_budget.get_shared_huggingface_token_counter', return_value=Counter()):
            self.assertEqual(episode_budget_options_from_role(resolved)['output_token_source'], 'local_content')
        bad = copy.deepcopy(role)
        bad['token_accounting']['output_token_source'] = 'typo'
        with self.assertRaises(ConfigurationError):
            resolve_role_config(bad)
        with self.assertRaises(ConfigurationError):
            token_accounting_preflight(bad)
        with self.assertRaises(ConfigurationError):
            EpisodeOutputTokenBudgetBackend(Backend(None), evaluated_role='user', evaluated_model='gpt-5.5',
                                            output_token_source='local_content')



if __name__ == '__main__':
    unittest.main()
