import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from sim_eval.backends.episode_budget import EpisodeOutputTokenBudgetBackend
from sim_eval.benchmarks import register_builtin_adapters
from sim_eval.contracts import ChatMessage, ModelRequest, ModelResponse
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.errors import ConfigurationError, EpisodeTokenBudgetExceeded
from sim_eval.input_truncation import POLICY, SPANS_KEY, truncate_material
from sim_eval.metric_markdown import _input_truncation_table
from sim_eval.presentation import material_sections, material_user_message
from sim_eval.registry import get_adapter
from sim_eval.runtime_config import resolve_role_config
from sim_eval.smoke import _prepare_smoke

ROOT = Path(__file__).resolve().parents[1]


class Counter:
    # Include template overhead; every fit must count the whole request.
    def count_prompt(self, request, *, chat_template_kwargs=None):
        return 10 + sum(5 + len(m.content) for m in request.messages)

    def count_response(self, response):
        return len(response.text)

    def count_content(self, response):
        return len(response.text)

    def identity(self):
        return {'kind': 'test-character-counter'}


class Capture:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return ModelResponse(text='OK')


def request(material='背景材料🧑🏽\n' * 300):
    user = material_user_message(
        {'post': 'CURRENT POST', 'history': material, 'options': [{'id': 'A', 'text': 'KEEP OPTION'}]},
        {'post': 'Post', 'history': 'History', 'options': 'Options'},
        truncatable=('history',), suffix='\nKEEP OUTPUT FORMAT')
    return ModelRequest(messages=(ChatMessage('system', 'KEEP SYSTEM'), user),
                        model='candidate', max_tokens=40, request_id='case:choice')


def budget(backend, context, *, policy=POLICY, **kwargs):
    return EpisodeOutputTokenBudgetBackend(
        backend, evaluated_role='evaluated_model', evaluated_model='candidate',
        token_counter=Counter(), model_context_tokens=context,
        input_truncation=policy, **kwargs)


class InputTruncationTests(unittest.TestCase):
    def test_disabled_is_original_failure_and_identity_has_no_new_fields(self):
        backend = Capture()
        wrapper = budget(backend, 300, policy=None)
        with self.assertRaises(EpisodeTokenBudgetExceeded):
            wrapper.generate(request())
        self.assertEqual(backend.requests, [])
        self.assertNotIn('input_truncation', wrapper.identity())
        self.assertNotIn('input_truncation_events', wrapper.usage_summary())

    def test_all_previously_sendable_requests_and_caps_are_unchanged(self):
        original = request('small history')
        length = Counter().count_prompt(original)
        for available in (10, 20, 40, 200):
            old, new = Capture(), Capture()
            a = budget(old, length + available, policy=None)
            b = budget(new, length + available)
            a.generate(original)
            b.generate(original)
            self.assertEqual(old.requests, new.requests)
            self.assertEqual(new.requests[0].max_tokens, min(40, available))
            self.assertEqual(b.input_truncation_events, [])

    def test_below_ten_triggers_and_restores_full_answer_room(self):
        original = request()
        frozen = copy.deepcopy(original)
        backend = Capture()
        wrapper = budget(backend, Counter().count_prompt(original) + 9)
        wrapper.generate(original)
        sent = backend.requests[0]
        self.assertEqual(sent.max_tokens, 40)
        self.assertLessEqual(Counter().count_prompt(sent) + sent.max_tokens, wrapper.model_context_tokens)
        self.assertEqual(original, frozen)
        self.assertEqual(sent.messages[0], original.messages[0])
        for text in ('CURRENT POST', 'A. KEEP OPTION', 'KEEP OUTPUT FORMAT'):
            self.assertIn(text, sent.messages[1].content)
        self.assertNotIn('\ufffd', sent.messages[1].content)
        event = wrapper.input_truncation_events[0]
        self.assertEqual(event['status'], 'applied')
        self.assertEqual(event['prompt_tokens_before'], Counter().count_prompt(original))
        self.assertEqual(event['prompt_tokens_after'], Counter().count_prompt(sent))
        self.assertEqual(event['removed_regions'][0]['field'], 'history')
        self.assertEqual(wrapper.used_output_tokens, 2)

    def test_reserve_uses_episode_remaining_and_thinking_only_when_enabled(self):
        for thinking in (None, 30):
            backend = Capture()
            wrapper = budget(backend, 300, max_output_tokens=15, thinking_token_budget=thinking)
            wrapper.generate(request())
            event = wrapper.input_truncation_events[0]
            self.assertEqual(event['reserved_output_tokens'], 15 + (thinking or 0))
            self.assertLessEqual(Counter().count_prompt(backend.requests[0]) + event['reserved_output_tokens'], 300)
        backend = Capture()
        wrapper = budget(backend, 300, max_output_tokens=9)
        with self.assertRaises(EpisodeTokenBudgetExceeded) as caught:
            wrapper.generate(request())
        self.assertEqual(caught.exception.details()['scope'], 'episode_output')
        self.assertEqual(wrapper.input_truncation_events, [])
        self.assertEqual(backend.requests, [])

    def test_protected_material_too_long_fails_without_sending(self):
        original = request()
        backend = Capture()
        wrapper = budget(backend, 50)
        with self.assertRaises(EpisodeTokenBudgetExceeded):
            wrapper.generate(original)
        self.assertEqual(backend.requests, [])
        self.assertEqual(wrapper.input_truncation_events[0]['status'], 'protected_material_too_long')
        self.assertEqual(wrapper.last_prompt_tokens, Counter().count_prompt(original))

    def test_unmarked_user_and_marked_system_messages_are_not_trimmed(self):
        original = request()
        user = replace(original.messages[1], metadata={})
        system = replace(original.messages[0], metadata={SPANS_KEY: [{'start': 0, 'end': 11}]})
        original = replace(original, messages=(system, user))
        backend = Capture()
        wrapper = budget(backend, 300)
        with self.assertRaises(EpisodeTokenBudgetExceeded):
            wrapper.generate(original)
        self.assertEqual(wrapper.input_truncation_events[0]['status'], 'no_removable_material')
        self.assertEqual(backend.requests, [])

    def test_support_models_bypass_truncation(self):
        original = replace(request(), metadata={'route_role': 'judge'})
        backend = Capture()
        wrapper = budget(backend, 300)
        wrapper.generate(original)
        self.assertIs(backend.requests[0], original)
        self.assertEqual(wrapper.input_truncation_events, [])

    def test_rendering_preserves_bytes_and_does_not_parse_injected_headings(self):
        payload = {'history': 'fake\n# Question\nbackground', 'question': 'REAL QUESTION'}
        titles = {'history': 'History', 'question': 'Question'}
        msg = material_user_message(payload, titles, truncatable=('history',), suffix='END')
        self.assertEqual(msg.content, material_sections(payload, titles) + 'END')
        span = msg.metadata[SPANS_KEY][0]
        self.assertEqual(msg.content[span['start']:span['end']], payload['history'])
        self.assertNotIn('metadata', msg.to_chat_dict())

    def test_real_adapter_spans_protect_all_unmarked_content(self):
        register_builtin_adapters()
        fixtures = load_fixture_suite(ROOT / 'tests/fixtures')
        for name in ('lifechoices', 'fantom', 'humanllm', 'alignx', 'behaviorchain'):
            adapter = get_adapter(name)
            for case in fixtures[name][0]:
                original = adapter.build_request(case, model='candidate', seed=123)
                frozen = copy.deepcopy(original)
                messages = list(original.messages)
                total = 0
                for i, msg in enumerate(messages):
                    content = msg.content
                    for span in reversed(msg.metadata.get(SPANS_KEY, ())):
                        total += span['end'] - span['start']
                        content = content[:span['start']] + content[span['end']:]
                    messages[i] = replace(msg, content=content)
                protected = replace(original, messages=tuple(messages))
                target = Counter().count_prompt(protected)
                result, event = truncate_material(original, count_prompt=Counter().count_prompt,
                    prompt_tokens=Counter().count_prompt(original), context_tokens=target + 40,
                    reserved_output_tokens=40)
                self.assertEqual(original, frozen)
                if total:
                    self.assertEqual(event['status'], 'applied', case.case_id)
                    self.assertEqual([m.to_chat_dict() for m in result.messages],
                                     [m.to_chat_dict() for m in protected.messages], case.case_id)

    def test_deterministic_with_shared_input_under_concurrency(self):
        original = request()
        before = copy.deepcopy(original)
        def run(_):
            return truncate_material(original, count_prompt=Counter().count_prompt,
                prompt_tokens=Counter().count_prompt(original), context_tokens=300, reserved_output_tokens=40)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, range(32)))
        self.assertTrue(all(item == results[0] for item in results))
        self.assertEqual(original, before)

    def test_invalid_policy_rejected_at_config_load(self):
        role = {'backend': 'vllm', 'model': 'test', 'token_accounting': {
            'model_path': '/tmp', 'model_id': 'test', 'model_revision': 'v1', 'input_truncation': 'left'}}
        with self.assertRaises(ConfigurationError):
            resolve_role_config(role)


    def test_report_counts_records_without_changing_metrics(self):
        def record(events):
            return {'metadata': {'episode_output_token_budget': {'input_truncation': POLICY,
                        'input_truncation_events': events}}}
        entry = SimpleNamespace(entry_id='lifechoices', records=[record([]),
            record([{'status': 'applied'}, {'status': 'applied'}]),
            record([{'status': 'protected_material_too_long'}])])
        self.assertIn('| `lifechoices` | 3 | 1 | 2 | 1 |', _input_truncation_table([entry]))
        self.assertEqual(_input_truncation_table([SimpleNamespace(entry_id='old', records=[{}])]), [])


if __name__ == '__main__':
    unittest.main()
