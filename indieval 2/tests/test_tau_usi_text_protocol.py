"""Regression tests for the DeepSeek/60-round Supplemental protocol adaptation."""
import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.tau_usi import TauUSIAdapter, SURVEY_SCALES
from sim_eval.contracts import ChatMessage, ModelRequest, ModelResponse
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.environments.tau_usi import ASSISTANT_ROLE, USER_ROLE, parse_assistant_response, parse_user_action
from sim_eval.errors import ValidationError
from sim_eval.integrations.tau_bench_local import TauBenchLocalSession
from sim_eval.integrations.tau_usi_protocol import (
    FIELD_ORDINAL, SURVEY_FIELD_NAMES, coerce_schema_value, extract_fn_call,
    fixture_survey_options, parse_survey_options, tool_call_text,
)
from sim_eval.runtime_config import apply_global_eval_model

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260812


class TauTextProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_fixture_suite(ROOT / 'tests/fixtures')['tau_usi'][0]

    def run_case(self, case, responses=None, adapter=None):
        adapter = adapter or TauUSIAdapter()
        return adapter.execute_case(case, backend=ReplayBackend(
            responses or adapter.replay_responses(case, seed=SEED)),
            run_id='protocol-test', seed=SEED, model='user')

    def test_multiple_calls_and_mixed_parameter_syntax(self):
        text = '<function=one>\n<id>wrong</id><parameter=id>right</parameter>\n</function>\n' \
               '<function=two>\n<parameter=count>2</parameter>\n</function>'
        action = parse_assistant_response(ModelResponse(text=text))
        self.assertEqual([call.name for call in action.tool_calls], ['one', 'two'])
        self.assertEqual(action.tool_calls[0].arguments, {'id': 'right'})
        self.assertEqual(action.message, text)
        self.assertEqual(extract_fn_call('normal reply'), None)

    def test_multi_call_executes_serially_in_one_step_and_combines_observations(self):
        adapter, case = TauUSIAdapter(), self.cases[0]
        state = adapter.environment.reset(case, seed=SEED)
        adapter.environment.apply_user(state, parse_user_action('Please help.'))
        calls = [step['output']['tool_call'] for step in case.metadata['replay']['steps']
                 if step['actor'] == ASSISTANT_ROLE and step['output']['tool_call']]
        action = parse_assistant_response(ModelResponse(text='\n'.join(tool_call_text(**call) for call in calls)))
        adapter.environment.apply_assistant(state, action)
        self.assertEqual(state.assistant_step_count, 1)
        self.assertEqual([e.content['name'] for e in state.trace if e.kind == 'tool_call'], [c['name'] for c in calls])
        request = adapter.build_assistant_request(case, state, adapter.runtime_for_case(case), seed=SEED)
        self.assertFalse(request.tools)
        self.assertEqual(request.messages[-1].role, 'system')
        self.assertIn('\n\n', request.messages[-1].content)
        self.assertEqual(sum(m.role == 'assistant' and '<function=' in m.content for m in request.messages), 1)

    def test_malformed_call_is_private_correction_and_consumes_one_step(self):
        adapter, case = TauUSIAdapter(), self.cases[0]
        state = adapter.environment.reset(case, seed=SEED)
        adapter.environment.apply_user(state, parse_user_action('help'))
        action = parse_assistant_response(ModelResponse(text='<function=lookup_order>\n<parameter>oops</parameter>'))
        self.assertIn('Tool Call Format Error', action.format_error)
        adapter.environment.apply_assistant(state, action)
        self.assertEqual(state.assistant_step_count, 1)
        self.assertEqual(state.next_actor, ASSISTANT_ROLE)
        self.assertFalse(any(e.kind == 'tool_call' for e in state.trace))
        request = adapter.build_assistant_request(case, state, adapter.runtime_for_case(case), seed=SEED)
        self.assertEqual(request.messages[-1].role, 'user')
        adapter.environment.handoff_assistant_step_limit(state, limit=1)
        self.assertEqual(adapter.environment.observation(state, actor=USER_ROLE)[-1].content, action.message)

    def test_schema_coercion_preserves_nested_types_and_integer_hashes(self):
        schema = {'type':'array','items':{'type':'object','properties':{'n':{'type':'integer'},'ok':{'type':'boolean'}}}}
        self.assertEqual(coerce_schema_value('[{"n":"2","ok":"false"}]', schema, 'x'), [{'n':2,'ok':False}])
        self.assertIs(type(coerce_schema_value('250', {'type':'number'}, 'x')), int)
        with self.assertRaises(ValueError):
            coerce_schema_value('bad', {'type':'integer'}, 'x')

    def test_survey_reuses_original_history_and_text_options(self):
        adapter, case = TauUSIAdapter(), self.cases[0]
        class Capture(ReplayBackend):
            def generate(self, request):
                requests.append(request)
                return super().generate(request)
        requests = []
        adapter.execute_case(case, backend=Capture(adapter.replay_responses(case, seed=SEED)),
                             run_id='history', seed=SEED, model='user')
        last_user = [r for r in requests if r.metadata.get('actor') == USER_ROLE and not r.metadata.get('stage')][-1]
        survey = requests[-1]
        self.assertEqual([(m.role,m.content) for m in survey.messages[:-2]], [(m.role,m.content) for m in last_user.messages])
        self.assertEqual((survey.messages[-2].role, survey.messages[-2].content), ('assistant','###STOP###'))
        self.assertIn('question_amount_preference:', survey.messages[-1].content)
        self.assertIn("['Too many', 'About right', 'Too few']", survey.messages[-1].content)
        self.assertIsNone(survey.response_format)

    def test_partial_survey_preserves_answers_and_matches_official_rng_stream(self):
        adapter = TauUSIAdapter()
        results = []
        for case in self.cases:
            responses = adapter.replay_responses(case, seed=SEED)
            responses[f'{case.case_id}:survey'] = {'text': json.dumps({'task_success':'Yes - Task completed', 'efficiency':99})}
            results.append(self.run_case(case, responses, adapter))
        metrics = adapter.aggregate(results)
        rng = np.random.default_rng(42)
        expected = {}
        ordered = sorted(results, key=lambda r:r.metadata['tau_usi']['task_key'])
        for batch in ('fixture_h1','fixture_h2','fixture_h3'):
            scores = []
            for result in ordered:
                payload = result.metadata['tau_usi']
                self.assertEqual(payload['simulator_survey'], {'task_success': .75})
                diffs = []
                for source, options in FIELD_ORDINAL.items():
                    name = SURVEY_FIELD_NAMES[source]
                    values = sorted(set(options.values()))
                    sim = .75 if source == 'task_success' else (rng.choice(values)-min(values))/(max(values)-min(values))
                    diffs.append(abs(payload['human_references'][batch]['survey'][name]-sim))
                scores.append((1-np.mean(diffs))*100)
            expected[batch] = np.mean(scores)
        for batch, value in expected.items():
            self.assertAlmostEqual(metrics['tau_usi.eval'].metadata['per_batch'][batch], value)
        self.assertEqual(adapter.aggregate(list(reversed(results)))['tau_usi.eval'], metrics['tau_usi.eval'])
        self.assertEqual(set(metrics['tau_usi.eval'].metadata['imputed_field_count_per_batch'].values()), {14})

    def test_text_option_normalization_does_not_accept_live_integer_answers(self):
        self.assertEqual(parse_survey_options({'overall_score': 5}), {})
        self.assertEqual(parse_survey_options({'question_amount_preference':'Too few'}), {'question_amount':0})
        self.assertEqual(parse_survey_options({'question_amount_preference':'About right'}), {'question_amount':1})

    @unittest.skipUnless((ROOT / "third_party/tau-bench").is_dir(), "optional external tau-bench checkout is not bundled")
    def test_transfer_freezes_database_and_reward_but_dialogue_can_continue(self):
        session = TauBenchLocalSession('retail', 0)
        _, terminal = session.execute_tool('transfer_to_human_agents', {'summary':'Cannot handle request'})
        self.assertTrue(terminal)
        reward = session.calculate_reward()
        before = json.dumps(session.data, sort_keys=True)
        count = len(session.actions)
        for action in session.task.actions:
            observation, ended = session.execute_tool(action.name, action.arguments)
            self.assertIn('already terminated', observation)
            self.assertTrue(ended)
        session.record_assistant_message('Transferred successfully.')
        self.assertEqual(len(session.actions), count)
        self.assertEqual(json.dumps(session.data, sort_keys=True), before)
        self.assertEqual(session.calculate_reward(), reward)

    def test_sixty_main_rounds_retain_the_last_customer_reply(self):
        adapter, case = TauUSIAdapter(), self.cases[0]
        adapter = TauUSIAdapter(runtime_provenance=replace(adapter.runtime_for_case(case), max_user_turns=60))
        requests = []
        class Endless:
            def generate(self, request):
                requests.append(request)
                if request.metadata.get('stage') == 'post_interaction_survey':
                    return ModelResponse(text=json.dumps(fixture_survey_options(case.metadata['replay']['survey'])))
                return ModelResponse(text='Please continue.' if request.metadata['actor'] == USER_ROLE else 'What else?')
        result = adapter.execute_case(case, backend=Endless(), run_id='60-rounds', seed=SEED, model='user')
        self.assertEqual(result.prediction['terminal_reason'], 'user_turn_limit')
        self.assertEqual(sum(r.metadata['actor'] == ASSISTANT_ROLE for r in requests), 60)
        self.assertEqual(result.prediction['user_turn_count'], 61)  # opening reply + 60 cycle replies
        self.assertEqual(requests[-1].messages[-2].content, 'Please continue.')

    def test_only_tau_text_tool_feedback_can_use_extra_system_messages(self):
        messages = (ChatMessage('system','policy'), ChatMessage('user','hello'),
                    ChatMessage('system','tool result', metadata={'source':'tau_tool_feedback'}))
        metadata = {'benchmark_id':'tau_usi','actor':ASSISTANT_ROLE,'tool_protocol':'supplemental-text-tool-calls-v4'}
        ModelRequest(messages=messages, model='deepseek', metadata=metadata)
        for altered in ({}, {**metadata,'benchmark_id':'sotopia'}, {**metadata,'actor':USER_ROLE}):
            with self.assertRaises(ValidationError):
                ModelRequest(messages=messages, model='deepseek', metadata=altered)
        with self.assertRaises(ValidationError):
            ModelRequest(messages=(*messages[:2], ChatMessage('system','unmarked')), model='deepseek', metadata=metadata)

    def test_deepseek_support_routing_is_unchanged(self):
        import yaml
        config = yaml.safe_load((ROOT/'sim_eval/resources/protocols/tau_usi.json').read_text())
        resolved = apply_global_eval_model(config, 'Deepseek')
        self.assertEqual(resolved['fixed_assistant']['model'], 'replace-with-support-model')
        self.assertEqual(resolved['limits']['max_user_turns'], 60)
