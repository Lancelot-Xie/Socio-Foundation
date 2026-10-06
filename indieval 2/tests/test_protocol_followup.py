import copy
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.backends.openai_compatible import OpenAICompatibleBackend
from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.alignx import AlignXAdapter
from sim_eval.benchmarks.behaviorchain import BehaviorChainAdapter
from sim_eval.benchmarks.behaviorchain_names import name_aliases_for_case, pseudonymize_material
from sim_eval.benchmarks.userlm import UserLMAdapter
from sim_eval.contracts import ModelResponse
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.errors import EpisodeTokenBudgetExceeded, ValidationError

ROOT = Path(__file__).resolve().parents[1]
from helpers.alignx_material import REVISION, project_signals, restore_row


class ProtocolFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_fixture_suite(ROOT/'tests/fixtures')

    def named_case(self):
        case = self.fixtures['behaviorchain'][0][0]
        return replace(case, input_data={**case.input_data,
            'persona': {'Name': 'Ann Stone', 'Relationships': {'Anna Stone': 'sister', 'Mr. Alex Reed': 'friend', 'Family': 'relatives'}},
            'history': ['Ann met Anna at an annual event. Will they return?'],
            'current_context': "Ann discusses Anna's tools with Mr. Alex Reed.",
            'candidates': ['Ann labels the tools', 'Anna loses the tools', 'Ann lends tools to Alex', 'Alex discards them']})

    def test_names_are_chain_stable_across_nodes_models_and_rollout_seeds(self):
        adapter = BehaviorChainAdapter()
        case = self.named_case()
        aliases = name_aliases_for_case(case)
        next_case = replace(case, case_id=case.case_id+'next', input_data={**case.input_data,
            'chain_index': 1, 'prior_nodes': [{'context': case.input_data['current_context'],
                                            'behavior': case.input_data['candidates'][case.gold]}],
            'current_context': 'Ann leaves.', 'candidates': ['future one', 'future two', 'future three', 'future four']})
        first = adapter.build_request(case, model='model-one', seed=111)
        second = adapter.build_request(next_case, model='model-two', seed=999)
        self.assertEqual(first.metadata['name_aliases'], second.metadata['name_aliases'])
        self.assertEqual(aliases, name_aliases_for_case(next_case))
        self.assertNotEqual(aliases, name_aliases_for_case(replace(case, group_id='another-chain')))

    def test_name_rewrite_preserves_word_boundaries_roles_possessives_and_source(self):
        case = self.named_case()
        original = copy.deepcopy(case.input_data)
        aliases = name_aliases_for_case(case)
        self.assertFalse(set(aliases.values()) & set(aliases))
        self.assertEqual(len(set(aliases.values())), len(aliases))
        transformed = pseudonymize_material(case.input_data, aliases)
        self.assertIn('annual event. Will they return?', transformed['history'][0])
        self.assertIn(f"{aliases['Anna']}'s tools", transformed['current_context'])
        self.assertIn('Mr. ', transformed['current_context'])
        self.assertIn('Family', transformed['persona']['Relationships'])
        self.assertEqual(transformed['persona']['Name'], f"{aliases['Ann']} {aliases['Stone']}")
        self.assertIn(f"{aliases['Anna']} {aliases['Stone']}", transformed['persona']['Relationships'])
        restored = pseudonymize_material(transformed, {v:k for k,v in aliases.items()})
        self.assertEqual(restored, original)
        self.assertEqual(case.input_data, original)

    def test_single_pass_has_no_cascade_or_option_id_replacement(self):
        self.assertEqual(pseudonymize_material('Ann Anna Annual', {'Ann':'Anna', 'Anna':'Beth'}), 'Anna Beth Annual')
        self.assertEqual(pseudonymize_material({'id':'Ann', 'text':'Ann'}, {'Ann':'Beth'}), {'id':'Ann', 'text':'Beth'})

    def test_actor_wire_does_not_expose_name_mapping_and_gold_choice_is_unchanged(self):
        adapter = BehaviorChainAdapter()
        case = self.named_case()
        request = adapter.build_request(case, model='audit', seed=17)
        payload = OpenAICompatibleBackend(base_url='http://fixture/v1').build_payload(request)
        wire = json.dumps(payload)
        self.assertNotIn('name_aliases', wire)
        self.assertNotIn('Ann Stone', wire)
        for index, label in enumerate('ABCD'):
            prediction = adapter.parse_response(case, ModelResponse(text=f'<answer>{label}</answer>'))
            self.assertEqual(prediction.source_id, str(index))
        result = adapter.execute_case(case, backend=ReplayBackend(adapter.replay_responses(case, seed=17)),
                                      run_id='test', seed=17, model='audit')
        self.assertEqual(result.status.value, 'completed')
        self.assertEqual(next(m.value for m in result.metrics if m.name=='behaviorchain.node_score'), 1)
        self.assertEqual(result.metadata['behaviorchain']['name_aliases'], request.metadata['name_aliases'])

    def test_generation_judge_uses_same_pseudonyms_without_transforming_model_output_again(self):
        adapter = BehaviorChainAdapter()
        case = self.named_case()
        aliases = name_aliases_for_case(case)
        behavior = aliases['Ann']+' labels the tools'
        request = adapter.build_judge_request(case, behavior, adapter.judge_provenance_for_case(case), seed=18)
        payload = json.loads(request.messages[-1].content)
        self.assertEqual(payload['persona']['Name'], f"{aliases['Ann']} {aliases['Stone']}")
        self.assertEqual(payload['generated_next_behavior'], behavior)
        self.assertNotIn('gold', payload)

    def test_description_only_profiles_are_not_guessed(self):
        case = self.fixtures['behaviorchain'][0][0]
        self.assertEqual(name_aliases_for_case(case), {})
        self.assertEqual(pseudonymize_material(case.input_data, {}), case.input_data)

    def test_behaviorchain_budget_failure_remains_unscored(self):
        class NoRoom:
            def generate(self, request):
                raise EpisodeTokenBudgetExceeded('no room', scope='model_context',
                    episode_remaining_tokens=100, context_remaining_tokens=0, prompt_tokens=40000)
        result = BehaviorChainAdapter().execute_case(self.named_case(), backend=NoRoom(), run_id='test', seed=17, model='audit')
        self.assertEqual(result.status.value, 'failed')
        self.assertEqual(result.error.kind, 'token_budget_exhausted')
        self.assertEqual(result.metadata['token_budget']['prompt_tokens'], 40000)
        self.assertFalse(result.metrics)

    def source_and_row(self):
        record = {'prompt':'TARGET', 'chosen':'GOOD', 'rejected':'BAD', 'profile':'SUMMARY',
                  'Demographic Information':'DEMO',
                  'Pair-wise Comparative Feedback':[{'prompt':'PAIR_POST','chosen':'PAIR_GOOD','rejected':'PAIR_BAD','Preference Direction':[1,0]}],
                  'User-Generated Content':[{'prompt':'UGC_POST','comment':'UGC_TEXT','Preference Direction':[0,1]}]}
        source = {'eval_id':'row', 'eval_core_id':'core', 'core_sha256':'sha', 'condition':'Reddit_arbitrary', 'record':record}
        row = {'source_id':'row','group_source_id':'core','metadata':{'core_sha256':'sha'},'gold':'chosen',
               'input':{'variant':'Reddit_arbitrary','prompt':'TARGET','chosen':'GOOD','rejected':'BAD','persona_components':'SUMMARY'}}
        return source, row

    def test_arbitrary_restores_only_frozen_signals_and_is_idempotent(self):
        source, row = self.source_and_row()
        original = copy.deepcopy((source,row))
        restored = restore_row(row,source)
        self.assertEqual(restored['input']['persona_components'], {
            'demographic_information':'DEMO',
            'pairwise_feedback':[{'prompt':'PAIR_POST','chosen':'PAIR_GOOD','rejected':'PAIR_BAD'}],
            'user_generated_content':[{'prompt':'UGC_POST','comment':'UGC_TEXT'}]})
        self.assertEqual(restore_row(restored,source), restored)
        for field in ['source_id','group_source_id','gold','metadata']:
            self.assertEqual(restored[field], row[field])
        self.assertEqual((source,row), original)
        self.assertNotIn('Preference Direction', json.dumps(restored))

    def test_arbitrary_mismatched_core_and_empty_signals_fail_closed(self):
        source,row = self.source_and_row()
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            restore_row({**row,'group_source_id':'wrong'},source)
        with self.assertRaisesRegex(ValueError,'no supplied signals'):
            project_signals({**source['record'],'Demographic Information':'',
                             'Pair-wise Comparative Feedback':[],'User-Generated Content':[]})

    def test_arbitrary_api_input_uses_mixture_without_summary_and_keeps_assignment(self):
        adapter = AlignXAdapter()
        base = self.fixtures['alignx'][0][0]
        source,row = self.source_and_row()
        restored = restore_row(row,source)
        case = replace(base, input_data=restored['input'],
                       metadata={**base.metadata,'strata':{**base.metadata['strata'],'test_variant':'Reddit_arbitrary'}})
        request = adapter.build_request(case, model='audit', seed=17)
        text = '\n'.join(m.content for m in request.messages)
        for part in ['DEMO','PAIR_POST','PAIR_GOOD','PAIR_BAD','UGC_POST','UGC_TEXT']:
            self.assertIn(part,text)
        self.assertNotIn('SUMMARY',text)
        self.assertNotIn('Preference Direction',text)
        old = replace(case,input_data=row['input'])
        self.assertEqual(adapter.choices_for_case(old,seed=17),adapter.choices_for_case(case,seed=17))
        with self.assertRaises(ValidationError):
            adapter.build_request(replace(case,input_data={**case.input_data,'persona_components':{'profile':'SUMMARY'}}),model='audit',seed=17)

    def test_opaque_string_shard_cannot_sneak_through_missing_text_guard(self):
        case = self.fixtures['userlm'][0][0]
        case = replace(case,input_data={**case.input_data,'information_shards':['shard_1']})
        with self.assertRaisesRegex(ValidationError,'requires its text'):
            UserLMAdapter().dialogue_spec_for_case(case)


if __name__ == '__main__':
    unittest.main()
