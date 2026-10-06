import copy
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.benchmarks.sotopia import SotopiaAdapter, SotopiaScorer
from sim_eval.contracts import CaseResult, ErrorState, ResultStatus
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.metric_markdown import metric_coverage

ROOT = Path(__file__).resolve().parents[1]


class SotopiaAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_fixture_suite(ROOT / 'tests/fixtures')['sotopia'][0][0]
        cls.adapter = SotopiaAdapter()
        cls.provenance = cls.adapter.provenance_for_case(cls.case)

    def result(self, evaluated, a, b, *, group='config', legacy=False):
        payload = copy.deepcopy(self.case.metadata['replay']['judge'])
        payload['scores']['a']['goal']['score'] = a
        payload['scores']['b']['goal']['score'] = b
        partner = 'b' if evaluated == 'a' else 'a'
        metrics = SotopiaScorer().score_payload(payload, agent_ids=('a', 'b'),
            provenance=self.provenance, evaluated_agent_id=evaluated, partner_agent_id=partner)
        subject = dict(evaluated_agent_id=evaluated, partner_agent_id=partner, position_designated=True)
        if not legacy:
            subject['evaluated_role_position'] = 'agent1' if evaluated == 'a' else 'agent2'
        else:
            metrics = tuple(replace(m, metadata={k: v for k, v in m.metadata.items()
                if k not in {'agent_id', 'agent_role'}}) if m.name.endswith('.normalized') else m for m in metrics)
        return CaseResult(run_id='test', benchmark_id='sotopia', case_id=f'{group}-{evaluated}',
            group_id=group, repetition=0, status=ResultStatus.COMPLETED, metrics=metrics,
            metadata={'formal_score_subject': subject, 'judge_status': 'available'})

    def test_opposite_roles_preserve_main_scores_and_separate_persona_slices(self):
        results = [self.result('a', 2, 8), self.result('b', 10, 4)]
        before = copy.deepcopy(results)
        aggregate = self.adapter.aggregate(results)
        self.assertEqual(results, before)
        expected = {'sotopia.evaluated_agent.goal': 3, 'sotopia.partner_agent.goal': 9,
                    'sotopia.configuration_mean.goal': 3,
                    'sotopia.diagnostic.mean_across_agents.goal': 6,
                    'sotopia.evaluated_agent_by_id.a.goal': 2,
                    'sotopia.partner_agent_by_id.a.goal': 10,
                    'sotopia.evaluated_agent_by_id.b.goal': 4,
                    'sotopia.partner_agent_by_id.b.goal': 8,
                    'sotopia.evaluated_agent_by_id.a.goal.normalized': .2,
                    'sotopia.partner_agent_by_id.a.goal.normalized': 1.0,
                    'sotopia.complete_role_pair_rate': 1.0}
        for name, value in expected.items():
            self.assertEqual(aggregate[name].value, value, name)
        self.assertFalse(any(name.startswith('sotopia.agent.') for name in aggregate))
        self.assertEqual(aggregate['sotopia.configuration_mean.goal'].denominator, 1)

    def test_four_one_sided_groups_are_excluded_without_changing_complete_group_scores(self):
        results = [self.result('a', 2, 8), self.result('b', 10, 4)]
        for index in range(4):
            first = self.result('a', 10, 10, group=f'partial-{index}')
            second = self.result('b', 10, 10, group=f'partial-{index}')
            second = replace(second, metrics=tuple(replace(m, value=None) for m in second.metrics),
                             metadata={**second.metadata, 'judge_status': 'unavailable'})
            results.extend((first, second))
        aggregate = self.adapter.aggregate(results)
        goal = aggregate['sotopia.configuration_mean.goal']
        self.assertEqual(goal.value, 3)
        self.assertEqual(goal.denominator, 1)
        self.assertEqual(goal.metadata['configuration_count'], 5)
        self.assertEqual(goal.metadata['available_configuration_count'], 1)
        self.assertEqual(goal.metadata['incomplete_or_unavailable_configuration_count'], 4)
        self.assertEqual(aggregate['sotopia.complete_role_pair_rate'].value, .2)
        partial = self.adapter.aggregate(results[2:])['sotopia.configuration_mean.goal']
        self.assertIsNone(partial.value)
        self.assertEqual(partial.denominator, 0)
        self.assertEqual(partial.metadata['availability'], 'unavailable_no_complete_role_pairs')
        coverage = metric_coverage(goal.name, {'value': goal.value, 'metadata': goal.metadata},
            record_counts={}, total_records=10, completed_records=10)
        self.assertEqual((coverage.valid_count, coverage.applicable_count), (1, 5))

    def test_invalid_pair_identities_duplicates_failed_and_nonfinite_values_are_excluded(self):
        first, second = self.result('a', 2, 8), self.result('b', 10, 4)
        bad_pairs = (
            [first], [first, first], [first, replace(first, case_id='duplicate-position')],
            [first, replace(second, repetition=1)],
            [first, replace(second, metadata={})],
            [first, replace(second, metadata={'formal_score_subject': 'invalid'})],
            [first, replace(second, metadata={'formal_score_subject': {
                'evaluated_agent_id': 'b', 'partner_agent_id': 'other', 'evaluated_role_position': 'agent2'}})],
            [first, replace(second, metadata={'formal_score_subject': {
                'evaluated_agent_id': 'b', 'partner_agent_id': 'a', 'evaluated_role_position': 'agent1'}})],
            [first, replace(second, status=ResultStatus.FAILED, error=ErrorState('test', 'test', 'failed'))],
            [first, replace(second, metrics=tuple(replace(m, value=float('nan'))
                if m.name == 'sotopia.evaluated_agent.goal' else m for m in second.metrics))],
        )
        for index, results in enumerate(bad_pairs):
            with self.subTest(index=index):
                aggregate = self.adapter.aggregate(results)
                self.assertIsNone(aggregate['sotopia.configuration_mean.goal'].value)
                self.assertEqual(aggregate['sotopia.complete_role_pair_rate'].value, 0)

    def test_valid_zero_and_legacy_opposite_identity_pairs_remain_valid(self):
        results = [self.result('a', 0, 8, legacy=True), self.result('b', 10, 0, legacy=True)]
        aggregate = self.adapter.aggregate(results)
        self.assertEqual(aggregate['sotopia.configuration_mean.goal'].value, 0)
        self.assertEqual(aggregate['sotopia.configuration_mean.goal'].denominator, 1)
        self.assertEqual(aggregate['sotopia.complete_role_pair_rate'].value, 1)
        self.assertEqual(aggregate['sotopia.evaluated_agent_by_id.a.goal.normalized'].value, 0)
        self.assertEqual(aggregate['sotopia.partner_agent_by_id.a.goal.normalized'].value, 1)

    def test_missing_one_dimension_does_not_erase_other_valid_paired_dimensions(self):
        first, second = self.result('a', 2, 8), self.result('b', 10, 4)
        second = replace(second, metrics=tuple(replace(m, value=None)
            if m.name == 'sotopia.evaluated_agent.goal' else m for m in second.metrics))
        aggregate = self.adapter.aggregate([first, second])
        self.assertIsNone(aggregate['sotopia.configuration_mean.goal'].value)
        self.assertIsNotNone(aggregate['sotopia.configuration_mean.believability'].value)
        self.assertEqual(aggregate['sotopia.complete_role_pair_rate'].value, 0)


if __name__ == '__main__':
    unittest.main()
