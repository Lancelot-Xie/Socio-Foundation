import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from sim_eval.artifacts import CheckpointStore, latest_checkpoint_record_dicts
from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks import register_builtin_adapters
from sim_eval.contracts import ResultStatus
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.errors import ArtifactError, BackendError, ResumeConflictError
from sim_eval.judge_resume import (
    merge_judge_result, needs_judge_resume, validate_judge_append, validate_judge_roles,
)
from sim_eval.json_utils import canonical_json
from sim_eval.registry import get_adapter

ROOT = Path(__file__).resolve().parents[1]


class RecordingReplay(ReplayBackend):
    def __init__(self, responses, *, fail=None, judge_only=False):
        super().__init__(responses)
        self.requests = []
        self.fail = fail
        self.judge_only = judge_only

    def generate(self, request):
        self.requests.append(request)
        if self.judge_only and ':judge' not in request.request_id:
            raise AssertionError(f'resuming a rollout request: {request.request_id}')
        if self.fail and self.fail in request.request_id:
            raise BackendError('injected judge outage')
        return super().generate(request)


class JudgeResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_builtin_adapters()
        cls.fixtures = load_fixture_suite(ROOT / 'tests/fixtures')

    def run_case(self, adapter, case, backend, previous=None):
        kwargs = {} if previous is None else {'previous_result': previous}
        return adapter.execute_case(case, backend=backend, run_id='run', seed=7, model='fixture', **kwargs)

    def assert_preserved(self, old, new):
        for key in ('prediction', 'trace', 'model_response', 'token_usage', 'latency_ms'):
            self.assertEqual(canonical_json(getattr(old, key)), canonical_json(getattr(new, key)), key)
        metrics = {m.name: m for m in new.metrics}
        for m in old.metrics:
            if m.value is not None and not m.name.endswith('.availability_rate'):
                self.assertEqual(canonical_json(m), canonical_json(metrics[m.name]), m.name)
        validate_judge_append(old.to_dict(), new.to_dict())

    def test_each_protocol_resumes_only_failed_judge_with_identical_request(self):
        for name in ('sotopia', 'coser', 'humanual', 'mirrorbench', 'agentsense', 'userlm', 'behaviorchain'):
            adapter = get_adapter(name)
            for case in self.fixtures[name][0]:
                with self.subTest(benchmark=name, case=case.case_id):
                    responses = adapter.replay_responses(case, seed=7)
                    full_backend = RecordingReplay(responses)
                    expected = self.run_case(adapter, case, full_backend)
                    self.assertFalse(needs_judge_resume(adapter, case, expected))
                    judge_ids = [r.request_id for r in full_backend.requests if ':judge' in r.request_id]
                    if not judge_ids:
                        continue
                    missing_id = judge_ids[-1]
                    failed = self.run_case(adapter, case, RecordingReplay(responses, fail=missing_id))
                    self.assertEqual(failed.status, ResultStatus.COMPLETED)
                    self.assertTrue(needs_judge_resume(adapter, case, failed))
                    resume_backend = RecordingReplay(responses, judge_only=True)
                    with patch.object(adapter, '_task_verifier', side_effect=AssertionError('verifier reran'), create=True), patch.object(adapter, '_ai_detector_score', side_effect=AssertionError('detector reran'), create=True):
                        scored = self.run_case(adapter, case, resume_backend, failed)
                    repaired = merge_judge_result(failed, scored)
                    self.assertEqual([r.request_id for r in resume_backend.requests], [missing_id])
                    original = next(r for r in full_backend.requests if r.request_id == missing_id)
                    self.assertEqual(original.fingerprint, resume_backend.requests[0].fingerprint)
                    self.assert_preserved(failed, repaired)
                    self.assertFalse(needs_judge_resume(adapter, case, repaired))
                    self.assertEqual({m.name: m.value for m in repaired.metrics}, {m.name: m.value for m in expected.metrics})
                    records = latest_checkpoint_record_dicts([failed.to_dict(), {**repaired.to_dict(), 'checkpoint_attempt': 1}])
                    self.assertEqual(len(records), 1)

    def test_mirror_partial_samples_reused_and_failed_repair_stays_pending(self):
        adapter = get_adapter('mirrorbench')
        case = self.fixtures['mirrorbench'][0][0]
        responses = adapter.replay_responses(case, seed=7)
        missing_id = f'{case.case_id}:judge:pi:main:1'
        failed = self.run_case(adapter, case, RecordingReplay(responses, fail=missing_id))
        initial = deepcopy(failed.metadata['mirrorbench']['judge_records']['pi']['main']['samples'])
        self.assertEqual(len(initial), 1)
        again = self.run_case(adapter, case, RecordingReplay(responses, fail=missing_id, judge_only=True), failed)
        merged = merge_judge_result(failed, again)
        self.assertTrue(needs_judge_resume(adapter, case, merged))
        backend = RecordingReplay(responses, judge_only=True)
        repaired = merge_judge_result(merged, self.run_case(adapter, case, backend, merged))
        self.assertEqual([r.request_id for r in backend.requests], [missing_id, f'{case.case_id}:judge:pi:main:2'])
        self.assertEqual(repaired.metadata['mirrorbench']['judge_records']['pi']['main']['samples'][:1], initial)
        self.assertFalse(needs_judge_resume(adapter, case, repaired))
        self.assert_preserved(merged, repaired)

    def test_n_a_local_resource_nulls_and_target_failures_do_not_schedule_judges(self):
        for name in ('coser', 'humanual', 'userlm', 'behaviorchain'):
            adapter = get_adapter(name)
            for case in self.fixtures[name][0]:
                result = self.run_case(adapter, case, RecordingReplay(adapter.replay_responses(case, seed=7)))
                self.assertTrue(any(m.value is None for m in result.metrics))
                self.assertFalse(needs_judge_resume(adapter, case, result))
                nulls = replace(result, metrics=tuple(replace(m, value=None) for m in result.metrics), metadata={**result.metadata, 'target_output_failure': {'kind': 'invalid_output'}})
                self.assertFalse(needs_judge_resume(adapter, case, nulls))

    def test_completed_result_and_available_zero_are_immutable(self):
        adapter = get_adapter('sotopia')
        case = self.fixtures['sotopia'][0][0]
        responses = adapter.replay_responses(case, seed=7)
        failed = self.run_case(adapter, case, RecordingReplay(responses, fail=':judge'))
        failed = replace(failed, metrics=(replace(failed.metrics[0], value=0), *failed.metrics[1:]))
        repaired = merge_judge_result(failed, self.run_case(adapter, case, RecordingReplay(responses), failed))
        self.assertEqual(repaired.metrics[0].value, 0)
        self.assert_preserved(failed, repaired)
        for mutation in (replace(repaired, prediction={}), replace(repaired, metrics=tuple(replace(m, value=1) for m in repaired.metrics))):
            with self.assertRaises(ArtifactError):
                validate_judge_append(failed.to_dict(), mutation.to_dict())
        with self.assertRaises(ArtifactError):
            latest_checkpoint_record_dicts([failed.to_dict(), failed.to_dict()])
        with self.assertRaises(ArtifactError):
            latest_checkpoint_record_dicts([repaired.to_dict(), repaired.to_dict()])

    def test_saved_judge_evidence_can_restore_missing_metric_without_network(self):
        for name in ('sotopia', 'coser', 'humanual', 'mirrorbench', 'agentsense', 'userlm', 'behaviorchain'):
            adapter = get_adapter(name)
            from sim_eval.judge_resume import required_judge_metrics
            for case in self.fixtures[name][0]:
                responses = adapter.replay_responses(case, seed=7)
                result = self.run_case(adapter, case, RecordingReplay(responses))
                required = required_judge_metrics(adapter, case, result)
                if required:
                    break
            self.assertTrue(required, name)
            missing = replace(result, metrics=tuple(m for m in result.metrics if m.name != required[-1]))
            self.assertTrue(needs_judge_resume(adapter, case, missing))
            backend = RecordingReplay({}, judge_only=True)
            repaired = merge_judge_result(missing, self.run_case(adapter, case, backend, missing))
            self.assertFalse(needs_judge_resume(adapter, case, repaired))
            self.assertEqual(backend.requests, [], name)
            self.assert_preserved(missing, repaired)

    def test_official_agentsense_reuses_interviews_and_parallel_judge_successes(self):
        from sim_eval.benchmarks.agentsense import AgentSenseAdapter
        for workers in (1, 3):
            with self.subTest(workers=workers):
                base = self.fixtures['agentsense'][0][0]
                questions = {a: [{'self': [{'obj': a, 'question': 'Have you succeeded?'}],
                                  'others': [{'obj': b, 'question': 'Did the other succeed?'}],
                                  'judge': [{'obj': 'judge', 'question': 'Did the target succeed?'}]}]
                             for a, b in (('a', 'b'), ('b', 'a'))}
                case = replace(base, metadata={**base.metadata, 'goal_evaluation_questions': questions})
                adapter = AgentSenseAdapter(judge_max_workers=workers)
                responses = dict(adapter.replay_responses(case, seed=7))
                provenance = adapter.provenance_for_case(case)
                for a, b in (('a', 'b'), ('b', 'a')):
                    for dimension, evaluator in [('self', a), ('others', b), *(('judge', j) for j in provenance.logical_judge_ids)]:
                        responses[f'{case.case_id}:goal:{a}:0:{dimension}:0:{evaluator}'] = json.dumps({'reasoning': 'Supported by conversation.', 'answer': 'No'})
                failed_id = f'{case.case_id}:goal:b:0:judge:0:{provenance.logical_judge_ids[-1]}'
                failed = self.run_case(adapter, case, RecordingReplay(responses, fail=failed_id))
                self.assertTrue(needs_judge_resume(adapter, case, failed))
                backend = RecordingReplay(responses, judge_only=True)
                repaired = merge_judge_result(failed, self.run_case(adapter, case, backend, failed))
                self.assertEqual([r.request_id for r in backend.requests], [failed_id])
                self.assertFalse(needs_judge_resume(adapter, case, repaired))
                self.assert_preserved(failed, repaired)
                self.assertEqual(next(m.value for m in repaired.metrics if m.name == 'agentsense.episode.judge_average'), 0)
                exhausted = replace(failed, metadata={**failed.metadata, 'judge_error': {'kind': 'token_budget_exhausted'}})
                self.assertFalse(needs_judge_resume(adapter, case, exhausted))

    def test_parse_failure_and_legacy_records_without_new_evidence_resume(self):
        for name in ('sotopia', 'humanual', 'agentsense'):
            adapter = get_adapter(name); case = self.fixtures[name][0][0]
            responses = dict(adapter.replay_responses(case, seed=7))
            failed_id = next(k for k in responses if ':judge' in k)
            broken = {**responses, failed_id: 'invalid judge JSON'}
            failed = self.run_case(adapter, case, RecordingReplay(broken))
            metadata = {k: v for k, v in failed.metadata.items() if k not in {'judge_payload', 'goal_evaluation_cache'}}
            failed = replace(failed, metadata=metadata)
            self.assertTrue(needs_judge_resume(adapter, case, failed))
            backend = RecordingReplay(responses, judge_only=True)
            repaired = merge_judge_result(failed, self.run_case(adapter, case, backend, failed))
            self.assertFalse(needs_judge_resume(adapter, case, repaired))
            self.assert_preserved(failed, repaired)

    def test_repair_audit_does_not_repeat_original_fallback_or_historical_warning(self):
        from sim_eval.execution_provenance import summarize_support_role_provenance, summarize_evaluation_warnings
        adapter = get_adapter('sotopia'); case = self.fixtures['sotopia'][0][0]
        responses = adapter.replay_responses(case, seed=7)
        failed = self.run_case(adapter, case, RecordingReplay(responses, fail=':judge'))
        failed = replace(failed, metadata={**failed.metadata, 'execution_provenance': {
            'support_roles': {'judge': {'model': 'primary', 'model_revision': 'v1'}},
            'api_calls': [{'route_role': 'judge', 'safety_fallback': {'used': True, 'fallback': {'model': 'fallback', 'model_revision': 'v1'}}}]}})
        again = merge_judge_result(failed, failed)
        successful = self.run_case(adapter, case, RecordingReplay(responses), again)
        successful = replace(successful, metadata={**successful.metadata, 'request_audits': [{'kind': 'BackendError', 'message': 'historical retry recovered'}]})
        repaired = merge_judge_result(again, successful)
        self.assertEqual(summarize_support_role_provenance([failed, again, repaired])['safety_fallback_call_count'], 1)
        self.assertEqual(summarize_evaluation_warnings([repaired]), [])
        self.assert_preserved(again, repaired)

    def test_support_model_change_is_rejected_but_concurrency_change_is_allowed(self):
        adapter = get_adapter('sotopia'); case = self.fixtures['sotopia'][0][0]
        result = self.run_case(adapter, case, RecordingReplay(adapter.replay_responses(case, seed=7)))
        result = replace(result, metadata={**result.metadata, 'execution_provenance': {'support_roles': {'judge': {'model': 'old', 'max_inflight': 1}}}})
        validate_judge_roles(result, {'judge': {'model': 'old', 'max_inflight': 4}})
        with self.assertRaises(ResumeConflictError):
            validate_judge_roles(result, {'judge': {'model': 'new', 'max_inflight': 1}})


class JudgeResumeRunnerTests(unittest.TestCase):
    @staticmethod
    def role(model):
        return {'backend': 'openai_compatible', 'base_url': 'http://unused.test/v1',
                'api_key_env': 'UNUSED_TEST_KEY', 'require_api_key': False,
                'model': model, 'model_revision': 'test-v1',
                'generation': {'temperature': 0, 'max_tokens': 2048}}

    def test_generic_runner_appends_repairs_then_third_run_has_zero_calls(self):
        from sim_eval.generic_runner import run_generic_import
        from sim_eval.benchmarks.mirrorbench import MirrorBenchAdapter
        from sim_eval.data.loaders import load_import_spec, load_local_cases
        for workers in (1, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory(dir=ROOT) as tmp:
                directory = Path(tmp)
                manifest = directory / 'import.json'
                manifest.write_text(json.dumps({'benchmark_id': 'mirrorbench', 'path': str(ROOT / 'tests/fixtures/mirrorbench.jsonl'), 'format': 'jsonl', 'source_kind': 'synthetic_fixture', 'source_revision': 'resume-test-v1', 'split': 'fixture'}))
                config = directory / 'runtime.yaml'
                config.write_text(json.dumps({'schema_version': '1.0', 'benchmark_id': 'mirrorbench',
                    'execution': {'max_workers': workers, 'repetitions': 2},
                    'roles': {k: self.role(k) for k in ('evaluated_user', 'fixed_assistant', 'judge')},
                    'routing': {'default_role': 'evaluated_user'}, 'prompts': {'user_simulation': {'source': 'official_repository', 'revision': 'test-v1', 'locator': 'test-only'}}, 'environment': {'max_retries': 0}}))
                cases, _ = load_local_cases(load_import_spec(manifest))
                responses = {}
                for case in cases:
                    responses.update(MirrorBenchAdapter().replay_responses(case, seed=7))
                backend = RecordingReplay(responses, fail=f'{cases[0].case_id}:judge:pi:main:1')
                arguments = dict(manifest_path=manifest, runtime_config_path=config, output_directory=directory/'output', catalog_path=ROOT/'sim_eval/resources/benchmarks.json', seed=7)
                with patch('sim_eval.generic_runner.build_role_routed_backend', return_value=backend):
                    first = run_generic_import(**arguments)
                    self.assertEqual(first['failed_count'], 0)
                    self.assertEqual(first['status'], 'completed_with_failures')
                    count = first['result_count']
                    pending = 2  # Only one case's two repetitions need repair.
                    self.assertEqual(first['pending_judge_count'], pending)
                    backend.fail = None; backend.judge_only = True; backend.requests.clear()
                    second = run_generic_import(**arguments)
                    self.assertEqual(second['status'], 'completed')
                    self.assertEqual(second['judge_resume_count'], pending)
                    self.assertEqual(second['result_count'], count)
                    self.assertEqual(len(backend.requests), pending * 2)
                    backend.requests.clear()
                    third = run_generic_import(**arguments)
                    self.assertEqual(backend.requests, [])
                    self.assertEqual(third['judge_resume_count'], 0)
                    self.assertEqual(second['metrics'], third['metrics'])
                records = [json.loads(line) for line in (directory/'output'/third['artifacts']['records']).read_text().splitlines()]
                self.assertEqual(len(records), count + pending)
                self.assertEqual(len(latest_checkpoint_record_dicts(records)), count)
                for record in records[count:]:
                    self.assertEqual(record['checkpoint_attempt'], 1)
                    self.assertIn('mirrorbench.judge.pi', record['metadata']['judge_resume_history'][0]['updated_metrics'])

    def test_interrupted_checkpoint_append_recovers_and_preserves_completed_record(self):
        from tests.test_core_artifacts import make_manifest
        from sim_eval.contracts import CaseResult
        for tail in (b'{"partial', b''):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory(dir=ROOT) as tmp:
                store = CheckpointStore(tmp); manifest = make_manifest(); store.initialize(manifest)
                first = CaseResult(run_id=manifest.run_id, benchmark_id='fantom', case_id='first', group_id='group', repetition=0, status=ResultStatus.COMPLETED)
                store.append_result(first)
                original = store.records_path.read_bytes()
                store.records_path.write_bytes(original + tail if tail else original.rstrip(b'\n'))
                resumed = CheckpointStore(tmp); resumed.initialize(manifest)
                resumed.append_result(replace(first, case_id='second'))
                self.assertEqual(len(list(resumed.iter_latest_record_dicts())), 2)
                self.assertEqual(resumed.records_path.read_bytes()[:len(original)], original)


    def test_userlm_section3_runner_resumes_intent_judge_and_preserves_zero(self):
        from sim_eval.external_runner import run_userlm_import
        from sim_eval.benchmarks.userlm import UserLMAdapter
        from sim_eval.data.loaders import load_import_spec, load_local_cases
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            directory = Path(tmp)
            data = directory/'cases.jsonl'
            data.write_text(json.dumps({'source_id': 'nq-1', 'group_source_id': 'nq-1', 'split': 'fixture',
                'input': {'variant': 'intrinsic_intent_adherence', 'execution_mode': 'section3_single_user_turn',
                          'intent': 'Who wrote the blue comet song?', 'question': 'Who wrote the song?', 'assistant_suggestion_turn': 'Shall we discuss telescopes instead?', 'conversation_history': '<user>: Who wrote the song?\n<assistant>: Shall we discuss telescopes instead?\n', 'turn': 1},
                'gold': None, 'strata': {'variant': 'intrinsic_intent_adherence', 'source_task': 'nq', 'intent': 'song', 'required_information_pattern': 'not_applicable'}})+'\n')
            manifest = directory/'import.json'
            manifest.write_text(json.dumps({'benchmark_id': 'userlm', 'path': str(data), 'format': 'jsonl', 'source_kind': 'synthetic_fixture', 'source_revision': 'resume-nq-v1', 'split': 'fixture'}))
            config = directory/'runtime.yaml'
            config.write_text(json.dumps({'schema_version': '1.0', 'benchmark_id': 'userlm', 'execution': {'max_workers': 2},
                'roles': {k: self.role(k) for k in ('evaluated_user', 'fixed_assistant', 'intent_judge', 'shard_judge')},
                'routing': {'default_role': 'evaluated_user'}, 'prompts': {'user_simulation': {'source': 'official_repository', 'revision': 'test-v1', 'locator': 'test-only'}},
                'resources': {'lic_repetitions': 1, 'ai_text_detector': {'enabled': False}, 'code_execution': {'enabled': False, 'backend': 'local_guarded'}}}))
            case = load_local_cases(load_import_spec(manifest))[0][0]
            base = load_fixture_suite(ROOT / "tests/fixtures")["userlm"][0][0]
            adapter = UserLMAdapter(runtime_provenance=UserLMAdapter().runtime_for_case(base))
            output = 'Yes, let us talk about telescopes.'
            # Request IDs belong to the normal Section 3 prompt builders.
            user_id = adapter.build_section3_request(case, model='evaluated_user', seed=7).request_id
            prov = replace(UserLMAdapter().provenance_for_case(base), intent_judge_model='intent_judge', intent_judge_revision='test-v1')
            judge_id = adapter.build_section3_intent_judge_request(case, output, prov, seed=7).request_id
            backend = RecordingReplay({user_id: output, judge_id: 'ACCEPTED'}, fail=judge_id)
            arguments = dict(manifest_path=manifest, runtime_config_path=config, output_directory=directory/'output', catalog_path=ROOT/'sim_eval/resources/benchmarks.json', seed=7)
            with patch('sim_eval.external_runner.build_role_routed_backend', return_value=backend):
                first = run_userlm_import(**arguments)
                self.assertEqual(first['pending_judge_count'], 1)
                backend.fail = None; backend.judge_only = True; backend.requests.clear()
                second = run_userlm_import(**arguments)
                self.assertEqual(second['pending_judge_count'], 0)
                self.assertEqual([r.request_id for r in backend.requests], [judge_id])
                self.assertEqual(second['metrics']['userlm.intrinsic.intent_adherence']['value'], 0)
                backend.requests.clear()
                third = run_userlm_import(**arguments)
                self.assertEqual(backend.requests, [])
                self.assertEqual(third['metrics'], second['metrics'])


if __name__ == '__main__':
    unittest.main()
