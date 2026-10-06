import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from sim_eval.backends.openai_compatible import OpenAICompatibleBackend
from sim_eval.benchmarks.agentsense import AgentSenseAdapter
from sim_eval.contracts import ChatMessage, ModelRequest, ModelResponse
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.generation_policy import apply_benchmark_generation_policy
from sim_eval.runtime_config import (
    GLOBAL_EVAL_MODEL_CHOICES, build_role_routed_backend,
    load_benchmark_runtime_config, resolve_role_config,
)
from sim_eval.smoke import _smoke_runtime_document

ROOT = Path(__file__).resolve().parents[1]


class GenerationPolicyTests(unittest.TestCase):

    def test_legacy_zero_temperature_and_fallback_are_overridden_without_mutation(self):
        raw = {'benchmark_id': 'agentsense', 'roles': {
            'evaluated_actor': {'generation': {'temperature': 0}},
            'judge_1': {'generation': {'temperature': 0, 'max_tokens': 4096},
                        'fallback': {'generation': {'temperature': 0}}},
        }}
        result = apply_benchmark_generation_policy(raw)
        self.assertEqual(raw['roles']['judge_1']['generation']['temperature'], 0)
        self.assertEqual(result['roles']['evaluated_actor'], raw['roles']['evaluated_actor'])
        judge = result['roles']['judge_1']
        for role in (judge, judge['fallback']):
            self.assertEqual(role['generation']['temperature'], 0.8)
            self.assertIs(role['send_sampling_params'], True)
        self.assertEqual(judge['generation']['max_tokens'], 4096)
        raw['benchmark_id'] = 'sotopia'
        self.assertEqual(apply_benchmark_generation_policy(raw), raw)

    def test_direct_routes_emit_actual_sampling_and_thinking_payloads(self):
        for preset in (None, *GLOBAL_EVAL_MODEL_CHOICES):
            with self.subTest(preset=preset):
                config = load_benchmark_runtime_config(
                    ROOT / 'sim_eval/resources/protocols/agentsense.json', global_eval_model=preset,
                    evaluate_overrides={'model': 'candidate-test', 'base_url': 'http://localhost:6666/v1'},
                )
                payloads = []

                class CapturePayload(OpenAICompatibleBackend):
                    def generate(self, request):
                        payloads.append(self.build_payload(request))
                        return ModelResponse(text='Yes')

                with patch('sim_eval.runtime_config.get_backend', side_effect=lambda name, **kwargs: CapturePayload(**kwargs)):
                    router = build_role_routed_backend(config)
                for name, role in config['roles'].items():
                    router.generate(ModelRequest(
                        messages=[ChatMessage('user', 'test')], model='adapter-model', temperature=0,
                        metadata={'route_role': name},
                    ))
                    self.assertEqual(payloads[-1]['temperature'], 0 if name == 'evaluated_actor' else 0.8)
                    if name == 'evaluated_actor':
                        self.assertFalse(payloads[-1]['chat_template_kwargs']['enable_thinking'])
                        self.assertEqual(payloads[-1]['model'], 'candidate-test')
                    fallback = role.get('fallback')
                    if fallback:
                        self.assertEqual(fallback['generation']['temperature'], 0.8)
                        self.assertTrue(fallback['send_sampling_params'])

    def test_suite_shared_zero_generation_cannot_erase_judge_exception(self):
        for preset in (None, *GLOBAL_EVAL_MODEL_CHOICES):
            with self.subTest(preset=preset):
                document = _smoke_runtime_document(
                    ROOT / 'sim_eval/resources/protocols/agentsense.json', benchmark_id='agentsense',
                    suite_id='test', model={
                        'backend': 'vllm', 'profile': 'vllm', 'model': 'example-model',
                        'model_revision': 'test', 'generation': {'temperature': 0},
                        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}},
                    }, execution_options={'local_device': 'cpu', 'max_workers': 8},
                    global_eval_model=preset,
                )
                for name, raw in document['roles'].items():
                    role = resolve_role_config(raw)
                    self.assertEqual(role['generation']['temperature'], 0 if name == 'evaluated_actor' else 0.8)
                    if name != 'evaluated_actor':
                        self.assertTrue(role['send_sampling_params'])
                self.assertEqual(document['execution']['max_workers'], 8)

    def test_adapter_official_and_legacy_judges_use_exception_only_for_judges(self):
        case = load_fixture_suite(ROOT / 'tests/fixtures')['agentsense'][0][0]
        adapter = AgentSenseAdapter()
        state = adapter.environment.reset(case, seed=7)
        for dimension, evaluator, role in [('self', 'a', 'evaluated_actor'),
                                          ('others', 'b', 'evaluated_actor'),
                                          ('judge', 'judge_1', 'judge_1')]:
            request = adapter.build_goal_evaluation_request(
                case, state, target_agent_id='a', goal_index=0, dimension=dimension,
                question_index=0, evaluator_id=evaluator, question='Achieved?',
                model='same-model', seed=7, route_role=role,
            )
            self.assertEqual(request.temperature, 0.8 if dimension == 'judge' else 0)
        provenance = adapter.provenance_for_case(case)
        seeds = set()
        for i, judge_id in enumerate(provenance.logical_judge_ids):
            request = adapter.build_judge_request(
                case, state, provenance, judge_model='same-model',
                judge_id=judge_id, judge_index=i, seed=7,
            )
            self.assertEqual(request.temperature, 0.8)
            seeds.add(request.seed)
        self.assertEqual(len(seeds), 3)
