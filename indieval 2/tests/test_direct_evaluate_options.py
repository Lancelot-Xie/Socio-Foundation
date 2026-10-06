import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_eval.errors import ConfigurationError
from sim_eval.runtime_config import apply_evaluate_overrides, load_benchmark_runtime_config
from sim_eval.external_runner import _load_runtime_config
from tests.test_cli_reporting_delivery import invoke_cli

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT.parent


class DirectEvaluateOptionsTests(unittest.TestCase):
    def test_direct_model_preserves_support_scoring_sampling_and_tokenizer(self):
        args = dict(global_eval_model='Deepseek', evaluated_model_config=ROOT/'tests/model_override.json')
        baseline = load_benchmark_runtime_config(ROOT/'sim_eval/resources/protocols/userlm.json', **args)
        current = load_benchmark_runtime_config(ROOT/'sim_eval/resources/protocols/userlm.json', **args,
            evaluate_overrides={'model': 'new-lora', 'base_url': 'http://localhost:6666/v1', 'max_workers': 16})
        for role in ('fixed_assistant', 'intent_judge', 'shard_judge'):
            self.assertEqual(baseline['roles'][role], current['roles'][role])
        self.assertEqual(current['roles']['evaluated_user']['model'], 'new-lora')
        self.assertEqual(current['roles']['evaluated_user']['model_revision'], 'new-lora')
        self.assertEqual(current['roles']['evaluated_user']['base_url'], 'http://localhost:6666/v1')
        self.assertEqual(current['execution']['max_workers'], 16)
        for key in ('generation', 'token_accounting'):
            self.assertEqual(baseline['roles']['evaluated_user'][key], current['roles']['evaluated_user'][key])
        self.assertEqual(current['resources'], baseline['resources'])
        self.assertNotIn('max_tokens', current['roles']['evaluated_user']['generation'])

    def test_no_override_explicit_revision_and_invalid_overrides(self):
        config = {'benchmark_id': 'humanual', 'roles': {'evaluated_model': {'model': 'old', 'model_revision': 'old-revision'}}}
        original = copy.deepcopy(config)
        self.assertEqual(apply_evaluate_overrides(config, None), config)
        changed = apply_evaluate_overrides(config, {'model': 'alias', 'model_revision': 'checkpoint-sha'})
        self.assertEqual(changed['roles']['evaluated_model']['model_revision'], 'checkpoint-sha')
        self.assertEqual(config, original)
        for invalid in (0, -1, True, '16'):
            with self.assertRaises(ConfigurationError):
                apply_evaluate_overrides(config, {'max_workers': invalid})
        with self.assertRaises(ConfigurationError):
            apply_evaluate_overrides(config, {'generation': {}})

    def test_tau_uses_existing_limits_and_target_schema(self):
        path = ROOT/'sim_eval/resources/protocols/tau_usi.json'
        _, original = _load_runtime_config(path, global_eval_model='Deepseek')
        _, changed = _load_runtime_config(path, global_eval_model='Deepseek',
            evaluate_overrides={'model': 'tau-lora', 'base_url': 'http://localhost:6666/v1', 'max_workers': 16})
        self.assertEqual(changed['target_user']['model'], 'tau-lora')
        self.assertEqual(changed['limits']['max_workers'], 16)
        self.assertEqual(changed['fixed_assistant'], original['fixed_assistant'])
        self.assertEqual(changed['scoring'], original['scoring'])

    def test_cli_passes_options_to_both_runners(self):
        for benchmark, target in [('social_r1', 'run_generic_import'), ('userlm', 'run_userlm_import')]:
            manifest = BUNDLE/'local_data/derived'/('eval_collection_v1/social_r1/import_manifest.json' if benchmark=='social_r1' else 'userlm_eval_v2/lic100/import_manifest.json')
            if not manifest.exists(): self.skipTest('bundled manifests are not installed')
            with patch('sim_eval.cli.'+target, return_value={'status': 'valid'}) as runner:
                code, _, err = invoke_cli(['evaluate', '--benchmark', benchmark, '--manifest', str(manifest),
                    '--runtime-config', str(ROOT/'sim_eval/resources/runtime'/f'{benchmark}.yaml'), '--output', str(ROOT/'artifacts/unused'),
                    '--model', 'new-lora', '--base-url', 'http://localhost:6666/v1', '--max-workers', '16', '--validate-only'])
                self.assertEqual(code, 0, err)
                self.assertEqual(runner.call_args.kwargs['evaluate_overrides'], {'model': 'new-lora', 'base_url': 'http://localhost:6666/v1', 'max_workers': 16})

    def test_humanual_domain_selects_one_hundred_without_new_manifest(self):
        manifest=BUNDLE/'local_data/derived/eval_scale_variants_v1/humanual_100/import_manifest.json'
        if not manifest.exists(): self.skipTest('bundled HUMANUAL collection is not installed')
        for domain in ('book', 'chat', 'opinion', 'politics'):
            code, stdout, stderr=invoke_cli(['evaluate', '--benchmark', 'humanual', '--manifest', str(manifest),
                '--runtime-config', str(ROOT/'sim_eval/resources/protocols/humanual.json'), '--catalog', str(ROOT/'sim_eval/resources/benchmarks.json'),
                '--output', str(ROOT/'artifacts/unused'), '--model', 'domain-'+domain, '--base-url', 'http://localhost:6666/v1',
                '--global-eval-model', 'Deepseek', '--max-workers', '16', '--humanual-domain', domain, '--validate-only'])
            self.assertEqual(code,0,stderr)
            payload=json.loads(stdout)
            self.assertEqual(payload['validated_case_count'],100)
            self.assertEqual(payload['max_workers'],16)
            self.assertEqual(payload['network_calls'],0)

    def test_domain_flag_rejects_other_benchmark(self):
        manifest=BUNDLE/'local_data/derived/eval_collection_v1/social_r1/import_manifest.json'
        if not manifest.exists(): self.skipTest('bundled manifest is not installed')
        code, _, err=invoke_cli(['evaluate', '--benchmark','social_r1','--manifest',str(manifest),
            '--runtime-config',str(ROOT/'sim_eval/resources/protocols/social_r1.json'),'--output',str(ROOT/'artifacts/unused'),
            '--humanual-domain','book','--validate-only'])
        self.assertNotEqual(code,0)
        self.assertIn('--humanual-domain',err)
