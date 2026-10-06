"""Prepare, train, evaluate and export the three-model ablation matrix."""
import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

from opd.checkpoints import atomic_json
from opd.config import load_config
from opd.data import load_data, write_rows
from opd.experiments import Runner

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = str(ROOT / 'outputs/hierarchy')
MODEL_ROOT = str(ROOT / 'models')
MODELS = {'qwen3_4b': 'Qwen3-4B', 'llama3_8b': 'Meta-Llama-3-8B-Instruct', 'qwen3_14b': 'Qwen3-14B'}
DIMS = tuple('FSUTN')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_identity(path):
    path = Path(path)
    if not (path / 'config.json').is_file() and not (path / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Model/adapter config missing: {path}')
    weights = list(path.glob('*.safetensors')) + list(path.glob('pytorch_model*.bin')) + list(path.glob('adapter_model.bin'))
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise FileNotFoundError(f'Nonempty weights missing: {path}')
    for index in path.glob('*.index.json'):
        for shard in set(json.loads(index.read_text())['weight_map'].values()):
            if not (path / shard).is_file():
                raise FileNotFoundError(path / shard)
    return {'path': str(path), 'small_files_sha256': {p.name: digest(p) for p in sorted(path.glob('*.json'))},
            'weights_size_mtime': {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(weights)}}


def read_source(args):
    source = args.reuse_from.resolve()
    if not (source / 'hierarchy_manifest.json').is_file():
        raise FileNotFoundError(source / 'hierarchy_manifest.json')
    base = load_config(source / 'configs/preflight.yaml')
    stages = json.loads((source / 'stages.json').read_text())
    teachers = []
    for dim in DIMS:
        stage = f'seed_{args.source_seed}/dimension_{dim}/offline_forward'
        if stages.get(stage, {}).get('status') != 'complete':
            raise ValueError(f'Unfinished expert: {stage}')
        path = source / 'runs' / stage / 'final'
        metadata = json.loads((path / 'opd_metadata.json').read_text())
        if metadata.get('dimension') != dim or metadata.get('student_mode') != 'lora':
            raise ValueError(f'Wrong expert metadata: {path}')
        teachers.append({'id': dim, 'adapter': str(path)})
    return base, teachers, source / 'hierarchy_data' / f'seed_{args.source_seed}'


def select_common(rows, tokenizers, max_prompt, eval_limit=0):
    from .text import prompt_ids
    kept, excluded, counts = [], [], Counter()
    for row in rows:
        lengths = {name: len(prompt_ids(tok, row['messages'], kwargs)) for name, tok, kwargs in tokenizers}
        if max(lengths.values()) > max_prompt:
            excluded.append({'id': row['id'], 'reason': 'overlong', 'lengths': lengths})
        elif not eval_limit or counts[row['task_id']] < eval_limit:
            kept.append(row)
            counts[row['task_id']] += 1
    missing = {r['task_id'] for r in rows} - set(counts)
    if missing:
        raise ValueError(f'Common-tokenizer filtering removed all rows for tasks: {sorted(missing)}')
    return kept, excluded


def split_guard(train, validation):
    def prompt_hash(row):
        return hashlib.sha256(json.dumps(row['messages'], sort_keys=True).encode()).hexdigest()
    ids = {r['id'] for r in validation}
    groups = {(r['task_id'], r['group_id']) for r in validation if r.get('group_id')}
    prompts = {prompt_hash(r) for r in validation}
    kept, removed = [], []
    for row in train:
        overlap = (row['id'] in ids or (row['task_id'], row.get('group_id')) in groups or prompt_hash(row) in prompts)
        (removed if overlap else kept).append(row)
    absent = {r['task_id'] for r in train} - {r['task_id'] for r in kept}
    if absent:
        raise ValueError(f'Leakage exclusion emptied train tasks {sorted(absent)}; supply a disjoint source split')
    return kept, [r['id'] for r in removed]


def evaluation_config(base, model, data, args, seed):
    cfg = copy.deepcopy(base)
    cfg['seed'] = seed
    cfg['model'].update(base_model=model, student_mode='full', dtype=args.dtype,
                        chat_template_kwargs={'enable_thinking': False})
    for key in ('tokenizer', 'student_init'):
        cfg['model'].pop(key, None)
    cfg['data'].update(train_file=str(data / 'train.jsonl'), eval_file=str(data / 'validation.jsonl'),
                       dimension=None, task_weights={}, demo_file=None)
    cfg['data'].pop('calibration_file', None)
    cfg['teachers'] = []
    cfg['routing'] = {'tasks': {}}
    cfg['qgpi'].update(enabled=False, registry_file=None)
    cfg['train'].update(stage='opd', objective='forward_kl', trajectory_source='student',
                        resume_from=None, sft_coef=0., anchor_coef=0., eval_every=0)
    cfg['rollout'].update(context_length=args.context_length, max_prompt_tokens=args.max_prompt_tokens,
                          max_new_tokens=args.teacher_tokens, generation_batch_size=1,
                          temperature=1., max_turns=1, environment_factory=None)
    cfg['inference']['batch_size'] = 1
    cfg['output_dir'] = str(data.parent / 'unused_eval')
    return cfg


def execute(args):
    if args.num_processes < 1 or args.micro_batch < 1 or args.global_batch < 1 or args.global_batch % (args.num_processes * args.micro_batch):
        raise ValueError('global-batch must be divisible by num-processes * micro-batch')
    if min(args.steps, args.save_every, args.teacher_tokens, args.max_prompt_tokens) < 1 or args.prefix_tokens < 0:
        raise ValueError('Positive steps/save/token budgets required; prefix-tokens >= 0')
    if args.max_prompt_tokens + args.prefix_tokens + args.teacher_tokens > args.context_length:
        raise ValueError('prompt + prefix + teacher token budgets must fit context-length')
    if args.learning_rate <= 0 or args.eval_limit < 0:
        raise ValueError('learning-rate > 0 and eval-limit >= 0 required')
    if len(set(args.students)) != len(args.students) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate students/seeds')
    root = args.output_root.resolve()
    source = args.reuse_from.resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('Output must be separate from the expert source')
    base, teachers, source_data = read_source(args)
    teacher_base = str(Path(args.teacher_base or base['model']['base_model']).resolve())
    models = {name: str((args.model_root / MODELS[name]).resolve()) for name in args.students}
    from .assets import resolve_teacher_tokenizer, student_tokenizers, check_tokenizers
    print(f'[source] hierarchy_root={source}', flush=True)
    print(f'[teacher] base_weights={teacher_base}', flush=True)
    for teacher in teachers:
        print(f"[teacher] dimension={teacher['id']} adapter={teacher['adapter']}", flush=True)
    for name, path in models.items():
        print(f'[student] name={name} weights={path}', flush=True)
    if Path(teacher_base) == source or (Path(teacher_base) / 'adapter_config.json').is_file():
        raise ValueError('--teacher-base must point to original FULL teacher backbone weights, '
                         'not the hierarchy root or a dimension LoRA. Experts are loaded separately.')
    teacher_tokenizer = resolve_teacher_tokenizer(teachers, base, teacher_base, args.teacher_tokenizer)
    tokenizer_paths = {'teacher': teacher_tokenizer, **student_tokenizers(models, args.student_tokenizer)}
    identities = {name: model_identity(path) for name, path in [('teacher', teacher_base), *models.items()]}
    identities.update({t['id']: model_identity(t['adapter']) for t in teachers})
    loaded, tokenizer_report, errors = check_tokenizers(tokenizer_paths, teachers)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'asset_report.json', {'source': str(source), 'teacher_base': teacher_base,
                'teachers': teachers, 'students': models, 'tokenizers': tokenizer_report, 'errors': errors})
    if errors:
        raise ValueError('Tokenizer preflight failed:\n' + '\n'.join(errors) +
                         f'\nFull report: {root / "asset_report.json"}')
    from transformers import AutoConfig
    tokenizers = []
    contexts = {}
    for name, path in [('teacher', teacher_base), *models.items()]:
        tok = loaded[name]
        kwargs = base['model']['chat_template_kwargs'] if name == 'teacher' else {'enable_thinking': False}
        tokenizers.append((name, tok, kwargs))
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        contexts[name] = getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', args.context_length))
        if args.context_length > contexts[name]:
            raise ValueError(f'{name} supports {contexts[name]} context tokens; requested {args.context_length}')
        if max(tok.get_vocab().values()) >= config.vocab_size:
            raise ValueError(f'{name}: tokenizer IDs exceed model vocab_size={config.vocab_size}; '
                             'use the tokenizer that belongs to these weights')
    if args.check_only:
        print(f'Asset check passed: {root / "asset_report.json"}', flush=True)
        return
    original_train = load_data(source_data / 'train.jsonl')
    original_eval = load_data(source_data / 'validation.jsonl')
    # Check against the entire held-out split BEFORE taking an evaluation limit.
    original_train, overlaps = split_guard(original_train, original_eval)
    train, excluded_train = select_common(original_train, tokenizers, args.max_prompt_tokens)
    validation, excluded_eval = select_common(original_eval, tokenizers, args.max_prompt_tokens, args.eval_limit)
    if any(not set(row['dimensions']) & set(DIMS) for row in train):
        raise ValueError('Some rows have no matching dimension expert')
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for folder in ('crossdistill', 'opd')
                   for p in sorted((ROOT / folder).glob('*.py'))}
    manifest = {'schema': 1, 'source': str(source), 'source_seed': args.source_seed,
                'models': models, 'teachers': teachers, 'identities': identities,
                'tokenizers': tokenizer_report,
                'code_sha256': code_hashes, 'source_data_sha256': {s: digest(source_data / f'{s}.jsonl') for s in ('train', 'validation')},
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                              if k not in ('resume', 'dry_run', 'prepare_only', 'check_only')},
                'evaluation': base['quality'], 'judge_model': os.environ.get('OPD_JUDGE_MODEL'),
                'judge_url': os.environ.get('OPD_JUDGE_BASE_URL'),
                'scope': 'held-out static prompt evaluation; not original interactive episode metrics'}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Run identity changed; choose a new output-root')
        if not args.resume:
            raise FileExistsError('Output exists; use --resume')
    if not args.dry_run and not args.prepare_only and base['quality']['evaluator'] == 'opd.hierarchy_judge:hierarchy_response':
        from opd.hierarchy_judge import settings
        settings()  # Fail before any expensive training if the evaluator is unconfigured.
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, manifest)
    data = root / 'data'
    write_rows(data / 'train.jsonl', train)
    write_rows(data / 'validation.jsonl', validation)
    atomic_json(data / 'audit.json', {'train': len(train), 'validation': len(validation),
                'train_by_task': dict(Counter(r['task_id'] for r in train)),
                'validation_by_task': dict(Counter(r['task_id'] for r in validation)),
                'excluded_train_overlong': excluded_train, 'excluded_eval_overlong': excluded_eval,
                'excluded_train_split_overlap': overlaps})
    if args.prepare_only:
        print(f'Prepared {data}', flush=True)
        return
    runner = Runner({'output_dir': str(root), 'num_processes': args.num_processes,
                     'inference_num_processes': args.num_processes}, resume=args.resume, dry_run=args.dry_run)
    results = []
    for seed in args.seeds:
        for name, model in models.items():
            stem = f'seed_{seed}/{name}'
            eval_cfg = evaluation_config(base, model, data, args, seed)
            eval_cfg['model']['tokenizer'] = tokenizer_paths[name]
            baseline = runner.evaluate(stem + '/base', eval_cfg, model=model)
            for method in args.methods:
                cfg = copy.deepcopy(eval_cfg)
                cfg['model'].update(student_mode=args.student_mode, lora_rank=32, lora_alpha=64,
                                     lora_targets='all-linear', lora_dropout=0., gradient_checkpointing=True)
                cfg['teachers'] = teachers
                cfg['teacher'].update(device='auto', dtype=args.dtype)
                cfg['output_dir'] = str(root / 'runs' / stem / method)
                cfg['train'].update(max_steps=args.steps, batch_size=args.micro_batch,
                    global_prompt_batch=args.global_batch, gradient_accumulation_steps=args.global_batch // (args.num_processes * args.micro_batch),
                    learning_rate=args.learning_rate, warmup_steps=min(20, args.steps), weight_decay=.1,
                    save_every=args.save_every, objective='text_continuation_ce',
                    trajectory_source='student_text_prefix' if method == 'text_opd' else 'teacher_text')
                cfg['crossdistill'] = {'method': method, 'teacher_base': teacher_base,
                    'teacher_tokenizer': teacher_tokenizer,
                    'teacher_chat_kwargs': base['model']['chat_template_kwargs'], 'prefix_tokens': args.prefix_tokens,
                    'teacher_tokens': args.teacher_tokens, 'teacher_context': args.context_length,
                    'identity': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}
                path = root / 'configs' / stem / f'{method}.json'
                atomic_json(path, cfg)
                cmd = [sys.executable, '-m', 'crossdistill.engine', '--config', str(path)]
                if args.backend == 'zero2':
                    launch = yaml.safe_load((ROOT / 'configs/accelerate_zero2.yaml').read_text())
                    launch.update(num_processes=args.num_processes,
                                  mixed_precision={'float32': 'no', 'bfloat16': 'bf16', 'float16': 'fp16'}[args.dtype])
                    launch_path = runner.config('accelerate', launch)
                    cmd = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file', str(launch_path),
                           '-m', 'crossdistill.engine', '--config', str(path)]
                elif args.num_processes > 1:
                    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                           f'--nproc_per_node={args.num_processes}', '--module', 'crossdistill.engine', '--config', str(path)]
                if args.resume:
                    cmd.append('--resume')
                final = Path(cfg['output_dir']) / 'final'
                runner.command('train/' + stem + '/' + method, cmd, [final / 'opd_metadata.json'], cfg)
                after = runner.evaluate(stem + '/' + method, eval_cfg, model=str(final))
                exported = root / 'full_models' / stem / method
                runner.command('export/' + stem + '/' + method,
                    [sys.executable, '-m', 'opd.full_export', '--model', str(final), '--base', model,
                     '--output', str(exported), '--dtype', args.dtype] + (['--resume'] if args.resume else []),
                    [exported / 'export_manifest.json'], {'config': cfg, 'final': str(final)})
                results.append({'student': name, 'seed': seed, 'method': method, 'baseline': baseline,
                                'evaluation': after, 'checkpoint': str(final), 'full_model': str(exported)})
    atomic_json(root / ('plan.json' if args.dry_run else 'results.json'), {'runs': results, 'dry_run': args.dry_run})
    if not args.dry_run:
        from .report import summarize
        summarize(root)
    print(f'{"Plan" if args.dry_run else "Results"}: {root}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reuse-from', type=Path, default=Path(os.environ.get('OPD_REUSE_FROM', DEFAULT_SOURCE)))
    p.add_argument('--source-seed', type=int, default=42)
    p.add_argument('--teacher-base', default=os.environ.get('OPD_TEACHER_BASE'))
    p.add_argument('--teacher-tokenizer', default=os.environ.get('OPD_TEACHER_TOKENIZER'),
                   help='Original teacher tokenizer; default: first dimension final with tokenizer assets')
    p.add_argument('--student-tokenizer', action='append', default=[], metavar='NAME=PATH',
                   help='Explicit tokenizer location for a selected student; repeatable')
    p.add_argument('--model-root', type=Path, default=Path(os.environ.get('OPD_MODEL_ROOT', MODEL_ROOT)))
    p.add_argument('--students', nargs='+', choices=list(MODELS), default=list(MODELS))
    p.add_argument('--methods', nargs='+', choices=['text_opd', 'sequence_kd'], default=['text_opd', 'sequence_kd'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--output-root', type=Path, default=ROOT / 'outputs/cross_model_v1')
    p.add_argument('--num-processes', type=int, default=int(os.environ.get('OPD_PROCESSES', '8')))
    p.add_argument('--backend', choices=['ddp', 'zero2'], default='zero2')
    p.add_argument('--student-mode', choices=['full', 'lora'], default='full')
    p.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='bfloat16')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--global-batch', type=int, default=32)
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=2e-6)
    p.add_argument('--save-every', type=int, default=200)
    p.add_argument('--context-length', type=int, default=8192)
    p.add_argument('--max-prompt-tokens', type=int, default=4096)
    p.add_argument('--prefix-tokens', type=int, default=128)
    p.add_argument('--teacher-tokens', type=int, default=512)
    p.add_argument('--eval-limit', type=int, default=64, help='Per task; 0 uses the entire held-out split')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Validate local inputs and emit commands; no GPU model loading')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--check-only', action='store_true', help='Check weights, tokenizer paths and context; no GPU/judge/data filtering')
    execute(p.parse_args())


if __name__ == '__main__':
    main()
