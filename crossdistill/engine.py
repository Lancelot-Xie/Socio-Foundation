"""Fresh text-prefix distillation. Supports DDP and DeepSpeed ZeRO-2."""
import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from opd.checkpoints import atomic_json, latest_checkpoint, archive_incomplete
from opd.data import TaskSampler, load_data, make_batch
from opd.losses import sequence_mean
from opd.models import TeacherBank, generation_model, load_student, load_tokenizer, response_logits, teacher_load_context
from opd.trainer import OPDAccelerator, export_model, save_checkpoint
from .text import continuation_example, generate, prompt_ids, teacher_config


def train(cfg, resume=False):
    tr, cross = cfg['train'], cfg['crossdistill']
    os.environ['ACCELERATE_GRADIENT_ACCUMULATION_STEPS'] = str(tr['gradient_accumulation_steps'])
    acc = OPDAccelerator(gradient_accumulation_steps=tr['gradient_accumulation_steps'],
                        mixed_precision={'float32': 'no', 'bfloat16': 'bf16', 'float16': 'fp16'}[cfg['model']['dtype']],
                        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)])
    if 'FSDP' in str(acc.distributed_type) or (acc.state.deepspeed_plugin and acc.state.deepspeed_plugin.zero_stage != 2):
        raise ValueError('Use DDP or ZeRO-2; sharded-parameter rollout is not supported')
    effective = tr['batch_size'] * tr['gradient_accumulation_steps'] * acc.num_processes
    if effective != tr['global_prompt_batch']:
        raise ValueError(f'Global batch mismatch: {effective} != {tr["global_prompt_batch"]}')
    if acc.state.deepspeed_plugin:
        acc.state.deepspeed_plugin.deepspeed_config.update(
            train_micro_batch_size_per_gpu=tr['batch_size'], gradient_accumulation_steps=tr['gradient_accumulation_steps'],
            train_batch_size=effective, gradient_clipping=tr['max_grad_norm'])
    output = Path(cfg['output_dir'])
    checkpoint = latest_checkpoint(output) if resume else None
    if acc.is_main_process and output.exists() and any(output.iterdir()) and not checkpoint:
        if not resume:
            raise FileExistsError(output)
        archive_incomplete(output)
    acc.wait_for_everyone()
    set_seed(cfg['seed'], device_specific=True)
    tokenizer = load_tokenizer(cfg)
    tcfg = teacher_config(cfg)
    ttok = load_tokenizer(tcfg)
    rows = load_data(cfg['data']['train_file'])
    sampler = TaskSampler(rows, cfg['data']['task_weights'], cfg['seed'] + acc.process_index)
    model = load_student(cfg)
    # Chat termination is model-specific (Llama <|eot_id|> differs from eos).
    ending = tokenizer.apply_chat_template([{'role': 'user', 'content': 'x'},
                                           {'role': 'assistant', 'content': 'x'}],
                                          tokenize=True, **cfg['model']['chat_template_kwargs'])
    from .text import stop_ids
    stops = stop_ids(model, tokenizer)
    eos = next((i for i in reversed(ending) if i in stops), tokenizer.eos_token_id)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=tr['learning_rate'], weight_decay=tr['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s+1)/max(1, tr['warmup_steps'])))
    model, optimizer = acc.prepare(model, optimizer)
    acc.register_for_checkpointing(scheduler)
    with teacher_load_context(acc):
        bank = TeacherBank(tcfg, ttok, acc.device)
    identity = copy.deepcopy(cfg)
    identity.pop('output_dir', None)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    step = 0
    if checkpoint:
        progress = json.loads((checkpoint / 'progress.json').read_text())
        if progress['fingerprint'] != fingerprint or progress['world_size'] != acc.num_processes:
            raise ValueError('Resume requires unchanged config/data/model identities and world size')
        acc.load_state(str(checkpoint / 'state'))
        state = json.loads((checkpoint / f'sampler_rank_{acc.process_index}.json').read_text())
        state['rng'] = (state['rng'][0], tuple(state['rng'][1]), state['rng'][2])
        sampler.load_state_dict(state)
        step = progress['step']
    if acc.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'config.resolved.json', cfg)
    acc.wait_for_everyone()
    optimizer.zero_grad()
    totals = torch.zeros(4, device=acc.device)
    while step < tr['max_steps']:
        with acc.accumulate(model):
            batch_rows = sampler.sample(tr['batch_size'])
            prompts, responses, masks = [], [], []
            # Select experts from a separate RNG stream tied to prompt order;
            # changing prefix length/method does not alter the teacher schedule.
            for row in batch_rows:
                choices = [t['id'] for t in cfg['teachers'] if t['id'] in row['dimensions']]
                teacher_id = sampler.rng.choice(choices)
                prompt = prompt_ids(tokenizer, row['messages'], cfg['model']['chat_template_kwargs'])
                prefix = ''
                if cross['method'] == 'text_opd':
                    budget = random.randint(0, cross['prefix_tokens'])
                    model.eval()
                    with generation_model(model, acc) as plain:
                        prefix_tokens, _ = generate(plain, tokenizer, prompt, budget, acc.device)
                    prefix = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
                # Teacher and student each use their own assistant generation
                # header. The decoded student prefix is appended as plain text.
                teacher_prompt = prompt_ids(ttok, row['messages'], tcfg['model']['chat_template_kwargs'])
                teacher_prompt += ttok.encode(prefix, add_special_tokens=False)
                budget = min(cross['teacher_tokens'], cross['teacher_context'] - len(teacher_prompt))
                if budget < 1:
                    raise ValueError(f'Teacher context overflow at {row["id"]}; lower prefix_tokens')
                teacher = bank.activate(teacher_id)
                for attempt in range(3):
                    tokens, ended = generate(teacher, ttok, teacher_prompt, budget, bank.device)
                    text = ttok.decode(tokens, skip_special_tokens=True)
                    try:
                        ids, mask, truncated = continuation_example(tokenizer, prompt, prefix, text, ended,
                                                                    cfg['rollout']['context_length'], eos)
                        break
                    except ValueError:
                        if attempt == 2:
                            raise
                prompts.append(prompt)
                responses.append(ids)
                masks.append(mask)
                totals[2] += int(truncated)
                totals[3] += len(tokens)
            batch = make_batch(prompts, responses, tokenizer.pad_token_id, acc.device)
            mask = torch.zeros_like(batch['response_mask'])
            for i, values in enumerate(masks):
                mask[i, :len(values)] = torch.tensor(values, device=acc.device)
            model.train()
            with acc.autocast():
                logits = response_logits(model, batch)
                # CE in chunks avoids a second complete float32 log-probability tensor.
                flat = logits.reshape(-1, logits.shape[-1])
                labels = batch['responses'].reshape(-1)
                nll = torch.cat([torch.nn.functional.cross_entropy(flat[s:s+128], labels[s:s+128], reduction='none')
                                 for s in range(0, len(labels), 128)]).reshape_as(mask)
                strength = torch.tensor([r['distill_strength'] for r in batch_rows], device=acc.device)
                loss = (sequence_mean(nll, mask) * strength).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            acc.backward(loss)
            totals[0] += loss.detach() / tr['gradient_accumulation_steps']
            totals[1] += mask.sum()
            if acc.sync_gradients:
                acc.clip_grad_norm_(model.parameters(), tr['max_grad_norm'])
            optimizer.step()
            if acc.sync_gradients and not acc.optimizer_step_was_skipped:
                scheduler.step()
            optimizer.zero_grad()
            del loss, logits, nll, flat
        if acc.sync_gradients:
            step += 1
            values = acc.reduce(totals, reduction='mean').tolist()
            if acc.is_main_process:
                record = dict(step=step, loss=values[0], supervised_tokens_per_rank=values[1],
                              truncated_targets_per_rank=values[2], teacher_tokens_per_rank=values[3],
                              method=cross['method'], lr=scheduler.get_last_lr()[0])
                with (output / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
            totals.zero_()
            if step % tr['save_every'] == 0 or step == tr['max_steps']:
                save_checkpoint(acc, model, tokenizer, sampler, output / f'step_{step:06d}', step, cfg, fingerprint)
    export_model(acc, model, tokenizer, output / 'final', cfg)
    acc.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text()), args.resume)
