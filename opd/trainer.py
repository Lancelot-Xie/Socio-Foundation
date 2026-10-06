"""Single-use fresh-rollout OPD and teacher-demo SFT with Accelerate."""

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from .data import TaskSampler, encode_prompt, load_data, make_batch, tokenizer_signature, warmup_response
from .losses import forward_loss, sampled_reverse_loss, sequence_mean
from .models import (
    TeacherBank,
    generation_model,
    load_student,
    load_tokenizer,
    response_logits,
    teacher_load_context,
)
from .rollout import collect_episode
from .routing import Router
from .checkpoints import archive_incomplete, atomic_json, complete_checkpoint


class OPDAccelerator(Accelerator):
    def wait_for_everyone(self):
        # Accelerate 1.12 passes GPU device_ids even for Gloo/CPU. On an MPS Mac,
        # PyTorch then selects an unsupported MPS barrier. CPU Gloo needs no device_ids.
        if self.device.type == "cpu" and torch.distributed.is_initialized():
            torch.distributed.barrier()
        else:
            super().wait_for_everyone()


def select_batch(batch, indices):
    return {k: v[indices] if torch.is_tensor(v) else v for k, v in batch.items()}


def teacher_targets(bank, batch, rows, router, cfg, device):
    groups = defaultdict(list)
    strengths = []
    for i, row in enumerate(rows):
        weights, strength = router.resolve(row)
        if row.get("_offline") and row.get("demo_teacher"):
            name = row["demo_teacher"]
            if name not in weights:
                raise ValueError(f"Demo teacher {name} is not active for task {row['task_id']}")
            weights = {name: 1.0}
        strengths.append(strength)
        for teacher, alpha in weights.items():
            groups[teacher].append((i, alpha))
    objective = cfg["train"]["objective"]
    combined = torch.zeros_like(batch["response_mask"])
    targets = []
    for teacher, pairs in groups.items():
        indices = torch.tensor([p[0] for p in pairs], device=device)
        alpha = torch.tensor([p[1] for p in pairs], device=device)
        result = bank.score(teacher, select_batch(batch, indices), objective, cfg["train"]["teacher_top_k"])
        result = {k: v.to(device) if torch.is_tensor(v) else v for k, v in result.items()}
        if objective == "sampled_reverse_kl":
            combined.index_add_(0, indices, result["sampled"] * alpha.unsqueeze(-1))
        else:
            targets.append({**result, "indices": indices, "alpha": alpha})
    return combined, targets, torch.tensor(strengths, device=device)


def _fingerprint(cfg):
    # max_steps may increase on resume; model/objective/data/optimizer settings must not change.
    train = {k: v for k, v in cfg["train"].items() if k not in ("resume_from", "max_steps", "save_every")}
    selected = {k: cfg[k] for k in ("model", "teacher", "teachers", "data", "routing", "rollout", "seed")
                if k in cfg}
    selected["train"] = train
    if not cfg.get("qgpi", {}).get("enabled") and cfg["quality"]["evaluator"] == "opd.objective:objective_response":
        from .objective import backend_hashes
        selected["quality"] = cfg["quality"]
        selected["objective_backend_hashes"] = backend_hashes()
    if cfg.get("qgpi", {}).get("enabled"):
        selected["qgpi"] = cfg["qgpi"]
        if cfg["qgpi"]["evaluator"] == "opd.objective:objective_candidates":
            from .objective import backend_hashes
            selected["objective_backend_hashes"] = backend_hashes()
        registry = cfg["qgpi"]["registry_file"]
        if registry:
            selected["registry_sha256"] = hashlib.sha256(Path(registry).read_bytes()).hexdigest()
    for key in ("train_file", "calibration_file", "demo_file", "eval_file"):
        path = cfg["data"].get(key)
        if path and Path(path).is_file():
            selected[key + "_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    path = cfg["routing"].get("calibration_file")
    if path:
        selected["routing_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def export_model(accelerator, model, tokenizer, folder, cfg):
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        folder.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(folder, state_dict=state, safe_serialization=True)
        tokenizer.save_pretrained(folder)
        (folder / "opd_metadata.json").write_text(json.dumps({
            "base_model": cfg["model"]["base_model"], "student_mode": cfg["model"]["student_mode"],
            "dimension": cfg["data"]["dimension"], "stage": cfg["train"]["stage"],
            "trajectory_source": cfg["train"]["trajectory_source"], "objective": cfg["train"]["objective"],
            "quality_basis": ["F", "S", "U", "T", "N"] if cfg.get("qgpi", {}).get("enabled") else None,
        }, indent=2))
    del state
    accelerator.wait_for_everyone()


def save_checkpoint(accelerator, model, tokenizer, sampler, folder, step, cfg, fingerprint):
    if complete_checkpoint(folder):
        raise FileExistsError(f"Refusing to overwrite complete checkpoint {folder}")
    temporary = folder.with_name("." + folder.name + ".pending")
    if accelerator.is_main_process:
        for path in (folder, temporary):
            if path.exists():
                archive_incomplete(path)
    accelerator.wait_for_everyone()
    export_model(accelerator, model, tokenizer, temporary / "model", cfg)
    accelerator.save_state(str(temporary / "state"))
    atomic_json(temporary / f"sampler_rank_{accelerator.process_index}.json", sampler.state_dict())
    # Every rank must finish its RNG/optimizer/sampler files before the commit.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        atomic_json(temporary / "progress.json", {"step": step, "fingerprint": fingerprint,
                                                  "world_size": accelerator.num_processes})
        temporary.rename(folder)
        atomic_json(folder.parent / "latest.json", {"checkpoint": str(folder)})
    accelerator.wait_for_everyone()


def train(cfg):
    precision = {"float32": "no", "float16": "fp16", "bfloat16": "bf16"}[cfg["model"]["dtype"]]
    # Accelerate's ZeRO launcher exports the YAML value "auto" here, but
    # Accelerator parses this environment variable with int() before we can
    # fill the DeepSpeed microbatch configuration below.
    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(cfg["train"]["gradient_accumulation_steps"])
    accelerator = OPDAccelerator(
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"], mixed_precision=precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)],
    )
    # Stage 2 shards optimizer/gradients while keeping rollout weights resident on each rank.
    # Stage 3/FSDP need a separate collective rollout service to avoid variable-length hook deadlocks.
    if "FSDP" in str(accelerator.distributed_type):
        raise ValueError("Use DDP or the supplied DeepSpeed ZeRO-2 config; FSDP rollout is not supported")
    if accelerator.state.deepspeed_plugin and accelerator.state.deepspeed_plugin.zero_stage == 3:
        raise ValueError("This rollout backend supports ZeRO-2, not ZeRO-3; use configs/accelerate_zero2.yaml")
    effective_batch = cfg["train"]["batch_size"] * cfg["train"]["gradient_accumulation_steps"] * accelerator.num_processes
    if cfg["train"]["global_prompt_batch"] not in (None, effective_batch):
        raise ValueError(f"Actual prompt batch {effective_batch} differs from requested "
                         f"{cfg['train']['global_prompt_batch']}; check launch world size")
    if accelerator.state.deepspeed_plugin:
        ds = accelerator.state.deepspeed_plugin.deepspeed_config
        # On-policy prompts are sampled directly, so no DataLoader supplies these to Accelerate.
        ds["train_micro_batch_size_per_gpu"] = cfg["train"]["batch_size"]
        ds["gradient_accumulation_steps"] = cfg["train"]["gradient_accumulation_steps"]
        ds["train_batch_size"] = (cfg["train"]["batch_size"] * cfg["train"]["gradient_accumulation_steps"]
                                  * accelerator.num_processes)
        ds["gradient_clipping"] = cfg["train"]["max_grad_norm"]
    set_seed(cfg["seed"], device_specific=True)
    output = Path(cfg["output_dir"])
    resume = cfg["train"]["resume_from"]
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is nonempty: {output}. Set resume_from or use a new directory")
    warmup = cfg["train"]["stage"] == "warmup"
    qgpi_enabled = cfg.get("qgpi", {}).get("enabled", False)
    rows = load_data(cfg["data"]["train_file"], cfg["data"]["dimension"], require_response=warmup)
    source = cfg["train"]["trajectory_source"]
    demos = []
    if not warmup and source != "student":
        demos = load_data(cfg["data"]["demo_file"], cfg["data"]["dimension"], require_response=True)
        absent = {r["task_id"] for r in rows} - {r["task_id"] for r in demos}
        if absent:
            raise ValueError(f"Teacher demonstrations missing entire tasks: {sorted(absent)}")
    router = Router(cfg)
    if qgpi_enabled:
        from .qgpi import Shortlist, collect_decision_batch, quality_loss, metric_values, METRIC_KEYS, audit_record
        shortlist = Shortlist(cfg, router)
    if not warmup:
        active = [router.resolve(row)[1] for row in rows]
        if not any(active):
            raise ValueError("No active teacher supervision; all routes were rejected or have zero strength")
    tokenizer = load_tokenizer(cfg)
    signature = tokenizer_signature(tokenizer, cfg)
    # Fail before model loading if persona/history context would be silently truncated.
    for row in rows:
        encode_prompt(tokenizer, row["messages"], cfg)
        if warmup:
            warmup_response(row, tokenizer, cfg, signature)
    for row in demos:
        encode_prompt(tokenizer, row["messages"], cfg)
        warmup_response(row, tokenizer, cfg, signature)
    if cfg["train"]["eval_every"]:
        from .validation import periodic_validation, validation_rows
        for row in validation_rows(cfg):
            encode_prompt(tokenizer, row["messages"], cfg)
    sampler = TaskSampler(demos if source == "teacher" and not warmup else rows,
                          cfg["data"]["task_weights"], cfg["seed"] + accelerator.process_index)
    replay = TaskSampler(demos, {}, 0).sources if demos else {}
    model = load_student(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    # Fixed warmup then constant LR permits extending max_steps without changing an existing schedule.
    warmup_steps = cfg["train"]["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s+1)/max(1, warmup_steps)))
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(scheduler)
    with teacher_load_context(accelerator):
        bank = None if warmup else TeacherBank(cfg, tokenizer, accelerator.device)
    fingerprint = _fingerprint(cfg)
    step = 0
    if resume:
        folder = Path(resume)
        progress = json.loads((folder / "progress.json").read_text())
        if progress["fingerprint"] != fingerprint or progress["world_size"] != accelerator.num_processes:
            raise ValueError("Resume requires unchanged model, data, routing, optimizer and world size")
        accelerator.load_state(str(folder / "state"))
        state = json.loads((folder / f"sampler_rank_{accelerator.process_index}.json").read_text())
        state["rng"] = (state["rng"][0], tuple(state["rng"][1]), state["rng"][2])
        sampler.load_state_dict(state)
        step = progress["step"]
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.resolved.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        (output / "model_info.json").write_text(json.dumps({"parameters": total, "trainable_parameters": trainable,
                                                            "world_size": accelerator.num_processes,
                                                            "effective_prompt_batch": effective_batch}, indent=2))
    accelerator.print(f"student={cfg['model']['student_mode']} trainable={trainable:,}/{total:,} "
                      f"objective={cfg['train']['objective']} device={accelerator.device}")
    accelerator.wait_for_everyone()
    # Repair a validation interrupted after a checkpoint was committed at this step.
    if step and cfg["train"]["eval_every"] and step % cfg["train"]["eval_every"] == 0:
        periodic_validation(accelerator, model, tokenizer, cfg, output, step)
    optimizer.zero_grad()
    aggregate = torch.zeros(5, device=accelerator.device)
    quality_aggregate = torch.zeros(len(METRIC_KEYS), device=accelerator.device) if qgpi_enabled else None
    started = time.monotonic()
    step_started = started
    rollout_seconds = 0.0
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    while step < cfg["train"]["max_steps"]:
        with accelerator.accumulate(model):
            rollout_started = time.monotonic()
            selected = sampler.sample(cfg["train"]["batch_size"])
            if qgpi_enabled:
                model.eval()
                with generation_model(model, accelerator) as generator:
                    decisions = collect_decision_batch(generator, bank, shortlist, selected, tokenizer, cfg,
                                                       accelerator.device)
                transitions = [d["candidates"][0] for d in decisions]
                prompts = [t["prompt_ids"] for t in transitions]
                responses = [t["response_ids"] for t in transitions]
                batch_rows = [t["row"] for t in transitions]
                if cfg["qgpi"]["audit_candidates"]:
                    with (output / f"candidates_rank_{accelerator.process_index}.jsonl").open("a") as stream:
                        for decision in decisions:
                            stream.write(json.dumps({"optimizer_step": step + 1, **audit_record(decision)}, ensure_ascii=False) + "\n")
            elif warmup:
                prompts = [encode_prompt(tokenizer, row["messages"], cfg) for row in selected]
                responses = [warmup_response(row, tokenizer, cfg, signature) for row in selected]
                batch_rows = selected
            else:
                model.eval()
                transitions = []
                with generation_model(model, accelerator) as generator:
                    for row in selected:
                        offline = source == "teacher" or (source == "mixed" and
                                   sampler.rng.random() < cfg["train"]["teacher_fraction"])
                        if offline:
                            demo = row if source == "teacher" else sampler.rng.choice(
                                sampler.rng.choice(replay[row["task_id"]]))
                            transitions.append({"prompt_ids": encode_prompt(tokenizer, demo["messages"], cfg),
                                                "response_ids": warmup_response(demo, tokenizer, cfg, signature),
                                                "row": {**demo, "_offline": True}})
                        else:
                            episode, _ = collect_episode(generator, row, tokenizer, cfg, accelerator.device)
                            transitions.extend(episode)
                prompts = [t["prompt_ids"] for t in transitions]
                responses = [t["response_ids"] for t in transitions]
                batch_rows = [t["row"] for t in transitions]
            rollout_seconds += time.monotonic() - rollout_started
            batch = make_batch(prompts, responses, tokenizer.pad_token_id, accelerator.device)
            mask = batch["response_mask"]
            if warmup:
                model.train()
                with accelerator.autocast():
                    logits = response_logits(model, batch)
                    logp = logits.log_softmax(-1).gather(-1, batch["responses"].unsqueeze(-1)).squeeze(-1)
                    loss = -sequence_mean(logp, mask).mean()
                diagnostic = -sequence_mean(logp.detach(), mask).mean()
                strength = torch.ones(len(batch_rows), device=accelerator.device)
            elif qgpi_enabled:
                loss, components, tokens, count = quality_loss(model, bank, decisions, tokenizer, cfg, accelerator)
                diagnostic = components["winner_kl"]
                strength = torch.tensor([d["weight"] for d in decisions], device=accelerator.device)
                quality_aggregate += metric_values(decisions, components, accelerator.device) / cfg["train"]["gradient_accumulation_steps"]
                logits, logp = None, None
            else:
                old_logp = None
                if cfg["train"]["objective"] == "sampled_reverse_kl" or cfg["train"]["anchor_coef"]:
                    with torch.no_grad(), accelerator.autocast():
                        logits = response_logits(model, batch, cfg["rollout"]["temperature"])
                        old_logp = logits.log_softmax(-1).gather(-1, batch["responses"].unsqueeze(-1)).squeeze(-1)
                    del logits
                teacher_logp, targets, strength = teacher_targets(bank, batch, batch_rows, router, cfg,
                                                                  accelerator.device)
                anchor = bank.anchor_score(batch).to(accelerator.device) if cfg["train"]["anchor_coef"] else None
                model.train()
                with accelerator.autocast():
                    logits = response_logits(model, batch, cfg["rollout"]["temperature"])
                    log_distribution = logits.log_softmax(-1)
                    logp = log_distribution.gather(-1, batch["responses"].unsqueeze(-1)).squeeze(-1)
                    if cfg["train"]["objective"] == "sampled_reverse_kl":
                        loss, per_sample = sampled_reverse_loss(logp, old_logp, teacher_logp, mask, strength,
                                                                cfg["train"]["advantage_clip"])
                    else:
                        loss, per_sample = forward_loss(log_distribution, targets, mask, strength)
                    if cfg["train"]["sft_coef"]:
                        offline_mask = torch.tensor([float(r.get("_offline", False)) for r in batch_rows],
                                                    device=accelerator.device)
                        # Use untempered probabilities for the auxiliary supervised objective.
                        supervised = (logits * cfg["rollout"]["temperature"]).log_softmax(-1)
                        supervised = supervised.gather(-1, batch["responses"].unsqueeze(-1)).squeeze(-1)
                        loss = loss - cfg["train"]["sft_coef"] * (
                            sequence_mean(supervised, mask) * offline_mask).sum() / offline_mask.sum().clamp_min(1)
                    if anchor is not None:
                        anchor_loss, _ = sampled_reverse_loss(logp, old_logp, anchor, mask,
                                                              torch.ones_like(strength), cfg["train"]["advantage_clip"])
                        loss = loss + cfg["train"]["anchor_coef"] * anchor_loss
                active = (strength > 0).float()
                diagnostic = (per_sample * active).sum() / active.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            accelerator.backward(loss)
            values = torch.stack([loss.detach(), diagnostic.detach(), strength.mean(), mask.sum(),
                                  torch.tensor(float(len(batch_rows)), device=accelerator.device)])
            aggregate += values / cfg["train"]["gradient_accumulation_steps"]
            if accelerator.sync_gradients and cfg["train"]["max_grad_norm"] > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"])
            optimizer.step()
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                scheduler.step()
            optimizer.zero_grad()
            del logits, logp, loss
            if not warmup and not qgpi_enabled:
                del log_distribution, teacher_logp, targets, old_logp
        if accelerator.sync_gradients:
            step += 1
            metric = accelerator.reduce(aggregate, reduction="mean").cpu().tolist()
            record = {"step": step, "loss": metric[0], "nll" if warmup else "sampled_log_ratio_or_kl": metric[1],
                      "strength": metric[2], "response_tokens_per_rank_microbatch": metric[3],
                      "turns_per_rank_microbatch": metric[4], "lr": scheduler.get_last_lr()[0],
                      "elapsed_seconds": time.monotonic()-started,
                      "teacher_calls_local": bank.calls if bank else 0,
                      "teacher_prefill_tokens_local": bank.tokens if bank else 0}
            # All ranks participate; use the slowest rank / largest peak, not GPU 0 alone.
            local_performance = torch.tensor([
                time.monotonic() - step_started, rollout_seconds,
                torch.cuda.max_memory_allocated(accelerator.device) / 2**30 if accelerator.device.type == "cuda" else 0,
                torch.cuda.max_memory_reserved(accelerator.device) / 2**30 if accelerator.device.type == "cuda" else 0,
            ], dtype=torch.float64, device=accelerator.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(local_performance, op=torch.distributed.ReduceOp.MAX)
            duration, rollout_duration, allocated, reserved = local_performance.cpu().tolist()
            record["performance"] = {"step_seconds_max": duration, "rollout_seconds_max": rollout_duration,
                                     "global_prompts_per_second": effective_batch / max(duration, 1e-9),
                                     "peak_allocated_gib_max": allocated, "peak_reserved_gib_max": reserved}
            record["trajectory_source"] = "demos_sft" if warmup else source
            record["offline_turns_last_microbatch"] = sum(bool(r.get("_offline")) for r in batch_rows)
            if qgpi_enabled:
                quality_metric = accelerator.reduce(quality_aggregate, reduction="mean").cpu().tolist()
                record["qgpi"] = dict(zip(METRIC_KEYS, quality_metric, strict=True))
                q = record["qgpi"]
                q["acceptance_rate"] = q["accepted"] / max(q["visited"], 1)
                q["mean_positive_advantage"] = q["advantage_sum"] / max(q["accepted"], 1e-12)
                q["scores"] = {d: {"student": q[d + "_student_sum"] / q[d + "_count"] if q[d + "_count"] else None,
                                    "winner": q[d + "_winner_sum"] / q[d + "_winner_count"] if q[d + "_winner_count"] else None}
                               for d in ("F", "S", "U", "T", "N")}
                record["trajectory_source"] = "student_states_quality_gated_teacher_targets"
                quality_aggregate.zero_()
            if accelerator.is_main_process:
                with (output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            aggregate.zero_()
            if step % cfg["train"]["save_every"] == 0 or step == cfg["train"]["max_steps"]:
                save_checkpoint(accelerator, model, tokenizer, sampler, output / f"step_{step:06d}",
                                step, cfg, fingerprint)
            if cfg["train"]["eval_every"] and step % cfg["train"]["eval_every"] == 0:
                periodic_validation(accelerator, model, tokenizer, cfg, output, step)
            step_started, rollout_seconds = time.monotonic(), 0.0
            if accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(accelerator.device)
    export_model(accelerator, model, tokenizer, output / "final", cfg)
    accelerator.end_training()
    return str(output / "final")
