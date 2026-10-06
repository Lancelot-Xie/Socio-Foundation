"""Generate BehaviorChain trajectories with a LoRA expert, then train rank-8 SFT.

The generation model is the common base model with the supplied BehaviorChain
LoRA adapter merged for inference.  The student is initialized from the same
base model and receives a fresh rank-8 LoRA adapter; the expert adapter is not
silently carried into the student.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml

from .behavior_chain_finetune import prepare_data
from .checkpoints import atomic_json
from .config import load_config, validate
from .data import load_data, read_rows, tokenizer_signature, write_rows
from .experiments import Runner
from .models import load_student, load_tokenizer
from .rollout import collect_episodes


def _generation_config(base_model: str, adapter: str, prepared: Path, output: Path,
                       max_prompt_tokens: int, max_new_tokens: int,
                       generation_batch_size: int = 1) -> dict:
    """Return a validated config for expert rollout only."""
    cfg = {
        "seed": 42,
        "output_dir": str(output),
        "model": {
            "base_model": str(Path(base_model).resolve()),
            "student_init": str(Path(adapter).resolve()),
            "tokenizer": str(Path(base_model).resolve()),
            "student_mode": "full",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_targets": "all-linear",
            "lora_dropout": 0.0,
            "dtype": "bfloat16",
            "trust_remote_code": False,
            "gradient_checkpointing": False,
            "chat_template_kwargs": {"enable_thinking": False, "thinking_budget": 0},
        },
        "teacher": {"device": "auto", "dtype": "bfloat16"},
        "data": {
            "train_file": str(prepared / "train.jsonl"),
            "eval_file": str(prepared / "eval.jsonl"),
            "dimension": "F",
            "task_weights": {"behavior_chain": 1.0},
        },
        "routing": {"tasks": {}},
        "rollout": {
            "context_length": max_prompt_tokens + max_new_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "temperature": 1.0,
            "max_turns": 1,
            "environment_factory": None,
            "generation_batch_size": generation_batch_size,
        },
        "inference": {"batch_size": 1},
        "train": {
            "stage": "warmup",
            "max_steps": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "global_prompt_batch": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "warmup_steps": 0,
            "save_every": 1,
            "eval_every": 1,
            "eval_limit_per_task": 0,
            "objective": "sampled_reverse_kl",
            "trajectory_source": "student",
            "teacher_fraction": 0.5,
            "teacher_top_k": 64,
            "advantage_clip": 5.0,
            "anchor_coef": 0.0,
            "sft_coef": 0.0,
            "max_grad_norm": 1.0,
        },
        "quality": {
            "evaluator": "opd.objective:objective_response",
            "min_score": 0.0,
            "filter_demos": False,
            "keep_per_prompt": 1,
            "task_min_scores": {},
        },
        "qgpi": {"enabled": False},
    }
    validate(cfg)
    return cfg


def _generate_worker(rank: int, world_size: int, cfg: dict, rows: list[dict],
                     part_prefix: str, split: str) -> None:
    """Generate one data shard on exactly one GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("Parallel trajectory generation requires CUDA")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tokenizer = load_tokenizer(cfg)
    signature = tokenizer_signature(tokenizer, cfg)
    model = load_student(cfg).to(device).eval().requires_grad_(False)
    local_rows = rows[rank::world_size]
    generated = []
    skipped = 0
    batch_size = cfg["rollout"]["generation_batch_size"]
    for start in range(0, len(local_rows), batch_size):
        batch_rows = local_rows[start:start + batch_size]
        episodes = collect_episodes(model, batch_rows, tokenizer, cfg, device, sample=False)
        for row, (transitions, response) in zip(batch_rows, episodes, strict=True):
            if not transitions or not response.strip():
                skipped += 1
                continue
            for turn, transition in enumerate(transitions):
                generated_response = tokenizer.decode(
                    transition["response_ids"], skip_special_tokens=True
                ).strip()
                if not generated_response:
                    continue
                generated.append({
                    **row,
                    "id": f"{row['id']}:expert_trajectory:{turn}",
                    "source_id": row.get("source_id", row["id"]),
                    "messages": transition["messages"],
                    "response": generated_response,
                    "response_token_ids": transition["response_ids"],
                    "prompt_token_ids": transition["prompt_ids"],
                    "tokenizer_signature": signature,
                    "trajectory_source": "behavior_chain_lora_expert",
                    "trajectory_split": split,
                    "trajectory_turn": turn,
                })
    write_rows(Path(f"{part_prefix}.rank{rank}.jsonl"), generated)


def generate_split(cfg: dict, source: Path, output: Path, split: str,
                   num_processes: int) -> dict:
    """Generate one split across GPUs, then merge rank files in source order."""
    rows = load_data(source, "F")
    if not torch.cuda.is_available():
        raise RuntimeError("Parallel trajectory generation requires CUDA")
    visible_gpus = torch.cuda.device_count()
    if num_processes < 1 or num_processes > visible_gpus:
        raise ValueError(f"num-processes={num_processes}, visible GPUs={visible_gpus}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output.parent / f".{output.stem}.parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)
    try:
        mp.spawn(
            _generate_worker,
            args=(num_processes, cfg, rows, str(parts_dir / output.stem), split),
            nprocs=num_processes,
            join=True,
        )
        generated = []
        for rank in range(num_processes):
            generated.extend(read_rows(parts_dir / f"{output.stem}.rank{rank}.jsonl"))
        generated.sort(key=lambda row: (row.get("source_index", 0), row["id"]))
        if not generated:
            raise RuntimeError(f"Expert generated no usable {split} trajectories from {source}")
        write_rows(output, generated)
        return {"source_rows": len(rows), "generated_rows": len(generated),
                "output": str(output), "num_processes": num_processes}
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)


def make_student_config(base_model: str, generated_dir: Path, output: Path, args) -> dict:
    cfg = _generation_config(base_model, base_model, generated_dir, output,
                             args.max_prompt_tokens, args.max_new_tokens)
    cfg["model"].pop("student_init", None)
    cfg["model"].update(student_mode="lora", lora_rank=8, lora_alpha=16,
                         gradient_checkpointing=True)
    cfg["data"].update(
        train_file=str(generated_dir / "train.jsonl"),
        eval_file=str(generated_dir / "eval.jsonl"),
    )
    cfg["train"].update(
        stage="warmup",
        max_steps=args.steps,
        batch_size=args.micro_batch,
        global_prompt_batch=args.global_batch,
        gradient_accumulation_steps=args.global_batch // (args.num_processes * args.micro_batch),
        learning_rate=args.learning_rate,
        warmup_steps=min(args.warmup_steps, args.steps),
        save_every=args.save_every,
        eval_every=args.eval_every,
    )
    cfg["output_dir"] = str(output / "training")
    validate(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--expert-adapter", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--upstream-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=2,
                        help="Per-GPU generation batch; start with 2 on 48GB RTX 4090")
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    if args.global_batch % (args.num_processes * args.micro_batch):
        raise ValueError("global-batch must be divisible by num-processes * micro-batch")
    if args.steps < 1 or args.eval_every < 1 or args.steps % args.eval_every:
        raise ValueError("steps must be a positive multiple of eval-every")

    root = Path(args.output).resolve()
    prepared = root / "prepared"
    generated = root / "expert_trajectories"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "run_manifest.json"
    atomic_json(manifest, {
        "workflow": "behavior_chain_expert_trajectory_rank8_sft",
        "base_model": str(Path(args.base_model).resolve()),
        "expert_adapter": str(Path(args.expert_adapter).resolve()),
        "rank": 8, "alpha": 16,
        "train_file": str(Path(args.train_file).resolve()),
        "eval_file": str(Path(args.eval_file).resolve()),
    })

    if not args.skip_generation:
        if prepared.exists() and any(prepared.iterdir()):
            raise FileExistsError(f"Prepared directory is nonempty: {prepared}")
        prepare_data(
            args.train_file, args.eval_file, args.upstream_repo,
            args.base_model, prepared, eval_limit=args.eval_limit,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        gen_cfg = _generation_config(
            args.base_model, args.expert_adapter, prepared,
            root / "generation_model", args.max_prompt_tokens,
            args.max_new_tokens, args.generation_batch_size,
        )
        generate_split(gen_cfg, prepared / "train.jsonl", generated / "train.jsonl",
                       "train", args.num_processes)
        generate_split(gen_cfg, prepared / "eval.jsonl", generated / "eval.jsonl",
                       "eval", args.num_processes)
    elif not (generated / "train.jsonl").is_file():
        raise FileNotFoundError(generated / "train.jsonl")

    cfg = make_student_config(args.base_model, generated, root, args)
    plan = {"output_dir": str(root), "num_processes": args.num_processes}
    runner = Runner(plan, resume=False, dry_run=False)
    if args.num_processes > 1:
        launch = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs/accelerate_zero2.yaml").read_text()
        )
        launch["num_processes"] = args.num_processes
        plan["accelerate_config"] = str(runner.config("accelerate", launch))
    cfg_path = runner.config("rank8_sft", cfg)
    final, _ = runner.train("rank8_sft", cfg)
    atomic_json(root / "result.json", {
        "adapter": final,
        "rank": 8,
        "alpha": 16,
        "generated_train": str(generated / "train.jsonl"),
        "generated_eval": str(generated / "eval.jsonl"),
        "config": str(cfg_path),
    })
    print(json.dumps({"adapter": final, "config": str(cfg_path)}, indent=2))


if __name__ == "__main__":
    main()
