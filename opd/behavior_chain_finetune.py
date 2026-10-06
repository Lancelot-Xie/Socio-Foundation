"""Prepare and evaluate a targeted BehaviorChain LoRA repair run.

This module deliberately uses the original BehaviorChain prompt builder and
objective scorer.  No LLM judge or API client is involved in either training
data construction or evaluation.
"""

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from .data import normalize_row, read_rows, write_rows
from .source_data import digest, group_id
from .upstream import ExternalRequired, canonical_task, render_prefix, source_hashes


TASK = "behavior_chain"


def _answer(raw):
    info = raw.get("extra_info", {})
    payload = info.get("raw", info)
    if isinstance(payload, str):
        payload = json.loads(payload)
    answer = str(payload.get("right_option_letter", "")).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"Missing/invalid BehaviorChain right_option_letter: {answer!r}")
    return answer


def _prompt_tokens(tokenizer, messages, template_kwargs):
    return len(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **template_kwargs,
    ))


def _convert(raw, index, split, source_file, upstream_repo, hashes, tokenizer,
             max_prompt_tokens, template_kwargs):
    if canonical_task(raw.get("data_source", "")) != TASK:
        raise ValueError(f"Unexpected data_source in {source_file}: {raw.get('data_source')}")
    seed_material = [TASK, split, str(source_file), index]
    seed = int(digest(seed_material)[:16], 16)
    messages = render_prefix(raw, TASK, upstream_repo, seed=seed)
    token_count = _prompt_tokens(tokenizer, messages, template_kwargs)
    if token_count > max_prompt_tokens:
        return None, "overlong"
    prompt_hash = digest(messages)
    group, group_field = group_id(raw, TASK)
    answer = _answer(raw)
    row = {
        "id": f"{TASK}:{prompt_hash[:24]}",
        "source_id": f"{TASK}:{group}:{index}",
        "task_id": TASK,
        "dimensions": ["F"],
        "messages": messages,
        "group_id": group,
        "source_split": split,
        "source_file": str(source_file),
        "source_index": index,
        "prompt_tokens": token_count,
        "evaluator_context": {
            "original_row": raw,
            "source_hashes": hashes,
        },
    }
    if split == "train":
        # The gold label is an exact task answer, not a synthetic rationale.
        row["response"] = f"<answer>{answer}</answer>"
    return normalize_row(row, index), group_field


def _load_split(path, split, upstream_repo, hashes, tokenizer, max_prompt_tokens,
                template_kwargs):
    rows, counts, groups = [], Counter(), Counter()
    seen = set()
    for index, raw in enumerate(read_rows(path)):
        counts["source_rows"] += 1
        try:
            row, detail = _convert(
                raw, index, split, path, upstream_repo, hashes, tokenizer,
                max_prompt_tokens, template_kwargs,
            )
        except ExternalRequired:
            counts["external_before_policy"] += 1
            continue
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Failed to convert {path} row {index}: {exc}") from exc
        if row is None:
            counts[detail] += 1
            continue
        prompt_hash = row["id"]
        if prompt_hash in seen:
            counts["duplicate_prompt"] += 1
            continue
        seen.add(prompt_hash)
        groups[detail] += 1
        counts["usable"] += 1
        rows.append(row)
    if not rows:
        raise ValueError(f"No usable BehaviorChain rows in {path}")
    return rows, counts, groups


def prepare_data(train_file, eval_file, upstream_repo, checkpoint, output,
                 eval_limit=0, seed=42, max_prompt_tokens=8192):
    from transformers import AutoTokenizer

    train_file = Path(train_file).resolve()
    eval_file = Path(eval_file).resolve()
    upstream_repo = Path(upstream_repo).resolve()
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prepared-data directory is nonempty: {output}")
    for path in (train_file, eval_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (upstream_repo / "agents/behavior_chain/agent.py").is_file():
        raise FileNotFoundError(f"BehaviorChain agent not found under {upstream_repo}")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint is not a complete HF model: {checkpoint}")
    if eval_limit < 0:
        raise ValueError("eval_limit must be >= 0 (0 means the full validation set)")

    template_kwargs = {"enable_thinking": False, "thinking_budget": 0}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if not tokenizer.chat_template:
        raise ValueError("Checkpoint tokenizer has no chat template")
    hashes = source_hashes(upstream_repo, TASK)

    # Load evaluation first so exact held-out prompt duplicates can be removed
    # from training. This keeps step-0 and all periodic reports comparable.
    evaluation, eval_counts, eval_grouping = _load_split(
        eval_file, "eval", upstream_repo, hashes, tokenizer,
        max_prompt_tokens, template_kwargs,
    )
    training, train_counts, train_grouping = _load_split(
        train_file, "train", upstream_repo, hashes, tokenizer,
        max_prompt_tokens, template_kwargs,
    )
    eval_ids = {row["id"] for row in evaluation}
    before = len(training)
    training = [row for row in training if row["id"] not in eval_ids]
    train_counts["excluded_eval_prompt_duplicate"] = before - len(training)
    if not training:
        raise ValueError("No training rows remain after held-out prompt de-duplication")

    if eval_limit and len(evaluation) > eval_limit:
        rng = random.Random(seed)
        evaluation = sorted(rng.sample(evaluation, eval_limit), key=lambda row: row["id"])

    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "train.jsonl", training)
    write_rows(output / "eval.jsonl", evaluation)
    report = {
        "task": TASK,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "upstream_repo": str(upstream_repo),
        "train_source": str(train_file),
        "eval_source": str(eval_file),
        "max_prompt_tokens": max_prompt_tokens,
        "eval_limit": eval_limit,
        "exported": {"train": len(training), "eval": len(evaluation)},
        "counts": {"train": dict(train_counts), "eval": dict(eval_counts)},
        "grouping": {"train": dict(train_grouping), "eval": dict(eval_grouping)},
        "group_overlap": len({r["group_id"] for r in training} & {r["group_id"] for r in evaluation}),
        "source_hashes": hashes,
        "train_sha256": hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest(),
        "eval_sha256": hashlib.sha256((output / "eval.jsonl").read_bytes()).hexdigest(),
        "evaluation": "fixed greedy objective node-choice accuracy; no API judge",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def make_config(checkpoint, prepared, output, config, rank=16, alpha=None,
                max_steps=200, eval_every=50, num_gpus=8, batch_size=1,
                gradient_accumulation_steps=4, learning_rate=5e-5,
                max_prompt_tokens=8192, max_new_tokens=32, inference_batch_size=1):
    checkpoint = str(Path(checkpoint).resolve())
    prepared = Path(prepared).resolve()
    output = str(Path(output).resolve())
    config = Path(config).resolve()
    if rank not in (8, 16):
        raise ValueError("LoRA rank must be 8 or 16")
    if alpha is None:
        alpha = rank * 2
    if max_steps < 1 or eval_every < 1 or max_steps % eval_every:
        raise ValueError("max_steps must be a positive multiple of eval_every")
    if num_gpus < 1 or batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("GPU and batch settings must be positive")
    for name in ("train.jsonl", "eval.jsonl", "report.json"):
        if not (prepared / name).is_file():
            raise FileNotFoundError(prepared / name)
    if config.exists():
        raise FileExistsError(config)

    cfg = {
        "seed": 42,
        "output_dir": output,
        "model": {
            "base_model": checkpoint,
            "tokenizer": checkpoint,
            "student_mode": "lora",
            "lora_rank": rank,
            "lora_alpha": alpha,
            "lora_targets": "all-linear",
            "lora_dropout": 0.0,
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
            "chat_template_kwargs": {"enable_thinking": False, "thinking_budget": 0},
        },
        "teacher": {"device": "auto", "dtype": "bfloat16"},
        "data": {
            "train_file": str(prepared / "train.jsonl"),
            "eval_file": str(prepared / "eval.jsonl"),
            "dimension": "F",
            "task_weights": {TASK: 1.0},
        },
        "routing": {"tasks": {}},
        "rollout": {
            "context_length": max_prompt_tokens + max_new_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "temperature": 1.0,
            "max_turns": 1,
            "environment_factory": None,
            "generation_batch_size": inference_batch_size,
        },
        "inference": {"batch_size": inference_batch_size},
        "train": {
            "stage": "warmup",
            "max_steps": max_steps,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "global_prompt_batch": batch_size * gradient_accumulation_steps * num_gpus,
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "warmup_steps": min(10, max_steps),
            "save_every": eval_every,
            "eval_every": eval_every,
            "eval_limit_per_task": 0,
            "objective": "sampled_reverse_kl",
            "trajectory_source": "student",
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
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return {"config": str(config), "output_dir": output,
            "lora_rank": rank, "lora_alpha": alpha,
            "effective_prompt_batch": cfg["train"]["global_prompt_batch"]}


def evaluate(config, model, output):
    from .config import load_config
    from .workflows import evaluate_model

    cfg = copy.deepcopy(load_config(config))
    model = str(Path(model).resolve())
    # Evaluate the supplied full checkpoint itself. Do not accidentally wrap a
    # fresh, untrained adapter around the step-0 or merged model.
    cfg["model"].update({
        "base_model": model,
        "student_init": model,
        "tokenizer": model,
        "student_mode": "full",
        "gradient_checkpointing": False,
    })
    return evaluate_model(cfg, str(Path(output).resolve()), model_path=model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--train-file", required=True)
    prepare.add_argument("--eval-file", required=True)
    prepare.add_argument("--upstream-repo", required=True)
    prepare.add_argument("--checkpoint", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--eval-limit", type=int, default=0)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-prompt-tokens", type=int, default=8192)

    config = sub.add_parser("make-config")
    config.add_argument("--checkpoint", required=True)
    config.add_argument("--prepared", required=True)
    config.add_argument("--output", required=True)
    config.add_argument("--config", required=True)
    config.add_argument("--rank", type=int, choices=(8, 16), default=16)
    config.add_argument("--alpha", type=int)
    config.add_argument("--max-steps", type=int, default=200)
    config.add_argument("--eval-every", type=int, default=50)
    config.add_argument("--num-gpus", type=int, default=8)
    config.add_argument("--batch-size", type=int, default=1)
    config.add_argument("--gradient-accumulation-steps", type=int, default=4)
    config.add_argument("--learning-rate", type=float, default=5e-5)
    config.add_argument("--max-prompt-tokens", type=int, default=8192)
    config.add_argument("--max-new-tokens", type=int, default=32)
    config.add_argument("--inference-batch-size", type=int, default=1)

    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--output", required=True)

    args = vars(parser.parse_args())
    command = args.pop("command")
    result = {"prepare": prepare_data, "make-config": make_config,
              "evaluate": evaluate}[command](**args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
