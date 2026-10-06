"""In-training validation on replicated DDP/ZeRO-2 weights, with isolated RNG.

Ranks generate disjoint validation rows; rank 0 merges once (including aggregate
UserLM F1). No extra model/backbone is loaded and validation never enters the loss.
"""

import contextlib
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .checkpoints import archive_incomplete, atomic_json
from .data import load_data, write_rows
from .models import generation_model
from .workflows import evaluate_rows, summarize_evaluation


@contextlib.contextmanager
def isolated_rng(seed, device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def validation_rows(cfg):
    rows = load_data(cfg["data"]["eval_file"], cfg["data"]["dimension"])
    limit = cfg["train"]["eval_limit_per_task"]
    if not limit:
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    rng = random.Random(cfg["seed"])
    # Fixed subset across steps; never select examples by current model scores.
    return [row for task in sorted(groups) for row in rng.sample(groups[task], min(limit, len(groups[task])))]


def periodic_validation(accelerator, model, tokenizer, cfg, output, step):
    output = Path(output) / "validation"
    folder = output / f"step_{step:06d}"
    temporary = output / f".step_{step:06d}.pending"
    accelerator.wait_for_everyone()
    # On resume a checkpoint may already have this step's committed validation.
    if (folder / "complete.json").is_file():
        return
    if accelerator.is_main_process:
        for path in (folder, temporary):
            if path.exists():
                archive_incomplete(path)
        temporary.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    started = time.monotonic()
    rows = validation_rows(cfg)
    local_rows = rows[accelerator.process_index::accelerator.num_processes]
    was_training = model.training
    try:
        model.eval()
        with isolated_rng(cfg["seed"] + accelerator.process_index, accelerator.device):
            with generation_model(model, accelerator) as generator:
                records = evaluate_rows(cfg, generator, tokenizer, local_rows, accelerator.device)
    finally:
        model.train(was_training)
    # Files avoid all_gather_object replicating large response strings on every GPU.
    write_rows(temporary / f"rank_{accelerator.process_index}.jsonl", records)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        from .data import read_rows
        records = [record for rank in range(accelerator.num_processes)
                   for record in read_rows(temporary / f"rank_{rank}.jsonl")]
        order = {row["id"]: index for index, row in enumerate(rows)}
        if len(records) != len(rows) or len({r["id"] for r in records}) != len(rows):
            raise ValueError("Distributed validation has missing or duplicate rows")
        records.sort(key=lambda r: order[r["id"]])
        write_rows(temporary / "responses.jsonl", records)
        summary = summarize_evaluation(cfg, records)
        atomic_json(temporary / "summary.json", summary)
        atomic_json(temporary / "complete.json", {
            "step": step, "rows": len(rows), "world_size": accelerator.num_processes,
            "eval_file": cfg["data"]["eval_file"], "limit_per_task": cfg["train"]["eval_limit_per_task"],
            "elapsed_seconds": time.monotonic() - started, "sampling": "greedy", "feeds_optimizer": False,
        })
        temporary.rename(folder)
        print(f"[validation] step={step} rows={len(rows)} summary={summary} "
              f"report={folder / 'summary.json'}", flush=True)
    accelerator.wait_for_everyone()
