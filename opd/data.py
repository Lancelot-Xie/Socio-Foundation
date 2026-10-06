"""Explicit prompt data, task-balanced sampling and response-only batches."""

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    elif path.suffix == ".json":
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError("JSON data must be a list")
        yield from rows
    else:
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_row(row, index):
    task = row.get("task_id", row.get("data_source"))
    messages = row.get("messages", row.get("prompt"))
    if "messages" not in row and "extra_info" in row and messages == [{"role": "user", "content": "x"}]:
        raise ValueError(f"Row {index}: raw Simulation placeholder prompt; export it through the original task agent")
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not task or not isinstance(messages, list) or not messages:
        raise ValueError(f"Row {index}: need task_id/data_source and explicit messages/prompt; "
                         "raw Simulation extra_info is not a rendered prompt")
    if any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool")
           or not isinstance(m.get("content"), str) for m in messages):
        raise ValueError(f"Row {index}: only text chat messages are supported")
    if messages[-1]["role"] == "assistant":
        raise ValueError(f"Row {index}: prompt ends with an assistant answer; move it to response for warmup")
    strength = row.get("distill_strength", 1.0)
    if not isinstance(strength, (int, float)) or not math.isfinite(strength) or strength < 0:
        raise ValueError(f"Row {index}: invalid distill_strength")
    dimensions = row.get("dimensions", [])
    if not isinstance(dimensions, list) or any(not isinstance(d, str) or d not in list("FSUTNC") for d in dimensions):
        raise ValueError(f"Row {index}: dimensions must be a list drawn from F/S/U/T/N/C")
    return {**row, "id": str(row.get("id", f"{task}-{index}")), "task_id": str(task),
            "messages": messages, "dimensions": dimensions, "distill_strength": float(strength)}


def load_data(path, dimension=None, require_response=False):
    rows = [normalize_row(r, i) for i, r in enumerate(read_rows(path))]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample ids")
    if dimension:
        rows = [r for r in rows if dimension in r["dimensions"]]
    if not rows:
        raise ValueError("No usable rows after dimension filtering")
    if require_response and any(not isinstance(r.get("response"), str) or not r["response"] for r in rows):
        raise ValueError("Warmup needs a nonempty response on every row; run build-demos first")
    return rows


class TaskSampler:
    def __init__(self, rows, weights, seed):
        self.groups = defaultdict(list)
        for row in rows:
            self.groups[row["task_id"]].append(row)
        if set(weights) - set(self.groups):
            raise ValueError("Task weights include tasks absent after data filtering")
        self.tasks = sorted(self.groups)
        self.weights = [weights.get(t, 1.0) for t in self.tasks]
        self.rng = random.Random(seed)
        # Multiple candidates/correction rounds must not multiply a source prompt's sampling mass.
        self.sources = {}
        for task, items in self.groups.items():
            sources = defaultdict(list)
            for row in items:
                sources[row.get("source_id", row["id"])].append(row)
            self.sources[task] = list(sources.values())

    def sample(self, n):
        return [self.rng.choice(self.rng.choice(self.sources[t]))
                for t in self.rng.choices(self.tasks, self.weights, k=n)]

    def state_dict(self):
        return {"rng": self.rng.getstate()}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])


def encode_prompt(tokenizer, messages, cfg):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        **cfg["model"]["chat_template_kwargs"])
    if len(ids) > cfg["rollout"]["max_prompt_tokens"]:
        raise ValueError(f"Prompt has {len(ids)} tokens, exceeding max_prompt_tokens; "
                         "truncate explicitly before training to preserve persona/system boundaries")
    if not ids:
        raise ValueError("Empty tokenized prompt")
    return ids


def tokenizer_signature(tokenizer, cfg):
    content = {"vocab": tokenizer.get_vocab(), "template": tokenizer.chat_template,
               "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id,
               "template_kwargs": cfg["model"]["chat_template_kwargs"]}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def warmup_response(row, tokenizer, cfg, signature):
    if "response_token_ids" in row:
        if row.get("tokenizer_signature") != signature:
            raise ValueError("Demonstration token ids were generated with a different tokenizer/template")
        ids = row["response_token_ids"]
        if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < len(tokenizer) for i in ids):
            raise ValueError("Invalid demonstration response_token_ids")
        # Preserve EOS when emitted, and do not invent EOS on max-length truncation.
        if row.get("prompt_token_ids") != encode_prompt(tokenizer, row["messages"], cfg):
            raise ValueError("Demonstration prefix differs from the current chat template")
    else:
        ids = tokenizer.encode(row["response"], add_special_tokens=False)
        if not ids or ids[-1] != tokenizer.eos_token_id:
            ids.append(tokenizer.eos_token_id)
    if len(ids) > cfg["rollout"]["max_new_tokens"]:
        raise ValueError("Warmup response exceeds max_new_tokens; filter or increase the limit")
    return ids


def make_batch(prompt_ids, response_ids, pad_id, device):
    """Left-pad prompts, right-pad responses; EOS stays in the loss even when EOS==PAD."""
    import torch
    if not prompt_ids or any(not p for p in prompt_ids) or any(not r for r in response_ids):
        raise ValueError("Every sample needs nonempty prompt and response tokens")
    p_len, r_len = max(map(len, prompt_ids)), max(map(len, response_ids))
    ids, attn, masks = [], [], []
    for p, r in zip(prompt_ids, response_ids, strict=True):
        ids.append([pad_id] * (p_len-len(p)) + p + r + [pad_id] * (r_len-len(r)))
        attn.append([0] * (p_len-len(p)) + [1] * (len(p)+len(r)) + [0] * (r_len-len(r)))
        masks.append([1] * len(r) + [0] * (r_len-len(r)))
    attention = torch.tensor(attn, device=device)
    positions = (attention.cumsum(-1)-1).clamp_min(0)
    return {"input_ids": torch.tensor(ids, device=device), "attention_mask": attention,
            "position_ids": positions, "response_mask": torch.tensor(masks, device=device).float(),
            "responses": torch.tensor([r+[pad_id]*(r_len-len(r)) for r in response_ids], device=device),
            "prompt_length": p_len}
