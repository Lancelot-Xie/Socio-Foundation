"""Fresh student rollout with optional real environment transitions."""

import importlib
import inspect

import torch
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin

from .data import encode_prompt


_NO_MODEL_DEFAULTS = ({"use_model_defaults": False} if
                      "use_model_defaults" in inspect.signature(GenerationMixin.generate).parameters else {})


def load_callable(spec):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


@torch.no_grad()
def generate_response(model, prompt_ids, tokenizer, cfg, device, sample=True):
    return generate_responses(model, [prompt_ids], tokenizer, cfg, device, sample)[0]


@torch.no_grad()
def generate_responses(model, prompts, tokenizer, cfg, device, sample=True):
    """Left-padded generation, bounded per call; preserve input order and real EOS.

    Callers must group requests by model/adapter. No concurrent adapter mutation.
    Sampling streams change with batch shape; unchanged runs remain resumable.
    """
    responses = []
    size = cfg["rollout"]["generation_batch_size"]
    settings = GenerationConfig(
        max_new_tokens=cfg["rollout"]["max_new_tokens"], do_sample=sample,
        temperature=cfg["rollout"]["temperature"] if sample else 1.0,
        top_p=1.0, top_k=0, repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, use_cache=True,
    )
    for start in range(0, len(prompts), size):
        chunk = prompts[start:start + size]
        if any(not p for p in chunk):
            raise ValueError("Cannot generate from an empty prompt")
        width = max(map(len, chunk))
        inputs = torch.full((len(chunk), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(inputs)
        for i, prompt in enumerate(chunk):
            inputs[i, -len(prompt):] = torch.tensor(prompt, device=device)
            mask[i, -len(prompt):] = 1
        output = model.generate(input_ids=inputs, attention_mask=mask,
                                generation_config=settings, synced_gpus=False, **_NO_MODEL_DEFAULTS)
        for generated in output[:, width:].tolist():
            if tokenizer.eos_token_id in generated:
                generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
            if not generated:
                raise RuntimeError("Model generated no tokens")
            responses.append(generated)
    return responses


def collect_episodes(model, rows, tokenizer, cfg, device, sample=True):
    """Batch independent single-turn tasks; keep real environments sequential."""
    if cfg["rollout"]["environment_factory"]:
        return [collect_episode(model, row, tokenizer, cfg, device, sample) for row in rows]
    prompts = [encode_prompt(tokenizer, row["messages"], cfg) for row in rows]
    responses = generate_responses(model, prompts, tokenizer, cfg, device, sample)
    return [([{"prompt_ids": prompt, "response_ids": response, "row": row, "turn": 0,
               "messages": [dict(m) for m in row["messages"]]}], tokenizer.decode(response, skip_special_tokens=True))
            for row, prompt, response in zip(rows, prompts, responses, strict=True)]


def collect_episode(model, row, tokenizer, cfg, device, sample=True):
    factory = cfg["rollout"]["environment_factory"]
    env = load_callable(factory)(row) if factory else None
    messages = env.reset() if env else row["messages"]
    transitions, outputs = [], []
    for turn in range(cfg["rollout"]["max_turns"]):
        prompt = encode_prompt(tokenizer, messages, cfg)
        response = generate_response(model, prompt, tokenizer, cfg, device, sample)
        text = tokenizer.decode(response, skip_special_tokens=True)
        transitions.append({"prompt_ids": prompt, "response_ids": response, "row": row, "turn": turn,
                            "messages": [dict(m) for m in messages]})
        outputs.append(text)
        if env is None:
            break
        # Environment must execute the student's action and return the NEXT full visible conversation.
        state = env.step(text)
        transitions[-1]["next_messages"] = [dict(m) for m in state.get("messages", [])]
        transitions[-1]["done"] = state["done"]
        if "reward" in state:
            transitions[-1]["reward"] = state["reward"]
        if state["done"]:
            break
        messages = state["messages"]
    return transitions, "\n".join(outputs)
