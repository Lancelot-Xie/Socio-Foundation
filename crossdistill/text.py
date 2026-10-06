"""Text is the only interface between incompatible teacher/student vocabularies."""
import copy

import torch
from transformers import GenerationConfig


def stop_ids(model, tokenizer):
    value = model.generation_config.eos_token_id
    result = list(value) if isinstance(value, (list, tuple)) else [value]
    result.append(tokenizer.eos_token_id)
    return list(dict.fromkeys(i for i in result if i is not None))


def prompt_ids(tokenizer, messages, kwargs):
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)


@torch.no_grad()
def generate(model, tokenizer, ids, budget, device, *, sample=True, temperature=1.0):
    if budget < 1:
        return [], False
    stops = stop_ids(model, tokenizer)
    settings = GenerationConfig(max_new_tokens=budget, do_sample=sample,
                                temperature=temperature if sample else 1.0,
                                top_k=0, top_p=1.0, eos_token_id=stops,
                                pad_token_id=tokenizer.pad_token_id, use_cache=True)
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            generation_config=settings, use_model_defaults=False,
                            synced_gpus=False)[0, len(ids):].tolist()
    ended = bool(result and result[-1] in stops)
    if ended:
        result = result[:-1]
    return result, ended


def continuation_example(tokenizer, prompt, prefix, continuation, ended, max_tokens, eos_id):
    """Mask prompt and prefix; mask any BPE token crossing the text boundary too.

    Prefix and continuation must be tokenized jointly: independently encoding
    either side can change the byte-level BPE boundary. No teacher IDs survive.
    Length-limited teacher generations do not acquire a fabricated EOS label.
    """
    if not tokenizer.is_fast:
        raise ValueError("A fast student tokenizer with offset mappings is required")
    encoded = tokenizer(prefix + continuation, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded['input_ids'])
    mask = [int(start >= len(prefix) and end > start)
            for start, end in encoded['offset_mapping']]
    if ended:
        ids.append(eos_id)
        mask.append(1)
    available = max_tokens - len(prompt)
    if available < 1:
        raise ValueError("No response capacity; lower the prompt/prefix budget")
    truncated = len(ids) > available
    ids, mask = ids[:available], mask[:available]
    # Empty textual continuations (e.g. a special-token-only generation) can
    # occur. Caller retries; silently inserting an EOS would change the target.
    if not any(mask):
        raise ValueError("Teacher continuation has no trainable student tokens")
    return ids, mask, truncated


def teacher_config(cfg):
    result = copy.deepcopy(cfg)
    result['model']['base_model'] = cfg['crossdistill']['teacher_base']
    result['model'].pop('student_init', None)
    result['model']['tokenizer'] = cfg['crossdistill'].get('teacher_tokenizer', cfg['crossdistill']['teacher_base'])
    result['model']['chat_template_kwargs'] = cfg['crossdistill']['teacher_chat_kwargs']
    return result
