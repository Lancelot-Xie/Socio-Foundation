"""Opt-in left trimming of adapter-declared material, never arbitrary messages."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from hashlib import sha256

from .contracts import ModelRequest
from .errors import ConfigurationError


POLICY = "protected_left_v1"
SPANS_KEY = "protected_left_v1_spans"


def validate_input_truncation(value: object) -> str | None:
    if value is not None and value != POLICY:
        raise ConfigurationError(f"token_accounting.input_truncation must be {POLICY!r}")
    return value


def truncate_material(
    request: ModelRequest, *, count_prompt: Callable[[ModelRequest], int],
    prompt_tokens: int, context_tokens: int, reserved_output_tokens: int,
) -> tuple[ModelRequest, dict]:
    """Return a copy fitting the full template, or the untouched original.

    Remove Unicode character prefixes, checking actual tokenizer counts on the
    complete request. No token decode/re-encode can corrupt a Unicode boundary.
    Binary search maintains a verified fitting upper bound; it need not assume
    token counts are strictly monotonic to guarantee a safe final request.
    """
    target = context_tokens - reserved_output_tokens
    audit = {"policy": POLICY, "request_id": request.request_id,
             "prompt_tokens_before": prompt_tokens,
             "reserved_output_tokens": reserved_output_tokens,
             "target_prompt_tokens": target}
    spans = []
    for index, message in enumerate(request.messages):
        if message.role != "user":
            continue
        previous_end = 0
        for raw in message.metadata.get(SPANS_KEY, ()):
            if not isinstance(raw, Mapping):
                raise ConfigurationError("invalid input truncation span")
            start, end = raw.get("start"), raw.get("end")
            if (type(start) is not int or type(end) is not int
                    or not previous_end <= start <= end <= len(message.content)):
                raise ConfigurationError("invalid or overlapping input truncation offsets")
            previous_end = end
            if end > start:
                spans.append((index, start, end, str(raw.get("field", ""))))
    if not spans:
        return request, {**audit, "status": "no_removable_material"}

    def candidate(remove: int) -> tuple[ModelRequest, list[dict]]:
        edits: dict[int, list[tuple[int, int]]] = {}
        removed = []
        for index, start, end, field in spans:
            take = min(remove, end - start)
            if take:
                edits.setdefault(index, []).append((start, start + take))
                removed.append({"message_index": index, "field": field,
                                "start_character": start, "removed_characters": take})
                remove -= take
        messages = list(request.messages)
        for index, ranges in edits.items():
            message = messages[index]
            content = message.content
            for start, end in reversed(ranges):
                content = content[:start] + content[end:]
            # These offsets belong to the original text, not the trimmed copy.
            metadata = {key: value for key, value in message.metadata.items() if key != SPANS_KEY}
            messages[index] = replace(message, content=content, metadata=metadata)
        return replace(request, messages=tuple(messages)), removed

    maximum = sum(end - start for _, start, end, _ in spans)
    smallest, _ = candidate(maximum)
    protected_tokens = count_prompt(smallest)
    if protected_tokens > target:
        return request, {**audit, "status": "protected_material_too_long",
                         "protected_prompt_tokens": protected_tokens}
    low, high = 0, maximum
    while low < high:
        mid = (low + high) // 2
        trial, _ = candidate(mid)
        if count_prompt(trial) <= target:
            high = mid
        else:
            low = mid + 1
    result, removed = candidate(high)
    final_tokens = count_prompt(result)
    if final_tokens > target:
        raise ConfigurationError("input truncation token counts changed during fitting")
    hashes = [{"message_index": i,
               "before_sha256": sha256(request.messages[i].content.encode()).hexdigest(),
               "after_sha256": sha256(result.messages[i].content.encode()).hexdigest()}
              for i in sorted({item["message_index"] for item in removed})]
    return result, {**audit, "status": "applied", "prompt_tokens_after": final_tokens,
                    "removed_prompt_tokens": prompt_tokens - final_tokens,
                    "removed_regions": removed, "message_hashes": hashes}
