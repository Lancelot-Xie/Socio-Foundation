"""Render existing task material without selecting, inferring, or truncating it."""

from collections.abc import Mapping, Sequence

from .contracts import ChatMessage
from typing import Any


def material_text(value: Any) -> str:
    """Keep every value and its order; express nested structure with labels."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "\n".join(f"{key}:\n{material_text(item)}" for key, item in value.items()) or "{}"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "\n\n".join(f"Item {index + 1}:\n{material_text(item)}" for index, item in enumerate(value)) or "[]"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def material_sections(payload: Mapping[str, Any], titles: Mapping[str, str]) -> str:
    """Render all fields, including empty ones, with benchmark-specific titles."""
    sections = []
    for key, value in payload.items():
        if key in {"options", "candidates", "responses"}:
            body = "\n".join(f"{choice['id']}. {choice['text']}" for choice in value)
        else:
            body = material_text(value)
        sections.append(f"# {titles[key]}\n{body}")
    return "\n\n".join(sections)


def material_user_message(
    payload: Mapping[str, Any], titles: Mapping[str, str], *,
    truncatable: Sequence[str], suffix: str,
) -> ChatMessage:
    """Preserve the wire prompt; annotate removable bodies using exact offsets.

    Section headings, unlisted fields, and suffix instructions are protected.
    Offsets are private metadata, never inferred from user-supplied headings.
    """
    parts = []
    spans = []
    offset = 0
    for key, value in payload.items():
        section = material_sections({key: value}, titles)
        if parts:
            offset += 2
        if key in truncatable:
            spans.append({"field": key, "start": offset + len(f"# {titles[key]}\n"),
                          "end": offset + len(section)})
        parts.append(section)
        offset += len(section)
    return ChatMessage("user", "\n\n".join(parts) + suffix,
                       metadata={"protected_left_v1_spans": spans})
