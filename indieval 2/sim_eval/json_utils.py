"""Canonical JSON and digest helpers used by manifests and cache identities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any


UNICODE_SANITIZATION_REVISION = "unicode-scalar-normalization-v1"


def _sanitize_unicode_text(text: str, stats: dict[str, int]) -> str:
    """Return valid Unicode scalar text, preserving legitimate non-BMP text."""

    output: list[str] = []
    index = 0
    while index < len(text):
        codepoint = ord(text[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 < len(text):
                following = ord(text[index + 1])
                if 0xDC00 <= following <= 0xDFFF:
                    output.append(
                        chr(
                            0x10000
                            + ((codepoint - 0xD800) << 10)
                            + (following - 0xDC00)
                        )
                    )
                    stats["surrogate_pairs_normalized"] += 1
                    index += 2
                    continue
            output.append("\ufffd")
            stats["unpaired_surrogates_replaced"] += 1
        elif 0xDC00 <= codepoint <= 0xDFFF:
            output.append("\ufffd")
            stats["unpaired_surrogates_replaced"] += 1
        else:
            output.append(text[index])
        index += 1
    return "".join(output)


def sanitize_unicode_scalars(value: Any) -> tuple[Any, dict[str, int]]:
    """Recursively make strings UTF-8 encodable and report exact repairs.

    JSON providers occasionally emit escaped lone UTF-16 surrogates.  Python's
    JSON decoder accepts them, but UTF-8 artifact writes correctly reject them.
    Valid surrogate pairs are combined into their Unicode scalar; only unpaired
    surrogates are replaced with U+FFFD.
    """

    stats = {
        "unpaired_surrogates_replaced": 0,
        "surrogate_pairs_normalized": 0,
    }

    def visit(item: Any) -> Any:
        if isinstance(item, str):
            return _sanitize_unicode_text(item, stats)
        if isinstance(item, dict):
            return {visit(key): visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, tuple):
            return tuple(visit(child) for child in item)
        if isinstance(item, set):
            return {visit(child) for child in item}
        return item

    return visit(value), stats


def jsonable(value: Any) -> Any:
    """Convert framework values into deterministic JSON-compatible objects."""

    if dataclasses.is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(jsonable(item) for item in value)
    return value


def canonical_json(value: Any) -> str:
    """Serialize with stable ordering and no insignificant whitespace."""

    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
