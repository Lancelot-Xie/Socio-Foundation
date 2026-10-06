"""Explicit, versioned adapters for evaluated models; absent means legacy behavior.

These settings are never inferred from a model name and never apply to support
roles.  Request transforms happen before context accounting.  Raw model text is
retained; choice normalization is only a parser input, never a rewritten record.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Mapping, Sequence

from .contracts import ModelRequest, ModelResponse
from .errors import ConfigurationError

USERLM_NATIVE = "userlm_native_v1"
COSER_FORMAT = "coser_format_v1"
MODEL_ADAPTERS = frozenset({USERLM_NATIVE, COSER_FORMAT})


def validate_model_adapter(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in MODEL_ADAPTERS:
        raise ConfigurationError(f"model_adapter must be one of {sorted(MODEL_ADAPTERS)}")
    return value


def adaptation_identity(role: Mapping[str, Any]) -> dict[str, str]:
    value = validate_model_adapter(role.get("model_adapter"))
    return {"model_adapter": value} if value else {}


def adapt_evaluated_request(request: ModelRequest, adapter: str | None) -> ModelRequest:
    if not adapter:
        return request
    validate_model_adapter(adapter)
    if request.metadata.get("model_adapter") == adapter:
        return request
    messages = request.messages
    if adapter == USERLM_NATIVE:
        # Framework histories use assistant=self and user=peer.  The native
        # UserLM template generates user=self, so reverse only API role labels.
        roles = {"assistant": "user", "user": "assistant"}
        messages = tuple(replace(m, role=roles.get(m.role, m.role)) for m in messages)
    return replace(request, messages=messages,
                   metadata={**request.metadata, "model_adapter": adapter})


def annotate_adapted_response(response: ModelResponse, adapter: str | None) -> ModelResponse:
    if not adapter:
        return response
    raw = dict(response.raw or {})
    raw["_sim_eval"] = {**dict(raw.get("_sim_eval") or {}), "model_adapter": adapter}
    return replace(response, raw=raw)


def adapted_choice_text(response: ModelResponse, choices: Sequence[Any], *, ranking: bool = False) -> str:
    """Accept only unambiguous labels or label + exact matching option text.

    Do not search arbitrary prose for letters, ignore contradictory text, repair
    incomplete answer tags, or use gold labels.  Complete original tags/JSON keep
    their original parser.  Ranked answers still require exactly five distinct
    candidates in HumanLLM's adapter.
    """
    raw = response.raw or {}
    if (raw.get("_sim_eval") or {}).get("model_adapter") not in MODEL_ADAPTERS:
        return response.text
    text = response.text.strip()
    if "<answer" in text.casefold() or text.startswith("{"):
        return response.text
    labels = {c.display_id.casefold(): c for c in choices}
    if ranking:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) == 5 and len({p.casefold() for p in parts}) == 5 and all(p.casefold() in labels for p in parts):
            return "<answer>" + ",".join(labels[p.casefold()].display_id for p in parts) + "</answer>"
        return response.text
    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)(?:[.)](?:\s+(.+))?)?", text, flags=re.DOTALL)
    if not match or match[1].casefold() not in labels:
        return response.text
    choice = labels[match[1].casefold()]
    if match[2] is not None:
        normalize = lambda s: " ".join(s.split()).casefold()
        if normalize(match[2]) != normalize(choice.text):
            return response.text
    return f"<answer>{choice.display_id}</answer>"
