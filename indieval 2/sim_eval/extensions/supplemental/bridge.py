"""Run frozen evaluation loops through the existing ModelBackend contract.

This replaces training/tokenization plumbing only. No credentials, SDK clients,
environment variables, monkeypatches, or global model routing are used here.
Each episode has its own context, including when the shared runner is threaded.
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import SimpleNamespace

from ...contracts import ChatMessage, ModelRequest, TraceEvent
from ...errors import BackendError, ParseError
from ...benchmarks.common import json_schema_response_format
from .vendor.text_utils import remove_think

ACTIVE: ContextVar = ContextVar("supplemental_episode")


@dataclass
class Session:
    backend: object
    case: object
    model: str
    seed: int
    evaluated_role: str
    roles: dict = field(default_factory=dict)
    counters: dict = field(default_factory=lambda: defaultdict(int))
    responses: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    support_failures: list = field(default_factory=list)
    last_candidate_response: object = None

    def generate(self, messages, *, role, model=None, reasoning_effort=None, response_format=None):
        index = self.counters[role]
        self.counters[role] += 1
        request = ModelRequest(
            request_id=f"{self.case.case_id}:{role}:{index}",
            messages=tuple(ChatMessage(m["role"], m.get("content") or "") for m in messages),
            model=self.model if role == self.evaluated_role else self.roles.get(role, {}).get("model", model or role),
            seed=self.seed,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            metadata={"benchmark_id": self.case.benchmark_id, "actor": role, "route_role": role},
        )
        self.trace.append(TraceEvent(index, role, "request", {
            "request_id": request.request_id,
            "messages": [m.to_chat_dict() for m in request.messages],
            "response_format": response_format,
        }, visible_to=("evaluator",)))
        response = self.backend.generate(request)
        self.responses.append(response)
        if role == self.evaluated_role:
            self.last_candidate_response = response
        self.trace.append(TraceEvent(index, role, "response", response.text, visible_to=("evaluator",)))
        return response


class Agent:
    """The subset of Supplemental Agent used by the frozen evaluation-only loops."""

    def __init__(self, llm_client, chat, tokenizer, config, **kwargs):
        self.chat = copy.deepcopy(chat)

    async def step(self, **kwargs):
        session = ACTIVE.get()
        response = session.generate(self.chat, role=session.evaluated_role)
        self.chat.append({"role": "assistant", "content": response.text})
        return response.text

    def append(self, message):
        self.chat.append(copy.deepcopy(message))

    async def get_agent_output(self, reward, extra_info=None):
        return SimpleNamespace(reward_score=reward, extra_fields={"reward_extra_info": extra_info or {}})


async def process_post_chat(*args, **kwargs):
    # The shared runner owns traces, checkpoints and reports.
    return None


def get_judge_model(default):
    return ACTIVE.get().roles.get("judge", {}).get("model", default)


def get_judge_reasoning(default):
    return ACTIVE.get().roles.get("judge", {}).get("generation", {}).get("reasoning_effort", default)


async def call_openai(messages, model="gpt-5-nano", reasoning_effort="minimal", **kwargs):
    # Infrastructure failures stay structured framework failures; never put an
    # HTTP error string into the simulated conversation as an assistant reply.
    return ACTIVE.get().generate(messages, role="fixed_assistant", model=model,
                                 reasoning_effort=reasoning_effort).text


def _strict_schema(schema):
    schema = copy.deepcopy(schema)
    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(schema)
    return schema


async def call_openai_parse(messages, text_format, model="gpt-5-nano", max_retries=3,
                            reasoning_effort="low", **kwargs):
    session = ACTIVE.get()
    schema = _strict_schema(text_format.model_json_schema())
    contract = json_schema_response_format(text_format.__name__.lstrip("_"), schema)
    from pydantic import ValidationError
    for attempt in range(max_retries + 1):
        try:
            response = session.generate(messages, role="judge", model=model,
                                        reasoning_effort=reasoning_effort, response_format=contract)
            return text_format.model_validate_json(response.text).model_dump(by_alias=True)
        except (BackendError, ParseError, ValidationError, json.JSONDecodeError) as exc:
            session.support_failures.append({"role": "judge", "attempt": attempt,
                                             "kind": type(exc).__name__, "message": str(exc)})
    # Frozen Supplemental scoring explicitly handles None, including its original
    # fixed-denominator reward. Preserve it and disclose the judge failures.
    return None
