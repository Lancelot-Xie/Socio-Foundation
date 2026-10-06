"""Independent MirrorBench adapter with lexical and judge metric separation."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import comb, sqrt
from typing import Any, Hashable, Mapping, Sequence

from ..contracts import (
    BenchmarkCase,
    CaseResult,
    ChatMessage,
    ErrorState,
    MetricValue,
    ModelRequest,
    ModelResponse,
    ResultStatus,
)
from ..data.schemas import probe_case
from ..environments.dialogue import (
    DialogueAction,
    DialogueEnvironment,
    DialogueSpec,
    DialogueState,
    EVALUATED_USER_ROLE,
    FIXED_ASSISTANT_ROLE,
)
from ..errors import (
    BackendError,
    BackendTimeoutError,
    ConfigurationError,
    EpisodeTokenBudgetExhausted,
    OptionalDependencyError,
    ParseError,
    ValidationError,
)
from ..interfaces import BenchmarkAdapter, ModelBackend
from ..json_utils import canonical_json
from ..registry import adapter
from .common import (
    DEFAULT_CONTRACT_RETRIES,
    aggregate_named_metrics,
    combine_usage,
    corrective_retry_request,
    json_schema_response_format,
)


LEXICAL_METRICS = ("mattr", "hdd", "yules_k")
JUDGE_METRICS = ("gteval", "pi", "rnr")
_WORD = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", re.IGNORECASE)
_TAGGED_REASONING = re.compile(
    r"<(?:seed:)?think>.*?</(?:seed:)?think>|<(?:seed:)?think>.*\Z",
    re.DOTALL | re.IGNORECASE,
)


def _parse_raw_utterance(text: str, *, actor: str | None = None) -> DialogueAction:
    """Normalize one official MirrorBench raw turn into the shared dialogue action."""

    message = _TAGGED_REASONING.sub("", text).strip()
    if actor == EVALUATED_USER_ROLE and message.casefold().startswith("user:"):
        message = message[5:].strip()
    if not message:
        raise ParseError("MirrorBench actor output must be one non-empty raw utterance")
    return DialogueAction(action="message", message=message)


@dataclass(frozen=True)
class MirrorRuntimeProvenance:
    fixed_assistant_model: str
    fixed_assistant_revision: str
    assistant_policy_revision: str
    max_user_turns: int = 12
    max_total_actions: int = 32
    request_timeout_seconds: float = 600.0
    max_retries: int = 2
    user_temperature: float = 0.0
    assistant_temperature: float = 0.0
    generation_max_tokens: int = 2048
    source: str = "configured"

    def __post_init__(self) -> None:
        if not all(
            (
                self.fixed_assistant_model,
                self.fixed_assistant_revision,
                self.assistant_policy_revision,
                self.source,
            )
        ):
            raise ConfigurationError("MirrorBench runtime provenance fields cannot be empty")
        if self.max_user_turns <= 0 or self.max_total_actions <= 0 or self.generation_max_tokens <= 0:
            raise ConfigurationError("MirrorBench limits must be positive")
        if self.request_timeout_seconds <= 0 or not math.isfinite(self.request_timeout_seconds):
            raise ConfigurationError("MirrorBench timeout must be positive and finite")
        if self.max_retries < 0:
            raise ConfigurationError("MirrorBench max_retries cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MirrorScoringProvenance:
    tokenizer_policy: str
    tokenizer_model: str
    tokenizer_revision: str
    judge_model: str | None
    judge_revision: str | None
    judge_temperature: float
    judge_max_tokens: int
    gteval_prompt_revision: str
    pi_prompt_revision: str
    rnr_prompt_revision: str
    gteval_samples: int = 1
    pi_samples: int = 3
    rnr_samples: int = 2
    compute_controls: bool = True
    replayed: bool = False
    source: str = "configured"

    def __post_init__(self) -> None:
        if not all(
            (
                self.tokenizer_policy,
                self.tokenizer_model,
                self.tokenizer_revision,
                self.gteval_prompt_revision,
                self.pi_prompt_revision,
                self.rnr_prompt_revision,
                self.source,
            )
        ):
            raise ConfigurationError("MirrorBench scoring provenance fields cannot be empty")
        if bool(self.judge_model) != bool(self.judge_revision):
            raise ConfigurationError("MirrorBench judge model and revision must be configured together")
        if min(self.gteval_samples, self.pi_samples, self.rnr_samples) <= 0:
            raise ConfigurationError("MirrorBench judge sample counts must be positive")
        if self.judge_max_tokens <= 0:
            raise ConfigurationError("MirrorBench judge_max_tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def mirror_tokenize(text: str, *, policy: str, model: str) -> list[Hashable]:
    """Tokenize without silently substituting for the configured policy."""

    if policy == "regex_word_v1":
        return [token.casefold() for token in _WORD.findall(text)]
    if policy == "tiktoken_model":
        try:
            tiktoken = importlib.import_module("tiktoken")
        except ImportError as exc:
            raise OptionalDependencyError(
                "MirrorBench canonical tiktoken policy was requested but tiktoken is not installed"
            ) from exc
        encoding = tiktoken.encoding_for_model(model)
        # Candidate text is untrusted natural-language output.  A base model
        # may emit a literal string such as ``<|endoftext|>``; tiktoken's
        # default ``encode`` rejects such strings to protect prompt-building
        # callers.  MirrorBench is only measuring lexical diversity, so encode
        # every visible character as ordinary text.  For text without special
        # literals this is token-for-token identical to the previous path.
        return list(encoding.encode_ordinary(text))
    raise ConfigurationError(f"unknown MirrorBench tokenizer policy {policy!r}")


def mattr(tokens: Sequence[Hashable], window: int = 50) -> float:
    if window <= 0:
        raise ValidationError("MATTR window must be positive")
    length = len(tokens)
    if not length:
        return 0.0
    if length <= window:
        return len(set(tokens)) / length
    return statistics.mean(
        len(set(tokens[index : index + window])) / window
        for index in range(length - window + 1)
    )


def hdd(tokens: Sequence[Hashable], sample_size: int = 42) -> float:
    if sample_size <= 0:
        raise ValidationError("HD-D sample size must be positive")
    population = len(tokens)
    if not population:
        return 0.0
    sample = min(sample_size, population)
    denominator = comb(population, sample)
    diversity = sum(
        1.0 - comb(population - count, sample) / denominator
        for count in Counter(tokens).values()
    )
    return diversity / sample


def yules_k(tokens: Sequence[Hashable]) -> float:
    total = len(tokens)
    if not total:
        return 0.0
    sum_sq = sum(count * count for count in Counter(tokens).values())
    return 10_000.0 * (sum_sq - total) / (total * total)


def _lexical_values(tokens: Sequence[Hashable]) -> Mapping[str, float]:
    return {"mattr": mattr(tokens, 50), "hdd": hdd(tokens, 42), "yules_k": yules_k(tokens)}


def _metric(
    name: str,
    value: float | int | bool | None,
    *,
    direction: str = "higher_is_better",
    unit: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MetricValue:
    return MetricValue(
        name=name,
        value=value,
        direction=direction,
        unit=unit,
        metadata={
            "availability": "available" if value is not None else "unavailable",
            **dict(metadata or {}),
        },
    )


def _reference_conversation(case: BenchmarkCase) -> tuple[Mapping[str, str], ...]:
    raw = case.input_data.get("reference_conversation")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 2:
        raise ValidationError("MirrorBench requires a structured reference_conversation with at least two turns")
    result: list[Mapping[str, str]] = []
    previous: str | None = None
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValidationError(f"MirrorBench reference turn #{index} must be an object")
        role = str(item.get("role") or "").casefold()
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValidationError(f"MirrorBench reference turn #{index} is invalid")
        if role == previous:
            raise ValidationError("MirrorBench reference roles must alternate")
        result.append({"role": role, "content": content.strip()})
        previous = role
    if not any(item["role"] == "user" for item in result):
        raise ValidationError("MirrorBench reference must contain at least one user turn")
    return tuple(result)


def _format_conversation(items: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{str(item.get('role') or item.get('actor')).upper()}: {str(item.get('content') or item.get('message'))}"
        for item in items
    )


def _user_proxy_system_prompt(case: BenchmarkCase) -> str:
    lines = [
        "You are simulating a real human user for the MirrorBench evaluation harness.",
        "Respond with the next USER turn only. Do not write assistant messages, notes, or any other analysis.",
        "Your utterance should be like a real user and the context should be based on the following information provided.",
    ]
    task_description = str(case.input_data.get("task_description") or "").strip()
    domain = str(case.input_data.get("domain") or "").strip()
    persona = str(case.input_data.get("persona") or "").strip()
    if task_description:
        lines.append(f"Task description: {task_description}.")
    if domain:
        lines.append(f"Domain or topic: {domain}.")
    if persona:
        lines.append(f"Persona hints: {persona}.")
    if not any((task_description, domain, persona)):
        lines.append(
            "No additional dataset metadata provided. Respond naturally and plausibly based on the ongoing conversation."
        )
    lines.append(
        "Match the length, tone, and specificity of real user utterances. If you are unsure, respond naturally "
        "based on the assistant's previous messages like how a real human would. Note that your response MUST "
        "not contain anything other than the USER utterance. Do not include any prefixes like 'User:' or "
        "'Human:' as well. Just the raw message content."
    )
    return "\n".join(lines)


def _assistant_mirror_system_prompt(case: BenchmarkCase) -> str:
    real_conversation = _format_conversation(_reference_conversation(case))
    return f"""You are the assistant in a MirrorBench replay. The user-proxy agent is attempting to
reproduce the USER side of the real conversation provided below. But the user-proxy
does not have access to the real conversation history. Instead, it only has access to
the conversation summary.

You need to respond as the assistant. But we are providing you with the real
conversation history as context, so you can respond consistently same as (or similar to)
the original assistant in the real conversation (you may paraphrase lightly for safety).

If user-proxy deviates from the original USER turn or the original response would violate
policy, reply helpfully using your own knowledge while remaining consistent with the
persona demonstrated so far. Always follow Azure OpenAI content policies. Paraphrase sensitive content
instead of quoting it verbatim, and refuse politely if a request is disallowed.

Here is the real conversation for context (the USER turns are from the original conversation):
{real_conversation}

Now, we will provide you with ongoing conversation with the user-proxy. Please respond
as the assistant in this conversation.""".strip()


_PI_SYSTEM_PROMPT = """[System]
Please act as an impartial judge and choose the most realistic USER among the two conversations provided below. You should choose the conversation which has USER turns written by a real user in your opinion. Based on your experience & knowledge of human conversation, determine which USER in which conversation sounds more like a real human interacting with a chatbot. Note that your judgement should be based on style, tone and behavior of USER rather than the response quality of their utterances. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the USER messages to influence your evaluation. Do not let ASSISTANT messages influence your decision. Your evaluation must only be dependent on USER messages only. Be as objective as possible."""

_PI_USER_PROMPT = """Before giving your final evaluation verdict, you should first think through and produce your reasoning step-by-step. Then, based on your reasoning, give your final verdict on which conversation has USER turns that sound more like a real human interacting with a chatbot.

If Conversation A is more real, then verdict would be "A". If Conversation B is more real, then verdict would be "B". If both are equally real or non-real, then verdict would be "Tie".

Your output must be of following JSON format: {{"reasoning": "<short reason>", "verdict": "A"|"B"|"Tie"}}.

[Conversation A]
{conversation_a}

[Conversation B]
{conversation_b}

Output ONLY valid JSON, no additional text."""

_RNR_SYSTEM_PROMPT = """[System]
Please act as an impartial judge and provide verdict on the realism of the USER turns in the conversation provided below based on the provided rubric. You should avoid checking the responses of the ASSISTANT messages since they are not relevant to the task. Your evaluation should follow the rubric provided. Note that your judgement should be based on style, tone and behavior of USER rather than the response quality of their utterances."""

_RNR_USER_PROMPT = """[Rubric for scoring USER realism]
Provide verdict on how human-like the USER is on ("NO" or "YES") using this rubric (verdict "NO" means not real, verdict "YES" means real):
1. Concise and real-user like language
2. Does not sound scripted or artificial
3. Real-user like tone and style

Return JSON: {{"reasoning": "<1-2 sentences>", "verdict": <"NO" or "YES">}}.

[Conversation]
{conversation}

Output ONLY valid JSON, no additional text."""

_GTEVAL_PROMPT = """You are an expert evaluator of conversational AI systems. Your task is to compare two conversations and assess whether the user responses in the proxy conversation exhibit similar style, tone, and realism to the user responses in the real conversation.

## Real Conversation:
{real_conversation}

## Proxy Conversation (where user turns are generated by user proxy agent):
{proxy_conversation}

## Evaluation Criteria:
1. **Style Similarity**: Do the proxy user responses match the conversational style of real user responses (formality, tone, verbosity)?
2. **Realism**: Do the proxy user responses sound natural and human-like?
3. **Contextual Appropriateness**: Are the proxy user responses appropriate given the conversation context? Note: You should not evaluate based on the content of the responses, only their style, realism, contextual appropriateness, and tone.

## Instructions:
- Focus exclusively on comparing USER responses (ignore assistant responses)
- Consider the overall conversation flow, not just individual turns
- Evaluate style, tone, and naturalness without encouraging unsafe behaviour
- If the comparison cannot be performed safely, explain the reason and provide a score of 0.0
- Otherwise, provide detailed reasoning about the comparison and assign a score from 0.0 to 1.0 where:
  - 0.0 = Proxy user responses are completely unlike real user responses
  - 0.5 = Proxy user responses are somewhat similar but have noticeable differences
  - 1.0 = Proxy user responses are highly similar in style and realism

Provide your evaluation as a JSON object with the following structure:
{{
    "reasoning": "Detailed explanation of your evaluation (2-3 sentences)",
    "score": <float between 0.0 and 1.0>
}}

Output ONLY valid JSON, no additional text."""


def _strict_json(text: str) -> Mapping[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError("MirrorBench judge response must be JSON") from exc
    if not isinstance(value, Mapping):
        raise ParseError("MirrorBench judge response must be an object")
    return value


def parse_gteval(text: str) -> Mapping[str, Any]:
    value = _strict_json(text)
    if set(value) != {"reasoning", "score"}:
        raise ParseError("GTEval requires exactly reasoning and score")
    reasoning, score = value["reasoning"], value["score"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ParseError("GTEval reasoning must be non-empty text")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
        raise ParseError("GTEval score must be a number in [0,1]")
    return {"reasoning": reasoning.strip(), "score": float(score)}


def parse_pi(text: str) -> Mapping[str, Any]:
    value = _strict_json(text)
    if set(value) != {"reasoning", "verdict"}:
        raise ParseError("PI requires exactly reasoning and verdict")
    reasoning, verdict = value["reasoning"], str(value["verdict"]).strip().upper()
    if not isinstance(reasoning, str) or not reasoning.strip() or verdict not in {"A", "B", "TIE"}:
        raise ParseError("PI requires non-empty reasoning and verdict A, B, or TIE")
    return {"reasoning": reasoning.strip(), "verdict": verdict}


def parse_rnr(text: str) -> Mapping[str, Any]:
    value = _strict_json(text)
    if set(value) != {"reasoning", "verdict"}:
        raise ParseError("RNR requires exactly reasoning and verdict")
    reasoning, verdict = value["reasoning"], str(value["verdict"]).strip().upper()
    if not isinstance(reasoning, str) or not reasoning.strip() or verdict not in {"YES", "NO"}:
        raise ParseError("RNR requires non-empty reasoning and verdict YES or NO")
    return {"reasoning": reasoning.strip(), "verdict": verdict, "score": 1.0 if verdict == "YES" else 0.0}


def _pi_order(case_id: str, seed: int, sample: int, scope: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{case_id}|{seed}|{sample}|{scope}|mirrorbench-pi-v1".encode()).digest()
    proxy_slot = "A" if digest[0] < 128 else "B"
    return proxy_slot, ("PH" if proxy_slot == "A" else "HP")


@adapter("mirrorbench")
class MirrorBenchAdapter(BenchmarkAdapter):
    benchmark_id = "mirrorbench"
    prompt_revision = "mirrorbench-official-raw-prompts-v6-single-reference"
    scorer_revision = "mirrorbench-pi-deviation-student-t-ci-and-lexical-directions-v3"

    def __init__(
        self,
        *,
        runtime_provenance: MirrorRuntimeProvenance | None = None,
        scoring_provenance: MirrorScoringProvenance | None = None,
    ) -> None:
        self.environment = DialogueEnvironment()
        self._runtime = runtime_provenance
        self._scoring = scoring_provenance

    @staticmethod
    def _metadata_mapping(case: BenchmarkCase, key: str) -> Mapping[str, Any] | None:
        direct = case.metadata.get(key)
        if isinstance(direct, Mapping):
            return direct
        replay = case.metadata.get("replay")
        value = replay.get(key) if isinstance(replay, Mapping) else None
        return value if isinstance(value, Mapping) else None

    def runtime_for_case(self, case: BenchmarkCase) -> MirrorRuntimeProvenance:
        if self._runtime is not None:
            return self._runtime
        raw = self._metadata_mapping(case, "runtime_provenance")
        if raw is None:
            raise ConfigurationError("MirrorBench requires fixed-assistant runtime provenance")
        return MirrorRuntimeProvenance(
            fixed_assistant_model=str(raw.get("fixed_assistant_model") or ""),
            fixed_assistant_revision=str(raw.get("fixed_assistant_revision") or ""),
            assistant_policy_revision=str(raw.get("assistant_policy_revision") or ""),
            max_user_turns=int(raw.get("max_user_turns", 12)),
            max_total_actions=int(raw.get("max_total_actions", 32)),
            request_timeout_seconds=float(raw.get("request_timeout_seconds", 600.0)),
            max_retries=int(raw.get("max_retries", 2)),
            user_temperature=float(raw.get("user_temperature", 0.0)),
            assistant_temperature=float(raw.get("assistant_temperature", 0.0)),
            generation_max_tokens=int(raw.get("generation_max_tokens", 2048)),
            source=str(raw.get("source") or "configured"),
        )

    def provenance_for_case(self, case: BenchmarkCase) -> MirrorScoringProvenance:
        if self._scoring is not None:
            return self._scoring
        raw = self._metadata_mapping(case, "scoring_provenance")
        if raw is None:
            raise ConfigurationError("MirrorBench requires scoring provenance")
        return MirrorScoringProvenance(
            tokenizer_policy=str(raw.get("tokenizer_policy") or ""),
            tokenizer_model=str(raw.get("tokenizer_model") or ""),
            tokenizer_revision=str(raw.get("tokenizer_revision") or ""),
            judge_model=str(raw["judge_model"]) if raw.get("judge_model") else None,
            judge_revision=str(raw["judge_revision"]) if raw.get("judge_revision") else None,
            judge_temperature=float(raw.get("judge_temperature", 0.0)),
            judge_max_tokens=int(raw.get("judge_max_tokens", 2048)),
            gteval_prompt_revision=str(raw.get("gteval_prompt_revision") or ""),
            pi_prompt_revision=str(raw.get("pi_prompt_revision") or ""),
            rnr_prompt_revision=str(raw.get("rnr_prompt_revision") or ""),
            gteval_samples=int(raw.get("gteval_samples", 1)),
            pi_samples=int(raw.get("pi_samples", 3)),
            rnr_samples=int(raw.get("rnr_samples", 2)),
            compute_controls=bool(raw.get("compute_controls", True)),
            replayed=bool(raw.get("replayed", False)),
            source=str(raw.get("source") or "configured"),
        )

    def assistant_or_partner_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        runtime = self.runtime_for_case(case)
        return {
            "role": FIXED_ASSISTANT_ROLE,
            "model": runtime.fixed_assistant_model,
            "model_revision": runtime.fixed_assistant_revision,
            "policy_revision": runtime.assistant_policy_revision,
            "temperature": runtime.assistant_temperature,
            "reference_anchored": True,
            "source": runtime.source,
        }

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        runtime = self.runtime_for_case(case)
        return {
            "revision": self.environment.environment_revision,
            "driver": "reference_role_schedule_with_fixed_assistant",
            "max_user_turns": runtime.max_user_turns,
            "max_total_actions": runtime.max_total_actions,
            "request_timeout_seconds": runtime.request_timeout_seconds,
            "max_retries": runtime.max_retries,
            "generation_max_tokens": runtime.generation_max_tokens,
        }

    def dialogue_spec_for_case(self, case: BenchmarkCase) -> DialogueSpec:
        runtime = self.runtime_for_case(case)
        reference = _reference_conversation(case)
        initial_messages: tuple[Mapping[str, Any], ...] = ()
        schedule_reference = reference
        if reference[0]["role"] == "assistant":
            initial_messages = (
                {"actor": FIXED_ASSISTANT_ROLE, "message": reference[0]["content"]},
            )
            schedule_reference = reference[1:]
        schedule = tuple(
            EVALUATED_USER_ROLE if item["role"] == "user" else FIXED_ASSISTANT_ROLE
            for item in schedule_reference
        )
        return DialogueSpec(
            user_context={
                "task_description": case.input_data.get("task_description"),
                "domain": case.input_data.get("domain"),
                "persona": case.input_data.get("persona"),
                "dataset": case.input_data.get("dataset"),
                "instruction": "Produce only the next user turn; the real conversation is hidden from you.",
            },
            assistant_context={
                "assistant_config": case.input_data.get("assistant_config"),
                # The assistant system template already renders the reference.
                # Keep its other private settings without injecting it twice.
                "instruction": "Respond consistently with the reference while adapting to the generated user turn.",
            },
            scheduled_actors=schedule,
            initial_messages=initial_messages,
            max_user_turns=runtime.max_user_turns,
            max_total_actions=runtime.max_total_actions,
            user_may_end=False,
            assistant_may_end=False,
        )

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"MirrorBenchAdapter cannot run {case.benchmark_id!r}")
        probe_case(case)
        _reference_conversation(case)
        self.runtime_for_case(case)
        provenance = self.provenance_for_case(case)
        if provenance.tokenizer_policy not in {"regex_word_v1", "tiktoken_model"}:
            raise ConfigurationError("MirrorBench tokenizer policy is unsupported")
        self.environment.reset_with_spec(case, self.dialogue_spec_for_case(case), seed=0)

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        self.validate_case(case)
        state = self.environment.reset_with_spec(case, self.dialogue_spec_for_case(case), seed=seed)
        if state.next_actor == EVALUATED_USER_ROLE:
            return self.build_user_request(case, state, model=model, seed=seed)
        return self.build_assistant_request(case, state, seed=seed)

    def parse_response(self, case: BenchmarkCase, response: ModelResponse) -> Any:
        return _parse_raw_utterance(response.text, actor=EVALUATED_USER_ROLE)

    def build_user_request(self, case: BenchmarkCase, state: DialogueState, *, model: str, seed: int) -> ModelRequest:
        runtime = self.runtime_for_case(case)
        messages = self.environment.observation(
            state, actor=EVALUATED_USER_ROLE, system_instruction=_user_proxy_system_prompt(case),
        )
        return ModelRequest(
            request_id=f"{case.case_id}:user:{state.user_turn_count}",
            messages=tuple(messages),
            model=model,
            temperature=runtime.user_temperature,
            max_tokens=runtime.generation_max_tokens,
            seed=seed + state.action_count,
            metadata={"benchmark_id": self.benchmark_id, "actor": EVALUATED_USER_ROLE, "route_role": "evaluated_user", "reference_visible": False},
        )

    def build_assistant_request(self, case: BenchmarkCase, state: DialogueState, *, seed: int) -> ModelRequest:
        runtime = self.runtime_for_case(case)
        messages = self.environment.observation(
            state, actor=FIXED_ASSISTANT_ROLE, system_instruction=_assistant_mirror_system_prompt(case),
        )
        return ModelRequest(
            request_id=f"{case.case_id}:assistant:{state.assistant_turn_count}",
            messages=tuple(messages),
            model=runtime.fixed_assistant_model,
            temperature=runtime.assistant_temperature,
            max_tokens=runtime.generation_max_tokens,
            seed=seed + state.action_count,
            metadata={"benchmark_id": self.benchmark_id, "actor": FIXED_ASSISTANT_ROLE, "route_role": "fixed_assistant", "reference_visible": True},
        )

    @staticmethod
    def _proxy_conversation(state: DialogueState) -> tuple[Mapping[str, str], ...]:
        return tuple(
            {
                "role": "user" if event.actor == EVALUATED_USER_ROLE else "assistant",
                "content": str(event.content),
            }
            for event in state.public_transcript
            if event.kind in {"message", "seed_message"}
        )

    def _judge_prompt(
        self,
        metric_name: str,
        *,
        real_text: str,
        proxy_text: str,
        proxy_slot: str | None = None,
    ) -> tuple[ChatMessage, ...]:
        if metric_name == "gteval":
            prompt = _GTEVAL_PROMPT.format(
                real_conversation=real_text,
                proxy_conversation=proxy_text,
            )
            return (ChatMessage("user", prompt, metadata={"visibility": "evaluator"}),)
        if metric_name == "pi":
            if proxy_slot == "A":
                conversation_a, conversation_b = proxy_text, real_text
            else:
                conversation_a, conversation_b = real_text, proxy_text
            return (
                ChatMessage(
                    "system",
                    _PI_SYSTEM_PROMPT,
                    metadata={"visibility": "evaluator"},
                ),
                ChatMessage(
                    "user",
                    _PI_USER_PROMPT.format(
                        conversation_a=conversation_a,
                        conversation_b=conversation_b,
                    ),
                    metadata={"visibility": "evaluator"},
                ),
            )
        return (
            ChatMessage(
                "system",
                _RNR_SYSTEM_PROMPT,
                metadata={"visibility": "evaluator"},
            ),
            ChatMessage(
                "user",
                _RNR_USER_PROMPT.format(conversation=proxy_text),
                metadata={"visibility": "evaluator"},
            ),
        )

    def build_judge_request(
        self,
        case: BenchmarkCase,
        state: DialogueState,
        provenance: MirrorScoringProvenance,
        *,
        metric_name: str,
        scope: str,
        sample: int,
        seed: int,
    ) -> tuple[ModelRequest, Mapping[str, Any]]:
        if not provenance.judge_model:
            raise ConfigurationError("MirrorBench judge suite is not configured")
        reference = _reference_conversation(case)
        proxy = self._proxy_conversation(state)
        real_text = _format_conversation(reference)
        proxy_text = _format_conversation(proxy)
        if scope == "hh":
            proxy_text = real_text
        elif scope == "pp":
            real_text = proxy_text
        proxy_slot = None
        order = None
        if metric_name == "pi":
            proxy_slot, order = _pi_order(case.case_id, seed, sample, scope)
        revision = {
            "gteval": provenance.gteval_prompt_revision,
            "pi": provenance.pi_prompt_revision,
            "rnr": provenance.rnr_prompt_revision,
        }[metric_name]
        if metric_name == "gteval":
            schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "minLength": 1},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["reasoning", "score"],
                "additionalProperties": False,
            }
            contract = "Return only JSON with exactly non-empty reasoning and numeric score in [0,1]."
        elif metric_name == "pi":
            schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "minLength": 1},
                    "verdict": {"type": "string", "enum": ["A", "B", "Tie"]},
                },
                "required": ["reasoning", "verdict"],
                "additionalProperties": False,
            }
            contract = "Return only JSON with exactly non-empty reasoning and verdict A, B, or Tie."
        else:
            schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "minLength": 1},
                    "verdict": {"type": "string", "enum": ["NO", "YES"]},
                },
                "required": ["reasoning", "verdict"],
                "additionalProperties": False,
            }
            contract = "Return only JSON with exactly non-empty reasoning and verdict NO or YES."
        request = ModelRequest(
            request_id=f"{case.case_id}:judge:{metric_name}:{scope}:{sample}",
            messages=self._judge_prompt(metric_name, real_text=real_text, proxy_text=proxy_text, proxy_slot=proxy_slot),
            model=provenance.judge_model,
            temperature=provenance.judge_temperature,
            max_tokens=provenance.judge_max_tokens,
            seed=seed + sample,
            response_format=json_schema_response_format(f"mirrorbench_{metric_name}", schema),
            metadata={
                "benchmark_id": self.benchmark_id,
                "route_role": "judge",
                "judge_revision": provenance.judge_revision,
                "prompt_revision": revision,
                "metric": metric_name,
                "scope": scope,
                "sample": sample,
                "replayed": provenance.replayed,
                "order": order,
                "proxy_slot": proxy_slot,
                "prompt_source": "SAP MirrorBench mirrorbench/metrics/judge/prompts.py",
                "output_contract": contract,
            },
        )
        return request, {"order": order, "proxy_slot": proxy_slot}

    def _generate_with_retries(
        self,
        backend: ModelBackend,
        request: ModelRequest,
        runtime: MirrorRuntimeProvenance,
        audits: list[Mapping[str, Any]],
    ) -> ModelResponse:
        last_error: BackendError | None = None
        for attempt in range(runtime.max_retries + 1):
            try:
                response = backend.generate(request)
                if response.latency_ms is not None and response.latency_ms > runtime.request_timeout_seconds * 1000:
                    raise BackendTimeoutError(
                        f"response latency {response.latency_ms}ms exceeded {runtime.request_timeout_seconds}s"
                    )
                audits.append({"request_id": request.request_id, "attempt": attempt, "status": "completed"})
                return response
            except BackendError as exc:
                last_error = exc
                audits.append(
                    {
                        "request_id": request.request_id,
                        "attempt": attempt,
                        "status": "retry" if attempt < runtime.max_retries else "failed",
                        "kind": type(exc).__name__,
                    }
                )
        assert last_error is not None
        raise last_error

    @staticmethod
    def _judge_scopes(metric_name: str, provenance: MirrorScoringProvenance) -> tuple[str, ...]:
        if not provenance.compute_controls:
            return ("main",)
        if metric_name in {"gteval", "pi"}:
            return ("main", "hh", "pp")
        return ("main", "hh")

    @staticmethod
    def _judge_sample_count(metric_name: str, provenance: MirrorScoringProvenance) -> int:
        return {
            "gteval": provenance.gteval_samples,
            "pi": provenance.pi_samples,
            "rnr": provenance.rnr_samples,
        }[metric_name]

    def replay_responses(self, case: BenchmarkCase, *, seed: int) -> Mapping[str, Any]:
        self.validate_case(case)
        replay = case.metadata.get("replay")
        if not isinstance(replay, Mapping):
            raise ConfigurationError("MirrorBench fixture lacks replay metadata")
        steps = replay.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise ConfigurationError("MirrorBench replay.steps must be an array")
        state = self.environment.reset_with_spec(case, self.dialogue_spec_for_case(case), seed=seed)
        responses: dict[str, Any] = {}
        for index, step in enumerate(steps):
            if state.terminal or not isinstance(step, Mapping):
                raise ConfigurationError(f"invalid MirrorBench replay step #{index}")
            actor = str(step.get("actor") or "")
            if actor != state.next_actor:
                raise ConfigurationError(f"MirrorBench replay actor {actor!r} does not match {state.next_actor!r}")
            output = step.get("output")
            if not isinstance(output, Mapping):
                raise ConfigurationError(f"MirrorBench replay step #{index} requires output object")
            if set(output) != {"action", "message"} or output.get("action") != "message":
                raise ConfigurationError(
                    f"MirrorBench replay step #{index} must contain one legacy message fixture"
                )
            text = output.get("message")
            if not isinstance(text, str) or not text.strip():
                raise ConfigurationError(f"MirrorBench replay step #{index} has no raw utterance")
            request = (
                self.build_user_request(case, state, model="offline-replay", seed=seed)
                if actor == EVALUATED_USER_ROLE
                else self.build_assistant_request(case, state, seed=seed)
            )
            responses[request.request_id or ""] = {
                "text": text,
                "finish_reason": "replayed",
                "usage": step.get("usage") or {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
            self.environment.apply(
                state,
                actor=actor,
                action=_parse_raw_utterance(text, actor=actor),
            )
        if not state.terminal or not state.protocol_complete:
            raise ConfigurationError("MirrorBench replay must exhaust the reference role schedule")

        provenance = self.provenance_for_case(case)
        if not provenance.judge_model:
            return responses
        judges = replay.get("judges")
        if not isinstance(judges, Mapping):
            raise ConfigurationError("configured MirrorBench judge requires replay.judges")
        for metric_name in JUDGE_METRICS:
            metric_payload = judges.get(metric_name)
            if not isinstance(metric_payload, Mapping):
                raise ConfigurationError(f"MirrorBench replay lacks {metric_name} judge payload")
            for scope in self._judge_scopes(metric_name, provenance):
                samples = metric_payload.get(scope)
                count = self._judge_sample_count(metric_name, provenance)
                if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or len(samples) != count:
                    raise ConfigurationError(f"MirrorBench {metric_name}.{scope} requires {count} replay samples")
                for sample, output in enumerate(samples):
                    request, _ = self.build_judge_request(
                        case, state, provenance, metric_name=metric_name, scope=scope, sample=sample, seed=seed
                    )
                    responses[request.request_id or ""] = {
                        "text": canonical_json(output),
                        "finish_reason": "replayed",
                    }
        return responses

    def _failure_result(
        self,
        case: BenchmarkCase,
        state: DialogueState,
        *,
        run_id: str,
        repetition: int,
        responses: Sequence[ModelResponse],
        audits: Sequence[Mapping[str, Any]],
        actor: str,
        stage: str,
        kind: str,
        message: str,
        retryable: bool,
    ) -> CaseResult:
        if not state.terminal:
            self.environment.force_terminate(state, actor=actor, reason=kind, stage=stage, kind=kind)
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.FAILED,
            prediction={"public_transcript": list(self._proxy_conversation(state)), "terminal_reason": state.terminal_reason},
            trace=tuple(state.trace),
            model_response=responses[-1] if responses else None,
            error=ErrorState(stage, kind, message, retryable=retryable, details={"request_audits": list(audits)}),
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "episode_complete": False,
                "terminal_reason": state.terminal_reason,
                "request_audits": list(audits),
                "partial_user_turn_count": state.user_turn_count,
            },
        )

    def _target_capability_result(
        self,
        case: BenchmarkCase,
        state: DialogueState,
        *,
        run_id: str,
        repetition: int,
        responses: Sequence[ModelResponse],
        audits: Sequence[Mapping[str, Any]],
        provenance: MirrorScoringProvenance,
        stage: str,
        kind: str,
        message: str,
    ) -> CaseResult:
        if not state.terminal:
            self.environment.force_terminate(
                state,
                actor=EVALUATED_USER_ROLE,
                reason=kind,
                stage=stage,
                kind=kind,
            )
        failure = {
            "stage": stage,
            "kind": kind,
            "message": message,
            "retryable": False,
            "scoring_policy": "target_capability_failure_scores_zero",
        }
        human_user_text = " ".join(
            item["content"] for item in _reference_conversation(case) if item["role"] == "user"
        )
        human_lexical: Mapping[str, float] = {}
        lexical_error: Mapping[str, Any] | None = None
        try:
            human_tokens = mirror_tokenize(
                human_user_text,
                policy=provenance.tokenizer_policy,
                model=provenance.tokenizer_model,
            )
            if not human_tokens:
                raise ValidationError("MirrorBench reference requires a non-empty user token stream")
            human_lexical = _lexical_values(human_tokens)
        except (ConfigurationError, OptionalDependencyError, ValidationError) as exc:
            lexical_error = {"kind": type(exc).__name__, "message": str(exc)}
        lexical_meta = {
            "tokenizer_policy": provenance.tokenizer_policy,
            "tokenizer_model": provenance.tokenizer_model,
            "tokenizer_revision": provenance.tokenizer_revision,
            "target_output_failure": failure,
            "error": lexical_error,
        }
        metrics: list[MetricValue] = []
        for name in LEXICAL_METRICS:
            metrics.append(
                _metric(
                    f"mirrorbench.lexical.{name}.proxy_raw",
                    0.0,
                    direction="descriptive",
                    metadata=lexical_meta,
                )
            )
            metrics.append(
                _metric(
                    f"mirrorbench.lexical.{name}.human_raw",
                    human_lexical.get(name),
                    direction="descriptive",
                    metadata=lexical_meta,
                )
            )
        judge_meta = {
            "judge_model": provenance.judge_model,
            "judge_revision": provenance.judge_revision,
            "judge_temperature": provenance.judge_temperature,
            "judge_max_tokens": provenance.judge_max_tokens,
            "target_output_failure": failure,
        }
        for name in JUDGE_METRICS:
            metrics.append(_metric(f"mirrorbench.judge.{name}", 0.0, metadata=judge_meta))
            for scope in ("hh", "pp"):
                metrics.append(
                    _metric(
                        f"mirrorbench.judge.{name}.{scope}_control",
                        None,
                        direction="descriptive",
                        metadata={**judge_meta, "availability": "not_run_target_output_failure"},
                    )
                )
        proxy = self._proxy_conversation(state)
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction={"public_transcript": list(proxy), "terminal_reason": state.terminal_reason, "parsed": False},
            metrics=tuple(metrics),
            trace=tuple(state.trace),
            model_response=responses[-1] if responses else None,
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "episode_complete": True,
                "natural_termination": False,
                "terminal_reason": state.terminal_reason,
                "evaluated_model_role": EVALUATED_USER_ROLE,
                "fixed_assistant_identity": self.assistant_or_partner_identity_for_case(case),
                "environment_identity": self.environment_identity_for_case(case),
                "scoring_provenance": provenance.to_dict(),
                "request_audits": list(audits),
                "target_output_failure": failure,
                "reference_visible_to_evaluated_user": False,
                "reference_visible_to_fixed_assistant": True,
                "mirrorbench": {
                    "lexical": {
                        name: {"proxy": 0.0, "human": human_lexical.get(name)}
                        for name in LEXICAL_METRICS
                    },
                    "lexical_error": lexical_error,
                    "judge_records": {
                        name: {"main": {"status": "target_output_failure", "score": 0.0}}
                        for name in JUDGE_METRICS
                    },
                },
            },
        )

    def _run_judge_metric(
        self,
        case: BenchmarkCase,
        state: DialogueState,
        backend: ModelBackend,
        runtime: MirrorRuntimeProvenance,
        provenance: MirrorScoringProvenance,
        audits: list[Mapping[str, Any]],
        responses: list[ModelResponse],
        *,
        metric_name: str,
        seed: int,
        previous_records: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        output: dict[str, Any] = {}
        for scope in self._judge_scopes(metric_name, provenance):
            saved = (previous_records or {}).get(scope, {})
            if saved.get("score") is not None:
                output[scope] = saved
                continue
            samples: list[Mapping[str, Any]] = list(saved.get("samples") or [])
            error: Mapping[str, Any] | None = None
            for sample in range(len(samples), self._judge_sample_count(metric_name, provenance)):
                request, order_meta = self.build_judge_request(
                    case,
                    state,
                    provenance,
                    metric_name=metric_name,
                    scope=scope,
                    sample=sample,
                    seed=seed,
                )
                try:
                    current_request = request
                    parsed: dict[str, Any] | None = None
                    for contract_attempt in range(DEFAULT_CONTRACT_RETRIES + 1):
                        response = self._generate_with_retries(backend, current_request, runtime, audits)
                        responses.append(response)
                        try:
                            if metric_name == "gteval":
                                parsed = dict(parse_gteval(response.text))
                            elif metric_name == "pi":
                                parsed = dict(parse_pi(response.text))
                                verdict = parsed["verdict"]
                                proxy_slot = order_meta["proxy_slot"]
                                parsed["score"] = 0.5 if verdict == "TIE" else float(verdict == proxy_slot)
                                parsed.update(order_meta)
                            else:
                                parsed = dict(parse_rnr(response.text))
                            break
                        except ParseError as exc:
                            audits.append(
                                {
                                    "request_id": request.request_id,
                                    "contract_attempt": contract_attempt,
                                    "status": (
                                        "contract_retry"
                                        if contract_attempt < DEFAULT_CONTRACT_RETRIES
                                        else "contract_failed"
                                    ),
                                    "kind": type(exc).__name__,
                                    "message": str(exc),
                                }
                            )
                            if contract_attempt >= DEFAULT_CONTRACT_RETRIES:
                                raise
                            current_request = corrective_retry_request(
                                request,
                                attempt=contract_attempt + 1,
                                reason=str(exc),
                                contract=str(request.metadata["output_contract"]),
                                previous_output=response.text,
                                request_suffix="judge_contract_retry",
                            )
                    assert parsed is not None
                    samples.append(parsed)
                except (BackendError, ParseError) as exc:
                    error = {"kind": type(exc).__name__, "message": str(exc), "sample": sample}
                    break
            if error is None and len(samples) == self._judge_sample_count(metric_name, provenance):
                output[scope] = {
                    "status": "available",
                    "score": statistics.mean(float(item["score"]) for item in samples),
                    "samples": samples,
                }
            else:
                output[scope] = {
                    "status": "unavailable",
                    "score": None,
                    "samples": samples,
                    "error": error,
                    "partial_sample_count": len(samples),
                }
        return output

    def execute_case(
        self,
        case: BenchmarkCase,
        *,
        backend: ModelBackend,
        run_id: str,
        seed: int,
        model: str,
        repetition: int = 0,
        previous_result: CaseResult | None = None,
    ) -> CaseResult:
        self.validate_case(case)
        runtime = self.runtime_for_case(case)
        provenance = self.provenance_for_case(case)
        state = self.environment.reset_with_spec(case, self.dialogue_spec_for_case(case), seed=seed)
        responses: list[ModelResponse] = []
        audits: list[Mapping[str, Any]] = []
        budget_termination: Mapping[str, Any] | None = None
        if previous_result is not None:
            from ..judge_resume import restore_judge_state
            restore_judge_state(state, previous_result)
        while previous_result is None and not state.terminal:
            actor = state.next_actor
            assert actor is not None
            if actor == EVALUATED_USER_ROLE and state.user_turn_count >= runtime.max_user_turns:
                return self._target_capability_result(
                    case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                    provenance=provenance, stage="user_turn_limit", kind="turn_limit",
                    message="MirrorBench user-turn limit reached before schedule completion",
                )
            request = (
                self.build_user_request(case, state, model=model, seed=seed)
                if actor == EVALUATED_USER_ROLE
                else self.build_assistant_request(case, state, seed=seed)
            )
            prefix = "user" if actor == EVALUATED_USER_ROLE else "assistant"
            try:
                response = self._generate_with_retries(backend, request, runtime, audits)
                responses.append(response)
            except EpisodeTokenBudgetExhausted as exc:
                budget_termination = exc.details()
                self.environment.terminate_capability_limit(
                    state,
                    actor=actor,
                    reason="token_budget_exhausted",
                    details=budget_termination,
                )
                break
            except BackendError as exc:
                return self._failure_result(
                    case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                    actor=actor, stage=f"{prefix}_backend", kind=type(exc).__name__, message=str(exc), retryable=True,
                )
            try:
                action = _parse_raw_utterance(response.text, actor=actor)
                self.environment.apply(state, actor=actor, action=action)
            except (ParseError, ValidationError) as exc:
                kind = "empty_turn" if "non-empty" in str(exc) else "invalid_action"
                if actor == EVALUATED_USER_ROLE:
                    return self._target_capability_result(
                        case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                        provenance=provenance, stage="user_parse", kind=kind, message=str(exc),
                    )
                return self._failure_result(
                    case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                    actor=actor, stage=f"{prefix}_parse", kind=kind, message=str(exc), retryable=False,
                )
            if state.terminal and not state.protocol_complete:
                if actor == EVALUATED_USER_ROLE:
                    return self._target_capability_result(
                        case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                        provenance=provenance, stage="user_refusal", kind="refusal",
                        message=f"{actor} refused before schedule completion",
                    )
                return self._failure_result(
                    case, state, run_id=run_id, repetition=repetition, responses=responses, audits=audits,
                    actor=actor, stage=f"{prefix}_refusal", kind="refusal",
                    message=f"{actor} refused before schedule completion", retryable=False,
                )

        proxy = self._proxy_conversation(state)
        reference = _reference_conversation(case)
        proxy_user_text = " ".join(item["content"] for item in proxy if item["role"] == "user")
        human_user_text = " ".join(item["content"] for item in reference if item["role"] == "user")
        lexical_error: Mapping[str, Any] | None = None
        proxy_lexical: Mapping[str, float] = {}
        human_lexical: Mapping[str, float] = {}
        if previous_result is not None:
            saved = previous_result.metadata.get("mirrorbench", {})
            proxy_lexical = {k: v.get("proxy") for k, v in saved.get("lexical", {}).items()}
            human_lexical = {k: v.get("human") for k, v in saved.get("lexical", {}).items()}
            lexical_error = saved.get("lexical_error")
        else:
            try:
                proxy_tokens = mirror_tokenize(proxy_user_text, policy=provenance.tokenizer_policy, model=provenance.tokenizer_model)
                human_tokens = mirror_tokenize(human_user_text, policy=provenance.tokenizer_policy, model=provenance.tokenizer_model)
                if not proxy_tokens or not human_tokens:
                    raise ValidationError("MirrorBench lexical metrics require non-empty user token streams")
                proxy_lexical = _lexical_values(proxy_tokens)
                human_lexical = _lexical_values(human_tokens)
            except (ConfigurationError, OptionalDependencyError, ValidationError) as exc:
                lexical_error = {"kind": type(exc).__name__, "message": str(exc)}

        from ..judge_resume import mirror_saved_records
        judge_records: dict[str, Any] = {}
        if provenance.judge_model:
            for metric_name in JUDGE_METRICS:
                judge_records[metric_name] = self._run_judge_metric(
                    case,
                    state,
                    backend,
                    runtime,
                    provenance,
                    audits,
                    responses,
                    metric_name=metric_name,
                    seed=seed,
                    previous_records=mirror_saved_records(previous_result, metric_name),
                )
        else:
            for metric_name in JUDGE_METRICS:
                judge_records[metric_name] = {
                    scope: {"status": "unavailable", "score": None, "reason": "judge_not_configured"}
                    for scope in self._judge_scopes(metric_name, provenance)
                }

        lexical_meta = {
            "tokenizer_policy": provenance.tokenizer_policy,
            "tokenizer_model": provenance.tokenizer_model,
            "tokenizer_revision": provenance.tokenizer_revision,
            "protocol_status": (
                "canonical_tokenization" if provenance.tokenizer_policy == "tiktoken_model" else "explicit_offline_protocol_variant"
            ),
            "error": lexical_error,
        }
        judge_meta = {
            "judge_model": provenance.judge_model,
            "judge_revision": provenance.judge_revision,
            "judge_temperature": provenance.judge_temperature,
            "judge_max_tokens": provenance.judge_max_tokens,
            "replayed": provenance.replayed,
        }
        metrics: list[MetricValue] = []
        for name in LEXICAL_METRICS:
            metrics.append(_metric(f"mirrorbench.lexical.{name}.proxy_raw", proxy_lexical.get(name), direction="descriptive", metadata=lexical_meta))
            metrics.append(_metric(f"mirrorbench.lexical.{name}.human_raw", human_lexical.get(name), direction="descriptive", metadata=lexical_meta))
        for name in JUDGE_METRICS:
            records = judge_records[name]
            metrics.append(_metric(f"mirrorbench.judge.{name}", records["main"]["score"], metadata={**judge_meta, "prompt_revision": getattr(provenance, f"{name}_prompt_revision"), "samples": records["main"].get("samples"), "error": records["main"].get("error")}))
            for scope in ("hh", "pp"):
                if scope in records:
                    metrics.append(_metric(f"mirrorbench.judge.{name}.{scope}_control", records[scope]["score"], direction="descriptive", metadata={**judge_meta, "scope": scope, "samples": records[scope].get("samples"), "error": records[scope].get("error")}))

        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction={"public_transcript": list(proxy), "terminal_reason": state.terminal_reason},
            metrics=tuple(metrics),
            trace=tuple(state.trace),
            model_response=responses[-1] if responses else None,
            latency_ms=sum(response.latency_ms or 0 for response in responses),
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "episode_complete": True,
                "terminal_reason": state.terminal_reason,
                "natural_termination": state.terminal_reason != "token_budget_exhausted",
                "budget_termination": budget_termination,
                "evaluated_model_role": EVALUATED_USER_ROLE,
                "fixed_assistant_identity": self.assistant_or_partner_identity_for_case(case),
                "environment_identity": self.environment_identity_for_case(case),
                "scoring_provenance": provenance.to_dict(),
                "request_audits": audits,
                "reference_visible_to_evaluated_user": False,
                "reference_visible_to_fixed_assistant": True,
                "mirrorbench": {
                    "lexical": {
                        name: {"proxy": proxy_lexical.get(name), "human": human_lexical.get(name)}
                        for name in LEXICAL_METRICS
                    },
                    "lexical_error": lexical_error,
                    "judge_records": judge_records,
                },
            },
        )

    @staticmethod
    def _mean_stdev_ci(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
        if not values:
            return None, None, None
        mean_value = statistics.mean(values)
        if len(values) == 1:
            # One observation cannot estimate sample variance or a mean CI.
            return mean_value, None, None
        stdev_value = statistics.stdev(values)
        degrees_of_freedom = len(values) - 1
        critical = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
            16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
            21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
            26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        }.get(degrees_of_freedom)
        if critical is None:
            # Fourth-order Student-t quantile expansion, used only for df > 30.
            # At p=.975 its absolute error is <3e-8 for df=31..1000;
            # unlike a constant 1.96, it retains the finite-sample correction.
            # Keep the existing small-df table and avoid a SciPy dependency.
            z = statistics.NormalDist().inv_cdf(0.975)
            df = degrees_of_freedom
            critical = (
                z + (z**3 + z) / (4 * df)
                + (5*z**5 + 16*z**3 + 3*z) / (96 * df**2)
                + (3*z**7 + 19*z**5 + 17*z**3 - 15*z) / (384 * df**3)
                + (79*z**9 + 776*z**7 + 1482*z**5 - 1920*z**3 - 945*z) / (92160 * df**4)
            )
        return mean_value, stdev_value, critical * stdev_value / sqrt(len(values))

    @staticmethod
    def _ci_metadata(sample_count: int) -> Mapping[str, Any]:
        return {
            "confidence_interval": "95_percent_student_t_half_width",
            "confidence_interval_method": "student_t_table_df_le30_cf4_df_gt30_v1",
            "confidence_interval_sample_count": sample_count,
            "confidence_interval_degrees_of_freedom": max(0, sample_count - 1),
            "confidence_interval_availability": "available" if sample_count >= 2 else "unavailable_insufficient_samples",
        }

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        metrics = dict(aggregate_named_metrics(results, namespace="mirrorbench"))
        completed = [result for result in results if result.status == ResultStatus.COMPLETED]
        for name in LEXICAL_METRICS:
            rows: list[tuple[float, float]] = []
            for result in completed:
                payload = result.metadata.get("mirrorbench")
                lexical = payload.get("lexical") if isinstance(payload, Mapping) else None
                row = lexical.get(name) if isinstance(lexical, Mapping) else None
                if isinstance(row, Mapping) and row.get("proxy") is not None and row.get("human") is not None:
                    rows.append((float(row["proxy"]), float(row["human"])))
            human_values = [human for _, human in rows]
            baseline_mean = statistics.mean(human_values) if human_values else None
            baseline_std = statistics.stdev(human_values) if len(human_values) > 1 else 0.0 if human_values else None
            z_scores = [
                0.0 if not baseline_std else (proxy - float(baseline_mean)) / baseline_std
                for proxy, _ in rows
            ]
            mean_z, stdev_z, ci_z = self._mean_stdev_ci(z_scores)
            metadata = {
                "baseline_mean": baseline_mean,
                "baseline_sample_stdev": baseline_std,
                "valid_episode_count": len(rows),
                "excluded_episode_count": len(completed) - len(rows),
                "aggregation": "per_episode_human_anchored_z_then_mean",
                **self._ci_metadata(len(z_scores)),
            }
            metrics[f"mirrorbench.lexical.{name}.z_score_mean"] = MetricValue(
                name=f"mirrorbench.lexical.{name}.z_score_mean",
                value=mean_z,
                direction="closer_to_zero",
                uncertainty={"sample_stdev": stdev_z, "ci95_half_width": ci_z},
                metadata={"availability": "available" if mean_z is not None else "unavailable", **metadata},
            )
            for scope in ("proxy_raw", "human_raw"):
                raw_name = f"mirrorbench.lexical.{name}.{scope}"
                raw = metrics.get(raw_name)
                if raw is not None:
                    metrics[raw_name] = MetricValue(
                        name=raw.name,
                        value=raw.value,
                        direction="descriptive",
                        unit=raw.unit,
                        numerator=raw.numerator,
                        denominator=raw.denominator,
                        uncertainty=raw.uncertainty,
                        metadata=raw.metadata,
                    )
        pi_values = [
            float(metric.value)
            for result in completed
            for metric in result.metrics
            if metric.name == "mirrorbench.judge.pi"
            and isinstance(metric.value, (int, float, bool))
        ]
        pi_mean, pi_stdev, pi_ci = self._mean_stdev_ci(pi_values)
        metrics["mirrorbench.judge.pi"] = MetricValue(
            "mirrorbench.judge.pi",
            pi_mean,
            direction="higher_is_better",
            numerator=sum(pi_values) if pi_values else None,
            denominator=len(pi_values),
            uncertainty={"sample_stdev": pi_stdev, "ci95_half_width": pi_ci},
            metadata={
                "aggregation": "mean_episode_proxy_win_rate",
                "paper_reporting_companion": "mirrorbench.judge.pi_deviation",
                "parity_neutral": 0.5,
                "availability": "available" if pi_mean is not None else "unavailable",
                **self._ci_metadata(len(pi_values)),
            },
        )
        pi_deviations = [value - 0.5 for value in pi_values]
        metrics["mirrorbench.judge.pi_deviation"] = MetricValue(
            "mirrorbench.judge.pi_deviation",
            pi_mean - 0.5 if pi_mean is not None else None,
            direction="higher_is_better",
            numerator=sum(pi_deviations) if pi_deviations else None,
            denominator=len(pi_deviations),
            uncertainty={"sample_stdev": pi_stdev, "ci95_half_width": pi_ci},
            metadata={
                "official_paper_symbol": "delta_w_PI",
                "formula": "mean_episode_proxy_win_rate - 0.5",
                "neutral_reference": 0.0,
                "raw_metric": "mirrorbench.judge.pi",
                "availability": "available" if pi_mean is not None else "unavailable",
                **self._ci_metadata(len(pi_values)),
            },
        )
        for name in JUDGE_METRICS:
            available = sum(
                any(metric.name == f"mirrorbench.judge.{name}" and metric.value is not None for metric in result.metrics)
                for result in completed
            )
            metrics[f"mirrorbench.judge.{name}.availability_rate"] = _metric(
                f"mirrorbench.judge.{name}.availability_rate",
                available / len(completed) if completed else None,
                metadata={"available_episode_count": available, "completed_episode_count": len(completed)},
            )
        return metrics


__all__ = [
    "JUDGE_METRICS",
    "LEXICAL_METRICS",
    "MirrorBenchAdapter",
    "MirrorRuntimeProvenance",
    "MirrorScoringProvenance",
    "hdd",
    "mattr",
    "mirror_tokenize",
    "parse_gteval",
    "parse_pi",
    "parse_rnr",
    "yules_k",
]
