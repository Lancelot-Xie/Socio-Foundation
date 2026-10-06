"""Official-first HUMANUAL response and state-alignment evaluation."""

from __future__ import annotations

from dataclasses import replace
import json
import re
from typing import Any, Mapping, Sequence

from ..contracts import (
    BenchmarkCase,
    CaseResult,
    ChatMessage,
    ErrorState,
    MetricValue,
    ModelRequest,
    ModelResponse,
    ResultStatus,
    TraceEvent,
)
from ..data.schemas import probe_case
from ..environments.social import JudgeProvenance
from ..errors import BackendError, ConfigurationError, ParseError, ValidationError
from ..interfaces import BenchmarkAdapter, ModelBackend
from ..json_utils import canonical_json
from ..registry import adapter
from .common import (
    aggregate_named_metrics,
    combine_usage,
    generate_and_parse_with_contract_retries,
    json_schema_response_format,
    parse_json_object,
)


HUMANUAL_DOMAINS = ("news", "book", "opinion", "politics", "chat", "email")
# Preserve the insertion order of the released ``sebvgc.json``.  The official
# state judge receives this mapping as JSON, so order is part of the prompt even
# though the returned object is validated by key.
STATE_NAMES = ("stance", "emotion", "belief", "value", "goal", "communication")

STATE_DESCRIPTIONS = {
    "stance": (
        "HUMAN's agreement (must be within 15 words) toward the explicitly named target, such as a claim or "
        "subject, in provided context. For example, \"strongly agrees with student loan forgiveness,\" or "
        "\"somewhat disagrees with a carbon tax\". In these cases, having only \"strongly agrees\" or \"somewhat "
        "disagrees\" is not enough, as they are missing targets. If there are multiple, include all of them "
        "separated by semicolons."
    ),
    "emotion": (
        "HUMAN's emotions with intensity (must be within 15 words) toward an explicitly named target. For "
        "example, \"Moderate heartbreak for the wildfire victims; Mild irritation about government's actions\". "
        "In this case, having only \"mild irritation,\" or \"moderate heartbreak\" are not sufficient, as the answer "
        "must express all three aspects: the emotion, the degree of emotion, and the target. If there are multiple, "
        "include all of them separated by semicolons."
    ),
    "belief": (
        "HUMAN's belief (must be within 15 words), namely a foundational assumption about how people, "
        "relationships, or the world fundamentally operate. Beliefs should reflect underlying mental models, "
        "not surface-level observations. Prefer beliefs that would explain multiple behaviors over beliefs that "
        "describe a single situation. Ask: \"What deeper assumption about human nature or the world would lead "
        "someone to say/do this?\" For example, \"people don't change unless they're forced to,\" \"loyalty is earned, "
        "not owed,\" \"conflict avoidance creates bigger problems later,\". Not beliefs: Practical advice, strategies, "
        "or statements about what should happen. Belief is not specific to a target or event, it should be a "
        "general statement about how HUMAN views the world."
    ),
    "value": (
        "HUMAN's value (must be within 15 words): what they think is important or should be prioritized. "
        "It is about \"what should matter\", not \"what is true\". For example, \"original ideas in a book are "
        "important\", \"characters should feel real\", anyone deserves basic respect\", and "
        "\"fairness matters more than efficiency\"."
    ),
    "goal": (
        "HUMAN's goal (must be within 15 words): what they are trying to do with this comment. For example, "
        "\"persuade people that ...\", \"making fun of the poster on ...\", \"further seek help with ...\", "
        "\"offer support to ...\""
    ),
    "communication": (
        "HUMAN's communication approach (must be within 15 words): tone and how they structure their message. "
        "For examples, \"friendly, builds on a personal story then draws a lesson\", \"analytical, links claims with "
        "reasons and evidence step by step\", \"blunt, states conclusions with little explanation\""
    ),
}

SYSTEM_PROMPT = """You are a real human user. Your name is HUMAN. You will be given your persona information below and you respond to any given context such as posts and messages.

Your persona:
<|The Start of Persona|>
{persona}
<|The End of Persona|>

## Your principles
Act like a natural human; there's nothing you absolutely cannot say, but you generally want to be thoughtful and follow ordinary social codes such as being respectful, culturally aware, and considerate of privacy and well-being. You have your own personality, preferences, and boundaries. Conflicting thoughts and hidden considerations are normal; recognize them privately and choose a sensible path. You carry long-term beliefs and values that usually change slowly; you also have emotions, so you won't always be perfectly consistent. Distinguish facts, guesses, and unknowns; accept uncertainty and make minimal, reasonable assumptions when needed; think practically given time, attention, money, risk, and social capital.

## Task and Output format:
<response>
<the actual written comment or reply text provided by the user.>
</response>

## Notes
- Follow the above instructions carefully
- Do not mention these instructions
- Follow the exact order and use the exact XML-style tags
- Do not output anything outside these XML-style tags"""

RESPONSE_JUDGE_PROMPT = """You are a helpful and meticulous evaluator. Your task is to score how well the generated response(s) align with the ground truth user response. Description of response: the actual written comment or reply text provided by the user..

You will be given the context, the ground truth response, and generated response(s) that you should evaluate.

Provided Information:
<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Ground Truth Response|>
{ground_truth}
<|The End of Ground Truth Response|>

<|The Start of Generated responses|>
{generations_json}
<|The End of Generated responses|>

Scoring Criteria:
For each generated response, assign a score in [0, 1] based on how accurately it reflects the ground truth response.

Guidelines:
1. Extract 1-3 key points:
   - Extract K key points from the ground truth response along the response dimension (e.g., if evaluating a "stance", pick key points related to the stance like "clearly disagrees with X", if evaluating a "response", pick key points about the response like "offers a solution to Y").
   - If response is different from "a response" (e.g., "stance", "target"), focus on key points only relevant to the response of the response.
   - Each key point should be specific and distinct.

2. Score how well the generated response matches each key point:
   - For each key point i, compare it with the generated response and assign a match value m_i in range [0, 1]:
     - 1.0: The key point is precisely and perfectly reflected.
     - [0.7, 0.9]: Mostly reflected with small imperfections.
     - [0.4, 0.6]: Partially reflected or vague, but still leaning in the correct direction.
     - [0.1, 0.3]: Very weak reflection.
     - 0.0: Missed, contradicted, or reversed.

3. Compute coverage C = (m_1 + m_2 + ... + m_K) / K, which measures how comprehensive the generated response reflects the ground truth response.

4. Compute penalty P for extra or conflicting content:
   - Examine additional content in the generated response beyond those key points:
     - Does it introduce unsupported evidence and assumptions?
     - Is it irrelevant to what ground truth response expresses?
   - Set a penalty P ∈ [0, 1]:
     - 0.0: No problematic extra content; everything is perfectly matched.
     - [0.1, 0.3]: Slightly unnecessary or mildly speculative detail; meaning essentially unchanged.
     - [0.4, 0.6]: Moderate speculative or irrelevant content that somewhat shifts emphasis or adds unsupported ideas.
     - [0.7, 0.9]: Significant speculative, misleading, or conflicting content that clearly changes the meaning.
     - 1.0: Mostly off-topic, contradictory, or dominated by incorrect/hallucinated content.

5. If you are evaluating generated responses (skip if response is not a response):
   - Length alone does NOT increase the score. Extra length is only ok if it is consistent and not redundant.
   - A generated response that is much longer than the ground truth response should be penalized via P.
   - The generated response may or may not reuse phrases from the context; however, if the generated response just directly copies previous context, without quoting them, treat that as off-task behavior and give a score of 0.

6. Compute the final score = max(0, min(1, C - P))

Additional considerations:
- Follow the instruction carefully.
- Be strict and reserve scores above 0.8 for clearly outstanding matches.
- If a response contains non-text content, unnecessary wrappers like XML-like markup, or is otherwise malformed, apply a penalty by multiplying its score by 0.5. If there are multiple responses, you should contrast them against each other to ensure that your evaluations are consistent and assign different scores to different generated responses.

Output format (JSON):
{{
    "key_points": "<analysis of key points from ground truth along response dimension>",
    "1": {{"thought": "<how well the 1st generated response matches each key point and compute the final score>", "score": <score>}},
    "2": ...
}}

Format Notes:
- All text in "key_points" and "thought" fields MUST be on a single line with no line breaks or newlines
- Use standard JSON string format with double quotes. For any quotes needed inside strings, use single quotes (')
- Double check the JSON array's format, especially for the comma and quotation marks
- Ensure that ALL fields, especially "thought" and "score", are present for each item
- You must provide exactly 1 scores for the generated response(s)

Your output:
"""

STATE_JUDGE_PROMPT = """You are a helpful and meticulous evaluator. Your task is to score how well the generated response matches the ground truth response with respect to each aspect below.

Aspects (aspect_i: description_i):
{state_desc}

You will be given the context, the ground truth response, and ONE generated response.

Provided Information:
<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Ground Truth Response|>
{ground_truth}
<|The End of Ground Truth Response|>

<|The Start of Generated Response|>
{generation}
<|The End of Generated Response|>

Scoring Criteria (for each aspect_i):
Assign a score in [0, 1] based on how much the aspects of generated response and the ground truth matches.

For each aspect_i, follow this procedure:
1. Extract 1-3 key points:
   - Extract K key points from the ground truth response only about the aspect based on the description (e.g., if evaluating a "stance", pick key points related to the stance like "clearly disagrees with X").
   - Each key point should be specific and distinct.

2. Score how well the generated response matches each key point:
   - For each key point i, compare it with the generated response and assign a match value m_i in range [0, 1]:
     - 1.0: The key point is precisely and perfectly reflected.
     - [0.7, 0.9]: Mostly reflected with small imperfections.
     - [0.4, 0.6]: Partially reflected or vague, but still leaning in the correct direction.
     - [0.1, 0.3]: Very weak reflection.
     - 0.0: Missed, contradicted, or reversed.

3. Compute coverage C = (m_1 + m_2 + ... + m_K) / K, which measures how comprehensive the generated response reflects the ground truth response.

4. Compute penalty P for extra or conflicting content:
   - Examine additional content in the generated response beyond those key points:
     - Does it introduce unsupported evidence and assumptions?
     - Is it irrelevant to what ground truth response expresses?
   - Set a penalty P ∈ [0, 1]:
     - 0.0: No problematic extra content; everything is perfectly matched.
     - [0.1, 0.3]: Slightly unnecessary or mildly speculative detail; meaning essentially unchanged.
     - [0.4, 0.6]: Moderate speculative or irrelevant content that somewhat shifts emphasis or adds unsupported ideas.
     - [0.7, 0.9]: Significant speculative, misleading, or conflicting content that clearly changes the meaning.
     - 1.0: Mostly off-topic, contradictory, or dominated by incorrect/hallucinated content.

5. If you are evaluating generated responses (skip if aspect_i is not response):
   - Length alone does NOT increase the score. Extra length is only ok if it is consistent and not redundant.
   - A generated response that is much longer than the ground truth response should be penalized via P.
   - The generated response may or may not reuse phrases from the context; however, if the generated response just directly copies previous context, without quoting them, treat that as off-task behavior and give a score of 0.

6. Compute the final score = max(0, min(1, C - P))

Additional considerations:
- Follow the instruction carefully.
- Be strict and reserve scores above 0.8 for clearly outstanding matches.


Output format (JSON):
{{
    "<aspect_1>": {{"thought": "<include (1) analysis of key points from ground truth along the aspect_1 dimension; (2) how well the generated response matches each key point; (3) compute the final score>", "score": <score>}},
    "<aspect_2>": {{"thought": "<include (1) analysis of key points from ground truth along the aspect_2 dimension; (2) how well the generated response matches each key point; (3) compute the final score>", "score": <score>}},
    "<aspect_3>": ...
}}

Format Notes:
- Make sure to score the responses with respect to each aspect and consider only one aspect at a time.
- All text in "thought" fields MUST be on a single line with no line breaks or newlines
- Use standard JSON string format with double quotes. For any quotes needed inside strings, use single quotes (')
- Double check the JSON array's format, especially for the comma and quotation marks
- Ensure that ALL fields, especially "thought" and "score", are present for each item
- You must provide exactly 1 score for each of the aspects: {state_names}.

Your evaluation:
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "key_points": {"type": "string"},
        "1": {
            "type": "object",
            "properties": {"thought": {"type": "string"}, "score": {"type": "number"}},
            "required": ["thought", "score"],
            "additionalProperties": False,
        },
    },
    "required": ["key_points", "1"],
    "additionalProperties": False,
}

_STATE_SCHEMA = {
    "type": "object",
    "properties": {
        name: {
            "type": "object",
            "properties": {"thought": {"type": "string"}, "score": {"type": "number"}},
            "required": ["thought", "score"],
            "additionalProperties": False,
        }
        for name in STATE_NAMES
    },
    "required": list(STATE_NAMES),
    "additionalProperties": False,
}

_THINK_PAIR = re.compile(r"<\s*think\b[^>]*>.*?</\s*think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_PREFIX = re.compile(r"^.*?</\s*think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<\s*think\b[^>]*>", re.IGNORECASE)
_RESPONSE_PAIR = re.compile(r"<\s*response\s*>(.*?)</\s*response\s*>", re.DOTALL | re.IGNORECASE)
_RESPONSE_OPEN = re.compile(r"<\s*response\s*>", re.IGNORECASE)
_RESPONSE_CLOSE = re.compile(r"</\s*response\s*>", re.IGNORECASE)


def extract_response(text: str) -> str:
    """Match the official evaluation-time permissive response parser."""

    stripped = text
    previous = None
    while previous != stripped:
        previous = stripped
        stripped = _THINK_PAIR.sub("", stripped)
    stripped = _THINK_CLOSE_PREFIX.sub("", stripped)
    stripped = _THINK_OPEN.sub("", stripped)
    original = stripped
    match = _RESPONSE_PAIR.search(stripped)
    if match:
        return match.group(1).strip()
    opening = _RESPONSE_OPEN.search(stripped)
    if opening:
        suffix = stripped[opening.end() :].strip()
        return suffix or original.strip()
    closing = _RESPONSE_CLOSE.search(stripped)
    if closing:
        return stripped[: closing.start()].strip()
    return stripped.strip()


def _score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError(f"{label} must be numeric")
    return max(0.0, min(1.0, float(value)))


@adapter("humanual")
class HumanualAdapter(BenchmarkAdapter):
    benchmark_id = "humanual"
    prompt_revision = "humanlm-official-system-response-r-v2-inline-speakers"
    scorer_revision = "humanlm-official-alignment-plus-project-pinned-embedding-v2"

    def __init__(
        self,
        *,
        judge_provenance: JudgeProvenance | None = None,
        embedding_scorer: Any = None,
    ) -> None:
        self._judge_provenance = judge_provenance
        self._embedding_scorer = embedding_scorer

    @property
    def embedding_batch_size(self) -> int:
        value = getattr(self._embedding_scorer, "batch_size", 1)
        return max(1, int(value))

    def embedding_protocol_identity(self) -> Mapping[str, Any]:
        method = getattr(self._embedding_scorer, "protocol_identity", None)
        return dict(method()) if callable(method) else {}

    def needs_embedding(self, case: BenchmarkCase, result: CaseResult) -> bool:
        del case
        return (
            self._embedding_scorer is not None
            and result.status == ResultStatus.COMPLETED
            and isinstance(result.prediction, Mapping)
            and isinstance(result.prediction.get("response"), str)
        )

    @staticmethod
    def _embedding_metrics(
        domain: str,
        similarity: float | None,
        evidence: Mapping[str, Any] | None,
    ) -> tuple[MetricValue, ...]:
        metadata = {
            "paper_metric": True,
            "official_encoder_identity_available": False,
            "encoder_selection": "project_pinned_user_requested",
            "semantic_evidence": dict(evidence) if evidence else None,
        }
        return (
            MetricValue(
                "humanual.embedding_cosine_similarity",
                similarity,
                unit="cosine_similarity",
                metadata=metadata,
            ),
            MetricValue(
                "humanual.embedding_cosine_similarity.availability_rate",
                int(similarity is not None),
                unit="proportion",
            ),
            MetricValue(
                f"humanual.domain.{domain}.embedding_cosine_similarity",
                similarity,
                unit="cosine_similarity",
                metadata=metadata,
            ),
        )

    def finalize_embedding_batch(
        self,
        items: Sequence[tuple[BenchmarkCase, CaseResult, int]],
    ) -> tuple[CaseResult, ...]:
        """Batch-score deferred response/reference pairs and preserve case order."""

        if not items:
            return ()
        score_batch = getattr(self._embedding_scorer, "score_batch", None)
        if not callable(score_batch):
            raise ConfigurationError(
                "deferred HUMANUAL embedding scoring requires a batch-capable scorer"
            )
        requests = tuple(
            (str(result.prediction["response"]), str(case.gold))
            for case, result, _ in items
        )
        evidence_items = tuple(score_batch(requests))
        if len(evidence_items) != len(items):
            raise ValidationError(
                "HUMANUAL embedding scorer result count does not match deferred case count"
            )
        finalized = []
        embedding_metric_names = {
            "humanual.embedding_cosine_similarity",
            "humanual.embedding_cosine_similarity.availability_rate",
        }
        for (case, result, _), raw_evidence in zip(items, evidence_items):
            if not isinstance(raw_evidence, Mapping):
                raise ValidationError("HUMANUAL embedding evidence must be an object")
            try:
                similarity = float(raw_evidence["similarity"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(
                    "HUMANUAL embedding evidence requires numeric similarity"
                ) from exc
            domain = str(case.input_data["domain"])
            embedding_metric_names.add(
                f"humanual.domain.{domain}.embedding_cosine_similarity"
            )
            metrics = tuple(
                metric for metric in result.metrics if metric.name not in embedding_metric_names
            ) + self._embedding_metrics(domain, similarity, raw_evidence)
            finalized.append(
                replace(
                    result,
                    metrics=metrics,
                    metadata={
                        **dict(result.metadata),
                        "embedding_cosine": dict(raw_evidence),
                    },
                )
            )
        return tuple(finalized)

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"HumanualAdapter cannot run {case.benchmark_id!r}")
        probe_case(case)
        if case.split not in {"test", "fixture"}:
            raise ValidationError("HUMANUAL evaluation requires the official test split")
        if not isinstance(case.gold, str) or not case.gold.strip():
            raise ValidationError("HUMANUAL requires a nonempty ground-truth response")

    @staticmethod
    def _messages(case: BenchmarkCase) -> tuple[ChatMessage, ...]:
        domain = str(case.input_data["domain"])
        target_user = str(case.input_data["target_user_id"])
        result: list[ChatMessage] = []
        for raw in case.input_data["prompt"]:
            source_role = str(raw["role"])
            content = str(raw["content"])
            if domain == "chat":
                is_assistant = "gpt" in source_role.casefold()
                result.append(
                    ChatMessage(
                        "assistant" if is_assistant else "user",
                        f"{'ASSISTANT' if is_assistant else 'HUMAN'}: {content}",
                        metadata={"humanual_source_role": source_role,
                                  "official_name": "ASSISTANT" if is_assistant else "HUMAN",
                                  "source_content": content},
                    )
                )
            else:
                official_name = "HUMAN" if source_role == target_user else source_role
                result.append(
                    ChatMessage(
                        "user",
                        f"{official_name}: {content}",
                        metadata={"humanual_source_role": source_role, "official_name": official_name,
                                  "source_content": content},
                    )
                )
        return tuple(result)

    @classmethod
    def _context(cls, case: BenchmarkCase) -> str:
        lines = []
        for message in cls._messages(case):
            # Keep the judge's existing context format; inline transport labels
            # belong to actor requests, not a second copy in judge text.
            official_name = message.metadata["official_name"]
            if case.input_data["domain"] == "chat" and message.role == "assistant":
                official_name = None
            lines.append(f"**{message.role.capitalize()} {official_name}**: {message.metadata['source_content']}")
        return "\n".join(lines)

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        self.validate_case(case)
        persona = str(case.input_data["persona"])
        return ModelRequest(
            request_id=f"{case.case_id}:response",
            messages=(ChatMessage("system", SYSTEM_PROMPT.format(persona=persona)), *self._messages(case)),
            model=model,
            temperature=0.0,
            max_tokens=1024,
            seed=seed,
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": "evaluated_model",
                "route_role": "evaluated_model",
                "prompt_revision": self.prompt_revision,
                "official_eval_temperature": 0.4,
                "configured_eval_temperature": 0.0,
                "official_eval_no_repeat_ngram_size": 4,
                "no_repeat_ngram_transport": "not_expressible_in_portable_openai_chat_api",
                "official_base_max_tokens": 512,
                "official_thinking_or_humanlm_max_tokens": 1024,
                "generation_role_transport": "assistant_api_surface_for_official_HUMAN_turn",
            },
        )

    def parse_response(self, case: BenchmarkCase, response: ModelResponse) -> str:
        del case
        return extract_response(response.text)

    def provenance_for_case(self, case: BenchmarkCase) -> JudgeProvenance | None:
        if self._judge_provenance is not None:
            return self._judge_provenance
        replay = case.metadata.get("replay")
        raw = replay.get("judge_provenance") if isinstance(replay, Mapping) else None
        if not isinstance(raw, Mapping):
            return None
        return JudgeProvenance(
            judge_models=tuple(str(value) for value in raw.get("judge_models") or ()),
            judge_revisions=tuple(str(value) for value in raw.get("judge_revisions") or ()),
            rubric_revision=str(raw.get("rubric_revision") or ""),
            calls_per_output=int(raw.get("calls_per_output", 2)),
            source=str(raw.get("source") or "fixture"),
        )

    def _judge_request(
        self,
        case: BenchmarkCase,
        generation: str,
        *,
        kind: str,
        seed: int,
        provenance: JudgeProvenance,
    ) -> ModelRequest:
        if len(provenance.judge_models) != 1:
            raise ConfigurationError("HUMANUAL requires exactly one pinned judge model")
        if kind == "response":
            prompt = RESPONSE_JUDGE_PROMPT.format(
                context=self._context(case),
                ground_truth=case.gold,
                generations_json=json.dumps({"1": generation.strip()}, indent=2),
            )
            response_format = json_schema_response_format("humanual_response_alignment", _RESPONSE_SCHEMA)
        elif kind == "state":
            prompt = STATE_JUDGE_PROMPT.format(
                state_desc=json.dumps(STATE_DESCRIPTIONS, indent=2),
                context=self._context(case),
                ground_truth=case.gold,
                generation=generation,
                state_names=list(STATE_NAMES),
            )
            response_format = json_schema_response_format("humanual_state_alignment", _STATE_SCHEMA)
        else:
            raise AssertionError(kind)
        return ModelRequest(
            request_id=f"{case.case_id}:judge:{kind}",
            messages=(ChatMessage("user", prompt),),
            model=provenance.judge_models[0],
            temperature=0.0,
            max_tokens=4096,
            seed=seed,
            response_format=response_format,
            metadata={
                "benchmark_id": self.benchmark_id,
                "actor": "judge",
                "route_role": "judge",
                "judge_kind": kind,
                "output_contract": "official HUMANUAL judge JSON schema",
                "preserve_request_sampling": True,
            },
        )

    @staticmethod
    def _parse_response_judgment(response: ModelResponse) -> Mapping[str, Any]:
        payload = parse_json_object(response, label="HUMANUAL response judge output")
        if set(payload) != {"key_points", "1"} or not isinstance(payload.get("key_points"), str):
            raise ParseError("HUMANUAL response judge requires exactly key_points and item 1")
        item = payload["1"]
        if not isinstance(item, Mapping) or not isinstance(item.get("thought"), str):
            raise ParseError("HUMANUAL response judge item 1 requires thought and score")
        return {
            "key_points": payload["key_points"],
            "thought": item["thought"],
            "score": _score(item.get("score"), "HUMANUAL response score"),
        }

    @staticmethod
    def _parse_state_judgment(response: ModelResponse) -> Mapping[str, Any]:
        payload = parse_json_object(response, label="HUMANUAL state judge output")
        if set(payload) != set(STATE_NAMES):
            raise ParseError(f"HUMANUAL state judge requires exactly {list(STATE_NAMES)}")
        parsed = {}
        for name in STATE_NAMES:
            item = payload[name]
            if not isinstance(item, Mapping) or not isinstance(item.get("thought"), str):
                raise ParseError(f"HUMANUAL state judge {name!r} requires thought and score")
            parsed[name] = {
                "thought": item["thought"],
                "score": _score(item.get("score"), f"HUMANUAL {name} score"),
            }
        return parsed

    def replay_responses(self, case: BenchmarkCase, *, seed: int) -> Mapping[str, Any]:
        del seed
        replay = case.metadata.get("replay")
        if not isinstance(replay, Mapping):
            raise ConfigurationError(f"fixture {case.case_id} requires HUMANUAL replay metadata")
        required = ("response", "response_judge", "state_judge")
        missing = [name for name in required if name not in replay]
        if missing:
            raise ConfigurationError(f"HUMANUAL fixture lacks replay fields: {missing}")
        return {
            f"{case.case_id}:response": replay["response"],
            f"{case.case_id}:judge:response": canonical_json(replay["response_judge"]),
            f"{case.case_id}:judge:state": canonical_json(replay["state_judge"]),
        }

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
        target_request = self.build_request(case, model=model, seed=seed)
        responses: list[ModelResponse] = []
        try:
            target_response = previous_result.model_response if previous_result is not None else backend.generate(target_request)
            assert target_response is not None
            if previous_result is None:
                responses.append(target_response)
        except BackendError as exc:
            return CaseResult(
                run_id=run_id,
                benchmark_id=self.benchmark_id,
                case_id=case.case_id,
                group_id=case.group_id,
                repetition=repetition,
                status=ResultStatus.FAILED,
                error=ErrorState("model_backend", "backend_failure", str(exc), retryable=True),
                metadata={"domain": case.input_data["domain"]},
            )

        generation = self.parse_response(case, target_response)
        provenance = self.provenance_for_case(case)
        judgments: dict[str, Any] = dict(previous_result.metadata.get("judge_records") or {}) if previous_result else {}
        if previous_result is not None:
            saved_scores = {m.name: m.value for m in previous_result.metrics}
            if "response" not in judgments and saved_scores.get("humanual.response_alignment") is not None:
                judgments["response"] = {"score": saved_scores["humanual.response_alignment"]}
            if "state" not in judgments and all(saved_scores.get(f"humanual.state.{name}_alignment") is not None for name in STATE_NAMES):
                judgments["state"] = {name: {"score": saved_scores[f"humanual.state.{name}_alignment"]} for name in STATE_NAMES}
        judge_errors: dict[str, Any] = {}
        scoring_shortcut: str | None = None
        if generation.strip() == case.gold.strip():
            scoring_shortcut = "official_exact_ground_truth"
            judgments = {
                "response": {
                    "key_points": "exact ground-truth response",
                    "thought": "Official reward code assigns one to an exact ground-truth match.",
                    "score": 1.0,
                },
                "state": {
                    name: {
                        "thought": "Official reward code assigns one to an exact ground-truth match.",
                        "score": 1.0,
                    }
                    for name in STATE_NAMES
                },
            }
        elif not generation.strip():
            scoring_shortcut = "official_empty_generation"
            judgments = {
                "response": {
                    "key_points": "",
                    "thought": "Official reward code assigns zero to an empty generation.",
                    "score": 0.0,
                },
                "state": {
                    name: {
                        "thought": "Official reward code assigns zero to an empty generation.",
                        "score": 0.0,
                    }
                    for name in STATE_NAMES
                },
            }
        elif provenance is None:
            judge_errors = {
                "response": {"kind": "missing_configuration", "message": "no HUMANUAL judge configured"},
                "state": {"kind": "missing_configuration", "message": "no HUMANUAL judge configured"},
            }
        else:
            for offset, kind in enumerate(("response", "state"), start=1):
                if kind in judgments:
                    continue
                request = self._judge_request(
                    case,
                    generation,
                    kind=kind,
                    seed=seed + offset,
                    provenance=provenance,
                )
                parser = self._parse_response_judgment if kind == "response" else self._parse_state_judgment
                try:
                    judgments[kind] = generate_and_parse_with_contract_retries(
                        backend=backend,
                        request=request,
                        parser=parser,
                        responses=responses,
                        contract=str(request.metadata["output_contract"]),
                        max_retries=4,
                    )
                except (BackendError, ParseError) as exc:
                    judge_errors[kind] = {"kind": type(exc).__name__, "message": str(exc)}

        response_score = judgments.get("response", {}).get("score")
        state_scores = {
            name: judgments.get("state", {}).get(name, {}).get("score")
            for name in STATE_NAMES
        }
        available_states = [float(value) for value in state_scores.values() if value is not None]
        state_mean = sum(available_states) / len(available_states) if len(available_states) == len(STATE_NAMES) else None
        domain = str(case.input_data["domain"])
        provenance_data = provenance.to_dict() if provenance else None
        metrics: list[MetricValue] = [
            MetricValue(
                "humanual.response_alignment",
                response_score,
                unit="proportion",
                metadata={"official_primary": True, "judge_provenance": provenance_data},
            ),
            MetricValue(
                "humanual.state_alignment",
                state_mean,
                unit="proportion",
                metadata={"official_metric": True, "aggregation": "mean_six_state_dimensions"},
            ),
            MetricValue(
                "humanual.response_alignment.availability_rate",
                int(response_score is not None),
                unit="proportion",
            ),
            MetricValue(
                "humanual.state_alignment.availability_rate",
                int(state_mean is not None),
                unit="proportion",
            ),
            MetricValue(f"humanual.domain.{domain}.response_alignment", response_score, unit="proportion"),
            MetricValue(f"humanual.domain.{domain}.state_alignment", state_mean, unit="proportion"),
            *self._embedding_metrics(domain, None, None),
        ]
        for name, value in state_scores.items():
            metrics.extend(
                (
                    MetricValue(f"humanual.state.{name}_alignment", value, unit="proportion"),
                    MetricValue(f"humanual.domain.{domain}.state.{name}_alignment", value, unit="proportion"),
                )
            )
        trace = [
            TraceEvent(
                turn=0,
                actor="evaluated_model",
                kind="response",
                content=generation,
                visible_to=("evaluator",),
                metadata={"raw_output": target_response.text},
            )
        ]
        for kind in ("response", "state"):
            if kind in judgments:
                trace.append(
                    TraceEvent(
                        turn=len(trace),
                        actor="judge",
                        kind=f"{kind}_alignment",
                        content=judgments[kind],
                        visible_to=("evaluator",),
                    )
                )
        known_latencies = [response.latency_ms for response in responses if response.latency_ms is not None]
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction={"response": generation},
            metrics=tuple(metrics),
            trace=tuple(trace),
            model_response=target_response,
            latency_ms=sum(known_latencies) if known_latencies else None,
            token_usage=combine_usage(response.usage for response in responses),
            metadata={
                "domain": domain,
                "evaluation_complete": True,
                "judge_provenance": provenance_data,
                "judge_error": judge_errors or None,
                "judge_records": judgments,
                "scoring_shortcut": scoring_shortcut,
                "response_tag_observed": bool(_RESPONSE_OPEN.search(target_response.text)),
                "output_truncated": bool(
                    isinstance(target_response.raw, Mapping)
                    and isinstance(target_response.raw.get("_sim_eval"), Mapping)
                    and target_response.raw["_sim_eval"].get("output_truncated")
                ),
            },
        )

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        metrics = dict(aggregate_named_metrics(results, namespace="humanual"))
        observed_domains = sorted(
            {
                str(result.metadata.get("domain"))
                for result in results
                if result.status == ResultStatus.COMPLETED and result.metadata.get("domain")
            }
        )
        for suffix in (
            "response_alignment",
            "state_alignment",
            "embedding_cosine_similarity",
            *(f"state.{name}_alignment" for name in STATE_NAMES),
        ):
            domain_values = [
                metrics[f"humanual.domain.{domain}.{suffix}"].value
                for domain in observed_domains
                if f"humanual.domain.{domain}.{suffix}" in metrics
                and metrics[f"humanual.domain.{domain}.{suffix}"].value is not None
            ]
            metric_name = f"humanual.{suffix}"
            metrics[metric_name] = MetricValue(
                metric_name,
                sum(float(value) for value in domain_values) / len(domain_values) if domain_values else None,
                unit="proportion",
                numerator=sum(float(value) for value in domain_values) if domain_values else None,
                denominator=len(domain_values),
                metadata={
                    "aggregation": "macro_mean_available_domains",
                    "observed_domains": observed_domains,
                    "official_six_domain_complete": set(observed_domains) == set(HUMANUAL_DOMAINS),
                },
            )
        return metrics

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        del case
        return {
            "revision": "humanual-official-context-role-mapping-v1",
            "domain_policy": "chat-assistant-role-branch;all-other-domains-named-user-messages",
            "response_parser": "official-permissive-response-tag-parser-v1",
            "target_max_tokens": 1024,
            "target_temperature": 0.0,
            "official_target_temperature": 0.4,
            "api_stop_sequences": [],
        }


__all__ = ["HumanualAdapter", "extract_response"]
