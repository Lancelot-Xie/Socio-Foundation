"""BehaviorChain longitudinal prediction and generation adapter."""

from __future__ import annotations

import json
from ..presentation import material_user_message

from collections import defaultdict
from dataclasses import dataclass
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
from ..errors import BackendError, BackendStructuredOutputError, ConfigurationError, EpisodeTokenBudgetExhausted, ParseError, ValidationError
from ..interfaces import BenchmarkAdapter, ModelBackend
from ..json_utils import canonical_json
from ..registry import adapter
from ..model_adaptation import adapted_choice_text
from .choice import (
    ChoicePrediction,
    DisplayedChoice,
    alphabetical_labels,
    indexed_choices,
    parse_failure_rate,
    parse_strict_choice,
)
from .common import (
    aggregate_named_metrics,
    combine_usage,
    generate_and_parse_with_contract_retries,
    json_schema_response_format,
)
from .behaviorchain_names import NAME_POLICY_REVISION, name_aliases_for_case, pseudonymize_material


@dataclass(frozen=True)
class BehaviorChainJudgeProvenance:
    model: str
    model_revision: str
    rubric_revision: str
    prompt_revision: str
    source: str
    replayed: bool

    def __post_init__(self) -> None:
        for field in ("model", "model_revision", "rubric_revision", "prompt_revision", "source"):
            if not getattr(self, field):
                raise ValidationError(f"BehaviorChain judge provenance requires {field}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_revision": self.model_revision,
            "rubric_revision": self.rubric_revision,
            "prompt_revision": self.prompt_revision,
            "source": self.source,
            "replayed": self.replayed,
            "temperature": 0.0,
            "output_schema": {"aligned": "boolean", "reasoning": "nonempty string"},
        }


def normalized_cumulative_chain_score(values: Sequence[int | bool]) -> float:
    """Official released-code CumScore: triangular credit for each correct run."""

    if not values:
        raise ValidationError("BehaviorChain CumScore requires at least one node")
    running = 0
    numerator = 0
    for value in values:
        if value not in (0, 1, False, True):
            raise ValidationError("BehaviorChain node values must be binary")
        if bool(value):
            running += 1
            numerator += running
        else:
            running = 0
    denominator = len(values) * (len(values) + 1) / 2
    return numerator / denominator


def _generation_payload(text: str) -> str:
    raw = text.strip()
    if not raw:
        raise ParseError("BehaviorChain generation response is empty")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError("BehaviorChain generation response must be JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"behavior"}:
        raise ParseError("BehaviorChain generation JSON requires exactly the behavior field")
    behavior = payload.get("behavior")
    if not isinstance(behavior, str) or not behavior.strip():
        raise ParseError("BehaviorChain generated behavior must be nonempty")
    return behavior.strip()


def _judge_payload(text: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ParseError("BehaviorChain judge response must be JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"aligned", "reasoning"}:
        raise ParseError("BehaviorChain judge JSON requires exactly aligned and reasoning")
    if not isinstance(payload["aligned"], bool):
        raise ParseError("BehaviorChain judge aligned must be boolean")
    if not isinstance(payload["reasoning"], str) or not payload["reasoning"].strip():
        raise ParseError("BehaviorChain judge reasoning must be nonempty")
    return {"aligned": payload["aligned"], "reasoning": payload["reasoning"].strip()}


@adapter("behaviorchain")
class BehaviorChainAdapter(BenchmarkAdapter):
    benchmark_id = "behaviorchain"
    prompt_revision = "behaviorchain-v4-chain-stable-pseudonyms"
    scorer_revision = "behaviorchain-avgscore-cumscore-v1"

    def __init__(
        self,
        *,
        judge_provenance: BehaviorChainJudgeProvenance | None = None,
    ) -> None:
        self._judge_provenance = judge_provenance

    @staticmethod
    def task_mode(case: BenchmarkCase) -> str:
        mode = str(case.input_data.get("task_mode"))
        if mode in {"prediction", "multiple_choice"}:
            return "prediction"
        if mode == "generation":
            return "generation"
        raise ValidationError("BehaviorChain task_mode must be prediction/multiple_choice or generation")

    @staticmethod
    def chain_metadata(case: BenchmarkCase) -> Mapping[str, Any]:
        strata = case.metadata.get("strata") or {}
        return {
            "chain_index": case.input_data.get("chain_index"),
            "chain_length": case.input_data.get("chain_length"),
            "task_mode": BehaviorChainAdapter.task_mode(case),
            "key_behavior_status": strata.get("key_behavior_status"),
            "chain_length_bucket": strata.get("chain_length_bucket"),
            "history_mode": case.input_data.get("history_mode", "gold_behavior_history"),
            "persona_type": case.input_data.get("persona_type", "unspecified"),
        }

    def judge_provenance_for_case(self, case: BenchmarkCase) -> BehaviorChainJudgeProvenance | None:
        if self._judge_provenance is not None:
            return self._judge_provenance
        raw = case.metadata.get("judge_provenance")
        replay = case.metadata.get("replay")
        if raw is None and isinstance(replay, Mapping):
            raw = replay.get("judge_provenance")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValidationError("BehaviorChain judge_provenance must be an object")
        replayed = raw.get("replayed", False)
        if not isinstance(replayed, bool):
            raise ValidationError("BehaviorChain judge_provenance.replayed must be boolean")
        return BehaviorChainJudgeProvenance(
            model=str(raw.get("model") or ""),
            model_revision=str(raw.get("model_revision") or ""),
            rubric_revision=str(raw.get("rubric_revision") or ""),
            prompt_revision=str(raw.get("prompt_revision") or ""),
            source=str(raw.get("source") or ""),
            replayed=replayed,
        )

    def provenance_for_case(self, case: BenchmarkCase) -> BehaviorChainJudgeProvenance | None:
        return self.judge_provenance_for_case(case)

    def prediction_choices(self, case: BenchmarkCase) -> tuple[DisplayedChoice, ...]:
        candidates = case.input_data.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or len(candidates) != 4:
            raise ValidationError("BehaviorChain prediction requires exactly four candidates")
        labels = alphabetical_labels(4)
        return indexed_choices(candidates, labels=labels, source_ids=tuple(str(index) for index in range(4)))

    def gold_source_id(self, case: BenchmarkCase) -> str:
        gold = case.gold
        if isinstance(gold, int) and not isinstance(gold, bool) and 0 <= gold < 4:
            return str(gold)
        if isinstance(gold, str):
            normalized = gold.strip().upper().rstrip(".)")
            if normalized in alphabetical_labels(4):
                return str(ord(normalized) - ord("A"))
            if normalized in {"0", "1", "2", "3"}:
                return normalized
        raise ValidationError("BehaviorChain prediction gold must identify one of four candidates")

    def validate_case(self, case: BenchmarkCase) -> None:
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError(f"BehaviorChainAdapter cannot run {case.benchmark_id!r}")
        probe_case(case)
        values = case.input_data
        if not isinstance(values.get("persona"), (str, Mapping)) or not values.get("persona"):
            raise ValidationError("BehaviorChain persona must be nonempty text or an object")
        history = values.get("history")
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise ValidationError("BehaviorChain history must be an array")
        chain_index = values.get("chain_index")
        chain_length = values.get("chain_length")
        if not isinstance(chain_index, int) or isinstance(chain_index, bool) or chain_index < 0:
            raise ValidationError("BehaviorChain chain_index must be a nonnegative integer")
        if not isinstance(chain_length, int) or isinstance(chain_length, bool) or chain_length <= 0:
            raise ValidationError("BehaviorChain chain_length must be positive")
        if chain_index >= chain_length:
            raise ValidationError("BehaviorChain chain_index must be smaller than chain_length")
        prior_nodes = values.get("prior_nodes", ())
        if not isinstance(prior_nodes, Sequence) or isinstance(prior_nodes, (str, bytes)):
            raise ValidationError("BehaviorChain prior_nodes must be an array")
        if len(prior_nodes) != chain_index:
            raise ValidationError("BehaviorChain prior_nodes count must equal chain_index")
        for index, node in enumerate(prior_nodes):
            if not isinstance(node, Mapping):
                raise ValidationError(f"BehaviorChain prior node #{index} must be an object")
            if set(node) != {"context", "behavior"}:
                raise ValidationError(
                    f"BehaviorChain prior node #{index} requires exactly context and behavior"
                )
            if not all(isinstance(node[field], str) and node[field].strip() for field in ("context", "behavior")):
                raise ValidationError(
                    f"BehaviorChain prior node #{index} context and behavior must be nonempty text"
                )
        current_context = values.get("current_context")
        if not isinstance(current_context, str) or not current_context.strip():
            raise ValidationError("BehaviorChain current_context must be nonempty")
        status = (case.metadata.get("strata") or {}).get("key_behavior_status")
        if status not in {"key", "subkey"}:
            raise ValidationError("BehaviorChain key_behavior_status must be key or subkey")
        mode = self.task_mode(case)
        if mode == "prediction":
            self.prediction_choices(case)
            self.gold_source_id(case)
        else:
            if not isinstance(case.gold, str) or not case.gold.strip():
                raise ValidationError("BehaviorChain generation gold/reference behavior must be nonempty")

    def _public_input(self, case: BenchmarkCase) -> Mapping[str, Any]:
        values = case.input_data
        return {
            "persona": values["persona"],
            "background_history": list(values.get("history") or ()),
            "prior_context_behavior_nodes": list(values.get("prior_nodes") or ()),
            "current_context": values["current_context"],
            "history_mode": values.get("history_mode", "gold_behavior_history"),
            "node_index": values["chain_index"],
        }

    def build_request(self, case: BenchmarkCase, *, model: str, seed: int) -> ModelRequest:
        self.validate_case(case)
        mode = self.task_mode(case)
        payload = dict(self._public_input(case))
        if mode == "prediction":
            payload["options"] = [choice.public_dict() for choice in self.prediction_choices(case)]
            output_instruction = (
                "Please think step by step, then put the final choice inside <answer></answer> tags, "
                "for example <answer>A</answer>. The answer must be exactly one of A, B, C, or D."
            )
            response_format = None
            max_tokens = 1024
        else:
            output_instruction = 'Return only JSON {"behavior":"a concise next behavior"}.'
            response_format = {"type": "json_object"}
            max_tokens = 1024
        aliases = name_aliases_for_case(case)
        payload = pseudonymize_material(payload, aliases)
        return ModelRequest(
            request_id=f"{case.case_id}:{'choice' if mode == 'prediction' else 'generation'}",
            messages=(
                ChatMessage(
                    "system",
                    "Continue the specified persona's behavior chain using only prior history and the current context.",
                ),
                material_user_message(payload, {'persona': 'Persona Profile', 'background_history': 'Historical Narrative Before the Behavior Chain', 'prior_context_behavior_nodes': 'Prior Context-Behavior Chain', 'current_context': 'Current Context', 'history_mode': 'History Mode', 'node_index': 'Node Index', 'options': 'Options'}, truncatable=('persona', 'background_history', 'prior_context_behavior_nodes'), suffix="\n" + output_instruction),
            ),
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            seed=seed,
            response_format=response_format,
            metadata={
                "benchmark_id": self.benchmark_id,
                "route_role": "evaluated_model",
                "prompt_revision": self.prompt_revision,
                "task_mode": mode,
                "chain_index": case.input_data["chain_index"],
                "gold_visible": False,
                "name_policy_revision": NAME_POLICY_REVISION,
                "name_aliases": aliases,
                "name_mapping_scope": "fixed_profile_and_background_per_chain_not_per_case_seed",
            },
        )

    def parse_response(self, case: BenchmarkCase, response: ModelResponse) -> ChoicePrediction | str:
        if self.task_mode(case) == "prediction":
            return parse_strict_choice(
                adapted_choice_text(response, self.prediction_choices(case)),
                self.prediction_choices(case),
                allow_answer_tag=True,
            )
        return _generation_payload(response.text)

    def build_judge_request(
        self,
        case: BenchmarkCase,
        behavior: str,
        provenance: BehaviorChainJudgeProvenance,
        *,
        seed: int,
    ) -> ModelRequest:
        payload = {
            **pseudonymize_material(self._public_input(case), name_aliases_for_case(case)),
            "generated_next_behavior": behavior,
            "evaluation_question": "Is this behavior consistent with both the current context and the supplied persona?",
        }
        output_contract = (
            'Return only one JSON object with exactly two fields: '
            '{"aligned":true,"reasoning":"a non-empty evidence-based explanation"}. '
            "aligned must be a JSON boolean, not a string; reasoning must cite the supplied context/persona; "
            "do not emit markdown or prose outside the object."
        )
        schema = {
            "type": "object",
            "properties": {
                "aligned": {"type": "boolean"},
                "reasoning": {"type": "string", "minLength": 1},
            },
            "required": ["aligned", "reasoning"],
            "additionalProperties": False,
        }
        return ModelRequest(
            request_id=f"{case.case_id}:judge",
            messages=(
                ChatMessage(
                    "system",
                    "Evaluate whether the generated next behavior is plausible given both the complete prior "
                    "behavior chain/current context and the supplied persona. Mark aligned=false if it contradicts "
                    "either source or introduces an unsupported behavioral jump. This is an indieval diagnostic "
                    f"judge because the released BehaviorChain code does not define a generation-judge prompt. {output_contract}",
                    metadata={"visibility": "evaluator_only"},
                ),
                ChatMessage("user", canonical_json(payload), metadata={"visibility": "evaluator_only"}),
            ),
            model=provenance.model,
            temperature=0.0,
            max_tokens=256,
            seed=seed + 1000,
            response_format=json_schema_response_format("behaviorchain_alignment", schema),
            metadata={
                "benchmark_id": self.benchmark_id,
                "route_role": "generation_judge",
                "rubric_revision": provenance.rubric_revision,
                "prompt_revision": provenance.prompt_revision,
                "gold_reference_visible": False,
                "prompt_source": "indieval_nonofficial_diagnostic_no_released_upstream_generation_judge",
                "output_contract": output_contract,
                "name_policy_revision": NAME_POLICY_REVISION,
            },
        )

    def replay_responses(self, case: BenchmarkCase, *, seed: int) -> Mapping[str, Any]:
        replay = case.metadata.get("replay")
        if not isinstance(replay, Mapping):
            raise ConfigurationError(f"fixture {case.case_id} requires metadata.replay")
        mode = self.task_mode(case)
        responses: dict[str, Any] = {}
        if mode == "prediction":
            if "response" in replay:
                responses[f"{case.case_id}:choice"] = replay["response"]
            else:
                source_id = str(replay.get("choice_source_index"))
                choices = self.prediction_choices(case)
                matched = [choice for choice in choices if choice.source_id == source_id]
                if len(matched) != 1:
                    raise ConfigurationError("BehaviorChain replay choice_source_index is invalid")
                responses[f"{case.case_id}:choice"] = f"<answer>{matched[0].display_id}</answer>"
            return responses
        behavior = replay.get("behavior")
        if not isinstance(behavior, str) or not behavior.strip():
            raise ConfigurationError("BehaviorChain generation replay requires behavior")
        responses[f"{case.case_id}:generation"] = canonical_json({"behavior": behavior})
        if "judge" in replay:
            if self.judge_provenance_for_case(case) is None:
                raise ConfigurationError("BehaviorChain replay judge requires judge provenance")
            responses[f"{case.case_id}:judge"] = canonical_json(replay["judge"])
        return responses

    def environment_identity_for_case(self, case: BenchmarkCase) -> Mapping[str, Any]:
        del case
        return {
            "revision": "static-longitudinal-chain-v3-pseudonyms",
            "name_policy_revision": NAME_POLICY_REVISION,
            "history_policy": "case-declared_gold_or_chosen_behavior_history",
            "prediction_option_policy": "preserve_authorized_source_order",
            "parser_revision": "prediction-answer-tag-legacy-json-compat-generation-json-v3",
        }

    def _base_metadata(self, case: BenchmarkCase) -> dict[str, Any]:
        return {
            "behaviorchain": {
                **dict(self.chain_metadata(case)),
                "gold_visible_to_model": False,
                "name_policy_revision": NAME_POLICY_REVISION,
                "name_aliases": name_aliases_for_case(case),
            },
            "evaluation_complete": False,
        }

    def _failed_result(
        self,
        case: BenchmarkCase,
        *,
        run_id: str,
        repetition: int,
        stage: str,
        exc: Exception,
        response: ModelResponse | None,
    ) -> CaseResult:
        if isinstance(exc, (ParseError, BackendStructuredOutputError)):
            mode = self.task_mode(case)
            failure = {
                "stage": stage,
                "kind": "parse_failure",
                "message": str(exc),
                "retryable": False,
                "scoring_policy": "target_capability_failure_scores_zero",
            }
            metrics = [
                MetricValue(
                    "behaviorchain.node_score",
                    0,
                    unit="proportion",
                    numerator=0,
                    denominator=1,
                    metadata={"parse_error": str(exc)},
                )
            ]
            metrics.append(
                MetricValue(
                    "behaviorchain.prediction.node_accuracy"
                    if mode == "prediction"
                    else "behaviorchain.generation.node_judge_score",
                    0,
                    unit="proportion",
                    numerator=0,
                    denominator=1,
                    metadata={"parse_error": str(exc)},
                )
            )
            metadata = self._base_metadata(case)
            metadata.update(
                {
                    "evaluation_complete": True,
                    "target_output_failure": failure,
                    "behaviorchain": {
                        **dict(metadata["behaviorchain"]),
                        "judge_error": None,
                    },
                }
            )
            return CaseResult(
                run_id=run_id,
                benchmark_id=self.benchmark_id,
                case_id=case.case_id,
                group_id=case.group_id,
                repetition=repetition,
                status=ResultStatus.COMPLETED,
                prediction={"raw_response": response.text if response else None, "parsed": False},
                metrics=tuple(metrics),
                trace=(
                    TraceEvent(
                        turn=int(case.input_data["chain_index"]),
                        actor="evaluated_model",
                        kind="target_output_parse_failure",
                        content={"raw_response": response.text if response else None, **failure},
                        visible_to=("evaluator",),
                    ),
                ),
                model_response=response,
                latency_ms=response.latency_ms if response else None,
                token_usage=response.usage if response else None,
                metadata=metadata,
            )
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.FAILED,
            prediction={"raw_response": response.text if response else None},
            trace=(
                TraceEvent(
                    turn=int(case.input_data["chain_index"]),
                    actor="evaluated_model",
                    kind="response_failure",
                    content=response.text if response else str(exc),
                    visible_to=("evaluator",),
                ),
            ),
            model_response=response,
            error=ErrorState(
                stage=stage,
                kind=("token_budget_exhausted" if isinstance(exc, EpisodeTokenBudgetExhausted)
                      else "backend_failure" if isinstance(exc, BackendError) else "parse_failure"),
                message=str(exc),
                retryable=isinstance(exc, BackendError),
            ),
            latency_ms=response.latency_ms if response else None,
            token_usage=response.usage if response else None,
            metadata={**self._base_metadata(case),
                      **({"token_budget": exc.details()} if isinstance(exc, EpisodeTokenBudgetExhausted) else {})},
        )

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
        request = self.build_request(case, model=model, seed=seed)
        responses: list[ModelResponse] = []
        try:
            response = previous_result.model_response if previous_result is not None else backend.generate(request)
            assert response is not None
            if previous_result is None:
                responses.append(response)
        except (BackendError, EpisodeTokenBudgetExhausted) as exc:
            return self._failed_result(
                case,
                run_id=run_id,
                repetition=repetition,
                stage="token_budget" if isinstance(exc, EpisodeTokenBudgetExhausted) else "model_backend",
                exc=exc,
                response=None,
            )
        try:
            prediction = self.parse_response(case, response)
        except ParseError as exc:
            return self._failed_result(
                case,
                run_id=run_id,
                repetition=repetition,
                stage="response_parse",
                exc=exc,
                response=response,
            )

        mode = self.task_mode(case)
        judge_record: Mapping[str, Any] | None = None
        judge_error: Mapping[str, Any] | None = None
        if mode == "prediction":
            assert isinstance(prediction, ChoicePrediction)
            node_score: int | None = int(prediction.source_id == self.gold_source_id(case))
            public_prediction: Any = prediction.to_dict()
            prediction_metric = node_score
            generation_metric = None
        else:
            assert isinstance(prediction, str)
            public_prediction = {"behavior": prediction}
            prediction_metric = None
            generation_metric = None
            provenance = self.judge_provenance_for_case(case)
            if provenance is None:
                judge_error = {"kind": "missing_configuration", "message": "no named generation judge configured"}
            else:
                judge_request = self.build_judge_request(case, prediction, provenance, seed=seed)
                try:
                    saved = previous_result.metadata.get("behaviorchain", {}).get("judge_record") if previous_result else None
                    if saved is not None and saved.get("status") == "available":
                        judgment = saved
                    else:
                        judgment = generate_and_parse_with_contract_retries(
                            backend=backend,
                            request=judge_request,
                            parser=lambda response: _judge_payload(response.text),
                            responses=responses,
                            contract=str(judge_request.metadata["output_contract"]),
                        )
                    generation_metric = int(bool(judgment["aligned"]))
                    judge_record = {
                        "status": "available",
                        "aligned": judgment["aligned"],
                        "reasoning": judgment["reasoning"],
                        "provenance": provenance.to_dict(),
                    }
                except (BackendError, ParseError) as exc:
                    judge_error = {"kind": type(exc).__name__, "message": str(exc)}
            node_score = generation_metric

        provenance = self.judge_provenance_for_case(case)
        metric_metadata = {
            "task_mode": mode,
            "judge_provenance": provenance.to_dict() if provenance else None,
            "availability": "available" if node_score is not None else "unavailable",
            "reason": judge_error,
        }
        metrics = (
            MetricValue(
                "behaviorchain.node_score",
                node_score,
                unit="proportion",
                numerator=node_score,
                denominator=1 if node_score is not None else 0,
                metadata=metric_metadata,
            ),
            MetricValue(
                "behaviorchain.prediction.node_accuracy",
                prediction_metric,
                unit="proportion",
                metadata={"task_mode": mode},
            ),
            MetricValue(
                "behaviorchain.generation.node_judge_score",
                generation_metric,
                unit="proportion",
                metadata=metric_metadata,
            ),
        )
        metadata = self._base_metadata(case)
        metadata["evaluation_complete"] = True
        metadata["behaviorchain"] = {
            **metadata["behaviorchain"],
            "judge_record": judge_record,
            "judge_error": judge_error,
            "judge_provenance": provenance.to_dict() if provenance else None,
        }
        traces = [
            TraceEvent(
                turn=int(case.input_data["chain_index"]),
                actor="evaluated_model",
                kind=mode,
                content=public_prediction,
                visible_to=("evaluator",),
            )
        ]
        if judge_record is not None:
            traces.append(
                TraceEvent(
                    turn=int(case.input_data["chain_index"]),
                    actor="generation_judge",
                    kind="judgment",
                    content={"aligned": judge_record["aligned"], "reasoning": judge_record["reasoning"]},
                    visible_to=("evaluator",),
                    metadata={"evaluator_only": True},
                )
            )
        return CaseResult(
            run_id=run_id,
            benchmark_id=self.benchmark_id,
            case_id=case.case_id,
            group_id=case.group_id,
            repetition=repetition,
            status=ResultStatus.COMPLETED,
            prediction=public_prediction,
            metrics=metrics,
            trace=tuple(traces),
            model_response=response,
            latency_ms=sum(item.latency_ms or 0 for item in responses),
            token_usage=combine_usage(item.usage for item in responses),
            metadata=metadata,
        )

    @staticmethod
    def _node_score(result: CaseResult) -> tuple[float | None, bool]:
        if result.status == ResultStatus.FAILED:
            return 0.0, True
        metric = next((item for item in result.metrics if item.name == "behaviorchain.node_score"), None)
        if metric is None or metric.value is None:
            return None, False
        return float(metric.value), True

    def aggregate(self, results: Sequence[CaseResult]) -> Mapping[str, MetricValue]:
        metrics = dict(aggregate_named_metrics(results, namespace=self.benchmark_id))
        grouped: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            grouped[result.group_id].append(result)
        chain_details: dict[str, Any] = {}
        mode_chains: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        structurally_complete = 0
        score_available = 0
        for group_id, items in sorted(grouped.items()):
            infos = [item.metadata.get("behaviorchain") for item in items]
            if not all(isinstance(info, Mapping) for info in infos):
                chain_details[group_id] = {"status": "invalid", "reason": "missing_chain_metadata"}
                continue
            modes = {str(info["task_mode"]) for info in infos if isinstance(info, Mapping)}
            lengths = {int(info["chain_length"]) for info in infos if isinstance(info, Mapping)}
            indices = [int(info["chain_index"]) for info in infos if isinstance(info, Mapping)]
            structure_ok = (
                len(modes) == 1
                and len(lengths) == 1
                and len(indices) == next(iter(lengths))
                and sorted(indices) == list(range(next(iter(lengths))))
            )
            mode = next(iter(modes)) if len(modes) == 1 else "mixed"
            ordered = sorted(
                zip(items, infos),
                key=lambda pair: int(pair[1]["chain_index"]) if isinstance(pair[1], Mapping) else -1,
            )
            values: list[int] = []
            unavailable_nodes = 0
            failed_nodes = 0
            for result, _info in ordered:
                score, available = self._node_score(result)
                if not available or score is None:
                    unavailable_nodes += 1
                else:
                    values.append(int(score))
                failed_nodes += int(result.status == ResultStatus.FAILED)
            chain_available = structure_ok and unavailable_nodes == 0 and len(values) == len(items)
            avg_score = sum(values) / len(values) if chain_available and values else None
            cum_score = normalized_cumulative_chain_score(values) if chain_available and values else None
            structurally_complete += int(structure_ok)
            score_available += int(chain_available)
            details = {
                "status": "available" if chain_available else "unavailable",
                "task_mode": mode,
                "expected_length": next(iter(lengths)) if len(lengths) == 1 else None,
                "observed_indices": sorted(indices),
                "structurally_complete": structure_ok,
                "failed_node_count": failed_nodes,
                "unavailable_node_count": unavailable_nodes,
                "node_values": values,
                "avg_score": avg_score,
                "cum_score": cum_score,
            }
            chain_details[group_id] = details
            mode_chains[mode].append(details)

        metrics["behaviorchain.chain_structure_complete_rate"] = MetricValue(
            "behaviorchain.chain_structure_complete_rate",
            structurally_complete / len(grouped) if grouped else None,
            unit="proportion",
            numerator=structurally_complete,
            denominator=len(grouped),
            metadata={"groups": chain_details},
        )
        metrics["behaviorchain.chain_score_availability_rate"] = MetricValue(
            "behaviorchain.chain_score_availability_rate",
            score_available / len(grouped) if grouped else None,
            unit="proportion",
            numerator=score_available,
            denominator=len(grouped),
            metadata={"groups": chain_details},
        )
        for mode in ("prediction", "generation"):
            chains = mode_chains.get(mode, [])
            complete = [chain for chain in chains if chain["status"] == "available"]
            all_available = bool(chains) and len(complete) == len(chains)
            avg_value = (
                sum(float(chain["avg_score"]) for chain in complete) / len(complete)
                if all_available
                else None
            )
            cum_value = (
                sum(float(chain["cum_score"]) for chain in complete) / len(complete)
                if all_available
                else None
            )
            avg_name = f"behaviorchain.{mode}.avg_score"
            cum_name = f"behaviorchain.{mode}.cum_score"
            common_metadata = {
                "official": True,
                "aggregation": "unweighted_mean_of_persona_chain_scores",
                "chain_count": len(chains),
                "available_chain_count": len(complete),
                "fail_closed_if_any_chain_unavailable": True,
                "groups": {key: value for key, value in chain_details.items() if value.get("task_mode") == mode},
            }
            metrics[avg_name] = MetricValue(
                avg_name,
                avg_value,
                unit="proportion",
                numerator=sum(float(chain["avg_score"]) for chain in complete) if all_available else None,
                denominator=len(chains),
                metadata=common_metadata,
            )
            metrics[cum_name] = MetricValue(
                cum_name,
                cum_value,
                unit="proportion",
                numerator=sum(float(chain["cum_score"]) for chain in complete) if all_available else None,
                denominator=len(chains),
                metadata={**common_metadata, "formula": "sum_triangular_correct_runs_over_triangular_chain_length"},
            )
        prediction = metrics["behaviorchain.prediction.avg_score"]
        generation = metrics["behaviorchain.generation.avg_score"]
        metrics["behaviorchain.behavior_prediction_accuracy"] = MetricValue(
            "behaviorchain.behavior_prediction_accuracy",
            prediction.value,
            unit="proportion",
            numerator=prediction.numerator,
            denominator=prediction.denominator,
            metadata={**dict(prediction.metadata), "official_primary": True, "alias_of": prediction.name},
        )
        metrics["behaviorchain.behavior_generation_judge_score"] = MetricValue(
            "behaviorchain.behavior_generation_judge_score",
            generation.value,
            unit="proportion",
            numerator=generation.numerator,
            denominator=generation.denominator,
            metadata={**dict(generation.metadata), "official_primary": False, "alias_of": generation.name},
        )

        by_key: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            info = result.metadata.get("behaviorchain")
            if isinstance(info, Mapping):
                by_key[str(info.get("key_behavior_status") or "unspecified")].append(result)
        for status, items in sorted(by_key.items()):
            scores = [self._node_score(item) for item in items]
            unavailable = sum(not available or value is None for value, available in scores)
            values = [float(value) for value, available in scores if available and value is not None]
            name = f"behaviorchain.node_score.{status}"
            metrics[name] = MetricValue(
                name,
                sum(values) / len(values) if values and unavailable == 0 else None,
                unit="proportion",
                numerator=sum(values) if values and unavailable == 0 else None,
                denominator=len(items),
                metadata={
                    "score_by_key_behavior_status": True,
                    "unavailable_count": unavailable,
                    "target_failures_count_as_incorrect": True,
                },
            )
        all_scores = [self._node_score(result) for result in results]
        available_scores = [float(value) for value, available in all_scores if available and value is not None]
        metrics["behaviorchain.diagnostic.node_micro_score"] = MetricValue(
            "behaviorchain.diagnostic.node_micro_score",
            sum(available_scores) / len(available_scores) if available_scores else None,
            unit="proportion",
            numerator=sum(available_scores) if available_scores else None,
            denominator=len(available_scores),
            metadata={
                "official_primary": False,
                "diagnostic": True,
                "does_not_replace_chain_macro_scores": True,
                "unavailable_count": len(results) - len(available_scores),
            },
        )
        parse_metric = parse_failure_rate(results)
        metrics["behaviorchain.parse_failure_rate"] = MetricValue(
            "behaviorchain.parse_failure_rate",
            parse_metric.value,
            direction="lower_is_better",
            unit="proportion",
            numerator=parse_metric.numerator,
            denominator=parse_metric.denominator,
        )
        return metrics


__all__ = [
    "BehaviorChainAdapter",
    "BehaviorChainJudgeProvenance",
    "normalized_cumulative_chain_score",
]
