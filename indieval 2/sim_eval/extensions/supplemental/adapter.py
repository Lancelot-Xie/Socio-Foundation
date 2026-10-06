"""Protocol adapters for the independent Supplemental supplementary suite."""
from __future__ import annotations

import asyncio
import copy
import importlib
import math
from types import SimpleNamespace
from collections.abc import Mapping

from ...contracts import CaseResult, ErrorState, MetricValue, ResultStatus
from ...errors import BackendError, EpisodeTokenBudgetExhausted, OptionalDependencyError, ValidationError
from ...interfaces import BenchmarkAdapter
from ...benchmarks.common import aggregate_named_metrics, combine_usage
from . import BENCHMARK_IDS, EVALUATED_ROLES, REVISION
from .bridge import ACTIVE, Session


PUBLIC_FIELDS = {
    "hitom": ("story", "question", "choices", "extra_info"),
    "paratomi": ("story", "question", "cands", "extra_info"),
    "mistakes": ("QuestionText", "AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText", "MisconceptionName"),
    "twinvoice": ("conversation_history", "anchor_post", "answer_choices"),
    "socsci210": ("prompt_text",),
    "sim_math": ("problem", "user_profile_text"),
    "sim_doc": ("document_type", "intent", "user_profile_text", "pre_writing_materials_text"),
}
GOLD_FIELDS = {
    "hitom": ("correct_answer",), "paratomi": ("correct_answer",),
    "mistakes": ("TargetOption",), "twinvoice": ("answer_idx",),
    "socsci210": ("response", "response_type", "r_min", "r_max"),
    "sim_math": ("human_user_queries", "human_public_conversation", "target_interaction_style_features"),
    "sim_doc": ("human_user_queries", "human_public_conversation", "target_writing_style_features",
                "target_interaction_style_features", "target_document_preferences"),
}
OPTIONAL_FIELDS = {"extra_info", "pre_writing_materials_text"}


def validate_case(case):
    if case.benchmark_id not in BENCHMARK_IDS:
        raise ValidationError(f"unknown Supplemental benchmark: {case.benchmark_id}")
    task = case.benchmark_id.removeprefix("supplemental_")
    row = case.input_data.get("row")
    if not isinstance(row, Mapping) or not isinstance(case.gold, Mapping):
        raise ValidationError("Supplemental cases require input.row and evaluator-only gold objects")
    unknown = set(row) - set(PUBLIC_FIELDS[task])
    if unknown:
        raise ValidationError(f"unexpected/private fields in model input: {sorted(unknown)}")
    for fields, values in ((PUBLIC_FIELDS[task], row), (GOLD_FIELDS[task], case.gold)):
        missing = set(fields) - OPTIONAL_FIELDS - set(values)
        if missing:
            raise ValidationError(f"{case.benchmark_id} missing fields: {sorted(missing)}")
    if not isinstance(case.metadata.get("strata"), Mapping):
        raise ValidationError("Supplemental cases require metadata.strata")
    if task == "twinvoice":
        if not isinstance(row["answer_choices"], list) or len(row["answer_choices"]) != 4:
            raise ValidationError("TwinVoice requires four answer choices")
        if type(case.gold["answer_idx"]) is not int or not 0 <= case.gold["answer_idx"] < 4:
            raise ValidationError("TwinVoice answer_idx must be in 0..3")
    if task == "mistakes" and case.gold["TargetOption"] not in {"A", "B", "C", "D"}:
        raise ValidationError("Mistakes requires TargetOption A..D")
    if task == "socsci210":
        if case.gold["response_type"] not in {"ordinal", "binary", "categorical"}:
            raise ValidationError("unsupported SocSci210 response type")
        if not all(math.isfinite(float(case.gold[k])) for k in ("response", "r_min", "r_max")):
            raise ValidationError("SocSci210 scores and bounds must be finite")


class _Captured(Exception):
    def __init__(self, request):
        self.request = request


class SupplementalAdapter(BenchmarkAdapter):
    def __init__(self, benchmark_id, *, config=None):
        if benchmark_id not in BENCHMARK_IDS:
            raise ValidationError(f"unknown Supplemental benchmark: {benchmark_id}")
        self.benchmark_id = benchmark_id
        self.task = benchmark_id.removeprefix("supplemental_")
        self.interactive = self.task in {"sim_doc", "sim_math"}
        self.prompt_revision = f"{REVISION}:{self.task}:prompt"
        self.scorer_revision = f"{REVISION}:{self.task}:reward"
        self.config = copy.deepcopy(config or {})
        try:
            self.module = importlib.import_module(f"{__package__}.vendor.{self.task}")
        except ModuleNotFoundError as exc:
            if exc.name == "pydantic":
                raise OptionalDependencyError("Supplemental interactive adapters require pydantic>=2; install .[supplemental]") from exc
            raise

    def validate_case(self, case):
        if case.benchmark_id != self.benchmark_id:
            raise ValidationError("adapter and case benchmark IDs differ")
        validate_case(case)

    def _run(self, case, session):
        self.validate_case(case)
        context = SimpleNamespace(llm_client=session.backend, tokenizer=None, config=None)
        row = {**copy.deepcopy(dict(case.input_data["row"])), **copy.deepcopy(dict(case.gold))}
        token = ACTIVE.set(session)
        try:
            return asyncio.run(self.module.agent_loop({"extra_info": row}, context))
        finally:
            ACTIVE.reset(token)

    def _session(self, case, backend, model, seed):
        return Session(backend, case, model, seed, EVALUATED_ROLES[self.benchmark_id],
                       roles=copy.deepcopy(self.config.get("roles", {})))

    def build_request(self, case, *, model, seed):
        class Capture:
            def generate(self, request):
                raise _Captured(request)
        try:
            self._run(case, self._session(case, Capture(), model, seed))
        except _Captured as captured:
            return captured.request
        raise ValidationError("protocol did not construct a candidate request")

    def parse_response(self, case, response):
        if self.interactive:
            return {"raw_response": response.text}
        class Fixed:
            def generate(self, request):
                return response
        output = self._run(case, self._session(case, Fixed(), "replay", 0))
        return {"reward": float(output.reward_score), "raw_response": response.text}

    def execute_case(self, case, *, backend, run_id, seed, model, repetition=0):
        session = self._session(case, backend, model, seed)
        try:
            output = self._run(case, session)
        except (BackendError, EpisodeTokenBudgetExhausted) as exc:
            return CaseResult(
                run_id, self.benchmark_id, case.case_id, case.group_id, repetition,
                ResultStatus.FAILED, trace=tuple(session.trace),
                error=ErrorState("execution", type(exc).__name__, str(exc), retryable=isinstance(exc, BackendError)),
                token_usage=combine_usage(r.usage for r in session.responses),
                metadata={"evaluation_complete": False, "protocol_revision": REVISION},
            )
        reward = float(output.reward_score)
        source_metrics = output.extra_fields.get("reward_extra_info", {})
        metrics = [MetricValue(f"{self.benchmark_id}.reward", reward, unit="proportion", numerator=reward, denominator=1)]
        for key, value in source_metrics.items():
            suffix = key.rsplit("/", 1)[-1]
            if key.startswith("all/") or suffix == "reward":
                continue
            metrics.append(MetricValue(f"{self.benchmark_id}.{suffix}", value))
        strata = case.metadata["strata"]
        from ...benchmarks.choice import safe_metric_token
        for field in ("deception", "story_length", "question_order", "question_type", "dimension", "target_option"):
            if field in strata:
                metrics.append(MetricValue(f"{self.benchmark_id}.by_{field}.{safe_metric_token(str(strata[field]))}",
                                           reward, numerator=reward, denominator=1, unit="proportion"))
        return CaseResult(
            run_id, self.benchmark_id, case.case_id, case.group_id, repetition, ResultStatus.COMPLETED,
            prediction={"reward": reward}, metrics=tuple(metrics), trace=tuple(session.trace),
            model_response=session.last_candidate_response,
            token_usage=combine_usage(r.usage for r in session.responses),
            metadata={"evaluation_complete": True, "protocol_revision": REVISION,
                      "protocol_label": "supplemental_compat_supplementary_not_source_author_canonical",
                      "source_metrics": source_metrics,
                      "rejudge_artifact": output.extra_fields.get("rejudge_artifact"),
                      "judge_failures": session.support_failures,
                      "missing_judge_policy": "original_Supplemental_fixed_denominator_zero_contribution",
                      "strata": dict(strata)},
        )

    def aggregate(self, results):
        metrics = dict(aggregate_named_metrics(results, namespace=self.benchmark_id))
        name = f"{self.benchmark_id}.reward"
        total = sum(float(m.value or 0) for r in results for m in r.metrics if m.name == name)
        metrics[name] = MetricValue(name, total / len(results) if results else None, numerator=total,
                                   denominator=len(results), unit="proportion",
                                   metadata={"aggregation": "row_mean", "failed_case_contribution": 0})
        failed_judges = sum(bool(r.metadata.get("judge_failures")) for r in results)
        metrics[f"{self.benchmark_id}.cases_with_judge_failures"] = MetricValue(
            f"{self.benchmark_id}.cases_with_judge_failures", failed_judges, direction="lower_is_better", unit="count")
        return metrics
