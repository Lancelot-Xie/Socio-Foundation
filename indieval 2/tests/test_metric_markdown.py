import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sim_eval.metric_markdown import (
    KEY_COMPONENT_METRICS,
    PRIMARY_METRICS,
    load_metric_entries,
    render_metric_markdown,
    write_metric_markdown,
)
from sim_eval.errors import ArtifactError


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class MetricMarkdownTests(unittest.TestCase):
    def test_explicit_posthoc_metrics_and_whole_entry_replacement_are_audited(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "base"
            replacement_root = Path(temporary) / "replacement"

            def write_entry(
                campaign: Path,
                entry_id: str,
                benchmark_id: str,
                run_id: str,
                metrics: dict,
            ) -> None:
                child = campaign / "entries" / entry_id
                records = child / benchmark_id / run_id / "records.jsonl"
                records.parent.mkdir(parents=True, exist_ok=True)
                records.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "case_id": f"{entry_id}-case",
                            "repetition": 0,
                            "checkpoint_attempt": 0,
                            "status": "completed",
                            "metrics": [
                                {
                                    "name": name,
                                    "value": metric.get("value"),
                                    "unit": metric.get("unit"),
                                    "direction": metric.get("direction"),
                                }
                                for name, metric in metrics.items()
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                _write_json(
                    child / "suite_summary.json",
                    {
                        "status": "completed",
                        "benchmark_id": benchmark_id,
                        "selected_case_count": 1,
                        "repetitions_per_case": 1,
                        "result_count": 1,
                        "completed_count": 1,
                        "failed_count": 0,
                        "metrics": metrics,
                        "artifacts": {"records": f"{benchmark_id}/{run_id}/records.jsonl"},
                    },
                )

            agentsense_metrics = {
                "agentsense.episode.judge_majority": {
                    "name": "agentsense.episode.judge_majority",
                    "value": 0.5,
                    "unit": "proportion",
                    "direction": "higher_is_better",
                },
                "agentsense.episode.private_information_accuracy": {
                    "name": "agentsense.episode.private_information_accuracy",
                    "value": 0.5,
                    "unit": "proportion",
                    "direction": "higher_is_better",
                },
                "agentsense.profile_sensitivity_index.goal": {
                    "name": "agentsense.profile_sensitivity_index.goal",
                    "value": 10.0,
                    "unit": "percentage_points",
                    "direction": "lower_is_better",
                },
                "agentsense.profile_sensitivity_index.information": {
                    "name": "agentsense.profile_sensitivity_index.information",
                    "value": 10.0,
                    "unit": "percentage_points",
                    "direction": "lower_is_better",
                },
            }
            old_coser_metrics = {
                "coser.scene.critic_average": {
                    "name": "coser.scene.critic_average",
                    "value": 99.0,
                    "unit": "score_0_to_100",
                    "direction": "higher_is_better",
                }
            }
            write_entry(root, "agentsense", "agentsense", "agentsense-old", agentsense_metrics)
            write_entry(root, "coser", "coser", "coser-old", old_coser_metrics)
            _write_json(
                root / "suite_summary.json",
                {
                    "status": "completed",
                    "suite_id": "base-suite",
                    "entries": [
                        {
                            "id": entry_id,
                            "benchmark_id": entry_id,
                            "status": "completed",
                            "selected_case_count": 1,
                            "child_summary": f"entries/{entry_id}/suite_summary.json",
                        }
                        for entry_id in ("agentsense", "coser")
                    ],
                },
            )
            suite_identity = {
                "model": {"model": "example-model", "model_revision": "checkpoint-1"},
                "global_eval_model": "Deepseek",
                "seed": 20260824,
            }
            _write_json(root / "suite_plan.json", suite_identity)

            corrected_dir = root / "entries" / "agentsense" / "agentsense" / "agentsense-old" / "posthoc_agentsense_metrics_v2"
            corrected_metrics = dict(agentsense_metrics)
            corrected_metrics["agentsense.profile_sensitivity_index.goal"] = {
                **corrected_metrics["agentsense.profile_sensitivity_index.goal"],
                "value": 70.710678,
            }
            _write_json(
                corrected_dir / "corrected_metrics.json",
                {
                    "source_run_id": "agentsense-old",
                    "metrics": corrected_metrics,
                },
            )
            _write_json(
                corrected_dir / "recomputation_manifest.json",
                {
                    "benchmark_id": "agentsense",
                    "source_run_id": "agentsense-old",
                },
            )

            new_coser_metrics = {
                "coser.scene.critic_average": {
                    "name": "coser.scene.critic_average",
                    "value": 42.0,
                    "unit": "score_0_to_100",
                    "direction": "higher_is_better",
                }
            }
            write_entry(replacement_root, "coser", "coser", "coser-new", new_coser_metrics)
            _write_json(
                replacement_root / "suite_summary.json",
                {
                    "status": "completed",
                    "suite_id": "replacement-suite",
                    "entries": [
                        {
                            "id": "coser",
                            "benchmark_id": "coser",
                            "status": "completed",
                            "selected_case_count": 1,
                            "child_summary": "entries/coser/suite_summary.json",
                        }
                    ],
                },
            )
            _write_json(replacement_root / "suite_plan.json", suite_identity)

            markdown = render_metric_markdown(
                input_path=root,
                corrected_metrics={"agentsense": corrected_dir},
                replacement_entries={"coser": replacement_root},
            )

            self.assertIn("## 修正与替换来源", markdown)
            self.assertIn("`agentsense` | `corrected_metrics`", markdown)
            self.assertIn("`coser` | `replace_entry`", markdown)
            self.assertIn("`agentsense.profile_sensitivity_index.goal` | 70.710678", markdown)
            self.assertIn("`coser.scene.critic_average` | 42", markdown)
            self.assertNotIn("`coser.scene.critic_average` | 99", markdown)

            _write_json(
                replacement_root / "suite_plan.json",
                {**suite_identity, "global_eval_model": "Qwen"},
            )
            with self.assertRaisesRegex(ArtifactError, "different suite identity"):
                render_metric_markdown(
                    input_path=root,
                    replacement_entries={"coser": replacement_root},
                )

    def test_requested_partial_data_scorecard_uses_node_and_item_metrics(self) -> None:
        self.assertEqual(
            PRIMARY_METRICS["behaviorchain"],
            ("behaviorchain.diagnostic.node_micro_score", "behaviorchain.prediction.cum_score"),
        )
        self.assertEqual(
            PRIMARY_METRICS["fantom"],
            ("fantom.item_correct",),
        )
        self.assertEqual(
            PRIMARY_METRICS["mirrorbench"],
            (
                "mirrorbench.judge.gteval",
                "mirrorbench.lexical.mattr.z_score_mean",
            ),
        )
        self.assertEqual(
            PRIMARY_METRICS["humanllm"],
            ("humanllm.diagnostic.top5_accuracy",),
        )
        self.assertEqual(KEY_COMPONENT_METRICS["behaviorchain"], ())
        self.assertEqual(KEY_COMPONENT_METRICS["fantom"], ())

    def test_current_campaign_reads_latest_attempts_and_renders_all_metric_fields(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "entries" / "lifechoices"
            records = child / "lifechoices" / "run-1" / "records.jsonl"
            records.parent.mkdir(parents=True)
            rows = [
                {
                    "case_id": "case-1",
                    "repetition": 0,
                    "status": "failed",
                    "checkpoint_attempt": 0,
                    "metrics": [],
                },
                {
                    "case_id": "case-1",
                    "repetition": 0,
                    "status": "completed",
                    "checkpoint_attempt": 1,
                    "metrics": [
                        {"name": "lifechoices.accuracy", "value": 1.0},
                        {"name": "sotopia.agent.persona-x.goal", "value": 8.0},
                        {
                            "name": "sotopia.agent.persona-x.goal.normalized",
                            "value": 0.8,
                            "unit": "0_to_1",
                        },
                        {"name": "sotopia.evaluated_agent.goal", "value": 8.0},
                    ],
                },
                {
                    "case_id": "case-2",
                    "repetition": 0,
                    "status": "completed",
                    "checkpoint_attempt": 0,
                    "metrics": [
                        {"name": "lifechoices.accuracy", "value": 0.0},
                        {"name": "sotopia.evaluated_agent.goal", "value": None},
                    ],
                },
            ]
            records.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            metrics = {
                "lifechoices.accuracy": {
                    "name": "lifechoices.accuracy",
                    "value": 0.5,
                    "unit": "proportion",
                    "direction": "higher_is_better",
                    "numerator": 1,
                    "denominator": 2,
                    "metadata": {},
                },
                "lifechoices.judge_score": {
                    "name": "lifechoices.judge_score",
                    "value": 0.75,
                    "unit": "score_0_to_1",
                    "direction": "higher_is_better",
                    "numerator": 0.75,
                    "denominator": 1,
                    "metadata": {},
                },
                "lifechoices.population_metric": {
                    "name": "lifechoices.population_metric",
                    "value": 2.0,
                    "unit": None,
                    "direction": "closer_to_zero",
                    "numerator": None,
                    "denominator": None,
                    "metadata": {"population_level": True},
                },
                "lifechoices.judge.availability_rate": {
                    "name": "lifechoices.judge.availability_rate",
                    "value": 0.5,
                    "unit": "proportion",
                    "direction": "higher_is_better",
                    "numerator": 1,
                    "denominator": 2,
                    "metadata": {},
                },
                "lifechoices.unavailable": {
                    "name": "lifechoices.unavailable",
                    "value": None,
                    "unit": "proportion",
                    "direction": "higher_is_better",
                    "numerator": None,
                    "denominator": 0,
                    "metadata": {"availability": "unavailable"},
                },
                "sotopia.agent.persona-x.goal": {
                    "name": "sotopia.agent.persona-x.goal",
                    "value": 8.0,
                    "unit": None,
                    "direction": "higher_is_better",
                    "numerator": 8.0,
                    "denominator": 1,
                    "metadata": {"unavailable_count": 0},
                },
                "sotopia.evaluated_agent.goal": {
                    "name": "sotopia.evaluated_agent.goal",
                    "value": 8.0,
                    "unit": None,
                    "direction": "higher_is_better",
                    "numerator": 8.0,
                    "denominator": 1,
                    "metadata": {"unavailable_count": 1},
                },
                "sotopia.agent.persona-x.goal.normalized": {
                    "name": "sotopia.agent.persona-x.goal.normalized",
                    "value": 0.8,
                    "unit": None,
                    "direction": "higher_is_better",
                    "numerator": 0.8,
                    "denominator": 1,
                    "metadata": {"unavailable_count": 0},
                },
            }
            _write_json(
                child / "suite_summary.json",
                {
                    "status": "completed",
                    "benchmark_id": "lifechoices",
                    "selected_case_count": 2,
                    "repetitions_per_case": 1,
                    "result_count": 2,
                    "completed_count": 2,
                    "failed_count": 0,
                    "metrics": metrics,
                    "artifacts": {"records": "lifechoices/run-1/records.jsonl"},
                },
            )
            _write_json(
                root / "suite_summary.json",
                {
                    "status": "completed",
                    "suite_id": "test-suite",
                    "entries": [
                        {
                            "id": "lifechoices",
                            "benchmark_id": "lifechoices",
                            "status": "completed",
                            "selected_case_count": 2,
                            "child_summary": "entries/lifechoices/suite_summary.json",
                            "records": "entries/lifechoices/lifechoices/run-1/records.jsonl",
                        }
                    ],
                },
            )

            markdown = render_metric_markdown(input_path=root)

            self.assertIn(
                "| Metric | Value | Unit | Direction | 适用范围 | 有效率（有效/适用） |",
                markdown,
            )
            self.assertIn(
                "`lifechoices.accuracy` | 0.5 | `proportion` | `↑ higher_is_better` | 2 / 2 (100.00%) | 2 / 2 (100.00%)",
                markdown,
            )
            self.assertIn(
                "`lifechoices.judge_score` | 0.75 | `score_0_to_1` | `↑ higher_is_better` | 1 aggregate units | 1 / 1 (100.00%)",
                markdown,
            )
            self.assertIn(
                "`lifechoices.population_metric` | 2 | `—` | `→0 closer_to_zero` | 2 / 2 (100.00%) | 2 / 2 (100.00%)",
                markdown,
            )
            self.assertIn(
                "`lifechoices.judge.availability_rate` | 0.5 | `proportion` | `↑ higher_is_better` | 2 / 2 (100.00%) | 2 / 2 (100.00%)",
                markdown,
            )
            self.assertIn(
                "`lifechoices.unavailable` | unavailable | `proportion` | `↑ higher_is_better` | 0 aggregate units | —",
                markdown,
            )
            self.assertIn(
                "`sotopia.agent.persona-x.goal` | 8 | `—` | `↑ higher_is_better` | 1 / 2 (50.00%) | 1 / 1 (100.00%)",
                markdown,
            )
            self.assertIn(
                "`sotopia.agent.persona-x.goal.normalized` | 0.8 | `0_to_1` | `↑ higher_is_better` | 1 / 2 (50.00%) | 1 / 1 (100.00%)",
                markdown,
            )
            self.assertIn(
                "`sotopia.evaluated_agent.goal` | 8 | `—` | `↑ higher_is_better` | 2 / 2 (100.00%) | 1 / 2 (50.00%)",
                markdown,
            )
            self.assertIn("## 五维主指标总览", markdown)
            self.assertIn(
                "| 能力维度 | Benchmark entry | 主要指标 | Value | Unit | Direction | 有效率（有效/适用） | 运行状态 |",
                markdown,
            )
            self.assertIn(
                "| `lifechoices` | `lifechoices.accuracy` | 0.5 | `proportion` | "
                "`↑ higher_is_better` | 2 / 2 (100.00%) | ✅ `completed` |",
                markdown,
            )
            self.assertIn("## 逐数据集指标明细", markdown)
            self.assertIn("## 关键组件指标", markdown)
            self.assertIn("## 运行健康度", markdown)
            self.assertIn("## 详细诊断附录", markdown)
            self.assertIn("<details>", markdown)
            self.assertNotIn("| Numerator |", markdown)
            self.assertNotIn("| Denominator |", markdown)

            output = write_metric_markdown(input_path=root)
            self.assertEqual(output, (root / "metric_summary.md").resolve())
            self.assertEqual(output.read_text(encoding="utf-8"), markdown)

    def test_legacy_suite_and_single_child_summary_are_supported(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "demo" / "run-1"
            records = run / "records.jsonl"
            records.parent.mkdir(parents=True)
            records.write_text(
                json.dumps(
                    {
                        "case_id": "c1",
                        "repetition": 0,
                        "status": "completed",
                        "metrics": [{"name": "demo.score", "value": 1}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            child = {
                "benchmark_id": "demo",
                "status": "completed",
                "selected_case_count": 1,
                "result_count": 1,
                "completed_count": 1,
                "failed_count": 0,
                "metrics": {
                    "demo.score": {
                        "name": "demo.score",
                        "value": 1,
                        "unit": "proportion",
                        "direction": "higher_is_better",
                    }
                },
                "artifact_paths": {"records": "demo/run-1/records.jsonl"},
            }
            _write_json(
                root / "suite_summary.json",
                {
                    "status": "completed",
                    "profile": "legacy",
                    "benchmarks": {"demo": child},
                },
            )
            _root, _path, _summary, entries = load_metric_entries(root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].entry_id, "demo")

            _write_json(run / "suite_summary.json", {**child, "artifacts": {"records": "records.jsonl"}})
            _root, _path, _summary, single = load_metric_entries(run / "suite_summary.json")
            self.assertEqual(len(single), 1)
            self.assertEqual(single[0].benchmark_id, "demo")

    def test_userlm_variant_placeholders_are_not_counted_as_applicable(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "userlm" / "run-1" / "records.jsonl"
            records.parent.mkdir(parents=True)
            rows = [
                {
                    "case_id": "prism",
                    "repetition": 0,
                    "status": "completed",
                    "metadata": {"variant": "intrinsic_prism"},
                    "metrics": [
                        {"name": "userlm.intrinsic.role_adherence", "value": None},
                        {"name": "userlm.intrinsic.intent_adherence", "value": None},
                        {"name": "userlm.intrinsic.ai_detector_human_likelihood", "value": 0.8},
                        {"name": "userlm.intrinsic.intent_decomposition_overlap", "value": 0.2},
                    ],
                },
                {
                    "case_id": "role",
                    "repetition": 0,
                    "status": "completed",
                    "metadata": {"variant": "intrinsic_role_adherence"},
                    "metrics": [
                        {"name": "userlm.intrinsic.role_adherence", "value": 0.9},
                        {"name": "userlm.intrinsic.intent_adherence", "value": None},
                        {
                            "name": "userlm.intrinsic.ai_detector_human_likelihood",
                            "value": 0,
                            "metadata": {"target_output_failure": {"kind": "invalid_output"}},
                        },
                    ],
                },
                {
                    "case_id": "intent",
                    "repetition": 0,
                    "status": "completed",
                    "metadata": {"variant": "intrinsic_intent_adherence"},
                    "metrics": [
                        {"name": "userlm.intrinsic.role_adherence", "value": None},
                        {"name": "userlm.intrinsic.intent_adherence", "value": 0.7},
                        {"name": "userlm.intrinsic.ai_detector_human_likelihood", "value": None},
                    ],
                },
            ]
            records.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            metrics = {
                name: {
                    "name": name,
                    "value": value,
                    "direction": "lower_is_better" if "decomposition" in name else "higher_is_better",
                    "metadata": metadata,
                }
                for name, value, metadata in (
                    ("userlm.intrinsic.role_adherence", 0.9, {}),
                    ("userlm.intrinsic.intent_adherence", 0.7, {}),
                    ("userlm.intrinsic.ai_detector_human_likelihood", 0.8, {}),
                    ("userlm.intrinsic.first_turn_diversity", 0.5, {}),
                    ("userlm.intrinsic.intent_decomposition_overlap", 0.2, {}),
                    (
                        "userlm.intrinsic.termination_f1",
                        0,
                        {"tp": 0, "fp": 0, "fn": 1, "denominator": 1},
                    ),
                )
            }
            _write_json(
                root / "suite_summary.json",
                {
                    "benchmark_id": "userlm",
                    "status": "completed",
                    "selected_case_count": 3,
                    "result_count": 3,
                    "completed_count": 3,
                    "failed_count": 0,
                    "metrics": metrics,
                    "artifacts": {"records": "userlm/run-1/records.jsonl"},
                },
            )

            markdown = render_metric_markdown(input_path=root)

            self.assertIn(
                "`userlm.intrinsic.role_adherence` | 0.9 | `—` | `↑ higher_is_better` | 1 / 3 (33.33%) | 1 / 1 (100.00%)",
                markdown,
            )
            self.assertIn(
                "`userlm.intrinsic.ai_detector_human_likelihood` | 0.8 | `—` | `↑ higher_is_better` | 1 / 3 (33.33%) | 1 / 1 (100.00%)",
                markdown,
            )
            self.assertIn(
                "`userlm.intrinsic.termination_f1` | 0 | `—` | `↑ higher_is_better` | 1 aggregate units | 1 / 1 (100.00%)",
                markdown,
            )
            self.assertIn("TP=0、FP=0、FN=1，适用 population=1", markdown)


if __name__ == "__main__":
    unittest.main()
