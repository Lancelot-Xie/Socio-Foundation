import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sim_eval.errors import ArtifactError
from sim_eval.metric_comparison import (
    discover_latest_candidate_paths,
    render_metric_comparison,
    render_three_way_metric_comparison,
    write_metric_comparison,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_run(
    root: Path,
    *,
    entry_id: str,
    metrics: dict,
    case_ids: tuple[str, ...] = ("c1", "c2"),
) -> None:
    child = root / "entries" / entry_id
    records = child / entry_id / "run-1" / "records.jsonl"
    records.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "case_id": case_id,
            "repetition": 0,
            "status": "completed",
            "metrics": [
                {"name": name, "value": metric["value"]}
                for name, metric in metrics.items()
            ],
        }
        for case_id in case_ids
    ]
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_json(
        child / "suite_summary.json",
        {
            "benchmark_id": entry_id,
            "status": "completed",
            "selected_case_count": len(case_ids),
            "result_count": len(case_ids),
            "completed_count": len(case_ids),
            "failed_count": 0,
            "metrics": metrics,
            "artifacts": {"records": f"{entry_id}/run-1/records.jsonl"},
        },
    )
    _write_json(
        root / "suite_summary.json",
        {
            "status": "completed",
            "suite_id": root.name,
            "entries": [
                {
                    "id": entry_id,
                    "benchmark_id": entry_id,
                    "status": "completed",
                    "child_summary": f"entries/{entry_id}/suite_summary.json",
                    "records": f"entries/{entry_id}/{entry_id}/run-1/records.jsonl",
                }
            ],
        },
    )


def _write_alignx_run(root: Path, *, variant_values: dict[str, tuple[int, ...]]) -> None:
    entry_id = "alignx"
    child = root / "entries" / entry_id
    records = child / entry_id / "run-1" / "records.jsonl"
    records.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, values in variant_values.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "case_id": f"{variant}-{index}",
                    "repetition": 0,
                    "status": "completed",
                    "metadata": {"alignx": {"variant": variant}},
                    "metrics": [
                        {"name": "alignx.direct_choice_accuracy", "value": value},
                        {"name": "alignx.alignment_accuracy", "value": None},
                    ],
                }
            )
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    direct = sum(sum(values) for values in variant_values.values()) / len(rows)
    metrics = {
        "alignx.direct_choice_accuracy": {
            "name": "alignx.direct_choice_accuracy",
            "value": direct,
            "unit": "proportion",
            "direction": "higher_is_better",
            "denominator": len(rows),
        },
        "alignx.alignment_accuracy": {
            "name": "alignx.alignment_accuracy",
            "value": None,
            "unit": "proportion",
            "direction": "higher_is_better",
            "denominator": 0,
        },
        "alignx.alignment_accuracy.variant.reddit_demo": {
            "name": "alignx.alignment_accuracy.variant.reddit_demo",
            "value": None,
            "unit": "proportion",
            "direction": "higher_is_better",
            "denominator": 0,
        },
        "alignx.alignment_score_availability_rate": {
            "name": "alignx.alignment_score_availability_rate",
            "value": None,
            "unit": "proportion",
            "direction": "higher_is_better",
            "denominator": 0,
        },
    }
    _write_json(
        child / "suite_summary.json",
        {
            "benchmark_id": "alignx",
            "status": "completed",
            "selected_case_count": len(rows),
            "result_count": len(rows),
            "completed_count": len(rows),
            "failed_count": 0,
            "metrics": metrics,
            "artifacts": {"records": "alignx/run-1/records.jsonl"},
        },
    )
    _write_json(
        root / "suite_summary.json",
        {
            "status": "completed",
            "suite_id": root.name,
            "entries": [
                {
                    "id": "alignx",
                    "benchmark_id": "alignx",
                    "status": "completed",
                    "child_summary": "entries/alignx/suite_summary.json",
                    "records": "entries/alignx/alignx/run-1/records.jsonl",
                }
            ],
        },
    )


def _write_humanual_run(
    root: Path,
    *,
    entry_id: str,
    domain_values: dict[str, tuple[float, ...]],
) -> None:
    child = root / "entries" / entry_id
    records = child / "humanual" / "run-1" / "records.jsonl"
    records.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain, values in domain_values.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "case_id": f"{domain}-{index}",
                    "repetition": 0,
                    "status": "completed",
                    "metadata": {"domain": domain},
                    "metrics": [
                        {
                            "name": "humanual.response_alignment",
                            "value": value,
                            "unit": "proportion",
                            "direction": "higher_is_better",
                        },
                        {
                            "name": f"humanual.domain.{domain}.response_alignment",
                            "value": value,
                            "unit": "proportion",
                            "direction": "higher_is_better",
                        },
                    ],
                }
            )
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    all_values = [value for values in domain_values.values() for value in values]
    metrics = {
        "humanual.response_alignment": {
            "name": "humanual.response_alignment",
            "value": sum(all_values) / len(all_values),
            "unit": "proportion",
            "direction": "higher_is_better",
            "denominator": len(all_values),
        }
    }
    for domain, values in domain_values.items():
        name = f"humanual.domain.{domain}.response_alignment"
        metrics[name] = {
            "name": name,
            "value": sum(values) / len(values),
            "unit": "proportion",
            "direction": "higher_is_better",
            "denominator": len(values),
        }
    _write_json(
        child / "suite_summary.json",
        {
            "benchmark_id": "humanual",
            "status": "completed",
            "selected_case_count": len(rows),
            "result_count": len(rows),
            "completed_count": len(rows),
            "failed_count": 0,
            "metrics": metrics,
            "artifacts": {"records": "humanual/run-1/records.jsonl"},
        },
    )
    _write_json(
        root / "suite_summary.json",
        {
            "status": "completed",
            "suite_id": root.name,
            "entries": [
                {
                    "id": entry_id,
                    "benchmark_id": "humanual",
                    "status": "completed",
                    "child_summary": f"entries/{entry_id}/suite_summary.json",
                    "records": f"entries/{entry_id}/humanual/run-1/records.jsonl",
                }
            ],
        },
    )


class MetricComparisonTests(unittest.TestCase):
    def test_three_way_comparison_uses_entry_union_and_marks_missing_whole_entry(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            third = root / "third_model"
            agentsense_metric = {
                "agentsense.episode.judge_majority": {
                    "name": "agentsense.episode.judge_majority",
                    "value": 0.4,
                    "unit": "proportion",
                    "direction": "higher_is_better",
                    "denominator": 2,
                }
            }
            _write_run(
                baseline,
                entry_id="agentsense",
                metrics=agentsense_metric,
            )
            _write_run(
                candidate,
                entry_id="lifechoices",
                metrics={
                    "lifechoices.accuracy": {
                        "name": "lifechoices.accuracy",
                        "value": 0.75,
                        "unit": "proportion",
                        "direction": "higher_is_better",
                        "denominator": 2,
                    }
                },
            )
            _write_run(
                third,
                entry_id="agentsense",
                metrics={
                    "agentsense.episode.judge_majority": {
                        **agentsense_metric["agentsense.episode.judge_majority"],
                        "value": 0.6,
                    }
                },
            )

            markdown = render_three_way_metric_comparison(
                baseline_path=baseline,
                candidate_paths=[candidate],
                third_path=third,
                baseline_label="baseline",
                candidate_label="Candidate",
                third_label="Third model",
            )

            self.assertIn("Candidate 缺失 entry：`agentsense`", markdown)
            self.assertIn("baseline 缺失 entry：`lifechoices`", markdown)
            self.assertIn(
                "| `agentsense` | **`agentsense.episode.judge_majority`** | "
                "0.4 | \\ | — | 0.6 | +0.2",
                markdown,
            )
            self.assertIn(
                "| `lifechoices` | **`lifechoices.accuracy`** | "
                "\\ | 0.75 | — | \\ | —",
                markdown,
            )
            self.assertIn(
                "`\\` 表示该模型没有评测整个 entry；`missing` 表示 entry 存在",
                markdown,
            )
            self.assertIn("| Third model | 2/2 (100.00%) | \\ | 2/2 (100.00%)", markdown)

    def test_three_way_entry_overrides_replace_only_requested_population(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline-full"
            candidate = root / "candidate-auto"
            third = root / "third_model-full"
            baseline_override = root / "behavior-v2-baseline"
            candidate_override = root / "behavior-v2-candidate"
            metric_name = "behaviorchain.diagnostic.node_micro_score"

            for path, value, case_ids in (
                (baseline, 0.1, ("old-1", "old-2")),
                (candidate, 0.2, ("old-1", "old-2")),
                (third, 0.6, ("v2-1", "v2-2")),
                (baseline_override, 0.4, ("v2-1", "v2-2")),
                (candidate_override, 0.7, ("v2-1", "v2-2")),
            ):
                _write_run(
                    path,
                    entry_id="behaviorchain",
                    metrics={
                        metric_name: {
                            "name": metric_name,
                            "value": value,
                            "unit": "proportion",
                            "direction": "higher_is_better",
                            "denominator": 2,
                        }
                    },
                    case_ids=case_ids,
                )

            markdown = render_three_way_metric_comparison(
                baseline_path=baseline,
                candidate_paths=[candidate],
                third_path=third,
                baseline_label="baseline",
                candidate_label="Candidate",
                third_label="Third model",
                baseline_entry_overrides={"behaviorchain": baseline_override},
                candidate_entry_overrides={"behaviorchain": candidate_override},
            )

            self.assertIn("Baseline entry override `behaviorchain`", markdown)
            self.assertIn("Candidate entry override `behaviorchain`", markdown)
            self.assertIn("✅ exact record keys (2)", markdown)
            self.assertIn(
                "| `behaviorchain` | **`behaviorchain.diagnostic.node_micro_score`** | "
                "0.4 | 0.7 | +0.3 | 0.6 | +0.2",
                markdown,
            )

    def test_three_way_comparison_renders_primary_values_deltas_and_winner(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            third = root / "unified"
            for path, value in ((baseline, 0.5), (candidate, 0.75), (third, 0.6)):
                _write_run(
                    path,
                    entry_id="lifechoices",
                    metrics={
                        "lifechoices.accuracy": {
                            "name": "lifechoices.accuracy",
                            "value": value,
                            "unit": "proportion",
                            "direction": "higher_is_better",
                            "denominator": 2,
                        }
                    },
                )

            markdown = render_three_way_metric_comparison(
                baseline_path=baseline,
                candidate_paths=[candidate],
                third_path=third,
                baseline_label="baseline",
                candidate_label="Candidate",
                third_label="Unified",
            )

            self.assertIn("评测指标三方对比：baseline vs Candidate vs Unified", markdown)
            self.assertIn("| Entry | Metric | baseline | Candidate | Δ Candidate | Unified | Δ Unified |", markdown)
            self.assertIn(
                "| `lifechoices` | **`lifechoices.accuracy`** | 0.5 | 0.75 | +0.25 | 0.6 | +0.1",
                markdown,
            )
            self.assertIn("Unified 有效/适用", markdown)
            self.assertIn("Candidate | 2/2 (100.00%) | 2/2 (100.00%) | 2/2 (100.00%)", markdown)
            self.assertIn("✅ 1 组三方比较", markdown)

    def test_three_way_comparison_slices_full_humanual_runs_for_domain_candidate(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            third = root / "unified"
            _write_humanual_run(
                baseline,
                entry_id="humanual",
                domain_values={"book": (0.8, 0.6), "chat": (0.2, 0.2)},
            )
            _write_humanual_run(
                candidate,
                entry_id="humanual_book",
                domain_values={"book": (0.9, 0.9)},
            )
            _write_humanual_run(
                third,
                entry_id="humanual",
                domain_values={"book": (0.5, 0.7), "chat": (0.4, 0.4)},
            )

            markdown = render_three_way_metric_comparison(
                baseline_path=baseline,
                candidate_paths=[candidate],
                third_path=third,
                baseline_label="baseline",
                candidate_label="Candidate",
                third_label="Unified",
            )

            self.assertIn("| `humanual_book` |", markdown)
            self.assertIn(
                "| `humanual_book` | **`humanual.response_alignment`** | 0.7 | 0.9 | +0.2 | 0.6 | -0.1",
                markdown,
            )
            self.assertGreaterEqual(markdown.count("exact record keys (2)"), 2)

    def test_exact_entry_comparison_renders_primary_delta_winner_and_coverage(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_run(
                baseline,
                entry_id="lifechoices",
                metrics={
                    "lifechoices.accuracy": {
                        "name": "lifechoices.accuracy",
                        "value": 0.5,
                        "unit": "proportion",
                        "direction": "higher_is_better",
                        "denominator": 2,
                    },
                    "lifechoices.parse_failure_rate": {
                        "name": "lifechoices.parse_failure_rate",
                        "value": 0.1,
                        "unit": "proportion",
                        "direction": "lower_is_better",
                        "denominator": 2,
                    },
                },
            )
            _write_run(
                candidate,
                entry_id="lifechoices",
                metrics={
                    "lifechoices.accuracy": {
                        "name": "lifechoices.accuracy",
                        "value": 0.75,
                        "unit": "proportion",
                        "direction": "higher_is_better",
                        "denominator": 2,
                    },
                    "lifechoices.parse_failure_rate": {
                        "name": "lifechoices.parse_failure_rate",
                        "value": 0.05,
                        "unit": "proportion",
                        "direction": "lower_is_better",
                        "denominator": 2,
                    },
                },
            )

            markdown = render_metric_comparison(
                baseline_path=baseline,
                candidate_paths=[candidate],
                baseline_label="baseline",
                candidate_label="Candidate",
            )

            self.assertIn("exact record keys (2)", markdown)
            self.assertIn(
                "| Entry | Metric | Baseline | Candidate | Delta | Unit | Direction |",
                markdown,
            )
            self.assertIn("| `lifechoices` | **`lifechoices.accuracy`** |", markdown)
            self.assertIn("**`lifechoices.accuracy`**", markdown)
            self.assertIn("0.5 | 0.75 | +0.25", markdown)
            self.assertIn("Candidate | 2/2 (100.00%) | 2/2 (100.00%)", markdown)
            self.assertIn("0.1 | 0.05 | -0.05", markdown)
            self.assertIn("✅ 1 组比较的 entry、benchmark", markdown)

            output = write_metric_comparison(
                baseline_path=baseline,
                candidate_paths=[candidate],
                output_path=root / "comparison.md",
            )
            self.assertTrue(output.is_file())
            self.assertIn("各 Benchmark 全部指标", output.read_text(encoding="utf-8"))

    def test_population_mismatch_is_reported(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            metric = {
                "lifechoices.accuracy": {
                    "name": "lifechoices.accuracy",
                    "value": 0.5,
                    "direction": "higher_is_better",
                }
            }
            _write_run(baseline, entry_id="lifechoices", metrics=metric)
            _write_run(
                candidate,
                entry_id="lifechoices",
                metrics=metric,
                case_ids=("c1", "c3"),
            )

            markdown = render_metric_comparison(
                baseline_path=baseline,
                candidate_paths=[candidate],
            )
            self.assertIn("record-key mismatch", markdown)
            self.assertIn("不能直接视为严格配对实验", markdown)

    def test_alignx_direct_choice_variants_are_derived_and_reference_margin_is_hidden(self) -> None:
        variants = {
            "reddit_arbitrary": (1, 0),
            "reddit_demo": (1, 0),
            "reddit_history16": (1, 0),
            "reddit_pair": (1, 0),
            "reddit_ugc": (1, 0),
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_alignx_run(baseline, variant_values=variants)
            _write_alignx_run(
                candidate,
                variant_values={name: (1, 1) for name in variants},
            )

            markdown = render_metric_comparison(
                baseline_path=baseline,
                candidate_paths=[candidate],
            )

            for variant in variants:
                self.assertIn(
                    f"alignx.direct_choice_accuracy.variant.{variant}",
                    markdown,
                )
            self.assertIn("0.5 | 1 | +0.5", markdown)
            self.assertNotIn("| `alignx.alignment_accuracy` |", markdown)
            self.assertNotIn("alignx.alignment_accuracy.variant", markdown)
            self.assertNotIn("alignx.alignment_score_availability_rate", markdown)

    def test_humanual_single_domain_candidate_uses_matching_baseline_domain(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_humanual_run(
                baseline,
                entry_id="humanual",
                domain_values={"book": (0.8, 0.6), "chat": (0.2, 0.2)},
            )
            _write_humanual_run(
                candidate,
                entry_id="humanual_book",
                domain_values={"book": (0.9, 0.9)},
            )

            markdown = render_metric_comparison(
                baseline_path=baseline,
                candidate_paths=[candidate],
                baseline_label="baseline",
                candidate_label="Candidate",
            )

            self.assertIn("| `humanual_book` |", markdown)
            self.assertIn("exact record keys (2)", markdown)
            self.assertIn(
                "| `humanual_book` | **`humanual.response_alignment`** | 0.7 | 0.9 | +0.2",
                markdown,
            )

    def test_candidate_root_discovery_uses_date_then_training_step(self) -> None:
        metric = {
            "alignx.direct_choice_accuracy": {
                "name": "alignx.direct_choice_accuracy",
                "value": 0.5,
                "direction": "higher_is_better",
            }
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            step150 = root / "candidate_8b_alignx_dapo_step150__eval-example-support-model__2026-08-26"
            step350 = root / "candidate_8b_alignx_dapo_step350__eval-example-support-model__2026-08-26"
            _write_run(step150, entry_id="alignx", metrics=metric)
            _write_run(step350, entry_id="alignx", metrics=metric)

            selected = discover_latest_candidate_paths(root)
            self.assertEqual(selected, (step350.resolve(),))

            summary_path = step350 / "suite_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["entries"][0]["status"] = "failed"
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(ArtifactError, "newest candidate.*not usable"):
                discover_latest_candidate_paths(root)


if __name__ == "__main__":
    unittest.main()
