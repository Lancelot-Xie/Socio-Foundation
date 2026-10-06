import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sim_eval.contracts import BenchmarkCase, ModelResponse
from sim_eval.coser_rejudge import (
    _OFFICIAL_CRITIC_TEMPLATE,
    _source_manifest_compatibility,
    build_official_coser_judge_request,
    parse_official_coser_judge_response,
    run_coser_rejudge,
    run_coser_rejudge_pilot,
    select_coser_pilot_cases,
    summarize_coser_rejudge_rows,
)
from sim_eval.data.loaders import load_fixture_suite, load_import_spec, load_local_cases
from sim_eval.errors import ParseError
from sim_eval.json_utils import jsonable


ROOT = Path(__file__).resolve().parents[1]


def _selection_case(case_id: str, stratum: str) -> BenchmarkCase:
    return BenchmarkCase(
        benchmark_id="coser",
        case_id=case_id,
        group_id=f"group-{case_id}",
        split="test",
        source_revision="revision",
        input_data={},
        metadata={"strata": {"in_domain_status": stratum}},
    )


class CoserRejudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = load_fixture_suite(ROOT / "tests" / "fixtures")["coser"][0][0]

    def test_embedded_template_is_frozen_upstream_text(self) -> None:
        self.assertEqual(
            hashlib.sha256(_OFFICIAL_CRITIC_TEMPLATE.encode("utf-8")).hexdigest(),
            "9a9bf1d745bd2e2ecc07b92394c566b6657e6708d9687d6b8ea342b26f60ab92",
        )

    def test_legacy_rollout_manifest_accepts_only_audited_context_upgrade(self) -> None:
        source_hash = "9b0abc0e43805a447bc7f6e1ac8cee7a7c7fc7c89a8013df5d757a794c8b0a2b"
        frozen = {
            "benchmark_id": "coser",
            "source_revision": "7cc80430f92532cda85df45015a4aca8ecc068d0",
            "source_kind": "official",
            "split": "test",
            "resolved_population": 200,
            "file_hashes": {
                "records.jsonl": "44e113a5f79dd6e61b5955c03781be64f96659a0a15c773d47903d089c36b722"
            },
            "metadata": {"source_sha256": source_hash},
        }
        loaded = SimpleNamespace(
            digest="different-after-audited-upgrade",
            benchmark_id="coser",
            source_revision="7cc80430f92532cda85df45015a4aca8ecc068d0",
            source_kind="official",
            split="test",
            resolved_population=200,
            file_hashes={
                "records.jsonl": "418363ad52bed9d8b0a6ba3b25c66bd3814625ab4db5da1961557c7c3a63efb5"
            },
            metadata={"source_sha256": source_hash},
        )
        self.assertEqual(
            _source_manifest_compatibility(frozen, loaded),
            "audited_legacy_rollout_to_gca_v2_context_upgrade",
        )

    def test_selection_is_deterministic_and_balanced_for_equal_strata(self) -> None:
        cases = tuple(
            [_selection_case(f"id-{index}", "id") for index in range(5)]
            + [_selection_case(f"ood-{index}", "ood") for index in range(5)]
        )
        first = select_coser_pilot_cases(
            cases,
            [case.case_id for case in cases],
            limit=4,
            seed=7,
        )
        second = select_coser_pilot_cases(
            tuple(reversed(cases)),
            [case.case_id for case in reversed(cases)],
            limit=4,
            seed=7,
        )
        self.assertEqual([case.case_id for case in first], [case.case_id for case in second])
        self.assertEqual(
            {label: sum(case.metadata["strata"]["in_domain_status"] == label for case in first) for label in ("id", "ood")},
            {"id": 2, "ood": 2},
        )

    def test_request_uses_official_context_without_goal_or_full_plot(self) -> None:
        record = {
            "prediction": {
                "public_dialogue": [
                    {
                        "speaker": "mara",
                        "content": "[private generated thought] We can begin. (opens the clock)",
                    },
                    {"speaker": "environment", "content": "Rain taps against the pier."},
                ]
            }
        }
        request = build_official_coser_judge_request(
            self.case,
            record,
            dimension="character_fidelity",
            judge_role={
                "model": "gpt-4o",
                "generation": {"temperature": 0.0, "max_tokens": 4096},
            },
            seed=11,
            max_context_tokens=20000,
        )
        self.assertEqual([message.role for message in request.messages], ["system", "user"])
        system, dialogue = (message.content for message in request.messages)
        self.assertTrue(system.startswith("You are a literary critic specializing in character analysis"))
        self.assertIn("Three harbor workers must repair a tide clock", system)
        self.assertIn("Rain has dampened the clock housing", system)
        self.assertIn("A patient apprentice clockmaker", system)
        self.assertIn("The escapement is wet", system)
        self.assertNotIn("Restore the tide clock before the ferry bell", system)
        self.assertNotIn("Mara secretly carries", system)
        self.assertNotIn("co-plot-1", system)
        self.assertIn("mara: We can begin. (opens the clock)", dialogue)
        self.assertNotIn("private generated thought", dialogue)
        self.assertFalse(request.metadata["explicit_character_goals_included"])
        self.assertFalse(request.metadata["full_structured_plot_included"])
        self.assertEqual(request.response_format["type"], "json_schema")
        self.assertEqual(request.temperature, 0.0)
        self.assertEqual(request.max_tokens, 4096)

    def test_response_parser_rejects_non_object_flaw_with_parse_error(self) -> None:
        with self.assertRaises(ParseError):
            parse_official_coser_judge_response(
                ModelResponse(
                    json.dumps({"Storyline Quality": {"flaws": ["bad"]}})
                ),
                dimension="storyline_quality",
            )

    def test_summary_compares_paired_old_and_new_scores(self) -> None:
        rows = []
        dimensions = (
            "storyline_consistency",
            "anthropomorphism",
            "character_fidelity",
            "storyline_quality",
        )
        for case_index, case_id in enumerate(("a", "b")):
            for dimension_index, dimension in enumerate(dimensions):
                old = 90.0 + case_index + dimension_index
                rows.append(
                    {
                        "case_id": case_id,
                        "dimension": dimension,
                        "attempt": 0,
                        "status": "completed",
                        "old_score": old,
                        "new_score": old - 5.0,
                        "flaw_count": 1,
                    }
                )
        summary = summarize_coser_rejudge_rows(rows, selected_case_ids=("a", "b"))
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["completed_judge_unit_count"], 8)
        self.assertEqual(summary["critic_average"]["paired_case_count"], 2)
        self.assertEqual(summary["critic_average"]["paired_delta_mean"], -5.0)

    def test_validate_only_builds_every_request_without_network_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            import_manifest = root / "import_manifest.json"
            import_manifest.write_text(
                json.dumps(
                    {
                        "benchmark_id": "coser",
                        "path": str(ROOT / "tests" / "fixtures" / "coser.jsonl"),
                        "format": "jsonl",
                        "source_kind": "synthetic_fixture",
                        "source_revision": "indieval-synthetic-v1",
                        "split": "fixture",
                        "license_acknowledged": True,
                        "license": "CC0-1.0 repository-owned synthetic text",
                        "metadata": {
                            "fixture_label": "synthetic_offline_smoke_not_a_benchmark_score"
                        },
                    }
                ),
                encoding="utf-8",
            )
            cases, source_manifest = load_local_cases(load_import_spec(import_manifest))
            case = cases[0]
            judge_role = {
                "backend": "chat_completions",
                "profile": "relay",
                "model": "fixture-coser-critic",
                "model_revision": "synthetic-v1",
                "generation": {"temperature": 0.0, "max_tokens": 4096},
            }
            runtime = root / "coser.json"
            runtime.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "benchmark_id": "coser",
                        "execution": {"request_timeout_seconds": 30},
                        "roles": {"judge": judge_role},
                        "prompts": {
                            "critic": {
                                "source": "official_repository",
                                "revision": "fixture",
                                "locator": "fixture",
                            }
                        },
                        "environment": {"max_judge_context_tokens": 20000},
                    }
                ),
                encoding="utf-8",
            )
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "coser-source-run",
                        "identity": {"benchmark_id": "coser", "judge": {}},
                        "metadata": {"runtime_config": str(runtime)},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "source_manifest.json").write_text(
                json.dumps(jsonable(source_manifest)), encoding="utf-8"
            )
            (run_dir / "sample_manifest.json").write_text(
                json.dumps({"selected_case_ids": [case.case_id]}), encoding="utf-8"
            )
            public_dialogue = [
                {"speaker": item["speaker"], "content": item["content"]}
                for item in case.gold["reference_dialogue"]
            ]
            source_record = {
                "run_id": "coser-source-run",
                "benchmark_id": "coser",
                "case_id": case.case_id,
                "group_id": case.group_id,
                "repetition": 0,
                "checkpoint_attempt": 0,
                "status": "completed",
                "prediction": {"public_dialogue": public_dialogue},
                "metrics": [
                    {"name": f"coser.scene.{dimension}", "value": 90.0}
                    for dimension in (
                        "storyline_consistency",
                        "anthropomorphism",
                        "character_fidelity",
                        "storyline_quality",
                    )
                ],
                "metadata": {
                    "judge_provenance": {
                        "judge_models": ["fixture-coser-critic"],
                        "judge_revisions": ["synthetic-v1"],
                    },
                    "runtime_provenance": {"max_judge_context_tokens": 20000},
                    "execution_provenance": {
                        "support_roles": {"judge": judge_role}
                    },
                },
            }
            (run_dir / "records.jsonl").write_text(
                json.dumps(source_record) + "\n", encoding="utf-8"
            )

            result = run_coser_rejudge_pilot(
                run_dir,
                import_manifest=import_manifest,
                runtime_config=runtime,
                limit=1,
                validate_only=True,
            )

            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["network_calls"], 0)
            self.assertEqual(
                result["pilot_manifest"]["request_validation"]["validated_request_count"],
                4,
            )
            self.assertEqual(
                result["pilot_manifest"]["judge"]["configuration_source"],
                "records.metadata.execution_provenance.support_roles.judge",
            )
            self.assertFalse(
                (run_dir / "rejudge_official_prompt_pilot_n1_seed20260826").exists()
            )

            formal_validation = run_coser_rejudge(
                run_dir,
                import_manifest=import_manifest,
                runtime_config=runtime,
                validate_only=True,
            )
            self.assertEqual(formal_validation["status"], "valid")
            self.assertEqual(
                formal_validation["rejudge_manifest"]["mode"],
                "formal_full_correction",
            )
            self.assertEqual(
                formal_validation["rejudge_manifest"]["selection"]["selected_case_ids"],
                [case.case_id],
            )

            responses = {
                f"coser-rejudge:{case.case_id}:{dimension}": json.dumps(
                    {
                        official_name: {
                            "flaws": [
                                {
                                    "instance": "fixture flaw",
                                    "type": "fixture",
                                    "severity": 5,
                                }
                            ]
                        }
                    }
                )
                for dimension, official_name in (
                    ("storyline_consistency", "Storyline Consistency"),
                    ("anthropomorphism", "Anthropomorphism"),
                    ("character_fidelity", "Character Fidelity"),
                    ("storyline_quality", "Storyline Quality"),
                )
            }
            from sim_eval.backends.replay import ReplayBackend

            corrected_dir = root / "corrected"
            with patch(
                "sim_eval.coser_rejudge.build_api_backend",
                return_value=ReplayBackend(responses),
            ):
                corrected = run_coser_rejudge(
                    run_dir,
                    import_manifest=import_manifest,
                    runtime_config=runtime,
                    output_directory=corrected_dir,
                    max_workers=1,
                )
            self.assertEqual(corrected["status"], "completed")
            for name in (
                "rejudge_manifest.json",
                "judge_records.jsonl",
                "corrected_records.jsonl",
                "corrected_metrics.json",
                "correction_manifest.json",
            ):
                self.assertTrue((corrected_dir / name).is_file(), name)
            corrected_record = json.loads(
                (corrected_dir / "corrected_records.jsonl").read_text(encoding="utf-8")
            )
            corrected_values = {
                item["name"]: item["value"] for item in corrected_record["metrics"]
            }
            self.assertIn("coser.scene.critic_average", corrected_values)
            self.assertNotEqual(corrected_values["coser.scene.critic_average"], 90.0)
            self.assertEqual(
                corrected_record["metadata"]["critic_prompt_revision"],
                "coser-self-play-deduct-template-upstream-exact-v1",
            )
            unchanged_source = json.loads(
                (run_dir / "records.jsonl").read_text(encoding="utf-8")
            )
            self.assertTrue(all(item["value"] == 90.0 for item in unchanged_source["metrics"]))


if __name__ == "__main__":
    unittest.main()
