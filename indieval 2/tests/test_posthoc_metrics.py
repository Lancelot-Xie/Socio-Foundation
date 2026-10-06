import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sim_eval.contracts import CaseResult, MetricValue, ResultStatus
from sim_eval.posthoc import recompute_posthoc_metrics


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(directory: Path, benchmark_id: str, results, old_metrics) -> None:
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": f"{benchmark_id}-run",
                "identity": {"benchmark_id": benchmark_id},
            }
        ),
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps({"run_id": f"{benchmark_id}-run", "metrics": old_metrics}),
        encoding="utf-8",
    )
    with (directory / "records.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            row = result.to_dict()
            row["checkpoint_attempt"] = 0
            handle.write(json.dumps(row) + "\n")


class PosthocMetricTests(unittest.TestCase):
    def test_agentsense_sidecar_corrects_psi_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            results = []
            for index, value in enumerate((0.0, 1.0)):
                results.append(
                    CaseResult(
                        run_id="agentsense-run",
                        benchmark_id="agentsense",
                        case_id=f"case-{index}",
                        group_id="template-1",
                        repetition=0,
                        status=ResultStatus.COMPLETED,
                        metrics=(
                            MetricValue("agentsense.episode.judge.j1", value),
                            MetricValue("agentsense.episode.judge.j2", 1.0 - value),
                            MetricValue("agentsense.episode.judge.j3", value),
                            MetricValue(
                                "agentsense.episode.private_information_accuracy", value
                            ),
                        ),
                        metadata={"profile_id": f"profile-{index}"},
                    )
                )
            old = {
                "agentsense.profile_sensitivity_index.goal": {
                    "name": "agentsense.profile_sensitivity_index.goal",
                    "value": 50.0,
                },
                "agentsense.profile_sensitivity_index.information": {
                    "name": "agentsense.profile_sensitivity_index.information",
                    "value": 50.0,
                },
            }
            _write_run(run_dir, "agentsense", results, old)
            source_hash = _hash(run_dir / "records.jsonl")

            output = recompute_posthoc_metrics(run_dir)

            corrected = json.loads((output / "corrected_metrics.json").read_text())
            metrics = corrected["metrics"]
            self.assertAlmostEqual(
                metrics["agentsense.profile_sensitivity_index.goal"]["value"],
                70.71067811865476,
            )
            self.assertEqual(
                metrics["agentsense.profile_sensitivity_index.goal"]["direction"],
                "lower_is_better",
            )
            self.assertAlmostEqual(
                metrics["agentsense.profile_sensitivity_index.information"]["value"],
                70.71067811865476,
            )
            self.assertEqual(source_hash, _hash(run_dir / "records.jsonl"))

    def test_mirrorbench_sidecar_adds_pi_deviation_and_preserves_lexical_z(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            results = []
            for index, pi in enumerate((0.25, 0.75)):
                lexical = {
                    "mattr": {"proxy": 0.6 + 0.1 * index, "human": 0.7 + 0.1 * index},
                    "hdd": {"proxy": 0.5 + 0.1 * index, "human": 0.6 + 0.1 * index},
                    "yules_k": {"proxy": 120.0 + 10 * index, "human": 100.0 + 10 * index},
                }
                raw_metrics = [MetricValue("mirrorbench.judge.pi", pi)]
                for name, values in lexical.items():
                    raw_metrics.extend(
                        (
                            MetricValue(
                                f"mirrorbench.lexical.{name}.proxy_raw",
                                values["proxy"],
                            ),
                            MetricValue(
                                f"mirrorbench.lexical.{name}.human_raw",
                                values["human"],
                            ),
                        )
                    )
                results.append(
                    CaseResult(
                        run_id="mirrorbench-run",
                        benchmark_id="mirrorbench",
                        case_id=f"case-{index}",
                        group_id=f"group-{index}",
                        repetition=0,
                        status=ResultStatus.COMPLETED,
                        metrics=tuple(raw_metrics),
                        metadata={"mirrorbench": {"lexical": lexical}},
                    )
                )
            _write_run(
                run_dir,
                "mirrorbench",
                results,
                {
                    "mirrorbench.judge.pi": {
                        "name": "mirrorbench.judge.pi",
                        "value": 0.5,
                    }
                },
            )

            output = recompute_posthoc_metrics(run_dir)

            metrics = json.loads((output / "corrected_metrics.json").read_text())["metrics"]
            self.assertEqual(metrics["mirrorbench.judge.pi_deviation"]["value"], 0.0)
            self.assertIn("ci95_half_width", metrics["mirrorbench.judge.pi"]["uncertainty"])
            self.assertEqual(
                metrics["mirrorbench.lexical.mattr.proxy_raw"]["direction"],
                "descriptive",
            )
            self.assertIn("mirrorbench.lexical.yules_k.z_score_mean", metrics)


if __name__ == "__main__":
    unittest.main()
