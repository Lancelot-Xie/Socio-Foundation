import copy
import csv
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sim_eval.capability_batch import (
    DEFAULT_MAPPING, FULL_ENTRIES, build_batch_report, load_mapping, write_batch_report,
)
from sim_eval.errors import ArtifactError

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def fixture(root, *, thinking=False):
    specs = load_mapping(DEFAULT_MAPPING, "by-type")["metrics"]
    plan = {"suite_mode": "formal", "global_eval_model": "Deepseek", "entries": [],
            "model": {"model": root.name, "model_revision": "weights-v1", "extra_body": {
                "chat_template_kwargs": {} if thinking is None else {"enable_thinking": thinking}}}}
    summary = {"suite_mode": "formal", "status": "completed", "failed_entry_count": 0, "entries": []}
    for entry in sorted(FULL_ENTRIES):
        repetitions = 10 if entry == "userlm_lic" else 1
        metrics = {r["metric"]: {"value": 0.123456789, "unit": "proportion", "direction": "higher_is_better", "metadata": {}}
                   for r in specs if r["entry"] == entry}
        benchmark = "userlm" if entry.startswith("userlm_") else entry
        child_dir = root / "entries" / entry
        rows = [{"case_id": entry + "-case", "repetition": rep, "status": "completed",
                 "metrics": [{"name": name, **value} for name, value in metrics.items()],
                 "metadata": {"execution_provenance": {"evaluated_role_identity": {
                     "extra_body": plan["model"]["extra_body"]}}}} for rep in range(repetitions)]
        child_dir.mkdir(parents=True, exist_ok=True)
        (child_dir / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        dump(child_dir / "run_manifest.json", {"identity": {"seed": 42, "prompt_revision": "p1", "scorer_revision": "s1"}})
        dump(child_dir / "suite_summary.json", {"benchmark_id": benchmark, "status": "completed",
             "selected_case_count": 1, "repetitions_per_case": repetitions,
             "result_count": repetitions, "completed_count": repetitions, "failed_count": 0,
             "pending_judge_count": 0, "metrics": metrics, "artifacts": {"records": "records.jsonl"}})
        summary["entries"].append({"id": entry, "benchmark_id": benchmark, "status": "completed",
                                   "child_summary": f"entries/{entry}/suite_summary.json"})
        plan["entries"].append({"id": entry, "selection_kind": "full_import_all_dependency_groups",
                                "selected_case_ids": [entry + "-case"], "selected_case_count": 1,
                                "source_population": 1, "source_manifest_digest": entry + "-source"})
    dump(root / "suite_summary.json", summary)
    dump(root / "suite_plan.json", plan)
    (root / "metric_summary.md").write_text("Existing rounded report; not the score source.")


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class CapabilityBatchTests(unittest.TestCase):
    def test_fixed_mapping_has_no_fantom_all_or_duplicate_types(self):
        mapping = load_mapping(DEFAULT_MAPPING, "by-type")
        self.assertEqual(len(mapping["metrics"]), 33)
        self.assertEqual(len(load_mapping(DEFAULT_MAPPING, "one-per-dataset")["metrics"]), 20)
        self.assertEqual([r["metric"] for r in mapping["metrics"] if r["entry"] == "fantom"], ["fantom.item_correct"])
        self.assertEqual(len({(r["dimension"], r["dataset"], r["metric_type"]) for r in mapping["metrics"]}), 33)
        with TemporaryDirectory() as tmp:
            mapping["metrics"].append(copy.deepcopy(mapping["metrics"][0]))
            path = Path(tmp) / "bad.json"
            dump(path, mapping)
            with self.assertRaises(ArtifactError):
                load_mapping(path, "by-type")

    def test_multiple_runs_preserve_zero_missing_precision_and_inputs(self):
        with TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"
            fixture(formal / "a")
            fixture(formal / "b", thinking=None)
            path = formal / "a/entries/lifechoices/suite_summary.json"
            d = json.loads(path.read_text())
            d["metrics"]["lifechoices.accuracy"]["value"] = 0
            dump(path, d)
            path = formal / "b/entries/fantom/suite_summary.json"
            d = json.loads(path.read_text()); d["metrics"].clear(); dump(path, d)
            before = hashes(formal)
            report, paths = write_batch_report(root=formal, output_dir=Path(tmp) / "out")
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual(hashes(formal), before)
            self.assertTrue(any("thinking" in w for w in report["runs"][1]["warnings"]))
            self.assertIn("未显式设置", paths["capability_summary.md"].read_text())
            with paths["capability_summary.csv"].open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(next(r for r in rows if r["metric"] == "lifechoices.accuracy")["a"], "0.0")
            self.assertEqual(next(r for r in rows if r["metric"] == "fantom.item_correct")["b"], "")
            self.assertEqual(next(r for r in rows if r["metric"] == "alignx.direct_choice_accuracy")["a"], "0.123456789")
            self.assertIn("—†", paths["capability_summary.md"].read_text())

    def test_failed_partial_pending_and_no_markdown_runs_keep_metrics(self):
        with TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"
            fixture(formal / "good")
            for name in ("partial", "failed", "pending", "missing_records", "no_md", "short_lic"):
                fixture(formal / name)
            p = formal / "partial/suite_summary.json"
            d = json.loads(p.read_text()); d["entries"] = d["entries"][:1]; dump(p, d)
            p = formal / "failed/suite_summary.json"
            d = json.loads(p.read_text()); d["status"] = "completed_with_failures"; dump(p, d)
            p = formal / "pending/entries/coser/suite_summary.json"
            d = json.loads(p.read_text()); d["pending_judge_count"] = 1; dump(p, d)
            (formal / "missing_records/entries/alignx/records.jsonl").unlink()
            (formal / "no_md/metric_summary.md").unlink()
            p = formal / "short_lic/entries/userlm_lic/records.jsonl"
            p.write_text(p.read_text().splitlines()[0] + "\n")
            report = build_batch_report(formal)
            self.assertEqual(len(report["runs"]), 7)
            self.assertEqual(len(report["skipped"]), 0)
            runs = {r["directory"]: r for r in report["runs"]}
            self.assertEqual(runs["pending"]["totals"]["pending_judge_count"], 1)
            self.assertEqual(runs["short_lic"]["health"]["userlm_lic"]["missing_count"], 9)
            self.assertEqual(runs["short_lic"]["health"]["userlm_lic"]["completed_count"], 1)
            self.assertFalse(runs["missing_records"]["health"]["alignx"]["records_available"])
            self.assertEqual(runs["failed"]["status"], "completed_with_failures")

    def test_wrong_sample_keys_and_stale_counts_are_flagged_without_dropping_scores(self):
        with TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"
            fixture(formal / "wrong_keys"); fixture(formal / "stale")
            p = formal / "wrong_keys/entries/lifechoices/records.jsonl"
            d = json.loads(p.read_text()); d["case_id"] = "wrong"; p.write_text(json.dumps(d) + "\n")
            p = formal / "stale/entries/lifechoices/suite_summary.json"
            d = json.loads(p.read_text()); d["result_count"] = 0; dump(p, d)
            report = build_batch_report(formal)
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual(len(report["skipped"]), 0)
            for run in report["runs"]:
                self.assertTrue(run["health"]["lifechoices"]["issues"])
                self.assertEqual(run["cells"][0]["value"], 0.123456789)

    def test_root_summary_missing_and_single_entry_runs_are_supported(self):
        with TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"; fixture(formal / "a")
            (formal / "a/suite_summary.json").unlink()
            (formal / "a/suite_plan.json").unlink()
            (formal / "a/metric_summary.md").unlink()
            import shutil
            for child in (formal / "a/entries").iterdir():
                if child.name != "coser":
                    shutil.rmtree(child)
            report = build_batch_report(formal)
            self.assertEqual(len(report["runs"]), 1)
            run = report["runs"][0]
            self.assertEqual(run["status"], "missing_root_summary")
            self.assertEqual(len(run["missing_entries"]), 13)
            self.assertIsNone(run["summary_sha256"])
            for spec, cell in zip(report["metrics"], run["cells"]):
                if spec["entry"] == "coser":
                    self.assertEqual(cell["value"], 0.123456789)
                else:
                    self.assertIsNone(cell["value"])

    def test_latest_attempt_counts_and_status_csv(self):
        with TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"; fixture(formal / "a")
            p = formal / "a/entries/lifechoices/records.jsonl"
            good = json.loads(p.read_text())
            old = {**good, "status": "failed", "metrics": []}
            p.write_text(json.dumps(old) + "\n" + json.dumps(good) + "\n")
            p = formal / "a/entries/alignx/records.jsonl"
            bad = json.loads(p.read_text()); bad["status"] = "failed"
            p.write_text(json.dumps(bad) + "\n")
            report, paths = write_batch_report(root=formal, output_dir=Path(tmp) / "out")
            h = report["runs"][0]["health"]
            self.assertEqual(h["lifechoices"]["result_count"], 1)
            self.assertEqual(h["lifechoices"]["failed_count"], 0)
            self.assertEqual(h["alignx"]["failed_count"], 1)
            self.assertEqual(report["runs"][0]["totals"]["failed_count"], 1)
            with paths["run_status.csv"].open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(next(r for r in rows if r["entry"] == "alignx")["failed_count"], "1")

    def test_corrupt_child_or_records_do_not_hide_other_aggregates(self):
        with TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"; fixture(formal / "a")
            (formal / "a/entries/lifechoices/suite_summary.json").write_text("invalid")
            (formal / "a/entries/alignx/records.jsonl").write_text("invalid\n")
            report = build_batch_report(formal)
            self.assertEqual(len(report["runs"]), 1)
            self.assertIsNone(report["runs"][0]["cells"][0]["value"])
            self.assertEqual(report["runs"][0]["cells"][1]["value"], 0.123456789)
            self.assertIn("lifechoices", report["runs"][0]["missing_entries"])

    def test_skip_only_runs_without_readable_selected_metric_values(self):
        with TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"; fixture(formal / "a")
            for p in (formal / "a/entries").glob("*/suite_summary.json"):
                d = json.loads(p.read_text()); d["metrics"].clear(); dump(p, d)
            report = build_batch_report(formal)
            self.assertEqual(len(report["runs"]), 0)
            self.assertEqual(len(report["skipped"]), 1)
            self.assertIn("没有可用的所选主指标", report["skipped"][0]["reason"])

    def test_not_applicable_does_not_become_zero(self):
        with TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"; fixture(formal / "a")
            p = formal / "a/entries/userlm_lic/suite_summary.json"
            d = json.loads(p.read_text()); m = d["metrics"]["userlm.lic.two_domain_macro.intent_coverage"]
            m.update(value=None, metadata={"availability": "not_applicable"}); dump(p, d)
            report, paths = write_batch_report(root=formal, output_dir=Path(tmp) / "out")
            i = next(i for i, r in enumerate(report["metrics"]) if r["metric"] == "userlm.lic.two_domain_macro.intent_coverage")
            self.assertIsNone(report["runs"][0]["cells"][i]["value"])
            self.assertIn("N/A", paths["capability_summary.md"].read_text())

    def test_output_cannot_overwrite_input_tree(self):
        with TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"; fixture(formal / "a")
            with self.assertRaises(ArtifactError):
                write_batch_report(root=formal, output_dir=formal / "a")

    def test_cli_from_other_directory_and_repeatable_output(self):
        with TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"; fixture(formal / "a")
            args = [sys.executable, str(ROOT / "scripts/summarize_capability_dimensions.py"),
                    "--root", str(formal), "--output-dir", str(Path(tmp) / "out"), "--selection", "one-per-dataset"]
            result = subprocess.run(args, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            before = hashes(Path(tmp) / "out")
            result = subprocess.run(args, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(hashes(Path(tmp) / "out"), before)
            d = json.loads((Path(tmp) / "out/capability_summary.json").read_text())
            self.assertEqual(len(d["metrics"]), 20)


if __name__ == "__main__":
    unittest.main()
