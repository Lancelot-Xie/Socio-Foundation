import csv
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sim_eval.capability_comparison import (
    build_capability_comparison, load_dimension_mapping, render_capability_markdown,
    render_capability_summary_markdown, write_capability_comparison,
)
from sim_eval.errors import ArtifactError

ROOT = Path(__file__).resolve().parents[1]


def metric(value, direction="higher_is_better", unit="proportion", **metadata):
    return {"value": value, "direction": direction, "unit": unit, "metadata": metadata}


def write_run(root, entry, metrics, *, records=None, selected=2, identity=None):
    child = root / "entries" / entry
    run = child / "run"
    run.mkdir(parents=True, exist_ok=True)
    benchmark = "userlm" if entry.startswith("userlm_") else entry
    if records is None:
        records = [{"case_id": f"c{i}", "repetition": 0, "status": "completed",
                    "metrics": [{"name": name, **value} for name, value in metrics.items()]}
                   for i in range(selected)]
    records = [{"repetition": 0, **r} for r in records]
    (run / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (run / "run_manifest.json").write_text(json.dumps({"identity": identity or {
        "seed": 17, "prompt_revision": "same-prompt", "source_revision": "same-source",
        "scorer_revision": "same-scorer", "model": root.name}}))
    latest = {(r["case_id"], r.get("repetition", 0)): r for r in records}
    summary = {"benchmark_id": benchmark, "metrics": metrics, "status": "completed",
               "selected_case_count": selected, "result_count": len(latest),
               "completed_count": sum(r["status"] == "completed" for r in latest.values()),
               "failed_count": sum(r["status"] == "failed" for r in latest.values()),
               "artifacts": {"records": "run/records.jsonl"}}
    (child / "suite_summary.json").write_text(json.dumps(summary))
    (root / "suite_summary.json").write_text(json.dumps({"status": "completed", "entries": [
        {"id": entry, "benchmark_id": benchmark, "child_summary": f"entries/{entry}/suite_summary.json"}]}))


def row(report, name):
    return next(r for r in report["rows"] if r["metric"] == name)


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class CapabilityComparisonTests(unittest.TestCase):
    def build(self, a, b):
        return build_capability_comparison(baseline_path=a, candidate_path=b)

    def test_taxonomy_uses_exact_protocols_and_no_composite(self):
        mapping = load_dimension_mapping()
        self.assertEqual([d["id"] for d in mapping["dimensions"]], list("FSUTN"))
        assignments = {r["metric"]: (r["dimension"], r["entry"]) for r in mapping["metrics"]}
        self.assertEqual(assignments["behaviorchain.prediction.avg_score"][0], "F")
        self.assertEqual(assignments["behaviorchain.prediction.cum_score"][0], "T")
        self.assertEqual(assignments["humanual.state.communication_alignment"][0], "F")
        self.assertEqual(assignments["userlm.intrinsic.termination_f1"], ("N", "userlm_section3"))
        self.assertEqual(assignments["userlm.lic.two_domain_macro.intent_coverage"], ("U", "userlm_lic"))
        self.assertNotIn("alignx.alignment_accuracy", assignments)
        self.assertNotIn("sotopia.diagnostic.mean_across_agents.goal", assignments)
        self.assertEqual(mapping["aggregation"], "group_only_no_composite_score")

    def test_summary_matches_all_five_dimension_headline_metrics(self):
        mapping = load_dimension_mapping()
        main = json.loads((ROOT / "sim_eval/resources/reporting/capability_main_metrics_v1.json").read_text())
        selected = [r for r in mapping["metrics"] if r["summary"]]
        keys = lambda rows: {(r["dimension"], r["entry"], r["metric"]) for r in rows}
        self.assertEqual(keys(selected), keys(main["metrics"]))
        self.assertEqual(len(selected), 33)
        self.assertEqual(mapping["dimensions"], main["dimensions"])
        from sim_eval.metric_markdown import PRIMARY_METRICS
        for entry in {r["entry"] for r in main["metrics"]}:
            self.assertEqual(set(PRIMARY_METRICS[entry]), {r["metric"] for r in main["metrics"] if r["entry"] == entry})
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.json"
            mapping["metrics"].append(dict(mapping["metrics"][0]))
            path.write_text(json.dumps(mapping))
            with self.assertRaises(ArtifactError):
                load_dimension_mapping(path)
            mapping["metrics"].pop()
            next(r for r in mapping["metrics"] if r["metric"] == "humanllm.diagnostic.top5_accuracy")["summary"] = False
            path.write_text(json.dumps(mapping))
            with self.assertRaises(ArtifactError):
                load_dimension_mapping(path)

    def test_summary_does_not_fallback_to_available_chain_average(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            for p in (a, b):
                write_run(p, "behaviorchain", {"behaviorchain.prediction.avg_score": metric(.8)})
            report = self.build(a, b)
            selected = next(r for r in report["summary_rows"] if r["dimension"] == "F" and r["dataset"] == "BehaviorChain")
            self.assertEqual(selected["metric"], "behaviorchain.diagnostic.node_micro_score")
            self.assertIsNone(selected["baseline"]["value"])
            self.assertEqual(selected["baseline"]["status"], "missing_metric")
            self.assertEqual(row(report, "behaviorchain.prediction.avg_score")["baseline"]["value"], .8)
            self.assertEqual(selected, row(report, selected["metric"]))

    def test_summary_is_one_table_and_csv_keeps_exact_values(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name="lifechoices.accuracy"
            write_run(a,"lifechoices",{name:metric(.61234567)})
            write_run(b,"lifechoices",{name:metric(.71234567)})
            outputs=write_capability_comparison(baseline_path=a,candidate_path=b,output_dir=Path(tmp)/"out")
            report=json.loads(outputs["json"].read_text())
            markdown=outputs["summary_md"].read_text()
            self.assertEqual(markdown.count("| 能力 |"),1)
            self.assertEqual(sum(line.startswith("| ") for line in markdown.splitlines()),34)
            self.assertNotIn("## F.",markdown)
            compact=list(csv.DictReader(io_string(outputs["summary_csv"])))
            full=list(csv.DictReader(io_string(outputs["csv"])))
            self.assertEqual(len(compact),33)
            self.assertEqual(len(full),53)
            selected=next(r for r in compact if r["metric"]==name)
            original=next(r for r in full if r["metric"]==name)
            self.assertEqual(selected,original)
            self.assertEqual(float(selected["baseline_value"]),.61234567)
            self.assertTrue(all(r["summary_reason"] for r in report["summary_rows"]))
            self.assertEqual(markdown,render_capability_summary_markdown(report))

    def test_zero_missing_and_nonfinite_are_distinct(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name = "lifechoices.accuracy"
            write_run(a, "lifechoices", {name: metric(0)})
            write_run(b, "lifechoices", {name: metric(None, availability="unavailable")})
            r = row(self.build(a,b), name)
            self.assertEqual(r["baseline"]["value"], 0)
            self.assertEqual(r["baseline"]["status"], "available")
            self.assertEqual(r["candidate"]["status"], "unavailable")
            self.assertIsNone(r["delta"])
            absent = row(self.build(a,b), "coser.scene.character_fidelity")
            self.assertEqual(absent["baseline"]["status"], "missing_entry")
            write_run(b, "lifechoices", {name: metric(float("nan"))})
            r = row(self.build(a,b), name)
            self.assertEqual(r["candidate"]["status"], "invalid_value")
            json.dumps(self.build(a,b), allow_nan=False)

    def test_raw_deltas_and_closer_to_zero_direction_are_preserved(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name = "mirrorbench.lexical.mattr.z_score_mean"
            write_run(a, "mirrorbench", {name: metric(-2, "closer_to_zero", None)})
            write_run(b, "mirrorbench", {name: metric(-0.5, "closer_to_zero", None)})
            report=self.build(a,b);r=row(report,name)
            self.assertEqual(r["delta"], 1.5)
            self.assertEqual(r["baseline"]["direction"], "closer_to_zero")
            self.assertIn("→0",render_capability_markdown(report))
            self.assertNotIn("winner",r)

    def test_unit_or_direction_mismatch_has_no_delta(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name="lifechoices.accuracy"
            write_run(a,"lifechoices",{name:metric(.5)})
            write_run(b,"lifechoices",{name:metric(60,unit="percent")})
            self.assertIsNone(row(self.build(a,b),name)["delta"])
            write_run(b,"lifechoices",{name:metric(.6,direction="lower_is_better")})
            self.assertIsNone(row(self.build(a,b),name)["delta"])

    def test_resume_dedup_and_different_valid_subsets_with_equal_counts(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name="coser.scene.character_fidelity"
            def record(key,value,status="completed"):
                return {"case_id":key,"repetition":0,"status":status,"metrics":[{"name":name,"value":value}]}
            write_run(a,"coser",{name:metric(50,unit="score")},records=[record("c0",None,"failed"),record("c0",50),record("c1",None)])
            write_run(b,"coser",{name:metric(60,unit="score")},records=[record("c0",None),record("c1",60)])
            report=self.build(a,b);r=row(report,name)
            self.assertEqual(r["baseline"]["coverage"]["valid"],1)
            self.assertEqual(r["baseline"]["coverage"]["applicable"],2)
            self.assertEqual(r["candidate"]["coverage"]["valid"],1)
            self.assertTrue(any("有效样本集合不同" in f for f in r["flags"]))
            audit=next(e for e in report["entries"] if e["entry"]=="coser")
            self.assertEqual(audit["population"],"same_record_keys")
            self.assertEqual(audit["baseline"]["observed_latest_records"],2)

    def test_chain_coverage_uses_chains_not_nodes(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name="behaviorchain.prediction.cum_score"
            value=metric(.6, chain_count=100, available_chain_count=96)
            records=[{"case_id":"node1","status":"completed","metrics":[]}]
            for p in (a,b):write_run(p,"behaviorchain",{name:value},records=records,selected=1529)
            cov=row(self.build(a,b),name)["baseline"]["coverage"]
            self.assertEqual((cov["valid"],cov["applicable"],cov["natural_unit"]),(96,100,"chain"))

    def test_missing_domain_does_not_become_available_macro(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name="userlm.lic.two_domain_macro.assistant_task_score"
            value=metric(None,availability="unavailable",aggregation="equal_weight_mean_of_code_and_math_task_means",
                         required_domains=["code","math"],available_domains=["code"])
            for p in (a,b):write_run(p,"userlm_lic",{name:value},records=[])
            r=row(self.build(a,b),name)
            self.assertIsNone(r["baseline"]["value"])
            self.assertEqual(r["baseline"]["coverage"]["applicable"],2)
            self.assertEqual(r["baseline"]["coverage"]["valid"],1)
            self.assertEqual(r["baseline"]["coverage"]["natural_unit"],"domain")

    def test_not_applicable_is_not_zero(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name="userlm.lic.two_domain_macro.intent_coverage"
            for p in (a,b):write_run(p,"userlm_lic",{name:metric(None,availability="not_applicable")})
            r=row(self.build(a,b),name)
            self.assertEqual(r["baseline"]["status"],"not_applicable")
            self.assertEqual(r["baseline"]["coverage"]["applicable"],0)
            self.assertIsNone(r["delta"])

    def test_fantom_set_denominator_is_not_double_counted(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            alias = {**metric(None, context_condition="short"), "denominator": 100}
            source = {**metric(None, incomplete_group_count=3, evaluator_unavailable_group_count=1), "denominator": 100}
            for p in (a,b):
                write_run(p,"fantom",{"fantom.all_star":alias,"fantom.short.inaccessible.all_star":source},records=[])
            cell=row(self.build(a,b),"fantom.all_star")["baseline"]
            self.assertIsNone(cell["value"])
            self.assertEqual(cell["coverage"]["valid"],96)
            self.assertEqual(cell["coverage"]["applicable"],100)
            self.assertEqual(cell["coverage"]["natural_unit"],"set")

    def test_changed_population_and_prompt_are_flagged(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name="lifechoices.accuracy"
            write_run(a,"lifechoices",{name:metric(.5)},identity={"prompt_revision":"old"})
            write_run(b,"lifechoices",{name:metric(.6)},identity={"prompt_revision":"new"},
                records=[{"case_id":"another","status":"completed","metrics":[{"name":name,"value":1}]}])
            report=self.build(a,b)
            audit=next(e for e in report["entries"] if e["entry"]=="lifechoices")
            self.assertEqual(audit["population"],"different_record_keys")
            self.assertIn("run_identity.prompt_revision",audit["identity_mismatches"])

    def test_failed_entry_without_child_summary_does_not_abort(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name="lifechoices.accuracy"
            write_run(a,"lifechoices",{name:metric(.5)})
            b.mkdir()
            (b/"suite_summary.json").write_text(json.dumps({"entries":[{"id":"lifechoices","status":"failed","selected_case_count":2}]}))
            report=self.build(a,b)
            self.assertTrue(report["warnings"])
            self.assertEqual(row(report,name)["candidate"]["status"],"missing_metric")

    def test_outputs_are_read_only_complete_and_portable_cli(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name="lifechoices.accuracy"
            for p in (a,b):write_run(p,"lifechoices",{name:metric(.5)})
            before=(hashes(a),hashes(b))
            outputs=write_capability_comparison(baseline_path=a,candidate_path=b,output_dir=Path(tmp)/"out")
            self.assertEqual(before,(hashes(a),hashes(b)))
            report=json.loads(outputs["json"].read_text())
            rows=list(csv.DictReader(io_string(outputs["csv"])))
            self.assertEqual(len(rows),len(report["rows"]))
            self.assertEqual(len(report["rows"]),len(load_dimension_mapping()["metrics"]))
            self.assertTrue(all("composite" not in d for d in report["dimensions"]))
            cli=subprocess.run([sys.executable,"-B",str(ROOT/"scripts/compare_capability_dimensions.py"),
                "--baseline",str(a),"--candidate",str(b),"--output-dir",str(Path(tmp)/"cli")],
                cwd=tmp,text=True,capture_output=True)
            self.assertEqual(cli.returncode,0,cli.stderr)
            self.assertTrue((Path(tmp)/"cli/capability_comparison.md").is_file())

    def test_missing_roots_compare_child_results_and_keep_missing_entries(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            name = "lifechoices.accuracy"
            for root, value in ((a, .4), (b, .7)):
                write_run(root, "lifechoices", {name: metric(value)})
                (root/"suite_summary.json").unlink()
                (root/"suite_plan.json").write_text(json.dumps({
                    "execution": {"userlm_lic_repetitions": 10},
                    "entries": [
                        {"id": "lifechoices", "benchmark_id": "lifechoices", "selected_case_count": 2},
                        {"id": "coser", "benchmark_id": "coser", "selected_case_count": 200},
                        {"id": "userlm_lic", "benchmark_id": "userlm", "selected_case_count": 100}]}))
            before = (hashes(a), hashes(b))
            with self.assertRaises(ArtifactError):
                self.build(a, b)
            output = write_capability_comparison(baseline_path=a, candidate_path=b,
                baseline_label="Qwen3-4B", candidate_label="Qwen3-4B Text-OPD",
                output_dir=Path(tmp)/"out", allow_missing_summary=True)
            report = json.loads(output['json'].read_text())
            actual = row(report, name)
            self.assertEqual(actual['baseline']['value'], .4)
            self.assertEqual(actual['candidate']['value'], .7)
            self.assertAlmostEqual(actual['delta'], .3)
            self.assertIsNone(row(report, 'coser.scene.character_fidelity')['candidate']['value'])
            lic = next(e for e in report['entries'] if e['entry'] == 'userlm_lic')
            self.assertEqual(lic['baseline']['expected_records'], 1000)
            self.assertIsNone(report['baseline']['summary'])
            self.assertIn('lifechoices', report['baseline']['child_summary_content_sha256'])
            self.assertIn('缺少根汇总', output['summary_md'].read_text())
            self.assertIn('Qwen3-4B Text-OPD', output['summary_md'].read_text())
            cli = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/compare_capability_dimensions.py'),
                '--baseline', str(a), '--candidate', str(b), '--allow-missing-summary',
                '--output-dir', str(Path(tmp)/'cli')], cwd=tmp, text=True, capture_output=True)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertTrue((Path(tmp)/'cli/capability_summary.csv').is_file())
            self.assertEqual(before, (hashes(a), hashes(b)))

    def test_allow_missing_summary_does_not_change_complete_reports(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            for root in (a, b):
                write_run(root, 'lifechoices', {'lifechoices.accuracy': metric(.5)})
            self.assertEqual(self.build(a, b), build_capability_comparison(
                baseline_path=a, candidate_path=b, allow_missing_summary=True))

    def test_missing_summary_requires_plan_and_rejects_corruption_or_escape(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            for root in (a, b):
                write_run(root, 'lifechoices', {'lifechoices.accuracy': metric(.5)})
            (a/'suite_summary.json').unlink()
            with self.assertRaises(ArtifactError):
                build_capability_comparison(baseline_path=a, candidate_path=b, allow_missing_summary=True)
            for entry in ({'id': '../b', 'benchmark_id': 'lifechoices'},
                          {'id': 'lifechoices', 'benchmark_id': 'coser'}):
                (a/'suite_plan.json').write_text(json.dumps({'entries': [entry]}))
                with self.assertRaises(ArtifactError):
                    build_capability_comparison(baseline_path=a, candidate_path=b, allow_missing_summary=True)
            (a/'suite_summary.json').write_text('broken json')
            with self.assertRaises(ArtifactError):
                build_capability_comparison(baseline_path=a, candidate_path=b, allow_missing_summary=True)


def io_string(path):
    import io
    return io.StringIO(path.read_text(encoding="utf-8-sig"))


if __name__=="__main__":
    unittest.main()
