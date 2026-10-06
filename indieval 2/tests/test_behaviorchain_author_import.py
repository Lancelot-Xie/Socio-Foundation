import copy
import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from sim_eval.benchmarks.behaviorchain import BehaviorChainAdapter
from sim_eval.contracts import CaseResult, MetricValue, ResultStatus
from sim_eval.data.loaders import load_import_spec, load_local_cases
from sim_eval.metric_markdown import _primary_metric_names

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from helpers.behaviorchain_material import normalize_persona


class AuthorBehaviorChainTests(unittest.TestCase):
    def fixture(self):
        def node(i):
            return {"summary_refined": f"context-{i}", "key_behavior": f"answer-{i}",
                    "meaningful": 1, "options_out_of_order": {
                        "d": f"wrong-d-{i}", "b": f"answer-{i}",
                        "c": f"wrong-c-{i}", "a": f"wrong-a-{i}"},
                    "right_option_index": "b", "original": "PRIVATE ANNOTATION"}
        raw = {"title": "Synthetic", "profile": {"Name": "Invented"}, "max_index": 1,
               "summary": [{"chapter_num": "before", "chapter_content": "past"},
                           {"chapter_num": "after", "chapter_content": "FUTURE SUMMARY"}],
               "examples": [{"key_behavior": "seed", "summary_refined": "seed context"}, node(1), node(2)]}
        selected = {"raw_node_count": 3, "valid_mcq_node_count": 2, "persona_id": "synthetic",
                    "source_folder": "fixture", "source_relpath": "fixture.json", "raw_file_sha256": "fixture",
                    "seen_in_odysim_post_train": False, "odysim_post_train_node_count": 0}
        return raw, selected

    def test_initial_context_is_unscored_and_no_future_or_annotation_is_exposed(self):
        raw, selected = self.fixture()
        before = copy.deepcopy(raw)
        rows = normalize_persona(raw, selected)
        self.assertEqual(raw, before)
        self.assertEqual([r["input"]["chain_index"] for r in rows], [0, 1])
        self.assertEqual([r["input"]["chain_length"] for r in rows], [2, 2])
        self.assertEqual(rows[0]["input"]["prior_nodes"], [])
        self.assertEqual(rows[1]["input"]["prior_nodes"], [{"context": "context-1", "behavior": "answer-1"}])
        self.assertEqual(rows[0]["input"]["history"][-1]["behavior"], "seed")
        self.assertEqual(rows[0]["gold"], 1)
        self.assertEqual(rows[0]["input"]["candidates"][1], "answer-1")
        visible = json.dumps(rows[0]["input"])
        self.assertNotIn("FUTURE SUMMARY", visible)
        self.assertNotIn("PRIVATE ANNOTATION", visible)
        self.assertNotIn("context-2", visible)
        self.assertNotIn("right_option_index", visible)

    def test_empty_seed_and_source_cutoff_are_audited_without_inventing_content(self):
        raw, selected = self.fixture()
        raw["examples"][0] = {"key_behavior": "", "meaningful": 1}
        raw["max_index"] = 10
        rows = normalize_persona(raw, selected)
        self.assertFalse(rows[0]["metadata"]["initial_context_has_behavior"])
        self.assertTrue(rows[0]["metadata"]["summary_cutoff_exceeds_available"])
        self.assertEqual(rows[0]["input"]["history"], raw["summary"])
        del raw["examples"][2]["options_out_of_order"]
        with self.assertRaisesRegex(ValueError, "all valid MCQ"):
            normalize_persona(raw, selected)



if __name__ == "__main__":
    unittest.main()
