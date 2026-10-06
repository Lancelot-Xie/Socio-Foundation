"""Offline protocol and isolation checks; these are not model benchmark scores."""
import copy
import importlib.util
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from sim_eval.benchmarks import register_builtin_adapters
from sim_eval.catalog import REQUESTED_BENCHMARK_IDS, load_catalog
from sim_eval.contracts import ModelResponse, ResultStatus, TokenUsage
from sim_eval.data.loaders import load_import_spec, load_local_cases
from sim_eval.data.schemas import SCHEMAS
from sim_eval.errors import BackendError, ValidationError
from sim_eval.extensions.supplemental import BENCHMARK_IDS, TASKS
from sim_eval.extensions.supplemental.adapter import SupplementalAdapter
from sim_eval.generic_runner import run_generic_import
from sim_eval.registry import registered_adapters
from sim_eval.runtime_config import apply_global_eval_model

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures_supplemental"
HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None


def fixture(task):
    return load_local_cases(load_import_spec(FIXTURES / f"supplemental_{task}.manifest.json"))[0][0]


class ProtocolBackend:
    """Deterministic synthetic dialogue, with hand-chosen Judge responses."""
    def __init__(self, *, invalid_judge=False, never_stop=False):
        self.requests = []
        self.invalid_judge = invalid_judge
        self.never_stop = never_stop

    def generate(self, request):
        self.requests.append(request)
        role = request.metadata["actor"]
        task = request.metadata["benchmark_id"].removeprefix("supplemental_")
        index = int(request.request_id.rsplit(":", 1)[-1])
        if role == "evaluated_model":
            text = {"hitom":"<answer>A</answer>","paratomi":"<answer>box</answer>",
                    "mistakes":"<answer>B</answer>","twinvoice":"<answer>C</answer>",
                    "socsci210":"<answer>7</answer>"}[task]
        elif role == "evaluated_user":
            text = "Please explain." if index == 0 or self.never_stop else "terminate conversation"
        elif role == "fixed_assistant":
            text = "A synthetic assistant reply."
        elif self.invalid_judge:
            text = "not JSON"
        else:
            name = request.response_format["name"]
            if name == "LikertResult":
                text = json.dumps({"key_differences":[],"similarity_score":4})
            elif name == "RatingResult":
                text = json.dumps({"analysis":"Synthetic rating.","rating":8})
            else:
                text = json.dumps({"results":[{"feature_name":"Ask short questions","analysis":"Synthetic match.","classification":"Match"} for _ in range(2 if task == "sim_doc" else 1)]})
        return ModelResponse(text, usage=TokenUsage(prompt_tokens=10,completion_tokens=10,total_tokens=20))


def execute(task, backend=None, case=None):
    return SupplementalAdapter("supplemental_"+task).execute_case(
        case or fixture(task),backend=backend or ProtocolBackend(),run_id="test",seed=19,model="candidate")


class SupplementalStaticTests(unittest.TestCase):
    def test_existing_catalog_schema_and_registry_are_still_exactly_thirteen(self):
        register_builtin_adapters()
        before = registered_adapters()
        for task in TASKS[:5]:
            execute(task)
        self.assertEqual(set(before), REQUESTED_BENCHMARK_IDS)
        self.assertEqual(registered_adapters(), before)
        self.assertEqual(set(SCHEMAS), REQUESTED_BENCHMARK_IDS)
        self.assertEqual(set(load_catalog(ROOT/"sim_eval/resources/benchmarks.json").benchmarks), REQUESTED_BENCHMARK_IDS)
        self.assertEqual(set(load_catalog(ROOT/"sim_eval/resources/supplemental/benchmarks.json").benchmarks), BENCHMARK_IDS)

    def test_static_expected_rewards_and_exact_tag_requirements(self):
        for task in TASKS[:5]:
            with self.subTest(task=task):
                result = execute(task)
                self.assertEqual(result.status, ResultStatus.COMPLETED)
                self.assertEqual(result.prediction["reward"], .5 if task == "socsci210" else 1)
                adapter = SupplementalAdapter("supplemental_"+task)
                self.assertEqual(adapter.parse_response(fixture(task), ModelResponse("A"))["reward"], 0)

    def test_original_parser_first_tag_and_think_stripping(self):
        adapter = SupplementalAdapter("supplemental_hitom")
        case = fixture("hitom")
        self.assertEqual(adapter.parse_response(case,ModelResponse("<think><answer>B</answer></think><answer>A</answer><answer>B</answer>"))["reward"],1)
        self.assertEqual(adapter.parse_response(case,ModelResponse('{"choice":"A"}'))["reward"],0)

    def test_paratomi_rejects_other_candidates_and_substrings(self):
        adapter = SupplementalAdapter("supplemental_paratomi")
        for text in ("<answer>box and basket</answer>","<answer>boxcar</answer>"):
            self.assertEqual(adapter.parse_response(fixture("paratomi"),ModelResponse(text))["reward"],0)

    def test_socsci_clamping_binary_and_missing_answer(self):
        adapter = SupplementalAdapter("supplemental_socsci210")
        case = fixture("socsci210")
        self.assertEqual(adapter.parse_response(case,ModelResponse("<answer>99</answer>"))["reward"],.5)
        binary = replace(case,gold={"response":1,"response_type":"binary","r_min":0,"r_max":1})
        self.assertEqual(adapter.parse_response(binary,ModelResponse("<answer>yes</answer>"))["reward"],1)
        self.assertEqual(adapter.parse_response(binary,ModelResponse("<answer>2</answer>"))["reward"],0)

    def test_private_gold_cannot_be_added_to_public_input(self):
        case = fixture("hitom")
        bad = replace(case,input_data={"row":{**case.input_data["row"],"correct_answer":"PRIVATE"}})
        with self.assertRaises(ValidationError):
            SupplementalAdapter(case.benchmark_id).validate_case(bad)

    def test_gold_changes_do_not_change_model_requests(self):
        for task in TASKS[:5]:
            case = fixture(task); adapter = SupplementalAdapter(case.benchmark_id)
            before = adapter.build_request(case,model="candidate",seed=1)
            gold = dict(case.gold)
            key = {"hitom":"correct_answer","paratomi":"correct_answer","mistakes":"TargetOption","twinvoice":"answer_idx","socsci210":"response"}[task]
            gold[key] = {"hitom":"other","paratomi":"basket","mistakes":"D","twinvoice":0,"socsci210":1}[task]
            after = adapter.build_request(replace(case,gold=gold),model="candidate",seed=1)
            self.assertEqual(before.messages, after.messages)

    def test_backend_failure_is_structured_and_remains_in_denominator(self):
        class Broken:
            def generate(self,request): raise BackendError("offline failure")
        failure = execute("hitom",Broken())
        self.assertEqual(failure.status,ResultStatus.FAILED)
        metric = SupplementalAdapter("supplemental_hitom").aggregate([execute("hitom"),failure])["supplemental_hitom.reward"]
        self.assertEqual((metric.value,metric.denominator),(.5,2))


@unittest.skipUnless(HAS_PYDANTIC,"optional Supplemental interactive dependency: pydantic>=2")
class SupplementalInteractiveTests(unittest.TestCase):


    def test_interactive_rewards_and_private_judge_visibility(self):
        for task,expected in (("sim_math",2.6/3),("sim_doc",4.4/5)):
            backend = ProtocolBackend(); result = execute(task,backend)
            self.assertAlmostEqual(result.prediction["reward"],expected)
            for request in backend.requests:
                text = "\n".join(m.content for m in request.messages)
                if request.metadata["actor"] != "judge":
                    self.assertNotIn("PRIVATE_REFERENCE",text)
            self.assertIsNotNone(result.metadata["rejudge_artifact"])
            self.assertEqual(sum(r.metadata["actor"]=="fixed_assistant" for r in backend.requests),2 if task=="sim_doc" else 1)

    def test_eight_turn_stop_is_preserved(self):
        backend=ProtocolBackend(never_stop=True); execute("sim_math",backend)
        self.assertEqual(sum(r.metadata["actor"]=="evaluated_user" for r in backend.requests),8)
        self.assertEqual(sum(r.metadata["actor"]=="fixed_assistant" for r in backend.requests),8)

    def test_math_empty_features_keeps_three_component_denominator(self):
        case=fixture("sim_math")
        case=replace(case,gold={**case.gold,"target_interaction_style_features":[]})
        backend=ProtocolBackend();result=execute("sim_math",backend,case)
        self.assertAlmostEqual(result.prediction["reward"],1.6/3)
        self.assertEqual(sum(r.metadata["actor"]=="judge" for r in backend.requests),2)

    def test_judge_missing_value_semantics_and_retry_limit(self):
        backend=ProtocolBackend(invalid_judge=True); result=execute("sim_math",backend)
        self.assertEqual(result.prediction["reward"],0)
        self.assertEqual(len(result.metadata["judge_failures"]),12)
        self.assertIsNone(result.metadata["source_metrics"]["sim_arena_math/writing_style_likert"])

    def test_threaded_episodes_do_not_share_sessions(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(execute,["sim_math","sim_doc"]*4))
        for result in results:
            for event in result.trace:
                if event.kind=="request":
                    self.assertTrue(event.content["request_id"].startswith(result.case_id+":"))

    def test_global_support_override_does_not_replace_candidate(self):
        from sim_eval.runtime_config import load_config_document
        _,raw=load_config_document(ROOT/"sim_eval/resources/protocols/supplemental_sim_math.json")
        changed=apply_global_eval_model(raw,"Deepseek")
        self.assertEqual(raw["roles"]["evaluated_user"],changed["roles"]["evaluated_user"])
        self.assertNotEqual(raw["roles"]["judge"]["model"],changed["roles"]["judge"]["model"])

    def test_all_seven_run_through_existing_runner_and_resume_without_calls(self):
        import yaml
        for task in TASKS:
            with self.subTest(task=task), tempfile.TemporaryDirectory(dir=ROOT) as tmp:
                path=Path(tmp);bid="supplemental_"+task
                config=yaml.safe_load((ROOT/f"sim_eval/resources/protocols/{bid}.json").read_text())
                config["execution"]["max_workers"]=1
                for role in config["roles"].values():
                    role["model"]="fixture-model"
                    role["model_revision"]="fixture-revision"
                runtime=path/"runtime.yaml";runtime.write_text(yaml.safe_dump(config))
                backend=ProtocolBackend()
                with patch("sim_eval.generic_runner.build_role_routed_backend",return_value=backend):
                    kwargs=dict(manifest_path=FIXTURES/f"{bid}.manifest.json",runtime_config_path=runtime,
                                catalog_path=ROOT/"sim_eval/resources/supplemental/benchmarks.json",output_directory=path/"out")
                    result=run_generic_import(**kwargs)
                    count=len(backend.requests); resumed=run_generic_import(**kwargs)
                self.assertEqual(result["status"],"completed")
                self.assertEqual(resumed["run_id"],result["run_id"])
                self.assertEqual(count,len(backend.requests))


if __name__=="__main__": unittest.main()
