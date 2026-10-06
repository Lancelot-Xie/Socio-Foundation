"""Public packaging and real HTTP transport tests with synthetic data only."""
import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import yaml
from sim_eval.cli import build_parser
from sim_eval.generic_runner import run_generic_import
from sim_eval.runtime_config import load_benchmark_runtime_config
ROOT = Path(__file__).resolve().parents[1]

class SyntheticCounter:
    def count_prompt(self, request, **kwargs): return 64
    def count_response(self, response): return 8
    def count_content(self, response): return 8
    def identity(self): return {"kind": "synthetic-test-only", "revision": "v1"}

class PublicReleaseTests(unittest.TestCase):
    def test_packaged_protocols_cover_all_twenty_and_preserve_critical_settings(self):
        root = ROOT / "sim_eval/resources/protocols"
        self.assertEqual(len(list(root.glob("*.json"))), 20)
        tau = json.loads((root / "tau_usi.json").read_text())
        self.assertEqual(tau["limits"]["max_user_turns"], 60)
        self.assertEqual(tau["limits"]["max_assistant_steps_per_user_turn"], 64)
        self.assertTrue(tau["limits"]["survey_required"])
        for path in root.glob("*.json"):
            if path.stem == "tau_usi": continue
            runtime = load_benchmark_runtime_config(path)
            self.assertEqual(runtime["benchmark_id"], path.stem)
        agent = load_benchmark_runtime_config(root / "agentsense.json")
        judges = [v for k, v in agent["roles"].items() if "judge" in k]
        self.assertEqual(len(judges), 3)
        self.assertTrue(all(v["generation"]["temperature"] == 0.8 for v in judges))

    def test_report_can_read_caller_owned_run_outside_installation(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr
        from sim_eval.cli import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run", "--profile", "offline_smoke", "--backend", "replay", "--benchmarks", "lifechoices", "--output", str(root/"run")]), 0)
                self.assertEqual(main(["report", "--run", str(root/"run"), "--output", str(root/"report.md")]), 0)
            self.assertIn("lifechoices", (root/"report.md").read_text().lower())

    def test_external_runtime_location_is_explicit_and_does_not_change_digest_logic(self):
        from sim_eval.integrations.tau_bench_local import TauBenchRepository
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tau_bench").mkdir()
            with patch.dict("os.environ", {"SIM_EVAL_TAU_BENCH_ROOT": str(root)}):
                self.assertEqual(TauBenchRepository().root, root.resolve())
                self.assertEqual(TauBenchRepository(root).root, root.resolve())

    def test_example_uses_http_no_thinking_and_resumes_without_extra_calls(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.server.paths.append(self.path)
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(request)
                data = json.dumps({"id":"synthetic", "choices":[{"message":{"role":"assistant","content":"<answer>A</answer>"},"finish_reason":"stop"}],"usage":{"prompt_tokens":64,"completion_tokens":8,"total_tokens":72}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.paths = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                example = ROOT / "configs/examples/qwen3_8b_nothinking_lifechoices.yaml"
                raw = yaml.safe_load(example.read_text())
                role = raw["roles"]["evaluated_model"]
                role["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
                role.pop("base_url_env", None)
                role["model_revision"] = "synthetic-server-v1"
                role["token_accounting"]["model_revision"] = "synthetic-counter-v1"
                runtime = tmp / "runtime.yaml"
                runtime.write_text(yaml.safe_dump(raw))
                kwargs = dict(manifest_path=ROOT/"examples/lifechoices/import_manifest.json", runtime_config_path=runtime, output_directory=tmp/"results", catalog_path=ROOT/"sim_eval/resources/benchmarks.json")
                with patch("sim_eval.backends.episode_budget.get_shared_huggingface_token_counter", return_value=SyntheticCounter()):
                    first = run_generic_import(**kwargs)
                    count = len(requests)
                    second = run_generic_import(**kwargs)
                self.assertEqual(first["status"], "completed")
                self.assertGreater(count, 0)
                self.assertEqual(len(requests), count)
                self.assertEqual(first["run_id"], second["run_id"])
                self.assertEqual(first["metrics"], second["metrics"])
                self.assertEqual(first["selected_case_count"], first["completed_count"])
                self.assertTrue(all(path == "/v1/chat/completions" for path in server.paths))
                for request in requests:
                    self.assertEqual(request["model"], "Qwen/Qwen3-8B")
                    self.assertIs(request["chat_template_kwargs"]["enable_thinking"], False)
                    self.assertEqual(request["temperature"], 0.0)
                    self.assertEqual(request["max_tokens"], 1024)
                    self.assertNotIn("gold", json.dumps(request))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

if __name__ == "__main__": unittest.main()
