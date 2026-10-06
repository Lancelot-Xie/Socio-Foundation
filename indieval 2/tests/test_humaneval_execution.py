import ast
import os
import platform
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from sim_eval.contracts import BenchmarkCase
from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.userlm import UserLMAdapter
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.data.loaders import load_import_spec, load_local_cases
from sim_eval.errors import ConfigurationError
from sim_eval.executors import CodeExecutionRequest, CodeExecutionResult, CodeExecutor
from sim_eval.executors.local_guarded import LocalGuardedExecutor, LocalGuardedPolicy
from sim_eval.executors.local_hardened_linux import (
    LocalHardenedLinuxExecutor,
    LocalHardenedLinuxPolicy,
)
from sim_eval.verifiers.cache import VerificationCache
from sim_eval.verifiers.humaneval import (
    HumanEvalExecutionConfig,
    HumanEvalVerifier,
    extract_python_completion,
)


ROOT = Path(__file__).resolve().parents[1]
LIC100_MANIFEST = (
    ROOT.parent
    / "temp_data_qa"
    / "datasets"
    / "derived"
    / "userlm_eval_v2"
    / "lic100"
    / "import_manifest.json"
)
LINUX_HARDENED_SUPPORTED = platform.system() == "Linux" and platform.machine().casefold() in {
    "x86_64",
    "amd64",
    "aarch64",
    "arm64",
}


def code_case() -> BenchmarkCase:
    return BenchmarkCase(
        benchmark_id="userlm",
        case_id="humaneval-test-case",
        group_id="humaneval-test-group",
        split="test",
        source_revision="test-v1",
        input_data={
            "assistant_task": {
                "kind": "code",
                "payload": {
                    "source": "humaneval",
                    "prompt": "def add_one(value):\n    \"\"\"Return value plus one.\"\"\"\n",
                    "test": "def check(candidate):\n    assert candidate(2) == 3\n",
                    "metadata": {"func_name": "add_one"},
                },
            }
        },
        metadata={
            "source_group_id": "sharded-HumanEval/0",
            "source_record": {"source": {"task_id": "sharded-HumanEval/0"}},
        },
    )


class CountingExecutor(CodeExecutor):
    name = "counting"
    revision = "counting-v1"

    def __init__(self) -> None:
        self.calls = 0
        self.programs = []

    def identity(self):
        return {"backend": self.name, "revision": self.revision}

    def execute(self, request):
        self.calls += 1
        self.programs.append(request.program)
        return CodeExecutionResult("completed", True, "passed", 1.25, exit_code=0)


class HumanEvalExecutionTests(unittest.TestCase):
    def test_official_suffix_extraction_preserves_leading_indentation(self) -> None:
        completion, mode = extract_python_completion(
            "    return value + 1\n",
            entry_point="add_one",
        )
        self.assertEqual(mode, "official_suffix")
        self.assertTrue(completion.startswith("    return"))

    def test_local_guarded_passes_safe_program_with_sanitized_environment(self) -> None:
        policy = LocalGuardedPolicy(timeout_seconds=2, cpu_seconds=1, memory_mb=1024)
        executor = LocalGuardedExecutor(policy)
        program = """import os
import socket
assert 'OPENAI_API_KEY' not in os.environ
try:
    socket.socket()
except PermissionError:
    pass
else:
    raise AssertionError('socket guard missing')
with open('temporary.txt', 'w') as handle:
    handle.write('ok')
assert os.path.exists('temporary.txt')
"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "must-not-reach-child"}):
            result = executor.execute(CodeExecutionRequest("safe", program))
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(result.reason, "passed")
        self.assertFalse(result.metadata["executor_identity"]["network_namespace_isolation"])

    def test_local_guarded_reports_failed_test_and_timeout(self) -> None:
        executor = LocalGuardedExecutor(
            LocalGuardedPolicy(timeout_seconds=0.25, cpu_seconds=1, memory_mb=1024)
        )
        failed = executor.execute(CodeExecutionRequest("failed", "assert False, 'expected failure'\n"))
        self.assertEqual(failed.status, "completed")
        self.assertFalse(failed.passed)
        self.assertEqual(failed.reason, "failed_test")

        timed_out = executor.execute(CodeExecutionRequest("timeout", "while True:\n    pass\n"))
        self.assertEqual(timed_out.status, "timed_out")
        self.assertFalse(timed_out.passed)
        self.assertEqual(timed_out.reason, "timeout")

    def test_humaneval_verifier_extracts_fenced_full_definition_and_caches(self) -> None:
        executor = CountingExecutor()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            cache = VerificationCache(Path(temporary) / "cache", store_completion=True)
            verifier = HumanEvalVerifier(
                executor=executor,
                verifier_revision="humaneval-test-v1",
                extraction_revision="extract-test-v1",
                cache=cache,
            )
            completion = "Here is the solution:\n```python\ndef add_one(value):\n    return value + 1\n```"
            first_score, first = verifier.verify_case(code_case(), completion)
            second_score, second = verifier.verify_case(code_case(), completion)
        self.assertEqual(first_score, 1.0)
        self.assertEqual(second_score, 1.0)
        self.assertEqual(executor.calls, 1)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertIn("check(add_one)", executor.programs[0])
        self.assertEqual(first["completion_mode"], "full_definition")

    def test_userlm_episode_runs_humaneval_verification_in_main_pipeline(self) -> None:
        base = load_fixture_suite(ROOT / "tests" / "fixtures")["userlm"][0][1]
        case = replace(
            base,
            input_data={
                **base.input_data,
                "assistant_task": {
                    "kind": "code",
                    "payload": {
                        "source": "humaneval",
                        "prompt": "def reverse_text(value):\n    \"\"\"Return reversed text.\"\"\"\n",
                        "test": "def check(candidate):\n    assert candidate('abc') == 'cba'\n",
                        "metadata": {"func_name": "reverse_text"},
                    },
                },
            },
            metadata={
                **base.metadata,
                "source_group_id": "sharded-HumanEval/test",
                "source_record": {"source": {"task_id": "sharded-HumanEval/test"}},
            },
        )
        executor = LocalGuardedExecutor(LocalGuardedPolicy(timeout_seconds=2, cpu_seconds=1, memory_mb=1024))
        verifier = HumanEvalVerifier(
            executor=executor,
            verifier_revision="humaneval-pipeline-test-v1",
            extraction_revision="extract-pipeline-test-v1",
            cache=None,
        )
        adapter = UserLMAdapter(code_task_verifier=verifier)
        result = adapter.execute_case(
            case,
            backend=ReplayBackend(adapter.replay_responses(case, seed=7)),
            run_id="humaneval-pipeline-test",
            seed=7,
            model="candidate-user",
        )
        task_metric = next(
            metric for metric in result.metrics if metric.name == "userlm.extrinsic.assistant_task_score"
        )
        self.assertEqual(task_metric.value, 1.0)
        self.assertEqual(result.metadata["task_verification"]["status"], "completed")
        self.assertTrue(result.metadata["task_verification"]["verifier_passed"])

    def test_enabled_local_guarded_requires_explicit_security_acknowledgment(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "acknowledge"):
            HumanEvalExecutionConfig.from_mapping(
                {"enabled": True, "backend": "local_guarded", "network": "unsupported"}
            )
        config = HumanEvalExecutionConfig.from_mapping(
            {
                "enabled": True,
                "backend": "local_guarded",
                "acknowledge_local_guarded_is_not_security_sandbox": True,
                "network": "unsupported",
            }
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.identity()["executor"]["security_level"], "best_effort_local_subprocess_not_security_sandbox")

    def test_hardened_config_is_explicit_and_fail_closed(self) -> None:
        config = HumanEvalExecutionConfig.from_mapping(
            {
                "enabled": True,
                "backend": "local_hardened_linux",
                "network": "seccomp_denied",
            }
        )
        self.assertEqual(config.backend, "local_hardened_linux")
        self.assertEqual(config.identity()["executor"]["security_level"], "linux_kernel_hardened_subprocess")
        self.assertTrue(config.identity()["executor"]["fail_closed"])
        preflight = config.preflight()
        self.assertEqual(preflight["ready"], not preflight["blocking_reasons"])
        if preflight["ready"]:
            self.assertTrue(LINUX_HARDENED_SUPPORTED)
            self.assertGreaterEqual(preflight["landlock_abi"], 1)
            self.assertTrue(preflight["seccomp_filter_available"])
        with self.assertRaisesRegex(ConfigurationError, "network must be declared seccomp_denied"):
            HumanEvalExecutionConfig.from_mapping(
                {
                    "enabled": True,
                    "backend": "local_hardened_linux",
                    "network": "unsupported",
                }
            )



@unittest.skipUnless(LINUX_HARDENED_SUPPORTED, "local_hardened_linux requires supported Linux")
class HardenedLinuxExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guarded = LocalGuardedExecutor(
            LocalGuardedPolicy(timeout_seconds=2, cpu_seconds=1, memory_mb=1024)
        )
        self.hardened = LocalHardenedLinuxExecutor(
            LocalHardenedLinuxPolicy(timeout_seconds=2, cpu_seconds=1, memory_mb=1024)
        )

    def test_kernel_hardening_evidence_and_safe_program(self) -> None:
        result = self.hardened.execute(
            CodeExecutionRequest(
                "hardened-safe",
                "from typing import List\nimport ast\nimport math\nimport textwrap\n"
                "assert math.isqrt(81) == 9\nassert ast.literal_eval('[1]') == [1]\n"
                "assert textwrap.dedent('  x') == 'x'\nassert List[int]\n",
            )
        )
        self.assertEqual(result.status, "completed", result.to_dict())
        self.assertTrue(result.passed, result.to_dict())
        evidence = result.metadata["security_evidence"]
        self.assertNotEqual(evidence["uid_after"], 0)
        self.assertTrue(evidence["no_new_privs"])
        self.assertGreaterEqual(evidence["landlock"]["abi"], 1)
        self.assertTrue(evidence["seccomp"]["enabled"])
        self.assertEqual(evidence["kernel_self_tests"]["read_etc_passwd"], "blocked")
        self.assertEqual(evidence["kernel_self_tests"]["socket_syscall"], "blocked_with_eperm")

    def test_candidate_cannot_read_host_or_import_unsafe_modules(self) -> None:
        program = """try:
    open('/etc/passwd').read()
except PermissionError:
    pass
else:
    raise AssertionError('host filesystem read was not blocked')
try:
    import ctypes
except ImportError:
    pass
else:
    raise AssertionError('unsafe import was not blocked')
"""
        result = self.hardened.execute(CodeExecutionRequest("hardened-host-deny", program))
        self.assertEqual(result.status, "completed", result.to_dict())
        self.assertTrue(result.passed, result.to_dict())

    def test_benign_score_parity_with_local_guarded(self) -> None:
        programs = (
            "def add_one(value):\n    return value + 1\nassert add_one(2) == 3\n",
            "def add_one(value):\n    return value\nassert add_one(2) == 3\n",
            "def broken(:\n    pass\n",
            "raise ValueError('expected probe error')\n",
        )
        for index, program in enumerate(programs):
            guarded = self.guarded.execute(CodeExecutionRequest(f"guarded-{index}", program))
            hardened = self.hardened.execute(CodeExecutionRequest(f"hardened-{index}", program))
            self.assertEqual(
                (hardened.status, hardened.passed, hardened.reason),
                (guarded.status, guarded.passed, guarded.reason),
                {"program_index": index, "guarded": guarded.to_dict(), "hardened": hardened.to_dict()},
            )


if __name__ == "__main__":
    unittest.main()
