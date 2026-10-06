import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_eval.errors import ConfigurationError
from sim_eval.local_env import configure_resource_cache_defaults, load_local_env


class LocalEnvTests(unittest.TestCase):
    def test_loads_local_values_without_overriding_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env.local"
            path.write_text(
                "SIM_EVAL_TEST_KEEP=file-value\n"
                "export SIM_EVAL_TEST_NEW='new=value'\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"SIM_EVAL_TEST_KEEP": "process-value"}, clear=False):
                os.environ.pop("SIM_EVAL_TEST_NEW", None)
                self.assertEqual(load_local_env(path), path.resolve())
                self.assertEqual(os.environ["SIM_EVAL_TEST_KEEP"], "process-value")
                self.assertEqual(os.environ["SIM_EVAL_TEST_NEW"], "new=value")
                os.environ.pop("SIM_EVAL_TEST_NEW", None)

    def test_rejects_invalid_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env.local"
            path.write_text("not-an-assignment\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_local_env(path)

    def test_configures_cluster_resource_cache_defaults_from_explicit_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(
                os.environ,
                {"SIM_EVAL_RESOURCE_BASE": temporary},
                clear=False,
            ):
                os.environ.pop("TIKTOKEN_CACHE_DIR", None)
                os.environ.pop("NLTK_DATA", None)
                configured = configure_resource_cache_defaults()
                self.assertEqual(
                    configured["TIKTOKEN_CACHE_DIR"],
                    str(Path(temporary).resolve() / ".cache" / "tiktoken"),
                )
                self.assertEqual(
                    configured["NLTK_DATA"],
                    str(Path(temporary).resolve() / ".cache" / "nltk_data"),
                )
                os.environ.pop("TIKTOKEN_CACHE_DIR", None)
                os.environ.pop("NLTK_DATA", None)

    def test_resource_cache_defaults_do_not_override_explicit_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            explicit_tiktoken = str(Path(temporary) / "custom-tiktoken")
            explicit_nltk = str(Path(temporary) / "custom-nltk")
            with patch.dict(
                os.environ,
                {
                    "SIM_EVAL_RESOURCE_BASE": temporary,
                    "TIKTOKEN_CACHE_DIR": explicit_tiktoken,
                    "NLTK_DATA": explicit_nltk,
                },
                clear=False,
            ):
                configured = configure_resource_cache_defaults()
                self.assertEqual(configured["TIKTOKEN_CACHE_DIR"], explicit_tiktoken)
                self.assertEqual(configured["NLTK_DATA"], explicit_nltk)


if __name__ == "__main__":
    unittest.main()
