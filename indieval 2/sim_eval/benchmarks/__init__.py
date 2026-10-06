"""Built-in benchmark adapters and registration entrypoint."""

from __future__ import annotations

import importlib


_BUILTIN_MODULES = (
    "sim_eval.benchmarks.fantom",
    "sim_eval.benchmarks.social_r1",
    "sim_eval.benchmarks.lifechoices",
    "sim_eval.benchmarks.behaviorchain",
    "sim_eval.benchmarks.alignx",
    "sim_eval.benchmarks.humanllm",
    "sim_eval.benchmarks.humanual",
    "sim_eval.benchmarks.coser",
    "sim_eval.benchmarks.sotopia",
    "sim_eval.benchmarks.agentsense",
    "sim_eval.benchmarks.tau_usi",
    "sim_eval.benchmarks.userlm",
    "sim_eval.benchmarks.mirrorbench",
)


def register_builtin_adapters() -> None:
    for module in _BUILTIN_MODULES:
        importlib.import_module(module)


__all__ = ["register_builtin_adapters"]
