"""Read original Simulation prompt/parser code without importing VERL or network clients.

Only audited agent modules and pure helper modules are supported. Imports into the
training framework and API helpers are removed from the AST; a capture Agent stops
at the first policy call. This is a static-prefix exporter, NOT a live simulator.
The original files remain untouched. Source hashes accompany exported data.
"""

import ast
import asyncio
import copy
import hashlib
import os
import random
import re
import sys
import types
from functools import lru_cache
from pathlib import Path


MODULES = {t: f"agents/{t}/agent.py" for t in (
    "lifechoices", "fantom", "social_r1", "behavior_chain", "userllm", "mirrorbench",
    "humanual", "alignx", "humanllm", "hitom", "paratomi", "mistakes", "twinvoice", "socsci210", "sotopia")}
MODULES.update(coser="agents/coser/coser_agent.py", sim_math="agents/sim_arena/agent_math.py",
               sim_doc="agents/sim_arena/agent_doc.py")
VERIFIABLE = {"lifechoices", "fantom", "social_r1", "behavior_chain", "alignx", "humanllm",
              "hitom", "paratomi", "mistakes", "twinvoice", "socsci210"}
HELPERS = {"agents.lifechoices.prompt", "agents.userllm.helpers", "agents.coser.prompt",
           "agents.sim_arena.math_prompts", "agents.sim_arena.doc_prompts"}


def canonical_task(task):
    task = task.strip().lower().replace("-", "_")
    return {"behaviorchain": "behavior_chain", "sim_arena_math": "sim_math",
            "sim_arena_doc": "sim_doc", "userlm": "userllm"}.get(task, task)


def agent_task(task):
    task = canonical_task(task)
    return "humanual" if task.startswith("humanual_") else task


def repository():
    default = Path(__file__).resolve().parents[1] / "framework"
    root = Path(os.environ.get("SIMULATION_REPO", default)).resolve()
    if not (root / "agents").is_dir():
        raise FileNotFoundError(f"Set SIMULATION_REPO to the framework directory: {root}")
    return root


class Captured(BaseException):
    def __init__(self, messages):
        self.messages = copy.deepcopy(messages)


class ExternalRequired(BaseException):
    # Original judges sometimes catch Exception and return a neutral score.
    # Export must stop immediately instead of entering those recovery paths.
    pass


async def no_external(*args, **kwargs):
    raise ExternalRequired("Original agent needs a partner/API before its next policy call; export rendered prefixes instead")


async def no_post(*args, **kwargs):
    return None


class CaptureAgent:
    def __init__(self, client, chat, *args, **kwargs):
        self.chat = copy.deepcopy(chat)
        self.client = client

    def append(self, turn, **kwargs):
        self.chat.append(copy.deepcopy(turn))

    async def step(self, *args, **kwargs):
        if self.client is None:
            raise Captured(self.chat)
        response = self.client
        self.append({"role": "assistant", "content": response})
        return response

    async def get_agent_output(self, reward, extra_info=None, **kwargs):
        return {"reward": float(reward), "metrics": extra_info or {}}


def no_anchor(data):
    if data.get("extra_info", {}).get("generic_anchor"):
        raise ValueError("The source exporter supports formal task agents, not generic-anchor ablations")
    return False


class Imports(ast.NodeTransformer):
    def __init__(self, root, namespace):
        self.root, self.namespace = root, namespace

    def visit_Import(self, node):
        names = [n for n in node.names if n.name.split(".")[0] in sys.stdlib_module_names or n.name == "regex"]
        return ast.Import(names=names) if names else None

    def visit_ImportFrom(self, node):
        name = node.module or ""
        if name.split(".")[0] in sys.stdlib_module_names or name == "pydantic":
            return node
        if name in HELPERS:
            helper = load_source(self.root, name.replace(".", "/") + ".py")
            for alias in node.names:
                self.namespace[alias.asname or alias.name] = getattr(helper, alias.name)
        # All unsupported API/framework imports remain unavailable. Calling them
        # fails explicitly; no placeholder observation or fabricated score is used.
        return None


@lru_cache(maxsize=32)
def load_source(root, relative):
    path = Path(root) / relative
    text = path.read_text()
    name = "_opd_source_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module  # dataclass resolves annotation namespaces here
    namespace = module.__dict__
    namespace.update(Agent=CaptureAgent, process_post_chat=no_post, call_openai=no_external,
                     call_openai_parse=no_external, editlens_score=no_external,
                     is_generic_anchor=no_anchor, get_judge_model=lambda *a, **k: "disabled",
                     get_judge_reasoning=lambda *a, **k: None)
    # Extract only these pure text functions from utils.py: never execute its imports
    # or credential/client initialization.
    utils = ast.parse((Path(root) / "agents/utils.py").read_text())
    functions = [n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name in ("remove_think", "split_think")]
    namespace["re"] = re
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    tree = Imports(root, namespace).visit(ast.parse(text))
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), namespace)
    return module


def context(response=None):
    return types.SimpleNamespace(llm_client=response, tokenizer=None, is_train=False, global_step=0,
                                 config=types.SimpleNamespace(algorithm=types.SimpleNamespace(agent_version="default")))


def render_prefix(raw, task, root=None, seed=0):
    task = agent_task(task)
    if task not in MODULES:
        raise ValueError(f"Unsupported source task {task}; provide explicitly rendered messages")
    root = str(root or repository())
    module = load_source(root, MODULES[task])
    previous_random = getattr(module, "random", None)
    if task == "coser":
        # The original NSP fallback samples list(set(character_names)). Seed
        # per exported prefix and sort that set so a different Python hash seed
        # or machine cannot change which policy prefixes are exported.
        rng = random.Random(seed)
        module.random = types.SimpleNamespace(choice=lambda candidates: rng.choice(sorted(candidates)))
    try:
        asyncio.run(module.agent_loop(copy.deepcopy(raw), context()))
    except Captured as captured:
        if not captured.messages or captured.messages[-1]["role"] == "assistant":
            raise ValueError("Original agent did not expose a next-answer prefix")
        return captured.messages
    finally:
        if task == "coser":
            module.random = previous_random
    raise ValueError("Original agent ended without a policy call")


@lru_cache(maxsize=32)
def source_hashes(root, task):
    paths = [MODULES[agent_task(task)], "agents/utils.py"]
    paths += [name.replace(".", "/") + ".py" for name in sorted(HELPERS)]
    return {p: hashlib.sha256((Path(root) / p).read_bytes()).hexdigest() for p in paths}


def task_or_rubric(row, response):
    task = agent_task(row.get("original_task_id", row["task_id"]))
    if task not in VERIFIABLE:
        from .judges import rubric_judge
        return rubric_judge(row, response)
    info = row.get("evaluator_context", {})
    raw = info.get("original_row")
    if not raw:
        raise ValueError("Original-task scoring needs evaluator_context.original_row")
    root = repository()
    expected = info.get("source_hashes", {})
    actual = source_hashes(root, task)
    if expected and expected != actual:
        raise ValueError("Original prompt/parser sources changed since data export; re-export in a new experiment directory")
    output = asyncio.run(load_source(str(root), MODULES[task]).agent_loop(copy.deepcopy(raw), context(response)))
    score = output["reward"]
    # This is an ORIGINAL TASK metric. Dimension labels control routing only;
    # replicating it into requested score slots is not evidence of disentanglement.
    return {"valid": True, "constraint_pass": True, "score_kind": "original_task",
            "scores": {d: score for d in row["dimensions"]}, "task_metrics": output["metrics"]}
