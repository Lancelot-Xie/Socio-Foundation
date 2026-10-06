"""Dependency-free, read-only bridge to the pinned local tau-bench snapshot.

The upstream package imports optional model clients at package import time.  An
evaluation environment only needs its literal task definitions, JSON database,
tool classes, and reward semantics, so this module loads precisely those pieces
without importing or modifying the upstream package.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import threading
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import ConfigurationError, ValidationError
from ..json_utils import canonical_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAU_BENCH_ROOT = PROJECT_ROOT / "third_party" / "tau-bench"
TERMINATE_TOOL = "transfer_to_human_agents"


@dataclass(frozen=True)
class TauBenchAction:
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class TauBenchTask:
    domain: str
    index: int
    user_id: str
    instruction: str
    actions: tuple[TauBenchAction, ...]
    outputs: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.domain}_{self.index}"


def _literal(node: ast.AST, *, context: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValidationError(f"tau-bench {context} is not a literal") from exc


def _keyword_map(call: ast.Call) -> dict[str, ast.AST]:
    return {item.arg: item.value for item in call.keywords if item.arg is not None}


def _parse_tasks(path: Path, domain: str) -> tuple[TauBenchTask, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    task_list: ast.List | ast.Tuple | None = None
    expected_name = "TASKS" if domain == "airline" else "TASKS_TEST"
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == expected_name for target in node.targets):
            if isinstance(node.value, (ast.List, ast.Tuple)):
                task_list = node.value
            break
    if task_list is None:
        raise ValidationError(f"cannot find literal {expected_name} in {path}")
    tasks: list[TauBenchTask] = []
    for index, node in enumerate(task_list.elts):
        if not isinstance(node, ast.Call):
            raise ValidationError(f"tau-bench {domain} task #{index} is not a Task call")
        fields = _keyword_map(node)
        raw_actions = fields.get("actions")
        if not isinstance(raw_actions, (ast.List, ast.Tuple)):
            raise ValidationError(f"tau-bench {domain} task #{index} actions are not literal")
        actions: list[TauBenchAction] = []
        for action_index, action_node in enumerate(raw_actions.elts):
            if not isinstance(action_node, ast.Call):
                raise ValidationError(
                    f"tau-bench {domain} task #{index} action #{action_index} is not an Action call"
                )
            action_fields = _keyword_map(action_node)
            name = _literal(action_fields["name"], context="action name")
            arguments = _literal(action_fields["kwargs"], context="action kwargs")
            if not isinstance(name, str) or not isinstance(arguments, Mapping):
                raise ValidationError("tau-bench action name/kwargs have invalid types")
            actions.append(TauBenchAction(name=name, arguments=dict(arguments)))
        user_id = _literal(fields["user_id"], context="task user_id")
        instruction = _literal(fields["instruction"], context="task instruction")
        outputs = _literal(fields["outputs"], context="task outputs")
        if (
            not isinstance(user_id, str)
            or not isinstance(instruction, str)
            or not isinstance(outputs, Sequence)
            or isinstance(outputs, (str, bytes))
            or not all(isinstance(value, str) for value in outputs)
        ):
            raise ValidationError(f"tau-bench {domain} task #{index} has invalid literal fields")
        tasks.append(
            TauBenchTask(
                domain=domain,
                index=index,
                user_id=user_id,
                instruction=instruction,
                actions=tuple(actions),
                outputs=tuple(outputs),
            )
        )
    return tuple(tasks)


class _ToolBase:
    @staticmethod
    def invoke(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def get_info() -> Mapping[str, Any]:
        raise NotImplementedError


_TOOL_IMPORT_LOCK = threading.Lock()


def _load_tool_class(path: Path) -> type[_ToolBase]:
    """Load one upstream tool file while satisfying only its Tool base import."""

    module_name = f"_indieval_tau_tool_{path.parent.parent.name}_{path.stem}"
    with _TOOL_IMPORT_LOCK:
        saved = {name: sys.modules.get(name) for name in ("tau_bench", "tau_bench.envs", "tau_bench.envs.tool")}
        tau_package = types.ModuleType("tau_bench")
        tau_package.__path__ = []  # type: ignore[attr-defined]
        env_package = types.ModuleType("tau_bench.envs")
        env_package.__path__ = []  # type: ignore[attr-defined]
        tool_module = types.ModuleType("tau_bench.envs.tool")
        tool_module.Tool = _ToolBase  # type: ignore[attr-defined]
        sys.modules.update(
            {
                "tau_bench": tau_package,
                "tau_bench.envs": env_package,
                "tau_bench.envs.tool": tool_module,
            }
        )
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ConfigurationError(f"cannot load tau-bench tool source {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            for name, previous in saved.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
    classes = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and value is not _ToolBase
        and issubclass(value, _ToolBase)
        and value.__module__ == module_name
    ]
    if len(classes) != 1:
        raise ValidationError(f"expected one tool class in {path}, found {len(classes)}")
    return classes[0]


def _to_hashable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((key, _to_hashable(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_to_hashable(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_to_hashable(item) for item in value))
    return value


def _data_hash(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(str(_to_hashable(data)).encode("utf-8")).hexdigest()


class TauBenchRepository:
    """Validated view over the immutable local tau-bench test runtime."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured_root = root if root is not None else os.environ.get("SIM_EVAL_TAU_BENCH_ROOT", DEFAULT_TAU_BENCH_ROOT)
        self.root = Path(configured_root).expanduser().resolve()
        self.package_root = self.root / "tau_bench"
        if not self.package_root.is_dir():
            raise ConfigurationError(f"tau-bench package not found under {self.root}")

    def _domain_root(self, domain: str) -> Path:
        if domain not in {"airline", "retail"}:
            raise ValidationError("tau-bench domain must be airline or retail")
        return self.package_root / "envs" / domain

    @lru_cache(maxsize=2)
    def tasks(self, domain: str) -> tuple[TauBenchTask, ...]:
        return _parse_tasks(self._domain_root(domain) / "tasks_test.py", domain)

    def task(self, domain: str, index: int) -> TauBenchTask:
        tasks = self.tasks(domain)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(tasks):
            raise ValidationError(f"invalid tau-bench {domain} task index {index!r}")
        return tasks[index]

    @lru_cache(maxsize=2)
    def tool_classes(self, domain: str) -> Mapping[str, type[_ToolBase]]:
        tool_dir = self._domain_root(domain) / "tools"
        values: dict[str, type[_ToolBase]] = {}
        for path in sorted(tool_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            tool_class = _load_tool_class(path)
            info = tool_class.get_info()
            try:
                name = str(info["function"]["name"])
            except (KeyError, TypeError) as exc:
                raise ValidationError(f"invalid tool schema in {path}") from exc
            if name in values:
                raise ValidationError(f"duplicate tau-bench tool name {name!r}")
            values[name] = tool_class
        if not values:
            raise ValidationError(f"no tau-bench tools found for {domain}")
        return values

    def tool_specs(self, domain: str) -> tuple[Mapping[str, Any], ...]:
        result = []
        for name, tool_class in sorted(self.tool_classes(domain).items()):
            raw = copy.deepcopy(dict(tool_class.get_info()))
            function = raw.get("function")
            if not isinstance(function, Mapping):
                raise ValidationError(f"tau-bench tool {name!r} has no function schema")
            parameters = function.get("parameters")
            if not isinstance(parameters, Mapping):
                raise ValidationError(f"tau-bench tool {name!r} has no parameter schema")
            required = parameters.get("required") or []
            result.append(
                {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "parameters": copy.deepcopy(dict(parameters)),
                    "required_arguments": list(required),
                }
            )
        return tuple(result)

    @lru_cache(maxsize=2)
    def policy(self, domain: str) -> str:
        value = (self._domain_root(domain) / "wiki.md").read_text(encoding="utf-8").strip()
        if not value:
            raise ValidationError(f"empty tau-bench policy for {domain}")
        return value

    @lru_cache(maxsize=2)
    def base_data(self, domain: str) -> Mapping[str, Any]:
        data_dir = self._domain_root(domain) / "data"
        names = ("flights", "reservations", "users") if domain == "airline" else ("orders", "products", "users")
        return {
            name: json.loads((data_dir / f"{name}.json").read_text(encoding="utf-8"))
            for name in names
        }

    def fresh_data(self, domain: str) -> dict[str, Any]:
        return copy.deepcopy(dict(self.base_data(domain)))

    @lru_cache(maxsize=1)
    def runtime_digest(self) -> str:
        files: list[Path] = [self.package_root / "envs" / "base.py", self.package_root / "envs" / "tool.py"]
        for domain in ("airline", "retail"):
            root = self._domain_root(domain)
            files.extend([root / "tasks_test.py", root / "wiki.md"])
            files.extend(sorted((root / "tools").glob("*.py")))
            files.extend(sorted((root / "data").glob("*.json")))
        digest = hashlib.sha256()
        for path in sorted(files):
            relative = path.relative_to(self.root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            payload = path.read_bytes()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()


class TauBenchLocalSession:
    """One isolated candidate interaction with official tau-bench reward logic."""

    def __init__(
        self,
        domain: str,
        task_index: int,
        *,
        repository: TauBenchRepository | None = None,
        expected_runtime_digest: str | None = None,
        expected_instruction: str | None = None,
    ) -> None:
        self.repository = repository or TauBenchRepository()
        if expected_runtime_digest and self.repository.runtime_digest() != expected_runtime_digest:
            raise ValidationError("local tau-bench runtime digest differs from the imported evaluation data")
        self.domain = domain
        self.task = self.repository.task(domain, task_index)
        if expected_instruction is not None and self.task.instruction.strip() != expected_instruction.strip():
            raise ValidationError("tau-USI user goal differs from the pinned tau-bench test task")
        self.tools = self.repository.tool_classes(domain)
        self.data = self.repository.fresh_data(domain)
        self.actions: list[TauBenchAction] = []
        self.terminal = False
        self.terminal_reward: float | None = None

    def execute_tool(self, name: str, arguments: Mapping[str, Any]) -> tuple[Any, bool]:
        if self.terminal:
            return "Error: TauBench environment has already terminated.", True
        self.actions.append(TauBenchAction(name=name, arguments=dict(arguments)))
        tool = self.tools.get(name)
        if tool is None:
            # Exact upstream tau-bench Env.step behavior: unknown tool names
            # become observations and the assistant retains the turn.
            return f"Unknown action {name}", False
        try:
            observation = tool.invoke(data=self.data, **dict(arguments))
        except Exception as exc:  # upstream runtime deliberately exposes tool errors as observations
            observation = f"Error: {exc}"
        self.terminal = name == TERMINATE_TOOL
        if self.terminal:
            self.terminal_reward = self.calculate_reward()
        return observation, self.terminal

    def record_assistant_message(self, content: str) -> None:
        if not self.terminal:
            self.actions.append(TauBenchAction(name="respond", arguments={"content": content}))

    def calculate_reward(self) -> float:
        if self.terminal_reward is not None:
            return self.terminal_reward
        candidate_hash = _data_hash(self.data)
        ground_truth_data = self.repository.fresh_data(self.domain)
        for action in self.task.actions:
            if action.name == TERMINATE_TOOL:
                continue
            tool = self.tools.get(action.name)
            if tool is None:
                raise ValidationError(f"ground-truth task names missing tool {action.name!r}")
            try:
                tool.invoke(data=ground_truth_data, **dict(action.arguments))
            except Exception:
                # Match tau-bench Env.step: exceptions become observations while state is retained.
                pass
        if candidate_hash != _data_hash(ground_truth_data):
            return 0.0
        for expected in self.task.outputs:
            if not any(
                action.name == "respond"
                and expected.lower() in str(action.arguments.get("content") or "").lower().replace(",", "")
                for action in self.actions
            ):
                return 0.0
        return 1.0


_SURVEY_ORDINAL: Mapping[str, Mapping[str, int]] = {
    "task_success": {
        "No - Task failed": 0,
        "No - Due to a policy issue, which the agent clearly explained": 1,
        "Partially - Some progress": 2,
        "Yes - Task completed": 3,
        "Fully - Exceeded expectations": 4,
    },
    "efficiency": {
        "Very inefficient - Too many steps": 0,
        "Somewhat inefficient": 1,
        "About right": 2,
        "Very efficient": 3,
    },
    "question_amount_preference": {"Too many": 0, "About right": 1, "Too few": 0},
    "answer_effort_time": {"High": 0, "Medium": 1, "Low": 2},
    "human_like": {"No": 0, "Partially": 1, "Yes": 2},
    "interaction_flow": {"Not smooth": 0, "OK": 1, "Smooth": 2, "Excellent": 3},
    "overall_score": {
        "1 (Very poor)": 0,
        "2 (Poor)": 1,
        "3 (Acceptable)": 2,
        "4 (Good)": 3,
        "5 (Excellent)": 4,
    },
    "reuse": {"Absolutely no": 0, "No": 1, "Maybe": 2, "Yes": 3, "Absolutely yes": 4},
}
_SURVEY_TARGET_NAMES = {
    "question_amount_preference": "question_amount",
    "answer_effort_time": "answer_effort",
    "human_like": "human_likeness",
    "overall_score": "overall",
    "reuse": "reuse_intent",
}


class TauUSIOfficialReferenceStore:
    """Lazy authorized reader; raw human messages are never copied into eval records."""

    def __init__(self, annotation_path: str | Path, *, expected_sha256: str | None = None) -> None:
        self.path = Path(annotation_path).resolve()
        if not self.path.is_file():
            raise ConfigurationError(f"tau-USI annotation file not found: {self.path}")
        actual = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if expected_sha256 and actual.lower() != expected_sha256.lower():
            raise ValidationError(
                f"tau-USI annotation checksum mismatch: expected {expected_sha256}, got {actual}"
            )
        self.sha256 = actual
        self._records: Mapping[str, Any] | None = None
        self._records_lock = threading.Lock()

    def _load(self) -> Mapping[str, Any]:
        if self._records is None:
            with self._records_lock:
                if self._records is None:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    if not isinstance(raw, Mapping):
                        raise ValidationError("tau-USI annotation file must contain an object")
                    self._records = raw
        return self._records

    @staticmethod
    def _user_messages(conversation: Any) -> list[str]:
        if not isinstance(conversation, Sequence) or isinstance(conversation, (str, bytes)):
            raise ValidationError("tau-USI conversation must be an array")
        messages = []
        for item in conversation:
            if not isinstance(item, Mapping) or item.get("role") != "user":
                continue
            content = str(item.get("content") or "").strip()
            if not content or content.startswith(("\\tau", "\\reward", "/stop")):
                continue
            if "<|canvas|>" in content or "<|highlight|>" in content or "<|survey|>" in content:
                continue
            messages.append(content)
        return messages

    @staticmethod
    def _survey(raw: Any) -> Mapping[str, int]:
        if not isinstance(raw, Mapping):
            raise ValidationError("tau-USI survey must be an object")
        result: dict[str, int] = {}
        for source_name, values in _SURVEY_ORDINAL.items():
            item = raw.get(source_name)
            answer = item.get("answer") if isinstance(item, Mapping) else None
            if answer not in values:
                raise ValidationError(f"unknown tau-USI survey answer for {source_name}: {answer!r}")
            result[_SURVEY_TARGET_NAMES.get(source_name, source_name)] = values[str(answer)]
        return result

    def references_for_ids(self, reference_ids: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
        records = self._load()
        result = []
        for index, reference_id in enumerate(reference_ids):
            raw = records.get(reference_id)
            if not isinstance(raw, Mapping):
                raise ValidationError(f"tau-USI reference ID not found: {reference_id!r}")
            reward = raw.get("reward")
            if isinstance(reward, bool):
                reward = float(reward)
            if not isinstance(reward, (int, float)) or not 0 <= float(reward) <= 1:
                raise ValidationError(f"invalid tau-USI reward for {reference_id!r}")
            result.append(
                {
                    "batch_id": f"human_batch_{index + 1}",
                    "user_messages": self._user_messages(raw.get("conversation")),
                    "survey": self._survey(raw.get("survey")),
                    "reward": float(reward),
                }
            )
        return tuple(result)

    def references_for_case(self, case: Any) -> tuple[Mapping[str, Any], ...]:
        evaluation = case.metadata.get("evaluation")
        ids = evaluation.get("human_reference_ids") if isinstance(evaluation, Mapping) else None
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or len(ids) != 3:
            raise ValidationError("tau-USI case requires exactly three human_reference_ids")
        return self.references_for_ids(tuple(str(value) for value in ids))


__all__ = [
    "DEFAULT_TAU_BENCH_ROOT",
    "TauBenchAction",
    "TauBenchLocalSession",
    "TauBenchRepository",
    "TauBenchTask",
    "TauUSIOfficialReferenceStore",
]
