"""Launch original VERL agent_hub evaluation, including real multi-agent episodes.

This entry point runs on the target training host with its original VERL/vLLM
environment. It intentionally does not substitute static rubric scores for the
original task metrics. No training updates are enabled.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .source_data import FILES
from .upstream import canonical_task


def command(repo, data, model, output, tasks, python, gpus=8, tensor_parallel=1, max_tokens=6144):
    repo, data, output = Path(repo), Path(data), Path(output)
    files = []
    for task in tasks:
        task = canonical_task(task)
        if task == "alignx":
            files.extend(data / "test" / f"alignx_{s}_val.parquet" for s in ("demo", "pair", "ugc", "arbitrary", "history16"))
        else:
            files.append(data / "test" / (FILES[task][1] + ".parquet"))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    entry = repo / "scripts/train_ppo_tf5.py"
    if not entry.is_file():
        raise FileNotFoundError(f"Original audited training/evaluation entry point missing: {entry}")
    if gpus < 1 or tensor_parallel < 1 or gpus % tensor_parallel:
        raise ValueError("gpus must be a positive multiple of tensor_parallel")
    paths = json.dumps(list(map(str, files)))
    return [python, str(entry),
            "algorithm.adv_estimator=foldgrpo",
            "actor_rollout_ref.rollout.agent.agent_loop_config_path=agents/agents.yaml",
            "actor_rollout_ref.rollout.agent.default_agent_loop=agent_hub",
            "data.train_files=" + paths, "data.val_files=" + paths,
            "data.train_batch_size=8", f"data.max_prompt_length={max_tokens}",
            f"data.max_response_length={max_tokens}", "data.filter_overlong_prompts=True", "data.truncation=error",
            "actor_rollout_ref.model.path=" + json.dumps(str(model)),
            "actor_rollout_ref.model.use_remove_padding=True", "actor_rollout_ref.model.use_fused_kernels=True",
            "actor_rollout_ref.actor.ppo_mini_batch_size=8", "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.actor.fsdp_config.param_offload=True",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True", "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel}",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
            f"actor_rollout_ref.rollout.max_model_len={2*max_tokens+1024}",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={2*max_tokens}",
            "actor_rollout_ref.rollout.max_num_seqs=64", "actor_rollout_ref.rollout.log_prob_micro_batch_size=1",
            "actor_rollout_ref.rollout.val_kwargs.n=1", "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
            "actor_rollout_ref.rollout.val_kwargs.temperature=0.0",
            "+actor_rollout_ref.rollout.agent.max_concurrent_rollouts=32",
            "algorithm.use_kl_in_reward=False", f"trainer.n_gpus_per_node={gpus}", "trainer.nnodes=1",
            'trainer.logger=["console"]', "trainer.project_name=evaluation",
            "trainer.experiment_name=original-task-eval", "trainer.val_before_train=True", "trainer.val_only=True",
            "trainer.default_local_dir=" + json.dumps(str(output)),
            "trainer.validation_data_dir=" + json.dumps(str(output / "generations")),
            "trainer.total_training_steps=1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("repo", "data-root", "model", "output"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--base", help="Required for merging a dimension/task adapter before evaluation")
    parser.add_argument("--upstream-python", default=os.environ.get("SIMULATION_PYTHON", sys.executable))
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    repo, data = Path(args.repo).resolve(), Path(args.data_root).resolve()
    model = Path(args.model).resolve()
    if not model.is_dir():
        raise FileNotFoundError(model)
    output.mkdir(parents=True, exist_ok=True)
    is_adapter = any((model / p / "adapter_config.json").is_file()
                     for p in (".", "actor/lora_adapter", "lora_adapter"))
    merge_cmd = None
    if is_adapter:
        if not args.base:
            raise ValueError("--base must be the original shared SFT checkpoint for adapter evaluation")
        merged = output / "merged_model"
        merge_cmd = [sys.executable, "-m", "opd", "merge-adapter", "--base", args.base,
                     "--adapter", str(model), "--output", str(merged), "--dtype", "bfloat16"]
        model = merged
    cmd = command(repo, data, model, output, args.tasks, args.upstream_python,
                  args.gpus, args.tensor_parallel, args.max_tokens)
    (output / "command.json").write_text(json.dumps({"merge": merge_cmd, "evaluate": cmd}, indent=2))
    if args.dry_run:
        print(json.dumps({"command_file": str(output / "command.json"), "dry_run": True}))
        return
    if (output / "complete.json").exists():
        raise FileExistsError("Original evaluation already completed; use its outputs or a new directory")
    if merge_cmd and not (model / "config.json").exists():
        subprocess.run(merge_cmd, check=True)
    env = {**os.environ, "SIMULATION_THINKING_MODE": "off", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONPATH": str(repo) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.pop("ACCELERATE_USE_CPU", None)
    with (output / "original_eval.log").open("a") as log:
        subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (output / "complete.json").write_text(json.dumps({"model": str(model), "tasks": args.tasks,
                                                     "scope": "original VERL task agent evaluation"}, indent=2))


if __name__ == "__main__":
    main()
