# Anonymous simulation code

This folder contains code and launch scripts for mid-training, task-level reinforcement learning, capability-expert distillation, and fusion ablations. It contains no datasets, model weights, checkpoints, generated samples, or experiment results.

## Install

Use Python 3.10+ and a CUDA environment compatible with the chosen PyTorch, vLLM, and distributed-training versions. From this folder:

```bash
python -m pip install -e ./framework
python -m pip install -e '.[parquet,objective]'
```

The framework's optional GPU backends must be installed for the selected launch mode. A full training run requires GPUs and externally supplied model weights and datasets.
The launch scripts use console logging and disable W&B by default.

## Inputs

- Set `SFT_BASE` to the common student/teacher base model or SFT checkpoint.
- Set `OPD_DATA_ROOT` to the post-training dataset directory containing `train/` and `test/` Parquet files.
- Set `OPD_EXPERT_ROOT` to a directory with one task adapter per entry in `configs/task_experts.yaml`. Update the relative `adapter` entries if your checkpoint layout differs.
- For rubric-scored tasks, set `OPD_JUDGE_BASE_URL`, `OPD_JUDGE_MODEL`, and `OPD_JUDGE_API_KEY` for an OpenAI-compatible judge.
- For mid-training, set `DATA_DIR` to the mid-training corpus and `ACTOR_MODEL_PATH` to the initial model. For task RL, set `DATA_DIR` to the post-training corpus and `ACTOR_MODEL_PATH` to the policy initialization.

## Run

```bash
bash run_sft.sh
bash run_rl.sh lifechoices
bash run_hierarchy.sh plan
bash run_hierarchy.sh prepare
bash run_hierarchy.sh train
```

The hierarchy stages are `prepare`, `trajectories`, `dimensions`, and `fusion`; `train` runs them in order. `plan` checks the configuration without starting training. Outputs are written under `outputs/` by default.

Optional ablations:

```bash
OPD_DATA_SOURCE=outputs/hierarchy bash run_equal_weight.sh
python -m fusion_ablation --help
bash run_cross_model.sh --help
```

The cross-model ablation reads a completed hierarchy run. Supply the model locations with its CLI options. The framework is included locally under `framework/`; no sibling repository is required.
