#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

if [[ -n "${OPD_PYTHON:-}" ]]; then
  PYTHON_BIN="$OPD_PYTHON"
elif [[ -x .venv-linux/bin/python ]]; then
  PYTHON_BIN="$PROJECT/.venv-linux/bin/python"
else
  PYTHON_BIN=python
fi

export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

exec "$PYTHON_BIN" direct_task_equal_opd_ablation/run.py \
  --data-source "${OPD_DATA_SOURCE:?Set OPD_DATA_SOURCE to the existing hierarchy output for prompts}" \
  --expert-manifest "${OPD_EXPERT_MANIFEST:-$PROJECT/configs/task_experts.yaml}" \
  --output "${OPD_OUTPUT:-$PROJECT/outputs/direct_task_equal_opd_v1}" \
  --seed "${OPD_SEED:-42}" \
  --train-per-task "${OPD_TRAIN_PER_TASK:-64}" \
  --validation-per-task "${OPD_VALIDATION_PER_TASK:-8}" \
  --steps "${OPD_STEPS:-100}" \
  --num-processes "${OPD_PROCESSES:-8}" \
  --micro-batch "${OPD_MICRO_BATCH:-1}" \
  --global-batch "${OPD_GLOBAL_BATCH:-16}" \
  --learning-rate "${OPD_LEARNING_RATE:-2e-6}" \
  --save-every "${OPD_SAVE_EVERY:-50}" \
  --max-new-tokens "${OPD_MAX_NEW_TOKENS:-256}" \
  "$@"
