#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  plan|prepare|trajectories|dimensions|fusion|train) ;;
  *) echo 'Usage: bash run_hierarchy.sh [plan|prepare|trajectories|dimensions|fusion|train] [options]' >&2; exit 2 ;;
esac

export PYTHONPATH="$PWD:$PWD/framework${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python}" -m opd.hierarchy "$mode" \
  --config "${HIERARCHY_CONFIG:-configs/hierarchy_five.yaml}" \
  --output "${HIERARCHY_OUTPUT:-outputs/hierarchy}" \
  --num-processes "${NUM_PROCESSES:-8}" \
  --dimension-steps "${DIMENSION_STEPS:-200}" \
  --fusion-steps "${FUSION_STEPS:-1000}" \
  "$@"
