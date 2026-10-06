#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD:$PWD/framework${PYTHONPATH:+:$PYTHONPATH}"
exec bash direct_task_equal_opd_ablation/run.sh "$@"
