#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${OPD_PYTHON:-$project_dir/.venv-linux/bin/python}"
if [[ ! -x "$python_bin" && -z "${OPD_PYTHON:-}" ]]; then python_bin=python; fi
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m fusion_ablation run "$@"
