#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/framework"
exec bash scripts/run_rl_grpo_dapo.sh "$@"
