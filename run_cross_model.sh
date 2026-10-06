#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD:$PWD/framework${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m crossdistill.run "$@"
