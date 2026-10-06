#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash fusion_ablation/run_all.sh SOURCE_HIERARCHY OUTPUT_ROOT [shared options]' >&2
  exit 2
fi
source_run="$1"
output_root="$2"
shift 2
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for method in dimension_offpolicy task_sft merge_dimensions merge_tasks; do
  method_output="$output_root/$method"
  echo "[fusion-ablation] starting method=$method output=$method_output"
  OPD_JUDGE_CACHE_DIR="$method_output/judge_cache" \
    bash "$script_dir/run.sh" --source "$source_run" --output "$method_output" --method "$method" "$@"
  echo "[fusion-ablation] completed method=$method"
done
echo "[fusion-ablation] all four methods completed"
