#!/bin/bash
set -e

ROLEPLAY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROLEPLAY_DIR/../.." && pwd)"
TASK="${1:-${TASK:-sotopia}}"
ENABLE_EXPERIMENTAL_ROLEPLAY_RL="${ENABLE_EXPERIMENTAL_ROLEPLAY_RL:-false}"
DAPO_ENABLED="${DAPO_ENABLED:-true}"

# Default to the FoldGRPO baseline. The CRPO/adaptive
# reward/anchor stack changes both the reward and the advantage estimator, so
# it is only suitable for an explicitly named ablation.
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-foldgrpo}"
export ROLEPLAY_RL_ENABLED="${ROLEPLAY_RL_ENABLED:-false}"
export GENERIC_ANCHOR_ENABLED="${GENERIC_ANCHOR_ENABLED:-false}"
export CLIP_LOW="${CLIP_LOW:-0.2}"

if [[ "$DAPO_ENABLED" == "true" ]]; then
  export CLIP_HIGH="${CLIP_HIGH_DAPO:-${CLIP_HIGH:-0.28}}"
  export CLIP_C="${CLIP_C_DAPO:-${CLIP_C:-10.0}}"
  export LOSS_AGG_MODE="${LOSS_AGG_MODE_DAPO:-${LOSS_AGG_MODE:-token-mean}}"
else
  export CLIP_HIGH="${CLIP_HIGH:-0.2}"
  export CLIP_C="${CLIP_C:-3.0}"
  export LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"
fi

if [[ "$ENABLE_EXPERIMENTAL_ROLEPLAY_RL" == "true" ]]; then
  case "$TASK" in
    sotopia|coser|mirrorbench|humanual_book|humanual_chat|humanual_email|humanual_news|humanual_opinion|humanual_politics|userllm|sim_math|sim_doc)
      export ADV_ESTIMATOR="${ADV_ESTIMATOR_EXPERIMENTAL:-crpo_foldgrpo}"
      export ROLEPLAY_RL_ENABLED=true
      export GENERIC_ANCHOR_ENABLED="${GENERIC_ANCHOR_ENABLED_EXPERIMENTAL:-true}"
      export CLIP_HIGH="${CLIP_HIGH_EXPERIMENTAL:-0.28}"
      export CLIP_C="${CLIP_C_EXPERIMENTAL:-10.0}"
      export LOSS_AGG_MODE="${LOSS_AGG_MODE_EXPERIMENTAL:-token-mean}"
      ;;
  esac
fi

echo "[INFO] Recipe: task=$TASK adv=$ADV_ESTIMATOR loss_agg=$LOSS_AGG_MODE clip=$CLIP_LOW/$CLIP_HIGH/$CLIP_C dapo=$DAPO_ENABLED experimental_roleplay_rl=$ENABLE_EXPERIMENTAL_ROLEPLAY_RL"

# Judge-task defaults favor lower variance and keep off-target terms out of the
# training reward. Every value remains environment-overridable.
case "$TASK" in
  mirrorbench)
    export MIRRORBENCH_JUDGE_SAMPLES="${MIRRORBENCH_JUDGE_SAMPLES:-3}"
    export MIRRORBENCH_LEXICAL_WEIGHT="${MIRRORBENCH_LEXICAL_WEIGHT:-0.10}"
    export MIRRORBENCH_PARSEFAIL="${MIRRORBENCH_PARSEFAIL:-0.5}"
    ;;
  coser)
    export COSER_ROUGE_WEIGHT="${COSER_ROUGE_WEIGHT:-0.05}"
    ;;
  humanual_*)
    export HUMANUAL_MASK_SILENT="${HUMANUAL_MASK_SILENT:-1}"
    export HUMANUAL_LENFACTOR="${HUMANUAL_LENFACTOR:-1}"
    export HUMANUAL_COPY_PENALTY="${HUMANUAL_COPY_PENALTY:-0.05}"
    ;;
  sim_doc)
    export SIMDOC_INCLUDE_DOCRATING="${SIMDOC_INCLUDE_DOCRATING:-0}"
    ;;
esac

exec bash "$REPO_ROOT/scripts/run_rl_grpo_dapo.sh" "$TASK"
