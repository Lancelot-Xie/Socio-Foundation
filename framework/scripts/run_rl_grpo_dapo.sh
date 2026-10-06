#!/bin/bash
# =============================================================================
# Parameterized FoldGRPO launcher with a complete DAPO mode.
#
# Override GPU count, context length, and batch settings for the target hardware.
#
# This launcher provides two options:
#   1. Drop RLVF  ->  agent_version=default  (vanilla GRPO, no verbal feedback)
#   2. Run DAPO's Clip-Higher, Dynamic Sampling, token-level loss, and
#      Overlong Reward Shaping on Simulation's agent-loop rewards.
#
# ── Clean baseline defaults ──────────────────────────────────────────────────
#   [BASE] Symmetric clip         clip_ratio_low/high=0.2
#   [BASE] Conservative dual clip clip_ratio_c=3.0
#   [BASE] Sequence-balanced loss actor.loss_agg_mode=seq-mean-token-mean
#   [ON ] Dynamic batch          actor.use_dynamic_bsz=True
#   [ON ] Overlong PROMPT filter data.filter_overlong_prompts=True
#
# ── DAPO mode ──────────────────────────────────────────────────────────────────
# DAPO is enabled by default. Set DAPO_ENABLED=false for the clean FoldGRPO
# control. Dynamic Sampling de-duplicates multi-agent sub-sequences by gen_uid
# and accumulates generation batches when one batch is insufficient.
#
# Usage:
#   bash scripts/run_rl_grpo_dapo.sh fantom
#   TASK=hitom ACTOR_MODEL_PATH=models/policy bash scripts/run_rl_grpo_dapo.sh
#
# DATA_DIR contains train/ and test/ Parquet files. Their locations may also
# be supplied explicitly through TRAIN_FILES and VAL_FILES.
# =============================================================================
set -e

# Run from the repo root regardless of where this script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Task ──────────────────────────────────────────────────────────────────────
TASK="${1:-${TASK:-fantom}}"   # default: a verifiable task (no judge API needed)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo-dapo-${TASK}}"
data_dir="${DATA_DIR:-data}"
rl_dir="$data_dir/train"
val_dir="$data_dir/test"

# TASK -> (train file, val file) — matches the HF dataset filenames.
case "$TASK" in
  sotopia)            train_rel=sotopia_clean_rl.parquet;        val_rel=sotopia_hard_val.parquet ;;
  coser)              train_rel=coser_rl_train.parquet;          val_rel=coser_val.parquet ;;
  lifechoices)        train_rel=lifechoices_hard_rl.parquet;     val_rel=lifechoices_val.parquet ;;
  userllm)            train_rel=userllm_rl_train.parquet;        val_rel=userllm_val.parquet ;;
  mirrorbench)        train_rel=mirrorbench_rl_train.parquet;    val_rel=mirrorbench_val.parquet ;;
  fantom)             train_rel=fantom_rl_train.parquet;         val_rel=fantom_val.parquet ;;
  hitom)              train_rel=hitom_rl_train.parquet;          val_rel=hitom_val.parquet ;;
  paratomi)           train_rel=paratomi_rl_train.parquet;       val_rel=paratomi_val.parquet ;;
  mistakes)           train_rel=mistakes_rl_train.parquet;       val_rel=mistakes_val.parquet ;;
  twinvoice)          train_rel=twinvoice_rl_train.parquet;      val_rel=twinvoice_val.parquet ;;
  social_r1)          train_rel=social_r1_rl.parquet;            val_rel=social_r1_val.parquet ;;
  behaviorchain)      train_rel=behaviorchain_rl_train.parquet;  val_rel=behaviorchain_val.parquet ;;
  sim_math)           train_rel=sim_math_rl.parquet;             val_rel=sim_math_val.parquet ;;
  sim_doc)            train_rel=sim_doc_rl.parquet;              val_rel=sim_doc_val.parquet ;;
  humanual_book)      train_rel=humanual_rl_book.parquet;        val_rel=humanual_book_val.parquet ;;
  humanual_chat)      train_rel=humanual_rl_chat.parquet;        val_rel=humanual_chat_val.parquet ;;
  humanual_email)     train_rel=humanual_rl_email.parquet;       val_rel=humanual_email_val.parquet ;;
  humanual_news)      train_rel=humanual_rl_news.parquet;        val_rel=humanual_news_val.parquet ;;
  humanual_opinion)   train_rel=humanual_rl_opinion.parquet;     val_rel=humanual_opinion_val.parquet ;;
  humanual_politics)  train_rel=humanual_rl_politics.parquet;    val_rel=humanual_politics_val.parquet ;;
  alignx)             train_rel=alignx_rl_8k.parquet;            val_rel=alignx_demo_val.parquet ;;
  socsci210)          train_rel=socsci210_rl_2k.parquet;         val_rel=socsci210_val.parquet ;;
  humanllm)           train_rel=humanllm_rl_train.parquet;       val_rel=humanllm_val.parquet ;;
  *) echo "Unknown TASK: $TASK" >&2; exit 1 ;;
esac

train_files="${TRAIN_FILES:-$rl_dir/$train_rel}"
val_files="${VAL_FILES:-$val_dir/$val_rel}"

# RL here expects an instruction-following or Simulation-midtrained checkpoint.
# Do not silently choose a machine-specific continued-pretraining checkpoint:
# model initialization is part of the experiment, not an implementation detail.
if [[ -z "${ACTOR_MODEL_PATH:-}" ]]; then
  echo "ACTOR_MODEL_PATH is required. Use the exact instruct/SFT/Simulation-midtrained checkpoint used for this run." >&2
  exit 1
fi
actor_model_path="$ACTOR_MODEL_PATH"
allow_base_model_for_rl="${ALLOW_BASE_MODEL_FOR_RL:-false}"
model_name_lower="$(basename "$actor_model_path" | tr '[:upper:]' '[:lower:]')"
if [[ "$model_name_lower" == *base* || "$model_name_lower" == *stage1a* ]]; then
  if [[ "$allow_base_model_for_rl" != "true" ]]; then
    echo "Refusing likely base/continued-pretraining initialization: $actor_model_path" >&2
    echo "Use an instruct/SFT/Simulation-midtrained checkpoint, or set ALLOW_BASE_MODEL_FOR_RL=true for an intentional ablation." >&2
    exit 1
  fi
fi

# ── GRPO / DAPO knobs ─────────────────────────────────────────────────────────
adv_estimator="${ADV_ESTIMATOR:-foldgrpo}"  # foldgrpo == GRPO + correct multi-agent grouping; use "grpo" to force plain
agent_version="default"                     # <<< pure GRPO: no RLVF/verbal feedback
loss_mode="vanilla"                         # policy-loss variant (not the agg mode)
loss_agg_mode="${LOSS_AGG_MODE:-seq-mean-token-mean}"
clip_ratio_low="${CLIP_LOW:-0.2}"
clip_ratio_high="${CLIP_HIGH:-0.2}"
clip_ratio_c="${CLIP_C:-3.0}"
roleplay_rl_enabled="${ROLEPLAY_RL_ENABLED:-false}"
generic_anchor_enabled="${GENERIC_ANCHOR_ENABLED:-false}"
use_remove_padding="${USE_REMOVE_PADDING:-true}"
use_fused_kernels="${USE_FUSED_KERNELS:-true}"
attn_implementation="${ATTN_IMPLEMENTATION:-flash_attention_2}"
entropy_from_logits_with_chunking="${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-true}"

actor_lr="${ACTOR_LR:-5e-6}"
actor_lr_warmup_steps="${ACTOR_LR_WARMUP_STEPS:-10}"
actor_weight_decay="${ACTOR_WEIGHT_DECAY:-0.1}"

# ========== 新增：KL 惩罚参数（支持环境变量覆盖） ==========
use_kl_in_reward="${USE_KL_IN_REWARD:-false}"
use_kl_loss="${USE_KL_LOSS:-false}"
kl_coef="${KL_COEF:-0.001}"
actor_kl_coef="${ACTOR_KL_COEF:-0.001}"
allow_dual_kl="${ALLOW_DUAL_KL:-false}"
judge_health_enabled="${JUDGE_HEALTH_ENABLED:-true}"
judge_min_valid_fraction="${JUDGE_MIN_VALID_FRACTION:-0.5}"
judge_failure_patience="${JUDGE_FAILURE_PATIENCE:-2}"
dapo_enabled="${DAPO_ENABLED:-true}"
dapo_gen_batch_multiplier="${DAPO_GEN_BATCH_MULTIPLIER:-3}"
dapo_max_num_gen_batches="${DAPO_MAX_NUM_GEN_BATCHES:-20}"
dapo_exhaustion_strategy="${DAPO_EXHAUSTION_STRATEGY:-use_partial}"
dapo_overlong_enabled="${DAPO_OVERLONG_ENABLED:-$dapo_enabled}"
dapo_overlong_buffer_len="${DAPO_OVERLONG_BUFFER_LEN:-1024}"
dapo_overlong_penalty_factor="${DAPO_OVERLONG_PENALTY_FACTOR:-1.0}"
# ===========================================================

if [[ "$dapo_enabled" == "true" ]]; then
  [[ -n "${LOSS_AGG_MODE+x}" ]] || loss_agg_mode="token-mean"
  [[ -n "${CLIP_HIGH+x}" ]] || clip_ratio_high="0.28"
  [[ -n "${CLIP_C+x}" ]] || clip_ratio_c="10.0"
fi

# ── Parameterized sizing ──────────────────────────────────────────────────────
max_prompt_length="${MAX_PROMPT_LEN:-6144}"
max_response_length="${MAX_RESP_LEN:-6144}"
actor_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$(((max_prompt_length + max_response_length) * 2))}"

n_gpus="${N_GPUS:-8}"
n_nodes="${NNODES:-1}"
train_batch_size="${TRAIN_BATCH:-32}"
ppo_mini_batch_size="${PPO_MINI:-8}"
n_resp_per_prompt="${N_RESP:-8}"             # GRPO group size
if [[ "$dapo_enabled" == "true" ]]; then
  gen_batch_size="${GEN_BATCH_SIZE:-$((train_batch_size * dapo_gen_batch_multiplier))}"
else
  gen_batch_size="${GEN_BATCH_SIZE:-$train_batch_size}"
fi
n_resp_per_prompt_val=1
infer_tp="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
agent_num_workers="${AGENT_NUM_WORKERS:-32}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-512}"
rollout_max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}"
rollout_max_model_len="${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length + 512))}"
rollout_free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE:-true}"
rollout_enable_sleep_mode="${ROLLOUT_ENABLE_SLEEP_MODE:-true}"

# vLLM sleep mode uses its CuMem memory pool. PyTorch expandable segments are
# currently incompatible with that pool and make every rollout worker fail at
# engine initialization. Keep any other allocator options supplied by the job,
# but force expandable_segments off before Ray inherits the environment.
if [[ "$rollout_enable_sleep_mode" == "true" && -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  cuda_alloc_entries=()
  cuda_alloc_conf_changed=false
  IFS=',' read -r -a cuda_alloc_conf_entries <<< "$PYTORCH_CUDA_ALLOC_CONF"
  for cuda_alloc_entry in "${cuda_alloc_conf_entries[@]}"; do
    cuda_alloc_key="${cuda_alloc_entry%%:*}"
    cuda_alloc_value="${cuda_alloc_entry#*:}"
    cuda_alloc_key="${cuda_alloc_key//[[:space:]]/}"
    cuda_alloc_value="${cuda_alloc_value//[[:space:]]/}"
    if [[ "${cuda_alloc_key,,}" == "expandable_segments" && ( "${cuda_alloc_value,,}" == "true" || "$cuda_alloc_value" == "1" ) ]]; then
      cuda_alloc_entry="expandable_segments:False"
      cuda_alloc_conf_changed=true
    fi
    cuda_alloc_entries+=("$cuda_alloc_entry")
  done
  if [[ "$cuda_alloc_conf_changed" == "true" ]]; then
    PYTORCH_CUDA_ALLOC_CONF="$(IFS=,; echo "${cuda_alloc_entries[*]}")"
    export PYTORCH_CUDA_ALLOC_CONF
    echo "[WARN] Disabled expandable_segments because ROLLOUT_ENABLE_SLEEP_MODE=true; allocator=$PYTORCH_CUDA_ALLOC_CONF" >&2
  fi
fi

rollout_enforce_eager="${ROLLOUT_ENFORCE_EAGER:-false}"
rollout_layered_summon="${ROLLOUT_LAYERED_SUMMON:-true}"
rollout_calculate_log_probs="${ROLLOUT_CALCULATE_LOG_PROBS:-true}"
rollout_temperature="${ROLLOUT_TEMPERATURE:-1.0}"
rollout_top_p="${ROLLOUT_TOP_P:-1.0}"
rollout_top_k="${ROLLOUT_TOP_K:--1}"
rollout_do_sample="${ROLLOUT_DO_SAMPLE:-true}"
val_temperature="${VAL_TEMPERATURE:-0.0}"
val_top_p="${VAL_TOP_P:-1.0}"
val_top_k="${VAL_TOP_K:--1}"
val_do_sample="${VAL_DO_SAMPLE:-false}"
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-8}"
fsdp_param_offload="${FSDP_PARAM_OFFLOAD:-true}"
fsdp_optimizer_offload="${FSDP_OPTIMIZER_OFFLOAD:-true}"

use_lora="${USE_LORA:-1}"                   # LoRA recommended on 24GB; set 0 for full-param
lora_rank="${LORA_RANK:-32}"
lora_alpha="${LORA_ALPHA:-64}"
lora_adapter_path="${LORA_ADAPTER_PATH:-}"

total_steps="${TOTAL_STEPS:-200}"
save_freq="${SAVE_FREQ:-50}"
test_freq="${TEST_FREQ:-50}"
val_before_train="${VAL_BEFORE_TRAIN:-true}"
resume_mode="${RESUME_MODE:-auto}"
checkpoint_save_contents="${CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"extra\"]}"
max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
trainer_project_name="${TRAINER_PROJECT_NAME:-grpo-dapo}"

if (( max_prompt_length + max_response_length > rollout_max_model_len )); then
  echo "MAX_PROMPT_LEN + MAX_RESP_LEN exceeds ROLLOUT_MAX_MODEL_LEN" >&2
  exit 1
fi
if [[ "$dapo_enabled" == "true" ]]; then
  (( n_resp_per_prompt >= 2 )) || { echo "DAPO requires N_RESP >= 2" >&2; exit 1; }
  (( gen_batch_size >= train_batch_size )) || { echo "GEN_BATCH_SIZE must be >= TRAIN_BATCH" >&2; exit 1; }
  (( dapo_gen_batch_multiplier >= 1 )) || { echo "DAPO_GEN_BATCH_MULTIPLIER must be >= 1" >&2; exit 1; }
  [[ "$dapo_exhaustion_strategy" == "error" || "$dapo_exhaustion_strategy" == "use_partial" ]] || {
    echo "DAPO_EXHAUSTION_STRATEGY must be error or use_partial" >&2
    exit 1
  }
  (( dapo_overlong_buffer_len > 0 && dapo_overlong_buffer_len <= max_response_length )) || {
    echo "DAPO_OVERLONG_BUFFER_LEN must be in [1, MAX_RESP_LEN]" >&2
    exit 1
  }
fi

# ── Setup ─────────────────────────────────────────────────────────────────────
export HF_HOME=$OUTPUT_DIR/hf_cache
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p "$OUTPUT_DIR/hf_cache" "$OUTPUT_DIR/$EXPERIMENT_NAME"

export SIMULATION_THINKING_MODE=off
export ALLOW_BASE_MODEL_FOR_RL="$allow_base_model_for_rl"

export TURNOFF_THINK=1
export LOGGING_LEVEL=ERROR

if [[ "$use_kl_in_reward" == "true" && "$use_kl_loss" == "true" && "$allow_dual_kl" != "true" ]]; then
  echo "Both reward KL and actor KL loss are enabled. Set ALLOW_DUAL_KL=true only for an intentional ablation." >&2
  exit 1
fi

# LoRA vs full-param arg block
if [ "$use_lora" = "1" ]; then
  lora_args=(
    actor_rollout_ref.model.lora_rank=$lora_rank
    actor_rollout_ref.model.lora_alpha=$lora_alpha
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=$rollout_layered_summon
  )
  if [ -n "$lora_adapter_path" ]; then
    lora_args+=(actor_rollout_ref.model.lora_adapter_path="$lora_adapter_path")
  fi
else
  lora_args=()
fi

echo "[INFO] KL config: use_kl_in_reward=$use_kl_in_reward, reward_kl_coef=$kl_coef, use_kl_loss=$use_kl_loss, actor_kl_coef=$actor_kl_coef"
echo "[INFO] Thinking mode=$SIMULATION_THINKING_MODE (formal default: fast/non-thinking)"
echo "[INFO] DAPO: enabled=$dapo_enabled gen_batch=$gen_batch_size dynamic_sampling=$dapo_enabled max_gen_batches=$dapo_max_num_gen_batches exhaustion=$dapo_exhaustion_strategy overlong=$dapo_overlong_enabled buffer=$dapo_overlong_buffer_len penalty=$dapo_overlong_penalty_factor"
echo "[INFO] Validation sampling: do_sample=$val_do_sample temperature=$val_temperature top_p=$val_top_p top_k=$val_top_k"
echo "[INFO] Optimizer: lr=$actor_lr warmup_steps=$actor_lr_warmup_steps weight_decay=$actor_weight_decay"
echo "[INFO] TOTAL_STEPS=$total_steps"
echo "[INFO] Entropy chunking=$entropy_from_logits_with_chunking"

NCCL_DEBUG=WARN python3 scripts/train_ppo_tf5.py \
  hydra.run.dir=$OUTPUT_DIR/hydra \
  algorithm.adv_estimator=$adv_estimator \
  algorithm.use_kl_in_reward=$use_kl_in_reward \
  algorithm.filter_groups.enable=$dapo_enabled \
  algorithm.filter_groups.metric=seq_reward \
  algorithm.filter_groups.max_num_gen_batches=$dapo_max_num_gen_batches \
  algorithm.filter_groups.exhaustion_strategy=$dapo_exhaustion_strategy \
  algorithm.dapo_overlong.enabled=$dapo_overlong_enabled \
  algorithm.dapo_overlong.buffer_length=$dapo_overlong_buffer_len \
  algorithm.dapo_overlong.penalty_factor=$dapo_overlong_penalty_factor \
  algorithm.dapo_overlong.log=True \
  algorithm.roleplay_rl.enabled=$roleplay_rl_enabled \
  algorithm.roleplay_rl.generic_anchor.enabled=$generic_anchor_enabled \
  +algorithm.agent_version=$agent_version \
  algorithm.kl_ctrl.kl_coef=$kl_coef \
  reward_model.enable=False \
  reward_model.launch_reward_fn_async=False \
  +algorithm.judge_health.enabled=$judge_health_enabled \
  +algorithm.judge_health.min_valid_fraction=$judge_min_valid_fraction \
  +algorithm.judge_health.failure_patience=$judge_failure_patience \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=agents/agents.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=agent_hub \
  actor_rollout_ref.rollout.agent.num_workers=$agent_num_workers \
  data.train_files="$train_files" \
  data.val_files="$val_files" \
  data.train_batch_size=$train_batch_size \
  data.gen_batch_size=$gen_batch_size \
  data.dataloader_num_workers=$dataloader_num_workers \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=$actor_model_path \
  actor_rollout_ref.model.use_remove_padding=$use_remove_padding \
  +actor_rollout_ref.model.override_config.attn_implementation=$attn_implementation \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=$use_fused_kernels \
  "${lora_args[@]}" \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=$entropy_from_logits_with_chunking \
  actor_rollout_ref.actor.optim.lr=$actor_lr \
  actor_rollout_ref.actor.optim.lr_warmup_steps=$actor_lr_warmup_steps \
  actor_rollout_ref.actor.optim.weight_decay=$actor_weight_decay \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
  actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
  actor_rollout_ref.actor.kl_loss_coef=$actor_kl_coef \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
  actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
  actor_rollout_ref.actor.clip_ratio_c=$clip_ratio_c \
  actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
  actor_rollout_ref.actor.fsdp_config.param_offload=$fsdp_param_offload \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=$fsdp_optimizer_offload \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.checkpoint.save_contents="$checkpoint_save_contents" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
  actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_memory_utilization \
  actor_rollout_ref.rollout.free_cache_engine=$rollout_free_cache_engine \
  +actor_rollout_ref.rollout.enable_sleep_mode=$rollout_enable_sleep_mode \
  actor_rollout_ref.rollout.enforce_eager=$rollout_enforce_eager \
  actor_rollout_ref.rollout.calculate_log_probs=$rollout_calculate_log_probs \
  actor_rollout_ref.rollout.temperature=$rollout_temperature \
  actor_rollout_ref.rollout.top_p=$rollout_top_p \
  actor_rollout_ref.rollout.top_k=$rollout_top_k \
  actor_rollout_ref.rollout.do_sample=$rollout_do_sample \
  actor_rollout_ref.rollout.max_model_len=$rollout_max_model_len \
  actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_num_batched_tokens \
  actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
  actor_rollout_ref.rollout.n=$n_resp_per_prompt \
  actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
  actor_rollout_ref.rollout.val_kwargs.top_p=$val_top_p \
  actor_rollout_ref.rollout.val_kwargs.top_k=$val_top_k \
  actor_rollout_ref.rollout.val_kwargs.do_sample=$val_do_sample \
  actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
  trainer.n_gpus_per_node=$n_gpus \
  trainer.nnodes=$n_nodes \
  trainer.logger='["console"]' \
  trainer.project_name=$trainer_project_name \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.val_before_train=$val_before_train \
  trainer.save_freq=$save_freq \
  trainer.resume_mode=$resume_mode \
  trainer.max_actor_ckpt_to_keep=$max_actor_ckpt_to_keep \
  trainer.default_local_dir=$OUTPUT_DIR/$EXPERIMENT_NAME \
  trainer.test_freq=$test_freq \
  trainer.total_training_steps=$total_steps \
  trainer.total_epochs=10000
