#!/usr/bin/env bash
#
# MED-R1 DAPO training via verl 0.5.0 (single-node, 8 GPUs).
#
# Architecture:
#   cuda:0-6 → verl training (actor/rollout/ref, TP=1, 7 DP groups)
#   cuda:7   → reward_server.py (PRM 4B + Judge 7B, ~22 GB bf16)
#
# The reward function (medical_reward.py) makes HTTP calls to the reward
# server, keeping the verl TaskRunner stateless.
#
# Prerequisites:
#   1. Activate verl environment:
#      source verl_env/bin/activate
#   2. Copy models to local disk:
#      cp -r ckpts/sft_stage1/checkpoint-1-3168/tfmr /tmp/sft_model
#      cp -r ckpts/prm_qwen3_4b/final /tmp/prm_qwen3_4b
#   3. Convert data:
#      python convert_data_to_parquet.py
#   4. Run:
#      bash run_verl_dapo.sh [MODEL_PATH]
#
# Resume (auto): 再次运行同一命令即可，verl 自动从最新 checkpoint 续训
# Resume (manual):
#      RESUME_PATH=./ckpts/dapo/global_step_100 bash run_verl_dapo.sh
#
# Logs:
#   logs/verl_dapo.log        - verl training
#   logs/reward_server.log    - PRM + Judge service
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH="${1:-/tmp/sft_model}"
DATA_DIR="data/verl"
TRAIN_FILE="${DATA_DIR}/medical_o1_train.parquet"
EVAL_FILE="${DATA_DIR}/medical_o1_eval.parquet"
OUTPUT_DIR="./ckpts/dapo"
LOG_DIR="logs"
RESUME_PATH="${RESUME_PATH:-}"
REWARD_FN_PATH="${SCRIPT_DIR}/verl_reward/medical_reward.py"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Data not found. Converting..."
    python convert_data_to_parquet.py --output_dir "$DATA_DIR"
fi

# ---------------------------------------------------------------------------
# Reward service configuration
#
# PRM (4B) + Judge (7B) share one GPU in a single process (~22 GB bf16).
# ---------------------------------------------------------------------------
PRM_MODEL_PATH="${PRM_MODEL_PATH:-/tmp/prm_qwen3_4b}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/tmp/sft_model}"
REWARD_PORT=8100

export REWARD_SERVICE_URL="http://localhost:${REWARD_PORT}"

# ---------------------------------------------------------------------------
# Cleanup: kill reward server on EXIT
# ---------------------------------------------------------------------------
REWARD_PID=""
VERL_PID=""

cleanup() {
    echo ""
    echo "Shutting down all processes..."
    [ -n "$VERL_PID"   ] && kill "$VERL_PID"   2>/dev/null && echo "  verl training stopped."
    [ -n "$REWARD_PID" ] && kill "$REWARD_PID" 2>/dev/null && echo "  Reward server stopped."
    wait 2>/dev/null
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helper: wait for service health
# ---------------------------------------------------------------------------
wait_for_service() {
    local url=$1
    local name=$2
    local max_wait=600   
    local waited=0
    echo "Waiting for ${name} at ${url}/health ..."
    while [ "$waited" -lt "$max_wait" ]; do
        if curl -sf "${url}/health" > /dev/null 2>&1; then
            echo "  ${name} is ready (${waited}s)"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 30)) -eq 0 ]; then
            echo "  Still waiting for ${name}... (${waited}s)"
        fi
    done
    echo "ERROR: ${name} did not start within ${max_wait}s"
    return 1
}

# ---------------------------------------------------------------------------
# 1. Start combined reward server (cuda:7)
# ---------------------------------------------------------------------------
pkill -9 -f "reward_server.py" 2>/dev/null || true
sleep 2

echo "Starting reward server on GPU 7 (PRM + Judge, port ${REWARD_PORT})..."
CUDA_VISIBLE_DEVICES=7 nohup python -u "${SCRIPT_DIR}/verl_reward/reward_server.py" \
    --prm_path "$PRM_MODEL_PATH" \
    --judge_path "$JUDGE_MODEL_PATH" \
    --port "$REWARD_PORT" \
    > "$LOG_DIR/reward_server.log" 2>&1 &
REWARD_PID=$!
echo "  Reward server PID: $REWARD_PID"

# ---------------------------------------------------------------------------
# 2. Wait for reward server to be healthy
# ---------------------------------------------------------------------------
wait_for_service "http://localhost:${REWARD_PORT}" "Reward server"

# ---------------------------------------------------------------------------
# 3. Resume logic
# ---------------------------------------------------------------------------
RESUME_ARGS=""
if [ -n "$RESUME_PATH" ]; then
    echo "Resuming from: $RESUME_PATH"
    RESUME_ARGS="trainer.resume_mode=resume_path trainer.resume_from_path=$RESUME_PATH"
fi

# When resuming with a new lr, skip loading optimizer state so the new lr takes effect.
# Set FRESH_OPTIMIZER=1 to only load model weights (not optimizer/extra).
if [ "${FRESH_OPTIMIZER:-0}" = "1" ]; then
    echo "FRESH_OPTIMIZER=1: will load model weights only (new optimizer with lr from config)"
    RESUME_ARGS="${RESUME_ARGS} actor_rollout_ref.actor.checkpoint.load_contents=\[model\]"
fi

# ---------------------------------------------------------------------------
# 4. verl DAPO training (cuda:0-6, 7 GPUs, TP=1)
#
#    verl 0.5.0 uses Hydra config: key=value syntax (no -- prefix).
#    TP=1 → 7 vLLM DP groups for maximum rollout throughput.
# ---------------------------------------------------------------------------
echo ""
echo "Starting verl DAPO training on GPUs 0-6 (7 GPUs, TP=1)..."
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 VLLM_USE_V1=0 nohup python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$EVAL_FILE" \
    data.train_batch_size=112 \
    data.val_batch_size=56 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=56 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.use_torch_compile=false \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    \
    reward_model.reward_manager=naive \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name=compute_score \
    \
    trainer.total_epochs=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.project_name=med_r1_dapo \
    trainer.experiment_name=dapo_prm \
    'trainer.logger=[console]' \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    ${RESUME_ARGS} \
    > "$LOG_DIR/verl_dapo.log" 2>&1 &

VERL_PID=$!

echo "verl DAPO training started."
echo "  verl PID:          $VERL_PID"
echo "  Reward server PID: $REWARD_PID"
echo ""
echo "Logs:"
echo "  Training : tail -f $LOG_DIR/verl_dapo.log"
echo "  Reward   : tail -f $LOG_DIR/reward_server.log"
echo ""
echo "Waiting for training to complete..."
echo "(Press Ctrl+C to stop — reward server will be cleaned up.)"

wait "$VERL_PID"
echo "Training complete (exit code: $?)."
