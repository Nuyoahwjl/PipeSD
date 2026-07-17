#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EDGE_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PHASE="${PHASE:-help}"
DATASET="${DATASET:-humaneval}"
SEED="${SEED:-3407}"
TARGET_OUTPUT_TOKENS="${TARGET_OUTPUT_TOKENS:-1000}"
RESULT_TAG="${RESULT_TAG:-four_mode_s1_paper}"
THREADS="${THREADS:-2}"
CTX_SIZE="${CTX_SIZE:-16384}"
MAX_GENERATED_TOKENS="${MAX_GENERATED_TOKENS:-128}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

BANDWIDTH_MBPS="${BANDWIDTH_MBPS:-2.5}"
DOWNLINK_BANDWIDTH_MBPS="${DOWNLINK_BANDWIDTH_MBPS:-25}"
DRAFT_N_GPU_LAYERS="${DRAFT_N_GPU_LAYERS:-0}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-120}"
SOFTWARE_UPLINK_STARTUP_MS="${SOFTWARE_UPLINK_STARTUP_MS:-25}"
SOFTWARE_DOWNLINK_STARTUP_MS="${SOFTWARE_DOWNLINK_STARTUP_MS:-0}"

if [[ "$DATASET" == "humaneval" ]]; then
  START_INDEX="${START_INDEX:-50}"
  END_INDEX="${END_INDEX:-163}"
  VANILLA_GAMMA="${VANILLA_GAMMA:-6}"
  VANILLA_VERIFY_NUM="${VANILLA_VERIFY_NUM:-6}"
  PIPESD_SINGLE_THRESH="${PIPESD_SINGLE_THRESH:-0.3514}"
  PIPESD_MULTI_THRESH="${PIPESD_MULTI_THRESH:-0.9}"
  PURE_EDGE_MODEL="${PURE_EDGE_MODEL:-pre_models/deepseek-coder-1.3b-instruct-GGUF/deepseek-coder-1.3b-instruct.Q4_K_M.gguf}"
  PURE_CLOUD_MODEL="${PURE_CLOUD_MODEL:-../cloud/pre_models/deepseek-coder-6.7B-instruct-GGUF/deepseek-coder-6.7b-instruct.Q4_K_M.gguf}"
elif [[ "$DATASET" == "gsm8k" ]]; then
  START_INDEX="${START_INDEX:-100}"
  END_INDEX="${END_INDEX:-1318}"
  VANILLA_GAMMA="${VANILLA_GAMMA:-4}"
  VANILLA_VERIFY_NUM="${VANILLA_VERIFY_NUM:-4}"
  PIPESD_SINGLE_THRESH="${PIPESD_SINGLE_THRESH:-0.4}"
  PIPESD_MULTI_THRESH="${PIPESD_MULTI_THRESH:-0.65}"
  PURE_EDGE_MODEL="${PURE_EDGE_MODEL:-pre_models/tinyllama-1.1b-chat-v1.0-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
  PURE_CLOUD_MODEL="${PURE_CLOUD_MODEL:-../cloud/pre_models/Llama-2-7b-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf}"
else
  echo "unsupported DATASET=$DATASET" >&2
  exit 2
fi

append_extra_args() {
  local -n array_ref=$1
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_parts=( $EXTRA_ARGS )
    array_ref+=( "${extra_parts[@]}" )
  fi
}

run_cmd() {
  local -n array_ref=$1
  echo "[run] ${array_ref[*]}"
  "${array_ref[@]}"
}

run_pure() {
  local mode=$1
  local model_path=$2
  local gpu_layers=0
  [[ "$mode" == "pure_cloud" ]] && gpu_layers=-1
  local cmd=( "$PYTHON_BIN" app/run_pure_baseline.py
    --mode "$mode"
    --dataset "$DATASET"
    --model_path "$model_path"
    --n_gpu_layers "$gpu_layers"
    --seed "$SEED"
    --threads "$THREADS"
    --ctx_size "$CTX_SIZE"
    --max_generated_tokens "$MAX_GENERATED_TOKENS"
    --start_index_of_sample "$START_INDEX"
    --end_index_of_sample "$END_INDEX"
    --target_output_tokens "$TARGET_OUTPUT_TOKENS"
    --result_tag "$RESULT_TAG"
  )
  append_extra_args cmd
  run_cmd cmd
}

collaborative_common=(
  --dataset "$DATASET"
  --seed "$SEED"
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --downlink_bandwidth_MBps "$DOWNLINK_BANDWIDTH_MBPS"
  --network_shaping_mode software
  --software_uplink_startup_ms "$SOFTWARE_UPLINK_STARTUP_MS"
  --software_downlink_startup_ms "$SOFTWARE_DOWNLINK_STARTUP_MS"
  --start_index_of_sample "$START_INDEX"
  --end_index_of_sample "$END_INDEX"
  --evaluation_protocol paper_table1
  --target_output_tokens "$TARGET_OUTPUT_TOKENS"
  --draft_n_gpu_layers "$DRAFT_N_GPU_LAYERS"
  --server_timeout_s "$SERVER_TIMEOUT_S"
  --result_tag "$RESULT_TAG"
)

run_collaborative() {
  local vanilla=( "$PYTHON_BIN" app/run_edge.py
    "${collaborative_common[@]}"
    --algorithm vanilla
    --verify_strategy fixed-num
    --gamma "$VANILLA_GAMMA"
    --verify_num "$VANILLA_VERIFY_NUM"
  )
  append_extra_args vanilla
  run_cmd vanilla

  local pipesd=( "$PYTHON_BIN" app/run_edge.py
    "${collaborative_common[@]}"
    --algorithm pipesd
    --verify_strategy hybrid
    --verify_thresh_single "$PIPESD_SINGLE_THRESH"
    --verify_thresh_multi "$PIPESD_MULTI_THRESH"
    --merge_policy "${PIPESD_MERGE_POLICY:-dp}"
  )
  if [[ -n "${PIPESD_BO_CONFIG:-}" ]]; then
    pipesd+=( --bo_config_path "$PIPESD_BO_CONFIG" )
  fi
  append_extra_args pipesd
  run_cmd pipesd
}

case "$PHASE" in
  pure_edge)
    run_pure pure_edge "$PURE_EDGE_MODEL"
    ;;
  pure_cloud)
    echo "Pure Cloud loads a second target model. Stop the FastAPI cloud service first on a single-GPU host."
    run_pure pure_cloud "$PURE_CLOUD_MODEL"
    ;;
  collaborative)
    echo "Vanilla and PipeSD require the cloud FastAPI service on PIPE_SD_SERVER_URL (default port 8000)."
    run_collaborative
    ;;
  compare)
    compare_cmd=( "$PYTHON_BIN" scripts/compare_four_modes.py
      exp/exp__wjl__four__modes
      --dataset "$DATASET"
      --result-tag "$RESULT_TAG"
      --output-dir "exp/exp__wjl__four__modes/$DATASET/comparison"
    )
    run_cmd compare_cmd
    ;;
  help|*)
    cat <<'EOF'
Run one safe phase at a time on a single server:

  PHASE=pure_edge    bash scripts/eval_four_modes.sh
  PHASE=pure_cloud   bash scripts/eval_four_modes.sh  # stop cloud API first
  PHASE=collaborative bash scripts/eval_four_modes.sh # start cloud API first
  PHASE=compare      bash scripts/eval_four_modes.sh

Set DATASET=gsm8k or DATASET=humaneval. All phases must use the same
RESULT_TAG, SEED, sample range, and TARGET_OUTPUT_TOKENS.
EOF
    [[ "$PHASE" == "help" ]] || exit 2
    ;;
esac
