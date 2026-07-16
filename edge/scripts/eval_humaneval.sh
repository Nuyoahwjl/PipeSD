#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EDGE_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-3407}"

DATASET="${DATASET:-humaneval}"
BANDWIDTH_MBPS="${BANDWIDTH_MBPS:-2.5}"
START_INDEX="${START_INDEX:-50}"
END_INDEX="${END_INDEX:-163}"
RESULT_TAG="${RESULT_TAG:-table1_s1_paper}"
EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL:-paper_table1}"
TARGET_OUTPUT_TOKENS="${TARGET_OUTPUT_TOKENS:-1000}"

VANILLA_GAMMA="${VANILLA_GAMMA:-6}"
VANILLA_VERIFY_NUM="${VANILLA_VERIFY_NUM:-6}"
HSL_THRESH="${HSL_THRESH:-0.99}"
EDGELLM_INIT_ALPHA="${EDGELLM_INIT_ALPHA:-0.92}"
EDGELLM_FULL_ACCEPT_DECAY="${EDGELLM_FULL_ACCEPT_DECAY:-0.5}"
PIPESD_MERGE_POLICY="${PIPESD_MERGE_POLICY:-dp}"

PIPESD_SINGLE_THRESH="${PIPESD_SINGLE_THRESH:-0.3514}"
PIPESD_MULTI_THRESH="${PIPESD_MULTI_THRESH:-0.9}"
PIPESD_BO_CONFIG="${PIPESD_BO_CONFIG:-}"

INITIAL_GENERATION_GAMMA="${INITIAL_GENERATION_GAMMA:-0.036}"
DRAFT_N_GPU_LAYERS="${DRAFT_N_GPU_LAYERS:-0}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-120}"
SOFTWARE_UPLINK_STARTUP_MS="${SOFTWARE_UPLINK_STARTUP_MS:-25}"
SOFTWARE_DOWNLINK_STARTUP_MS="${SOFTWARE_DOWNLINK_STARTUP_MS:-0}"
SOFTWARE_BANDWIDTH_PROFILE="${SOFTWARE_BANDWIDTH_PROFILE:-}"
SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S="${SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S:-20}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$PIPESD_SINGLE_THRESH" || -z "$PIPESD_MULTI_THRESH" ]]; then
  PIPESD_BO_CONFIG="${PIPESD_BO_CONFIG:-exp/exp__wjl/humaneval/pipesd/latest_bayes_best.json}"
  if [[ ! -f "$PIPESD_BO_CONFIG" ]]; then
    echo "missing PipeSD BO config: $PIPESD_BO_CONFIG; run scripts/bo_humaneval.sh first" >&2
    exit 1
  fi
  read -r CONFIG_SINGLE CONFIG_MULTI < <(
    "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); n=p.get("network_emulation", {}); assert n.get("emulator_version") == "shared-fifo-v1", "BO config predates shared software link; rerun BO"; print(p["best_thresh_single"], p["best_thresh_multi"])' "$PIPESD_BO_CONFIG"
  )
  PIPESD_SINGLE_THRESH="${PIPESD_SINGLE_THRESH:-$CONFIG_SINGLE}"
  PIPESD_MULTI_THRESH="${PIPESD_MULTI_THRESH:-$CONFIG_MULTI}"
fi

append_extra_args() {
  local -n arr_ref=$1
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_parts=($EXTRA_ARGS)
    arr_ref+=( "${extra_parts[@]}" )
  fi
}

run_cmd() {
  local -n cmd_ref=$1
  echo "[run] ${cmd_ref[*]}"
  "${cmd_ref[@]}"
  sleep 2
}

cmd=( "$PYTHON_BIN" app/run_edge.py
  --dataset "$DATASET"
  --seed "$SEED"
  --algorithm vanilla
  --verify_strategy fixed-num
  --gamma "$VANILLA_GAMMA"
  --verify_num "$VANILLA_VERIFY_NUM"
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --downlink_bandwidth_MBps "${DOWNLINK_BANDWIDTH_MBPS:-25}"
  --network_shaping_mode "${NETWORK_SHAPING_MODE:-software}"
  --software_uplink_startup_ms "$SOFTWARE_UPLINK_STARTUP_MS"
  --software_downlink_startup_ms "$SOFTWARE_DOWNLINK_STARTUP_MS"
  --software_bandwidth_profile "$SOFTWARE_BANDWIDTH_PROFILE"
  --software_bandwidth_change_interval_s "$SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S"
  --start_index_of_sample "$START_INDEX"
  --end_index_of_sample "$END_INDEX"
  --initial_generation_gamma "$INITIAL_GENERATION_GAMMA"
  --evaluation_protocol "$EVALUATION_PROTOCOL"
  --target_output_tokens "$TARGET_OUTPUT_TOKENS"
  --draft_n_gpu_layers "$DRAFT_N_GPU_LAYERS"
  --server_timeout_s "$SERVER_TIMEOUT_S"
  --result_tag "$RESULT_TAG"
)
append_extra_args cmd
run_cmd cmd

cmd=( "$PYTHON_BIN" app/run_edge.py
  --dataset "$DATASET"
  --seed "$SEED"
  --algorithm hsl
  --verify_strategy single-token
  --verify_thresh_single "$HSL_THRESH"
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --downlink_bandwidth_MBps "${DOWNLINK_BANDWIDTH_MBPS:-25}"
  --network_shaping_mode "${NETWORK_SHAPING_MODE:-software}"
  --software_uplink_startup_ms "$SOFTWARE_UPLINK_STARTUP_MS"
  --software_downlink_startup_ms "$SOFTWARE_DOWNLINK_STARTUP_MS"
  --software_bandwidth_profile "$SOFTWARE_BANDWIDTH_PROFILE"
  --software_bandwidth_change_interval_s "$SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S"
  --start_index_of_sample "$START_INDEX"
  --end_index_of_sample "$END_INDEX"
  --initial_generation_gamma "$INITIAL_GENERATION_GAMMA"
  --evaluation_protocol "$EVALUATION_PROTOCOL"
  --target_output_tokens "$TARGET_OUTPUT_TOKENS"
  --draft_n_gpu_layers "$DRAFT_N_GPU_LAYERS"
  --server_timeout_s "$SERVER_TIMEOUT_S"
  --result_tag "$RESULT_TAG"
)
append_extra_args cmd
run_cmd cmd

cmd=( "$PYTHON_BIN" app/run_edge.py
  --dataset "$DATASET"
  --seed "$SEED"
  --algorithm edgeLLM
  --init_alpha "$EDGELLM_INIT_ALPHA"
  --edge_llm_full_accept_decay "$EDGELLM_FULL_ACCEPT_DECAY"
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --downlink_bandwidth_MBps "${DOWNLINK_BANDWIDTH_MBPS:-25}"
  --network_shaping_mode "${NETWORK_SHAPING_MODE:-software}"
  --software_uplink_startup_ms "$SOFTWARE_UPLINK_STARTUP_MS"
  --software_downlink_startup_ms "$SOFTWARE_DOWNLINK_STARTUP_MS"
  --software_bandwidth_profile "$SOFTWARE_BANDWIDTH_PROFILE"
  --software_bandwidth_change_interval_s "$SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S"
  --start_index_of_sample "$START_INDEX"
  --end_index_of_sample "$END_INDEX"
  --initial_generation_gamma "$INITIAL_GENERATION_GAMMA"
  --evaluation_protocol "$EVALUATION_PROTOCOL"
  --target_output_tokens "$TARGET_OUTPUT_TOKENS"
  --draft_n_gpu_layers "$DRAFT_N_GPU_LAYERS"
  --server_timeout_s "$SERVER_TIMEOUT_S"
  --result_tag "$RESULT_TAG"
)
append_extra_args cmd
run_cmd cmd

cmd=( "$PYTHON_BIN" app/run_edge.py
  --dataset "$DATASET"
  --seed "$SEED"
  --algorithm pipesd
  --verify_strategy hybrid
  --verify_thresh_single "$PIPESD_SINGLE_THRESH"
  --verify_thresh_multi "$PIPESD_MULTI_THRESH"
  --merge_policy "$PIPESD_MERGE_POLICY"
  --bo_config_path "$PIPESD_BO_CONFIG"
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --downlink_bandwidth_MBps "${DOWNLINK_BANDWIDTH_MBPS:-25}"
  --network_shaping_mode "${NETWORK_SHAPING_MODE:-software}"
  --software_uplink_startup_ms "$SOFTWARE_UPLINK_STARTUP_MS"
  --software_downlink_startup_ms "$SOFTWARE_DOWNLINK_STARTUP_MS"
  --software_bandwidth_profile "$SOFTWARE_BANDWIDTH_PROFILE"
  --software_bandwidth_change_interval_s "$SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S"
  --start_index_of_sample "$START_INDEX"
  --end_index_of_sample "$END_INDEX"
  --initial_generation_gamma "$INITIAL_GENERATION_GAMMA"
  --evaluation_protocol "$EVALUATION_PROTOCOL"
  --target_output_tokens "$TARGET_OUTPUT_TOKENS"
  --draft_n_gpu_layers "$DRAFT_N_GPU_LAYERS"
  --server_timeout_s "$SERVER_TIMEOUT_S"
  --result_tag "$RESULT_TAG"
)
append_extra_args cmd
run_cmd cmd
