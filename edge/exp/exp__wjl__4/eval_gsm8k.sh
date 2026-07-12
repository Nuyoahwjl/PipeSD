#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EDGE_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET="${DATASET:-gsm8k}"
BANDWIDTH_MBPS="${BANDWIDTH_MBPS:-2.5}"
START_INDEX="${START_INDEX:-12}"
END_INDEX="${END_INDEX:-12}"
RESULT_TAG="${RESULT_TAG:-table1_s1_paper_12_token1024}"

VANILLA_GAMMA="${VANILLA_GAMMA:-4}"
VANILLA_VERIFY_NUM="${VANILLA_VERIFY_NUM:-4}"
HSL_THRESH="${HSL_THRESH:-0.7}"
EDGELLM_INIT_ALPHA="${EDGELLM_INIT_ALPHA:-0.5}"
EDGELLM_MULTIPLY_TIMES="${EDGELLM_MULTIPLY_TIMES:-0.7}"
PIPESD_SINGLE_THRESH="${PIPESD_SINGLE_THRESH:-0.8}"
PIPESD_MULTI_THRESH="${PIPESD_MULTI_THRESH:-0.5}"
PIPESD_MERGE_POLICY="${PIPESD_MERGE_POLICY:-dp}"

DEFAULT_TOKEN_COMPUTE="${DEFAULT_TOKEN_COMPUTE:-0.036}"
DRAFT_N_GPU_LAYERS="${DRAFT_N_GPU_LAYERS:-0}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-120}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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
  --algorithm vanilla
  --verify_strategy fixed-num
  --gamma "$VANILLA_GAMMA"
  --verify_num "$VANILLA_VERIFY_NUM"
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --start_index_of_sample "$START_INDEX"
  --end_index_of_sample "$END_INDEX"
  --default_token_compute "$DEFAULT_TOKEN_COMPUTE"
  --draft_n_gpu_layers "$DRAFT_N_GPU_LAYERS"
  --server_timeout_s "$SERVER_TIMEOUT_S"
  --result_tag "$RESULT_TAG"
)
append_extra_args cmd
run_cmd cmd

cmd=( "$PYTHON_BIN" app/run_edge.py
  --dataset "$DATASET"
  --algorithm hsl
  --verify_strategy single-token
  --verify_thresh_single "$HSL_THRESH"
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --start_index_of_sample "$START_INDEX"
  --end_index_of_sample "$END_INDEX"
  --default_token_compute "$DEFAULT_TOKEN_COMPUTE"
  --draft_n_gpu_layers "$DRAFT_N_GPU_LAYERS"
  --server_timeout_s "$SERVER_TIMEOUT_S"
  --result_tag "$RESULT_TAG"
)
append_extra_args cmd
run_cmd cmd

cmd=( "$PYTHON_BIN" app/run_edge.py
  --dataset "$DATASET"
  --algorithm edgeLLM
  --init_alpha "$EDGELLM_INIT_ALPHA"
  --multiply_times "$EDGELLM_MULTIPLY_TIMES"
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --start_index_of_sample "$START_INDEX"
  --end_index_of_sample "$END_INDEX"
  --default_token_compute "$DEFAULT_TOKEN_COMPUTE"
  --draft_n_gpu_layers "$DRAFT_N_GPU_LAYERS"
  --server_timeout_s "$SERVER_TIMEOUT_S"
  --result_tag "$RESULT_TAG"
)
append_extra_args cmd
run_cmd cmd

cmd=( "$PYTHON_BIN" app/run_edge.py
  --dataset "$DATASET"
  --algorithm pipesd
  --verify_strategy hybrid
  --verify_thresh_single "$PIPESD_SINGLE_THRESH"
  --verify_thresh_multi "$PIPESD_MULTI_THRESH"
  --merge_policy "$PIPESD_MERGE_POLICY"
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --start_index_of_sample "$START_INDEX"
  --end_index_of_sample "$END_INDEX"
  --default_token_compute "$DEFAULT_TOKEN_COMPUTE"
  --draft_n_gpu_layers "$DRAFT_N_GPU_LAYERS"
  --server_timeout_s "$SERVER_TIMEOUT_S"
  --result_tag "$RESULT_TAG"
)
append_extra_args cmd
run_cmd cmd
