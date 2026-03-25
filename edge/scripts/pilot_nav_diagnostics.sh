#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET="${DATASET:-humaneval}"
BANDWIDTH_MBPS="${BANDWIDTH_MBPS:-2.5}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-4}"
RESULT_TAG="${RESULT_TAG:-nav_diag_pilot}"

VANILLA_GAMMA="${VANILLA_GAMMA:-6}"
HSL_THRESH="${HSL_THRESH:-0.99}"
PIPESD_SINGLE_THRESH="${PIPESD_SINGLE_THRESH:-0.9}"
PIPESD_MULTI_THRESH="${PIPESD_MULTI_THRESH:-0.95}"
EDGELLM_INIT_ALPHA="${EDGELLM_INIT_ALPHA:-0.92}"
EDGELLM_MULTIPLY_TIMES="${EDGELLM_MULTIPLY_TIMES:-0.95}"
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
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --start_index_of_sample "$START_INDEX"
  --end_index_of_sample "$END_INDEX"
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
  --merge_policy dp
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --start_index_of_sample "$START_INDEX"
  --end_index_of_sample "$END_INDEX"
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
  --result_tag "$RESULT_TAG"
)
append_extra_args cmd
run_cmd cmd
