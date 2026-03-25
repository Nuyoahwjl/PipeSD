#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET="${DATASET:-humaneval}"
BANDWIDTH_MBPS="${BANDWIDTH_MBPS:-2.5}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-4}"
VERIFY_THRESH_SINGLE="${VERIFY_THRESH_SINGLE:-0.9}"
VERIFY_THRESH_MULTI="${VERIFY_THRESH_MULTI:-0.95}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

append_extra_args() {
  local -n arr_ref=$1
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_parts=($EXTRA_ARGS)
    arr_ref+=( "${extra_parts[@]}" )
  fi
}

run_variant() {
  local merge_policy="$1"
  local label="$2"

  cmd=( "$PYTHON_BIN" app/run_edge.py
    --dataset "$DATASET"
    --algorithm pipesd
    --verify_strategy hybrid
    --verify_thresh_single "$VERIFY_THRESH_SINGLE"
    --verify_thresh_multi "$VERIFY_THRESH_MULTI"
    --bandwidth_MBps "$BANDWIDTH_MBPS"
    --start_index_of_sample "$START_INDEX"
    --end_index_of_sample "$END_INDEX"
    --merge_policy "$merge_policy"
  )
  append_extra_args cmd

  echo "[pilot] ${label}: dataset=${DATASET} bw=${BANDWIDTH_MBPS}MB samples=${START_INDEX}-${END_INDEX}"
  "${cmd[@]}"
  sleep 2
}

run_variant "dp" "pipesd_dp"
run_variant "immediate" "pipesd_immediate"
run_variant "no_early" "pipesd_no_early"
