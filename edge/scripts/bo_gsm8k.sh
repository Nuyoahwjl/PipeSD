#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EDGE_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
BO_TOKENS_PER_TRIAL="${BO_TOKENS_PER_TRIAL:-1000}"

cmd=( "$PYTHON_BIN" app/run_edge.py
  --dataset gsm8k
  --seed "${SEED:-1234}"
  --algorithm pipesd
  --verify_strategy hybrid
  --merge_policy "${PIPESD_MERGE_POLICY:-dp}"
  --bandwidth_MBps "${BANDWIDTH_MBPS:-2.5}"
  --downlink_bandwidth_MBps "${DOWNLINK_BANDWIDTH_MBPS:-25}"
  --network_shaping_mode "${NETWORK_SHAPING_MODE:-software}"
  --start_index_of_sample "${START_INDEX:-0}"
  --end_index_of_sample "${END_INDEX:-1318}"
  --initial_generation_gamma "${INITIAL_GENERATION_GAMMA:-0.036}"
  --draft_n_gpu_layers "${DRAFT_N_GPU_LAYERS:-0}"
  --server_timeout_s "${SERVER_TIMEOUT_S:-120}"
  --bayes_optimize
  --bayes_only
  --bayes_calls "${BO_CALLS:-16}"
  --bayes_init_points "${BO_INIT_POINTS:-1}"
  --bo_protocol paper
  --bayes_tokens_per_trial "$BO_TOKENS_PER_TRIAL"
  --bayes_ei_xi "${BO_EI_XI:-0.1}"
  --bayes_single_min "${BO_SINGLE_MIN:-0.000001}"
  --bayes_single_max "${BO_SINGLE_MAX:-1.0}"
  --bayes_multi_min "${BO_MULTI_MIN:-0.000001}"
  --bayes_multi_max "${BO_MULTI_MAX:-1.0}"
)

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_parts=($EXTRA_ARGS)
  cmd+=( "${extra_parts[@]}" )
fi

echo "[run] ${cmd[*]}"
"${cmd[@]}"
