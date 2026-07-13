#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EDGE_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cmd=( "$PYTHON_BIN" app/run_edge.py
  --dataset humaneval
  --algorithm pipesd
  --verify_strategy hybrid
  --merge_policy "${PIPESD_MERGE_POLICY:-dp}"
  --bandwidth_MBps "${BANDWIDTH_MBPS:-2.5}"
  --start_index_of_sample "${START_INDEX:-0}"
  --end_index_of_sample "${END_INDEX:-9}"
  --default_token_compute "${DEFAULT_TOKEN_COMPUTE:-0.036}"
  --draft_n_gpu_layers "${DRAFT_N_GPU_LAYERS:-0}"
  --server_timeout_s "${SERVER_TIMEOUT_S:-120}"
  --bayes_optimize
  --bayes_only
  --bayes_calls "${BO_CALLS:-16}"
  --bayes_init_points "${BO_INIT_POINTS:-1}"
  --bayes_tokens_per_sample "${BO_TOKENS_PER_SAMPLE:-${BO_TOKENS_PER_TRIAL:-20}}"
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
