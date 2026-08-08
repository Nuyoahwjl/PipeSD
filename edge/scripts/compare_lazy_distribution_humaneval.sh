#!/usr/bin/env bash
set -euo pipefail

# Standalone A/B evaluation for PipeSD's probability transport.
# This script deliberately does not source or call eval_humaneval.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$EDGE_DIR/.." && pwd)"
CLOUD_DIR="$REPO_DIR/cloud"
cd "$EDGE_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-3407}"
START_INDEX="${START_INDEX:-50}"
END_INDEX="${END_INDEX:-163}"
TARGET_OUTPUT_TOKENS="${TARGET_OUTPUT_TOKENS:-1000}"
EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL:-paper_table1}"

PIPESD_SINGLE_THRESH="${PIPESD_SINGLE_THRESH:-0.3514}"
PIPESD_MULTI_THRESH="${PIPESD_MULTI_THRESH:-0.9}"
PIPESD_MERGE_POLICY="${PIPESD_MERGE_POLICY:-dp}"
PIPESD_BO_CONFIG="${PIPESD_BO_CONFIG:-}"
INITIAL_GENERATION_GAMMA="${INITIAL_GENERATION_GAMMA:-0.036}"
DRAFT_N_GPU_LAYERS="${DRAFT_N_GPU_LAYERS:-0}"

BANDWIDTH_MBPS="${BANDWIDTH_MBPS:-2.5}"
DOWNLINK_BANDWIDTH_MBPS="${DOWNLINK_BANDWIDTH_MBPS:-25}"
NETWORK_SHAPING_MODE="${NETWORK_SHAPING_MODE:-software}"
SOFTWARE_UPLINK_STARTUP_MS="${SOFTWARE_UPLINK_STARTUP_MS:-25}"
SOFTWARE_DOWNLINK_STARTUP_MS="${SOFTWARE_DOWNLINK_STARTUP_MS:-0}"
SOFTWARE_BANDWIDTH_PROFILE="${SOFTWARE_BANDWIDTH_PROFILE:-}"
SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S="${SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S:-20}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-120}"

# The comparison defaults to the main-branch-style serial backend.
START_CLOUD="${START_CLOUD:-1}"
CLOUD_BACKEND="${CLOUD_BACKEND:-serial}"
CLOUD_PORT="${CLOUD_PORT:-8000}"
CLOUD_CTX_SIZE="${CLOUD_CTX_SIZE:-16384}"
CLOUD_MAX_SEQUENCES="${CLOUD_MAX_SEQUENCES:-1}"
CLOUD_BATCH_SIZE="${CLOUD_BATCH_SIZE:-1024}"
CLOUD_UBATCH_SIZE="${CLOUD_UBATCH_SIZE:-64}"
CLOUD_BATCH_WAIT_MS="${CLOUD_BATCH_WAIT_MS:-2}"
CLOUD_THREADS="${CLOUD_THREADS:-4}"
CLOUD_MAX_TOKENS="${CLOUD_MAX_TOKENS:-128}"
CLOUD_CPUSET="${CLOUD_CPUSET:-}"
CLOUD_START_TIMEOUT_S="${CLOUD_START_TIMEOUT_S:-300}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d-%H%M%S)}"
FULL_RESULT_TAG="${FULL_RESULT_TAG:-lazy_ab_full_${RUN_LABEL}}"
LAZY_RESULT_TAG="${LAZY_RESULT_TAG:-lazy_ab_lazy_${RUN_LABEL}}"
SKIP_FULL="${SKIP_FULL:-0}"
SKIP_LAZY="${SKIP_LAZY:-0}"

PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN:-<not found>}" >&2
  exit 1
fi
if [[ "$CLOUD_BACKEND" != "serial" && "$CLOUD_BACKEND" != "batched" ]]; then
  echo "CLOUD_BACKEND must be serial or batched: $CLOUD_BACKEND" >&2
  exit 1
fi
if [[ "$SKIP_FULL" != "0" && "$SKIP_FULL" != "1" ]]; then
  echo "SKIP_FULL must be 0 or 1: $SKIP_FULL" >&2
  exit 1
fi
if [[ "$SKIP_LAZY" != "0" && "$SKIP_LAZY" != "1" ]]; then
  echo "SKIP_LAZY must be 0 or 1: $SKIP_LAZY" >&2
  exit 1
fi
if [[ -n "$CLOUD_CPUSET" ]] && ! command -v taskset >/dev/null 2>&1; then
  echo "taskset is required when CLOUD_CPUSET is set" >&2
  exit 1
fi

export PIPE_SD_SERVER_URL="${PIPE_SD_SERVER_URL:-http://127.0.0.1:${CLOUD_PORT}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPU_ENERGY_DEVICE="${GPU_ENERGY_DEVICE:-0}"
HEALTH_URL="${PIPE_SD_SERVER_URL%/}/health"
CLOUD_LOG="${CLOUD_LOG:-$CLOUD_DIR/logs/lazy-ab-humaneval-${RUN_LABEL}.log}"
CLOUD_PID=""

cleanup() {
  if [[ -n "$CLOUD_PID" ]] && kill -0 "$CLOUD_PID" 2>/dev/null; then
    kill "$CLOUD_PID"
    wait "$CLOUD_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$START_CLOUD" == "1" ]] && "$PYTHON_BIN" -c \
  'import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=1)' \
  "$HEALTH_URL" >/dev/null 2>&1; then
  echo "A server is already listening at $PIPE_SD_SERVER_URL." >&2
  echo "Stop it, or rerun with START_CLOUD=0 to reuse it explicitly." >&2
  exit 1
fi

mkdir -p "$CLOUD_DIR/logs"
if [[ "$START_CLOUD" == "1" ]]; then
  cloud_cmd=( "$PYTHON_BIN" -m src.speculative_server
    --dataset humaneval
    --backend "$CLOUD_BACKEND"
    --max_sequences "$CLOUD_MAX_SEQUENCES"
    --ctx_size "$CLOUD_CTX_SIZE"
    --batch_size "$CLOUD_BATCH_SIZE"
    --ubatch_size "$CLOUD_UBATCH_SIZE"
    --batch_wait_ms "$CLOUD_BATCH_WAIT_MS"
    --batch_request_timeout_s "$SERVER_TIMEOUT_S"
    --threads "$CLOUD_THREADS"
    --max_tokens "$CLOUD_MAX_TOKENS"
    --temp 0
    --top_k 1
    --top_p 1
    --seed "$SEED"
    --port "$CLOUD_PORT"
  )
  if [[ -n "$CLOUD_CPUSET" ]]; then
    cloud_cmd=( taskset -c "$CLOUD_CPUSET" "${cloud_cmd[@]}" )
  fi
  (
    cd "$CLOUD_DIR"
    exec "${cloud_cmd[@]}"
  ) >"$CLOUD_LOG" 2>&1 &
  CLOUD_PID=$!
  echo "[cloud] pid=$CLOUD_PID backend=$CLOUD_BACKEND log=$CLOUD_LOG"
fi

echo "[preflight] waiting for $CLOUD_BACKEND Cloud at $HEALTH_URL"
cloud_ready=0
for ((attempt = 0; attempt < CLOUD_START_TIMEOUT_S; attempt++)); do
  if "$PYTHON_BIN" -c \
    'import json,sys,urllib.request; d=json.load(urllib.request.urlopen(sys.argv[1], timeout=2)); assert d.get("backend") == sys.argv[2], d' \
    "$HEALTH_URL" "$CLOUD_BACKEND" >/dev/null 2>&1; then
    cloud_ready=1
    break
  fi
  if [[ -n "$CLOUD_PID" ]] && ! kill -0 "$CLOUD_PID" 2>/dev/null; then
    echo "Cloud exited during startup. Last log lines:" >&2
    tail -n 80 "$CLOUD_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ "$cloud_ready" != "1" ]]; then
  echo "Timed out waiting for $CLOUD_BACKEND Cloud at $HEALTH_URL" >&2
  [[ -f "$CLOUD_LOG" ]] && tail -n 80 "$CLOUD_LOG" >&2 || true
  exit 1
fi

echo "[configuration] dataset=humaneval backend=$CLOUD_BACKEND samples=${START_INDEX}-${END_INDEX} target=$TARGET_OUTPUT_TOKENS"

# Original PipeSD: upload every complete draft-token probability distribution.
full_cmd=( "$PYTHON_BIN" app/run_edge.py
  --dataset humaneval
  --seed "$SEED"
  --algorithm pipesd
  --verify_strategy hybrid
  --verify_thresh_single "$PIPESD_SINGLE_THRESH"
  --verify_thresh_multi "$PIPESD_MULTI_THRESH"
  --merge_policy "$PIPESD_MERGE_POLICY"
  --bo_config_path "$PIPESD_BO_CONFIG"
  --prob_transport full
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --downlink_bandwidth_MBps "$DOWNLINK_BANDWIDTH_MBPS"
  --network_shaping_mode "$NETWORK_SHAPING_MODE"
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
  --result_tag "$FULL_RESULT_TAG"
)
if [[ "$SKIP_FULL" == "1" ]]; then
  echo "[1/2 original PipeSD] skipped; reusing tag=$FULL_RESULT_TAG"
else
  echo "[1/2 original PipeSD] ${full_cmd[*]}"
  "${full_cmd[@]}"
  sleep 2
fi

# Improved PipeSD: upload scalar draft-token probabilities and fetch a full
# distribution only after a rejection.
lazy_cmd=( "$PYTHON_BIN" app/run_edge.py
  --dataset humaneval
  --seed "$SEED"
  --algorithm pipesd
  --verify_strategy hybrid
  --verify_thresh_single "$PIPESD_SINGLE_THRESH"
  --verify_thresh_multi "$PIPESD_MULTI_THRESH"
  --merge_policy "$PIPESD_MERGE_POLICY"
  --bo_config_path "$PIPESD_BO_CONFIG"
  --prob_transport lazy_distribution
  --bandwidth_MBps "$BANDWIDTH_MBPS"
  --downlink_bandwidth_MBps "$DOWNLINK_BANDWIDTH_MBPS"
  --network_shaping_mode "$NETWORK_SHAPING_MODE"
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
  --result_tag "$LAZY_RESULT_TAG"
)
if [[ "$SKIP_LAZY" == "1" ]]; then
  echo "[2/2 lazy_distribution PipeSD] skipped; reusing tag=$LAZY_RESULT_TAG"
else
  echo "[2/2 lazy_distribution PipeSD] ${lazy_cmd[*]}"
  "${lazy_cmd[@]}"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_lazy_distribution_humaneval.py" \
  --edge-dir "$EDGE_DIR" \
  --full-tag "$FULL_RESULT_TAG" \
  --lazy-tag "$LAZY_RESULT_TAG"
