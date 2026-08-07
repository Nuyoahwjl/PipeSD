#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$EDGE_DIR/.." && pwd)"
CLOUD_DIR="$REPO_DIR/cloud"
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
HSL_THRESH="${HSL_THRESH:-0.95}"
EDGELLM_INIT_ALPHA="${EDGELLM_INIT_ALPHA:-0.92}"
EDGELLM_FULL_ACCEPT_DECAY="${EDGELLM_FULL_ACCEPT_DECAY:-0.8}"
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

PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN:-<not found>}" >&2
  exit 1
fi
if [[ "$CLOUD_BACKEND" != "serial" && "$CLOUD_BACKEND" != "batched" ]]; then
  echo "CLOUD_BACKEND must be serial or batched: $CLOUD_BACKEND" >&2
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
CLOUD_LOG="${CLOUD_LOG:-$CLOUD_DIR/logs/eval-${DATASET}-${RUN_LABEL}.log}"
CLOUD_PID=""

cleanup() {
  if [[ -n "$CLOUD_PID" ]] && kill -0 "$CLOUD_PID" 2>/dev/null; then
    kill "$CLOUD_PID"
    wait "$CLOUD_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

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
    --dataset "$DATASET"
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
  if [[ -f "$CLOUD_LOG" ]]; then
    tail -n 80 "$CLOUD_LOG" >&2 || true
  fi
  exit 1
fi
"$PYTHON_BIN" -c \
  'import json,sys,urllib.request; d=json.load(urllib.request.urlopen(sys.argv[1], timeout=5)); assert d.get("backend") == sys.argv[2], d; print(json.dumps(d, indent=2))' \
  "$HEALTH_URL" "$CLOUD_BACKEND"

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
