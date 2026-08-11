#!/usr/bin/env bash
set -euo pipefail

# One-pass HumanEval matrix:
#   1/2/4/8 clients x vanilla/HSL/EdgeLLM/PipeSD x full/lazy transport.
# Every condition uses one shared batched Cloud, a synchronized warm-up, and a
# 600-second measurement window. The script is intentionally nohup-safe.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$EDGE_DIR/.." && pwd)"
CLOUD_DIR="$REPO_DIR/cloud"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/pipesd/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPU_ENERGY_DEVICE="${GPU_ENERGY_DEVICE:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

START_CLOUD="${START_CLOUD:-1}"
CLOUD_BACKEND="${CLOUD_BACKEND:-batched}"
CLOUD_PORT="${CLOUD_PORT:-8000}"
CLOUD_CPUSET="${CLOUD_CPUSET:-0-3}"
CLOUD_MAX_SEQUENCES="${CLOUD_MAX_SEQUENCES:-8}"
CLOUD_CTX_SIZE="${CLOUD_CTX_SIZE:-1024}"
CLOUD_BATCH_SIZE="${CLOUD_BATCH_SIZE:-1024}"
CLOUD_UBATCH_SIZE="${CLOUD_UBATCH_SIZE:-64}"
CLOUD_BATCH_WAIT_MS="${CLOUD_BATCH_WAIT_MS:-2}"
CLOUD_THREADS="${CLOUD_THREADS:-4}"
CLOUD_START_TIMEOUT_S="${CLOUD_START_TIMEOUT_S:-300}"

CLIENT_COUNTS="${CLIENT_COUNTS:-1 2 4 8}"
ALGORITHMS="${ALGORITHMS:-vanilla hsl edgeLLM pipesd}"
PROB_TRANSPORTS="${PROB_TRANSPORTS:-full lazy}"
EDGE_CPUSETS="${EDGE_CPUSETS:-4-5;6-7;8-9;10-11;12-13;14-15;16-17;18-19}"

SEED="${SEED:-3407}"
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
DURATION_SECONDS="${DURATION_SECONDS:-300}"
MAX_GENERATED_TOKENS="${MAX_GENERATED_TOKENS:-128}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-300}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d-%H%M%S)}"
RESUME="${RESUME:-0}"

VANILLA_GAMMA="${VANILLA_GAMMA:-6}"
VANILLA_VERIFY_NUM="${VANILLA_VERIFY_NUM:-6}"
HSL_THRESH="${HSL_THRESH:-0.95}"
EDGELLM_INIT_ALPHA="${EDGELLM_INIT_ALPHA:-0.92}"
EDGELLM_FULL_ACCEPT_DECAY="${EDGELLM_FULL_ACCEPT_DECAY:-0.8}"
PIPESD_SINGLE_THRESH="${PIPESD_SINGLE_THRESH:-0.3514}"
PIPESD_MULTI_THRESH="${PIPESD_MULTI_THRESH:-0.9}"
PIPESD_MERGE_POLICY="${PIPESD_MERGE_POLICY:-dp}"
INITIAL_GENERATION_GAMMA="${INITIAL_GENERATION_GAMMA:-0.036}"

BANDWIDTH_MBPS="${BANDWIDTH_MBPS:-2.5}"
DOWNLINK_BANDWIDTH_MBPS="${DOWNLINK_BANDWIDTH_MBPS:-25}"
SOFTWARE_UPLINK_STARTUP_MS="${SOFTWARE_UPLINK_STARTUP_MS:-25}"
SOFTWARE_DOWNLINK_STARTUP_MS="${SOFTWARE_DOWNLINK_STARTUP_MS:-0}"
LINK_PROFILE="${LINK_PROFILE:-1.25:12.5,2.5:25,5:50,2.5:25}"
SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S="${SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S:-20}"

LAZY_COMM_PROBE_SIZES="${LAZY_COMM_PROBE_SIZES:-1,4,16,64,256,1024,2048,4096}"
LAZY_COMM_PROBE_REPETITIONS="${LAZY_COMM_PROBE_REPETITIONS:-3}"
LAZY_COMM_MIN_R_SQUARED="${LAZY_COMM_MIN_R_SQUARED:-0.8}"

PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN:-<not found>}" >&2
  exit 1
fi
if [[ "$(uname -s)" != Linux* ]]; then
  echo "Duration-mode CPU affinity requires Linux." >&2
  exit 1
fi
if [[ "$CLOUD_BACKEND" != "batched" ]]; then
  echo "This experiment requires CLOUD_BACKEND=batched: $CLOUD_BACKEND" >&2
  exit 1
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "RESUME must be 0 or 1: $RESUME" >&2
  exit 1
fi
if [[ "$START_CLOUD" == "1" ]] && ! command -v taskset >/dev/null 2>&1; then
  echo "taskset is required for Cloud CPU affinity." >&2
  exit 1
fi

max_clients=0
client_count_entries=0
for clients in $CLIENT_COUNTS; do
  if [[ "$clients" -lt 1 || "$clients" -gt 8 ]]; then
    echo "CLIENT_COUNTS must contain values from 1 through 8: $clients" >&2
    exit 1
  fi
  (( clients > max_clients )) && max_clients="$clients"
  (( client_count_entries += 1 ))
done
if (( client_count_entries == 0 )); then
  echo "CLIENT_COUNTS must not be empty" >&2
  exit 1
fi
if (( CLOUD_MAX_SEQUENCES < max_clients )); then
  echo "CLOUD_MAX_SEQUENCES=$CLOUD_MAX_SEQUENCES is smaller than max clients=$max_clients" >&2
  exit 1
fi
IFS=';' read -r -a edge_cpuset_specs <<< "$EDGE_CPUSETS"
if (( ${#edge_cpuset_specs[@]} < max_clients )); then
  echo "EDGE_CPUSETS provides ${#edge_cpuset_specs[@]} sets, but $max_clients clients were requested" >&2
  exit 1
fi

algorithm_count=0
for algorithm in $ALGORITHMS; do
  case "$algorithm" in
    vanilla|hsl|edgeLLM|pipesd) ;;
    *) echo "Unsupported algorithm: $algorithm" >&2; exit 1 ;;
  esac
  (( algorithm_count += 1 ))
done
transport_count=0
for transport in $PROB_TRANSPORTS; do
  case "$transport" in
    full|lazy|lazy_distribution) ;;
    *) echo "Unsupported probability transport: $transport" >&2; exit 1 ;;
  esac
  (( transport_count += 1 ))
done
if (( algorithm_count == 0 || transport_count == 0 )); then
  echo "ALGORITHMS and PROB_TRANSPORTS must not be empty" >&2
  exit 1
fi
condition_count=$((client_count_entries * algorithm_count * transport_count))

export PIPE_SD_SERVER_URL="${PIPE_SD_SERVER_URL:-http://127.0.0.1:${CLOUD_PORT}}"
HEALTH_URL="${PIPE_SD_SERVER_URL%/}/health"
OUTPUT_DIR="$EDGE_DIR/exp/full-lazy-multiclient/$RUN_LABEL"
CLOUD_LOG="$CLOUD_DIR/logs/multiedge-full-lazy-${RUN_LABEL}.log"
mkdir -p "$CLOUD_DIR/logs" "$OUTPUT_DIR"

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

if [[ "$START_CLOUD" == "1" ]]; then
  (
    cd "$CLOUD_DIR"
    exec taskset -c "$CLOUD_CPUSET" "$PYTHON_BIN" -m src.speculative_server \
      --dataset humaneval \
      --backend "$CLOUD_BACKEND" \
      --max_sequences "$CLOUD_MAX_SEQUENCES" \
      --ctx_size "$CLOUD_CTX_SIZE" \
      --batch_size "$CLOUD_BATCH_SIZE" \
      --ubatch_size "$CLOUD_UBATCH_SIZE" \
      --batch_wait_ms "$CLOUD_BATCH_WAIT_MS" \
      --batch_request_timeout_s "$SERVER_TIMEOUT_S" \
      --threads "$CLOUD_THREADS" \
      --max_tokens "$MAX_GENERATED_TOKENS" \
      --temp 0 \
      --top_k 1 \
      --top_p 1 \
      --seed "$SEED" \
      --port "$CLOUD_PORT"
  ) >"$CLOUD_LOG" 2>&1 &
  CLOUD_PID=$!
  echo "[cloud] pid=$CLOUD_PID backend=$CLOUD_BACKEND log=$CLOUD_LOG"
fi

echo "[preflight] waiting for batched Cloud at $HEALTH_URL"
cloud_ready=0
for ((attempt = 0; attempt < CLOUD_START_TIMEOUT_S; attempt++)); do
  if "$PYTHON_BIN" -c \
    'import json,sys,urllib.request; d=json.load(urllib.request.urlopen(sys.argv[1], timeout=2)); assert d.get("backend") == "batched", d' \
    "$HEALTH_URL" >/dev/null 2>&1; then
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
  echo "Timed out waiting for batched Cloud at $HEALTH_URL" >&2
  [[ -f "$CLOUD_LOG" ]] && tail -n 80 "$CLOUD_LOG" >&2 || true
  exit 1
fi
"$PYTHON_BIN" -c \
  'import json,sys,urllib.request; d=json.load(urllib.request.urlopen(sys.argv[1], timeout=5)); assert d.get("backend") == "batched", d; print(json.dumps(d, indent=2))' \
  "$HEALTH_URL"

echo "[configuration] run_label=$RUN_LABEL"
echo "[configuration] clients=($CLIENT_COUNTS) algorithms=($ALGORITHMS) transports=($PROB_TRANSPORTS)"
echo "[configuration] one pass; conditions=$condition_count warmup=${WARMUP_SECONDS}s measurement=${DURATION_SECONDS}s per condition"
echo "[configuration] output_dir=$OUTPUT_DIR"

cd "$EDGE_DIR"
run_condition() {
  local algorithm="$1"
  local clients="$2"
  local transport_label="$3"
  local prob_transport="$transport_label"
  local tag summary
  local algorithm_args=()
  local transport_args=()

  if [[ "$transport_label" == "lazy" ]]; then
    prob_transport="lazy_distribution"
  fi
  tag="${RUN_LABEL}_${algorithm}_${transport_label}_c${clients}"
  summary="$OUTPUT_DIR/${algorithm}_${transport_label}_c${clients}.json"

  if [[ "$RESUME" == "1" && -s "$summary" ]]; then
    echo "[skip] existing summary=$summary"
    return 0
  fi
  if [[ "$RESUME" == "1" ]]; then
    # Never mix raw results left by an interrupted attempt into a resumed one.
    tag="${tag}_attempt$(date +%Y%m%d-%H%M%S)"
  fi

  case "$algorithm" in
    vanilla)
      algorithm_args=(
        --forward_arg=--verify_strategy --forward_arg=fixed-num
        --forward_arg=--gamma --forward_arg="$VANILLA_GAMMA"
        --forward_arg=--verify_num --forward_arg="$VANILLA_VERIFY_NUM"
      )
      ;;
    hsl)
      algorithm_args=(
        --forward_arg=--verify_strategy --forward_arg=single-token
        --forward_arg=--verify_thresh_single --forward_arg="$HSL_THRESH"
      )
      ;;
    edgeLLM)
      algorithm_args=(
        --forward_arg=--init_alpha --forward_arg="$EDGELLM_INIT_ALPHA"
        --forward_arg=--edge_llm_full_accept_decay --forward_arg="$EDGELLM_FULL_ACCEPT_DECAY"
      )
      ;;
    pipesd)
      algorithm_args=(
        --forward_arg=--verify_strategy --forward_arg=hybrid
        --forward_arg=--verify_thresh_single --forward_arg="$PIPESD_SINGLE_THRESH"
        --forward_arg=--verify_thresh_multi --forward_arg="$PIPESD_MULTI_THRESH"
        --forward_arg=--merge_policy --forward_arg="$PIPESD_MERGE_POLICY"
        --forward_arg=--initial_generation_gamma --forward_arg="$INITIAL_GENERATION_GAMMA"
      )
      ;;
  esac

  if [[ "$prob_transport" == "lazy_distribution" ]]; then
    transport_args=(
      --forward_arg=--lazy_comm_probe_sizes --forward_arg="$LAZY_COMM_PROBE_SIZES"
      --forward_arg=--lazy_comm_probe_repetitions --forward_arg="$LAZY_COMM_PROBE_REPETITIONS"
      --forward_arg=--lazy_comm_min_r_squared --forward_arg="$LAZY_COMM_MIN_R_SQUARED"
    )
  fi

  echo "[run] algorithm=$algorithm transport=$prob_transport clients=$clients"
  "$PYTHON_BIN" scripts/run_multiclient_pilot.py \
    --dataset humaneval \
    --algorithm "$algorithm" \
    --num_clients "$clients" \
    --pilot_samples 164 \
    --workload_mode replicated \
    --duration_s "$DURATION_SECONDS" \
    --warmup_s "$WARMUP_SECONDS" \
    --barrier_timeout_s 1800 \
    --workload_seed "$SEED" \
    --cpu_sets "$EDGE_CPUSETS" \
    --base_tag "$tag" \
    --summary_path "$summary" \
    --python_bin "$PYTHON_BIN" \
    --forward_arg=--seed --forward_arg="$SEED" \
    --forward_arg=--max_generated_tokens --forward_arg="$MAX_GENERATED_TOKENS" \
    --forward_arg=--threads --forward_arg=2 \
    --forward_arg=--ctx_size --forward_arg="$CLOUD_CTX_SIZE" \
    --forward_arg=--draft_n_gpu_layers --forward_arg=0 \
    --forward_arg=--evaluation_protocol --forward_arg=sample_index \
    --forward_arg=--prob_transport --forward_arg="$prob_transport" \
    --forward_arg=--network_shaping_mode --forward_arg=software \
    --forward_arg=--bandwidth_MBps --forward_arg="$BANDWIDTH_MBPS" \
    --forward_arg=--downlink_bandwidth_MBps --forward_arg="$DOWNLINK_BANDWIDTH_MBPS" \
    --forward_arg=--software_uplink_startup_ms --forward_arg="$SOFTWARE_UPLINK_STARTUP_MS" \
    --forward_arg=--software_downlink_startup_ms --forward_arg="$SOFTWARE_DOWNLINK_STARTUP_MS" \
    --forward_arg=--software_bandwidth_profile --forward_arg="$LINK_PROFILE" \
    --forward_arg=--software_bandwidth_change_interval_s --forward_arg="$SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S" \
    --forward_arg=--server_timeout_s --forward_arg="$SERVER_TIMEOUT_S" \
    "${transport_args[@]}" \
    "${algorithm_args[@]}"
}

# Keep each full/lazy pair adjacent to reduce time-dependent system drift.
for clients in $CLIENT_COUNTS; do
  for algorithm in $ALGORITHMS; do
    for transport in $PROB_TRANSPORTS; do
      run_condition "$algorithm" "$clients" "$transport"
    done
  done
done

echo "[done] summaries=$OUTPUT_DIR/*.json"
