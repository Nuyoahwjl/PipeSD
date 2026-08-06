#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$EDGE_DIR/.." && pwd)"
CLOUD_DIR="$REPO_DIR/cloud"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/pipesd/bin/python}"
export PIPE_SD_SERVER_URL="${PIPE_SD_SERVER_URL:-http://127.0.0.1:8000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPU_ENERGY_DEVICE="${GPU_ENERGY_DEVICE:-0}"
START_CLOUD="${START_CLOUD:-1}"
CLOUD_CPUSET="${CLOUD_CPUSET:-0-3}"
EDGE_CPUSETS="${EDGE_CPUSETS:-4-5;6-7;8-9;10-11;12-13;14-15;16-17;18-19}"
CLIENT_COUNTS="${CLIENT_COUNTS:-2 4 8}"
ALGORITHMS="${ALGORITHMS:-vanilla pipesd}"
REPEATS="${REPEATS:-3}"
WARMUP_SECONDS="${WARMUP_SECONDS:-60}"
DURATION_SECONDS="${DURATION_SECONDS:-600}"
MAX_GENERATED_TOKENS="${MAX_GENERATED_TOKENS:-128}"
BATCH_WAIT_MS="${BATCH_WAIT_MS:-2}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d-%H%M%S)}"
LINK_PROFILE="${LINK_PROFILE:-1.25:12.5,2.5:25,5:50,2.5:25}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ "$(nproc)" -lt 24 ]]; then
  echo "This preset expects at least 24 logical CPUs; found $(nproc)." >&2
  exit 1
fi
if [[ "$START_CLOUD" == "1" ]] && ! command -v taskset >/dev/null 2>&1; then
  echo "taskset is required for Cloud CPU affinity." >&2
  exit 1
fi
for clients in $CLIENT_COUNTS; do
  if [[ "$clients" -lt 1 || "$clients" -gt 8 ]]; then
    echo "CLIENT_COUNTS must contain values from 1 through 8: $clients" >&2
    exit 1
  fi
done

if [[ "$START_CLOUD" == "1" ]] && "$PYTHON_BIN" -c \
  "import urllib.request; urllib.request.urlopen('${PIPE_SD_SERVER_URL%/}/health', timeout=1)" \
  >/dev/null 2>&1; then
  echo "A server is already listening at $PIPE_SD_SERVER_URL." >&2
  echo "Stop it, or rerun with START_CLOUD=0 to reuse it explicitly." >&2
  exit 1
fi

mkdir -p "$CLOUD_DIR/logs" "$EDGE_DIR/exp-multi/multiclient"
CLOUD_PID=""
cleanup() {
  if [[ -n "$CLOUD_PID" ]] && kill -0 "$CLOUD_PID" 2>/dev/null; then
    kill "$CLOUD_PID"
    wait "$CLOUD_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$START_CLOUD" == "1" ]]; then
  CLOUD_LOG="$CLOUD_DIR/logs/multiedge-${RUN_LABEL}.log"
  (
    cd "$CLOUD_DIR"
    exec taskset -c "$CLOUD_CPUSET" "$PYTHON_BIN" -m src.speculative_server \
      --dataset humaneval \
      --backend batched \
      --max_sequences 8 \
      --ctx_size 1024 \
      --batch_size 1024 \
      --ubatch_size 64 \
      --batch_wait_ms "$BATCH_WAIT_MS" \
      --batch_request_timeout_s 300 \
      --threads 4 \
      --max_tokens "$MAX_GENERATED_TOKENS" \
      --temp 0 \
      --top_k 1 \
      --top_p 1 \
      --seed 3407 \
      --port 8000
  ) >"$CLOUD_LOG" 2>&1 &
  CLOUD_PID=$!
  echo "[cloud] pid=$CLOUD_PID log=$CLOUD_LOG"
fi

echo "[preflight] waiting for batched Cloud at ${PIPE_SD_SERVER_URL%/}/health"
for _ in $(seq 1 300); do
  if "$PYTHON_BIN" -c \
    "import json,urllib.request; d=json.load(urllib.request.urlopen('${PIPE_SD_SERVER_URL%/}/health', timeout=2)); assert d.get('backend') == 'batched'" \
    >/dev/null 2>&1; then
    break
  fi
  if [[ -n "$CLOUD_PID" ]] && ! kill -0 "$CLOUD_PID" 2>/dev/null; then
    echo "Cloud exited during startup. Last log lines:" >&2
    tail -n 80 "$CLOUD_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
"$PYTHON_BIN" -c \
  "import json,urllib.request; d=json.load(urllib.request.urlopen('${PIPE_SD_SERVER_URL%/}/health', timeout=5)); assert d.get('backend') == 'batched', d; print(json.dumps(d, indent=2))"

cd "$EDGE_DIR"
run_condition() {
  local algorithm="$1"
  local clients="$2"
  local repeat="$3"
  local tag="${RUN_LABEL}_${algorithm}_c${clients}_r${repeat}"
  local summary="exp-multi/multiclient/${tag}.json"
  local algorithm_args=()

  if [[ "$algorithm" == "vanilla" ]]; then
    algorithm_args=(
      --forward_arg=--verify_strategy
      --forward_arg=fixed-num
      --forward_arg=--gamma
      --forward_arg=6
      --forward_arg=--verify_num
      --forward_arg=6
    )
  elif [[ "$algorithm" == "pipesd" ]]; then
    algorithm_args=(
      --forward_arg=--verify_strategy
      --forward_arg=hybrid
      --forward_arg=--verify_thresh_single
      --forward_arg=0.3514
      --forward_arg=--verify_thresh_multi
      --forward_arg=0.9
      --forward_arg=--merge_policy
      --forward_arg=dp
      --forward_arg=--initial_generation_gamma
      --forward_arg=0.036
    )
  else
    echo "Unsupported algorithm: $algorithm" >&2
    return 2
  fi

  echo "[run] algorithm=$algorithm clients=$clients repeat=$repeat"
  "$PYTHON_BIN" scripts/run_multiclient_pilot.py \
    --dataset humaneval \
    --algorithm "$algorithm" \
    --num_clients "$clients" \
    --pilot_samples 164 \
    --workload_mode replicated \
    --duration_s "$DURATION_SECONDS" \
    --warmup_s "$WARMUP_SECONDS" \
    --barrier_timeout_s 1800 \
    --workload_seed "$((3407 + repeat))" \
    --cpu_sets "$EDGE_CPUSETS" \
    --base_tag "$tag" \
    --summary_path "$summary" \
    --python_bin "$PYTHON_BIN" \
    --forward_arg=--seed \
    --forward_arg="$((3407 + repeat))" \
    --forward_arg=--max_generated_tokens \
    --forward_arg="$MAX_GENERATED_TOKENS" \
    --forward_arg=--threads \
    --forward_arg=2 \
    --forward_arg=--ctx_size \
    --forward_arg=1024 \
    --forward_arg=--draft_n_gpu_layers \
    --forward_arg=0 \
    --forward_arg=--evaluation_protocol \
    --forward_arg=sample_index \
    --forward_arg=--network_shaping_mode \
    --forward_arg=software \
    --forward_arg=--bandwidth_MBps \
    --forward_arg=2.5 \
    --forward_arg=--downlink_bandwidth_MBps \
    --forward_arg=25 \
    --forward_arg=--software_uplink_startup_ms \
    --forward_arg=25 \
    --forward_arg=--software_downlink_startup_ms \
    --forward_arg=0 \
    --forward_arg=--software_bandwidth_profile \
    --forward_arg="$LINK_PROFILE" \
    --forward_arg=--software_bandwidth_change_interval_s \
    --forward_arg=20 \
    --forward_arg=--server_timeout_s \
    --forward_arg=300 \
    "${algorithm_args[@]}"
}

for repeat in $(seq 1 "$REPEATS"); do
  for clients in $CLIENT_COUNTS; do
    for algorithm in $ALGORITHMS; do
      run_condition "$algorithm" "$clients" "$repeat"
    done
  done
done

echo "[done] summaries: $EDGE_DIR/exp-multi/multiclient/${RUN_LABEL}_*.json"
