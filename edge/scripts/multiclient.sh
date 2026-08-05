#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EDGE_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/pipesd/bin/python}"
export PIPE_SD_SERVER_URL="${PIPE_SD_SERVER_URL:-http://127.0.0.1:8000}"

DATASET="${DATASET:-humaneval}"
PILOT_SAMPLES="${PILOT_SAMPLES:-32}"
BASE_TAG="${BASE_TAG:-vanilla_c8_n32}"
SUMMARY_PATH="${SUMMARY_PATH:-exp-multi/multiclient/${BASE_TAG}.json}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if command -v curl >/dev/null 2>&1; then
  echo "[preflight] Checking Cloud health at ${PIPE_SD_SERVER_URL%/}/health"
  curl --fail --silent --show-error --max-time 10 \
    "${PIPE_SD_SERVER_URL%/}/health"
  echo
else
  echo "[preflight] curl is unavailable; skipping Cloud health check" >&2
fi

echo "[run] Vanilla HumanEval: clients=8 samples=$PILOT_SAMPLES tag=$BASE_TAG"

"$PYTHON_BIN" scripts/run_multiclient_pilot.py \
  --dataset "$DATASET" \
  --algorithm vanilla \
  --num_clients 8 \
  --pilot_samples "$PILOT_SAMPLES" \
  --workload_mode distinct \
  --base_tag "$BASE_TAG" \
  --python_bin "$PYTHON_BIN" \
  --summary_path "$SUMMARY_PATH" \
  --forward_arg=--seed \
  --forward_arg=3407 \
  --forward_arg=--max_generated_tokens \
  --forward_arg=128 \
  --forward_arg=--threads \
  --forward_arg=2 \
  --forward_arg=--verify_strategy \
  --forward_arg=fixed-num \
  --forward_arg=--gamma \
  --forward_arg=6 \
  --forward_arg=--verify_num \
  --forward_arg=6 \
  --forward_arg=--evaluation_protocol \
  --forward_arg=sample_index \
  --forward_arg=--draft_n_gpu_layers \
  --forward_arg=0 \
  --forward_arg=--server_timeout_s \
  --forward_arg=300

  # --forward_arg=--verify_strategy \
  # --forward_arg=hybrid \
  # --forward_arg=--verify_thresh_single \
  # --forward_arg=0.3514 \
  # --forward_arg=--verify_thresh_multi \
  # --forward_arg=0.9 \

echo "[done] Summary written to $EDGE_DIR/$SUMMARY_PATH"
