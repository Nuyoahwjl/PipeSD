#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_humaneval.sh"
cd "$EDGE_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN:-<not found>}" >&2
  exit 1
fi

# Paper Figure 5 varies only the Scenario-1 uplink at 10/20/40/80 Mbps. The
# current CLI uses decimal MB/s, hence 1.25/2.5/5/10 below. Algorithm and cloud
# parameters come from eval_humaneval.sh; this wrapper fixes only the bandwidth
# experiment protocol, probability transport, and backend.
UPLINK_BANDWIDTHS_MBPS="${UPLINK_BANDWIDTHS_MBPS:-${BANDWIDTHS_MBPS:-1.25 2.5 5 10}}"
read -r -a bandwidths <<< "$UPLINK_BANDWIDTHS_MBPS"
if (( ${#bandwidths[@]} == 0 )); then
  echo "UPLINK_BANDWIDTHS_MBPS must contain at least one bandwidth." >&2
  exit 1
fi
for bw in "${bandwidths[@]}"; do
  if ! "$PYTHON_BIN" -c \
    'import math,sys; value=float(sys.argv[1]); assert math.isfinite(value) and value > 0' \
    "$bw" >/dev/null 2>&1; then
    echo "Invalid uplink bandwidth (expected positive MB/s): $bw" >&2
    exit 1
  fi
done

REPEATS="${REPEATS:-1}"
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be a positive integer: $REPEATS" >&2
  exit 1
fi

SEED="${SEED:-3407}"
START_INDEX="${START_INDEX:-50}"
END_INDEX="${END_INDEX:-163}"
EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL:-paper_table1}"
TARGET_OUTPUT_TOKENS="${TARGET_OUTPUT_TOKENS:-1000}"
DOWNLINK_BANDWIDTH_MBPS="${DOWNLINK_BANDWIDTH_MBPS:-25}"
SOFTWARE_UPLINK_STARTUP_MS="${SOFTWARE_UPLINK_STARTUP_MS:-25}"
SOFTWARE_DOWNLINK_STARTUP_MS="${SOFTWARE_DOWNLINK_STARTUP_MS:-0}"
NETWORK_SHAPING_MODE="${NETWORK_SHAPING_MODE:-software}"
ALTERNATE_BANDWIDTH_ORDER="${ALTERNATE_BANDWIDTH_ORDER:-1}"
DRY_RUN="${DRY_RUN:-0}"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d-%H%M%S)}"
BASE_RESULT_TAG="${RESULT_TAG:-bandwidth_humaneval_full_serial_${SWEEP_ID}}"
USER_EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "$NETWORK_SHAPING_MODE" != "software" ]]; then
  echo "This sweep requires NETWORK_SHAPING_MODE=software." >&2
  echo "In os mode, changing --bandwidth_MBps does not configure tc/QoS." >&2
  exit 1
fi
if [[ "$EVALUATION_PROTOCOL" != "paper_table1" ]]; then
  echo "This sweep requires EVALUATION_PROTOCOL=paper_table1." >&2
  exit 1
fi
if ! [[ "$TARGET_OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TARGET_OUTPUT_TOKENS must be a positive integer: $TARGET_OUTPUT_TOKENS" >&2
  exit 1
fi
if ! [[ "$START_INDEX" =~ ^[0-9]+$ && "$END_INDEX" =~ ^[0-9]+$ ]] \
  || (( START_INDEX > END_INDEX )); then
  echo "Invalid sample range: START_INDEX=$START_INDEX END_INDEX=$END_INDEX" >&2
  exit 1
fi
if [[ "$ALTERNATE_BANDWIDTH_ORDER" != "0" && "$ALTERNATE_BANDWIDTH_ORDER" != "1" ]]; then
  echo "ALTERNATE_BANDWIDTH_ORDER must be 0 or 1." >&2
  exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 1
fi
if [[ "$USER_EXTRA_ARGS" =~ (^|[[:space:]])--prob_transport($|[=[:space:]]) ]]; then
  echo "Do not set --prob_transport through EXTRA_ARGS; this sweep fixes it to full." >&2
  exit 1
fi
EVAL_EXTRA_ARGS="${USER_EXTRA_ARGS:+$USER_EXTRA_ARGS }--prob_transport full"

echo "[configuration] dataset=humaneval transport=full backend=serial protocol=paper_table1"
echo "[configuration] uplink=10/20/40/80Mbps (${UPLINK_BANDWIDTHS_MBPS} MB/s) downlink=${DOWNLINK_BANDWIDTH_MBPS}MB/s"
echo "[configuration] samples=${START_INDEX}-${END_INDEX} accepted_token_budget=${TARGET_OUTPUT_TOKENS} repeats=${REPEATS}"

run_bandwidth() {
  local repeat="$1"
  local repeat_seed="$2"
  local result_tag="$3"
  local bw="$4"
  local bw_label="${bw//./p}"
  local paper_mbps
  paper_mbps="$($PYTHON_BIN -c 'import sys; print(f"{float(sys.argv[1]) * 8:g}")' "$bw")"

  local -a cmd=(
    env
    "PYTHON_BIN=$PYTHON_BIN"
    DATASET=humaneval
    "SEED=$repeat_seed"
    "BANDWIDTH_MBPS=$bw"
    "DOWNLINK_BANDWIDTH_MBPS=$DOWNLINK_BANDWIDTH_MBPS"
    NETWORK_SHAPING_MODE=software
    "SOFTWARE_UPLINK_STARTUP_MS=$SOFTWARE_UPLINK_STARTUP_MS"
    "SOFTWARE_DOWNLINK_STARTUP_MS=$SOFTWARE_DOWNLINK_STARTUP_MS"
    SOFTWARE_BANDWIDTH_PROFILE=
    "START_INDEX=$START_INDEX"
    "END_INDEX=$END_INDEX"
    EVALUATION_PROTOCOL=paper_table1
    "TARGET_OUTPUT_TOKENS=$TARGET_OUTPUT_TOKENS"
    CLOUD_BACKEND=serial
    "EXTRA_ARGS=$EVAL_EXTRA_ARGS"
    "RESULT_TAG=$result_tag"
    "RUN_LABEL=${SWEEP_ID}_humaneval_full_serial_r${repeat}_bw${bw_label}"
    bash "$EVAL_SCRIPT"
  )

  echo "[sweep] dataset=humaneval repeat=$repeat seed=$repeat_seed uplink=${paper_mbps}Mbps/${bw}MB/s tag=$result_tag"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  repeat_seed=$((SEED + repeat - 1))
  result_tag="$BASE_RESULT_TAG"
  if (( REPEATS > 1 )); then
    result_tag="${BASE_RESULT_TAG}_r${repeat}"
  fi

  if [[ "$ALTERNATE_BANDWIDTH_ORDER" == "1" ]] && (( repeat % 2 == 0 )); then
    for ((index = ${#bandwidths[@]} - 1; index >= 0; index--)); do
      run_bandwidth "$repeat" "$repeat_seed" "$result_tag" "${bandwidths[$index]}"
    done
  else
    for bw in "${bandwidths[@]}"; do
      run_bandwidth "$repeat" "$repeat_seed" "$result_tag" "$bw"
    done
  fi
done
