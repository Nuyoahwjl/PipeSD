#!/usr/bin/env bash
# Sweep verify thresholds for hybrid/single/multi strategies at fixed bandwidth.
# Default: bandwidth=5MB, samples 0-4, single thresholds 0.80-0.95 step 0.05,
# multi thresholds 0.05-0.35 step 0.10, algorithm=vanilla-with-merge.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ALGORITHM="${ALGORITHM:-vanilla-with-merge}"
BANDWIDTH="${BANDWIDTH:-5}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-4}"
SINGLE_VALUES="${SINGLE_VALUES:-}"
SINGLE_START="${SINGLE_START:-0.90}"
SINGLE_STOP="${SINGLE_STOP:-0.96}"
SINGLE_STEP="${SINGLE_STEP:-0.03}"
MULTI_VALUES="${MULTI_VALUES:-}"
MULTI_START="${MULTI_START:-0.10}"
MULTI_STOP="${MULTI_STOP:-0.90}"
MULTI_STEP="${MULTI_STEP:-0.05}"
EXP_PREFIX="${EXP_PREFIX:-thresh_sweep}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

build_values() {
  local raw="$1" start="$2" stop="$3" step="$4"
  if [[ -n "$raw" ]]; then
    echo "$raw"
  else
    seq "$start" "$step" "$stop"
  fi
}

format_label() {
  local val="$1"
  printf "%s" "${val/./p}"
}

SINGLE_LIST=($(build_values "$SINGLE_VALUES" "$SINGLE_START" "$SINGLE_STOP" "$SINGLE_STEP"))
MULTI_LIST=($(build_values "$MULTI_VALUES" "$MULTI_START" "$MULTI_STOP" "$MULTI_STEP"))
DEFAULT_SINGLE="${SINGLE_LIST[0]}"
DEFAULT_MULTI="${MULTI_LIST[0]}"
BANDWIDTH_LABEL=$(printf "%g" "$BANDWIDTH")

# Strategies to run; adjust as needed.
# STRATEGIES=("hybrid" "single-token" "multiple-tokens")
STRATEGIES=("multiple-tokens")

for strategy in "${STRATEGIES[@]}"; do
  case "$strategy" in
    "single-token")
      for single_thresh in "${SINGLE_LIST[@]}"; do
        s_label=$(format_label "$single_thresh")
        exp_name="${EXP_PREFIX}_s${s_label}"
        cmd=( "$PYTHON_BIN" -m benchmark.eval_Draft
          --algorithm "$ALGORITHM"
          --bandwidth_MBps "$BANDWIDTH"
          --start_index_of_sample "$START_INDEX"
          --end_index_of_sample "$END_INDEX"
          --verify_strategy "$strategy"
          --verify_thresh_single "$single_thresh"
          --verify_thresh_multi "$DEFAULT_MULTI"
          # --exp_name "$exp_name"
        )
        if [[ -n "$EXTRA_ARGS" ]]; then
          # shellcheck disable=SC2206
          extra_parts=($EXTRA_ARGS)
          cmd+=( "${extra_parts[@]}" )
        fi
        printf "[run] strategy=%s single=%s -> %s\n" \
          "$strategy" "$single_thresh" "$(printf "%q " "${cmd[@]}")"
        [[ "$DRY_RUN" == "1" ]] || "${cmd[@]}"
      done
      ;;
    "multiple-tokens")
      for multi_thresh in "${MULTI_LIST[@]}"; do
        m_label=$(format_label "$multi_thresh")
        exp_name="${EXP_PREFIX}_m${m_label}"
        cmd=( "$PYTHON_BIN" -m benchmark.eval_Draft
          --algorithm "$ALGORITHM"
          --bandwidth_MBps "$BANDWIDTH"
          --start_index_of_sample "$START_INDEX"
          --end_index_of_sample "$END_INDEX"
          --verify_strategy "$strategy"
          --verify_thresh_single "$DEFAULT_SINGLE"
          --verify_thresh_multi "$multi_thresh"
          # --exp_name "$exp_name"
        )
        if [[ -n "$EXTRA_ARGS" ]]; then
          # shellcheck disable=SC2206
          extra_parts=($EXTRA_ARGS)
          cmd+=( "${extra_parts[@]}" )
        fi
        printf "[run] strategy=%s multi=%s -> %s\n" \
          "$strategy" "$multi_thresh" "$(printf "%q " "${cmd[@]}")"
        [[ "$DRY_RUN" == "1" ]] || "${cmd[@]}"
      done
      ;;
    "hybrid")
      for single_thresh in "${SINGLE_LIST[@]}"; do
        for multi_thresh in "${MULTI_LIST[@]}"; do
          s_label=$(format_label "$single_thresh")
          m_label=$(format_label "$multi_thresh")
          exp_name="${EXP_PREFIX}_s${s_label}_m${m_label}"
          cmd=( "$PYTHON_BIN" -m benchmark.eval_Draft
            --algorithm "$ALGORITHM"
            --bandwidth_MBps "$BANDWIDTH"
            --start_index_of_sample "$START_INDEX"
            --end_index_of_sample "$END_INDEX"
            --verify_strategy "$strategy"
            --verify_thresh_single "$single_thresh"
            --verify_thresh_multi "$multi_thresh"
            # --exp_name "$exp_name"
          )
          if [[ -n "$EXTRA_ARGS" ]]; then
            # shellcheck disable=SC2206
            extra_parts=($EXTRA_ARGS)
            cmd+=( "${extra_parts[@]}" )
          fi
          printf "[run] strategy=%s single=%s multi=%s -> %s\n" \
            "$strategy" "$single_thresh" "$multi_thresh" "$(printf "%q " "${cmd[@]}")"
          [[ "$DRY_RUN" == "1" ]] || "${cmd[@]}"
        done
      done
      ;;
  esac
done
