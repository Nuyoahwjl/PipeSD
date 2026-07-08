#!/usr/bin/env bash
# Sweep single-threshold HSL and hybrid (single+multi) thresholds for pipesd, in batches.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# Bandwidths and sample range.
BANDWIDTHS_MBPS=(${BANDWIDTHS_MBPS:-2.5})
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-9}
BATCH_SIZE=${BATCH_SIZE:-2}

HSL_THRESHOLDS=(${HSL_THRESHOLDS:-0.7})
PIPESD_SINGLE_THRESHOLDS=(${PIPESD_SINGLE_THRESHOLDS:-0.8})
PIPESD_MULTI_THRESHOLDS=(${PIPESD_MULTI_THRESHOLDS:-0.5})
INIT_ALPHA=(${INIT_ALPHA:-0.5})
MULTIPLY_TIMES=(${MULTIPLY_TIMES:-0.7})

# Extra args to pass through, e.g. --max_len 512
EXTRA_ARGS="${EXTRA_ARGS:-}"

append_extra_args() {
  local -n arr_ref=$1
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_parts=($EXTRA_ARGS)
    arr_ref+=( "${extra_parts[@]}" )
  fi
}

for ((start=START_INDEX; start<=END_INDEX; start+=BATCH_SIZE)); do
  end=$((start + BATCH_SIZE - 1))
  if (( end > END_INDEX )); then
    end=$END_INDEX
  fi

  for bw in "${BANDWIDTHS_MBPS[@]}"; do

    cmd=( "$PYTHON_BIN" app/run_edge.py
          --algorithm vanilla
          --verify_strategy fixed-num
          --gamma 6
          --bandwidth_MBps "$bw"
          --start_index_of_sample "$start"
          --end_index_of_sample "$end"
        )
    append_extra_args cmd
    echo "[run] vanilla gamma=6 bw=${bw}MB samples=${start}-${end}"
    "${cmd[@]}"
    sleep 2

    

    # # HSL: sweep single thresholds.
    for single_thresh in "${HSL_THRESHOLDS[@]}"; do
      cmd=( "$PYTHON_BIN" app/run_edge.py
        --algorithm hsl
        --verify_strategy single-token
        --verify_thresh_single "$single_thresh"
        --bandwidth_MBps "$bw"
        --start_index_of_sample "$start"
        --end_index_of_sample "$end"
      )
      append_extra_args cmd
      echo "[run] hsl single=$single_thresh bw=${bw}MB samples=${start}-${end}"
      "${cmd[@]}"
      sleep 2
    done

    # pipesd: hybrid thresholds (single x multi).
    for single_thresh in "${PIPESD_SINGLE_THRESHOLDS[@]}"; do
      for multi_thresh in "${PIPESD_MULTI_THRESHOLDS[@]}"; do
        cmd=( "$PYTHON_BIN" app/run_edge.py
          --algorithm pipesd
          --verify_strategy hybrid
          --verify_thresh_single "$single_thresh"
          --verify_thresh_multi "$multi_thresh"
          --bandwidth_MBps "$bw"
          --start_index_of_sample "$start"
          --end_index_of_sample "$end"
        )
        append_extra_args cmd
        echo "[run] pipesd single=$single_thresh multi=$multi_thresh bw=${bw}MB samples=${start}-${end}"
        "${cmd[@]}"
        sleep 2
      done
    done

    for init_alpha in "${INIT_ALPHA[@]}"; do
      for multiply_times in "${MULTIPLY_TIMES[@]}"; do
        cmd=( "$PYTHON_BIN" app/run_edge.py
              --algorithm edgeLLM
              --init_alpha "$init_alpha"
              --multiply_times "$multiply_times"
              --bandwidth_MBps "$bw"
              --start_index_of_sample "$start"
              --end_index_of_sample "$end"
            )
        append_extra_args cmd
        echo "[run] edgeLLM bw=${bw}MB samples=${start}-${end}"
        "${cmd[@]}"
        sleep 2
      done
    done

    

  done
done
