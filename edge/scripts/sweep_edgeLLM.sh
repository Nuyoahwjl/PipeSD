#!/usr/bin/env bash
# Sweep single-threshold HSL and hybrid (single+multi) thresholds for pipesd, in batches.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# Bandwidths and sample range.
BANDWIDTHS_MBPS=(${BANDWIDTHS_MBPS:-10})
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-59}
BATCH_SIZE=${BATCH_SIZE:-2}

# Thresholds.
INIT_ALPHA=(${INIT_ALPHA:-0.98})
MULTIPLY_TIMES=(${MULTIPLY_TIMES:-0.99})

for ((start=START_INDEX; start<=END_INDEX; start+=BATCH_SIZE)); do
  end=$((start + BATCH_SIZE - 1))
  if (( end > END_INDEX )); then
    end=$END_INDEX
  fi

  
  for init_alpha in "${INIT_ALPHA[@]}"; do
    for multiply_times in "${MULTIPLY_TIMES[@]}"; do
      cmd=( "$PYTHON_BIN" -m benchmark.eval_Draft
        --algorithm edgeLLM
        --verify_strategy multiple-tokens
        --init_alpha "$init_alpha"
        --multiply_times "$multiply_times"
        --start_index_of_sample "$start"
        --end_index_of_sample "$end"
      )
      echo "[run] edgeLLM init_alpha=$init_alpha multiply_times=$multiply_times samples=${start}-${end}"
      "${cmd[@]}"
      sleep 2
    done
  done
done
