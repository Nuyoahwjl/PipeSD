#!/usr/bin/env bash
# Sweep single-threshold HSL and hybrid (single+multi) thresholds for pipesd, in batches.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# Bandwidths and sample range.
BANDWIDTHS_MBPS=(${BANDWIDTHS_MBPS:-2.5})
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-9}
BATCH_SIZE=${BATCH_SIZE:-2}

# Thresholds.
INIT_ALPHA=(${INIT_ALPHA:-0.4})
MULTIPLY_TIMES=(${MULTIPLY_TIMES:-0.4 0.6 })

for ((start=START_INDEX; start<=END_INDEX; start+=BATCH_SIZE)); do
  end=$((start + BATCH_SIZE - 1))
  if (( end > END_INDEX )); then
    end=$END_INDEX
  fi

  # cmd=( "$PYTHON_BIN" -m benchmark.eval_Draft
  #   --algorithm pipesd
  #   --verify_strategy multiple-tokens
  #   --bandwidth_MBps "${BANDWIDTHS_MBPS[0]}"
  #   --init_alpha 0.95
  #   --multiply_times 0.97
  #   --start_index_of_sample "$start"
  #   --end_index_of_sample "$end"
  #   --ablation_study
  # )
  # echo "[run] pipesd init_alpha=0.95 multiply_times=0.97 samples=${start}-${end}"
  # "${cmd[@]}"
  # sleep 2

  # cmd=( "$PYTHON_BIN" -m benchmark.eval_Draft
  #   --algorithm pipesd
  #   --verify_strategy single-token
  #   --verify_thresh_single 0.99
  #   --bandwidth_MBps "${BANDWIDTHS_MBPS[0]}"
  #   --start_index_of_sample "$start"
  #   --end_index_of_sample "$end"
  #   --ablation_study
  # )
  # echo "[run] pipesd verify_thresh_single=0.99 samples=${start}-${end}"
  # "${cmd[@]}"
  # sleep 2

  # cmd=( "$PYTHON_BIN" -m benchmark.eval_Draft
  #   --algorithm pipesd
  #   --verify_strategy fixed-num
  #   --gamma 6
  #   --bandwidth_MBps "${BANDWIDTHS_MBPS[0]}"
  #   --start_index_of_sample "$start"
  #   --end_index_of_sample "$end"
  #   --ablation_study
  # )
  # echo "[run] pipesd gamma=6 samples=${start}-${end}"
  # "${cmd[@]}"
  # sleep 2

  cmd=( "$PYTHON_BIN" -m benchmark.eval_Draft
    --algorithm pipesd
    --bandwidth_MBps "${BANDWIDTHS_MBPS[0]}"
    --start_index_of_sample "$start"
    --end_index_of_sample "$end"
    --ablation_study
    --nomerge
  )
  echo "[run] pipesd gamma=6 samples=${start}-${end}"
  "${cmd[@]}"
  sleep 2

done
