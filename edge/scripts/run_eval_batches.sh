#!/usr/bin/env bash
# Run HumanEval batches for the four algorithms over indices 20-159 in chunks of 10,
# sweeping the specified bandwidths.
set -euo pipefail

# ALGORITHMS=(vanilla vanilla-with-merge-no-send vanilla-with-merge vanilla-with-send hsl pipesd)
ALGORITHMS=(edgeLLM)
# ALGORITHMS=(vanilla vanilla-with-merge-no-send vanilla-with-merge vanilla-with-send hsl pipesd edgeLLM)
BANDWIDTHS_MBPS=(1 2.5 20)
# BANDWIDTHS_MBPS=(10)
SINGLE_THRESHOLDS=(0.99)
# BANDWIDTHS_MBPS=(8)
START_INDEX=0
END_INDEX=59
BATCH_SIZE=2

for ((start=START_INDEX; start<=END_INDEX; start+=BATCH_SIZE)); do
    end=$((start + BATCH_SIZE - 1))
    if (( end > END_INDEX )); then
        end=$END_INDEX
    fi

    for bw in "${BANDWIDTHS_MBPS[@]}"; do
        for alg in "${ALGORITHMS[@]}"; do
            for thresh in "${SINGLE_THRESHOLDS[@]}"; do
                echo "Running eval for algorithm: $alg, bandwidth: ${bw}MBps, single threshold: $thresh, indices: $start to $end"
                python -m benchmark.eval_Draft \
                    --algorithm="${alg}" \
                    --gamma=6 \
                    --bandwidth_MBps="${bw}" \
                    --start_index_of_sample="${start}" \
                    --end_index_of_sample="${end}" \
                    --verify_thresh_single="${thresh}"
                sleep 2
            done
        done
    done
done
