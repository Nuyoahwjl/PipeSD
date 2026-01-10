# PipeSD: Parallel Speculative Decoding

This repository implements PipeSD, a parallel speculative decoding framework for LLM inference. It uses pre-verify and post-verify strategies to adapt draft length while keeping generation lossless and training-free.

## Highlights
- Parallel speculative decoding with adaptive draft length
- Lossless generation, no extra training or memory
- Works with draft-then-verify setups (EAGLE, Medusa style)
- GGUF support via llama-cpp-python
- Benchmarks for HumanEval, GSM8K, MT-Bench, and MGSM

## Project Structure
- `src/`: PipeSD runtime
  - `engine.py`: draft/target interplay and decoding loops
  - `speculator.py`: verification logic
  - `util.py`: model/data paths and shared utilities (edit before running)
- `benchmark/`: evaluation entry points
- `scripts/`: reproducible pipelines used in experiments
- `exp/`, `exp_bw/`, `exp_abl/`, `exp_bo/`: experiment outputs and plots
- `logs/`: benchmark logs and JSON summaries
- `static/`, `applications.py`: optional demo assets/UI harness

## Setup
```sh
sh install.sh
```

Then update model and data paths in `src/util.py` (required before running any pipeline).

## Quick Start
```sh
sh scripts/run_para_sd.sh
```

Autoregressive baseline:
```sh
sh scripts/run_ar.sh
```

Custom experiment template:
```sh
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
  benchmark/eval_humaneval.py --eval_mode para_sd --gamma 5 -n 1 \
  -e H_PSD_codellama_7_70b --draft_model codellama-7b --target_model codellama-70b
```

## Analysis and Plots
- Summary tables: `exp/humaneval/summary_tables.py`
- Bandwidth plots: `exp_bw/bw.py`

## Testing and Validation
- Throughput limiter: `python scripts/test_bandwidth_limiter.py`
- Lint/compile: `ruff check` or `python -m compileall src`
- Algorithm checks: run `benchmark/eval_*.py` in both `para_sd` and `ar` modes and compare JSON summaries in `logs/`

## Notes
- Keep proprietary checkpoints outside the repo and reference them via absolute paths in `src/util.py`.
- Outputs for benchmarks are written to `logs/` and experiment folders under `exp/`.
