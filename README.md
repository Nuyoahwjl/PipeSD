# PipeSD

PipeSD is a Parallel Speculative Decoding framework that boosts LLM inference throughput while keeping generation lossless. This repository contains two independent subdirectories for different deployment scenarios.

## Project Structure
- `edge/`: primary research/experiment code (runtime, scripts, benchmarks, and experiment outputs).
- `cloud/`: lightweight code and resources for cloud environments.

## Quick Start (edge)
```sh
cd edge
sh install.sh
```

Configure model and data paths in `edge/src/util.py`, then run:
```sh
sh scripts/run_para_sd.sh
```

For more details, see `edge/README.md`.
