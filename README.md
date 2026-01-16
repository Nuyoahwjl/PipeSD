# PipeSD

Speculative decoding project split into a cloud verification service and an edge-side runner for experiments.

## Repository layout
- `cloud/`: FastAPI service that hosts the target model and verifies speculative tokens
- `edge/`: client/runner that drives experiments and logs results
- `README.md`: top-level overview (this file)

## What lives where
- **Models**
  - `cloud/pre_models/`: target-model weights used by the cloud service
  - `edge/pre_models/`: draft-model weights used by the edge runner
  - `cloud/hfd.sh`, `edge/hfd.sh`: helper scripts for model downloads
- **Data & results** (edge-side)
  - `edge/data/`: datasets in JSONL (e.g., `humaneval.jsonl`, `gsm8k.jsonl`)
  - `edge/exp/`: experiment outputs (created at runtime)
  - `edge/figs/`: pre-generated figures
- **Code**
  - `cloud/src/`: FastAPI server + utilities
  - `edge/src/`: client/runner, comms, merge logic, optimization
  - `edge/benchmark/`: evaluation entrypoint
  - `edge/scripts/`: sweep and ablation scripts

## Setup
Each side has its own install script:
- Cloud: `cloud/install.sh`
- Edge: `edge/install.sh`

Use them as references for required packages; adapt to your environment as needed.

## Typical flow
1) Start the cloud service (see `cloud/README.md`).
2) Run an edge evaluation or sweep (see `edge/README.md`).
3) Results are written under `edge/exp/`.

## Notes
- The edge client talks to a hard-coded cloud URL in `edge/src/engine.py`. Update it to match your server address.
- Power-integral metrics are produced by the cloud service if NVML is available.
