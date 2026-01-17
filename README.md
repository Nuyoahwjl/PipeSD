# PipeSD

PipeSD is a cloud-edge collaborative inference framework based on speculative decoding, with two core mechanisms:
- Token-batch pipeline scheduling
- Dual-threshold NAV (non-autoregressive verification)

The project includes a cloud verification service and an edge-side runner for experiments.

## Repository layout
- `cloud/`: FastAPI service that hosts the target model and verifies speculative tokens
- `edge/`: generates draft tokens autoregressively and sends them to the cloud for NAV

## Setup
Cloud environment requirements:
- Ubuntu 22.04
- CUDA 12.1
- Python 3.10

Edge environment notes:
- Windows 11 24H2 (CPU-only) is used in our setup.
- Ubuntu 22.04 should also work.

Each side has its own install script:
- Cloud: `cloud/install.sh`
- Edge: `edge/install.sh`

Use them as references for required packages.

## Typical flow
1) Start the cloud service on the cloud server (see `cloud/README.md`).
2) Run an edge evaluation on the edge device (see `edge/README.md`).

## Notes
- Ubuntu >= 22.04 is recommended.
- The edge client talks to a hard-coded cloud URL in `edge/src/engine.py`. Update it to match your server address.
