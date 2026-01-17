# Edge (experiment runner)

Client-side code that drives speculative decoding experiments, sends proposals to the cloud service, and writes results.

## Key files
- `src/engine.py`: core speculative decoding logic and cloud communication
- `src/comm.py`: bandwidth-limited sender
- `src/merge.py`: merge planning for speculative batches
- `src/bayes.py`: Bayesian threshold optimization helpers
- `src/util.py`: argument parsing, dataset settings, helpers
- `app/run_edge.py`: main evaluation entrypoint
- `scripts/`: sweep and ablation scripts
- `install.sh`: dependency

## Directory layout
- `data/`: datasets (JSONL). Examples: `humaneval.jsonl`, `gsm8k.jsonl`
- `pre_models/`: draft-model GGUF files
- `exp/`: experiment outputs (created at runtime)
- `figs/`: exp results figures

## Configuration
- Cloud URL is hard-coded in `src/engine.py` (`URL = ...`). Update it to point at your running cloud service.
- CLI flags are defined in `src/util.py` (e.g. `--dataset`, `--algorithm`, thresholds, bandwidth settings).

## Running an evaluation
From the `edge/` directory:
```bash
python app/run_edge.py --dataset humaneval --algorithm pipesd
```

Common arguments (see `src/util.py` for the full list):
- `--dataset`: `humaneval`, `gsm8k`
- `--algorithm`: `vanilla`, `edgeLLM`, `hsl`, `pipesd`
- `--verify_strategy`: `fixed-num`, `single-token`, `multiple-tokens`, `hybrid`
- `--bandwidth_MBps`: network bandwidth limit used by the sender

## Sweep scripts
In `edge/scripts/`:
- `sweep.sh`, `sweep_gsm8k.sh`: run multi-config sweeps
- `vary_bandwidth.sh`, `vary_bandwidth_gsm8k.sh`: dynamic bandwidth
- `ablation_study.sh`: ablation variants

Example (from `edge/`):
```bash
bash scripts/sweep.sh
```

## Outputs
Results are written under `edge/exp/` as JSON (one entry per run).
