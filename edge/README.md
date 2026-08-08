# Edge (experiment runner)

Client-side code that drives speculative decoding experiments, sends proposals to the cloud service, and writes results.

Detailed repository walkthrough: `docs/repo-summary.md`

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
- Set `PIPE_SD_SERVER_URL` to the cloud service URL (for example `http://127.0.0.1:8000`).
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
- `--prob_transport`: `full` (the unchanged default) or `lazy_distribution`.
  Lazy mode sends only the probability of each draft token and uploads one
  float32 vocabulary distribution if that token is rejected.

Upload behavior is algorithm-specific: Vanilla/HSL upload the complete draft
once at NAV; EdgeLLM uploads proactively only while NAV is pending, in the
current moving-average `N-hat` window; PipeSD applies its DP batch schedule both
before NAV and while NAV is pending. Bayesian threshold optimization is a
PipeSD-only entrypoint.

## Sweep scripts
In `edge/scripts/`:
- `sweep.sh`, `sweep_gsm8k.sh`: run multi-config sweeps
- `vary_bandwidth.sh`, `vary_bandwidth_gsm8k.sh`: dynamic bandwidth
- `ablation_study.sh`: ablation variants

Example (from `edge/`):
```bash
bash scripts/sweep.sh
```

To compare the original PipeSD probability upload against lazy distribution
transfer on HumanEval with identical thresholds and network settings:

```bash
bash scripts/compare_lazy_distribution_humaneval.sh
```

Use `TARGET_OUTPUT_TOKENS=100` for a pilot. The default is the 1000-token paper
protocol. The comparison starts the serial cloud backend by default; use
`START_CLOUD=0` only when explicitly reusing an already-running server.

## Outputs
Results are written under `edge/exp/` as JSON (one entry per run).

For the paper-aligned 1000-token Table 1 protocol, BO procedure, four-method
commands, metrics, and correctness evaluation, see
[`docs/table1-paper-protocol.md`](docs/table1-paper-protocol.md).

For the deployment comparison requested by the reproduction assignment—Pure
Cloud, Pure Edge, serial Edge-Cloud SD (the existing Vanilla algorithm), and
PipeSD—see [`docs/four-mode-evaluation.md`](docs/four-mode-evaluation.md).
