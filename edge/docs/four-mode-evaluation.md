# Pure Cloud / Pure Edge / Serial SD / PipeSD evaluation

This extension adds two standalone single-model baselines without changing the
existing speculative-decoding engine or cloud API:

| Mode | Model and placement | Network/NAV in measured decode time |
| --- | --- | --- |
| Pure Cloud | target model, cloud GPU (`n_gpu_layers=-1`) | no; local cloud-side decode baseline |
| Pure Edge | draft model, edge CPU (`n_gpu_layers=0`) | no |
| Serial Edge-Cloud SD | existing `vanilla` algorithm | yes; edge draft, software-shaped upload, cloud NAV |
| PipeSD | existing `pipesd` algorithm | yes; pipelined upload, dual-threshold NAV, DP batching |

The paper evaluates Vanilla, HSL, EdgeLLM, and PipeSD. Pure Cloud and Pure Edge
are assignment-level deployment baselines rather than rows from the paper's
Table 1. The single-model runner therefore stays separate from `src/engine.py`.
It excludes model loading and prompt prefill from TPT, reports prefill
separately, and never contacts port 8000.

## One-server procedure

Run commands from `edge/`. Keep `DATASET`, `RESULT_TAG`, `SEED`, sample range,
generation cap, and accepted-token budget identical across every phase.

### 1. Pure Edge

The cloud service may be stopped or running; Pure Edge uses only the CPU draft
model.

```bash
DATASET=humaneval PHASE=pure_edge bash scripts/eval_four_modes.sh
DATASET=gsm8k PHASE=pure_edge bash scripts/eval_four_modes.sh
```

### 2. Pure Cloud

Stop `python -m src.speculative_server` first. Otherwise the standalone runner
loads a second copy of the target model and can exhaust GPU memory.

```bash
DATASET=humaneval PHASE=pure_cloud bash scripts/eval_four_modes.sh
DATASET=gsm8k PHASE=pure_cloud bash scripts/eval_four_modes.sh
```

Pure Cloud measures the target model where it is hosted. It intentionally does
not add artificial prompt/output transfer. This is a model-serving lower bound,
not an end-user request-latency number. The comparison report prints this
warning explicitly.

### 3. Serial Edge-Cloud SD and PipeSD

Start the existing cloud service in a separate terminal:

```bash
cd cloud
python -m src.speculative_server
```

Then run the two collaborative methods from `edge/`:

```bash
export PIPE_SD_SERVER_URL=http://127.0.0.1:8000
DATASET=humaneval PHASE=collaborative bash scripts/eval_four_modes.sh
DATASET=gsm8k PHASE=collaborative bash scripts/eval_four_modes.sh
```

The defaults reproduce Scenario 1's software network settings: 2.5 MB/s
uplink, 25 MB/s downlink, 25 ms upload startup, CPU draft
`DRAFT_N_GPU_LAYERS=0`, and 1000 generated/accepted tokens. Override PipeSD
thresholds with `PIPESD_SINGLE_THRESH`, `PIPESD_MULTI_THRESH`, and optionally
record a BO file with `PIPESD_BO_CONFIG`.

### 4. Compare

```bash
DATASET=humaneval PHASE=compare bash scripts/eval_four_modes.sh
DATASET=gsm8k PHASE=compare bash scripts/eval_four_modes.sh
```

Reports are written beside the dataset's experiment results under
`exp/exp__wjl/<dataset>/comparison` as Markdown, CSV, and
JSON. The comparator selects the newest matching run for each method. To
compare exact files or a different directory, invoke it directly:

```bash
python scripts/compare_four_modes.py PATH... \
  --dataset humaneval \
  --result-tag four_mode_s1_paper \
  --output-dir exp/exp__wjl/humaneval/comparison/manual
```

## Reported metrics

The common table contains weighted TPT, throughput, speedup over serial SD,
P50/P95/P99 token latency, true time to the first accepted token (TTFT), total
time, and token/sample counts. The efficiency table contains measured energy per 100
tokens with its hardware scope, NAV frequency, mean draft length, acceptance
rate, rollback rate, and actual batch size. The network table contains total
upload/download volume, upload volume per 100 output tokens, upload count,
average upload size, software-link queue time, and link service time. Cap-hit
and EOS rates remain as runtime termination diagnostics; no correctness metric
or completion correctness score is produced. The selected raw generations are
still exported, one JSONL file per mode, as
`<dataset>_<mode>_completions.jsonl`. These files preserve `task_id`,
`completion`, `method`, `run_id`, and `sample_index`; the comparison script does
not execute or score the completions.

Pure modes have no speculative rounds, so NAV, draft, acceptance, rollback,
and batch fields are shown as `—`, not zero. `missing` means an applicable
metric was not recorded and the affected mode should be rerun. GPU energy is sampled with NVML for Pure
Cloud. Pure Edge CPU package energy is reported only when Linux exposes Intel
RAPL counters; unavailable Pure Edge energy remains `N/A` and is not replaced
with zero or an estimate. Energy values with different scopes must not be
presented as a whole-system ranking.

Pure Cloud is labeled `model-only` because it excludes client-cloud transport,
and Pure Edge is labeled `local-only`. Collaborative artifacts created before
the TTFT instrumentation change will show `missing`; rerun Serial SD and PipeSD
with the same result tag to populate TTFT.
