# Four-mode local and collaborative speculative-decoding evaluation

All four modes now use the same dataset-specific draft/target pair and stop
after exactly 1,000 target-accepted draft tokens.

| Mode | Draft | Target | Verification path |
| --- | --- | --- | --- |
| Pure Cloud SD | cloud | cloud | fixed-window synchronous local handoff |
| Pure Edge SD | edge | edge | fixed-window synchronous local handoff |
| Serial Edge-Cloud SD | edge | cloud | fixed-window WAN upload, NAV, then wait |
| PipeSD | edge | cloud | threshold NAV, DP batching and overlap |

The first three modes use the Vanilla fixed-window policy: HumanEval proposes
at most 6 tokens and GSM8K at most 4. A shorter proposal is verified at draft
EOS, the output cap, or the remaining accepted-token budget. Pure modes execute
the same proposal/verify/wait transition in process and do not add WAN delay.

## Common defaults

- seed `3407`
- exactly `1000` target-accepted draft tokens per mode
- at most `128` committed output tokens per sample
- HumanEval indices `50..163`, DeepSeek-Coder 1.3B/6.7B, fixed window `6`
- GSM8K indices `100..1318`, TinyLlama 1.1B/Llama-2 7B, fixed window `4`
- collaborative link: 2.5 MB/s up, 25 MB/s down, 25 ms upload startup
- Pure Edge: both models default to `n_gpu_layers=0`
- Pure Cloud: both models default to `n_gpu_layers=-1`

`TARGET_ACCEPTED_TOKENS` is the preferred environment variable. The legacy
`TARGET_OUTPUT_TOKENS` variable remains an alias.

## Run procedure

Run from `edge/`. Pure Edge requires enough RAM for both models. Before Pure
Cloud, stop the FastAPI target service on a single-GPU host so it does not keep
a second target-model copy resident.

```bash
DATASET=humaneval PHASE=pure_edge bash scripts/eval_four_modes.sh
DATASET=humaneval PHASE=pure_cloud bash scripts/eval_four_modes.sh

DATASET=gsm8k PHASE=pure_edge bash scripts/eval_four_modes.sh
DATASET=gsm8k PHASE=pure_cloud bash scripts/eval_four_modes.sh
```

`PHASE=local_modes` runs both local placements sequentially for one dataset.
Placement can be overridden with
`PURE_{EDGE,CLOUD}_{DRAFT,TARGET}_N_GPU_LAYERS`.

For Serial Edge-Cloud SD and PipeSD, start the cloud service using the target
model for the selected dataset, then run:

```bash
export PIPE_SD_SERVER_URL=http://127.0.0.1:8000
DATASET=humaneval PHASE=collaborative bash scripts/eval_four_modes.sh
DATASET=gsm8k PHASE=collaborative bash scripts/eval_four_modes.sh
```

Restart the service with the matching target when changing datasets. Generate
reports with the exact shared tag:

```bash
DATASET=humaneval PHASE=compare bash scripts/eval_four_modes.sh
DATASET=gsm8k PHASE=compare bash scripts/eval_four_modes.sh
```

Reports are written under
`exp/exp__wjl/<dataset>/comparison/<result-tag>/`.

## Metric boundary

All four TPT/throughput values use target-accepted draft tokens. Target
continuation tokens appear in `actual_output_tokens` but do not consume the
1,000-token budget. Local decode timing starts after both prompt prefills, like
the current collaborative engine. Model load, tokenization, prefill and final
detokenization are excluded from TPT and recorded separately.

Pure Cloud NVML energy covers both local model prefills and speculative decode.
Pure Edge remains unmeasured without RAPL. Collaborative energy covers the
cloud target scope, so these values are not whole-system energy equivalents.

Completion JSONL files remain unscored. Run HumanEval pass@1 and GSM8K answer
accuracy separately before claiming equal output quality.
