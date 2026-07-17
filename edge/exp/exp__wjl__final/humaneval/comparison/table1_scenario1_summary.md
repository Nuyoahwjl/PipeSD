# Table 1 Scenario 1 summary: humaneval

> Lower is better for TPT, latency, energy, NAV, rollback, traffic, and queue time. Energy covers the cloud GPU only.

## Conclusions

- Best TPT: **PipeSD**.
- Lowest recorded GPU energy per 100 tokens: **PipeSD**.
- PipeSD speedup over the best baseline: **1.320x**.

## Performance, latency, and energy

| Method | TPT ms↓ | tok/s↑ | vs Vanilla↑ | GPU J/100↓ | Energy Δ | P50↓ | P95↓ | P99↓ | TTFT↓ | Sample CV↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 516.587 | 1.936 | 1.000 | 4921.003 | 0.0% | 386.934 | 1278.710 | 2669.294 | missing | 13.7% |
| HSL | 552.550 | 1.810 | 0.935 | 5417.605 | 10.1% | 451.951 | 957.246 | 1844.093 | missing | 17.1% |
| EdgeLLM | 722.554 | 1.384 | 0.715 | 7174.303 | 45.8% | 800.672 | 1285.318 | 2157.378 | missing | 39.5% |
| PipeSD | 391.419 | 2.555 | 1.320 | 3827.028 | -22.2% | 238.022 | 1064.176 | 1516.848 | missing | 12.1% |

## Speculative-decoding behavior

| Method | Draft | Accept↑ | NAV/100↓ | Accepted/NAV↑ | Rollback↓ | Batch | Reuse | Discard | Discard rate↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 5.866 | 70.9% | 19.400 | 4.160 | 44.8% | 5.866 | 0 | 0 | — |
| HSL | 3.107 | 95.8% | 25.200 | 2.976 | 12.7% | 3.107 | 0 | 0 | — |
| EdgeLLM | 2.614 | 78.6% | 32.900 | 2.055 | 9.4% | 2.206 | 274 | 139 | 33.7% |
| PipeSD | 5.808 | 93.7% | 15.600 | 5.442 | 20.5% | 1.725 | 470 | 132 | 21.9% |

## Network behavior

| Method | Upload MiB↓ | MiB/100↓ | Uploads↓ | Download KiB↓ | Queue s↓ | Service s↓ | Primary req | Proactive req |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 315.101 | 31.510 | 218 | 22.507 | 0.000 | 137.614 | 218 | 0 |
| HSL | 216.824 | 21.682 | 276 | 28.645 | 0.000 | 97.844 | 276 | 0 |
| EdgeLLM | 277.223 | 27.722 | 477 | 53.264 | 54.768 | 128.203 | 208 | 269 |
| PipeSD | 330.131 | 33.013 | 656 | 66.966 | 35.943 | 154.870 | 316 | 340 |

## Runtime termination

| Method | Cap hit | EOS | Total s | Avg GPU W |
| --- | --- | --- | --- | --- |
| Vanilla | 87.5% | 0 | 516.587 | 95.260 |
| HSL | 87.5% | 0 | 552.550 | 98.047 |
| EdgeLLM | 87.5% | 0 | 722.554 | 99.291 |
| PipeSD | 87.5% | 0 | 391.419 | 97.773 |

## Comparability warnings

- At least one artifact was produced from a dirty worktree.
- At least one artifact predates true TTFT instrumentation.
- The cloud target-model hash is missing from at least one artifact.
- At least one method has only one matching run; no cross-run confidence interval is available.
- The selected PipeSD run does not record a BO configuration path.
- Energy covers the cloud GPU only; edge CPU, memory, network devices, and idle system power are excluded.
