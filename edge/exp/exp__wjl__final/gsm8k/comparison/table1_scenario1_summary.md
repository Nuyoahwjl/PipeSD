# Table 1 Scenario 1 summary: gsm8k

> Lower is better for TPT, latency, energy, NAV, rollback, traffic, and queue time. Energy covers the cloud GPU only.

## Conclusions

- Best TPT: **PipeSD**.
- Lowest recorded GPU energy per 100 tokens: **PipeSD**.
- PipeSD speedup over the best baseline: **1.090x**.

## Performance, latency, and energy

| Method | TPT ms↓ | tok/s↑ | vs Vanilla↑ | GPU J/100↓ | Energy Δ | P50↓ | P95↓ | P99↓ | TTFT↓ | Sample CV↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 700.350 | 1.428 | 1.000 | 6900.050 | 0.0% | 461.095 | 2211.665 | 2286.116 | missing | 12.2% |
| HSL | 667.692 | 1.498 | 1.049 | 6632.790 | -3.9% | 562.516 | 1772.897 | 1954.646 | missing | 26.6% |
| EdgeLLM | 841.294 | 1.189 | 0.832 | 8552.868 | 24.0% | 854.083 | 1821.358 | 1989.035 | missing | 23.3% |
| PipeSD | 612.468 | 1.633 | 1.143 | 6092.901 | -11.7% | 471.827 | 1817.762 | 2155.488 | 1838.885 | 40.0% |

## Speculative-decoding behavior

| Method | Draft | Accept↑ | NAV/100↓ | Accepted/NAV↑ | Rollback↓ | Batch | Reuse | Discard | Discard rate↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 3.813 | 57.1% | 31.500 | 2.178 | 59.0% | 3.813 | 0 | 0 | — |
| HSL | 2.950 | 71.6% | 32.200 | 2.112 | 45.7% | 2.950 | 0 | 0 | — |
| EdgeLLM | 2.002 | 62.9% | 44.300 | 1.260 | 29.6% | 1.539 | 200 | 496 | 71.3% |
| PipeSD | 3.141 | 78.2% | 29.000 | 2.455 | 50.0% | 1.354 | 313 | 385 | 55.2% |

## Network behavior

| Method | Upload MiB↓ | MiB/100↓ | Uploads↓ | Download KiB↓ | Queue s↓ | Service s↓ | Primary req | Proactive req |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 329.923 | 32.992 | 339 | 35.629 | 0.000 | 146.856 | 339 | 0 |
| HSL | 260.984 | 26.098 | 349 | 36.506 | 0.000 | 118.191 | 349 | 0 |
| EdgeLLM | 380.302 | 38.030 | 916 | 114.587 | 68.411 | 182.415 | 332 | 584 |
| PipeSD | 411.073 | 41.107 | 1025 | 116.025 | 66.489 | 198.046 | 519 | 506 |

## Runtime termination

| Method | Cap hit | EOS | Total s | Avg GPU W |
| --- | --- | --- | --- | --- |
| Vanilla | 87.5% | 0 | 700.350 | 98.523 |
| HSL | 77.8% | 1 | 667.692 | 99.339 |
| EdgeLLM | 87.5% | 0 | 841.294 | 101.663 |
| PipeSD | 77.8% | 1 | 612.468 | 99.481 |

## Comparability warnings

- Sample-index sets differ; the comparison is not fully paired.
- At least one artifact was produced from a dirty worktree.
- At least one artifact predates true TTFT instrumentation.
- The cloud target-model hash is missing from at least one artifact.
- At least one method has only one matching run; no cross-run confidence interval is available.
- The selected PipeSD run does not record a BO configuration path.
- Energy covers the cloud GPU only; edge CPU, memory, network devices, and idle system power are excluded.
