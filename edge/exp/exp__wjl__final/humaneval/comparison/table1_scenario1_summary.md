# Table 1 Scenario 1 summary: humaneval

> TPT, throughput, GPU J/100, NAV/100, and MiB/100 are normalized by cloud-accepted draft tokens. Token-latency percentiles and TTFT describe committed output tokens. Energy includes cloud prompt prefill and target-model NAV compute only.

## Conclusions

- Best TPT: **PipeSD**.
- Lowest recorded GPU energy per 100 accepted draft tokens: **PipeSD**.
- PipeSD speedup over the best baseline: **1.249x**.

## Performance, latency, and energy

| Method | Accepted | Output | TPT ms/accepted↓ | accepted tok/s↑ | vs Vanilla↑ | GPU J/100 accepted↓ | Energy Δ | Output P50↓ | Output P95↓ | Output P99↓ | Output TTFT↓ | Sample accepted-TPT CV↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 1000 | 1234 | 628.705 | 1.591 | 1.000 | 30.060 | 0.0% | 386.652 | 1266.437 | 2665.768 | 2593.122 | 16.5% |
| HSL | 1000 | 1354 | 773.934 | 1.292 | 0.812 | 29.048 | -3.4% | 460.315 | 959.618 | 1867.952 | 1915.668 | 24.4% |
| EdgeLLM | 1000 | 1567 | 1258.404 | 0.795 | 0.500 | 39.091 | 30.0% | 827.053 | 1290.572 | 2426.259 | 1989.173 | 43.1% |
| PipeSD | 1000 | 1200 | 503.444 | 1.986 | 1.249 | 19.121 | -36.4% | 255.726 | 1140.075 | 1917.818 | 2101.268 | 32.1% |

## Speculative-decoding behavior

| Method | Draft | Accept↑ | NAV/100 accepted↓ | Accepted/NAV↑ | Rollback↓ | Batch | Reuse | Discard | Discard rate↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 5.881 | 72.0% | 23.600 | 4.237 | 42.8% | 5.881 | 0 | 0 | — |
| HSL | 2.955 | 95.1% | 35.600 | 2.809 | 14.3% | 2.955 | 0 | 0 | — |
| EdgeLLM | 2.597 | 66.8% | 57.600 | 1.736 | 11.5% | 2.245 | 401 | 354 | 46.9% |
| PipeSD | 5.279 | 92.9% | 20.400 | 4.902 | 24.0% | 1.670 | 528 | 185 | 25.9% |

## Network behavior

| Method | Upload MiB↓ | MiB/100 accepted↓ | Uploads↓ | Download KiB↓ | Queue s↓ | Service s↓ | Primary req | Proactive req |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 384.323 | 38.432 | 266 | 138.111 | 0.000 | 167.853 | 266 | 0 |
| HSL | 291.317 | 29.132 | 389 | 204.548 | 0.000 | 131.921 | 389 | 0 |
| EdgeLLM | 512.346 | 51.235 | 861 | 358.454 | 85.210 | 236.433 | 400 | 461 |
| PipeSD | 401.036 | 40.104 | 823 | 181.584 | 39.848 | 188.789 | 409 | 414 |

## Runtime termination

| Method | Cap hit | EOS | Total s | Active GPU s | Prefill J | NAV J | NAV mean J | NAV P95 J | Active GPU W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 90.0% | 0 | 628.705 | 3.198 | 28.107 | 272.488 | 1.155 | 1.305 | 93.983 |
| HSL | 90.9% | 0 | 773.934 | 2.951 | 17.678 | 272.801 | 0.766 | 1.254 | 98.436 |
| EdgeLLM | 92.3% | 0 | 1258.404 | 3.969 | 21.241 | 369.673 | 0.642 | 1.038 | 98.491 |
| PipeSD | 90.0% | 0 | 503.444 | 1.978 | 15.521 | 175.687 | 0.861 | 1.359 | 96.674 |

## Comparability warnings

- Sample-index sets differ; the comparison is not fully paired.
- At least one artifact was produced from a dirty worktree.
- The cloud target-model hash is missing from at least one artifact.
- At least one method has only one matching run; no cross-run confidence interval is available.
- The selected PipeSD run does not record a BO configuration path.
- Energy follows the original-repository active-compute scope: cloud prompt prefill plus each target-model NAV. GPU idle between NAVs, edge-draft wait, network transfer, proactive wait/transfer, model load, and state restore/save are excluded.
