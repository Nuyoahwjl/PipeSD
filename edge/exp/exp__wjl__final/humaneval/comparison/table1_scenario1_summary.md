# Table 1 Scenario 1 summary: humaneval

> TPT, throughput, GPU J/100, NAV/100, and MiB/100 are normalized by cloud-accepted draft tokens. Token-latency percentiles and TTFT describe committed output tokens. Energy includes cloud prompt prefill and target-model NAV compute only.

## Conclusions

- Best TPT: **PipeSD**.
- Lowest recorded GPU energy per 100 accepted draft tokens: **PipeSD**.
- PipeSD speedup over the best baseline: **1.090x**.

## Performance, latency, and energy

| Method | Accepted | Output | TPT ms/accepted↓ | accepted tok/s↑ | vs Vanilla↑ | GPU J/100 accepted↓ | Energy Δ | Output P50↓ | Output P95↓ | Output P99↓ | Output TTFT↓ | Sample accepted-TPT CV↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 1000 | 1234 | 628.705 | 1.591 | 1.000 | 30.060 | 0.0% | 386.652 | 1266.437 | 2665.768 | 2593.122 | 16.5% |
| HSL | 1000 | 1240 | 578.558 | 1.728 | 1.087 | 21.643 | -28.0% | 350.212 | 956.762 | 1851.974 | 1970.022 | 28.0% |
| EdgeLLM | 1000 | 1212 | 548.539 | 1.823 | 1.146 | 20.552 | -31.6% | 268.789 | 1145.125 | 2333.992 | 1978.717 | 38.6% |
| PipeSD | 1000 | 1200 | 503.444 | 1.986 | 1.249 | 19.121 | -36.4% | 255.726 | 1140.075 | 1917.818 | 2101.268 | 32.1% |

## Speculative-decoding behavior

| Method | Draft | Accept↑ | NAV/100 accepted↓ | Accepted/NAV↑ | Rollback↓ | Batch | Reuse | Discard | Discard rate↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 5.881 | 72.0% | 23.600 | 4.237 | 42.8% | 5.881 | 0 | 0 | — |
| HSL | 4.408 | 94.5% | 24.000 | 4.167 | 20.8% | 4.408 | 0 | 0 | — |
| EdgeLLM | 5.841 | 80.0% | 21.400 | 4.673 | 31.3% | 4.547 | 372 | 328 | 46.9% |
| PipeSD | 5.279 | 92.9% | 20.400 | 4.902 | 24.0% | 1.670 | 528 | 185 | 25.9% |

## Network behavior

| Method | Upload MiB↓ | MiB/100 accepted↓ | Uploads↓ | Download KiB↓ | Queue s↓ | Service s↓ | Primary req | Proactive req |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 384.323 | 38.432 | 266 | 138.111 | 0.000 | 167.853 | 266 | 0 |
| HSL | 292.960 | 29.296 | 270 | 140.552 | 0.000 | 129.632 | 270 | 0 |
| EdgeLLM | 436.953 | 43.695 | 374 | 145.500 | 66.121 | 192.627 | 176 | 198 |
| PipeSD | 401.036 | 40.104 | 823 | 181.584 | 39.848 | 188.789 | 409 | 414 |

## Runtime termination

| Method | Cap hit | EOS | Total s | Active GPU s | Prefill J | NAV J | NAV mean J | NAV P95 J | Active GPU W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 90.0% | 0 | 628.705 | 3.198 | 28.107 | 272.488 | 1.155 | 1.305 | 93.983 |
| HSL | 90.0% | 0 | 578.558 | 2.318 | 25.091 | 191.343 | 0.797 | 1.318 | 93.371 |
| EdgeLLM | 90.0% | 0 | 548.539 | 2.139 | 26.486 | 179.030 | 0.837 | 1.397 | 96.087 |
| PipeSD | 90.0% | 0 | 503.444 | 1.978 | 15.521 | 175.687 | 0.861 | 1.359 | 96.674 |

## Comparability warnings

- At least one artifact was produced from a dirty worktree.
- The cloud target-model hash is missing from at least one artifact.
- At least one method has only one matching run; no cross-run confidence interval is available.
- The selected PipeSD run does not record a BO configuration path.
- Energy follows the original-repository active-compute scope: cloud prompt prefill plus each target-model NAV. GPU idle between NAVs, edge-draft wait, network transfer, proactive wait/transfer, model load, and state restore/save are excluded.
