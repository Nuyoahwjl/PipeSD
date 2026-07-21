# Table 1 Scenario 1 summary: gsm8k

> TPT, throughput, GPU J/100, NAV/100, and MiB/100 are normalized by cloud-accepted draft tokens. Token-latency percentiles and TTFT describe committed output tokens. Energy includes cloud prompt prefill and target-model NAV compute only.

## Conclusions

- Best TPT: **PipeSD**.
- Lowest recorded GPU energy per 100 accepted draft tokens: **PipeSD**.
- PipeSD speedup over the best baseline: **1.088x**.

## Performance, latency, and energy

| Method | Accepted | Output | TPT ms/accepted↓ | accepted tok/s↑ | vs Vanilla↑ | GPU J/100 accepted↓ | Energy Δ | Output P50↓ | Output P95↓ | Output P99↓ | Output TTFT↓ | Sample accepted-TPT CV↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 1000 | 1477 | 1062.706 | 0.941 | 1.000 | 45.453 | 0.0% | 461.008 | 2221.609 | 2291.564 | 1735.283 | 18.7% |
| HSL | 1000 | 1452 | 949.016 | 1.054 | 1.120 | 36.254 | -20.2% | 532.261 | 1781.417 | 1940.024 | 1741.783 | 30.4% |
| EdgeLLM | 1000 | 1794 | 1498.874 | 0.667 | 0.709 | 54.021 | 18.9% | 860.097 | 1826.136 | 1975.462 | 1768.365 | 39.4% |
| PipeSD | 1000 | 1417 | 872.472 | 1.146 | 1.218 | 34.743 | -23.6% | 494.411 | 1779.891 | 2121.968 | 1824.054 | 42.0% |

## Speculative-decoding behavior

| Method | Draft | Accept↑ | NAV/100 accepted↓ | Accepted/NAV↑ | Rollback↓ | Batch | Reuse | Discard | Discard rate↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 3.797 | 55.1% | 47.800 | 2.092 | 60.9% | 3.797 | 0 | 0 | — |
| HSL | 2.998 | 73.5% | 45.400 | 2.203 | 46.5% | 2.998 | 0 | 0 | — |
| EdgeLLM | 1.830 | 68.3% | 80.000 | 1.250 | 28.1% | 1.439 | 386 | 694 | 64.3% |
| PipeSD | 2.995 | 79.3% | 42.100 | 2.375 | 46.3% | 1.371 | 443 | 502 | 53.1% |

## Network behavior

| Method | Upload MiB↓ | MiB/100 accepted↓ | Uploads↓ | Download KiB↓ | Queue s↓ | Service s↓ | Primary req | Proactive req |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 498.593 | 49.859 | 514 | 271.875 | 0.000 | 221.986 | 514 | 0 |
| HSL | 373.892 | 37.389 | 490 | 258.843 | 0.000 | 169.082 | 490 | 0 |
| EdgeLLM | 593.003 | 59.300 | 1541 | 548.122 | 90.232 | 287.271 | 618 | 923 |
| PipeSD | 548.189 | 54.819 | 1366 | 347.570 | 87.786 | 264.092 | 697 | 669 |

## Runtime termination

| Method | Cap hit | EOS | Total s | Active GPU s | Prefill J | NAV J | NAV mean J | NAV P95 J | Active GPU W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 91.7% | 0 | 1062.706 | 4.712 | 26.387 | 428.139 | 0.896 | 1.027 | 96.451 |
| HSL | 75.0% | 2 | 949.016 | 3.706 | 17.188 | 345.357 | 0.761 | 1.290 | 97.828 |
| EdgeLLM | 86.7% | 1 | 1498.874 | 5.400 | 21.242 | 518.970 | 0.649 | 0.947 | 100.043 |
| PipeSD | 83.3% | 1 | 872.472 | 3.532 | 16.651 | 330.782 | 0.786 | 1.159 | 98.356 |

## Comparability warnings

- Sample-index sets differ; the comparison is not fully paired.
- At least one artifact was produced from a dirty worktree.
- The cloud target-model hash is missing from at least one artifact.
- At least one method has only one matching run; no cross-run confidence interval is available.
- The selected PipeSD run does not record a BO configuration path.
- Energy follows the original-repository active-compute scope: cloud prompt prefill plus each target-model NAV. GPU idle between NAVs, edge-draft wait, network transfer, proactive wait/transfer, model load, and state restore/save are excluded.
