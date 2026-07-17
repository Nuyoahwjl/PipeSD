# Four-mode comparison: gsm8k

> `—` means not applicable; `missing` means the metric should exist but was not recorded; `N/A` is retained only for unavailable Pure Edge energy.

## Selected artifacts and protocol

| Method | Run ID | Commit | Seed | Tokens | Network | Emulator | Up MB/s | Down MB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud (model-only) | fe1dad7935a142a1b64a389024ca8102 | 3902f28a5ce9 | 3407 | 1000 | local | — | — | — |
| Pure Edge (local-only) | 62626df8321b48b8a6033e18af30e66a | 3902f28a5ce9 | 3407 | 1000 | local | — | — | — |
| Serial Edge-Cloud SD | c256bf81805449e8b1b5027ab7e5c620 | 3902f28a5ce9 | 3407 | 1000 | software | shared-fifo-v1 | 2.500 | 25.000 |
| PipeSD | 8b6c138d0505449294b10307826157fc | 3902f28a5ce9 | 3407 | 1000 | software | shared-fifo-v1 | 2.500 | 25.000 |

## Latency and throughput

> Every selected run contains exactly 1,000 output tokens, so TPT in ms/token is numerically equal to total measured time in seconds: TPT = total_time_seconds × 1000 / 1000. The report retains both columns and the general token-normalized definition.

| Method | TPT ms↓ | token/s↑ | vs Serial↑ | Total s↓ | P50 ms↓ | P95 ms↓ | P99 ms↓ | TTFT ms↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud (model-only) | 3.989 | 250.661 | 175.195 | 3.989 | 3.946 | 4.090 | 4.672 | 6.179 |
| Pure Edge (local-only) | 28.223 | 35.432 | 24.765 | 28.223 | 28.090 | 29.176 | 29.729 | 29.431 |
| Serial Edge-Cloud SD | 698.932 | 1.431 | 1.000 | 698.932 | 461.033 | 2186.003 | 2289.455 | 1731.888 |
| PipeSD | 610.392 | 1.638 | 1.145 | 610.392 | 478.256 | 1796.162 | 2135.024 | 1833.356 |

## Energy and speculative-decoding behavior

> Average power is derived as measured energy divided by reported total time. The result artifacts do not store a separate NVML sampling-window duration.

| Method | Measured energy J/100↓ | Avg power W↓ | Energy scope | NAV/100↓ | Draft len | Accept↑ | Rollback↓ | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud (model-only) | 171.317 | 429.425 | cloud_gpu | — | — | — | — | — |
| Pure Edge (local-only) | N/A | N/A | edge_cpu_package | — | — | — | — | — |
| Serial Edge-Cloud SD | 6846.153 | 97.952 | cloud_gpu | 31.500 | 3.813 | 57.1% | 59.0% | 3.813 |
| PipeSD | 6163.528 | 100.977 | cloud_gpu | 29.000 | 3.141 | 78.2% | 50.0% | 1.354 |

## Network behavior

| Method | Upload MiB↓ | MiB/100 tok↓ | Uploads↓ | Avg upload KiB | Download MiB↓ | Queue s↓ | Service s↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud (model-only) | — | — | — | — | — | — | — |
| Pure Edge (local-only) | — | — | — | — | — | — | — |
| Serial Edge-Cloud SD | 329.923 | 32.992 | 339 | 996.581 | 0.035 | 0.000 | 146.856 |
| PipeSD | 411.073 | 41.107 | 1025 | 410.672 | 0.113 | 66.556 | 198.046 |

## Runtime termination diagnostics

| Method | Cap hit | EOS |
| --- | --- | --- |
| Pure Cloud (model-only) | 60.0% | 40.0% |
| Pure Edge (local-only) | 9.5% | 90.5% |
| Serial Edge-Cloud SD | 87.5% | 0.0% |
| PipeSD | 77.8% | 11.1% |

## Comparability warnings

- Pure Cloud (model-only) reports local target-model decode time and excludes client-cloud transfer; collaborative modes include emulated transport.
