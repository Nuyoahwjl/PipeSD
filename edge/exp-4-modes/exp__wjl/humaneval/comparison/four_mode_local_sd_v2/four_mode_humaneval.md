# Four-mode comparison: humaneval

> `—` means not applicable; `missing` means the metric should exist but was not recorded; `--` means Pure Edge energy is intentionally not measured because RAPL access is unavailable.

## Selected artifacts and protocol

| Method | Run ID | Commit | Seed | Norm tokens | Output | TPT token type | Network | Emulator | Up MB/s | Down MB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud SD | 6e8e9a0a11fa40999bbb00d67d5393a2 | 3fa3f35f8ee1 | 3407 | 1000 | 1212 | target_accepted_draft_tokens | local | — | — | — |
| Pure Edge SD | b51eee23249643a194db18f917f70065 | 3fa3f35f8ee1 | 3407 | 1000 | 1213 | target_accepted_draft_tokens | local | — | — | — |
| Serial Edge-Cloud SD | e116e8d2b4ce4ae290736eba7b1d93f0 | 1af4ddae5b51 | 3407 | 1000 | 1234 | target_accepted_draft_tokens | software | shared-fifo-v1 | 2.500 | 25.000 |
| PipeSD | 43b8378b9dcb48bbbbf866489d5c9dbd | 08f33c53a7bc | 3407 | 1000 | 1200 | target_accepted_draft_tokens | software | shared-fifo-v1 | 2.500 | 25.000 |

## Latency and throughput

> Every selected run contains exactly 1,000 benchmark-normalization tokens, so TPT in ms/token is numerically equal to total measured time in seconds: TPT = total_time_seconds × 1000 / 1000. The report retains both columns. All four modes use target-accepted draft tokens.

| Method | TPT ms/benchmark token↓ | benchmark token/s↑ | vs Serial↑ | Total s↓ | P50 ms↓ | P95 ms↓ | P99 ms↓ | TTFT ms↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud SD | 5.532 | 180.762 | 113.921 | 5.532 | 3.636 | 8.340 | 25.233 | 29.445 |
| Pure Edge SD | 204.839 | 4.882 | 3.077 | 204.839 | 134.661 | 314.394 | 939.528 | 943.478 |
| Serial Edge-Cloud SD | 630.228 | 1.587 | 1.000 | 630.228 | 387.747 | 1272.975 | 2674.895 | 2596.919 |
| PipeSD | 503.759 | 1.985 | 1.251 | 503.759 | 254.618 | 1143.649 | 1929.144 | 2094.993 |

## Energy and speculative-decoding behavior

> Energy is normalized by target-accepted draft tokens. Pure Cloud energy covers co-located draft/target prompt prefill and speculative decode. Average power uses the recorded energy-window duration when available; legacy artifacts fall back to reported total time.

| Method | Measured energy J/100 benchmark tokens↓ | Avg power W↓ | Energy scope | NAV/100↓ | Draft len | Accept↑ | Rollback↓ | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud SD | 222.132 | 376.497 | co_located_cloud_gpu_draft_and_target_prefill_plus_decode | 22.100 | 5.814 | 77.8% | 31.7% | 5.814 |
| Pure Edge SD | -- | -- | not_measured_no_rapl_permission | 22.000 | 5.873 | 77.4% | 34.1% | 5.873 |
| Serial Edge-Cloud SD | 30.292 | 94.680 | cloud_gpu_prompt_prefill_plus_nav_compute | 23.600 | 5.881 | 72.0% | 42.8% | 5.881 |
| PipeSD | 20.026 | 99.136 | cloud_gpu_prompt_prefill_plus_nav_compute | 20.400 | 5.279 | 92.9% | 24.0% | 1.670 |

## Network behavior

| Method | Upload MiB↓ | MiB/100 benchmark tokens↓ | Uploads↓ | Avg upload KiB | Download MiB↓ | Queue s↓ | Service s↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud SD | — | — | — | — | — | — | — |
| Pure Edge SD | — | — | — | — | — | — | — |
| Serial Edge-Cloud SD | 384.323 | 38.432 | 266 | 1479.501 | 0.135 | 0.000 | 167.853 |
| PipeSD | 401.036 | 40.104 | 823 | 498.980 | 0.177 | 39.841 | 188.789 |

## Runtime termination diagnostics

| Method | Cap hit | EOS |
| --- | --- | --- |
| Pure Cloud SD | 90.0% | 0.0% |
| Pure Edge SD | 90.0% | 0.0% |
| Serial Edge-Cloud SD | 90.0% | 0.0% |
| PipeSD | 90.0% | 0.0% |

## Comparability warnings

- Pure Cloud SD co-locates draft and target and excludes client-cloud transfer; Serial Edge-Cloud SD and PipeSD include emulated transport.
- Energy values use different hardware scopes; do not rank them as whole-system energy.
