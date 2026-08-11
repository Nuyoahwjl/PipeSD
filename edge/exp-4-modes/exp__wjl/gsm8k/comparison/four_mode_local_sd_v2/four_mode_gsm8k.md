# Four-mode comparison: gsm8k

> `—` means not applicable; `missing` means the metric should exist but was not recorded; `--` means Pure Edge energy is intentionally not measured because RAPL access is unavailable.

## Selected artifacts and protocol

| Method | Run ID | Commit | Seed | Norm tokens | Output | TPT token type | Network | Emulator | Up MB/s | Down MB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud SD | 4b7ebf2eaf864f00b426e6dbcf7ecf74 | 3fa3f35f8ee1 | 3407 | 1000 | 1381 | target_accepted_draft_tokens | local | — | — | — |
| Pure Edge SD | 2abcb73e072344f29e2cf117c956e416 | 3fa3f35f8ee1 | 3407 | 1000 | 1401 | target_accepted_draft_tokens | local | — | — | — |
| Serial Edge-Cloud SD | f7f1d375791249b4b3079c29cf1df429 | 08f33c53a7bc | 3407 | 1000 | 1477 | target_accepted_draft_tokens | software | shared-fifo-v1 | 2.500 | 25.000 |
| PipeSD | a9c3398f3dab4013ae94b3b9f86f09a1 | 08f33c53a7bc | 3407 | 1000 | 1417 | target_accepted_draft_tokens | software | shared-fifo-v1 | 2.500 | 25.000 |

## Latency and throughput

> Every selected run contains exactly 1,000 benchmark-normalization tokens, so TPT in ms/token is numerically equal to total measured time in seconds: TPT = total_time_seconds × 1000 / 1000. The report retains both columns. All four modes use target-accepted draft tokens.

| Method | TPT ms/benchmark token↓ | benchmark token/s↑ | vs Serial↑ | Total s↓ | P50 ms↓ | P95 ms↓ | P99 ms↓ | TTFT ms↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud SD | 7.172 | 139.438 | 148.735 | 7.172 | 3.736 | 13.397 | 19.029 | 15.189 |
| Pure Edge SD | 263.538 | 3.795 | 4.048 | 263.538 | 133.848 | 666.722 | 669.987 | 404.567 |
| Serial Edge-Cloud SD | 1066.676 | 0.937 | 1.000 | 1066.676 | 463.752 | 2233.876 | 2294.833 | 1748.500 |
| PipeSD | 876.745 | 1.141 | 1.217 | 876.745 | 503.547 | 1802.361 | 2122.362 | 1839.752 |

## Energy and speculative-decoding behavior

> Energy is normalized by target-accepted draft tokens. Pure Cloud energy covers co-located draft/target prompt prefill and speculative decode. Average power uses the recorded energy-window duration when available; legacy artifacts fall back to reported total time.

| Method | Measured energy J/100 benchmark tokens↓ | Avg power W↓ | Energy scope | NAV/100↓ | Draft len | Accept↑ | Rollback↓ | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud SD | 298.163 | 395.229 | co_located_cloud_gpu_draft_and_target_prefill_plus_decode | 38.800 | 3.807 | 67.7% | 42.8% | 3.807 |
| Pure Edge SD | -- | -- | not_measured_no_rapl_permission | 40.500 | 3.785 | 65.2% | 45.9% | 3.785 |
| Serial Edge-Cloud SD | 45.592 | 96.557 | cloud_gpu_prompt_prefill_plus_nav_compute | 47.800 | 3.797 | 55.1% | 60.9% | 3.797 |
| PipeSD | 34.930 | 98.043 | cloud_gpu_prompt_prefill_plus_nav_compute | 42.100 | 2.995 | 79.3% | 46.3% | 1.371 |

## Network behavior

| Method | Upload MiB↓ | MiB/100 benchmark tokens↓ | Uploads↓ | Avg upload KiB | Download MiB↓ | Queue s↓ | Service s↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud SD | — | — | — | — | — | — | — |
| Pure Edge SD | — | — | — | — | — | — | — |
| Serial Edge-Cloud SD | 498.593 | 49.859 | 514 | 993.306 | 0.266 | 0.000 | 221.986 |
| PipeSD | 548.189 | 54.819 | 1366 | 410.941 | 0.339 | 89.761 | 264.092 |

## Runtime termination diagnostics

| Method | Cap hit | EOS |
| --- | --- | --- |
| Pure Cloud SD | 61.5% | 30.8% |
| Pure Edge SD | 53.8% | 38.5% |
| Serial Edge-Cloud SD | 91.7% | 0.0% |
| PipeSD | 83.3% | 8.3% |

## Comparability warnings

- Pure Cloud SD co-locates draft and target and excludes client-cloud transfer; Serial Edge-Cloud SD and PipeSD include emulated transport.
- Energy values use different hardware scopes; do not rank them as whole-system energy.
