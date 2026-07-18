# Four-mode comparison: gsm8k

> `—` means not applicable; `missing` means the metric should exist but was not recorded; `--` means Pure Edge energy is intentionally not measured because RAPL access is unavailable.

## Selected artifacts and protocol

| Method | Run ID | Commit | Seed | Norm tokens | Output | TPT token type | Network | Emulator | Up MB/s | Down MB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud  | 5954891cdcaf4d429d571321d302dbbd | 1af4ddae5b51 | 3407 | 1000 | 1000 | committed_output_tokens | local | — | — | — |
| Pure Edge  | 5837e37e568244e6a12376d80c15c3ae | 1af4ddae5b51 | 3407 | 1000 | 1000 | committed_output_tokens | local | — | — | — |
| Serial Edge-Cloud SD | f7f1d375791249b4b3079c29cf1df429 | 08f33c53a7bc | 3407 | 1000 | 1477 | cloud_accepted_draft_tokens | software | shared-fifo-v1 | 2.500 | 25.000 |
| PipeSD | a9c3398f3dab4013ae94b3b9f86f09a1 | 08f33c53a7bc | 3407 | 1000 | 1417 | cloud_accepted_draft_tokens | software | shared-fifo-v1 | 2.500 | 25.000 |

## Latency and throughput

> Every selected run contains exactly 1,000 benchmark-normalization tokens, so TPT in ms/token is numerically equal to total measured time in seconds: TPT = total_time_seconds × 1000 / 1000. The report retains both columns. Collaborative modes use cloud-accepted draft tokens; pure modes use committed output tokens because they have no NAV acceptance stage.

| Method | TPT ms/benchmark token↓ | benchmark token/s↑ | vs Serial↑ | Total s↓ | P50 ms↓ | P95 ms↓ | P99 ms↓ | TTFT ms↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud  | 4.221 | 236.925 | 252.723 | 4.221 | 3.943 | 4.070 | 4.641 | 29.786 |
| Pure Edge  | 33.941 | 29.463 | 31.428 | 33.941 | 28.108 | 29.289 | 29.782 | 119.160 |
| Serial Edge-Cloud SD | 1066.676 | 0.937 | 1.000 | 1066.676 | 463.752 | 2233.876 | 2294.833 | 1748.500 |
| PipeSD | 876.745 | 1.141 | 1.217 | 876.745 | 503.547 | 1802.361 | 2122.362 | 1839.752 |

## Energy and speculative-decoding behavior

> Energy is normalized by the same benchmark-token denominator as TPT. Pure Cloud energy covers prompt prefill plus the complete autoregressive decode. Average power uses the recorded energy-window duration when available; legacy artifacts fall back to reported total time.

| Method | Measured energy J/100 benchmark tokens↓ | Avg power W↓ | Energy scope | NAV/100↓ | Draft len | Accept↑ | Rollback↓ | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud  | 182.116 | 431.526 | cloud_gpu_prompt_prefill_plus_autoregressive_decode | — | — | — | — | — |
| Pure Edge  | -- | -- | not_measured_no_rapl_permission | — | — | — | — | — |
| Serial Edge-Cloud SD | 45.592 | 96.557 | cloud_gpu_prompt_prefill_plus_nav_compute | 47.800 | 3.797 | 55.1% | 60.9% | 3.797 |
| PipeSD | 34.930 | 98.043 | cloud_gpu_prompt_prefill_plus_nav_compute | 42.100 | 2.995 | 79.3% | 46.3% | 1.371 |

## Network behavior

| Method | Upload MiB↓ | MiB/100 benchmark tokens↓ | Uploads↓ | Avg upload KiB | Download MiB↓ | Queue s↓ | Service s↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud  | — | — | — | — | — | — | — |
| Pure Edge  | — | — | — | — | — | — | — |
| Serial Edge-Cloud SD | 498.593 | 49.859 | 514 | 993.306 | 0.266 | 0.000 | 221.986 |
| PipeSD | 548.189 | 54.819 | 1366 | 410.941 | 0.339 | 89.761 | 264.092 |

## Runtime termination diagnostics

| Method | Cap hit | EOS |
| --- | --- | --- |
| Pure Cloud  | 60.0% | 40.0% |
| Pure Edge  | 9.5% | 90.5% |
| Serial Edge-Cloud SD | 91.7% | 0.0% |
| PipeSD | 83.3% | 8.3% |

## Comparability warnings

- Pure Cloud (model-only) TPT covers the warm-model local request end to end and excludes model load and client-cloud transfer; its energy covers prompt prefill plus complete decode. Collaborative modes include emulated transport.
- Energy values use different hardware scopes; do not rank them as whole-system energy.
- Energy-per-100 values use different token denominators (accepted draft tokens versus committed output tokens) and are not directly comparable.
