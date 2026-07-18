# Four-mode comparison: humaneval

> `—` means not applicable; `missing` means the metric should exist but was not recorded; `--` means Pure Edge energy is intentionally not measured because RAPL access is unavailable.

## Selected artifacts and protocol

| Method | Run ID | Commit | Seed | Norm tokens | Output | TPT token type | Network | Emulator | Up MB/s | Down MB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud  | 774006d514b44aab8944f5a6390c2468 | 1af4ddae5b51 | 3407 | 1000 | 1000 | committed_output_tokens | local | — | — | — |
| Pure Edge  | 5fbb3f8c0f9443ee90a53b2bcbaef0be | 1af4ddae5b51 | 3407 | 1000 | 1000 | committed_output_tokens | local | — | — | — |
| Serial Edge-Cloud SD | e116e8d2b4ce4ae290736eba7b1d93f0 | 1af4ddae5b51 | 3407 | 1000 | 1234 | cloud_accepted_draft_tokens | software | shared-fifo-v1 | 2.500 | 25.000 |
| PipeSD | 43b8378b9dcb48bbbbf866489d5c9dbd | 08f33c53a7bc | 3407 | 1000 | 1200 | cloud_accepted_draft_tokens | software | shared-fifo-v1 | 2.500 | 25.000 |

## Latency and throughput

> Every selected run contains exactly 1,000 benchmark-normalization tokens, so TPT in ms/token is numerically equal to total measured time in seconds: TPT = total_time_seconds × 1000 / 1000. The report retains both columns. Collaborative modes use cloud-accepted draft tokens; pure modes use committed output tokens because they have no NAV acceptance stage.

| Method | TPT ms/benchmark token↓ | benchmark token/s↑ | vs Serial↑ | Total s↓ | P50 ms↓ | P95 ms↓ | P99 ms↓ | TTFT ms↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud  | 4.281 | 233.577 | 147.206 | 4.281 | 3.976 | 4.259 | 4.785 | 36.087 |
| Pure Edge  | 45.934 | 21.770 | 13.720 | 45.934 | 42.444 | 44.530 | 44.988 | 444.092 |
| Serial Edge-Cloud SD | 630.228 | 1.587 | 1.000 | 630.228 | 387.747 | 1272.975 | 2674.895 | 2596.919 |
| PipeSD | 503.759 | 1.985 | 1.251 | 503.759 | 254.618 | 1143.649 | 1929.144 | 2094.993 |

## Energy and speculative-decoding behavior

> Energy is normalized by the same benchmark-token denominator as TPT. Pure Cloud energy covers prompt prefill plus the complete autoregressive decode. Average power uses the recorded energy-window duration when available; legacy artifacts fall back to reported total time.

| Method | Measured energy J/100 benchmark tokens↓ | Avg power W↓ | Energy scope | NAV/100↓ | Draft len | Accept↑ | Rollback↓ | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud  | 176.164 | 411.501 | cloud_gpu_prompt_prefill_plus_autoregressive_decode | — | — | — | — | — |
| Pure Edge  | -- | -- | not_measured_no_rapl_permission | — | — | — | — | — |
| Serial Edge-Cloud SD | 30.292 | 94.680 | cloud_gpu_prompt_prefill_plus_nav_compute | 23.600 | 5.881 | 72.0% | 42.8% | 5.881 |
| PipeSD | 20.026 | 99.136 | cloud_gpu_prompt_prefill_plus_nav_compute | 20.400 | 5.279 | 92.9% | 24.0% | 1.670 |

## Network behavior

| Method | Upload MiB↓ | MiB/100 benchmark tokens↓ | Uploads↓ | Avg upload KiB | Download MiB↓ | Queue s↓ | Service s↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud  | — | — | — | — | — | — | — |
| Pure Edge  | — | — | — | — | — | — | — |
| Serial Edge-Cloud SD | 384.323 | 38.432 | 266 | 1479.501 | 0.135 | 0.000 | 167.853 |
| PipeSD | 401.036 | 40.104 | 823 | 498.980 | 0.177 | 39.841 | 188.789 |

## Runtime termination diagnostics

| Method | Cap hit | EOS |
| --- | --- | --- |
| Pure Cloud  | 100.0% | 0.0% |
| Pure Edge  | 100.0% | 0.0% |
| Serial Edge-Cloud SD | 90.0% | 0.0% |
| PipeSD | 90.0% | 0.0% |

## Comparability warnings

- Pure Cloud (model-only) TPT covers the warm-model local request end to end and excludes model load and client-cloud transfer; its energy covers prompt prefill plus complete decode. Collaborative modes include emulated transport.
- Energy values use different hardware scopes; do not rank them as whole-system energy.
- Energy-per-100 values use different token denominators (accepted draft tokens versus committed output tokens) and are not directly comparable.
