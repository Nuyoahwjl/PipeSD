# Four-mode comparison: humaneval

> `—` means not applicable; `missing` means the metric should exist but was not recorded; `N/A` is retained only for unavailable Pure Edge energy.

## Selected artifacts and protocol

| Method | Run ID | Commit | Seed | Tokens | Network | Emulator | Up MB/s | Down MB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud (model-only) | f5b432204bb84f50b1cb08d1a72af255 | 3902f28a5ce9 | 3407 | 1000 | local | — | — | — |
| Pure Edge (local-only) | 1b25638f2daf4a89890b2694b81591be | 3902f28a5ce9 | 3407 | 1000 | local | — | — | — |
| Serial Edge-Cloud SD | 7c8b7314ab774b89afa91c8244c04900 | 3902f28a5ce9 | 3407 | 1000 | software | shared-fifo-v1 | 2.500 | 25.000 |
| PipeSD | 8f7b8beaedd942adaa751fabdd712477 | 3902f28a5ce9 | 3407 | 1000 | software | shared-fifo-v1 | 2.500 | 25.000 |

## Latency and throughput

| Method | TPT ms↓ | token/s↑ | vs Serial↑ | Total s↓ | P50 ms↓ | P95 ms↓ | P99 ms↓ | TTFT ms↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud (model-only) | 4.068 | 245.795 | 127.440 | 4.068 | 3.991 | 4.269 | 4.807 | 6.693 |
| Pure Edge (local-only) | 42.670 | 23.436 | 12.151 | 42.670 | 42.348 | 44.555 | 44.956 | 43.901 |
| Serial Edge-Cloud SD | 518.481 | 1.929 | 1.000 | 518.481 | 389.310 | 1279.295 | 2679.731 | 2586.206 |
| PipeSD | 393.155 | 2.544 | 1.319 | 393.155 | 238.174 | 1090.059 | 1530.660 | 2117.161 |

## Energy and speculative-decoding behavior

| Method | Measured energy J/100↓ | Energy scope | NAV/100↓ | Draft len | Accept↑ | Rollback↓ | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud (model-only) | 168.247 | cloud_gpu | — | — | — | — | — |
| Pure Edge (local-only) | N/A | edge_cpu_package | — | — | — | — | — |
| Serial Edge-Cloud SD | 4930.713 | cloud_gpu | 19.400 | 5.866 | 70.9% | 44.8% | 5.866 |
| PipeSD | 3849.947 | cloud_gpu | 15.600 | 5.808 | 93.7% | 20.5% | 1.742 |

## Network behavior

| Method | Upload MiB↓ | MiB/100 tok↓ | Uploads↓ | Avg upload KiB | Download MiB↓ | Queue s↓ | Service s↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Cloud (model-only) | — | — | — | — | — | — | — |
| Pure Edge (local-only) | — | — | — | — | — | — | — |
| Serial Edge-Cloud SD | 315.101 | 31.510 | 218 | 1480.107 | 0.022 | 0.000 | 137.614 |
| PipeSD | 329.853 | 32.985 | 649 | 520.446 | 0.065 | 35.893 | 154.578 |

## Runtime termination diagnostics

| Method | Cap hit | EOS |
| --- | --- | --- |
| Pure Cloud (model-only) | 100.0% | 0.0% |
| Pure Edge (local-only) | 100.0% | 0.0% |
| Serial Edge-Cloud SD | 87.5% | 0.0% |
| PipeSD | 87.5% | 0.0% |

## Comparability warnings

- Pure Cloud (model-only) reports local target-model decode time and excludes client-cloud transfer; collaborative modes include emulated transport.
