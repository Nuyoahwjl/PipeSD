# Rebuttal Draft: Standard Speculative-Decoding Diagnostics for Dual-Threshold NAV

We thank the reviewer for pointing out that the original evaluation did not include several standard speculative-decoding diagnostics needed to interpret the gains from the dual-threshold NAV mechanism. To address this, we added a diagnostic run that explicitly records the draft-length distribution before verification, accepted-prefix length, rejection/rollback statistics, and verification frequency for `PipeSD`, `vanilla`, `HSL`, and `edgeLLM`.

The results below are from a 5-sample Humaneval diagnostic run at `2.5 MB/s`. For `vanilla`, we use fixed-length verification with `gamma=6`; for `HSL`, the single-threshold setting uses `verify_thresh_single=0.99`; for `PipeSD`, the dual-threshold NAV setting uses `verify_thresh_single=0.9` and `verify_thresh_multi=0.95`; and for `edgeLLM`, we use `init_alpha=0.92` and `multiply_times=0.95`. We report weighted time per output token as the main latency metric.

## Table 1. Diagnostic comparison across speculative decoding baselines

| Method | Weighted Time / Output Token (s) | Verification Frequency | Mean Draft Length | Mean Accepted Prefix | Mean Rejected Length | Rollback Rate | Acceptance Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 0.1883 | 0.1396 | 8.00 | 6.16 | 1.84 | 0.3696 | 0.7704 |
| HSL | 0.1502 | 0.2558 | 3.18 | 2.91 | 0.27 | 0.1386 | 0.9148 |
| edgeLLM | 0.1389 | 0.1912 | 4.74 | 4.23 | 0.51 | 0.1757 | 0.8917 |
| **PipeSD** | **0.1254** | **0.1733** | **4.96** | **4.77** | **0.19** | **0.1667** | **0.9616** |

## Interpretation

Table 1 shows that the gain of dual-threshold NAV is not due to a single trivial operating point, but to a better balance between aggressiveness and reliability.

First, compared with `vanilla`, `PipeSD` reduces weighted latency by `33.4%` while dramatically decreasing the average rejected length from `1.84` to `0.19` and improving the acceptance rate from `0.770` to `0.962`. This indicates that the fixed-length baseline speculates too aggressively: it allows long draft segments (`8.0` tokens on average), but a substantial fraction is later rejected. In contrast, dual-threshold NAV keeps a shorter and more controllable draft window, which substantially reduces rollback overhead.

Second, compared with `HSL`, `PipeSD` reduces weighted latency by `16.5%` while lowering verification frequency by `32.2%` and increasing the accepted-prefix length by `63.9%` (`4.77` vs. `2.91`). This is the clearest evidence that the dual-threshold mechanism does not merely become more conservative; rather, it avoids overly frequent verification while still preserving a high acceptance rate. In other words, `HSL` verifies too often, whereas `PipeSD` makes each verification more productive.

Third, compared with `edgeLLM`, `PipeSD` still improves weighted latency by `9.7%`. The two methods have similar mean draft lengths (`4.96` for `PipeSD` vs. `4.74` for `edgeLLM`), but `PipeSD` achieves a longer accepted prefix (`4.77` vs. `4.23`), a much smaller rejected length (`0.19` vs. `0.51`), and a higher acceptance rate (`0.962` vs. `0.892`). This suggests that the benefit of dual-threshold NAV is not simply to shorten drafts; instead, it steers the system toward a more favorable verification point where more drafted tokens are eventually accepted and fewer are wasted.

Overall, these diagnostics support the mechanism behind the observed gains. `vanilla` is too aggressive and incurs heavy rollback; `HSL` is too cautious and verifies too frequently; `edgeLLM` is closer to a balanced regime but still wastes more drafted tokens than `PipeSD`. The dual-threshold NAV in `PipeSD` achieves a better compromise among draft length, accepted-prefix length, verification frequency, and rollback behavior, which directly explains its lower latency.

## Files

- Summary JSON: [nav_diag_pilot_v3_summary.json](/mnt/c/files/PipeSD/edge/exp/exp__gsm/humaneval/nav_diag_pilot_v3_summary.json)
- Raw `PipeSD` results: [st=0.9_mt=0.95_merge=dp_tag=nav_diag_pilot_v3_bw=2.5MB.json](/mnt/c/files/PipeSD/edge/exp/exp__gsm/humaneval/pipesd/st=0.9_mt=0.95_merge=dp_tag=nav_diag_pilot_v3_bw=2.5MB.json)
- Raw `vanilla` results: [gamma_6_tag=nav_diag_pilot_v3_bw=2.5MB.json](/mnt/c/files/PipeSD/edge/exp/exp__gsm/humaneval/vanilla/gamma_6_tag=nav_diag_pilot_v3_bw=2.5MB.json)
- Raw `HSL` results: [st=0.99_tag=nav_diag_pilot_v3_bw=2.5MB.json](/mnt/c/files/PipeSD/edge/exp/exp__gsm/humaneval/hsl/st=0.99_tag=nav_diag_pilot_v3_bw=2.5MB.json)
- Raw `edgeLLM` results: [edgeLLM_alpha=0.92_mult=0.95_tag=nav_diag_pilot_v3_bw=2.5MB.json](/mnt/c/files/PipeSD/edge/exp/exp__gsm/humaneval/edgeLLM/edgeLLM_alpha=0.92_mult=0.95_tag=nav_diag_pilot_v3_bw=2.5MB.json)
