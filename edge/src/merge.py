import sys
from typing import List, Tuple


def dynamic_token_scheduling_dp(
    token_compute_times: List[float],
    C: float,
    d: float,
    verbose: bool = False,
) -> Tuple[List[List[int]], float]:
    """
    Solve the dynamic token scheduling problem with dynamic programming.

    Args:
        token_compute_times: Per-token compute times in seconds.
        C: Fixed transmission overhead per batch (seconds).
        d: Per-token transmission time (seconds).
        verbose: Whether to print debug information.

    Returns:
        A tuple (batches, min_completion_time):
            batches: List of batches, each batch is a list of token indices.
            min_completion_time: Minimum completion time achieved by the plan.
    """
    N = len(token_compute_times)
    if N == 0:
        return [], 0.0

    # Compute absolute ready times for each token.
    T_ready = [0.0] * N
    T_ready[0] = token_compute_times[0]
    for i in range(1, N):
        T_ready[i] = T_ready[i - 1] + token_compute_times[i]

    # DP[i] stores the optimal completion time for tokens 0..i.
    DP = [0.0] * N
    # P[i] stores the start index of the last batch that gives DP[i].
    P = [0] * N

    for i in range(N):
        min_total_time = sys.float_info.max
        best_j = 0

        for j in range(i + 1):
            prev_batch_finish_time = DP[j - 1] if j > 0 else 0.0
            data_ready_time = T_ready[i]
            batch_start_time = max(prev_batch_finish_time, data_ready_time)
            batch_size = i - j + 1
            batch_duration = C + batch_size * d
            current_total_time = batch_start_time + batch_duration

            if current_total_time < min_total_time:
                min_total_time = current_total_time
                best_j = j

        DP[i] = min_total_time
        P[i] = best_j

    batches: List[List[int]] = []
    current_idx = N - 1
    while current_idx >= 0:
        batch_start_idx = P[current_idx]
        batch = list(range(batch_start_idx, current_idx + 1))
        batches.append(batch)
        current_idx = batch_start_idx - 1

    batches.reverse()
    min_completion_time = DP[N - 1]

    if verbose:
        print(f"[DP] batches: {batches}, min completion time: {min_completion_time:.6f}")

    return batches, min_completion_time


def baseline_full_merge(token_compute_times: List[float], C: float, d: float) -> float:
    """
    Compute the completion time if all tokens are merged and transmitted together.
    """
    N = len(token_compute_times)
    if N == 0:
        return 0.0

    ready_last = sum(token_compute_times)
    batch_start_time = max(0.0, ready_last)
    batch_duration = C + N * d
    return batch_start_time + batch_duration


if __name__ == "__main__":
    import pandas as pd

    compute_times = [0.037] * 20
    C_value = 0.05
    token_size_MB = 0.29
    bandwidths_MBps = list(range(1, 21))  # 1 to 20 MB/s

    results = []

    print(f"开始实验: N={len(compute_times)}, C={C_value}, t_comp=0.02, token_size={token_size_MB}MB")
    print("对比 DP 算法 vs. '全部合并' 基线")
    print("-" * 70)

    for bw in bandwidths_MBps:
        d_value = token_size_MB / bw
        baseline_time = baseline_full_merge(compute_times, C_value, d_value)
        dp_batches, dp_time = dynamic_token_scheduling_dp(compute_times, C_value, d_value)
        time_saved = baseline_time - dp_time
        improvement_percent = (time_saved / baseline_time) * 100

        results.append({
            "BW (MB/s)": bw,
            "d (sec)": d_value,
            "Baseline Time (s)": baseline_time,
            "DP Time (s)": dp_time,
            "DP Batches": str(dp_batches),
            "Improvement (%)": improvement_percent,
        })

    df_results = pd.DataFrame(results)
    max_row = df_results.loc[df_results["Improvement (%)"].idxmax()]

    print("实验结果:")
    print(df_results.to_string(index=False, float_format="%.4f"))
    print("\n" + "=" * 70)
    print("最大提升")
    print(max_row.to_string(float_format="%.4f"))
    print(
        f"\n结论: 在 {max_row['BW (MB/s)']} MB/s 的带宽下 (d={max_row['d (sec)']:.4f}s)，"
        f"DP 算法的提升最大，达到 {max_row['Improvement (%)']:.2f}%。"
    )
    print(f"此时 DP 算法的决策是: {max_row['DP Batches']}")
