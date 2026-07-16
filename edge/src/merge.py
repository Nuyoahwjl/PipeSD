import sys
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


def next_plan_index(current_index: int, plan: List[int]) -> int:
    """Advance a batch plan cyclically across consecutive scheduling windows."""
    if not plan:
        raise ValueError("plan must contain at least one batch")
    return (int(current_index) + 1) % len(plan)


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


class PaperDPScheduler:
    """Online scheduler matching Algorithm 1 and Appendix D.2 of PipeSD."""

    def __init__(self, alpha: float, beta: float, gamma: float, initial_window: int = 20,
                 history_size: int = 100, update_threshold: Optional[float] = None,
                 gamma_update_threshold: float = 0.2,
                 communication_update_threshold: float = 0.2) -> None:
        if initial_window <= 0 or history_size <= 0:
            raise ValueError("window sizes must be positive")
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.initial_window = int(initial_window)
        self.window = self.initial_window
        if update_threshold is not None:
            gamma_update_threshold = update_threshold
            communication_update_threshold = update_threshold
        self.gamma_update_threshold = float(gamma_update_threshold)
        self.communication_update_threshold = float(communication_update_threshold)
        self.draft_lengths: Deque[int] = deque(maxlen=history_size)
        self._plan: Optional[List[int]] = None
        self.update_history: Deque[Dict[str, object]] = deque(maxlen=history_size)

    def reset_workload_history(self) -> None:
        """Reset candidate-specific draft-length state without discarding environment estimates."""
        self.draft_lengths.clear()
        self.window = self.initial_window
        self._plan = None

    @staticmethod
    def _relative_change(old: float, new: float) -> float:
        return (0.0 if new == 0 else float("inf")) if old == 0 else abs(new - old) / abs(old)

    def observe_draft_length(self, length: int) -> bool:
        if length <= 0:
            return False
        self.draft_lengths.append(int(length))
        new_window = max(1, int(round(sum(self.draft_lengths) / len(self.draft_lengths))))
        if new_window == self.window:
            return False
        self.window, self._plan = new_window, None
        return True

    def update_parameters(self, *, alpha: Optional[float] = None,
                          beta: Optional[float] = None,
                          gamma: Optional[float] = None) -> bool:
        proposed = {"alpha": self.alpha if alpha is None else float(alpha),
                    "beta": self.beta if beta is None else float(beta),
                    "gamma": self.gamma if gamma is None else float(gamma)}
        relative_changes = {
            name: self._relative_change(getattr(self, name), value)
            for name, value in proposed.items()
        }
        changed = (
            relative_changes['gamma'] > self.gamma_update_threshold
            or relative_changes['alpha'] > self.communication_update_threshold
            or relative_changes['beta'] > self.communication_update_threshold
        )
        self.update_history.append({
            'old': {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma},
            'new': dict(proposed),
            'relative_change': relative_changes,
            'triggered': changed,
        })
        if changed:
            self.alpha, self.beta, self.gamma = proposed["alpha"], proposed["beta"], proposed["gamma"]
            self._plan = None
        return changed

    def plan(self) -> List[int]:
        if self._plan is None:
            batches, _ = dynamic_token_scheduling_dp([self.gamma] * self.window,
                                                      self.alpha, self.beta)
            self._plan = [len(batch) for batch in batches if batch] or [self.window]
        return list(self._plan)

    def snapshot(self) -> Dict[str, object]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "window": self.window,
            "plan": self.plan(),
            "draft_length_history_size": len(self.draft_lengths),
            "gamma_update_threshold": self.gamma_update_threshold,
            "communication_update_threshold": self.communication_update_threshold,
            "last_parameter_update": self.update_history[-1] if self.update_history else None,
        }


class OnlineEnvironmentEstimator:
    """Estimate PipeSD DP parameters from recent runtime measurements."""

    def __init__(self, history_size: int = 100, min_comm_samples: int = 8) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if min_comm_samples <= 1:
            raise ValueError("min_comm_samples must be greater than 1")
        self.min_comm_samples = int(min_comm_samples)
        self.history_size = int(history_size)
        self.comm_samples: Deque[Tuple[int, float]] = deque(maxlen=history_size)
        self.generation_samples: Deque[Tuple[int, float]] = deque(maxlen=history_size)

    def observe_communication(self, token_count: int, elapsed_seconds: float) -> None:
        if token_count <= 0 or elapsed_seconds <= 0:
            return
        self.comm_samples.append((int(token_count), float(elapsed_seconds)))

    def observe_generation(self, token_count: int, elapsed_seconds: float) -> None:
        if token_count <= 0 or elapsed_seconds <= 0:
            return
        self.generation_samples.append((int(token_count), float(elapsed_seconds)))

    def missing_batch_sizes(self, required_sizes) -> List[int]:
        observed = {count for count, _ in self.comm_samples}
        return [int(size) for size in required_sizes if int(size) not in observed]

    def estimate(self) -> Dict[str, float]:
        estimates: Dict[str, float] = {}
        regression = self.communication_regression()
        if regression is not None:
            estimates["alpha"] = regression["alpha"]
            estimates["beta"] = regression["beta"]

        total_generated_tokens = sum(count for count, _ in self.generation_samples)
        total_generation_time = sum(elapsed for _, elapsed in self.generation_samples)
        if total_generated_tokens > 0 and total_generation_time > 0:
            estimates["gamma"] = total_generation_time / total_generated_tokens

        return estimates

    def communication_regression(self) -> Optional[Dict[str, object]]:
        raw_comm = list(self.comm_samples)
        grouped: Dict[int, List[float]] = {}
        for count, elapsed in raw_comm:
            grouped.setdefault(count, []).append(elapsed)
        comm = [
            (count, sum(elapsed_values) / len(elapsed_values))
            for count, elapsed_values in sorted(grouped.items())
        ]
        required_distinct_sizes = min(self.min_comm_samples, 8)
        is_initial_bootstrap = len(raw_comm) == self.min_comm_samples
        is_full_runtime_window = len(raw_comm) >= self.history_size
        if (is_initial_bootstrap or is_full_runtime_window) and len(comm) >= required_distinct_sizes:
            n = float(len(comm))
            sum_x = sum(float(count) for count, _ in comm)
            sum_y = sum(elapsed for _, elapsed in comm)
            sum_xx = sum(float(count * count) for count, _ in comm)
            sum_xy = sum(float(count) * elapsed for count, elapsed in comm)
            denominator = n * sum_xx - sum_x * sum_x
            if denominator > 0:
                beta = (n * sum_xy - sum_x * sum_y) / denominator
                alpha = (sum_y - beta * sum_x) / n
                fitted = [alpha + beta * count for count, _ in comm]
                residuals = [elapsed - predicted for (_, elapsed), predicted in zip(comm, fitted)]
                mean_y = sum_y / n
                ss_res = sum(value * value for value in residuals)
                ss_tot = sum((elapsed - mean_y) ** 2 for _, elapsed in comm)
                r_squared = 1.0 if ss_tot == 0 and ss_res == 0 else (0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot)
                return {
                    "alpha": max(0.0, alpha),
                    "beta": max(1e-9, beta),
                    "r_squared": float(r_squared),
                    "residuals": residuals,
                    "samples_per_batch_size": {str(count): len(grouped[count]) for count, _ in comm},
                    "mean_seconds_per_batch_size": {str(count): elapsed for count, elapsed in comm},
                    "window_full": is_full_runtime_window,
                }
        return None

    def snapshot(self) -> Dict[str, object]:
        return {
            "comm_samples": len(self.comm_samples),
            "generation_samples": len(self.generation_samples),
            "distinct_comm_batch_sizes": sorted({count for count, _ in self.comm_samples}),
            "min_comm_samples": self.min_comm_samples,
            "estimate": self.estimate(),
            "communication_regression": self.communication_regression(),
        }


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
