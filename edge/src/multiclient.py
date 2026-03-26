import math
from typing import Dict, List


def partition_sample_indices(
    total_samples: int,
    num_clients: int,
    workload_mode: str,
    pilot_samples: int,
) -> List[List[int]]:
    if total_samples <= 0 or num_clients <= 0 or pilot_samples <= 0:
        return [[] for _ in range(max(0, num_clients))]

    capped = min(total_samples, pilot_samples)
    if workload_mode == "same":
        per_client = max(1, math.ceil(capped / num_clients))
        subset = list(range(min(total_samples, per_client)))
        return [subset.copy() for _ in range(num_clients)]

    if workload_mode != "distinct":
        raise ValueError(f"Unsupported workload_mode: {workload_mode}")

    indices = list(range(capped))
    base = capped // num_clients
    remainder = capped % num_clients
    assignments: List[List[int]] = []
    cursor = 0
    for client_idx in range(num_clients):
        size = base + (1 if client_idx < remainder else 0)
        assignments.append(indices[cursor : cursor + size])
        cursor += size
    return assignments


def summarize_multiclient_metrics(entries: List[Dict[str, float]], makespan: float) -> Dict[str, float]:
    total_output_tokens = sum(int(entry.get("output_length", 0) or 0) for entry in entries)
    num_completed_samples = len(entries)
    total_cloud_energy = float(
        sum(float(entry.get("gpu_power_integral_joules", 0.0) or 0.0) for entry in entries)
    )
    token_throughput = float(total_output_tokens / makespan) if makespan > 0 else 0.0
    sample_throughput = float(num_completed_samples / makespan) if makespan > 0 else 0.0
    energy_per_token = float(total_cloud_energy / total_output_tokens) if total_output_tokens > 0 else 0.0
    energy_per_sample = float(total_cloud_energy / num_completed_samples) if num_completed_samples > 0 else 0.0
    sample_starts = [
        float(entry["sample_started_at"])
        for entry in entries
        if entry.get("sample_started_at") is not None and entry.get("sample_finished_at") is not None
    ]
    sample_finishes = [
        float(entry["sample_finished_at"])
        for entry in entries
        if entry.get("sample_started_at") is not None and entry.get("sample_finished_at") is not None
    ]
    sample_window_makespan = None
    sample_window_token_throughput = None
    sample_window_sample_throughput = None
    if sample_starts and sample_finishes:
        sample_window_makespan = max(0.0, max(sample_finishes) - min(sample_starts))
        if sample_window_makespan > 0:
            sample_window_token_throughput = float(total_output_tokens / sample_window_makespan)
            sample_window_sample_throughput = float(num_completed_samples / sample_window_makespan)
        else:
            sample_window_token_throughput = 0.0
            sample_window_sample_throughput = 0.0
    return {
        "makespan_seconds": float(makespan),
        "total_output_tokens": total_output_tokens,
        "num_completed_samples": num_completed_samples,
        "total_cloud_energy_joules": total_cloud_energy,
        "token_throughput_tps": token_throughput,
        "sample_throughput_sps": sample_throughput,
        "energy_per_token_joules": energy_per_token,
        "energy_per_sample_joules": energy_per_sample,
        "sample_window_makespan_seconds": sample_window_makespan,
        "sample_window_token_throughput_tps": sample_window_token_throughput,
        "sample_window_sample_throughput_sps": sample_window_sample_throughput,
    }


def build_client_result_tag(base_tag: str, client_idx: int) -> str:
    return f"{base_tag}_client{client_idx}"


def build_client_command(
    python_bin: str,
    dataset: str,
    algorithm: str,
    start_index: int,
    end_index: int,
    task_id_offset: int,
    result_tag: str,
    extra_args: List[str],
) -> List[str]:
    command = [
        python_bin,
        "app/run_edge.py",
        "--dataset",
        dataset,
        "--algorithm",
        algorithm,
        "--start_index_of_sample",
        str(start_index),
        "--end_index_of_sample",
        str(end_index),
        "--task_id_offset",
        str(task_id_offset),
        "--result_tag",
        result_tag,
    ]
    command.extend(extra_args)
    return command
