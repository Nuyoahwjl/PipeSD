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
    if workload_mode == "replicated":
        subset = list(range(capped))
        return [subset.copy() for _ in range(num_clients)]

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


def summarize_multiclient_metrics(
    entries: List[Dict[str, float]],
    makespan: float,
    num_clients: int = None,
) -> Dict[str, float]:
    def percentile(values, fraction):
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return None
        position = (len(ordered) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

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
    latencies = [
        float(entry.get("total_time", 0.0) or 0.0) for entry in entries
        if entry.get("total_time") is not None
    ]
    ttft_values = [
        float(entry["time_to_first_token_seconds"]) for entry in entries
        if entry.get("time_to_first_token_seconds") is not None
    ]
    tokens_by_client = {}
    samples_by_client = {}
    for entry in entries:
        client_id = str(entry.get("client_id", 0))
        tokens_by_client[client_id] = tokens_by_client.get(client_id, 0) + int(
            entry.get("output_length", 0) or 0
        )
        samples_by_client[client_id] = samples_by_client.get(client_id, 0) + 1
    if num_clients is not None:
        for client_id in range(num_clients):
            tokens_by_client.setdefault(str(client_id), 0)
            samples_by_client.setdefault(str(client_id), 0)
    client_rates = [value / makespan for value in tokens_by_client.values()] if makespan > 0 else []
    fairness = 0.0
    if client_rates and sum(rate * rate for rate in client_rates) > 0:
        fairness = (sum(client_rates) ** 2) / (
            len(client_rates) * sum(rate * rate for rate in client_rates)
        )
    cloud_batches = [
        batch
        for entry in entries
        for batch in (entry.get("cloud_batch_trace", []) or [])
        if isinstance(batch, dict)
    ]
    verify_batches = [
        batch for batch in cloud_batches
        if batch.get("batch_stage", "verify") == "verify"
    ]
    prefill_batches = [
        batch for batch in cloud_batches if batch.get("batch_stage") == "prefill"
    ]
    actual_batch_sizes = [
        int(batch.get("actual_batch_size", 0) or 0) for batch in verify_batches
    ]
    prefill_batch_sizes = [
        int(batch.get("actual_batch_size", 0) or 0) for batch in prefill_batches
    ]
    batch_queue_seconds = [
        float(batch.get("batch_queue_seconds", 0.0) or 0.0)
        for batch in verify_batches
    ]
    batch_decode_seconds = [
        float(batch.get("batch_decode_seconds", 0.0) or 0.0)
        for batch in verify_batches
    ]
    total_speculative = sum(
        sum(int(value) for value in (entry.get("verify_spec_lengths", []) or []))
        for entry in entries
    )
    total_accepted = sum(
        sum(int(value) for value in (entry.get("verify_accept_lengths", []) or []))
        for entry in entries
    )
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
        "sample_latency_p50_seconds": percentile(latencies, 0.50),
        "sample_latency_p95_seconds": percentile(latencies, 0.95),
        "ttft_p50_seconds": percentile(ttft_values, 0.50),
        "ttft_p95_seconds": percentile(ttft_values, 0.95),
        "tokens_by_client": tokens_by_client,
        "completed_samples_by_client": samples_by_client,
        "jain_token_throughput_fairness": fairness,
        "total_speculative_tokens_verified": total_speculative,
        "total_draft_tokens_accepted": total_accepted,
        "draft_acceptance_rate": (
            total_accepted / total_speculative if total_speculative else 0.0
        ),
        "cloud_verify_requests": len(verify_batches),
        "cloud_actual_batch_size_mean": (
            sum(actual_batch_sizes) / len(actual_batch_sizes)
            if actual_batch_sizes else 0.0
        ),
        "cloud_actual_batch_size_p95": percentile(actual_batch_sizes, 0.95),
        "cloud_batched_request_fraction": (
            sum(size > 1 for size in actual_batch_sizes) / len(actual_batch_sizes)
            if actual_batch_sizes else 0.0
        ),
        "cloud_batch_queue_p50_seconds": percentile(batch_queue_seconds, 0.50),
        "cloud_batch_queue_p95_seconds": percentile(batch_queue_seconds, 0.95),
        "cloud_batch_decode_p50_seconds": percentile(batch_decode_seconds, 0.50),
        "cloud_batch_decode_p95_seconds": percentile(batch_decode_seconds, 0.95),
        "cloud_prefill_requests": len(prefill_batches),
        "cloud_actual_prefill_batch_size_mean": (
            sum(prefill_batch_sizes) / len(prefill_batch_sizes)
            if prefill_batch_sizes else 0.0
        ),
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
