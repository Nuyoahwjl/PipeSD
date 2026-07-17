#!/usr/bin/env python3
"""Compare Pure Cloud, Pure Edge, serial Vanilla SD, and PipeSD results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METHODS = ("pure_cloud", "pure_edge", "vanilla", "pipesd")
DISPLAY_NAMES = {
    "pure_cloud": "Pure Cloud (model-only)",
    "pure_edge": "Pure Edge (local-only)",
    "vanilla": "Serial Edge-Cloud SD",
    "pipesd": "PipeSD",
}


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower = int(math.floor(rank))
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def result_tag(payload: Dict[str, Any]) -> str:
    manifest = payload.get("manifest", {})
    return str(
        manifest.get("result_tag")
        or manifest.get("arguments", {}).get("result_tag")
        or ""
    )


def iter_result_files(inputs: Sequence[Path]) -> Iterable[Path]:
    seen = set()
    for item in inputs:
        candidates = [item] if item.is_file() else item.rglob("*_run=*.json")
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def load_candidates(
    inputs: Sequence[Path], dataset: Optional[str], tag: Optional[str]
) -> Dict[Tuple[str, str], List[Tuple[Path, Dict[str, Any]]]]:
    grouped: Dict[Tuple[str, str], List[Tuple[Path, Dict[str, Any]]]] = defaultdict(list)
    for path in iter_result_files(inputs):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
            continue
        manifest = payload["manifest"]
        method = str(manifest.get("algorithm", ""))
        current_dataset = str(manifest.get("dataset", "")).lower()
        summary = payload.get("summary", {})
        if method not in METHODS or current_dataset not in {"humaneval", "gsm8k"}:
            continue
        if summary.get("evaluation_protocol") != "paper_table1":
            continue
        if dataset and current_dataset != dataset:
            continue
        if tag and result_tag(payload) != tag:
            continue
        grouped[(current_dataset, method)].append((path, payload))
    return grouped


def select_results(
    grouped: Dict[Tuple[str, str], List[Tuple[Path, Dict[str, Any]]]]
) -> Dict[str, Dict[str, Tuple[Path, Dict[str, Any]]]]:
    selected: Dict[str, Dict[str, Tuple[Path, Dict[str, Any]]]] = defaultdict(dict)
    for (dataset, method), candidates in grouped.items():
        selected[dataset][method] = max(
            candidates,
            key=lambda item: (
                float(item[1].get("manifest", {}).get("created_at_unix", 0.0)),
                item[0].stat().st_mtime,
            ),
        )
    return selected


def completion_rows(
    payload: Dict[str, Any], method: str
) -> List[Dict[str, Any]]:
    """Return evaluator-compatible completions without scoring them."""
    manifest = payload.get("manifest", {})
    run_id = manifest.get("run_id")
    rows = []
    for index, sample in enumerate(payload.get("samples", [])):
        rows.append(
            {
                "task_id": sample.get("dataset_task_id", sample.get("task_id")),
                "completion": sample.get("generated_text", ""),
                "method": method,
                "run_id": run_id,
                "sample_index": sample.get("sample_index", index),
            }
        )
    return rows


def write_completion_jsonl(
    output_dir: Path,
    dataset: str,
    method: str,
    payload: Dict[str, Any],
) -> Path:
    """Write one completion JSONL for one selected mode/run."""
    path = output_dir / f"{dataset}_{method}_completions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in completion_rows(payload, method):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def aggregate_network(samples: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    totals = {
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "uplink_transfers": 0,
        "downlink_transfers": 0,
        "network_transfers": 0,
        "network_queue_wait_seconds": 0.0,
        "network_service_seconds": 0.0,
    }
    found = False
    for sample in samples:
        directions = (
            sample.get("environment_measurements", {})
            .get("software_link", {})
            .get("totals", {})
        )
        for direction in ("uplink", "downlink"):
            values = directions.get(direction)
            if not isinstance(values, dict):
                continue
            found = True
            totals[f"{direction}_bytes"] += int(values.get("bytes", 0) or 0)
            transfers = int(values.get("transfers", 0) or 0)
            totals[f"{direction}_transfers"] += transfers
            totals["network_transfers"] += transfers
            totals["network_queue_wait_seconds"] += float(
                values.get("queue_wait_seconds", 0.0) or 0.0
            )
            totals["network_service_seconds"] += float(
                values.get("service_seconds", 0.0) or 0.0
            )
    if not found:
        return {key: None for key in totals}
    return totals


def normalize_result(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    manifest = payload["manifest"]
    summary = payload.get("summary", {})
    samples = payload.get("samples", [])
    tokens = int(summary.get("actual_output_tokens", 0) or 0)
    total_time = float(summary.get("total_time_seconds", 0.0) or 0.0)
    durations = [
        float(value)
        for sample in samples
        for value in sample.get("token_durations", [])
    ]
    ttft = [
        float(sample["time_to_first_token_seconds"])
        for sample in samples
        if sample.get("time_to_first_token_seconds") is not None
    ]
    energy = summary.get("model_energy_joules")
    energy_per_100 = summary.get("model_energy_joules_per_100_tokens")
    if energy is None:
        energy = summary.get("gpu_energy_joules")
    if energy_per_100 is None:
        energy_per_100 = summary.get("gpu_energy_joules_per_100_tokens")
    average_power_watts = (
        float(energy) / total_time
        if energy is not None and total_time > 0.0
        else None
    )
    method = str(manifest["algorithm"])
    network = aggregate_network(samples)
    network_emulation = manifest.get("network_emulation") or {}
    row = {
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        "path": str(path),
        "run_id": manifest.get("run_id"),
        "git_commit": manifest.get("git_commit"),
        "seed": manifest.get("seed"),
        "result_tag": result_tag(payload),
        "network_shaping_mode": manifest.get("network_shaping_mode"),
        "network_emulator_version": network_emulation.get("emulator_version"),
        "uplink_bandwidth_MBps": manifest.get("uplink_bandwidth_MBps"),
        "downlink_bandwidth_MBps": manifest.get("downlink_bandwidth_MBps"),
        "target_tokens": summary.get("target_output_tokens"),
        "actual_tokens": tokens,
        "num_samples": summary.get("num_samples", len(samples)),
        "total_time_seconds": total_time,
        "tpt_ms": summary.get("weighted_tpt_ms"),
        "throughput_tokens_per_second": (
            summary.get("throughput_tokens_per_second")
            or (tokens / total_time if total_time else None)
        ),
        "token_latency_p50_ms": 1000.0 * (
            summary.get("token_latency_p50_seconds")
            if summary.get("token_latency_p50_seconds") is not None
            else percentile(durations, 0.50) or 0.0
        ) if durations or summary.get("token_latency_p50_seconds") is not None else None,
        "token_latency_p95_ms": 1000.0 * (
            summary.get("token_latency_p95_seconds")
            if summary.get("token_latency_p95_seconds") is not None
            else percentile(durations, 0.95) or 0.0
        ) if durations or summary.get("token_latency_p95_seconds") is not None else None,
        "token_latency_p99_ms": 1000.0 * (
            summary.get("token_latency_p99_seconds")
            if summary.get("token_latency_p99_seconds") is not None
            else percentile(durations, 0.99) or 0.0
        ) if durations or summary.get("token_latency_p99_seconds") is not None else None,
        "mean_ttft_ms": 1000.0 * (
            summary.get("mean_ttft_seconds")
            if summary.get("mean_ttft_seconds") is not None
            else statistics.fmean(ttft)
        ) if ttft or summary.get("mean_ttft_seconds") is not None else None,
        "energy_joules": energy,
        "energy_joules_per_100_tokens": energy_per_100,
        "average_power_watts": average_power_watts,
        "power_time_seconds": total_time if average_power_watts is not None else None,
        "power_calculation": (
            "energy_joules / total_time_seconds"
            if average_power_watts is not None
            else None
        ),
        "energy_scope": summary.get("energy_scope", "cloud_gpu" if method != "pure_edge" else "edge_cpu_package"),
        "energy_source": summary.get("energy_source", "cloud_service_nvml" if energy is not None else "unavailable"),
        "verification_frequency": summary.get("verification_frequency"),
        "nav_per_100_tokens": (
            100.0 * summary["verification_frequency"]
            if summary.get("verification_frequency") is not None
            else None
        ),
        "mean_draft_length": summary.get("mean_draft_length"),
        "acceptance_rate": summary.get("acceptance_rate"),
        "rollback_rate": summary.get("rollback_rate"),
        "mean_actual_batch_size": summary.get("mean_actual_batch_size"),
        "cap_hit_rate": summary.get("cap_hit_rate"),
        "eos_rate": (
            float(summary.get("eos_count", 0) or 0) / len(samples) if samples else None
        ),
        **network,
    }
    row["uplink_mib_per_100_tokens"] = (
        row["uplink_bytes"] / 1024 / 1024 * 100.0 / tokens
        if row["uplink_bytes"] is not None and tokens
        else None
    )
    row["average_uplink_transfer_kib"] = (
        row["uplink_bytes"] / 1024.0 / row["uplink_transfers"]
        if row["uplink_bytes"] is not None and row["uplink_transfers"]
        else None
    )
    row["network_queue_seconds_per_100_tokens"] = (
        row["network_queue_wait_seconds"] * 100.0 / tokens
        if row["network_queue_wait_seconds"] is not None and tokens
        else None
    )
    row["network_service_seconds_per_100_tokens"] = (
        row["network_service_seconds"] * 100.0 / tokens
        if row["network_service_seconds"] is not None and tokens
        else None
    )
    return row


def fmt(
    value: Any,
    digits: int = 3,
    percent: bool = False,
    none_label: str = "missing",
) -> str:
    if value is None:
        return none_label
    if isinstance(value, float):
        if percent:
            return f"{100.0 * value:.1f}%"
        return f"{value:.{digits}f}"
    return str(value)


def fmt_for_method(
    row: Dict[str, Any],
    value: Any,
    *,
    applies_to: str = "all",
    digits: int = 3,
    percent: bool = False,
    missing_label: str = "missing",
) -> str:
    collaborative = row["method"] in {"vanilla", "pipesd"}
    if applies_to == "collaborative" and not collaborative:
        return "—"
    if applies_to == "network" and row.get("network_shaping_mode") is None:
        return "—"
    return fmt(value, digits=digits, percent=percent, none_label=missing_label)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(dataset: str, rows: Sequence[Dict[str, Any]], warnings: Sequence[str]) -> str:
    by_method = {row["method"]: row for row in rows}
    vanilla_tpt = by_method.get("vanilla", {}).get("tpt_ms")
    common = []
    for row in rows:
        speedup = vanilla_tpt / row["tpt_ms"] if vanilla_tpt and row.get("tpt_ms") else None
        common.append(
            [
                row["display_name"],
                fmt(row["tpt_ms"]),
                fmt(row["throughput_tokens_per_second"]),
                fmt(speedup),
                fmt(row["total_time_seconds"]),
                fmt(row["token_latency_p50_ms"]),
                fmt(row["token_latency_p95_ms"]),
                fmt(row["token_latency_p99_ms"]),
                fmt_for_method(row, row["mean_ttft_ms"]),
            ]
        )
    efficiency = [
        [
            row["display_name"],
            fmt(
                row["energy_joules_per_100_tokens"],
                none_label="N/A" if row["method"] == "pure_edge" else "missing",
            ),
            fmt(
                row["average_power_watts"],
                none_label="N/A" if row["method"] == "pure_edge" else "missing",
            ),
            row["energy_scope"],
            fmt_for_method(row, row["nav_per_100_tokens"], applies_to="collaborative"),
            fmt_for_method(row, row["mean_draft_length"], applies_to="collaborative"),
            fmt_for_method(row, row["acceptance_rate"], applies_to="collaborative", percent=True),
            fmt_for_method(row, row["rollback_rate"], applies_to="collaborative", percent=True),
            fmt_for_method(row, row["mean_actual_batch_size"], applies_to="collaborative"),
        ]
        for row in rows
    ]
    network_rows = [
        [
            row["display_name"],
            fmt_for_method(
                row,
                row["uplink_bytes"] / 1024 / 1024 if row["uplink_bytes"] is not None else None,
                applies_to="network",
            ),
            fmt_for_method(row, row["uplink_mib_per_100_tokens"], applies_to="network"),
            fmt_for_method(row, row["uplink_transfers"], applies_to="network"),
            fmt_for_method(row, row["average_uplink_transfer_kib"], applies_to="network"),
            fmt_for_method(
                row,
                row["downlink_bytes"] / 1024 / 1024 if row["downlink_bytes"] is not None else None,
                applies_to="network",
            ),
            fmt_for_method(row, row["network_queue_wait_seconds"], applies_to="network"),
            fmt_for_method(row, row["network_service_seconds"], applies_to="network"),
        ]
        for row in rows
    ]
    termination_rows = [
        [
            row["display_name"],
            fmt(row["cap_hit_rate"], percent=True),
            fmt(row["eos_rate"], percent=True),
        ]
        for row in rows
    ]
    provenance = [
        [
            row["display_name"],
            row["run_id"],
            str(row["git_commit"] or "N/A")[:12],
            row["seed"],
            row["actual_tokens"],
            row["network_shaping_mode"] or "local",
            row["network_emulator_version"] or "—",
            fmt_for_method(row, row["uplink_bandwidth_MBps"], applies_to="network"),
            fmt_for_method(row, row["downlink_bandwidth_MBps"], applies_to="network"),
        ]
        for row in rows
    ]
    text = [
        f"# Four-mode comparison: {dataset}",
        "",
        "> `—` means not applicable; `missing` means the metric should exist but was not recorded; `N/A` is retained only for unavailable Pure Edge energy.",
        "",
        "## Selected artifacts and protocol",
        "",
        markdown_table(
            ["Method", "Run ID", "Commit", "Seed", "Tokens", "Network", "Emulator", "Up MB/s", "Down MB/s"],
            provenance,
        ),
        "",
        "## Latency and throughput",
        "",
        (
            "> Every selected run contains exactly 1,000 output tokens, so TPT in "
            "ms/token is numerically equal to total measured time in seconds: "
            "TPT = total_time_seconds × 1000 / 1000. The report retains both "
            "columns and the general token-normalized definition."
            if rows and all(row.get("actual_tokens") == 1000 for row in rows)
            else "> TPT is token-normalized: TPT(ms/token) = total_time_seconds × 1000 / actual_tokens."
        ),
        "",
        markdown_table(
            ["Method", "TPT ms↓", "token/s↑", "vs Serial↑", "Total s↓", "P50 ms↓", "P95 ms↓", "P99 ms↓", "TTFT ms↓"],
            common,
        ),
        "",
        "## Energy and speculative-decoding behavior",
        "",
        "> Average power is derived as measured energy divided by reported total time. The result artifacts do not store a separate NVML sampling-window duration.",
        "",
        markdown_table(
            ["Method", "Measured energy J/100↓", "Avg power W↓", "Energy scope", "NAV/100↓", "Draft len", "Accept↑", "Rollback↓", "Batch size"],
            efficiency,
        ),
        "",
        "## Network behavior",
        "",
        markdown_table(
            ["Method", "Upload MiB↓", "MiB/100 tok↓", "Uploads↓", "Avg upload KiB", "Download MiB↓", "Queue s↓", "Service s↓"],
            network_rows,
        ),
        "",
        "## Runtime termination diagnostics",
        "",
        markdown_table(
            ["Method", "Cap hit", "EOS"],
            termination_rows,
        ),
    ]
    if warnings:
        text.extend(["", "## Comparability warnings", ""] + [f"- {warning}" for warning in warnings])
    return "\n".join(text) + "\n"


def comparability_warnings(rows: Sequence[Dict[str, Any]]) -> List[str]:
    warnings = []
    if len({row.get("actual_tokens") for row in rows}) > 1:
        warnings.append("Actual accepted-token budgets differ across methods.")
    if len({row.get("seed") for row in rows}) > 1:
        warnings.append("Seeds differ across methods.")
    if len({row.get("result_tag") for row in rows}) > 1:
        warnings.append("Result tags differ; confirm that all runs belong to the same scenario.")
    if any(row["method"] == "pure_cloud" for row in rows):
        warnings.append(
            "Pure Cloud (model-only) reports local target-model decode time and excludes client-cloud transfer; collaborative modes include emulated transport."
        )
    collaborative = [row for row in rows if row["method"] in {"vanilla", "pipesd"}]
    if any(row.get("mean_ttft_ms") is None for row in collaborative):
        warnings.append(
            "At least one collaborative artifact predates TTFT instrumentation; rerun Serial SD and PipeSD."
        )
    if any(
        row["method"] != "pure_edge"
        and row.get("energy_joules_per_100_tokens") is None
        for row in rows
    ):
        warnings.append("At least one non-Pure-Edge run is missing its expected energy measurement.")
    network_configs = {
        (
            row.get("network_shaping_mode"),
            row.get("network_emulator_version"),
            row.get("uplink_bandwidth_MBps"),
            row.get("downlink_bandwidth_MBps"),
        )
        for row in collaborative
    }
    if len(network_configs) > 1:
        warnings.append("Vanilla and PipeSD use different network configurations.")
    if any(
        row.get("network_shaping_mode") != "software"
        or row.get("network_emulator_version") != "shared-fifo-v1"
        for row in collaborative
    ):
        warnings.append("At least one collaborative run is not from the current shared-fifo-v1 software emulator.")
    scopes = {row.get("energy_scope") for row in rows if row.get("energy_joules_per_100_tokens") is not None}
    if len(scopes) > 1:
        warnings.append("Energy values use different hardware scopes; do not rank them as whole-system energy.")
    return warnings


def resolve_output_dir(
    dataset: str, result_tag_value: Optional[str], explicit: Optional[Path] = None
) -> Path:
    if explicit is not None:
        return explicit
    return (
        Path("exp/exp__wjl__four__modes")
        / dataset
        / "comparison"
        / (result_tag_value or "latest")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("exp/exp__wjl")],
        help="Result JSON files or directories (default: exp/exp__wjl).",
    )
    parser.add_argument("--dataset", choices=("humaneval", "gsm8k"))
    parser.add_argument("--result-tag", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. By default reports are written beside the "
            "dataset results under exp/exp__wjl__four__modes/<dataset>/comparison."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a partial report instead of failing when a method is absent.",
    )
    args = parser.parse_args()

    grouped = load_candidates(args.inputs, args.dataset, args.result_tag)
    selected = select_results(grouped)
    if not selected:
        raise SystemExit("no matching paper_table1 result bundles found")
    all_reports = {}
    for dataset, methods in sorted(selected.items()):
        output_dir = resolve_output_dir(dataset, args.result_tag, args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        missing = [method for method in METHODS if method not in methods]
        if missing and not args.allow_missing:
            raise SystemExit(
                f"{dataset}: missing {', '.join(missing)}; pass --allow-missing for a partial report"
            )
        rows = []
        for method in METHODS:
            if method not in methods:
                continue
            path, payload = methods[method]
            rows.append(normalize_result(path, payload))
            write_completion_jsonl(output_dir, dataset, method, payload)
        warnings = comparability_warnings(rows)
        report = {"dataset": dataset, "methods": rows, "warnings": warnings}
        all_reports[dataset] = report
        (output_dir / f"four_mode_{dataset}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (output_dir / f"four_mode_{dataset}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        markdown = build_markdown(dataset, rows, warnings)
        (output_dir / f"four_mode_{dataset}.md").write_text(markdown, encoding="utf-8")
        (output_dir / "four_mode_all.json").write_text(
            json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(markdown)


if __name__ == "__main__":
    main()
