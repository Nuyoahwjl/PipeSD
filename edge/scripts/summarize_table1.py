#!/usr/bin/env python3
"""Summarize Table 1 Scenario 1 results for each dataset independently."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = ("vanilla", "hsl", "edgeLLM", "pipesd")
DATASETS = ("humaneval", "gsm8k")


DISPLAY_NAMES = {
    "vanilla": "Vanilla",
    "hsl": "HSL",
    "edgeLLM": "EdgeLLM",
    "pipesd": "PipeSD",
}


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower = int(math.floor(rank))
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def safe_div(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def result_tag(payload):
    manifest = payload.get("manifest", {})
    return str(
        manifest.get("result_tag")
        or manifest.get("arguments", {}).get("result_tag")
        or ""
    )


def network_matches(manifest, implementation):
    mode = manifest.get("network_shaping_mode")
    emulator = (manifest.get("network_emulation") or {}).get("emulator_version")
    if implementation == "current_software":
        return mode == "software" and emulator == "shared-fifo-v1"
    if implementation == "os":
        return mode == "os"
    return True


def load_candidates(root, dataset, tag, bandwidth_mbps, network_implementation):
    grouped = defaultdict(list)
    for path in root.rglob("*_run=*.json"):
        if path.name.startswith("bayes_trials_run="):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        manifest = payload.get("manifest", {})
        summary = payload.get("summary", {})
        current_dataset = str(manifest.get("dataset", "")).lower()
        method = manifest.get("algorithm")
        if current_dataset not in DATASETS or method not in METHODS:
            continue
        if dataset and current_dataset != dataset:
            continue
        if summary.get("evaluation_protocol") != "paper_table1":
            continue
        if tag and result_tag(payload) != tag:
            continue
        bandwidth = manifest.get("uplink_bandwidth_MBps")
        if bandwidth_mbps is not None and (
            bandwidth is None
            or not math.isclose(float(bandwidth), bandwidth_mbps, rel_tol=0, abs_tol=1e-9)
        ):
            continue
        if not network_matches(manifest, network_implementation):
            continue
        grouped[(current_dataset, method)].append((path, payload))
    return grouped


def select_latest(grouped):
    selected = defaultdict(dict)
    for (dataset, method), candidates in grouped.items():
        path, payload = max(
            candidates,
            key=lambda item: (
                float(item[1].get("manifest", {}).get("created_at_unix", 0.0)),
                item[0].stat().st_mtime,
            ),
        )
        selected[dataset][method] = (path, payload, len(candidates))
    return selected


def aggregate_network(samples):
    keys = (
        "uplink_bytes", "downlink_bytes", "uplink_transfers", "downlink_transfers",
        "network_queue_wait_seconds", "network_service_seconds",
        "primary_requests", "proactive_requests",
    )
    totals = {key: 0 for key in keys}
    found = False
    for sample in samples:
        environment = sample.get("environment_measurements") or {}
        link = (environment.get("software_link") or {}).get("totals") or {}
        for direction in ("uplink", "downlink"):
            values = link.get(direction)
            if not isinstance(values, dict):
                continue
            found = True
            totals[f"{direction}_bytes"] += int(values.get("bytes", 0) or 0)
            totals[f"{direction}_transfers"] += int(values.get("transfers", 0) or 0)
            totals["network_queue_wait_seconds"] += float(
                values.get("queue_wait_seconds", 0) or 0
            )
            totals["network_service_seconds"] += float(
                values.get("service_seconds", 0) or 0
            )
        for sender, key in (("primary_sender", "primary_requests"),
                            ("proactive_sender", "proactive_requests")):
            values = (environment.get(sender) or {}).get("totals") or {}
            totals[key] += int(values.get("requests", 0) or 0)
    return totals if found else {key: None for key in keys}


def normalize_result(dataset, method, path, payload, candidate_count):
    manifest = payload.get("manifest", {})
    arguments = manifest.get("arguments", {})
    summary = payload.get("summary", {})
    samples = payload.get("samples", [])
    tokens = int(summary.get("actual_output_tokens", 0) or 0)
    total_time = float(summary.get("total_time_seconds", 0) or 0)
    durations = [
        float(value)
        for sample in samples
        for value in (sample.get("token_durations") or [])
    ]
    ttfts = [
        float(sample["time_to_first_token_seconds"])
        for sample in samples
        if sample.get("time_to_first_token_seconds") is not None
    ]
    sample_tpts = [
        1000 * float(sample["total_time"]) / int(sample["output_length"])
        for sample in samples
        if sample.get("output_length")
    ]
    spec_tokens = sum(
        sum(int(value) for value in (sample.get("verify_spec_lengths") or []))
        for sample in samples
    )
    accepted_tokens = sum(
        sum(int(value) for value in (sample.get("verify_accept_lengths") or []))
        for sample in samples
    )
    nav_count = int(summary.get("num_verifications", 0) or 0)
    reused = sum(int(sample.get("reused_proactive_tokens", 0) or 0) for sample in samples)
    discarded = sum(
        int(sample.get("discarded_proactive_tokens", 0) or 0) for sample in samples
    )
    energy = summary.get("gpu_energy_joules")
    tpt_ms = summary.get("weighted_tpt_ms")
    sample_mean = statistics.fmean(sample_tpts) if sample_tpts else None
    sample_std = (
        statistics.stdev(sample_tpts)
        if len(sample_tpts) > 1
        else (0.0 if sample_tpts else None)
    )
    row = {
        "dataset": dataset,
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        "source_path": str(path.resolve()),
        "run_id": manifest.get("run_id"),
        "git_commit": manifest.get("git_commit"),
        "git_dirty": bool(str(manifest.get("git_status") or "").strip()),
        "seed": manifest.get("seed"),
        "result_tag": result_tag(payload),
        "candidate_runs": candidate_count,
        "num_samples": len(samples),
        "sample_indices": summary.get("sample_indices", []),
        "target_tokens": summary.get("target_output_tokens"),
        "actual_tokens": tokens,
        "total_time_seconds": total_time,
        "tpt_ms": tpt_ms,
        "throughput_tokens_per_second": safe_div(tokens, total_time),
        "token_latency_p50_ms": 1000 * percentile(durations, 0.50) if durations else None,
        "token_latency_p95_ms": 1000 * percentile(durations, 0.95) if durations else None,
        "token_latency_p99_ms": 1000 * percentile(durations, 0.99) if durations else None,
        "mean_ttft_ms": 1000 * statistics.fmean(ttfts) if ttfts else None,
        "sample_tpt_mean_ms": sample_mean,
        "sample_tpt_std_ms": sample_std,
        "sample_tpt_cv": safe_div(sample_std, sample_mean),
        "sample_tpt_min_ms": min(sample_tpts) if sample_tpts else None,
        "sample_tpt_max_ms": max(sample_tpts) if sample_tpts else None,
        "sample_tpt_p95_ms": percentile(sample_tpts, 0.95),
        "gpu_energy_joules": energy,
        "gpu_energy_joules_per_100_tokens": summary.get("gpu_energy_joules_per_100_tokens"),
        "average_gpu_power_watts": safe_div(energy, total_time),
        "energy_scope": "cloud_gpu",
        "num_verifications": nav_count,
        "nav_per_100_tokens": 100 * safe_div(nav_count, tokens) if tokens else None,
        "tokens_per_nav": safe_div(tokens, nav_count),
        "mean_draft_length": summary.get("mean_draft_length"),
        "acceptance_rate": summary.get("acceptance_rate"),
        "spec_tokens": spec_tokens,
        "accepted_spec_tokens": accepted_tokens,
        "accepted_spec_tokens_per_nav": safe_div(accepted_tokens, nav_count),
        "rejected_spec_tokens_per_nav": safe_div(spec_tokens - accepted_tokens, nav_count),
        "rollback_rate": summary.get("rollback_rate"),
        "mean_actual_batch_size": summary.get("mean_actual_batch_size"),
        "proactive_reused_tokens": reused,
        "proactive_discarded_tokens": discarded,
        "proactive_discard_rate": safe_div(discarded, reused + discarded),
        "cap_hit_count": summary.get("cap_hit_count"),
        "cap_hit_rate": summary.get("cap_hit_rate"),
        "eos_count": summary.get("eos_count"),
        "eos_rate": safe_div(summary.get("eos_count"), len(samples)),
        "verify_strategy": arguments.get("verify_strategy"),
        "verify_thresh_single": arguments.get("verify_thresh_single"),
        "verify_thresh_multi": arguments.get("verify_thresh_multi"),
        "merge_policy": arguments.get("merge_policy"),
        "bo_config_path": manifest.get("bo_config_path") or arguments.get("bo_config_path") or "",
        "network_shaping_mode": manifest.get("network_shaping_mode"),
        "network_emulator_version": (manifest.get("network_emulation") or {}).get("emulator_version"),
        "uplink_bandwidth_MBps": manifest.get("uplink_bandwidth_MBps"),
        "downlink_bandwidth_MBps": manifest.get("downlink_bandwidth_MBps"),
        "target_model_sha256": manifest.get("target_model_sha256"),
        **aggregate_network(samples),
    }
    row["uplink_mib"] = (
        row["uplink_bytes"] / 1024 / 1024 if row["uplink_bytes"] is not None else None
    )
    row["downlink_kib"] = (
        row["downlink_bytes"] / 1024 if row["downlink_bytes"] is not None else None
    )
    row["uplink_mib_per_100_tokens"] = (
        row["uplink_mib"] * 100 / tokens if row["uplink_mib"] is not None and tokens else None
    )
    return row


def add_comparisons(rows):
    by_method = {row["method"]: row for row in rows}
    vanilla = by_method.get("vanilla")
    pipesd = by_method.get("pipesd")
    for row in rows:
        row["speedup_vs_vanilla"] = (
            safe_div(vanilla.get("tpt_ms"), row.get("tpt_ms")) if vanilla else None
        )
        tpt_ratio = safe_div(row.get("tpt_ms"), vanilla.get("tpt_ms")) if vanilla else None
        energy_ratio = (
            safe_div(
                row.get("gpu_energy_joules_per_100_tokens"),
                vanilla.get("gpu_energy_joules_per_100_tokens"),
            )
            if vanilla else None
        )
        row["tpt_change_vs_vanilla"] = tpt_ratio - 1 if tpt_ratio is not None else None
        row["energy_change_vs_vanilla"] = (
            energy_ratio - 1 if energy_ratio is not None else None
        )
        row["pipesd_speedup_over_method"] = (
            safe_div(row.get("tpt_ms"), pipesd.get("tpt_ms")) if pipesd else None
        )


def comparability_warnings(rows, missing):
    warnings = []
    if missing:
        warnings.append("Missing methods: " + ", ".join(missing) + ".")
    if len({row.get("actual_tokens") for row in rows}) > 1:
        warnings.append("Actual output-token budgets differ across methods.")
    if len({tuple(row.get("sample_indices") or []) for row in rows}) > 1:
        warnings.append("Sample-index sets differ; the comparison is not fully paired.")
    if len({row.get("seed") for row in rows}) > 1:
        warnings.append("Seeds differ across methods.")
    if any(row.get("git_dirty") for row in rows):
        warnings.append("At least one artifact was produced from a dirty worktree.")
    if any(row.get("mean_ttft_ms") is None for row in rows):
        warnings.append("At least one artifact predates true TTFT instrumentation.")
    if any(not row.get("target_model_sha256") for row in rows):
        warnings.append("The cloud target-model hash is missing from at least one artifact.")
    if any(row.get("candidate_runs", 0) < 2 for row in rows):
        warnings.append("At least one method has only one matching run; no cross-run confidence interval is available.")
    if any(row["method"] == "pipesd" and not row.get("bo_config_path") for row in rows):
        warnings.append("The selected PipeSD run does not record a BO configuration path.")
    warnings.append(
        "Energy covers the cloud GPU only; edge CPU, memory, network devices, and idle system power are excluded."
    )
    return warnings


def build_conclusions(rows):
    tpt_rows = [row for row in rows if row.get("tpt_ms") is not None]
    energy_rows = [
        row for row in rows if row.get("gpu_energy_joules_per_100_tokens") is not None
    ]
    pipesd = next((row for row in rows if row["method"] == "pipesd"), None)
    baselines = [row for row in tpt_rows if row["method"] != "pipesd"]
    best_baseline = min(baselines, key=lambda row: row["tpt_ms"]) if baselines else None
    return {
        "best_tpt_method": min(tpt_rows, key=lambda row: row["tpt_ms"])["method"] if tpt_rows else None,
        "best_energy_method": (
            min(energy_rows, key=lambda row: row["gpu_energy_joules_per_100_tokens"])["method"]
            if energy_rows else None
        ),
        "best_baseline_method": best_baseline["method"] if best_baseline else None,
        "pipesd_speedup_over_best_baseline": (
            safe_div(best_baseline["tpt_ms"], pipesd["tpt_ms"])
            if best_baseline and pipesd else None
        ),
    }


def fmt(value, digits=3, percent=False, missing="missing"):
    if value is None:
        return missing
    if isinstance(value, float):
        return f"{100 * value:.1f}%" if percent else f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(report):
    rows = report["methods"]
    conclusions = report["conclusions"]
    best_tpt = DISPLAY_NAMES.get(conclusions["best_tpt_method"], "missing")
    best_energy = DISPLAY_NAMES.get(conclusions["best_energy_method"], "missing")
    sections = [
        f"# Table 1 Scenario 1 summary: {report['dataset']}",
        "",
        "> Lower is better for TPT, latency, energy, NAV, rollback, traffic, and queue time. Energy covers the cloud GPU only.",
        "",
        "## Conclusions",
        "",
        f"- Best TPT: **{best_tpt}**.",
        f"- Lowest recorded GPU energy per 100 tokens: **{best_energy}**.",
        f"- PipeSD speedup over the best baseline: **{fmt(conclusions['pipesd_speedup_over_best_baseline'])}x**.",
        "",
        "## Performance, latency, and energy",
        "",
        markdown_table(
            ["Method", "TPT ms↓", "tok/s↑", "vs Vanilla↑", "GPU J/100↓", "Energy Δ", "P50↓", "P95↓", "P99↓", "TTFT↓", "Sample CV↓"],
            [[row["display_name"], fmt(row["tpt_ms"]), fmt(row["throughput_tokens_per_second"]),
              fmt(row["speedup_vs_vanilla"]), fmt(row["gpu_energy_joules_per_100_tokens"]),
              fmt(row["energy_change_vs_vanilla"], percent=True), fmt(row["token_latency_p50_ms"]),
              fmt(row["token_latency_p95_ms"]), fmt(row["token_latency_p99_ms"]),
              fmt(row["mean_ttft_ms"]), fmt(row["sample_tpt_cv"], percent=True)]
             for row in rows],
        ),
        "",
        "## Speculative-decoding behavior",
        "",
        markdown_table(
            ["Method", "Draft", "Accept↑", "NAV/100↓", "Accepted/NAV↑", "Rollback↓", "Batch", "Reuse", "Discard", "Discard rate↓"],
            [[row["display_name"], fmt(row["mean_draft_length"]),
              fmt(row["acceptance_rate"], percent=True), fmt(row["nav_per_100_tokens"]),
              fmt(row["accepted_spec_tokens_per_nav"]), fmt(row["rollback_rate"], percent=True),
              fmt(row["mean_actual_batch_size"]), row["proactive_reused_tokens"],
              row["proactive_discarded_tokens"], fmt(row["proactive_discard_rate"], percent=True, missing="—")]
             for row in rows],
        ),
        "",
        "## Network behavior",
        "",
        markdown_table(
            ["Method", "Upload MiB↓", "MiB/100↓", "Uploads↓", "Download KiB↓", "Queue s↓", "Service s↓", "Primary req", "Proactive req"],
            [[row["display_name"], fmt(row["uplink_mib"]), fmt(row["uplink_mib_per_100_tokens"]),
              fmt(row["uplink_transfers"]), fmt(row["downlink_kib"]),
              fmt(row["network_queue_wait_seconds"]), fmt(row["network_service_seconds"]),
              fmt(row["primary_requests"]), fmt(row["proactive_requests"])]
             for row in rows],
        ),
        "",
        "## Runtime termination",
        "",
        markdown_table(
            ["Method", "Cap hit", "EOS", "Total s", "Avg GPU W"],
            [[row["display_name"], fmt(row["cap_hit_rate"], percent=True), row["eos_count"],
              fmt(row["total_time_seconds"]), fmt(row["average_gpu_power_watts"])]
             for row in rows],
        ),
    ]
    if report["warnings"]:
        sections.extend(["", "## Comparability warnings", ""])
        sections.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(sections) + "\n"


def completion_rows(payload, method):
    run_id = payload.get("manifest", {}).get("run_id")
    return [
        {
            "task_id": sample.get("dataset_task_id", sample.get("task_id")),
            "completion": sample.get("generated_text", ""),
            "method": method,
            "run_id": run_id,
            "sample_index": sample.get("sample_index", index),
        }
        for index, sample in enumerate(payload.get("samples", []))
    ]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset_outputs(root, dataset, report, selected):
    output_dir = root / dataset / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "table1_scenario1_summary"
    (output_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = report["methods"]
    if rows:
        with (output_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / f"{stem}.md").write_text(build_markdown(report), encoding="utf-8")
    for method, (_, payload, _) in selected.items():
        write_jsonl(
            output_dir / f"{dataset}_{method}_completions.jsonl",
            completion_rows(payload, method),
        )
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", nargs="?", type=Path, default=Path("exp/exp__wjl"),
        help="Result root (default: exp/exp__wjl).",
    )
    parser.add_argument("--dataset", choices=("humaneval", "gsm8k"))
    parser.add_argument("--result-tag", default="table1_s1_paper")
    parser.add_argument("--bandwidth-mbps", type=float, default=2.5)
    parser.add_argument(
        "--network-implementation", choices=("current_software", "os", "any"),
        default="current_software",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional combined JSON index retained for compatibility.",
    )
    parser.add_argument(
        "--humaneval-jsonl", type=Path, default=None,
        help="Optional combined HumanEval completion JSONL retained for compatibility.",
    )
    args = parser.parse_args()

    grouped = load_candidates(
        args.root, args.dataset, args.result_tag, args.bandwidth_mbps,
        args.network_implementation,
    )
    selected_by_dataset = select_latest(grouped)
    if not selected_by_dataset:
        raise SystemExit("no matching paper_table1 result bundles found")

    combined = {
        "result_tag": args.result_tag,
        "bandwidth_mbps": args.bandwidth_mbps,
        "network_implementation_filter": args.network_implementation,
        "datasets": {},
    }
    combined_humaneval = []
    for dataset, selected in sorted(selected_by_dataset.items()):
        missing = [method for method in METHODS if method not in selected]
        if missing and not args.allow_missing:
            raise SystemExit(
                f"{dataset}: missing {', '.join(missing)}; pass --allow-missing for a partial report"
            )
        rows = [
            normalize_result(dataset, method, *selected[method])
            for method in METHODS if method in selected
        ]
        add_comparisons(rows)
        report = {
            "dataset": dataset,
            "filters": {
                "result_tag": args.result_tag,
                "bandwidth_mbps": args.bandwidth_mbps,
                "network_implementation": args.network_implementation,
            },
            "methods": rows,
            "conclusions": build_conclusions(rows),
            "warnings": comparability_warnings(rows, missing),
        }
        combined["datasets"][dataset] = report
        output_dir = write_dataset_outputs(args.root, dataset, report, selected)
        print(f"[{dataset}] wrote Table 1 summary to {output_dir}")
        print(build_markdown(report))
        if dataset == "humaneval":
            for method in METHODS:
                if method in selected:
                    combined_humaneval.extend(completion_rows(selected[method][1], method))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.humaneval_jsonl is not None:
        args.humaneval_jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.humaneval_jsonl, combined_humaneval)


if __name__ == "__main__":
    main()
