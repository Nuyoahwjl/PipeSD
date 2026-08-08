#!/usr/bin/env python3
"""Summarize the two result files produced by the lazy-distribution A/B run."""

import argparse
import glob
import json
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-dir", required=True)
    parser.add_argument("--full-tag", required=True)
    parser.add_argument("--lazy-tag", required=True)
    return parser.parse_args()


def load_result(edge_dir, tag):
    pattern = os.path.join(
        edge_dir,
        "exp",
        "exp__wjl",
        "humaneval",
        "pipesd",
        f"*_tag={tag}_bw=*MB*.json",
    )
    paths = glob.glob(pattern)
    if not paths:
        raise SystemExit(f"result not found: {pattern}")
    path = max(paths, key=os.path.getmtime)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    summary = payload.get("summary", {})
    upload_bytes = 0
    resolve_requests = 0
    for sample in payload.get("samples", []):
        measurements = sample.get("environment_measurements", {})
        for sender_name in ("primary_sender", "proactive_sender"):
            sender = measurements.get(sender_name) or {}
            totals = sender.get("totals", {})
            upload_bytes += int(totals.get("payload_bytes", 0) or 0)
        lazy_metrics = sample.get("lazy_distribution", {})
        resolve_requests += int(lazy_metrics.get("resolve_requests", 0) or 0)

    return {
        "path": path,
        "tpt_ms": summary.get("weighted_tpt_ms"),
        "accepted_tokens": summary.get("actual_accepted_draft_tokens"),
        "total_time_s": summary.get("total_time_seconds"),
        "upload_mib": upload_bytes / (1024 * 1024),
        "resolve_requests": resolve_requests,
    }


def display_number(value):
    return float("nan") if value is None else float(value)


def main():
    args = parse_arguments()
    full = load_result(args.edge_dir, args.full_tag)
    lazy = load_result(args.edge_dir, args.lazy_tag)

    print("\n=== PipeSD HumanEval probability-transport comparison ===")
    print(
        f"{'mode':20} {'tokens':>10} {'TPT(ms)':>12} "
        f"{'time(s)':>12} {'upload(MiB)':>14} {'resolves':>10}"
    )
    for name, row in (("full (original)", full), ("lazy_distribution", lazy)):
        print(
            f"{name:20} {int(row['accepted_tokens'] or 0):10d} "
            f"{display_number(row['tpt_ms']):12.3f} "
            f"{display_number(row['total_time_s']):12.3f} "
            f"{row['upload_mib']:14.3f} {row['resolve_requests']:10d}"
        )

    if full["tpt_ms"] and lazy["tpt_ms"] is not None:
        print(f"TPT change: {(lazy['tpt_ms'] / full['tpt_ms'] - 1) * 100:+.2f}%")
    if full["upload_mib"]:
        print(
            "Upload change: "
            f"{(lazy['upload_mib'] / full['upload_mib'] - 1) * 100:+.2f}%"
        )
    print(f"original result: {full['path']}")
    print(f"lazy result:     {lazy['path']}")


if __name__ == "__main__":
    main()
