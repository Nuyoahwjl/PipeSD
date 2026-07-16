#!/usr/bin/env python3
"""Validate shared-link invariants in PipeSD software-mode result files."""

import argparse
import json
from pathlib import Path


def iter_result_files(inputs):
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            yield from sorted(path.rglob("*.json"))
        elif path.is_file():
            yield path


def validate_file(path, strict_legacy=False):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"cannot read JSON: {exc}"]
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        return None, []
    manifest = payload.get("manifest", {})
    if manifest.get("network_shaping_mode") != "software":
        return None, []
    network = manifest.get("network_emulation", {})
    if network.get("emulator_version") != "shared-fifo-v1":
        if strict_legacy:
            return None, ["software result predates shared-fifo-v1"]
        return {"path": str(path), "legacy_skipped": True}, []

    issues = []
    totals = {
        "samples": 0,
        "requests": 0,
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "uplink_queue_wait_seconds": 0.0,
        "uplink_service_seconds": 0.0,
        "downlink_service_seconds": 0.0,
    }
    for sample_index, sample in enumerate(payload["samples"]):
        environment = sample.get("environment_measurements", {})
        link = environment.get("software_link")
        if not isinstance(link, dict):
            issues.append(f"sample {sample_index}: missing software_link snapshot")
            continue
        link_totals = link.get("totals", {})
        uplink = link_totals.get("uplink", {})
        downlink = link_totals.get("downlink", {})
        sender_requests = 0
        for sender_name in ("primary_sender", "proactive_sender"):
            sender = environment.get(sender_name) or {}
            sender_requests += int((sender.get("totals") or {}).get("requests", 0))
        uplink_transfers = int(uplink.get("transfers", 0))
        downlink_transfers = int(downlink.get("transfers", 0))
        if uplink_transfers != sender_requests:
            issues.append(
                f"sample {sample_index}: {uplink_transfers} uplink transfers != "
                f"{sender_requests} sender requests"
            )
        if downlink_transfers != sender_requests:
            issues.append(
                f"sample {sample_index}: {downlink_transfers} downlink transfers != "
                f"{sender_requests} sender requests"
            )
        totals["samples"] += 1
        totals["requests"] += sender_requests
        totals["uplink_bytes"] += int(uplink.get("bytes", 0))
        totals["downlink_bytes"] += int(downlink.get("bytes", 0))
        totals["uplink_queue_wait_seconds"] += float(uplink.get("queue_wait_seconds", 0.0))
        totals["uplink_service_seconds"] += float(uplink.get("service_seconds", 0.0))
        totals["downlink_service_seconds"] += float(downlink.get("service_seconds", 0.0))

    regression = None
    if payload["samples"]:
        regression = (
            payload["samples"][-1]
            .get("environment_measurements", {})
            .get("estimator", {})
            .get("communication_regression")
        )
    return {
        "path": str(path),
        "dataset": manifest.get("dataset"),
        "algorithm": manifest.get("algorithm"),
        "run_id": manifest.get("run_id"),
        "network": network,
        "totals": totals,
        "communication_regression": regression,
        "valid": not issues,
    }, issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Result JSON files or directories")
    parser.add_argument("--output", help="Optional output JSON path")
    parser.add_argument("--strict-legacy", action="store_true", help="Fail instead of skipping pre-fix software results")
    args = parser.parse_args()

    reports = []
    legacy_skipped = []
    failures = []
    for path in iter_result_files(args.paths):
        report, issues = validate_file(path, strict_legacy=args.strict_legacy)
        if report is not None and report.get("legacy_skipped"):
            legacy_skipped.append(report["path"])
        elif report is not None:
            reports.append(report)
        failures.extend(f"{path}: {issue}" for issue in issues)
    output = {
        "files": reports,
        "legacy_skipped": legacy_skipped,
        "failures": failures,
        "valid": not failures and bool(reports),
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if output["valid"] else 1)


if __name__ == "__main__":
    main()
