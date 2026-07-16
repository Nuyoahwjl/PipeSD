#!/usr/bin/env python3
"""Summarize paper_table1 JSON bundles for both datasets and four methods."""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = ("vanilla", "hsl", "edgeLLM", "pipesd")
PAPER_SCENARIO1_TPT_MS = {
    "humaneval": {"vanilla": 194, "hsl": 155, "edgeLLM": 153, "pipesd": 129},
    "gsm8k": {"vanilla": 193, "hsl": 174, "edgeLLM": 169, "pipesd": 145},
}


def extract_number(text):
    if not text:
        return None
    marked = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    candidates = marked or re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not candidates:
        return None
    return candidates[-1].replace(",", "")


def gsm8k_exact_match(samples):
    outcomes = []
    for sample in samples:
        reference = extract_number(sample.get("reference_answer"))
        prediction = extract_number(sample.get("generated_text"))
        if reference is not None:
            outcomes.append(prediction == reference)
    return (sum(outcomes) / len(outcomes)) if outcomes else None


def mean_std(values):
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Result directory, normally edge/exp/exp__wjl")
    parser.add_argument("--output", type=Path, default=Path("table1_summary.json"))
    parser.add_argument("--humaneval-jsonl", type=Path, default=Path("humaneval_completions.jsonl"))
    args = parser.parse_args()

    grouped = defaultdict(list)
    humaneval_rows = []
    for path in args.root.rglob("*_run=*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            # BO trial logs are JSON lists and also contain "_run=" in their
            # filename; they are not formal Table 1 evaluation artifacts.
            continue
        manifest = payload.get("manifest", {})
        summary = payload.get("summary", {})
        if summary.get("evaluation_protocol") != "paper_table1":
            continue
        dataset = str(manifest.get("dataset", "")).lower()
        method = manifest.get("algorithm")
        if dataset not in PAPER_SCENARIO1_TPT_MS or method not in METHODS:
            continue
        samples = payload.get("samples", [])
        grouped[(dataset, method)].append({
            "path": str(path),
            "run_id": manifest.get("run_id"),
            "tpt_ms": summary.get("weighted_tpt_ms"),
            "tokens": summary.get("actual_output_tokens"),
            "verification_frequency": summary.get("verification_frequency"),
            "mean_draft_length": summary.get("mean_draft_length"),
            "acceptance_rate": summary.get("acceptance_rate"),
            "rollback_rate": summary.get("rollback_rate"),
            "gsm8k_exact_match": gsm8k_exact_match(samples) if dataset == "gsm8k" else None,
        })
        if dataset == "humaneval":
            for sample in samples:
                humaneval_rows.append({
                    "task_id": sample.get("dataset_task_id", sample.get("task_id")),
                    "completion": sample.get("generated_text", ""),
                    "method": method,
                    "run_id": manifest.get("run_id"),
                })

    report = {"paper_scenario1_tpt_ms": PAPER_SCENARIO1_TPT_MS, "results": {}}
    for dataset in PAPER_SCENARIO1_TPT_MS:
        report["results"][dataset] = {}
        pipesd_tpts = [
            run["tpt_ms"] for run in grouped[(dataset, "pipesd")]
            if run["tpt_ms"] is not None
        ]
        pipesd_mean = statistics.fmean(pipesd_tpts) if pipesd_tpts else None
        for method in METHODS:
            runs = grouped[(dataset, method)]
            tpts = [run["tpt_ms"] for run in runs if run["tpt_ms"] is not None]
            method_mean = statistics.fmean(tpts) if tpts else None
            report["results"][dataset][method] = {
                "runs": runs,
                "num_runs": len(runs),
                "tpt_ms": mean_std(tpts),
                "speedup_of_pipesd_over_method": (
                    method_mean / pipesd_mean
                    if method_mean is not None and pipesd_mean not in (None, 0)
                    else None
                ),
                "paper_tpt_ms": PAPER_SCENARIO1_TPT_MS[dataset][method],
                "relative_error_vs_paper": (
                    (method_mean - PAPER_SCENARIO1_TPT_MS[dataset][method])
                    / PAPER_SCENARIO1_TPT_MS[dataset][method]
                    if method_mean is not None else None
                ),
            }

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.humaneval_jsonl.open("w", encoding="utf-8") as handle:
        for row in humaneval_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
