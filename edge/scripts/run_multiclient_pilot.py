import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

sys.path.append(os.path.join(sys.path[0], "../"))

from src.multiclient import (
    build_client_command,
    build_client_result_tag,
    partition_sample_indices,
    summarize_multiclient_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-client edge pilot against one shared server.")
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument("--algorithm", default="pipesd")
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--pilot_samples", type=int, default=8)
    parser.add_argument("--workload_mode", choices=["distinct", "same"], default="distinct")
    parser.add_argument("--base_tag", default="multiclient_pilot")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument(
        "--forward_arg",
        action="append",
        default=[],
        help="Additional argument pair forwarded to app/run_edge.py, e.g. --forward_arg=--bandwidth_MBps --forward_arg=2.5",
    )
    parser.add_argument("--summary_path", default=None)
    return parser.parse_args()


def load_total_samples(dataset: str) -> int:
    data_path = Path("data") / f"{dataset}.jsonl"
    with data_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_entries_for_tag(dataset: str, algorithm: str, result_tag: str) -> List[dict]:
    exp_dir = Path("exp") / "exp__gsm" / dataset / algorithm
    entries: List[dict] = []
    if not exp_dir.exists():
        return entries
    for path in exp_dir.glob(f"*tag={result_tag}*"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, list):
            entries.extend(payload)
    return entries


def main() -> int:
    args = parse_args()
    total_samples = load_total_samples(args.dataset)
    assignments = partition_sample_indices(
        total_samples=total_samples,
        num_clients=args.num_clients,
        workload_mode=args.workload_mode,
        pilot_samples=args.pilot_samples,
    )
    processes = []
    started_at = time.time()
    for client_idx, indices in enumerate(assignments):
        if not indices:
            continue
        result_tag = build_client_result_tag(args.base_tag, client_idx)
        command = build_client_command(
            python_bin=args.python_bin,
            dataset=args.dataset,
            algorithm=args.algorithm,
            start_index=min(indices),
            end_index=max(indices),
            task_id_offset=client_idx * 1_000_000,
            result_tag=result_tag,
            extra_args=args.forward_arg,
        )
        processes.append((result_tag, subprocess.Popen(command)))

    exit_code = 0
    for result_tag, process in processes:
        code = process.wait()
        if code != 0:
            exit_code = code
            print(f"[multiclient] client tag={result_tag} exited with code {code}", file=sys.stderr)
    ended_at = time.time()

    all_entries: List[dict] = []
    for result_tag, _ in processes:
        all_entries.extend(load_entries_for_tag(args.dataset, args.algorithm, result_tag))
    metrics = summarize_multiclient_metrics(all_entries, makespan=ended_at - started_at)
    metrics.update(
        {
            "dataset": args.dataset,
            "algorithm": args.algorithm,
            "num_clients": args.num_clients,
            "pilot_samples": args.pilot_samples,
            "workload_mode": args.workload_mode,
            "base_tag": args.base_tag,
        }
    )

    summary_path = Path(args.summary_path) if args.summary_path else (
        Path("exp") / "multiclient" / f"{args.base_tag}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
