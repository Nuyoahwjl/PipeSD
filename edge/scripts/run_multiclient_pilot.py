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
    parser.add_argument("--workload_mode", choices=["distinct", "same", "replicated"], default="distinct")
    parser.add_argument("--base_tag", default="multiclient_pilot")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument(
        "--forward_arg",
        action="append",
        default=[],
        help="Additional argument pair forwarded to app/run_edge.py, e.g. --forward_arg=--bandwidth_MBps --forward_arg=2.5",
    )
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--duration_s", type=float, default=0.0)
    parser.add_argument("--warmup_s", type=float, default=0.0)
    parser.add_argument("--barrier_timeout_s", type=float, default=1800.0)
    parser.add_argument("--barrier_root", default="exp-multi/barriers")
    parser.add_argument("--workload_seed", type=int, default=3407)
    parser.add_argument(
        "--cpu_sets",
        default="4-5;6-7;8-9;10-11;12-13;14-15;16-17;18-19",
        help="semicolon-separated CPU sets, one per edge process",
    )
    return parser.parse_args()


def parse_cpu_set(value: str) -> set:
    cpus = set()
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = (int(item) for item in part.split('-', 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError(f'invalid empty CPU set: {value!r}')
    return cpus


def affinity_preexec(cpus: set):
    def apply_affinity():
        os.sched_setaffinity(0, cpus)
    return apply_affinity


def load_total_samples(dataset: str) -> int:
    data_path = Path("data") / f"{dataset}.jsonl"
    with data_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_entries_for_tag(dataset: str, algorithm: str, result_tag: str) -> List[dict]:
    exp_dir = Path("exp-multi") / "exp__wjl" / dataset / algorithm
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
    barrier_dir = None
    cpu_specs = [item.strip() for item in args.cpu_sets.split(';') if item.strip()]
    cpu_assignments = []
    if args.duration_s > 0:
        if args.num_clients <= 0:
            raise ValueError('--num_clients must be positive')
        if args.workload_mode != 'replicated':
            raise ValueError('duration mode requires --workload_mode replicated')
        if len(cpu_specs) < args.num_clients:
            raise ValueError('--cpu_sets does not provide one set per client')
        if os.name != 'posix' or not hasattr(os, 'sched_setaffinity'):
            raise RuntimeError('CPU affinity requires Linux sched_setaffinity')
        available = set(range(os.cpu_count() or 1))
        for client_idx in range(args.num_clients):
            cpus = parse_cpu_set(cpu_specs[client_idx])
            if not cpus.issubset(available):
                raise ValueError(
                    f'client {client_idx} CPU set {sorted(cpus)} exceeds available CPUs'
                )
            if any(cpus & previous for previous in cpu_assignments):
                raise ValueError('client CPU sets must not overlap')
            cpu_assignments.append(cpus)
        barrier_dir = (
            Path(args.barrier_root)
            / f'{args.base_tag}-{os.getpid()}-{int(time.time())}'
        )
        barrier_dir.mkdir(parents=True, exist_ok=False)
    launch_started_at = time.time()
    for client_idx, indices in enumerate(assignments):
        if not indices:
            continue
        result_tag = build_client_result_tag(args.base_tag, client_idx)
        client_extra_args = list(args.forward_arg)
        if args.duration_s > 0:
            client_extra_args.extend([
                '--client_id', str(client_idx),
                '--run_duration_s', str(args.duration_s),
                '--warmup_duration_s', str(args.warmup_s),
                '--barrier_dir', str(barrier_dir),
                '--barrier_timeout_s', str(args.barrier_timeout_s),
                '--workload_seed', str(args.workload_seed),
                '--software_bandwidth_profile_offset', str(client_idx),
            ])
        command = build_client_command(
            python_bin=args.python_bin,
            dataset=args.dataset,
            algorithm=args.algorithm,
            start_index=min(indices),
            end_index=max(indices),
            task_id_offset=client_idx * 1_000_000,
            result_tag=result_tag,
            extra_args=client_extra_args,
        )
        popen_kwargs = {}
        if args.duration_s > 0:
            popen_kwargs['preexec_fn'] = affinity_preexec(
                cpu_assignments[client_idx]
            )
        processes.append((result_tag, subprocess.Popen(command, **popen_kwargs)))

    measurement_start = None
    measurement_end = None
    if args.duration_s > 0:
        try:
            ready_deadline = time.monotonic() + args.barrier_timeout_s
            while True:
                ready_count = len(list(barrier_dir.glob('ready-*.json')))
                stopped = [
                    (tag, process.returncode)
                    for tag, process in processes
                    if process.poll() is not None
                ]
                if stopped:
                    raise RuntimeError(f'clients stopped before barrier: {stopped}')
                if ready_count == len(processes):
                    break
                if time.monotonic() >= ready_deadline:
                    raise TimeoutError(
                        f'timed out waiting for clients: {ready_count}/{len(processes)} ready'
                    )
                time.sleep(0.1)
            warmup_start = time.time() + 2.0
            measurement_start = warmup_start + args.warmup_s
            measurement_end = measurement_start + args.duration_s
            schedule = {
                'warmup_start_epoch': warmup_start,
                'measurement_start_epoch': measurement_start,
                'measurement_end_epoch': measurement_end,
            }
            (barrier_dir / 'start.json').write_text(
                json.dumps(schedule, ensure_ascii=False, indent=2), encoding='utf-8'
            )
        except Exception:
            for _, process in processes:
                if process.poll() is None:
                    process.terminate()
            for _, process in processes:
                process.wait()
            raise

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
    excluded_entries = 0
    if measurement_start is not None and measurement_end is not None:
        filtered_entries = [
            entry for entry in all_entries
            if entry.get('measurement_phase') == 'measurement'
            and float(entry.get('sample_started_at', 0.0)) >= measurement_start
            and float(entry.get('sample_finished_at', float('inf'))) <= measurement_end
        ]
        excluded_entries = len(all_entries) - len(filtered_entries)
        all_entries = filtered_entries
        makespan = args.duration_s
    else:
        makespan = ended_at - launch_started_at
    metrics = summarize_multiclient_metrics(
        all_entries, makespan=makespan, num_clients=args.num_clients
    )
    metrics.update(
        {
            "dataset": args.dataset,
            "algorithm": args.algorithm,
            "num_clients": args.num_clients,
            "pilot_samples": args.pilot_samples,
            "workload_mode": args.workload_mode,
            "base_tag": args.base_tag,
            "warmup_seconds": args.warmup_s,
            "measurement_duration_seconds": args.duration_s,
            "measurement_window_start": measurement_start,
            "measurement_window_end": measurement_end,
            "excluded_warmup_or_partial_samples": excluded_entries,
            "cpu_sets": cpu_specs[:args.num_clients] if args.duration_s > 0 else [],
            "barrier_dir": str(barrier_dir) if barrier_dir is not None else None,
        }
    )

    summary_path = Path(args.summary_path) if args.summary_path else (
        Path("exp-multi") / "multiclient" / f"{args.base_tag}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
