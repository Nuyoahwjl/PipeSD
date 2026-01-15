"""Analyze verify threshold sweeps in exp_test_thresh."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:  # pragma: no cover - plotting is optional
    plt = None
    np = None


@dataclass
class Result:
    thresh_single: float
    thresh_multi: float
    avg_time_per_token: float
    sample_count: int
    strategy: str
    avg_total_time: Optional[float]
    avg_gpu_energy: Optional[float]
    avg_gpu_power_integral: Optional[float]
    avg_task_energy: Optional[float]
    avg_num_verifications: Optional[float]
    file_name: str


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def fmt_optional(value: Optional[float], fmt: str) -> str:
    return f"{value:{fmt}}" if value is not None else "N/A"


def parse_indices(spec: Optional[str]) -> Optional[Set[int]]:
    """Parse a comma-separated list of indices or ranges like 0-3,5,7."""
    if not spec:
        return None
    indices: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            if end < start:
                start, end = end, start
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices if indices else None


def load_file(path: Path, keep_indices: Optional[Set[int]]) -> Result:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"{path} is empty")
    meta = records[0]
    single = float(meta.get("thresh_single", 0.0))
    multi = float(meta.get("thresh_multi", 0.0))
    strategy = meta.get("strategy", "unknown")
    times = []
    totals = []
    energies = []
    power_integrals = []
    task_energies = []
    num_verifications = []
    for idx, entry in enumerate(records):
        task_id = entry.get("task_id")
        if keep_indices is not None and task_id not in keep_indices:
            continue
        total = entry.get("total_time")
        out_len = entry.get("output_length")
        energy = entry.get("gpu_energy_joules")
        power = entry.get("gpu_power_integral_joules")
        task_energy = entry.get("task_energy_joules")
        verify_stats = entry.get("verify_stats", {})
        num_verify = verify_stats.get("num_verifications")
        if total is not None and out_len not in (None, 0):
            times.append(total / out_len)
        if total is not None:
            totals.append(float(total))
        if energy is not None:
            energies.append(float(energy))
        if power is not None:
            power_integrals.append(float(power))
        if task_energy is not None:
            task_energies.append(float(task_energy))
        if num_verify is not None:
            num_verifications.append(float(num_verify))
    if not times:
        raise ValueError(f"No usable entries in {path}")
    avg_tpt = sum(times) / len(times)
    return Result(
        thresh_single=single,
        thresh_multi=multi,
        avg_time_per_token=avg_tpt,
        sample_count=len(times),
        strategy=strategy,
        avg_total_time=safe_mean(totals),
        avg_gpu_energy=safe_mean(energies),
        avg_gpu_power_integral=safe_mean(power_integrals),
        avg_task_energy=safe_mean(task_energies),
        avg_num_verifications=safe_mean(num_verifications),
        file_name=path.name,
    )


def gather_results(exp_dir: Path, keep_indices: Optional[Set[int]]) -> Dict[str, List[Result]]:
    results: Dict[str, List[Result]] = {}
    for path in exp_dir.glob("*bw=*.json"):
        try:
            result = load_file(path, keep_indices)
        except Exception as exc:  # pragma: no cover - robustness for partial data
            print(f"[warn] skip {path.name}: {exc}")
            continue
        results.setdefault(result.strategy, []).append(result)
    return results


def plot_line(ax, x_vals, y_vals, title, xlabel):
    ax.plot(x_vals, y_vals, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Avg time per token (s)")
    ax.grid(True, linestyle="--", alpha=0.5)


def plot_heatmap(ax, data: List[Result], title: str) -> None:
    if np is None:
        raise ImportError("numpy is required for the hybrid heatmap")
    singles = sorted({r.thresh_single for r in data})
    multis = sorted({r.thresh_multi for r in data})
    grid = np.full((len(multis), len(singles)), np.nan)
    for row in data:
        i = multis.index(row.thresh_multi)
        j = singles.index(row.thresh_single)
        grid[i, j] = row.avg_time_per_token
    im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(singles)), [f"{v:.2f}" for v in singles], rotation=45)
    ax.set_yticks(range(len(multis)), [f"{v:.2f}" for v in multis])
    ax.set_xlabel("thresh_single")
    ax.set_ylabel("thresh_multi")
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Avg time per token (s)")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze threshold sweep results.")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of task indices or ranges (e.g., 0-9,12,15). "
        "If provided, only these task_ids are used to compute averages.",
    )
    args = parser.parse_args(argv)

    exp_dir = Path(__file__).parent
    keep_indices = parse_indices(args.indices)
    lines: List[str] = []

    def log(message: str = "") -> None:
        lines.append(message)
        print(message)

    base_and_subdirs = [exp_dir] + sorted(
        [p for p in exp_dir.iterdir() if p.is_dir()], key=lambda p: p.name
    )
    any_results = False
    base_dir_results: Optional[Dict[str, List[Result]]] = None

    for directory in base_and_subdirs:
        dir_results = gather_results(directory, keep_indices)
        if not dir_results:
            continue
        any_results = True
        if directory == exp_dir:
            base_dir_results = dir_results
            dir_label = "."
        else:
            dir_label = directory.name
        source = f"subset={len(keep_indices)}" if keep_indices is not None else "all"
        log()
        log(f"Directory: {dir_label}")
        log("Averages by strategy:")
        for strategy, rows in dir_results.items():
            rows = sorted(rows, key=lambda r: (r.thresh_single, r.thresh_multi))
            for row in rows:
                log(
                    f"{strategy:<15} st={row.thresh_single:.2f} mt={row.thresh_multi:.2f} "
                    f"avg_tpt={row.avg_time_per_token:.4f}s "
                    f"avg_total={fmt_optional(row.avg_total_time, '.3f')}s "
                    f"avg_verifications={fmt_optional(row.avg_num_verifications, '.1f')} "
                    f"gpu_energy={fmt_optional(row.avg_gpu_energy, '.2f')}J "
                    f"gpu_power_int={fmt_optional(row.avg_gpu_power_integral, '.2f')}J "
                    f"task_energy={fmt_optional(row.avg_task_energy, '.2f')}J "
                    f"(n={row.sample_count}, {source}) file={row.file_name}"
                )

    if not any_results:
        raise SystemExit("No result files found.")

    if plt is None:
        log("matplotlib not installed; skipping plots.")
    elif not base_dir_results:
        log("No top-level results to plot.")
    else:
        # Single-token strategy: plot vs thresh_single.
        if "single-token" in base_dir_results:
            single_rows = sorted(base_dir_results["single-token"], key=lambda r: r.thresh_single)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_line(
                ax,
                [r.thresh_single for r in single_rows],
                [r.avg_time_per_token for r in single_rows],
                "Single-token strategy",
                "thresh_single",
            )
            fig.tight_layout()
            fig.savefig(exp_dir / "single_token_vs_thresh.png", dpi=300)
            log(f"Saved {exp_dir / 'single_token_vs_thresh.png'}")

        # Multiple-tokens strategy: plot vs thresh_multi.
        if "multiple-tokens" in base_dir_results:
            multi_rows = sorted(base_dir_results["multiple-tokens"], key=lambda r: r.thresh_multi)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_line(
                ax,
                [r.thresh_multi for r in multi_rows],
                [r.avg_time_per_token for r in multi_rows],
                "Multiple-tokens strategy",
                "thresh_multi",
            )
            fig.tight_layout()
            fig.savefig(exp_dir / "multiple_tokens_vs_thresh.png", dpi=300)
            log(f"Saved {exp_dir / 'multiple_tokens_vs_thresh.png'}")

        # Hybrid: heatmap over both thresholds.
        if "hybrid" in base_dir_results:
            hybrid_rows = base_dir_results["hybrid"]
            if hybrid_rows:
                fig, ax = plt.subplots(figsize=(7, 5))
                plot_heatmap(ax, hybrid_rows, "Hybrid strategy heatmap")
                fig.tight_layout()
                fig.savefig(exp_dir / "hybrid_heatmap.png", dpi=300)
                log(f"Saved {exp_dir / 'hybrid_heatmap.png'}")

    output_path = exp_dir / "analysis_summary.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary written to {output_path}")


if __name__ == "__main__":
    main()
