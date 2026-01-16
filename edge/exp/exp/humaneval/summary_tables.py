"""
Generate tables from exp/humaneval/analysis_summary.txt.

The script groups results by bandwidth and prints avg_tpt, gpu_power_int,
and gpu_energy for each method. HSL entries can be filtered by st values
so different thresholds can be compared as separate methods.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SUMMARY_PATH = Path(__file__).with_name("analysis_summary.txt")
DEFAULT_TABLE_PATH = Path(__file__).with_name("summary_tables.txt")
DEFAULT_RELATIVE_PATH = Path(__file__).with_name("summary_tables_relative.txt")
DEFAULT_PIPESD_RELATIVE_PATH = Path(__file__).with_name("summary_tables_pipesd_relative.txt")


@dataclass
class Record:
    directory: str
    strategy: str
    st: Optional[str]
    mt: Optional[str]
    avg_tpt: float
    gpu_power_int: float
    gpu_energy: float
    bandwidth: str
    file: str

    @property
    def method_label(self) -> str:
        if self.directory == "hsl":
            return f"{self.directory} st={self.st} mt={self.mt}"
        if self.strategy:
            return f"{self.directory} ({self.strategy})"
        return self.directory


LINE_RE = re.compile(
    r"^(?P<strategy>\S+)\s+"
    r"st=(?P<st>[0-9.]+)\s+mt=(?P<mt>[0-9.]+)\s+"
    r"avg_tpt=(?P<avg_tpt>[0-9.]+)s.*?"
    r"gpu_energy=(?P<gpu_energy>[0-9.]+)J\s+gpu_power_int=(?P<gpu_power_int>[0-9.]+)J"
    r".*?file=(?P<file>[^\s]+)"
)

BANDWIDTH_RE = re.compile(r"bw=(\d+MB)")


def parse_summary(path: Path) -> List[Record]:
    records: List[Record] = []
    current_dir: Optional[str] = None

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Directory:"):
                current_dir = line.split(":", 1)[1].strip()
                continue
            if current_dir is None:
                continue

            match = LINE_RE.match(line)
            if not match:
                continue

            file_name = match.group("file")
            bw_match = BANDWIDTH_RE.search(file_name)
            bandwidth = bw_match.group(1) if bw_match else "unknown"

            records.append(
                Record(
                    directory=current_dir,
                    strategy=match.group("strategy"),
                    st=match.group("st"),
                    mt=match.group("mt"),
                    avg_tpt=float(match.group("avg_tpt")),
                    gpu_power_int=float(match.group("gpu_power_int")),
                    gpu_energy=float(match.group("gpu_energy")),
                    bandwidth=bandwidth,
                    file=file_name,
                )
            )

    return records


def choose_hsl_thresholds(records: Sequence[Record], user_values: Optional[Sequence[str]]) -> List[str]:
    available = sorted({rec.st for rec in records if rec.directory == "hsl" and rec.st is not None})
    if not available:
        return []

    if user_values is not None:
        return [val for val in user_values if val in available]

    print("HSL st values found:", ", ".join(available))
    raw = input("Enter comma-separated st values to include (blank for all): ").strip()
    if not raw:
        return available
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    return [val for val in requested if val in available]


def filter_records(records: Iterable[Record], hsl_st_values: List[str]) -> List[Record]:
    filtered: List[Record] = []
    for rec in records:
        if rec.directory == "hsl" and rec.st not in hsl_st_values:
            continue
        filtered.append(rec)
    return filtered


def group_by_bandwidth(records: Iterable[Record]) -> Dict[str, List[Record]]:
    grouped: Dict[str, List[Record]] = {}
    for rec in records:
        grouped.setdefault(rec.bandwidth, []).append(rec)
    for bw in grouped:
        grouped[bw].sort(key=lambda r: r.method_label)
    return grouped


def render_table(bandwidth: str, records: Sequence[Record]) -> str:
    header = ["method", "avg_tpt (s)", "gpu_power_int (J)", "gpu_energy (J)"]
    rows = [
        [
            rec.method_label,
            f"{rec.avg_tpt:.4f}",
            f"{rec.gpu_power_int:.2f}",
            f"{rec.gpu_energy:.2f}",
        ]
        for rec in records
    ]

    col_widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header)]
    lines = [f"Bandwidth {bandwidth}"]
    lines.append(" | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)))
    lines.append("-+-".join("-" * col_widths[i] for i in range(len(header))))
    for row in rows:
        lines.append(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(header))))
    return "\n".join(lines)


def render_relative_table(bandwidth: str, records: Sequence[Record]) -> str:
    header = [
        "method",
        "avg_tpt vs vanilla",
        "gpu_power_int vs vanilla",
        "gpu_energy vs vanilla",
    ]

    vanilla = next((rec for rec in records if rec.directory == "vanilla"), None)
    if vanilla is None:
        return f"Bandwidth {bandwidth}\n(no vanilla baseline found)"

    def format_delta(base: float, value: float) -> str:
        if base == 0:
            return "n/a"
        delta = (base - value) / base * 100
        return f"{delta:+.2f}%"

    rows = []
    for rec in records:
        if rec is vanilla:
            rows.append([rec.method_label, "baseline", "baseline", "baseline"])
            continue
        rows.append(
            [
                rec.method_label,
                format_delta(vanilla.avg_tpt, rec.avg_tpt),
                format_delta(vanilla.gpu_power_int, rec.gpu_power_int),
                format_delta(vanilla.gpu_energy, rec.gpu_energy),
            ]
        )

    col_widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header)]
    lines = [f"Bandwidth {bandwidth} (relative to vanilla)"]
    lines.append(" | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)))
    lines.append("-+-".join("-" * col_widths[i] for i in range(len(header))))
    for row in rows:
        lines.append(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(header))))
    return "\n".join(lines)


def render_pipesd_relative_table(bandwidth: str, records: Sequence[Record]) -> str:
    header = [
        "method",
        "pipesd vs method avg_tpt",
        "pipesd vs method gpu_power_int",
        "pipesd vs method gpu_energy",
    ]

    pipesd = next((rec for rec in records if rec.directory == "pipesd"), None)
    if pipesd is None:
        return f"Bandwidth {bandwidth} (relative to pipesd)\n(no pipesd baseline found)"

    def format_delta(method_value: float, pipesd_value: float) -> str:
        if method_value == 0:
            return "n/a"
        # Improvement of pipesd relative to the method's value.
        delta = (method_value - pipesd_value) / method_value * 100
        return f"{delta:+.2f}%"

    rows = []
    for rec in records:
        if rec is pipesd:
            rows.append([rec.method_label, "baseline", "baseline", "baseline"])
            continue
        rows.append(
            [
                rec.method_label,
                format_delta(rec.avg_tpt, pipesd.avg_tpt),
                format_delta(rec.gpu_power_int, pipesd.gpu_power_int),
                format_delta(rec.gpu_energy, pipesd.gpu_energy),
            ]
        )

    col_widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header)]
    lines = [f"Bandwidth {bandwidth} (relative to pipesd)"]
    lines.append(" | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)))
    lines.append("-+-".join("-" * col_widths[i] for i in range(len(header))))
    for row in rows:
        lines.append(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(header))))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tables from analysis_summary.txt")
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="Path to analysis_summary.txt (default: %(default)s)",
    )
    parser.add_argument(
        "--hsl-st",
        type=str,
        default=None,
        help="Comma-separated st values to include for HSL (defaults to prompting and keeping all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TABLE_PATH,
        help="Path to write the absolute tables (default: %(default)s)",
    )
    parser.add_argument(
        "--relative-output",
        type=Path,
        default=DEFAULT_RELATIVE_PATH,
        help="Path to write the relative tables vs vanilla (default: %(default)s)",
    )
    parser.add_argument(
        "--pipesd-output",
        type=Path,
        default=DEFAULT_PIPESD_RELATIVE_PATH,
        help="Path to write the relative tables vs pipesd (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = parse_summary(args.summary)

    hsl_values_arg = None
    if args.hsl_st:
        hsl_values_arg = [item.strip() for item in args.hsl_st.split(",") if item.strip()]
    hsl_values = choose_hsl_thresholds(records, hsl_values_arg)

    filtered = filter_records(records, hsl_values)
    grouped = group_by_bandwidth(filtered)

    def bw_key(value: str) -> Tuple[int, str]:
        match = re.match(r"(\d+)", value)
        if match:
            return (0, f"{int(match.group(1)):08d}")
        return (1, value)

    ordered_bandwidths = sorted(grouped.keys(), key=bw_key)

    absolute_sections = []
    relative_sections = []
    pipesd_relative_sections = []
    for bandwidth in ordered_bandwidths:
        records_for_bw = grouped[bandwidth]
        absolute_sections.append(render_table(bandwidth, records_for_bw))
        relative_sections.append(render_relative_table(bandwidth, records_for_bw))
        pipesd_relative_sections.append(render_pipesd_relative_table(bandwidth, records_for_bw))

    args.output.write_text("\n\n".join(absolute_sections) + "\n", encoding="utf-8")
    args.relative_output.write_text("\n\n".join(relative_sections) + "\n", encoding="utf-8")
    args.pipesd_output.write_text("\n\n".join(pipesd_relative_sections) + "\n", encoding="utf-8")

    print(f"Wrote absolute tables to {args.output}")
    print(f"Wrote relative tables to {args.relative_output}")
    print(f"Wrote pipesd-relative tables to {args.pipesd_output}")


if __name__ == "__main__":
    main()
