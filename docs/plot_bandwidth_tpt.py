#!/usr/bin/env python3
"""Plot the HumanEval bandwidth sweep in the style of PipeSD Figure 5.

The experiment values are intentionally hard-coded so the figure is a stable,
self-contained record of the run in edge/exp-bandwidth-2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BANDWIDTHS_MBPS = np.array([10, 20, 40, 80])

# Average milliseconds per cloud-accepted draft token.
TPT_MS = {
    "Vanilla": np.array([791.280031, 631.404350, 551.964049, 511.796763]),
    "HSL": np.array([702.608645, 580.643831, 520.602459, 489.869692]),
    "EdgeLLM": np.array([682.014448, 546.169273, 482.938376, 456.717116]),
    "PipeSD": np.array([616.523522, 505.461902, 449.664524, 429.574869]),
}

# Figure 5 visual language: tan cross-hatch, blue backslash, green horizontal,
# and orange-red forward slash. Dark edges preserve readability in grayscale.
COLORS = {
    "Vanilla": "#F1D08A",
    "HSL": "#3D7FB4",
    "EdgeLLM": "#78A88C",
    "PipeSD": "#F04B18",
}
HATCHES = {
    "Vanilla": "xx",
    "HSL": "\\\\",
    "EdgeLLM": "--",
    "PipeSD": "//",
}


def configure_style() -> None:
    """Apply a compact paper-style configuration close to Figure 5."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.0,
            "hatch.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def build_figure() -> tuple[plt.Figure, plt.Axes]:
    configure_style()
    # A wider, flatter canvas keeps the four bar groups visually compact while
    # leaving enough horizontal room for values above every bar.
    fig, ax = plt.subplots(figsize=(11.4, 3.8), constrained_layout=True)

    algorithms = list(TPT_MS)
    positions = np.arange(len(BANDWIDTHS_MBPS), dtype=float)
    bar_width = 0.19
    offsets = (np.arange(len(algorithms)) - (len(algorithms) - 1) / 2) * bar_width

    for offset, algorithm in zip(offsets, algorithms):
        bars = ax.bar(
            positions + offset,
            TPT_MS[algorithm],
            width=bar_width,
            label=algorithm,
            color=COLORS[algorithm],
            edgecolor="#4A4A4A",
            linewidth=0.7,
            hatch=HATCHES[algorithm],
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in TPT_MS[algorithm]],
            padding=3,
            fontsize=8.5,
            rotation=0,
            color="#333333",
        )

    ax.set_xlabel("Bandwidth (Mbps)")
    ax.set_ylabel("Avg. TPT (ms)")
    ax.set_xticks(positions, [str(value) for value in BANDWIDTHS_MBPS])
    ax.set_ylim(0, 880)
    ax.set_yticks(np.arange(0, 801, 200))
    ax.set_xlim(-0.62, len(BANDWIDTHS_MBPS) - 0.38)
    ax.tick_params(direction="out", length=3.5, width=0.9)
    ax.legend(loc="upper right", frameon=False, borderaxespad=0.55)

    # Figure 5 has a clean background without prominent grid lines.
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#4A4A4A")
        spine.set_linewidth(0.9)

    return fig, ax


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "figures" / "bandwidth_tpt_humaneval"
    parser = argparse.ArgumentParser(
        description="Draw the hard-coded HumanEval four-bandwidth TPT results."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output path without extension (default: docs/figures/bandwidth_tpt_humaneval).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution (default: 600; PDF and SVG remain vector graphics).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = build_figure()
    png_path = output.with_suffix(".png")
    pdf_path = output.with_suffix(".pdf")
    svg_path = output.with_suffix(".svg")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
