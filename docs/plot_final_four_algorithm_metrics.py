"""Create publication-style figures for the final four-algorithm evaluation.

All values are intentionally hard-coded from the final HumanEval and GSM8K
summary reports. The script writes raster (PNG) and vector (PDF) versions to
``docs/figures``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

METHODS = ("Vanilla", "HSL", "EdgeLLM", "PipeSD")
DATASETS = ("HumanEval", "GSM8K")

# Color-blind-friendly palette paired with hatch patterns for grayscale output.
PERFORMANCE_COLORS = {
    "Vanilla": "#9ECAE1",
    "HSL": "#6BAED6",
    "EdgeLLM": "#3182BD",
    "PipeSD": "#EF3B2C",
}
HATCHES = {
    "Vanilla": "///",
    "HSL": "\\\\\\",
    "EdgeLLM": "xxx",
    "PipeSD": "...",
}

PERFORMANCE_DATA = {
    "HumanEval": {
        "TPT": [628.705, 578.558, 548.539, 503.444],
        "Energy": [30.060, 21.643, 20.552, 19.121],
    },
    "GSM8K": {
        "TPT": [1062.706, 949.016, 910.411, 872.472],
        "Energy": [45.453, 36.254, 36.900, 34.743],
    },
}

def configure_style() -> None:
    """Apply restrained styling suitable for papers and grayscale printing."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "figure.titleweight": "semibold",
            "hatch.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="both", length=3, width=0.7, color="#555555")


def add_labels(
    ax: plt.Axes,
    containers,
    formatter,
    fontsize: float = 7.2,
) -> None:
    for container in containers:
        labels = []
        for bar in container:
            value = bar.get_height()
            labels.append("" if not np.isfinite(value) else formatter(value))
        ax.bar_label(
            container,
            labels=labels,
            padding=2,
            fontsize=fontsize,
            rotation=0,
            color="#222222",
        )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_performance_and_energy(orientation: str) -> None:
    if orientation == "horizontal":
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    elif orientation == "vertical":
        fig, axes = plt.subplots(2, 1, figsize=(6.6, 7.6))
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")

    axes = np.atleast_1d(axes)
    group_centers = np.array([0.0, 1.35])
    width = 0.19
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width

    for ax, dataset in zip(axes, DATASETS):
        energy_ax = ax.twinx()
        tpt_values = PERFORMANCE_DATA[dataset]["TPT"]
        energy_values = PERFORMANCE_DATA[dataset]["Energy"]
        tpt_containers = []
        energy_containers = []

        for method_index, method in enumerate(METHODS):
            common_style = {
                "width": width,
                "color": PERFORMANCE_COLORS[method],
                "edgecolor": "#222222",
                "linewidth": 0.65,
                "hatch": HATCHES[method],
                "zorder": 3,
            }
            tpt_containers.append(
                ax.bar(
                    group_centers[0] + offsets[method_index],
                    tpt_values[method_index],
                    **common_style,
                )
            )
            energy_containers.append(
                energy_ax.bar(
                    group_centers[1] + offsets[method_index],
                    energy_values[method_index],
                    **common_style,
                )
            )

        style_axis(ax)
        energy_ax.grid(False)
        energy_ax.spines["top"].set_visible(False)
        energy_ax.spines["left"].set_visible(False)
        energy_ax.spines["right"].set_color("#555555")
        energy_ax.spines["bottom"].set_visible(False)
        energy_ax.tick_params(axis="y", length=3, width=0.7, color="#555555")
        energy_ax.tick_params(axis="x", bottom=False, labelbottom=False)

        ax.set_title(dataset, pad=8)
        ax.set_ylabel("TPT (ms/tok.)")
        energy_ax.set_ylabel("Energy (J/100 tok.)")
        ax.set_xticks(group_centers, ("TPT", "Energy"))
        ax.set_xlim(group_centers[0] - 0.62, group_centers[1] + 0.62)
        ax.set_ylim(200, max(tpt_values) * 1.2)
        energy_ax.set_ylim(10, max(energy_values) * 1.4)
        add_labels(ax, tpt_containers, lambda value: f"{value:.1f}", fontsize=7.5)
        add_labels(energy_ax, energy_containers, lambda value: f"{value:.1f}", fontsize=7.5)

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PERFORMANCE_COLORS[method],
            edgecolor="#222222",
            linewidth=0.65,
            hatch=HATCHES[method],
        )
        for method in METHODS
    ]
    legend_anchor = 0.925 if orientation == "horizontal" else 0.955
    fig.legend(
        legend_handles,
        METHODS,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, legend_anchor),
        handlelength=2.0,
    )
    if orientation == "horizontal":
        fig.suptitle("Performance and Cloud GPU Energy", y=0.985)
        fig.subplots_adjust(left=0.075, right=0.925, top=0.77, bottom=0.15, wspace=0.38)
    else:
        fig.suptitle("Performance and Cloud GPU Energy", y=0.992)
        fig.subplots_adjust(left=0.13, right=0.88, top=0.875, bottom=0.07, hspace=0.30)
    save_figure(fig, f"four_algorithms_performance_energy_{orientation}")


def main() -> None:
    configure_style()
    plot_performance_and_energy("horizontal")
    plot_performance_and_energy("vertical")
    print(f"Figures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
