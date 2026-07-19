"""Create publication-style TPT and energy figures for four deployment modes.

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
DATASETS = ("HumanEval", "GSM8K")
MODES = ("Pure Cloud", "Pure Edge", "Serial Edge-Cloud", "PipeSD")

# User-selected palette for the four deployment modes.
COLORS = {
    "Pure Cloud": "#2751CE",
    "Pure Edge": "#CE2751",
    "Serial Edge-Cloud": "#51CE27",
    "PipeSD": "#EA4E1A",
}
HATCHES = {
    "Pure Cloud": "///",
    "Pure Edge": "\\\\\\",
    "Serial Edge-Cloud": "xxx",
    "PipeSD": "...",
}

MODE_DATA = {
    "HumanEval": {
        "TPT": [4.281, 45.934, 630.228, 503.759],
        "Energy": [176.164, np.nan, 30.292, 20.026],
    },
    "GSM8K": {
        "TPT": [4.221, 33.941, 1066.676, 876.745],
        "Energy": [182.116, np.nan, 45.592, 34.930],
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
            "legend.fontsize": 8.7,
            "figure.titlesize": 13,
            "figure.titleweight": "semibold",
            "hatch.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def style_tpt_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="#D5D5D5", linewidth=0.65, alpha=0.85)
    ax.grid(axis="y", which="minor", color="#ECECEC", linewidth=0.45, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="both", length=3, width=0.7, color="#555555")


def style_energy_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_color("#555555")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=3, width=0.7, color="#555555")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)


def legend_handles():
    return [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS[mode],
            edgecolor="#222222",
            linewidth=0.65,
            hatch=HATCHES[mode],
        )
        for mode in MODES
    ]


def add_labels(ax: plt.Axes, containers) -> None:
    for container in containers:
        ax.bar_label(
            container,
            labels=[f"{bar.get_height():.1f}" for bar in container],
            padding=2,
            fontsize=7.3,
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
    offsets = (np.arange(len(MODES)) - (len(MODES) - 1) / 2) * width

    for ax, dataset in zip(axes, DATASETS):
        energy_ax = ax.twinx()
        tpt_values = MODE_DATA[dataset]["TPT"]
        energy_values = MODE_DATA[dataset]["Energy"]
        tpt_containers = []
        energy_containers = []

        for mode_index, mode in enumerate(MODES):
            common_style = {
                "width": width,
                "color": COLORS[mode],
                "edgecolor": "#222222",
                "linewidth": 0.65,
                "hatch": HATCHES[mode],
                "zorder": 3,
            }
            tpt_containers.append(
                ax.bar(
                    group_centers[0] + offsets[mode_index],
                    tpt_values[mode_index],
                    **common_style,
                )
            )
            if np.isfinite(energy_values[mode_index]):
                energy_containers.append(
                    energy_ax.bar(
                        group_centers[1] + offsets[mode_index],
                        energy_values[mode_index],
                        **common_style,
                    )
                )
            else:
                energy_ax.annotate(
                    "N/A",
                    xy=(group_centers[1] + offsets[mode_index], 0),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    color="#555555",
                )

        ax.set_yscale("log")
        style_tpt_axis(ax)
        style_energy_axis(energy_ax)

        finite_energy = [value for value in energy_values if np.isfinite(value)]
        ax.set_title(dataset, pad=8)
        ax.set_ylabel("TPT (ms/tok., log)")
        energy_ax.set_ylabel("Energy (J/100 tok.)")
        ax.set_xticks(group_centers, ("TPT", "Energy"))
        ax.set_xlim(group_centers[0] - 0.62, group_centers[1] + 0.62)
        ax.set_ylim(min(tpt_values) / 2.2, max(tpt_values) * 2.0)
        energy_ax.set_ylim(0, max(finite_energy) * 1.22)
        add_labels(ax, tpt_containers)
        add_labels(energy_ax, energy_containers)

    legend_anchor = 0.925 if orientation == "horizontal" else 0.955
    fig.legend(
        legend_handles(),
        MODES,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, legend_anchor),
        columnspacing=1.5,
        handlelength=2.0,
    )
    if orientation == "horizontal":
        fig.suptitle("Four-Mode Performance and Energy", y=0.985)
        fig.subplots_adjust(left=0.075, right=0.925, top=0.77, bottom=0.15, wspace=0.38)
    else:
        fig.suptitle("Four-Mode Performance and Energy", y=0.992)
        fig.subplots_adjust(left=0.13, right=0.88, top=0.875, bottom=0.07, hspace=0.30)
    save_figure(fig, f"four_modes_performance_energy_{orientation}")


def main() -> None:
    configure_style()
    plot_performance_and_energy("horizontal")
    plot_performance_and_energy("vertical")
    print(f"Figures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
