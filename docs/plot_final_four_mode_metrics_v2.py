"""Create V2 four-mode TPT and energy figures.

The values are intentionally hard-coded from the ``four_mode_local_sd_v2``
HumanEval and GSM8K reports.  Unlike the legacy figure, Pure Cloud and Pure
Edge are dual-model, co-located speculative-decoding deployments.  Raster and
vector outputs use a V2-specific filename prefix so historical figures are
never overwritten.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_PREFIX = "four_modes_local_sd_v2_performance_energy"
DATASETS = ("HumanEval", "GSM8K")
MODES = (
    "Pure Cloud SD",
    "Pure Edge SD",
    "Serial Edge-Cloud SD",
    "PipeSD",
)

# Keep mode colors compatible with the original figure while using a distinct
# V2 title treatment, precise accepted-token labels, and V2 output names.
COLORS = {
    "Pure Cloud SD": "#2751CE",
    "Pure Edge SD": "#CE2751",
    "Serial Edge-Cloud SD": "#51A52E",
    "PipeSD": "#EA4E1A",
}
HATCHES = {
    "Pure Cloud SD": "///",
    "Pure Edge SD": "\\\\\\",
    "Serial Edge-Cloud SD": "xxx",
    "PipeSD": "...",
}

# Fixed values recorded from the four_mode_local_sd_v2 experiment summary.
# The plotting path performs no result-file discovery or parsing.
# TPT and energy are normalized by target-accepted draft tokens.
MODE_DATA = {
    "HumanEval": {
        "TPT": [5.532, 204.839, 630.228, 503.759],
        "Energy": [222.132, np.nan, 30.292, 20.026],
    },
    "GSM8K": {
        "TPT": [7.172, 263.538, 1066.676, 876.745],
        "Energy": [298.163, np.nan, 45.592, 34.930],
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
            "legend.fontsize": 8.5,
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


def save_figure(fig: plt.Figure, orientation: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{OUTPUT_PREFIX}_{orientation}"
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_performance_and_energy(orientation: str) -> None:
    if orientation == "horizontal":
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.35))
    elif orientation == "vertical":
        fig, axes = plt.subplots(2, 1, figsize=(6.7, 7.85))
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
        ax.set_ylabel("TPT (ms/accepted tok., log)")
        energy_ax.set_ylabel("Measured energy (J/100 accepted tok.)")
        ax.set_xticks(group_centers, ("TPT", "Energy"))
        ax.set_xlim(group_centers[0] - 0.62, group_centers[1] + 0.62)
        ax.set_ylim(min(tpt_values) / 2.2, max(tpt_values) * 2.0)
        energy_ax.set_ylim(0, max(finite_energy) * 1.22)
        add_labels(ax, tpt_containers)
        add_labels(energy_ax, energy_containers)

    legend_anchor = 0.89 if orientation == "horizontal" else 0.895
    fig.legend(
        legend_handles(),
        MODES,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, legend_anchor),
        columnspacing=1.35,
        handlelength=1.9,
    )
    if orientation == "horizontal":
        fig.suptitle("Four-Mode SD Performance and Energy · V2", y=0.982)
        fig.subplots_adjust(left=0.075, right=0.925, top=0.75, bottom=0.13, wspace=0.38)
    else:
        fig.suptitle("Four-Mode SD Performance and Energy · V2", y=0.988)
        fig.subplots_adjust(left=0.13, right=0.88, top=0.80, bottom=0.06, hspace=0.32)
    save_figure(fig, orientation)


def main() -> None:
    configure_style()
    plot_performance_and_energy("horizontal")
    plot_performance_and_energy("vertical")
    print(f"V2 figures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
