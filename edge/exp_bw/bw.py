# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import numpy as np
import os

# ============================
# Data
# ============================
bandwidths_mb = [1, 2.5, 5, 10]
bandwidths_mbps = [bw * 8 for bw in bandwidths_mb]   # Mbps
bandwidths_mbps = [int(bw) for bw in bandwidths_mbps]
x = np.arange(len(bandwidths_mbps))

data = {
    "Vanilla": {
        "avg_tpt":       [366.6, 193.5, 115.6, 95.2],
        "gpu_power_int": [132,  135,  126.46, 156.05],
    },
    "HSL": {
        "avg_tpt":       [289.2, 154.8, 100.8, 86.0],
        "gpu_power_int": [148,  141,  124.72, 123.91],
    },
    "edgeLLM": {
        "avg_tpt":       [306.3, 153.0, 89.8, 81.2],
        "gpu_power_int": [150,  132,  115.40, 147.01],
    },
    "PipeSD": {
        "avg_tpt":       [284.4, 131.5, 79.7, 71.3],
        "gpu_power_int": [114,  113.52,  97.47,  104.38],
    },
}

methods = ["Vanilla", "HSL", "edgeLLM", "PipeSD"]

# ============================
# Style (from your template)
# ============================
color_mine1 = [1, 0.270588235294118, 0]            # red
color_mine2 = [57/255, 112/255, 165/255]           # blue
color_mine3 = [130/255, 178/255, 154/255]          # green
color_mine4 = [242/255, 204/255, 142/255]          # yellow

colors = {
    "Vanilla": color_mine4,
    "HSL":     color_mine2,
    "edgeLLM": color_mine3,
    "PipeSD":  color_mine1,
}

hatches = {
    "Vanilla": "x",
    "HSL":     "\\",
    "edgeLLM": "-",
    "PipeSD":  "/",
}

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ============================
# Plot function (template-style)
# ============================
def plot_bar(metric, ylabel, out_name):
    plt.figure(figsize=(8, 5), dpi=900)

    bar_width = 0.18

    for i, m in enumerate(methods):
        plt.bar(
            x + i * bar_width,
            data[m][metric],
            bar_width,
            color=colors[m],
            hatch=hatches[m],
            edgecolor="k",
            label=m,
        )

    plt.xlabel("Bandwidth (Mbps)", size=25)
    plt.ylabel(ylabel, size=25)

    

    plt.xticks(
        x + 1.5 * bar_width,
        [str(bw) for bw in bandwidths_mbps],
        size=25
    )

    ax = plt.gca()
    ax.set_ylim(bottom=60)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    plt.yticks(size=25)

    if metric == "avg_tpt":
        plt.legend(fontsize=22, frameon=False, loc="best")
    else:
        plt.legend(
        ncol=4,
        fontsize=22,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.2,
        handlelength=1.5,
        handletextpad=0.5,
    )

    # thick spines
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(out_name, bbox_inches="tight")
    # plt.show()


# ============================
# Draw figures
# ============================
plot_bar(
    metric="avg_tpt",
    ylabel="Avg. TPT (ms)",
    out_name="avg_tpt.png",
)

plot_bar(
    metric="gpu_power_int",
    ylabel="Avg. ECS (J)",
    out_name="avg_ecs.png",
)
