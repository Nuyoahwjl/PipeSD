import requests
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from .merge import dynamic_token_scheduling_dp

URL = "http://106.63.100.63:30007"
DELAY_ENDPOINT = f"{URL}/delay"

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示

accumulated_time = []
# payloads = [b'a' * i * 2900000 for i in range(10)]  # 0MB ~ 9MB
# payloads = [b'a' * i * 290000 for i in range(3)]  # 0MB ~ 9MB
payloads = [b''] * 10  # 0MB ~ 9MB
   
def test_latency():
    for j in range(10):
        send_time = time.time()
        response = requests.post(DELAY_ENDPOINT, data=payloads[j])
        r_time = time.time()
        print(f"j={j}, RTT: {r_time - send_time:.8f} 秒")
        accumulated_time.append(r_time - send_time)

def linear_fit(xs, ys, xlabel="Number of Tokens", ylabel="Communication Time (s)"):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    # -------- 线性拟合 y = kx + b --------
    k, b = np.polyfit(xs, ys, 1)
    y_pred = k * xs + b

    # R^2
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # 打印结果（保留你的输出习惯）
    print("\n====== 线性拟合结果 ======")
    print(f"斜率 k（每单位 x 增加的时间）: {k:.6f}")
    print(f"截距 b（固定时延）           : {b:.6f}")
    print(f"R^2 拟合优度                 : {r2:.6f}")

    # -------- 科研风格：全局参数（局部设置，不污染其他图）--------
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "figure.dpi": 140,
        "savefig.dpi": 1200,
        "mathtext.fontset": "stix",
        "font.family": "DejaVu Sans",  # 跨平台稳定；若你有 Times New Roman 可自行换
    })

    # -------- 画布与坐标轴（推荐使用 constrained_layout）--------
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)

    # 数据点：空心点更“论文”
    ax.scatter(
    xs, ys,
    s=48,
    edgecolors="black",
    facecolors="white",   # 关键
    linewidths=1.2,
    zorder=3,
    label="Measured"
)

    ax.plot(
        xs, y_pred,
        linewidth=2.0,
        zorder=2,
        label=rf"Linear fit $\alpha={b:.4f}$, $\beta={k:.4f}$"
    )

    # 坐标轴标签
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 科研常用：只保留左/下边框（干净）
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 主/次刻度 + 刻度朝内
    ax.tick_params(which="both", direction="in", top=False, right=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # 网格：淡一点，且主/次区分
    ax.grid(True, which="major", linewidth=0.7, alpha=0.25)
    ax.grid(True, which="minor", linewidth=0.5, alpha=0.12)

    # 刻度：只保留主刻度
    ax.minorticks_off()
    ax.tick_params(which="major", direction="in")

    # 网格：要么不要，要么极淡
    # ax.grid(False)

    # 图例：短小即可
    ax.legend(frameon=False, loc="upper left")

    # 把拟合信息放图内（更像 paper）
    # 这里按你写法：alpha=b, beta=k
    # text = (
    #     rf"$y=\beta x+\alpha$" "\n"
    #     rf"$\alpha={b:.4f}$, $\beta={k:.4f}$" "\n"
    #     rf"$R^2={r2:.4f}$"
    # )
    # ax.text(
    #     0.98, 0.05, text,
    #     transform=ax.transAxes,
    #     ha="right", va="bottom",
    #     bbox=dict(boxstyle="round,pad=0.25", alpha=0.12, linewidth=0.8)
    # )

    # 让拟合线覆盖范围更自然：用 x 的最小/最大延伸
    x_min, x_max = xs.min(), xs.max()
    ax.set_xlim(x_min - 0.02*(x_max-x_min), x_max + 0.02*(x_max-x_min))

    # plt.show()
    plt.savefig("linear_fit_communication_time.png", bbox_inches="tight")

    return k, b, r2

if __name__ == "__main__":

    st = time.time()
    test_latency()
    et = time.time()
    print(f"Total test latency time: {et - st:.6f} seconds")

    # X：每个 j 对应的 payload 大小（MB）
    xs = np.array([i for i in range(1, 10)])  # 因为 1MB≈1,000,000 而你用 2,900,000

    # Y：测到的 RTT
    ys = np.array(accumulated_time)
    ys = np.array([
    # 0.02397871,
    0.13031721,
    0.21771407,
    0.29609466,
    0.40678072,
    0.48185134,
    0.56132030,
    0.63236594,
    0.73809338,
    0.80314398
]) * 0.29

    # linear_fit(xs, ys)

    # time1 = time.time()
    # batches, _ = dynamic_token_scheduling_dp(
    #             [0.03] * 8, 0.025,
    #             0.025)
        
    # merge_plan_batches = [len(batch) for batch in batches if batch]
    # time2 = time.time()
    # print(f"[DP] 计算时间: {time2 - time1:.6f} 秒")
    # # merge_plan_batches = [7, 10]
    # print(f"[Edge] 计算得到合并计划: {merge_plan_batches}")
