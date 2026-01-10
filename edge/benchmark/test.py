from llama_cpp import Llama
import numpy as np
import time
import matplotlib.pyplot as plt
llm = Llama(model_path="pre_models/qwen1.5/qwen1_5-1_8b-chat-q4_k_m.gguf", n_gpu_layers=0, n_threads=4, verbose=False, logits_all=True)

prefix = 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements",'
prefix = 'I am'
# import pynvml

# # 初始化 NVML
# pynvml.nvmlInit()

# def print_gpu_memory(device_id=0):
#     handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
#     info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     print(f"[GPU{device_id}] 显存占用: {info.used / 1024**2:.1f} MB / {info.total / 1024**2:.1f} MB "
#           f"({info.used / info.total * 100:.1f}%)")
print(llm.n_vocab())
out = llm.tokenize(prefix.encode("utf-8"), add_bos=True)
# time1 = time.time()
llm.eval(out)
# time2 = time.time()
# print(f'time:{time2-time1}')
# print(type(llm.scores[0:llm.n_tokens]))
# s1 = np.vstack([llm.scores[0:llm.n_tokens], llm.scores[0]])
# print(s1.shape)
# print(llm.scores[0:llm.n_tokens].shape)

tokens = []
token_times = []
for i in range(200):
    next_token = llm.sample(top_k=5, top_p=0.9, temp=0)
    time1 = time.time()
    llm.eval([next_token])
    time2 = time.time()
    # print(f'time{i}:{time2-time1}')
    tokens.append(next_token)
    token_times.append(time2 - time1)
    if next_token == llm.token_eos():
        break

# print(np.softmax(llm.scores[llm.n_tokens-1]))
# print(tokens)

# token_times: List[float]
t = np.asarray(token_times, dtype=float)
x = np.arange(1, len(t) + 1)  # token序号(也可理解为token长度)

# 可选：滑动平均，让曲线更稳、更像论文图
window = max(1, min(15, len(t)//10))  # 自动给个不太夸张的窗口
kernel = np.ones(window) / window
# t_ma = np.convolve(t, kernel, mode="same")
def moving_average_adaptive(x, window):
    out = np.empty_like(x)
    for i in range(len(x)):
        left = max(0, i - window // 2)
        right = min(len(x), i + window // 2 + 1)
        out[i] = x[left:right].mean()
    return out

t_ma = moving_average_adaptive(t, window)

# 累计时间（有时也很有参考价值）
t_cum = np.cumsum(t)

# ====== 论文风格参数 ======
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 1200,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.0,
})

fig, ax1 = plt.subplots(figsize=(7.2, 4.2))

# 主轴：单token耗时
ax1.plot(x, t * 1000, linewidth=1.2, alpha=0.35, label="Per-token generation time")
ax1.plot(x, t_ma * 1000, linewidth=2.0, label=f"Moving avg token generation time (window={window})")
ax1.set_xlabel("Prefix Length")
ax1.set_ylabel("Time per token (ms)")
ax1.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
ax1.minorticks_on()
ax1.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.35)

# 副轴：累计时间
ax2 = ax1.twinx()
ax2.plot(x, t_cum, linewidth=1.6, linestyle="-.", alpha=0.9, label="Cumulative token generation time")
ax2.set_ylabel("Cumulative time (s)")

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

ax1.set_title("Token generation latency vs. prefix length")
fig.tight_layout()

# 导出：PNG(快速看) + PDF(论文矢量)
plt.savefig("figs/token_latency.png", bbox_inches="tight")
plt.savefig("figs/token_latency.pdf", bbox_inches="tight")
plt.show()