# Lazy Distribution 最新实验分析

本文只分析 `20260808-080512` 的最新一组 Full/Lazy 配对实验。实验采用 HumanEval、PipeSD、串行 cloud 后端、DP merge，上行带宽为 2.5 MB/s，上行固定启动开销为 25 ms；两组均以 1000 个 accepted draft tokens 为停止条件。

## 1. Accepted-token TPT

| 模式 | Accepted-token TPT | Lazy 相对 Full |
|---|---:|---:|
| Full distribution | 506.728 ms | — |
| Lazy distribution | 422.601 ms | **降低 16.60%** |

Lazy 将每个 accepted draft token 的平均端到端时间减少约 84.13 ms。两组的验证次数、平均 draft 长度、接受率和 rollback rate 均相同，因此该收益主要来自概率分布传输方式变化，而不是验证工作量变化。

## 2. α、β、γ 估计

DP 使用如下通信模型：

$$
T_{\mathrm{comm}}(n)=\alpha+\beta n
$$

- α：一次上传的固定开销，主要对应网络 startup、请求和序列化开销。
- β：每增加一个 token 所增加的上传时间。
- γ：edge 生成一个 draft token 的平均时间。

实验中调度参数的初值为 `initial_generation_gamma=0.036`，即 36 ms/token。

### 初始值

| 模式 | α₀ | β₀ | γ₀ |
|---|---:|---:|---:|
| Full | 25 ms | 116 ms/token | 36 ms/token |
| Lazy | 25 ms | 4.801 µs/token | 36 ms/token |

Full 按每 token 约 0.29 MB 的序列化完整分布估算 β₀；Lazy 的普通消息每 token 原始数据约为 12 bytes，因此初始 β 降低了约四个数量级。

### 在线测量结果

| 参数 | Full 测量范围（中位数） | Lazy 测量范围（中位数） |
|---|---:|---:|
| α | 31.795–37.185 ms（33.687 ms） | 27.140–27.479 ms（27.318 ms） |
| β | 115.044–116.114 ms/token（115.591 ms） | 4.603–4.760 µs/token（4.689 µs/token） |
| γ | 42.824–44.492 ms/token（43.215 ms） | 43.477–44.820 ms/token（44.399 ms） |
| 回归 R² | 0.999789–0.999980 | 0.996017–0.998269 |

Lazy 的回归均超过配置的最低要求：

$$
R^2 \ge 0.8
$$

其测得有效带宽为 2.520–2.606 MB/s，与配置的 2.5 MB/s 接近；每 token 序列化数据稳定在约 11.99 bytes。因此这次 Lazy 的 α/β 估计是可信的。


## 3. DP 调度结果

Lazy 上传一个普通 token 的预计时间为：

$$
T_{\mathrm{lazy}}(1)
=26.791\text{ ms}+0.004849\text{ ms}
\approx26.796\text{ ms}
$$

它小于约 44.087 ms 的 draft token 生成间隔。因此当前 token 的上传能在下一个 token 生成前完成，立即发送比等待合并更有利。Lazy 的 DP 计划因此全部为：

```text
[1, 1, 1, ...]
```

其平均实际 batch size 为 1.0。这不是 DP 退化，而是由高质量回归参数支持的合理结果。

Full 上传一个 token 约需：

$$
T_{\mathrm{full}}(1)
\approx35.320+115.455
=150.775\text{ ms}
$$

该时间显著大于约 43 ms 的 token 生成间隔，逐 token 上传会造成队列积压，所以 Full 需要通过 `[1,2]`、`[1,3]`、`[1,4]`、`[1,2,3]` 等计划合并上传，平均实际 batch size 为 1.632。

## 4. 通信量变化

以下统计包含在线通信探测流量：

| 指标 | Full | Lazy | 变化 |
|---|---:|---:|---:|
| 上传请求数 | 816 | 1543 | +89.09% |
| 总上传量 | 389.959 MiB | 8.184 MiB | **-97.90%** |
| 累计 uplink service time | 183.961 s | 42.008 s | **-77.17%** |
| 累计上传排队时间 | 36.735 s | 0.047 s | **-99.87%** |

Lazy 因逐 token 上传和更密集的回归探测而增加了请求数，但普通 token 的 payload 极小，所以总上传量和排队时间仍大幅下降。其约 42 秒累计 service time 中，仅 1543 次请求的 25 ms startup 理论上就占约 38.58 秒，说明 Lazy 当前已经从“payload 主导”转为“请求固定开销主导”。这些请求可以与 draft generation 流水重叠，因此不会全部进入端到端关键路径。

排除通信探测后，主要业务上传量约为：

| 业务流量 | Full | Lazy |
|---|---:|---:|
| 普通 proposal | 340.672 MiB | 0.269 MiB |
| 拒绝后的 residual distribution | 0 | 5.045 MiB |
| 合计 | 340.672 MiB | 约 5.314 MiB |

按业务流量估算，Lazy 减少约 98.44% 的上传数据。本次共有 41 次 rollback，Lazy 对应发起 41 次 resolve；每次 residual distribution 的大小为：

$$
32256\times4=129024\text{ bytes}=126\text{ KiB}
$$

这说明完整概率分布只在拒绝时补传，而正常 proposal 只携带轻量信息。

