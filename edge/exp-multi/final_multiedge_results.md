# PipeSD 多 Edge 单 Cloud 最终实验结果

## 1. 实验范围与统计口径

本报告基于 `edge/exp-multi/multiclient/` 下的六份最终汇总结果：

- 算法：Vanilla、PipeSD；
- Edge client 数：2、4、8；
- 每组 warmup 60 秒，正式 measurement 600 秒；
- 数据集：HumanEval；
- 每个完成样本生成 128 个 output tokens；
- 仅统计完整位于 measurement window 内的样本；
- 当前目录包含一轮结果（`r1`）。

本文使用以下口径：

$$
\text{Output Throughput}=\frac{\text{total output tokens}}{600}
$$

$$
\text{System TPT}=\frac{600}{\text{total output tokens}}\times1000
=\frac{1000}{\text{Output Throughput}}
$$

System TPT 是系统总体吞吐的倒数，用于表示 Cloud 同时服务多个 Edge 时，每产出一个系统级 output token 所需的墙钟时间。

$$
\text{User-weighted TPT}
=\frac{\sum_i\text{sample total time}_i}
{\sum_i\text{sample output tokens}_i}\times1000
$$
User-weighted TPT 表示客户端实际经历的加权平均每 output token 时间。它与多客户端系统吞吐倒数不同。


## 2. 总体性能

| Clients | Algorithm | 完成样本 | Output tokens | Output throughput (tok/s) | System TPT (ms/tok) | User-weighted TPT (ms/tok) | Sample P50 / P95 (s) | TTFT P50 / P95 (s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | Vanilla | 48 | 6,144 | 10.24 | 97.66 | 182.08 | 20.83 / 35.42 | 1.017 / 1.715 |
| 2 | PipeSD | 57 | 7,296 | **12.16** | **82.24** | **158.45** | **18.72 / 30.17** | **0.365 / 0.840** |
| 4 | Vanilla | 91 | 11,648 | 19.41 | 51.51 | 191.96 | 22.60 / 36.61 | 1.039 / 1.756 |
| 4 | PipeSD | 110 | 14,080 | **23.47** | **42.61** | **161.61** | **19.98 / 29.47** | **0.376 / 0.846** |
| 8 | Vanilla | 167 | 21,376 | 35.63 | 28.07 | 210.87 | 25.45 / 38.60 | 1.102 / 1.830 |
| 8 | PipeSD | 179 | 22,912 | **38.19** | **26.19** | **178.73** | **21.13 / 33.27** | **0.399 / 0.875** |

PipeSD 在 2、4、8 clients 下均获得更高的系统吞吐、更低的用户侧 TPT、更低的样本长尾延迟和更低的 TTFT。

## 3. PipeSD 相对 Vanilla 的改进

| Clients | Output throughput | User-weighted TPT | Sample P50 | Sample P95 | TTFT P50 | TTFT P95 | Cloud GPU J/output token | Acceptance rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | **+18.75%** | **-12.98%** | -10.10% | -14.83% | **-64.14%** | -51.02% | **-18.39%** | +11.05 pp |
| 4 | **+20.88%** | **-15.81%** | -11.59% | -19.49% | **-63.77%** | -51.82% | **-14.66%** | +12.59 pp |
| 8 | **+7.19%** | **-15.24%** | -16.97% | -13.83% | **-63.76%** | -52.22% | **-12.33%** | +13.50 pp |

主要观察：

1. PipeSD 在所有 client 数下都提高了 output throughput；2、4 clients 时提升约 19%--21%，8 clients 时提升 7.19%。
2. User-weighted TPT 稳定降低约 13%--16%，说明 PipeSD 不仅提高系统容量，也降低客户端实际经历的单位 token 生成时间。
3. TTFT P50 稳定降低约 64%，是最一致、幅度最大的延迟收益。
4. Cloud GPU 每 output token 能耗降低约 12%--18%。
5. PipeSD 的 draft acceptance rate 比 Vanilla 高 11.05--13.50 个百分点。

## 4. Speculative decoding、能耗与公平性

| Clients | Algorithm | Verified draft tokens | Accepted draft tokens | Acceptance rate | Accepted-draft throughput (tok/s) | Cloud GPU energy (J) | GPU J/output token | Jain fairness |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | Vanilla | 6,376 | 5,057 | 79.31% | 8.43 | 1,480.75 | 0.241 | 0.9983 |
| 2 | PipeSD | 6,515 | 5,887 | **90.36%** | **9.81** | 1,434.97 | **0.197** | **0.9997** |
| 4 | Vanilla | 12,221 | 9,571 | 78.32% | 15.95 | 3,298.93 | 0.283 | 0.9968 |
| 4 | PipeSD | 12,472 | 11,338 | **90.91%** | **18.90** | 3,403.13 | **0.242** | **0.9997** |
| 8 | Vanilla | 22,526 | 17,550 | 77.91% | 29.25 | 7,762.61 | 0.363 | **0.9986** |
| 8 | PipeSD | 20,496 | 18,736 | **91.41%** | **31.23** | **7,294.64** | **0.318** | 0.9282 |

这些数据说明：

- Vanilla 的 acceptance rate 随 client 数增加略有下降，而 PipeSD 从 90.36% 上升到 91.41%，双阈值 NAV 在多 Edge 并发下仍保持稳定。
- PipeSD C8 验证的 draft tokens 少于 Vanilla，但接受的 draft tokens 更多，说明 Cloud verification 工作的有效率更高。
- C2、C4 下两种算法的 Jain fairness 均高于 0.996；C8 下 Vanilla 为 0.9986，PipeSD 为 0.9282，说明高并发时 PipeSD 的跨 client 服务均衡性仍有优化空间。
- GPU J/output token 随 client 数增加而上升，说明 8-client 阶段已经出现更明显的 Cloud 排队、碎片化 decode 或资源竞争。
