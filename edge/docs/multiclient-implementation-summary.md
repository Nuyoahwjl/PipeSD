# Multi-Client Implementation Summary

本文记录 2026-03-26 当前仓库里“单 server 对多 client”实验链路的实际实现状态，目标是回答四个问题：

1. 当前多客户端实验到底是怎么跑起来的
2. cloud 和 edge 两侧各自共享了什么、隔离了什么
3. 汇总吞吐和能耗是按什么口径统计的
4. 当前实现有哪些已确认限制

这份文档描述的是“当前实现”，不是最初设计目标。设计意图仍见：

- `docs/plans/2026-03-26-multiclient-server-design.md`
- `docs/plans/2026-03-26-multiclient-server-experiments.md`

## 1. 当前实验拓扑

当前多客户端实验的真实拓扑是：

- 一个远端 cloud server
- 一份共享 target model 权重
- 多个本地 edge client 进程
- 本地 client 通过同一个 server URL 与远端交互

这里的“多客户端”不是多台边缘设备，而是同一台本地机器上并发启动多个 `run_edge.py` 进程。因此：

- cloud 端会发生真实的多任务竞争
- edge 端也会发生本地资源竞争

这点对解释吞吐结果非常重要。单客户端结果不能直接拿来和多客户端 aggregate throughput 对齐比较。

## 2. Cloud 侧实现

当前远端 server 入口仍然是：

```bash
./run_server.sh
```

核心逻辑在远端 `src/speculative_server.py`。当前实现已经不是最早那种“全局上下文直接被后来的 task 覆盖”的版本，而是：

- 保留一份共享 `Llama` target model 权重
- 每个 task 保存自己的 `LlamaState`
- `/init` 后保存 prefix 对应的 state
- `/propose` 验证前恢复该 task 的 state
- 验证完成后再保存更新后的 state

因此 cloud 侧目前是：

- 单份 target model 权重
- 多任务独立上下文
- verify 在共享 model 上串行执行

这满足了“一个模型轮着处理多个任务”的要求，但没有做跨 task batching。

## 3. Edge 侧实现

多客户端主入口是：

```bash
python3 scripts/run_multiclient_pilot.py ...
```

它的职责是：

- 按 `pilot_samples` 和 `num_clients` 切分样本
- 为每个 client 构造独立的 `run_edge.py` 命令
- 给每个 client 分配独立 `result_tag`
- 给每个 client 分配 `task_id_offset`
- 并发启动多个 `run_edge.py`
- 等全部 client 完成后，聚合结果写入 `exp/multiclient/<base_tag>.json`

当前 task id 隔离方式是：

- client 0: `task_id_offset = 0`
- client 1: `task_id_offset = 1_000_000`
- client 2: `task_id_offset = 2_000_000`
- client 3: `task_id_offset = 3_000_000`

这样即使原始数据集里 task id 相同，也不会在 cloud 侧冲突。

## 4. 单个 client 的执行链路

每个 client 实际上还是普通的：

```bash
python3 app/run_edge.py ...
```

单个 client 的关键链路如下：

1. 本地加载 draft model
2. 对 prefix 做 tokenize
3. 调 cloud `/init`
4. 本地 draft model 逐 token speculative generation
5. 把 speculative token 和 draft 概率发给 cloud `/propose`
6. cloud 返回 `n_accepted` 和 `final_token`
7. 本地更新 draft 状态，继续下一轮
8. 结束时调 `/exit`，取回 cloud GPU 能耗积分

注意：当前 `/propose` 发给 cloud 的不仅是 token，还包括每个 speculative token 的全词表概率分布。这意味着通信 payload 很大，在低带宽设定下会直接主导整体时延。

## 5. 吞吐与能耗统计口径

当前多客户端聚合结果同时记录两种吞吐口径。

### 5.1 Wall-clock makespan

聚合脚本把：

- 从第一个 client 进程启动
- 到最后一个 client 进程退出

之间的整段墙钟时间记为：

- `makespan_seconds`

对应吞吐是：

- `token_throughput_tps`
- `sample_throughput_sps`

这个口径会把以下开销都算进去：

- client 进程启动
- draft model 首次加载
- 每个 client 的第一条样本初始化

### 5.2 Sample-window makespan

当前实现另外记录每条样本的：

- `sample_started_at`
- `sample_finished_at`

然后用所有样本的最早开始和最晚结束构造：

- `sample_window_makespan_seconds`

对应吞吐是：

- `sample_window_token_throughput_tps`
- `sample_window_sample_throughput_sps`

这个口径仍然是端到端实验时间，但比 wall-clock 更少受进程冷启动污染，更接近 steady-state。

### 5.3 能耗

当前多客户端文档和脚本统一只统计 cloud GPU 能耗：

- 每个 task 的 `gpu_power_integral_joules` 来自 cloud `/exit`
- 聚合后得到 `total_cloud_energy_joules`
- 再计算 `energy_per_token_joules`
- 再计算 `energy_per_sample_joules`

本地 edge 端能耗当前没有计入正式结果。

## 6. Draft model 的 GPU 配置现状

当前 edge 侧已经支持通过 CLI 配置：

```bash
--draft_n_gpu_layers <int>
```

对应实现位于：

- `src/util.py`
- `src/engine.py`

理论上，传：

```bash
--draft_n_gpu_layers -1
```

意味着尽可能把 draft model 全部放到 GPU。

但在当前本地环境里，这个设置没有真正生效。已确认事实是：

- 本地 `llama_cpp` 运行时 `llama_supports_gpu_offload()` 返回 `False`
- 运行多客户端实验时 `nvidia-smi` 一直显示 `0 MiB / 0%`

因此当前本地 draft model 仍然实际上跑在 CPU 路径上。换句话说：

- 参数已接通
- 但本地 `llama-cpp-python` 不是 GPU-offload 可用构建

所以不能把当前结果解释成“4 个 GPU draft client”的实验。

## 7. 当前已确认的性能限制

截至目前，已经确认的多客户端瓶颈有四类。

### 7.1 Cloud verify 串行

cloud 端只有一份 target model。虽然每个 task 有独立 state，但 verify 仍然是轮流执行的，没有做跨 task batching。

### 7.2 Edge 端本地竞争

当前多个 client 都在同一台本地机器上运行，不是多个真实独立 edge 设备。因此：

- CPU
- 内存
- 本地 I/O

都会被多个 draft client 共享。

### 7.3 通信 payload 大

当前 `/propose` 会发送全词表概率，`32256` 词表下单个 speculative token 的 payload 就接近 `0.29 MB`。因此低带宽设定会迅速把 PipeSD 推到“通信主导”的区域。

### 7.4 Cold-start 污染

如果 `pilot_samples` 太小，例如总共只有 `4` 个样本，那么：

- 每个 client 可能只跑 `1` 个样本
- draft model 加载和样本初始化会严重污染 wall-clock 吞吐

因此当前正式判断趋势时，更应优先看：

- `pilot_samples >= 8`
- `sample_window_*` 指标

## 8. 当前实验结果该如何解释

到 2026-03-26 为止，已经可以确认两点：

1. 之前“多客户端吞吐低得离谱”并不完全是算法本身慢，确实有 cold-start 和统计口径污染。
2. 但即使使用修正后的 `sample-window` 口径，在当前这套实现和参数下，PipeSD 仍然不一定优于 `vanilla`。

所以目前不能再把问题简单归因成“算错了”，而应该分开看：

- 统计口径是否公平
- 当前参数是否落在有利于 PipeSD 的区域
- 当前系统实现是否已经接近预期实验设定

## 9. 当前推荐的阅读顺序

如果之后要继续接手这个多客户端实验，建议按下面顺序回看：

1. `docs/multiclient-implementation-summary.md`
2. `docs/cloud-server-summary.md`
3. `docs/repo-summary.md`
4. `scripts/run_multiclient_pilot.py`
5. `src/multiclient.py`
6. `src/engine.py`

这样会比直接从 `engine.py` 开始读更快进入状态。
