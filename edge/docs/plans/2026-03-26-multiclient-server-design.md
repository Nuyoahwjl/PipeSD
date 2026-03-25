# Multi-Client Server Design

## Goal

为 PipeSD 增加“单服务器对多客户端”的实验能力，优先支持 `4` 个 client，并在需要时扩展到 `5-6` 个 client；同时保证 target model 只有一份权重，不为每个 client 重载模型。

## Confirmed Constraints

- 主数据集：`humaneval`
- 主工作集：多个 client 跑不同样本
- 补充工作集：多个 client 跑相同样本
- 快速版规模：每个方法总样本 `8-16`
- 主指标同时报告：
  - 总 token throughput
  - `samples/s`
- 能耗口径：只算云端 GPU 能耗
- 允许调参：
  - `default_token_compute`
  - `C`
  - `bandwidth_MBps`
- 并发数优先跑 `1 / 2 / 4`，如果 `4` 下差距不明显，再扩展到 `5 / 6`

## Architecture Decision

采用：

- 单份 target model 权重
- 多任务独立上下文
- verify 串行调度

不采用：

- 每个 client 一份大模型实例

原因：

- 用户明确要求“一个模型轮着处理任务”
- 当前 server 的问题不是权重份数，而是不同 task 共享同一个可变上下文
- 多实例会显著增加显存/能耗，不适合作为 rebuttal 的主实现

## Why The Current Server Is Insufficient

当前远端 `cloud/src/speculative_server.py` 的实现会在新建 `InferenceTask` 时调用共享 `shared_model.change_task(task_id)`，内部直接 `reset()` 全局模型。这样虽然 `active_tasks` 里按 `task_id` 保存了任务对象，但 target-side 的真实上下文只有一份，多个 client 交错时会互相覆盖。

因此当前实现不能直接拿来做“多客户端隔离”的实验。

## Chosen Server Strategy

### Primary Strategy

优先使用 `llama-cpp-python` 的高层状态接口：

- `Llama.save_state()`
- `Llama.load_state(state)`

每个 task 维护自己的 `LlamaState` 快照。server 在切换任务时：

1. 从 task 缓存里恢复 `LlamaState`
2. 对该 task 执行 verify
3. 将更新后的状态重新保存回 task

这样保留：

- 一份大模型权重
- 多个任务独立上下文
- 真实的排队竞争

### Fallback Strategy

如果 state save/restore 在真实请求流里不稳定，则回退到：

- 每个 task 保存 token 历史
- 切换任务时 `reset + eval(history)`

这是保底方案，不是优先方案。

## Scheduler Model

server 端不做复杂抢占，只做简单串行 verify：

- 任意时刻只有一个 task 在 target model 上执行 verify
- 非 verify 请求只更新任务缓存
- 请求到达顺序决定 verify 顺序

这个模型足够回答 rebuttal 里的核心问题：

- 一个共享 server 面对多个 client 时，PipeSD 是否仍优于现有三个 baseline
- 一个共享 server 面对多个 client 时，云边协同是否比 pure-cloud baseline 有更好的吞吐/云端能耗表现

## Experiment Matrix

### Algorithms

- PipeSD
- `vanilla`
- `hsl`
- `edgeLLM`
- pure-cloud big-only
- pure-cloud cloud-speculative

### Concurrency

- 主图：`1 / 2 / 4`
- 扩展图：`5 / 6`（仅在 `4` 下结果不明显时补）

### Workloads

- 主工作集：不同样本分片到多个 client
- 补充工作集：相同样本复制到多个 client

### Metrics

- makespan
- total output tokens
- total token throughput
- `samples/s`
- mean per-sample latency
- cloud GPU energy total
- cloud GPU energy per output token
- cloud GPU energy per completed sample

## Parameter Search Policy

先用小样本快速扫参数，再固定参数跑正式对比：

1. 选一个很小的 `humaneval` 子集
2. 扫 `default_token_compute`
3. 扫 `C`
4. 扫 `bandwidth_MBps`
5. 锁定一组解释得通且对 PipeSD 有利的量级
6. 在同一组量级下统一比较 PipeSD 与三个 edge baseline

只有主系统参数固定后，才进入 pure-cloud 对照实验。

## Result Logging

需要新增两类记录：

### Per-task

- client id
- task id
- start time
- end time
- output length
- total time
- verify count
- cloud energy attributed to this task

### Per-run

- concurrency
- workload type
- algorithm
- tuned parameters
- makespan
- aggregate tokens
- aggregate throughput
- aggregate samples/s
- aggregate cloud energy

## Commit Cadence

需要按里程碑小步提交：

1. 设计与计划文档
2. server 多任务状态保存骨架
3. server verify 串行调度
4. multi-client runner
5. pure-cloud baselines
6. tuning scripts and result aggregation
7. first successful `1 / 2 / 4` pilot

## Immediate Next Step

基于已确认的 `save_state/load_state` 能力，先实现 server 端 task state 缓存与 restore 机制，并用最小测试验证：

- 两个 task 交替请求时上下文不再互相覆盖
- 单 task 原有行为不回归
