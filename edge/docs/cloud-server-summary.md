# Cloud Server Summary

本文记录 2026-03-26 对远端服务仓库 `/vepfs_hyh/hyh/PipeSD` 的代码检查结果，目的有两个：

1. 以后回看 server 端架构时，不需要重新登录远端翻代码。
2. 为“一台 server 对多客户端”的实验设计提供准确前提，而不是只看 edge 端做猜测。

这份文档是对本地 [`docs/repo-summary.md`](/mnt/c/files/PipeSD/edge/docs/repo-summary.md) 的补充。前者偏 edge/client 侧，这份偏 cloud/server 侧，并且把两边如何对接也写清楚。

---

## 1. 远端仓库定位

远端仓库是目标模型所在的验证服务，不是训练仓库，也不是完整的前后端系统。它的职责很集中：

- 接收 edge 端发来的 prefix，建立任务上下文
- 接收 speculative token 和 draft 概率
- 用 target model 做 acceptance check
- 返回 `n_accepted` 和 `final_token`
- 统计每次验证阶段的 GPU 功耗积分
- 在任务结束时返回累积的能耗和验证次数

从代码上看，它就是一个单进程 FastAPI 服务，加上一个常驻的共享 `Llama` 模型实例。

---

## 2. 远端仓库结构

远端 `/vepfs_hyh/hyh/PipeSD` 当前文件非常少，核心基本都在一个文件里：

- `run_server.sh`
  - 实际启动入口
- `src/speculative_server.py`
  - FastAPI app
  - 共享 target model 初始化
  - `InferenceTask` 状态对象
  - `/init`、`/propose`、`/exit`、`/health`、`/delay`
- `src/util.py`
  - 参数解析
  - 概率工具函数
  - GPU 功耗采样与积分逻辑
- `logs/communication_service.log`
  - 运行日志，包含 verify 细节、采样 token、GPU 功耗积分
- `README.md`
  - 服务用途和 API 的高层说明

换句话说，这个 server 仓库还没有明显模块化；如果后面要做多客户端实验或多任务隔离，`src/speculative_server.py` 会是第一修改点。

---

## 3. 启动链路

远端入口是：

```bash
./run_server.sh
```

`run_server.sh` 只做三件事：

```bash
cd "$(dirname "$0")"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python -m src.speculative_server "$@"
```

也就是说真正的服务入口是 `python -m src.speculative_server`。

`src/speculative_server.py` 在 `__main__` 里做这些事：

1. `parse_arguments()`
2. `seed_everything(args.seed)`
3. 创建全局 `shared_model = MyModel(args.target_model, 16384)`
4. `uvicorn.run(app, host="0.0.0.0", port=APP_PORT, workers=1)`

关键结论：

- 服务是单 worker 启动的
- target model 只初始化一次
- 这个 target model 不是“每个 task 一份”，而是“全局共享一份”

这对多客户端实验是核心事实。

---

## 4. API 与 edge 端对接关系

本地 edge 端在 [`src/engine.py`](/mnt/c/files/PipeSD/edge/src/engine.py#L24) 到 [`src/engine.py`](/mnt/c/files/PipeSD/edge/src/engine.py#L31) 固定了服务地址和三个主要 endpoint：

- `POST /init`
- `POST /propose`
- `POST /exit`

另一个本地脚本 [`scripts/measure_rtt.py`](/mnt/c/files/PipeSD/edge/scripts/measure_rtt.py#L12) 额外使用了 `POST /delay`。

server 端对应暴露的是：

- `POST /init`
- `POST /propose`
- `POST /exit`
- `POST /delay`
- `GET /health`
- `GET /`

因此 edge 和 server 的接口面基本是对齐的。

需要注意两个实现层面的不一致：

1. 远端 `README.md` 写的是默认端口 `8000`，但 `src/speculative_server.py` 里 `APP_PORT = 8001`
2. 本地 edge 端默认访问的是 `http://115.190.90.101:1597`

因此可以合理推断：你当前实际使用的 `1597` 并不是 repo 内 uvicorn 直接暴露的默认端口，外面大概率还有端口映射、反向代理或平台分配端口。这个结论是推断，不是代码内显式配置。

---

## 5. Server 端核心对象

### 5.1 `MyModel`

`MyModel` 是对 `llama_cpp.Llama` 的一层薄包装，职责主要有：

- 初始化 target model
- 记录当前 `task_id`
- 在 `sample()` 前后打印调试日志
- 提供 `change_task(task_id)`，内部直接调用 `self.model.reset()`

这里最重要的不是日志，而是 `change_task()` 的行为：切换任务时会直接 reset 那个共享模型。

### 5.2 `InferenceTask`

每个请求任务在 `/init` 时会创建一个 `InferenceTask`，对象里保存：

- `task_id`
- `prefix`
- `n_past`
- `final_token`
- `accumulated_tokens`
- `accumulated_probs`
- 局部采样/随机数状态
- GPU 功耗积分累计
- `last_verify_pass`

但是它并没有真正拥有独立的 target model。初始化时它会执行：

1. `shared_model.change_task(task_id)`
2. `self.target_model = shared_model.model`

也就是说：

- `InferenceTask` 只是“任务状态对象”
- 所有任务复用同一个底层 `Llama`
- 新任务初始化时会重置这个全局模型上下文

这意味着当前实现并不是真正的“多 client 隔离上下文”架构。

### 5.3 `active_tasks`

server 侧用一个全局字典：

```python
active_tasks: Dict[int, InferenceTask] = {}
```

它只是按 `task_id` 保存任务对象，方便 `/propose` 和 `/exit` 查回对应任务。

这层映射只能说明“任务有 ID”，不能说明“任务有独立模型状态”。真正的模型状态仍然挂在全局 `shared_model.model` 上。

---

## 6. 请求生命周期

### 6.1 `/init`

`/init` 接收：

- `task_id`
- `tokens`

执行流程：

1. `parse_arguments()`
2. 新建 `InferenceTask`
3. 在 `InferenceTask.__init__()` 中调用 `shared_model.change_task(task_id)`，这一步会 reset 全局 target model
4. `proc_prefix()` 对 prefix 执行一次 `target_model.eval(prefix)`
5. 把 task 放入 `active_tasks[task_id]`

所以 `/init` 的真实含义不是“分配一个独立 session”，而是“把共享模型切到这个 task，并把 prefix 跑进去”。

### 6.2 `/propose`

`/propose` 接收 msgpack，主要字段有：

- `task_id`
- `tokens`
- `probs`
- `index`
- `n_past`
- `should_verify`
- `type`

分两种路径：

- `should_verify = False`
  - 只把 token/prob 追加到 `accumulated_tokens` / `accumulated_probs`
- `should_verify = True`
  - 先追加数据
  - 再调用 `verify_tokens(n_past_at_receive)`
  - 返回 `n_accepted`、`final_token`、`gpu_power_integral`

如果是 `type == 'propose_waiting'`，还会检查上一次验证是否全通过；如果没通过，直接返回 drop，不继续验证等待期 token。

### 6.3 `verify_tokens()`

这是 server 端真正的算法核心，做的事是：

1. 用当前累计的 speculative token 构造 `draft_probs`
2. 必要时把 `target_model.n_tokens` 回退到 `n_past_at_verify`
3. 对 speculative token 做一次 target model `eval`
4. 取出 target logits，算 `target_probs`
5. 对每个 speculative token 计算 `p_ratio = target_prob / draft_prob`
6. 用固定 seed + 序列位置生成随机数，按 speculative decoding 接受规则逐个判断
7. 如果中间拒绝，则从 `target_probs - draft_probs` 采样 `final_token`
8. 如果全通过，则直接调用共享模型 `sample()`
9. 返回 `n_accepted` 和 `final_token`

这里的随机数设计是按绝对位置复现 acceptance decision，而不是按请求到达顺序共享一个全局 RNG，这一点对可重复性是有帮助的。

### 6.4 `/exit`

`/exit` 会：

1. 从 `active_tasks` 弹出对应 task
2. 返回任务累计的 `gpu_power_integral_joules`
3. 返回 `verify_num`

因此 edge 端最终写回结果文件里的功耗积分，其实就是从 server `/exit` 带回来的。

---

## 7. 与本地 edge 端的关系

本地 edge 端的总体结构已经在 [`docs/repo-summary.md`](/mnt/c/files/PipeSD/edge/docs/repo-summary.md) 里写过，这里只补充 server 对接视角下最重要的几点。

### 7.1 edge 端是串行样本驱动

[`app/run_edge.py`](/mnt/c/files/PipeSD/edge/app/run_edge.py#L197) 到 [`app/run_edge.py`](/mnt/c/files/PipeSD/edge/app/run_edge.py#L227) 的 `eval()` 是逐样本串行跑的。当前本地仓库默认不是多 client 压测器。

### 7.2 edge 端每个样本新建一个 `BandwidthSender`

[`src/engine.py`](/mnt/c/files/PipeSD/edge/src/engine.py#L75) 到 [`src/engine.py`](/mnt/c/files/PipeSD/edge/src/engine.py#L106) 的 `_reset_state()` 里，每个样本都会新建一个 [`src/comm.py`](/mnt/c/files/PipeSD/edge/src/comm.py#L32) `BandwidthSender`。

这个 sender 自己只有一个工作线程，因此“单个 edge client 内部”的上传仍然是单 worker 串行出队，只是应用层可以异步提交多个 future。

### 7.3 edge 端已有“生成与通信重叠”

[`src/engine.py`](/mnt/c/files/PipeSD/edge/src/engine.py#L381) 之后的主循环里，`pipesd`/`edgeLLM` 会在等待 verify 返回时继续生成并提交 `propose_waiting`。

所以你现在代码里的“并发”主要是：

- 单 client 内部的 speculative generation 和网络发送重叠

而不是：

- 多个独立 client 同时占用同一个 server

这正是 rebuttal 会盯住的点。

---

## 8. 对多客户端实验最关键的架构事实

下面这几条是后面做 4-client 并发实验时必须先承认的前提。

### 8.1 现在的 server 不是多会话隔离架构

虽然 server 用 `active_tasks[task_id]` 管任务，但所有 task 共用同一个 `shared_model.model`。而且每次 `/init` 都会执行 `shared_model.change_task(task_id)`，直接 reset 全局模型。

如果两个 client 真正并发交错：

- client A 刚 init 完前缀
- client B 又调 `/init`
- 共享 target model 被 reset

那么 A 之前建立的 target-side KV/cache 上下文就不再可靠了。

### 8.2 单 worker 不能自动保证任务安全

`uvicorn.run(..., workers=1)` 只表示单个 worker 进程，并不等于“多请求语义上自动串成一条任务流”。FastAPI 单 worker 下仍然可能在请求级别交错执行，而且即使完全串行，只要任务 A/B 轮流到达，也会因为共用同一模型状态而互相覆盖。

### 8.3 当前实现更接近“单活动任务 server”

从模型状态角度看，当前服务最稳妥的理解不是“一台 server 可服务多个 client”，而是：

- 一台 server
- 一个共享 target model 会话
- 通过 `task_id` 勉强记录任务元数据

因此，如果直接拿它跑 4-client 并发，测出来的很可能不是“算法在多客户端负载下的性能”，而是“共享模型状态互相踩踏后的混合结果”。

---

## 9. 日志与可观测性

日志文件在远端：

- `logs/communication_service.log`

从现有日志可以看到：

- 每次 verify 的 `task_id`
- speculative token 文本
- `target_token_probs`
- `draft_token_probs`
- `p_ratios`
- 每个位置的随机数和 accept/reject
- `model.sample` 的 top-k 概率
- 每次 verify 的 GPU 功耗积分
- 任务结束时的总功耗积分

这说明如果后面做并发实验，server 端已经有足够日志用于排查：

- task 是否交叉污染
- `n_tokens` 是否异常跳变
- 某个 task 的 verify 是否在另一个 task init 之后失真

---

## 10. 当前代码对 rebuttal 的直接含义

如果 rebuttal 指出“没有考虑一台 server 对多个 client”，从代码事实出发，应该把问题拆成两层：

### 10.1 论文/实验层

目前 edge 端实验主要证明的是：

- 单 client 条件下
- PipeSD 相比若干基线
- 能否通过通信和生成重叠降低 token latency

它还没有自然扩展成“多客户端共享 server 的系统实验”。

### 10.2 实现层

目前 server 端并没有准备好支撑真正的多客户端隔离实验，因为：

- 共享一个全局 target model
- `/init` 会 reset 共享模型
- task 级状态和模型级状态没有解耦

所以如果下一步要做 4-client 并发实验，首先要明确你想做的是哪一种：

- “现有 server 原样下的 contention experiment”
- “修正为 task-isolated server 后的真实多客户端实验”

这两种实验回答的问题不一样。

---

## 11. 建议的后续阅读顺序

如果后面要继续推进多客户端实验，建议按这个顺序回看代码：

1. 先看本地 [`src/engine.py`](/mnt/c/files/PipeSD/edge/src/engine.py)
2. 再看本地 [`src/comm.py`](/mnt/c/files/PipeSD/edge/src/comm.py)
3. 再看远端 `src/speculative_server.py`
4. 最后再回到本地 [`app/run_edge.py`](/mnt/c/files/PipeSD/edge/app/run_edge.py)

原因很简单：

- `engine.py` 决定 edge 端什么时候发、什么时候验
- `comm.py` 决定上传如何异步化
- `speculative_server.py` 决定这些请求在 target model 上到底怎么落地
- `run_edge.py` 只是外层评测驱动

---

## 12. 一句话结论

当前系统是“单 client speculative decoding 实验框架 + 单共享 target model 验证服务”，不是天然支持多客户端隔离的 server/client 系统；如果直接做 4-client 并发实验，首先要处理或至少明确共享 target model 带来的任务互相覆盖问题。
