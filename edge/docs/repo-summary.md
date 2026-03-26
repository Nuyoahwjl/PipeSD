# Edge 仓库总结

本文基于当前仓库状态整理，目的是让第一次接手项目的人能尽快回答三个问题：

1. 这个仓库在做什么
2. 代码实际是怎么跑起来的
3. 哪些地方是“设计如此”，哪些地方是“当前实现细节”

如果要看 2026-03-26 当前“单 server 对多 client”实验链路的真实实现状态，请优先结合：

- `docs/multiclient-implementation-summary.md`
- `docs/cloud-server-summary.md`

## 1. 项目定位

这是一个边缘端实验仓库，用来运行 speculative decoding 相关实验。它本身不是完整的云服务，而是客户端/实验驱动端：

- 在本地加载一个较小的 draft model（GGUF，经 `llama_cpp` 推理）
- 按 token 逐步做 speculative generation
- 把候选 token 和对应概率发送到云端服务
- 让云端 target model 决定接受多少 token，并返回最终 token
- 记录延迟、验证统计、能耗积分等实验结果

从代码和脚本来看，这个仓库主要服务于以下几类实验对比：

- `vanilla`
- `hsl`
- `edgeLLM`
- `pipesd`

其中 `pipesd` 是当前最像“主角算法”的实现，`vanilla`/`hsl`/`edgeLLM` 更像对照组或变体。

## 2. 仓库核心结构

顶层目录大致可以按职责理解：

- `app/`
  - 实验入口。`app/run_edge.py` 是整个仓库最重要的启动脚本。
- `src/`
  - 核心逻辑实现。
  - `engine.py`：speculative decoding 主循环、状态管理、与云端交互。
  - `comm.py`：带宽受限异步发送器。
  - `util.py`：命令行参数、数据集默认值、算法到策略的映射。
  - `merge.py`：token 批量发送计划的动态规划实现。
  - `bayes.py`：一个历史遗留/独立实验脚本风格的文件，当前主流程未使用。
- `scripts/`
  - 批量 sweep、变带宽实验、ablation 脚本，以及 RTT 测量脚本。
- `tests/`
  - 当前新增测试主要覆盖通信器、RTT 脚本、运行器初始化和部分状态逻辑。
- `data/`
  - 数据集 JSONL 文件。当前包含 `humaneval.jsonl` 和 `gsm8k.jsonl`。
- `pre_models/`
  - draft model 的本地 GGUF 文件。
- `exp/`
  - 运行时产物目录，保存实验结果 JSON。
- `figs/`
  - 离线整理出的图。

## 3. 一句话理解运行链路

执行：

```bash
python3 app/run_edge.py --dataset humaneval --algorithm pipesd
```

背后的真实链路是：

1. `src.util.parse_arguments()` 解析参数，并基于 `dataset`/`algorithm` 自动补一些默认值。
2. `CloudEdgeSpeculativeEval` 初始化，继承自 `src.engine.Decoding`。
3. `run_edge.py` 从 `data/*.jsonl` 读取样本。
4. 每个样本进入 `Decoding.edge_process_draft_model()`。
5. draft model 在本地逐 token 生成候选。
6. `BandwidthSender` 按带宽和固定时延约束把候选发给云端。
7. 云端返回接受的 token 数 `n_accepted` 和最终 token `final_token`。
8. 本地根据返回值更新草稿模型状态，继续下一轮 speculative generation。
9. 每条样本结束后，把统计信息写入 `exp/...json`。

这个仓库真正的“算法核心”就在第 4 到第 8 步。

## 4. 入口文件 `app/run_edge.py`

`CloudEdgeSpeculativeEval` 是对 `Decoding` 的一个具体任务封装，职责主要有四类：

- 数据加载
- 按数据集格式做预处理
- 可选的贝叶斯阈值搜索
- 驱动逐样本评测

### 4.1 数据加载逻辑

当前支持的主要数据集格式：

- `humaneval`
  - 读取 `prompt`
  - `task_id` 优先用原始字段，否则回退到 `question_id` 或索引
- `gsm8k`
  - 读取 `question`
  - 转成统一字段 `prompt`
  - `task_id` 直接使用样本顺序索引

另外：

- `--start_index_of_sample` / `--end_index_of_sample` 用于切片数据集
- `--max_samples` 用于快速调试

### 4.2 后处理逻辑

`postprocess()` 更偏向 HumanEval 风格：

- 去掉 BOS/EOS 痕迹
- 遇到 `\nclass`、`\ndef`、`\n#`、`\n@`、`\nprint`、代码块符号等停止
- 最终把生成内容拼回原 prompt

这说明仓库最初的评测重心应该是代码生成任务，后续才扩展到了 `gsm8k`。

### 4.3 贝叶斯优化

如果传入 `--bayes_optimize`：

- 程序不会走正常评测流程
- 会在 hybrid 策略下循环试验阈值组合
- 目标函数是平均 token latency
- 日志写到 `exp/.../bayes_trials.json`

一个容易误判的点：

- README 把 `src/bayes.py` 说成 Bayesian helper
- 但当前真正被调用的贝叶斯优化逻辑实际上在 `app/run_edge.py`

## 5. 核心抽象 `src/engine.py`

`Decoding` 是整个项目最重要的类。它既保存实验配置，也承载具体的 speculative decoding 状态机。

### 5.1 初始化阶段

构造时做了几件事：

- 读取 CLI 参数并保存常用字段
- 设置随机种子
- 设置实验目录 `exp/exp__gsm/<dataset>/<algorithm>`
- 根据算法决定是否允许“边生成边发送”

其中：

- `vanilla` 和 `hsl`：`send_while_generating = False`
- 其他算法：`send_while_generating = True`

这基本决定了是否尝试通信与生成重叠。

### 5.2 `_reset_state()`

每个样本开始前都会重置一次状态，主要包括：

- 验证长度统计
- 接受长度统计
- 推测 token 计数
- token 级延迟统计
- 动态阈值状态
- 加载 `llama_cpp.Llama` draft model（首次）
- 构造新的 `BandwidthSender`

这里有两个很关键的现实细节：

1. 模型加载发生在 `_reset_state()` 内，但因为 `self.draft_model` 会缓存，所以只在第一次真正加载。
2. 每个样本都会新建一个 `BandwidthSender`，样本结束时关闭。

### 5.3 验证触发策略 `if_verify()`

这个方法决定什么时候向云端请求验证，支持：

- `fixed-num`
  - speculative token 数达到 `verify_num` 就验证
- `single-token`
  - 最近一个 token 最大概率低于 `verify_thresh_single`
- `multiple-tokens`
  - 所有 speculative token 最大概率乘积低于 `alpha`
- `hybrid`
  - 同时检查 single 和 multi 两种条件，任一触发即验证

算法和策略的关系由 `src/util.py` 做默认映射：

- `vanilla` -> `fixed-num`
- `hsl` -> `single-token`
- `edgeLLM` -> `multiple-tokens`
- `pipesd` -> 默认 `hybrid`，但 ablation/no-merge 时可改

### 5.4 主状态机 `edge_process_draft_model()`

这是仓库最难读、也最关键的函数。可以把它拆成下面几个阶段。

#### 阶段 A：初始化云端上下文

- 把 prefix tokenize
- 向云端 `/init` 发送初始 token 序列
- 本地 draft model 也先对 prefix 做一次 `eval`

此时本地和云端应该都进入同一个上下文。

#### 阶段 B：逐 token speculative generation

循环内每次：

- 本地 sample 一个 token
- 记录该 token 的概率分布
- 调用 `draft_model.eval([next_token])`
- 视情况 sleep 到 `default_token_compute`

所以这里不是单纯“越快越好”的真实推理，而是显式模拟了每 token 计算耗时。

#### 阶段 C：判断“发送”还是“验证”

代码维护了两组缓存：

- `total_speculative_*`
  - 当前整个 speculative 序列，供验证使用
- `current_batch_*`
  - 当前准备发送的这一批 token

每轮都会检查：

- 是否应该发送一批 token
- 是否应该触发验证
- 是否遇到 EOS

如果验证和发送同时满足，优先验证。

#### 阶段 D：验证期间继续生成

如果算法允许 `send_while_generating`，那么在等待验证返回时，本地不会完全停住，而是继续：

- speculative 生成额外 token
- 以 `propose_waiting` 的形式异步发到云端
- 视情况在等待序列内部再触发一次验证

这部分是整个项目最复杂的并发逻辑，也是 `pipesd`/`edgeLLM` 类算法的关键实验点。

#### 阶段 E：处理验证结果

云端返回两个核心字段：

- `n_accepted`
- `final_token`

本地随后：

- 把被接受的 speculative token 加进输出
- 再追加 `final_token`
- 更新 `draft_model.n_tokens`
- 必要时重新 `eval(final_token)`
- 更新接受率、验证长度分布、token latency 等统计

### 5.5 结果落盘

单个样本结束后会写入 JSON，内容包括：

- `task_id`
- `output_length`
- `total_time`
- `output`
- `gamma`
- `strategy`
- `bandwidth_MBps`
- 单阈值/多阈值
- `verify_stats`
- `token_durations`
- `avg_token_time`
- `gpu_power_integral_joules`
- `acc_ratio`
- `verify_his`

这说明仓库当前更偏实验记录工具，而不是纯在线服务客户端。

## 6. 通信模块 `src/comm.py`

`BandwidthSender` 是一个很实用的基础组件，负责把 HTTP POST 发送过程做成“近似带宽受限链路”的异步队列。

核心机制：

- 主线程 `submit()` 只负责把请求放进队列并返回 `Future`
- 后台工作线程 `_worker()` 串行处理发送
- 根据 `payload_size / bandwidth + base_latency` 计算理论耗时
- 如果真实发送比理论耗时快，就 `sleep` 补足

这使得仓库可以在普通网络环境里近似模拟“固定 RTT + 限速带宽”的传输。

### 6.1 tag 机制

请求可以带 `tag`，从而支持：

- `drain_tag()`
  - 取消还没开始发的请求
- `cancel_and_resubmit()`
  - 清空旧请求并重新提交
- `list_tags()`
  - 查看当前未完成任务

这套机制主要服务于 `engine.py` 中“等待验证期间继续 speculative 并可能重打包”的逻辑。

### 6.2 代理行为

默认情况下：

- `requests.Session.trust_env = False`

也就是默认不吃系统代理环境变量。只有显式打开 `use_env_proxy` 才会使用。这个行为已经有测试覆盖。

## 7. 参数与模式映射 `src/util.py`

这个文件决定了仓库“跑实验时长什么样”。

### 7.1 关键参数

最常用的一组参数：

- `--dataset`
- `--algorithm`
- `--gamma`
- `--verify_strategy`
- `--verify_thresh_single`
- `--verify_thresh_multi`
- `--init_alpha`
- `--multiply_times`
- `--C`
- `--bandwidth_MBps`
- `--default_token_compute`
- `--token_size_MB`
- `--start_index_of_sample`
- `--end_index_of_sample`
- `--max_samples`

### 7.2 数据集默认模型

当前默认映射是：

- `humaneval`
  - draft model: `deepseek-coder-1.3b-instruct-GGUF`
- `gsm8k`
  - draft model: `tinyllama-1.1b-chat-v1.0-gguf`
- `mt_bench`
  - 也走 `tinyllama`

target model 名称会在参数处理中设置，但本仓库并不直接加载 target model，它只与远端服务通信。

### 7.3 算法到策略的自动映射

这是理解脚本输出时必须知道的默认行为：

- 如果 `algorithm == edgeLLM`，会强制 `verify_strategy = multiple-tokens`
- 如果 `algorithm == hsl`，会强制 `verify_strategy = single-token`
- 如果 `algorithm == pipesd`，在非 ablation 或 `nomerge` 情况下会强制 `verify_strategy = hybrid`

也就是说，命令行上显式传的 `verify_strategy` 有时会被二次覆盖。

## 8. 批量发送规划 `src/merge.py`

`dynamic_token_scheduling_dp()` 用动态规划求 token 应如何分批发送，以最小化总完成时间。

输入是：

- 每个 token 的计算时间
- 固定启动开销 `C`
- 单 token 传输时间 `d`

输出是：

- 分批方案 `batches`
- 最优完成时间

这个模块本身是清晰独立的，也带了一个可直接运行的 `__main__` 做对比实验。

但是有一个非常重要的“当前实现事实”：

- `engine.py` 里虽然调用了 `dynamic_token_scheduling_dp()`
- 但紧接着把结果覆盖成了 `merge_plan_batches = [100]`

这意味着当前主流程实际上没有真正使用 DP 规划结果，而是近似固定成“很大批次后再发”。如果后续有人要研究 merge 效果，这一行是首先要确认的地方。

## 9. 脚本体系 `scripts/`

这些脚本的价值在于把实验组合固化下来，避免手工拼参数。

### 9.1 `sweep.sh`

面向固定带宽的一组对照实验，典型流程是：

- 先跑 `vanilla`
- 再跑一组 `hsl`
- 再跑 `pipesd`
- 最后跑 `edgeLLM`

通过 `START_INDEX`、`END_INDEX`、`BATCH_SIZE` 控制样本切片。

### 9.2 `vary_bandwidth.sh`

支持两种模式：

- `BW_LIST` 按样本索引逐条指定带宽
- `BANDWIDTHS_MBPS` 对一个样本区间统一扫带宽

适合做“动态网络条件”实验。

### 9.3 `swee_gsm8k.sh` / `vary_bandwidth_gsm8k.sh`

这些脚本和 Humaneval 版本结构基本一致，但阈值默认值更偏向 GSM8K 场景。

一个小细节：

- 文件名 `swee_gsm8k.sh` 少了一个 `p`
- 看起来是命名遗留，不影响运行，但容易让后来者搜索不到

### 9.4 `ablation_study.sh`

针对 `pipesd` 做消融：

- multi-token
- single-token
- fixed-num
- `--nomerge`

也就是把 `pipesd` 拆解为若干策略变体比较。

### 9.5 `measure_rtt.py`

这是一个独立小工具，用来测量到某个 HTTP endpoint 的 RTT：

- 支持 warmup
- 汇总 min/avg/p50/p95/max
- 可输出文本或 JSON

默认探测地址是：

```text
http://115.190.90.101:1597/delay
```

这和主实验默认云服务地址是同一台机器。

## 10. 数据与结果格式

### 10.1 输入数据

当前仓库已包含：

- `data/humaneval.jsonl`：164 条
- `data/gsm8k.jsonl`：1319 条

格式分别是：

- Humaneval：`task_id`、`prompt`、`canonical_solution`、`test` 等字段
- GSM8K：`question`、`answer`

### 10.2 结果数据

实验结果写在 `exp/exp__gsm/<dataset>/<algorithm>/...json`。

虽然目录名里写的是 `exp__gsm`，但当前实际也拿它存 `humaneval`，所以这里更像历史命名残留，而不是严格表示数据集类型。

结果文件名会编码当前配置，例如：

- `gamma_6_bw=2.5MB.json`
- `st=0.99_bw=2.5MB.json`
- `edgeLLM_alpha=0.92_mult=0.95_bw=2.5MB.json`
- `st=0.9_mt=0.95_bw=2.5MB.json`

## 11. 云端接口约定

当前边缘端默认连接：

```text
http://115.190.90.101:1597
```

使用的接口：

- `/init`
- `/propose`
- `/exit`

大致协议如下：

- `/init`
  - 发送 prefix token 和 `task_id`
- `/propose`
  - 发送 speculative token、每 token 概率、`n_past`、序列偏移、是否验证
- `/exit`
  - 结束当前任务，并取回统计信息，例如能耗积分

从边缘端代码推断，`/propose` 返回至少需要包含：

- `n_accepted`
- `final_token`

如果缺这两个字段，边缘端会直接认为响应异常。

## 12. 依赖与运行环境

### 12.1 Python 侧依赖

`install.sh` 展示的主要依赖包括：

- `transformers`
- `accelerate`
- `numpy`
- `fastapi`
- `uvicorn`
- `pydantic`
- `msgpack`
- `pynvml`

除此之外，按源码实际还需要：

- `requests`
- `scikit-optimize`
- `pandas`
- `llama-cpp-python`

如果没有 `llama_cpp`，`engine.py` 会打印警告并关闭 GGUF 支持。

### 12.2 模型依赖

当前 `pre_models/` 下能看到两份 draft model：

- `deepseek-coder-1.3b-instruct.Q4_K_M.gguf`
- `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`

这与 `src/util.py` 的默认数据集映射一致。

### 12.3 当前环境观察

在本次整理时：

- `python` 命令不存在
- `python3` 可用
- `python3 -m pytest tests -q` 通过，共 9 个测试

所以如果有人直接复制 README 里的 `python app/run_edge.py ...`，在某些机器上可能需要替换成 `python3`。

## 13. 测试现状

测试数量不多，但覆盖了最近比较容易出问题的几块。

### 13.1 `tests/test_comm.py`

验证 `BandwidthSender` 的代理默认值：

- 默认禁用环境代理
- 显式打开后才启用

### 13.2 `tests/test_run_edge.py`

覆盖点包括：

- `load_data()` 是否尊重 `max_samples`
- `_reset_state()` 是否初始化追踪字段
- `use_env_proxy` 是否向 `BandwidthSender` 透传
- token duration 记录逻辑是否正确

这里大量使用 stub module，避免真实依赖 `torch`、`llama_cpp`、`skopt` 等重库。

### 13.3 `tests/test_measure_rtt.py`

覆盖点包括：

- RTT 统计汇总
- 请求成功/失败计数
- 默认禁用环境代理

### 13.4 测试边界

目前没有覆盖的高风险区域主要是：

- `engine.py` 主循环中的等待期并发分支
- 与真实云端服务的协议兼容性
- merge 计划与实际发送行为的一致性
- 结果 JSON 的长期兼容性

也就是说，当前测试更偏“单元级防回归”，不是“端到端保证”。

## 14. 现在最值得记住的几个事实

如果只想快速接手项目，优先记住下面这些：

1. 这是边缘端实验驱动仓库，不是云端推理服务本体。
2. 真正入口是 `app/run_edge.py`，真正核心是 `src/engine.py`。
3. 本地只跑 draft model，target model 在远端。
4. `BandwidthSender` 用队列和 sleep 模拟受限带宽链路。
5. `pipesd` 的复杂度主要来自“验证期间继续生成并异步发送”。
6. README 提到的 `src/bayes.py` 并不是当前主流程里的贝叶斯优化入口。
7. `merge.py` 的 DP 结果在主流程里目前被固定值覆盖，实际 merge 规划并未真正启用。
8. 结果文件是追加写 JSON 列表，而不是每次单独一个样本文件。
9. 仓库当前测试能过，但没有端到端覆盖远端协议。

## 15. 后续维护建议

如果后续要继续把这个仓库作为研究或复现实验的基线，建议优先做下面几件事：

- 把云端 URL 从 `engine.py` 硬编码改成 CLI 参数或环境变量。
- 明确区分“当前启用逻辑”和“历史实验代码”，尤其是 `src/bayes.py`。
- 决定 `merge_plan_batches = [100]` 是临时调试还是正式策略，并写进 README。
- 补一个最小端到端协议文档，说明 `/init`、`/propose`、`/exit` 的请求/响应字段。
- 给 `exp/` 结果格式补一个 schema 说明，方便后处理脚本稳定解析。
- 如果需要长期维护，建议把 Humaneval/GSM8K 的后处理和评测逻辑彻底拆开，避免一个类里混合多任务分支。

## 16. 快速上手建议

第一次接手这个仓库，推荐按这个顺序读：

1. `README.md`
2. `app/run_edge.py`
3. `src/util.py`
4. `src/engine.py`
5. `src/comm.py`
6. `scripts/sweep.sh`
7. `tests/test_run_edge.py`

这样能先建立“实验是怎么被启动的”，再去理解“每个 token 在循环里经历了什么”。
