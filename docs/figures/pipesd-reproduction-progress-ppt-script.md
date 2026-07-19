# PipeSD 复现进度汇报：PPT 逐页内容与讲稿

> 适用场景：基于当前仓库、论文和现有测评结果制作一次约 18～22 分钟的阶段汇报。  
> 主体共 18 页，附录 4 页。每页都包含“页面内容”“建议图示”和“讲稿”。  
> 论文：[PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](https://arxiv.org/abs/2605.13319)

## 使用前必须统一的表述

- 本次是**协议与机制复现**，不是论文全部结果的完整复刻。当前已重点复现论文 Scenario 1 的四种算法协议，并增加了纯边、纯云、边云串行、PipeSD 四种部署模式的扩展对比。
- 当前实验绝对时延明显高于论文，不能说“复现了论文数值”；可以说“复现了 PipeSD 的核心机制，并在当前软硬件条件下观察到相对收益”。
- Git 中查询到的作者名是 `Nuyoahwjl`，不是题目中拼写的 `Nuyoshwjl`。下文按仓库实际作者名统计，共 8 个相关提交。
- “纯云”结果覆盖 warm target 请求从预处理、prompt prefill、自回归 decode 到 detokenization，但不含模型加载和客户端—云端传输；“纯边”使用 draft 小模型，输出质量与 target 大模型不等价。纯模式按 committed output token 归一化，协同模式按 cloud-accepted draft token 归一化，因此四模式结果用于拆解系统开销，不能解释为四种同质量、同口径服务的直接竞赛。
- 当前能耗是**云端 GPU 能耗**，纯边没有 RAPL/外接功率计数据，不能据此给出端到端系统总能耗结论。

---

## 第 1 页：封面

### 页面内容

**标题：** PipeSD 云边协同推测解码复现进度  
**副标题：** 从论文机制、代码实现到四算法与四部署模式测评  
**页脚：** 汇报人 / 日期 / 项目仓库

### 建议图示

使用论文工作流图作为淡化背景，或者放置本仓库的多轮流水线图缩略图：

![PipeSD 多轮流水线](./pipesd-humaneval-trace-multi-round.png)

### 讲稿

今天汇报的是 PipeSD 的阶段性复现。我会先解释它为什么需要云边流水线，以及 DP 批调度、双阈值 NAV 和 BO 自动调参分别解决什么问题；然后结合当前代码说明我已经对齐了哪些机制；最后展示两组实验：第一组比较 Vanilla、HSL、EdgeLLM 和 PipeSD，第二组拆解纯边、纯云、边云串行和 PipeSD，并说明目前距离论文完整复现还差哪些工作。

---

## 第 2 页：汇报结论先行

### 页面内容

**已完成**

- PipeSD 核心闭环：边缘 draft、边生成边上传、云端 NAV、返回 accepted tokens、下一轮衔接。
- DP token-batch 调度、双阈值触发、16 次 BO、在线环境估计和共享 FIFO 软件链路。
- Scenario 1 风格的四算法 1000-token 测评；两任务均观察到 PipeSD 相对收益。
- 纯边 / 纯云 / 边云串行 / PipeSD 四模式拆解，以及可追踪的多轮流水线图。

**关键结果**

- HumanEval：PipeSD 相对 Vanilla 加速 **1.249×**，云 GPU 能耗降低 **36.4%**。
- GSM8K：PipeSD 相对 Vanilla 加速 **1.218×**，相对最佳基线 HSL 加速 **1.088×**，云 GPU 能耗降低 **23.6%**。

**尚未完成**

- 论文 Scenario 2～4 的完整正式结果、带宽扫描、消融、BO 对照、正确率和多次重复置信区间。
- 论文原硬件、真实公网链路与论文绝对数值的复刻。

### 建议图示

三栏：已完成 / 关键结果 / 未完成。关键数字用大号字体。

### 讲稿

先给出结论：核心系统机制已经跑通，而且 PipeSD 在两个任务上都优于串行 Vanilla。HumanEval 加速 1.249 倍，GSM8K 加速 1.218 倍；两项任务的云 GPU active-compute 能耗分别降低 36.4% 和 23.6%。这里的完成主要指协议、并发和测评链路已经形成，不代表论文所有表格都复刻出来。当前最重要的缺口是测试床不一致、正式实验覆盖不足，以及只有单次 1000-token 结果，没有置信区间和任务正确率。

---

## 第 3 页：问题背景——为什么传统云边推测解码还不够快

### 页面内容

传统云边推测解码一轮通常是：

1. 边缘小模型串行生成一段 draft tokens；
2. 生成结束后才整体上传；
3. 云端大模型执行 Non-Autoregressive Verification，简称 NAV；
4. 边缘等待验证结果，再开始下一轮。

两个主要瓶颈：

- **生成与通信串行**：上传阶段边缘算力空闲，生成阶段链路空闲。
- **固定长度或单阈值触发僵化**：过早 NAV 会产生高固定通信开销；过晚 NAV 会累积低质量 token，导致回滚。

### 建议图示

画两条时间轴：传统流程的 Generate → Upload → NAV → Wait，与 PipeSD 的重叠时间块对比。

### 讲稿

PipeSD 关注的不是简单地把小模型放边缘、大模型放云端，而是如何让这两个异构阶段持续工作。传统流程里，draft 生成、网络传输和云端验证基本串行，导致设备和网络交替空闲。触发 NAV 也存在两难：发得太频繁，启动开销很大；积攒太久，一旦后面的 token 置信度下降，就会回滚更多无效计算。论文的两个核心机制正好对应这两个问题：DP 批调度负责重叠生成与传输，双阈值加 BO 负责决定何时验证。

---

## 第 4 页：推测解码与 NAV 到底做了什么

### 页面内容

设边缘 draft 模型生成候选序列：

\[
D=(d_1,d_2,\ldots,d_n)
\]

云端 target 模型一次前向并行检查这段候选：

- 从前向后接受与 target 采样规则一致的最长前缀；
- 在 accepted prefix 后，target 还会给出一个额外 token；
- 若发生拒绝，拒绝点之后的 draft token 全部失效；
- 下一轮上下文必须以云端确认后的序列为准。

**关键边界：** target 额外 token 与边缘提前生成的下一轮首 token 相同，才可将该提前结果晋升为下一轮正式候选；否则必须丢弃并重新开始。

### 建议图示

示例：draft 为 A B C D，云端接受 A B，拒绝 C，并返回 X；结果上下文为 A B X，C D 及其派生 token 作废。

### 讲稿

NAV 的核心是用 target 大模型一次性验证一批 draft token，而不是逐 token 调用大模型。云端总会在接受前缀后提供一个 target token，用它保证序列能继续前进。需要强调的是，PipeSD 为了隐藏等待时间，会在 NAV 返回前继续生成甚至提前上传下一轮候选。但这并不代表这些 token 已经确认。如果云端只接受部分 token，或者即使全部接受但额外 token 与边缘预测的衔接 token 不一致，那么提前生成的数据都不能直接进入正式上下文，必须取消或丢弃。当前实现专门维护父轮次和父 token 元数据来保证这一点。

---

## 第 5 页：PipeSD 总体架构

### 页面内容

**Edge**

- Draft Model：自回归产生候选 token 与置信度。
- Transmission Controller：DP 批调度器 + 双阈值 NAV 触发器。
- Environment Monitor：估计生成时间 \(\gamma\)、通信启动开销 \(\alpha\) 和单位 token 通信时间 \(\beta\)。
- Parameter Updater：BO 更新双阈值，DP 根据环境参数重算批边界。
- Communication Interface：上传 draft batch，接收 NAV 结果。

**Cloud**

- Communication API：接收带轮次元数据的候选批次。
- Target Model：执行 NAV，返回 accepted prefix 和额外 target token。

### 建议图示

画两个大框 Edge / Cloud，中间画双向箭头。Edge 内将 Controller 标成核心模块。

### 讲稿

架构上，PipeSD 把决策主要放在边缘 Transmission Controller。它一方面根据当前生成和通信速度，把一个预测窗口切成若干批；另一方面根据 token 置信度和累计序列置信度决定何时触发 NAV。环境监控模块持续更新参数，避免调度只适用于固定网络。云端逻辑相对集中：接收批次、按轮次组装候选、用 target 模型验证并返回确认结果。

---

## 第 6 页：重点——一轮 PipeSD 流水线如何运行

### 页面内容

一轮的时间顺序：

1. **Draft generation**：边缘逐 token 生成并计算置信度。
2. **Token-batch scheduling**：达到当前 DP 批边界后，异步上传已完成的 batch；边缘继续生成下一批。
3. **Dual-threshold trigger**：每个 token 后检查单 token 与累计置信度，任一越界就触发 NAV。
4. **NAV**：云端验证已上传的本轮候选并返回 accepted tokens + 额外 target token。
5. **Speculative continuation**：等待 NAV 时，边缘建立隔离的 proactive buffer，尝试生成并上传下一轮候选。
6. **Promote or discard**：若父轮全部接受且额外 token 与预期相同，将 proactive 数据晋升为下一轮正式候选；否则取消发送并丢弃。

### 建议图示

本页建议直接使用：

![HumanEval Task 50 前 12 个触发窗口](./pipesd-humaneval-trace-multi-round.png)

图中重点标注：蓝色 draft、橙色 upload batch、红色 NAV、绿色 accepted、灰色 discarded/proactive。

### 讲稿

这一页是整个汇报的重点。PipeSD 不等一整段 draft 生成完再发，而是把 token 按 DP 选出的边界分批上传，所以生成和通信可以重叠。触发 NAV 以后，边缘也不立即停机，而是在隔离缓冲区里推测下一轮。云端结果返回后再决定是否提交这部分工作。这样做的收益来自隐藏网络与 NAV 等待，但正确性约束也更复杂：提前工作只能作为 speculation，不能提前写入正式序列。图里的每个窗口都来自正式结果文件，不是示意数据；详细字段映射见附录引用。

---

## 第 7 页：DP token-batch pipeline scheduling

### 页面内容

论文用线性模型描述上传一个 batch 的时间：

$$
T_{comm}(b)=\alpha+\beta b
$$
生成 \(b\) 个 token 的时间近似为：

$$
T_{gen}(b)=\gamma b
$$
调度目标：在预测窗口 \(\hat N\) 内选择批边界，使最后一批上传完成时间最小。

- 新 batch 只有在“该批 token 已生成”且“上一批通信已完成”后才能发送。
- 动态规划枚举前一个切分点，复杂度 \(O(\hat N^2)\)。
- \(\hat N\) 初始为 20，之后使用最近 100 轮 draft 长度移动平均更新。
- NAV 提前返回时中断当前计划，未发送 token 被清理。

### 建议图示

展示窗口 `[1…N]` 被切为 `[1,2] [3,4,5] [6…]`，下方以甘特图显示 generate 与 upload 重叠。

### 讲稿

为什么不是每生成一个 token 就立即上传？因为每次网络请求都有固定启动开销 alpha；为什么也不是全部生成完一次发送？因为那样无法与生成重叠。因此批大小是一个折中。DP 利用当前测得的 alpha、beta 和 gamma，寻找预测窗口里的最优切分。当前代码还实现了环境估计和 20% 变化门限，避免轻微抖动导致频繁重算。正式结果中 PipeSD 的平均实际 batch 大约是 HumanEval 1.670、GSM8K 1.371，说明调度确实采用了细粒度流水批次，而不是整段上传。

### 实现对应

- [`edge/src/merge.py`](../edge/src/merge.py)：`PaperDPScheduler`、窗口更新和在线环境估计。
- [`edge/src/engine.py`](../edge/src/engine.py)：按批边界异步发送、NAV 中断和 trace 记录。

---

## 第 8 页：双阈值 NAV 与 BO 自动调参

### 页面内容

对第 \(n\) 个 draft token：

- 单 token 置信度：\(P(d_n)\)
- 累计序列置信度：\(C_n=\prod_{i=1}^{n}P(d_i)\)

触发规则：

\[
C_n\le R_1 \quad \text{或}\quad P(d_n)\le R_2
\]

- 单 token 阈值捕捉局部突降。
- 累计阈值捕捉多个“看似还可以”的 token 连乘后整体风险升高。
- BO 以平均 TPT 为目标，在当前设备与网络下搜索 \((R_1,R_2)\)。
- 当前实现按论文配置进行 16 次采样，使用 EI acquisition，`xi=0.1`。

### 建议图示

两条曲线：token confidence 与 cumulative confidence，画两条阈值线，并标出触发点。

### 讲稿

只看单 token 会漏掉一种情况：连续多个 token 每个置信度都不算特别低，但整段都正确的概率已经很小。只看累计值又可能对单点置信度骤降反应不够直观，所以论文采用 OR 关系的双阈值。最优阈值依赖模型、任务、网络和设备，不能只用一个人工常数，因此用轻量 BO 在线寻找。这里要区分“功能已实现”和“论文对照已复现”：BO 代码和独立 trial 已经完成，但最终正式结果中记录的阈值与最新 BO 输出没有完整绑定，BO、Grid、Random 的 Table 3 对照也还没做完，所以 BO 只能标为机制已实现、实验闭环待补。

### 实现对应

- [`edge/src/engine.py`](../edge/src/engine.py)：双阈值判定。
- [`edge/app/run_edge.py`](../edge/app/run_edge.py)：BO 目标、16 次预算与实验隔离。

---

## 第 9 页：并发流水线中的正确性与数据生命周期

### 页面内容

| 云端返回情况 | proactive token 能否晋升 | 处理方式 |
|---|---:|---|
| 只接受部分本轮 draft | 否 | 回滚拒绝点以后 token；取消/丢弃提前发送的下一轮数据 |
| 接受全部 draft，但额外 target token 不匹配 | 否 | proactive 分支上下文错误，全部丢弃并从 target token 重启 |
| 接受全部 draft，且额外 token 匹配 | 是 | 将隔离 buffer 提升为下一轮正式候选 |

当前实现的保护措施：

- 每批携带 round、parent round、parent final token 等元数据。
- proactive 与正式 batch 使用不同逻辑缓冲区。
- 两者共享同一物理 `SoftwareLink`，避免虚构额外带宽。
- 云端只在父轮验证通过后 promote，否则 discard。

### 建议图示

画三条分支：partial accept / full accept + mismatch / full accept + match，只有第三条进入绿色 commit。

### 讲稿

流水线复现最容易“看起来更快但语义不正确”的地方，就是把提前生成的 token 无条件晋升。正确条件其实很严格：父轮必须全部接受，而且 target 给出的额外 token 必须等于 proactive 分支假设的第一个上下文 token。当前版本在边缘和云端都做了元数据校验，失败时会清理缓冲区。因此提前上传是允许的，但它只是待确认数据，不能绕过 NAV。这个改动也是当前复现相对早期版本最关键的正确性补齐之一。

### 实现对应

- [`edge/src/engine.py`](../edge/src/engine.py)：proactive sender、晋升与取消。
- [`cloud/src/speculative_server.py`](../cloud/src/speculative_server.py)：父轮校验、buffer promote/discard。
- [`edge/src/software_link.py`](../edge/src/software_link.py)：双向共享 FIFO 网络仿真。

---

## 第 10 页：当前项目中的端到端实现映射

### 页面内容

| 论文组件 | 当前实现 | 复现状态 |
|---|---|---|
| Edge draft autoregressive generation | `edge/src/engine.py` | 已实现 |
| Token-batch DP scheduler | `edge/src/merge.py` | 已按论文逻辑对齐 |
| Dual-threshold NAV trigger | `edge/src/engine.py` | 已实现 |
| BO parameter updater | `edge/app/run_edge.py` | 已实现，正式结果溯源待补 |
| Environment monitor | `merge.py` + trace/result 字段 | 已实现，论文 Figure 6 对照未完成 |
| Cloud NAV | `cloud/src/speculative_server.py` | 已实现 |
| Dynamic software network | `edge/src/software_link.py` | 已实现 |
| Four-mode evaluation | `edge/src/pure_baseline.py` + scripts | 仓库扩展，已测评 |

### 建议图示

左边放论文架构模块，右边放代码文件，中间连线。

### 讲稿

这页把论文术语落到代码上。核心控制路径主要在 engine、merge 和 speculative server 三个文件中；run_edge 负责实验预算、BO 和结果清单；software_link 负责把上传和下载限制在共享的物理链路模型上。四模式比较不是论文 Table 1 的组成部分，而是为了回答“收益究竟来自模型、链路还是并发”而增加的系统拆解实验。

---

## 第 11 页：我完成了什么——Nuyoahwjl 提交时间线

### 页面内容

| 提交 | 主要工作 | 对复现的意义 |
|---|---|---|
| `c3a904d` | 规范 GSM sweep 和实验输出目录 | 结果组织 |
| `4c8a975` | 增加四算法实验结果 | 形成 Table 1 风格比较 |
| `4915a13` | 对齐 DP 调度与 BO | 核心方法复现 |
| `c4a8b5e` | 隔离 BO trial、bootstrap 通信回归 | 避免调参实验互相污染 |
| `5a7f3f1` | 严格 token / generation budget | 保证 1000-token 口径 |
| `9375fa1` | 对齐真实并发、proactive 元数据和评测协议 | 修复流水正确性与可比性 |
| `3902f28` | 对齐共享 FIFO 软件网络与动态带宽 | 更可信的网络仿真 |
| `f27a203` | 增加四模式测评和综合报告 | 系统开销拆解 |

### 建议图示

按日期画水平时间线，将 8 个提交归为“算法对齐—协议正确性—网络仿真—扩展测评”四段。

### 讲稿

这里的表述要准确：仓库原本已经有 PipeSD 基础框架，我的工作不是从零发明所有模块，而是围绕论文协议做对齐、补齐和验证。前几次提交先形成四算法结果，随后把 DP、BO、实验隔离和预算做得更接近论文。后面的重点是并发正确性、共享链路和结果可追溯，最后增加四模式测评。因此当前最有价值的复现成果，不只是能运行，而是把“并发是否真实、提前 token 是否安全、不同方法是否同口径”这些影响结论的问题显式化了。

---

## 第 12 页：单卡 RTX PRO 6000 上如何复现与测评

### 页面内容

**部署与模型落点：同一台机器上的逻辑 cloud-edge**

| 角色 | 本次实际设置 | 模型与推理配置 |
|---|---|---|
| Cloud / target | 单卡 **NVIDIA RTX PRO 6000**；本地 FastAPI `127.0.0.1:8000`，单 worker | HumanEval：DeepSeek-Coder-6.7B-Instruct Q4_K_M；GSM8K：Llama-2-7B-Chat Q4_K_M；`llama-cpp-python`，`n_gpu_layers=-1`，全部 target 层放入 GPU，`n_ctx=16384` |
| Edge / draft | 与 cloud 同一工作站，但 draft 明确在 **CPU** 上运行 | HumanEval：DeepSeek-Coder-1.3B-Instruct Q4_K_M；GSM8K：TinyLlama-1.1B-Chat-v1.0 Q4_K_M；`n_gpu_layers=0`、2 CPU threads、`n_ctx=16384`、`temperature=0`、`top_k=1` |

因此“单卡测评”指 target NAV 使用一张 RTX PRO 6000；draft 没有与 target 共享这张 GPU。它复现的是 cloud/edge 的进程、协议和资源分工，不是物理跨地域部署。论文原环境则是边缘端 Core Ultra 9 185H CPU、云端 A800 40 GB 和真实城域网，二者不能直接比较绝对 TPT（[论文 §5.1](https://arxiv.org/pdf/2605.13319#page=6)）。

**论文模块在代码中如何实现**

| 论文模块 | 本仓库的复现实现 |
|---|---|
| Draft Model | `edge/src/engine.py` 用 GGUF draft 自回归逐 token 生成，同时从 logits 计算每个 token 的最大概率 `P(Dn)`。 |
| 双阈值 NAV Trigger | `if_verify(..., "hybrid")` 同时计算单 token 置信度和累计序列置信度乘积；`P(Dn) < R2` **或** `∏P(Di) < R1` 即触发 NAV。 |
| Token-batch Pipeline Scheduler | `edge/src/merge.py` 按论文递推式求最小完成时间的 DP batch；初始 `N-hat=20`、`alpha=25 ms`、`beta=0.29/2.5 s/token`、`gamma=0.036 s/token`，随后用最近 100 条生成/通信记录更新。只有 PipeSD 在 NAV 前按 DP 分批上传，并在等待 NAV 时继续生成和上传。 |
| Communication / proactive continuation | 主上传和 proactive 上传各有一个 worker，但共享同一个 `SoftwareLink`；请求携带 round/window/batch/index/prefix 元数据。只有“父轮全部接受且 target extra token 与 edge 预期 token 相同”时，才将等待期 token 晋升为下一轮正式候选；否则 edge/cloud 两端都丢弃并回滚。 |
| Cloud NAV | `cloud/src/speculative_server.py` 暴露 `/init`、`/start`、`/propose`、`/exit`；target 一次前向计算验证整段 draft，按 `min(1,p/q)` 接受，首次拒绝时从正差分分布采样修正 token，全部接受时由 target 再产生一个 extra token。 |
| 环境自适应与 BO | 先用 1--8 token 探针拟合通信 `alpha/beta`，在线估计 draft `gamma`，变化超过 20% 时重算 DP。BO 已实现 `(R1,R2) in (0,1)^2`、16 次调用、1 个随机初始点、Matern GP、EI `xi=0.1`、每候选 20 个 accepted token；但本页正式结果使用脚本中的固定阈值，manifest 的 `bo_config_path` 为空，不能声称该次结果由已记录 BO 自动选出。 |

**四算法的实际参数**

| 数据集 | Vanilla | HSL | EdgeLLM | PipeSD |
|---|---|---|---|---|
| HumanEval | 固定 `N=6` | token 阈值 `R2=0.99` | 初始序列阈值 `R1=0.92`，全接受后乘 `0.5`，等待 NAV 时继续生成 | `R1(sequence)=0.90`，`R2(token)=0.3514`，DP merge |
| GSM8K | 固定 `N=4` | token 阈值 `R2=0.70` | 初始序列阈值 `R1=0.50`，全接受后乘 `0.5`，等待 NAV 时继续生成 | `R1(sequence)=0.65`，`R2(token)=0.40`，DP merge |

**网络条件如何模拟**

- 论文场景 1 的 20 Mbps 上行和 200 Mbps 下行换算为 `2.5 MB/s` 与 `25 MB/s`；本次通过应用层 `SoftwareLink` 在真正发出 loopback HTTP 请求前后整形。
- 每次传输服务时间为 `startup + serialized_bytes / bandwidth`：上行 startup 为 `25 ms`，下行为 `0 ms`。25 ms 是本次复现为通信模型设置的每 batch 启动开销 `alpha`，**不是论文报告的 RTT**。
- 上、下行是两个独立 FIFO，形成全双工；同方向内所有正式和 proactive 请求共享一条队列，因此不会各自获得一条 20 Mbps“虚拟专线”。实际 loopback HTTP 与 target 计算时间仍保留在端到端耗时中。
- 本页是固定带宽、无人工 edge 计算延迟的 Scenario-1-style 测试；`enable_compute_emulation=false`。Scenario 2/3 的逐 token 人工延迟和 Scenario 4 的 20 秒动态带宽 profile 均未用于本页正式结果。

**具体测评流程与统计口径**

1. 按数据集启动/重启 cloud，加载对应 target；edge 依次运行 Vanilla、HSL、EdgeLLM、PipeSD。HumanEval 从样本 50 开始，GSM8K 从样本 100 开始；每个样本最多生成 128 token。
2. edge 侧 manifest 记录 seed `3407`。每种方法持续取样，直到 cloud 累计恰好接受 **1000 个 draft token**；不是固定样本数。当前结果中 HumanEval 使用约 8 个样本，GSM8K 使用 8--9 个样本。
3. `comparison` 中的 TPT 为 `sum(end-to-end time) / sum(cloud-accepted draft tokens)`，吞吐、NAV/100 和 GPU J/100 也按 accepted draft token 归一化；P50/P95/P99 与 TTFT 描述已提交输出 token。
4. cloud 用 NVML 每 5 ms 采样板卡功率；能耗只积分 prompt prefill 与 target NAV active compute，排除模型加载、NAV 间 GPU idle、edge CPU、网络和整机功耗。
5. 每次运行保存原始 sample、completion、batch/NAV/proactive trace、网络队列统计、manifest 与 SHA-256，再由 `edge/scripts/summarize_table1.py` 生成 Markdown/CSV/JSON 汇总。

### 建议图示

用一张自左向右的复现链路图：`CPU draft -> 双阈值/DP -> shared FIFO software link -> FastAPI -> RTX PRO 6000 target NAV`；图下再放四算法参数表和“20/200 Mbps、alpha=25 ms、1000 accepted draft tokens”三个醒目标注。

### 讲稿

这不是把 cloud 和 edge 都塞进一张 GPU。实际运行时，edge draft 固定 `n_gpu_layers=0`，由 CPU 逐 token 生成；cloud target 固定 `n_gpu_layers=-1`，整模型放到单卡 RTX PRO 6000 做 NAV。二者是同一台机器上的两个进程，通过 `127.0.0.1:8000` 通信，所以我们再把论文场景 1 的 20/200 Mbps 放进应用层链路模拟器。模拟器在 HTTP 请求真正到达 cloud 之前等待上传时间，在响应交给 edge 之前等待下载时间；两个上传 worker 共享同一条上行 FIFO。这样保留了真实 target 计算，同时把通信带宽和 batch 启动开销变成可重复的受控变量。

一次任务先由 `/init` 在 target 上执行 prompt prefill，然后 edge 逐 token draft。Vanilla 和 HSL 在 NAV 前不提前上传，触发时一次发送整段；EdgeLLM 只在等待 NAV 时继续生成；PipeSD 在 NAV 前与等待期间都按 DP batch 发送。cloud 收到 `should_verify=true` 后做一次 NAV，返回接受长度和 target extra token。只有上一轮全接受且 extra token 对上，等待期分支才能晋升；否则分支被丢弃并从已确认前缀恢复。这一条件保证 pipeline 优化不会把未经 target 确认的 token 提交到最终输出。

建议复现实验时先为两个数据集分别启动 cloud，并显式统一 seed：

```bash
cd cloud
GPU_POWER_SAMPLE_INTERVAL=0.005 python -m src.speculative_server --dataset humaneval --seed 3407
# HumanEval 完成后停止，再启动：
GPU_POWER_SAMPLE_INTERVAL=0.005 python -m src.speculative_server --dataset gsm8k --seed 3407
```

然后在 `edge/` 中固定网络、预算和 tag，先跑 100-token pilot，检查 accepted-token 预算、batch trace、proactive 晋升/丢弃计数和共享链路总量，再跑正式 1000-token 测试并汇总：

```bash
export PIPE_SD_SERVER_URL=http://127.0.0.1:8000
export NETWORK_SHAPING_MODE=software BANDWIDTH_MBPS=2.5 DOWNLINK_BANDWIDTH_MBPS=25
export SOFTWARE_UPLINK_STARTUP_MS=25 SOFTWARE_DOWNLINK_STARTUP_MS=0
SEED=3407 TARGET_OUTPUT_TOKENS=1000 RESULT_TAG=table1_s1_paper bash scripts/eval_humaneval.sh
SEED=3407 TARGET_OUTPUT_TOKENS=1000 RESULT_TAG=table1_s1_paper bash scripts/eval_gsm8k.sh
python scripts/summarize_table1.py exp/exp__wjl --network-implementation current_software --result-tag table1_s1_paper --bandwidth-mbps 2.5
```

这里还要说明当前仓库状态：现有 HumanEval 脚本包含四算法顺序运行，而当前 `eval_gsm8k.sh` 中前三种算法命令已被注释，只会运行 PipeSD；现有 GSM8K 四算法结果来自分别保存的正式 artifact。做干净重跑时必须恢复三段命令或逐条执行。当前每方法只有一次匹配运行、部分 artifact 来自 dirty worktree，target 模型 hash 和 cloud seed 没有写入 edge manifest，PipeSD 正式结果也没有记录 BO 配置路径；此外尚未计算 HumanEval pass@1 与 GSM8K exact match。因此本页证明的是“当前代码在受控单机模拟链路上的性能与协议行为”，不是完整的论文数值复现或最终准确率复现。

### 结果来源

- [HumanEval 四算法汇总](../edge/exp/exp__wjl__final/humaneval/comparison/table1_scenario1_summary.md)
- [GSM8K 四算法汇总](../edge/exp/exp__wjl__final/gsm8k/comparison/table1_scenario1_summary.md)

---

## 第 13 页：四算法结果——PipeSD 的收益来自哪里

### 页面内容

#### 性能与能耗

| 任务 / 方法 | TPT (ms/accepted token) | 吞吐 (accepted token/s) | 云 GPU 能耗 (J/100 accepted token) |
|---|---:|---:|---:|
| HumanEval Vanilla | 628.705 | 1.591 | 30.060 |
| HumanEval HSL | 773.934 | 1.292 | 29.048 |
| HumanEval EdgeLLM | 1258.404 | 0.795 | 39.091 |
| **HumanEval PipeSD** | **503.444** | **1.986** | **19.121** |
| GSM8K Vanilla | 1062.706 | 0.941 | 45.453 |
| GSM8K HSL | 949.016 | 1.054 | 36.254 |
| GSM8K EdgeLLM | 1498.874 | 0.667 | 54.021 |
| **GSM8K PipeSD** | **872.472** | **1.146** | **34.743** |

#### 流水行为

| 任务 / 方法 | 平均 draft 长度 | 总生成 draft token | Cloud accept 率 | NAV 总次数 | Rollback 率 | Promoted | Discard | Discard 率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HumanEval Vanilla | 5.881 | 1388 | 72.0% | 236 | 42.8% | 0 | 0 | — |
| HumanEval HSL | 2.955 | 1052 | 95.1% | 356 | 14.3% | 0 | 0 | — |
| HumanEval EdgeLLM | 2.597 | 1496 | 66.8% | 576 | 11.5% | 401 | 354 | 46.9% |
| **HumanEval PipeSD** | **5.279** | **1077** | **92.9%** | **204** | **24.0%** | **528** | **185** | **25.9%** |
| GSM8K Vanilla | 3.797 | 1815 | 55.1% | 478 | 60.9% | 0 | 0 | — |
| GSM8K HSL | 2.998 | 1361 | 73.5% | 454 | 46.5% | 0 | 0 | — |
| GSM8K EdgeLLM | 1.830 | 1464 | 68.3% | 800 | 28.1% | 386 | 694 | 64.3% |
| **GSM8K PipeSD** | **2.995** | **1261** | **79.3%** | **421** | **46.3%** | **443** | **502** | **53.1%** |

> “总生成 draft token”指进入正式 NAV 窗口的 draft token 总数；Promoted、Discard 只统计 NAV 等待期的 proactive token。Vanilla 与 HSL 不生成 proactive token，因此 Discard 率记为“—”。

### 建议图示

主图用两组 TPT 柱状图；右侧用三个数字卡片展示 1.249×、1.218×、能耗下降。

### 讲稿

HumanEval 上，PipeSD 将 TPT 从 Vanilla 的 628.7 ms 降到 503.4 ms，加速 1.249 倍，云 GPU active-compute 能耗降低 36.4%。它只需 204 次 NAV，并将 528 个 proactive token 晋升为下一轮正式候选；185 个被丢弃，Discard 率为 25.9%。GSM8K 上，PipeSD 将 TPT 从 1062.7 ms 降到 872.5 ms，加速 1.218 倍，但 Discard 率升至 53.1%，说明该任务上的等待期推测浪费更高。HSL 是 GSM8K 最强基线，PipeSD 仍比它快 1.088 倍；EdgeLLM 在两个数据集上均最慢，与论文趋势不完全一致，说明硬件、基线参数和网络实现仍需进一步对齐。

#### 指标口径补充

- **平均 draft 长度**：正式送入 NAV 的 draft token 总数除以 NAV 总次数；**总生成 draft token**是所有正式 NAV 窗口长度之和，不包含单独统计的 proactive Promoted/Discard。
- **Cloud accept 率**：cloud 接受的 draft token 数除以正式送验 draft token 总数；**NAV 总次数**是 cloud 实际执行验证的次数。
- **Rollback 率**：至少拒绝了一个 draft token 的 NAV 次数除以 NAV 总次数。它衡量“有多少轮发生回滚”，不是“被拒绝 token 占比”。
- **Promoted**：等待父轮 NAV 时提前生成并上传的 proactive token，在父轮 draft 全部接受且 target extra token 与提前分支预期一致时，被晋升为下一轮正式候选的 token 数。Promoted 不代表已经被 cloud 接受。
- **Discard**：父轮未全部接受、extra token 不匹配或分支已过期时，被清除且不能进入输出的 proactive token 数；**Discard 率**为 `Discard / (Promoted + Discard)`。Vanilla 与 HSL 没有等待期 proactive generation，因此该比率无定义，记为“—”。

Vanilla 测评时固定的是常规触发窗口：HumanEval `N=6`、GSM8K `N=4`。代码在累计 draft 达到 `N` 时触发 NAV，但输出 token 上限、剩余 cloud-accepted-token 预算不足 `N`，或 draft 模型提前生成 EOS 时，也会强制提前验证。因此表中显示的是**实际送验长度的平均值**，不会严格等于配置值。本次 HumanEval 的 236 次 NAV 中，225 次长度为 6，其余 11 次为样本或预算结束前的短窗口，所以平均为 `1388/236=5.881`；GSM8K 的 478 次 NAV 中，437 次长度为 4，其余 41 次更短，其中 30 次由 draft EOS 提前触发，最终平均为 `1815/478=3.797`。draft EOS 触发 NAV 不代表最终输出一定以 EOS 结束，因为 cloud 仍可能拒绝该 EOS 并继续生成。

---

## 第 14 页：从真实 trace 看多轮流水线，而不是只看均值

### 页面内容

**图中元素与原始结果的对应关系**

- Draft token 生成区间：逐 token 的生成时间戳和 token index。
- 橙色发送块：`batch_trace` 中每个 batch 的 start/end、batch size 和 round。
- NAV 触发线：触发条件、触发时间及本轮候选长度。
- 云端返回：accepted count、额外 target token 和 NAV latency。
- Proactive 区间：等待 NAV 时提前生成/上传的下一轮候选。
- Promoted / discard：父轮结果返回后对提前数据的最终处置。

![多轮流水线结果](./pipesd-humaneval-trace-multi-round.png)

### 讲稿

均值只能告诉我们整体快了多少，trace 才能证明流水线是否真的发生。这张图选取 HumanEval Task 50 的前 12 个触发窗口。我们能看到 draft 生成与多个小 batch 上传重叠，NAV 等待期仍有 proactive 工作；返回后有些提前结果被晋升为下一轮正式候选，有些因为父轮没有完全接受或额外 token 不匹配而丢弃。也就是说，PipeSD 的收益不是“把所有提前 token 都算作有效”，而是在严格校验下，用能够安全晋升的那一部分隐藏等待时间。

### 详细解释

- [多轮流水线逐窗口分析](./pipesd-humaneval-trace-multi-round-analysis.md)
- [单轮字段与原始 JSON 对应](./pipesd-humaneval-trace-analysis.md)

---

## 第 15 页：四部署模式——拆解模型、网络与流水线贡献

### 页面内容

| 模式 | 模型与执行路径 | 网络 / NAV | 统计预算 |
|---|---|---|---|
| 纯云（Pure Cloud） | target 大模型在单卡 GPU 上完整自回归解码 | warm local request；无客户端—云端传输、无 NAV | 1000 committed output tokens |
| 纯边（Pure Edge） | draft 小模型在 CPU 上独立自回归解码 | 无网络、无 cloud、无 NAV | 1000 committed output tokens |
| 边云串行（Serial Edge-Cloud SD） | CPU draft 固定窗口后整段上传，GPU target 执行 NAV | 20/200 Mbps 软件链路；生成、上传、NAV 串行 | 1000 cloud-accepted draft tokens |
| PipeSD | 与串行模式使用相同 draft/target 与软件链路 | DP 分批上传、生成—传输重叠、NAV 等待期继续推测 | 1000 cloud-accepted draft tokens |

| 能耗口径 | 包含内容 | 不包含内容 |
|---|---|---|
| 纯云 | prompt prefill + 完整 autoregressive decode | 模型加载、客户端传输、整机其他部件 |
| 纯边 | 未测量：当前环境无 RAPL 权限 | — |
| 边云串行 / PipeSD | cloud prompt prefill + target NAV active compute | NAV 间 GPU idle、edge CPU、网络与整机功耗 |

统一条件：两个数据集均使用 seed `3407`，每个模式恰好统计 1000 个各自口径的 benchmark tokens。四模式用于系统开销拆解；只有边云串行与 PipeSD 具有相同模型、网络路径、归一化 token 和云端能耗边界。

### 建议图示

四条横向执行路径；用虚线标出纯模式绕过网络/NAV，并强调边云串行与 PipeSD 共享同一模型和链路。

### 讲稿

四模式实验不要和论文的四算法表混为一谈。纯云测的是 warm target 模型本地请求，包含 prefill 与完整 decode，但绕过客户端—云端传输；纯边完全使用较小的 draft 模型。它们能给出计算参考，却不能作为同质量服务与协同模式直接排名。边云串行和 PipeSD 才是最有解释力的对照：二者共享模型、20/200 Mbps 软件链路、accepted-token 预算和云端能耗范围，差别主要是是否进行 DP 分批与跨阶段重叠。

### 结果来源

- [HumanEval 四模式汇总](../edge/exp/exp__wjl__four__modes__final/humaneval/comparison/four_mode_humaneval.md)
- [GSM8K 四模式汇总](../edge/exp/exp__wjl__four__modes__final/gsm8k/comparison/four_mode_gsm8k.md)

---

## 第 16 页：四部署模式结果——流水线降低了串行云边开销

### 页面内容

#### 延迟、吞吐与能耗

| 任务 / 模式 | TPT (ms/benchmark token) | 吞吐 (benchmark token/s) | TTFT (ms) | 测得能耗 (J/100 benchmark tokens) |
|---|---:|---:|---:|---:|
| HumanEval 纯云* | 4.281 | 233.577 | 36.087 | 176.164 |
| HumanEval 纯边† | 45.934 | 21.770 | 444.092 | — |
| HumanEval 边云串行 | 630.228 | 1.587 | 2596.919 | 30.292 |
| **HumanEval PipeSD** | **503.759** | **1.985** | **2094.993** | **20.026** |
| GSM8K 纯云* | 4.221 | 236.925 | 29.786 | 182.116 |
| GSM8K 纯边† | 33.941 | 29.463 | 119.160 | — |
| GSM8K 边云串行 | 1066.676 | 0.937 | 1748.500 | 45.592 |
| **GSM8K PipeSD** | **876.745** | **1.141** | **1839.752** | **34.930** |

\* 纯云按 committed output token 归一化，包含 prompt prefill 与完整 decode，但不含客户端传输。  
† 纯边按 committed output token 归一化，使用质量不同的 draft 模型，且未取得 RAPL 能耗。协同模式则按 cloud-accepted draft token 归一化，因此纯模式与协同模式的绝对数值不可直接排名。

#### 协同路径行为：相同口径下的公平比较

| 任务 / 模式 | NAV/100 | Draft 长度 | Cloud accept 率 | Rollback 率 | Batch | Upload MiB | Upload 次数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HumanEval 边云串行 | 23.6 | 5.881 | 72.0% | 42.8% | 5.881 | 384.323 | 266 |
| **HumanEval PipeSD** | **20.4** | **5.279** | **92.9%** | **24.0%** | **1.670** | **401.036** | **823** |
| GSM8K 边云串行 | 47.8 | 3.797 | 55.1% | 60.9% | 3.797 | 498.593 | 514 |
| **GSM8K PipeSD** | **42.1** | **2.995** | **79.3%** | **46.3%** | **1.371** | **548.189** | **1366** |

**可直接汇报的同口径比较：** PipeSD 相对边云串行，HumanEval 加速 **1.251×**、云 GPU active-compute 能耗降低 **33.9%**；GSM8K 加速 **1.217×**、能耗降低 **23.4%**。

### 建议图示

横排与竖排两版均在每个数据集子图中并列展示 TPT 与 Energy；TPT 使用对数轴，纯边能耗标为 N/A。四模式依次采用蓝、红、绿、橙配色。

![四模式性能与能耗（横排）](./figures/four_modes_performance_energy_horizontal.png)

![四模式性能与能耗（竖排）](./figures/four_modes_performance_energy_vertical.png)

### 讲稿

四模式结果显示，纯云和纯边的 TPT 远低于协同路径，但前者绕过网络且采用不同归一化，后者更换为小模型，因此只能作为计算参考。公平比较中，PipeSD 在 HumanEval 和 GSM8K 上分别比串行路径快 1.251 倍和 1.217 倍，并降低 33.9% 和 23.4% 的同口径云 GPU active-compute 能耗。PipeSD 通过更小 batch 和更多上传请求换取生成—传输重叠，同时提高接受率并降低 NAV 与回滚频率。HumanEval 的 TTFT 降低约 19.3%，但 GSM8K 上反而增加约 5.2%，说明流水线主要改善总体 TPT，不保证所有任务的首 token 延迟都下降。

---

## 第 17 页：复现完成度——哪些已经完成，哪些还没有

### 页面内容

| 论文/项目内容 | 状态 | 当前证据或缺口 |
|---|---|---|
| 云边 draft + NAV 基础闭环 | 已完成 | 两任务正式结果与完成文本 |
| DP token-batch pipeline | 已完成 | 代码、batch trace、多轮图 |
| 双阈值触发 | 已完成 | 代码与正式运行参数 |
| 等待期继续生成及安全晋升 | 已完成 | parent 元数据、promoted/discard 统计 |
| 软件网络与动态带宽能力 | 代码完成 | 共享 FIFO 与动态 profile；Scenario 4 正式结果缺失 |
| BO 自动调参 | 部分完成 | 16-trial 代码和输出存在；正式 run 与最新 BO 参数溯源未闭环 |
| Scenario 1 四算法 | 部分完成 | 有 1000-token 单次结果；多数 manifest 为 dirty，不同方法跨两个 commit，GSM8K 样本索引未完全一致 |
| Scenario 2 / 3 / 4 | 未完成 | 缺正式对比结果 |
| 带宽扫描 Figure 5 | 未完成 | 缺 10/20/40/80 Mbps 系统测评 |
| 参数拟合 Figure 6 | 部分完成 | 在线估计已实现，缺拟合曲线和误差报告 |
| BO / Grid / Random、固定阈值表 | 未完成 | 论文 Table 3/4 未复刻 |
| 消融、调度策略与 overhead | 未完成 | 论文 Table 5/6、Appendix 对照缺正式报告 |
| HumanEval / GSM8K 正确率 | 未完成 | completion 已保存，尚未算 pass@1 / exact match |
| 端到端能耗与多边缘客户端 | 未完成 | 仅云 GPU；多客户端实验缺失 |
| 自动化测试 | 部分完成 | Cloud 11/11 通过；Edge 69/70 通过，剩余 1 项是默认输出目录断言与当前路径不一致 |

### 建议图示

用绿 / 黄 / 灰三色矩阵；不要使用一个笼统的“完成百分比”。

### 讲稿

当前可以把完成度分为三层。第一层是核心协议闭环，已经完成并有 trace 证明；第二层是论文 Scenario 1 风格实验，已经有结果，但由于硬件、模型制品、单次运行、dirty worktree 和参数溯源问题，只能叫部分复现；第三层是论文完整实验矩阵，包括场景、带宽、消融、BO 对照和多边缘，目前还没有完成。自动化测试也不是全绿，Edge 侧还剩一个实验输出目录断言需要修正。这样的划分比说一个百分比更准确，也能明确下一步工作的优先级。

---

## 第 18 页：后续计划与验收标准

### 页面内容

**P0：先把现有结论做扎实**

1. 修正剩余的输出目录测试，冻结 clean commit、模型 revision/hash、依赖和实验配置。
2. 将 BO 最优参数自动写入正式 run manifest，并重跑四算法。
3. 每个任务至少 3～5 个 seed，报告 mean ± std / 95% CI。
4. 计算 HumanEval pass@1 与 GSM8K exact match，确认加速不以质量下降为代价。

**P1：覆盖论文关键实验**

5. 重跑 Scenario 2、3，完成 10/20/40/80 Mbps 带宽扫描。
6. 构造并公开 Scenario 4 动态带宽 trace，验证在线自适应。
7. 完成 BO vs Grid vs Random、固定阈值、pipeline/trigger 消融和 DP 策略对照。

**P2：增强测试床与系统结论**

8. 补边缘端功耗，给出端到端能耗。
9. 在可获得的真实边缘/云 GPU 和真实链路上验证绝对性能。
10. 扩展多边缘客户端并发和云端排队实验。

**验收标准：** 可复现脚本 + clean manifest + 多次重复 + 正确率 + 原始 trace + 自动汇总报告。

### 建议图示

三阶段路线图：可信结果 → 论文覆盖 → 系统扩展。

### 讲稿

后续计划首先不是马上堆更多场景，而是把现有结果变成可复核的证据：固定版本、打通 BO 参数溯源、多次重复并补正确率。第二阶段再覆盖论文最关键的场景、带宽和消融。第三阶段才是更昂贵的真实测试床、端侧功耗和多客户端。最终验收不以“跑出一个更快的数字”为准，而以任何人能从 clean commit 和 manifest 复现实验、验证质量并追到原始 trace 为准。

---

# 附录

## 附录 A：与论文原始结果如何对照

论文 Scenario 1 的 TPT（ms/token）：

| 任务 | Vanilla | HSL | EdgeLLM | PipeSD | PipeSD vs Vanilla |
|---|---:|---:|---:|---:|---:|
| HumanEval | 194 | 155 | 153 | 129 | 1.50× |
| GSM8K | 193 | 174 | 169 | 145 | 1.33× |

当前结果：

| 任务 | Vanilla | HSL | EdgeLLM | PipeSD | PipeSD vs Vanilla |
|---|---:|---:|---:|---:|---:|
| HumanEval | 628.705 | 773.934 | 1258.404 | 503.444 | 1.249× |
| GSM8K | 1062.706 | 949.016 | 1498.874 | 872.472 | 1.218× |

### 讲稿

这张表说明为什么不能说“复现了论文数值”。当前绝对 TPT 是论文的数倍，而且 HSL、EdgeLLM 的排序也没有完全复现。可信的阶段结论只有两个：第一，核心机制已经在当前环境运行；第二，在同一当前协议下 PipeSD 优于 Vanilla。差异可能来自硬件、模型格式与推理后端、真实网络和软件仿真、阈值、样本集以及实现细节，后续必须通过冻结配置和逐项消融定位。

---

## 附录 B：正式结果目录和字段来源

### 四算法

- `edge/exp/exp__wjl__final/humaneval/{vanilla,hsl,edgellm,pipesd}/`
- `edge/exp/exp__wjl__final/gsm8k/{vanilla,hsl,edgellm,pipesd}/`
- 汇总位于各任务的 `comparison/table1_scenario1_summary.md`。

结果文件通常包含：

- `weighted_tpt_ms` / 总耗时与总 token：主性能指标。
- `gpu_energy_per_100_tokens_j`：云 GPU 能耗。
- `draft_tokens`、`accepted_tokens`、`acceptance_rate`：候选质量。
- `nav_calls`、`nav_per_100_tokens`：验证频率。
- `rollback_*`：被云端否决造成的无效 draft。
- `batch_trace`：每批发送时间、大小和轮次。
- `proactive_reused_*` / `proactive_discarded_*`：等待期提前工作的原始统计字段；PPT 中分别展示为 Promoted / Discard。
- manifest：seed、commit、模型、网络、预算与阈值。

### 四模式

- `edge/exp/exp__wjl__four__modes__final/humaneval/`
- `edge/exp/exp__wjl__four__modes__final/gsm8k/`
- 汇总位于各任务 `comparison/four_mode_*.md`。

---

## 附录 C：论文实验设置与当前覆盖关系

| 项目 | 论文 | 当前正式结果 |
|---|---|---|
| Edge | Lenovo ThinkBook 16+，Core Ultra 9 185H，32 GB，Windows 11 | 当前本地环境，非同一测试床 |
| Cloud | A800 40 GB，Xeon，Ubuntu 22.04 | 当前云端/服务环境，未严格同配 |
| 基础网络 | 上行 20 Mbps，下行 200 Mbps | 软件链路 20/200 Mbps + 25 ms 启动延迟 |
| Scenario 1 | Laptop | 有四算法正式结果 |
| Scenario 2 | 2.5 GHz emulated phone | 无正式结果 |
| Scenario 3 | 1.2 GHz IoT | 无正式结果 |
| Scenario 4 | 动态上行 10～80、下行 150～280 Mbps，每 20 秒变化 | 代码支持 profile，缺正式结果 |
| 统计预算 | 每方法统计 1000 accepted tokens | 每方法恰好 1000 个 cloud-accepted draft tokens，单次运行；accepted token 的具体定义仍需与论文进一步核对 |

---

## 附录 D：汇报时可能被问到的问题

### 1. Cloud 只接受部分 token，但 edge 已生成并上传后续 token，怎么办？

后续 token 属于隔离的 speculative/proactive 数据。父轮部分接受会改变下一轮上下文，因此这些 token 必须取消或丢弃；它们可能已经占用上传带宽，但不会被提交到正式序列。

### 2. Cloud 是否总会在 accepted prefix 后多给一个 token？

是。标准推测解码需要 target 在接受前缀后给出一个继续 token；当前云端 NAV 路径实现了这一返回，并将它用于下一轮衔接校验。

### 3. 如果全部接受，但额外 token 与 edge 已继续生成所假设的 token 不一致呢？

仍然不能晋升 proactive 分支。只有“全部接受 + 额外 token 匹配”同时成立，提前结果才会 promote；否则 discard 并从 target token 重新生成。

### 4. 为什么 PipeSD 不是每个 token 都立刻发？

网络请求存在固定启动开销 \(\alpha\)。逐 token 发送会重复支付启动开销，整段发送又失去重叠。DP 根据 \(\alpha,\beta,\gamma\) 选择中间的批边界。

### 5. 为什么当前结果比论文慢很多？

硬件、推理后端、模型制品、真实网络、样本和参数没有完全一致；目前只适合比较同一当前环境下的方法相对差异，不能跨测试床比较绝对 TPT。

### 6. 现在能否证明输出质量不下降？

还不能。协议上 target NAV 保证了推测解码的序列校验，但当前 comparison 没有给出 HumanEval pass@1 和 GSM8K exact match；这是下一轮正式验收的 P0 工作。

---

## 制作 PPT 时的版式建议

- 第 3～9 页使用统一颜色：边缘为蓝色、上传为橙色、云 NAV 为红色、accepted/promoted 为绿色、discard 为灰色。
- 第 13、16 页只突出最关键的 TPT 和相对收益，完整小数表放附录。
- 第 14 页流水线图建议全页展示，讲解时按一个窗口从左到右走一遍，再说明跨轮晋升条件。
- 每一张结果图页脚都标注：任务、1000 cloud-accepted draft tokens、edge seed 3407、单次运行、当前软件链路配置。
- 对纯边/纯云结果始终保留星号说明，避免被误解为同质量、同服务边界的直接对比。
