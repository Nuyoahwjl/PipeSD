## 第1页：标题

各位老师好，我是王家乐。今天汇报的是论文《PipeSD：一种基于推测解码的高效云边协同流水推理框架》的理解与复现工作。

## 第2页：Motivation

这篇论文研究的是云边协同大模型推理。

基本方案是在边缘设备上部署较小的 draft model，快速生成候选 token；云端部署较大的 target model，对候选序列进行验证，从而降低云端decoding的计算量。

传统流程通常是串行的：边缘先生成一段 draft，生成完成后再上传；云端验证时，边缘暂停并等待结果。因此，边缘计算、网络通信和云端验证之间存在大量空闲时间。

我对这篇论文的理解是：它不只是改进推测解码算法，而是把推测解码建模成一个由边缘生成、网络传输和云端验证组成的系统流水线问题。优化目标是让这三个阶段尽可能重叠。

## 第3页：NAV

这里先介绍 NAV，也就是非自回归验证。

边缘模型生成一段 draft token 后，云端 target model可以在一次前向计算中并行验证整段候选序列。云端比较两个模型的输出，接受最长的匹配前缀，并给出一个额外的 target token；遇到第一个不匹配位置后，后续 draft token 全部失效。

因此，下一轮生成必须从云端确认后的上下文继续。

这也带来一个重要约束：这篇文章中NAV 等待期间提前生成的 token，只有在父轮全部被接受，并且云端返回的额外 token 与提前生成分支的第一个 token 一致时，才能安全复用。

## 第4页：PipeSD Architecture

云端主要包含通信接口和 target model，负责缓存上传的 draft token，并执行 NAV。

边缘端包括 draft model、传输控制器、环境监控器和参数更新器。

传输控制器使用动态规划决定 token 如何分批上传，并使用双阈值决定何时触发 NAV。环境监控器持续估计固定通信开销 alpha、单位 token 传输开销 beta，以及单位 token 生成时间 gamma。参数更新器则根据这些估计重新规划上传批次，并通过 BO 搜索更合适的置信度阈值。

## 第5页：Workflow

完整流程是：边缘逐 token 生成并记录置信度；达到 DP 计算出的批次边界后，异步上传当前批次，同时继续生成；任一置信度阈值被触发后请求 NAV；等待结果期间仍可继续生成，并把提前数据放入隔离的 proactive buffer；最终再根据父轮验证结果决定提升为正式候选还是丢弃。核心思想就是把原来的串行阶段重叠起来，同时保证提前执行不会破坏正确性。

## 第6页：Dynamic Programming

DP 使用三个环境参数进行调度。

PipeSD 将一批 token 的通信时间近似为固定启动开销 α 加上与 token 数量成正比的 β，将生成时间近似为 γ。逐 token 发送虽然重叠充分，但会重复支付 α；整段发送又无法利用流水线。

调度窗口 N-hat 初始为 20，之后根据最近 100 轮的平均 draft 长度更新。DP 在预测窗口内枚前一个分割点，寻找最终上传完成时间最小的分批方案，计算每个上传批次应该包含多少 token。

例如第 9 页中的计划是 1、1、4、14，表示前两个 token 分别上传，接下来上传 4 个，最后上传剩余的 14 个。但它只是计划：如果双阈值提前触发 NAV，当前批次会立即截断并发送。

## 第7页：Bayesian Optimization 贝叶斯优化

触发 NAV 时使用两个阈值。

R1 对应整段 draft 的累计序列置信度，R2 对应当前单 token 的置信度。任意一个置信度低于对应阈值，就触发 NAV。论文使用贝叶斯优化，根据实际 TPT 搜索 R1 和 R2，并在推理期间异步在线更新阈值。

目前的实现是正式测评开始前的 BO：使用 16 个候选点，每个候选测量 20 个accept token，找到最优阈值后，在正式测评中保持固定。目前还没有实现论文中的在线异步 BO。

## 第8页：Proactive Token Lifecycle

提前生成的 proactive token 有三种处理结果。

第一种是父轮出现拒绝，此时上下文已经变化，所有 proactive token 都必须丢弃并回滚。

第二种是父轮全部接受，但云端额外生成的 token 与 proactive 分支的第一个 token 不一致。此时两条生成路径仍然不同，也必须丢弃。

第三种是父轮全部接受，并且额外 token 匹配。此时 proactive token 才能被晋升为下一轮正式候选。

我在代码中补充了 round、window、batch、prefix 等元数据，并在云端严格执行这两个复用条件。

## 第9页：An Example

这一页来自 HumanEval Task 50 的真实运行结果，展示最开始的两轮 NAV。

初始 DP 计划是 1、1、4、14。

第一轮中，D1 和 D2 分别作为 B1、B2 上传，但不触发 NAV。第三批原计划包含 4 个 token，但是生成到 D4 时已经满足 NAV 条件，因此第三批实际只有 D3 和 D4，并立即发起 NAV1。

最后一次请求虽然只携带 D3、D4，但云端已经缓存了 D1、D2，所以 NAV1 实际验证的是 D1 到 D4。

在 NAV1 运行期间，边缘没有暂停，而是继续生成 D5 到 D15，并按照 1、1、4、5 分为 P1 到 P4 上传。P4 原计划最多包含 14 个 token，但生成 5 个以后再次满足 NAV 条件，因此提前发起 NAV2。

NAV1 接受 4 个中的全部 4 个。由于父轮全部接受，并且额外 target token 与 proactive 分支匹配，这 11 个提前生成的 token 被晋升为第二轮正式候选。

NAV2 验证这 11 个 token，其中接受前 10 个、拒绝最后 1 个，因此系统回滚被拒绝的后缀，并从云端返回的纠正 token 继续生成。

## 第10页：Reproduction Work

云端 target model运行在单张 NVIDIA RTX PRO 6000 上；边缘 draft model运行在 CPU 上，设置为HumanEval 使用 DeepSeek-Coder 1.3B 和 6.7B，GSM8K 使用 TinyLlama 1.1B 和 Llama-2 7B Chat。

网络方面，我实现了应用层 SoftwareLink，将论文的 20 Mbps 上行和 200 Mbps 下行换算为 2.5 MB/s 和 25 MB/s。上下行使用两条独立 FIFO；正式请求和 proactive 请求共享同一条上行队列，因此需要竞争相同的带宽资源。真实 loopback HTTP 和 target model计算时间仍然保留在端到端时间中。

并发方面，主请求与 proactive 请求使用独立发送线程；云端执行 target verification 时释放任务缓存锁，使 proactive HTTP 请求确实能够在 NAV 期间到达。每个批次携带 round、window、batch 和 prefix 信息，。云端根据父轮结果安全晋升或丢弃缓存。

测评每个样本最多生成 128 个 token，每种方法一直运行到云端恰好接受 1000 个 token。TPT 使用所有样本的总端到端时间除以接受 token 数量。

能耗通过每 5 毫秒采样一次，只积分 prompt prefill 和 target NAV active compute，不包括模型加载、NAV 间 GPU idle、边缘 CPU、网络和整机功耗。

## 第11页：Baselines

这里比较四种算法。

Vanilla 使用固定 draft 长度，HumanEval 为 6，GSM8K 为 4；生成完整序列后一次上传，等待 NAV 时暂停。

HSL 使用单 token 置信度触发 NAV，当前阈值为 HumanEval 0.95、GSM8K 0.7。

EdgeLLM 使用累计序列置信度触发 NAV，并在等待 NAV 时继续生成。初始阈值分别为 HumanEval 0.92 和 GSM8K 0.5。全接受后的衰减系数调为 0.8。

PipeSD 同时使用单 token 和累计序列阈值，并结合 DP 分批上传。HumanEval 使用约 0.9 和 0.3514，GSM8K 使用约 0.65 和 0.4。

## 第12页：Performance and Energy

在 HumanEval 上，PipeSD 的 TPT 为 503.4 毫秒每 token，优于 EdgeLLM 的 548.5 毫秒；相对于当前最好的基线是 1.09 倍加速，相对于 Vanilla 是1.249倍。云端能耗为每 100 个接受 token 19.1 焦耳，相对于 Vanilla 下降36.4%。

在 GSM8K 上，PipeSD 的 TPT 为 872.5 毫秒，优于 EdgeLLM 的 910.4 毫秒；相对于最好基线为1.043倍，相对于 Vanilla 为1.218倍。能耗为34.7焦耳，相对于 Vanilla 下降23.6%。

这说明 PipeSD 在两个数据集上都取得了最低 TPT 和最低的云端 active-compute 能耗，但 HumanEval 上的优势更加明显。

## 第13页：Pipeline Behavior

流水行为可以进一步解释性能差异。

Rollback rate 表示发生至少一次 token 拒绝的 NAV 轮次比例；discard rate 等于丢弃 token 数除以晋升与丢弃 token 总数。它衡量的是 proactive 计算的浪费程度。

HumanEval 上，PipeSD 的接受率为92.9%，只进行了204次 NAV，discard rate 为25.9%，说明大部分 proactive 工作都能被利用。

GSM8K 上，PipeSD 的接受率下降到79.3%，NAV 增加到421次，discard rate 上升到53.1%。这说明推理任务中 draft model 与 target model的一致性较弱，更多 proactive token 最终失效。

## 第14页：Four Modes

四种部署模式。

Pure Cloud 使用 target model在 GPU 上完成完整自回归生成；Pure Edge 使用 draft model在 CPU 上本地生成；Serial Edge–Cloud SD 按固定窗口串行执行 draft、上传和 NAV；PipeSD 使用相同的模型和链路。

## 第15页：Plan

后续工作主要有五项。

第一，实现推理期间的在线异步 BO，动态更新双阈值。

第二，测评不同带宽、时延和动态网络条件，验证算法的场景适应性。

第三，继续调整 HSL 和 EdgeLLM 的阈值，使基线比较更加充分和公平。

第四，在时间和计算资源允许的情况下增加样本数量，并进行多次重复实验，给出置信区间。

第五，实现多个 edge 共享一个，进一步研究多用户竞争条件下的资源调度、公平性和系统吞吐。

## 第16页：结束

以上是我对 PipeSD 论文的理解、当前复现进展以及后续计划。

我认为云边大模型推理的关键不仅是模型本身，还包括计算、通信、调度和推测正确性之间的协同优化。

我的汇报结束，谢谢各位老师。



https://github.com/Ghanyunhe/PipeSD

| 方向         | 主要改动                                                     |
| ------------ | :----------------------------------------------------------- |
| PipeSD 调度  | 重构 DP 调度器，引入移动平均窗口 \(\hat N\)、循环 batch plan、在线估计通信参数 \(\alpha/\beta\) 和生成时间 \(\gamma\)，并按环境变化重新规划。[merge.py](D:/Desktop/PipeSD/edge/src/merge.py) |
| 贝叶斯优化   | 将 BO 限定为 PipeSD，按照论文设置使用 GP/Matern、EI、16 次调用、1 个初始点和每个候选 20 个 accepted token；隔离候选间状态并保存最佳配置。 |
| 并发流水线   | 将主 NAV 和等待期间的 proactive 上传拆成独立异步 HTTP 通道；支持请求取消、排空、重发、批次跟踪和精确 payload 大小统计。[comm.py](D:/Desktop/PipeSD/edge/src/comm.py) |
| 云端缓存     | 增加 round/window/batch/index/prefix 元数据、父 NAV 状态判断、proactive buffer、cache version，以及复用/废弃统计；云端验证时允许下一轮请求并发进入缓存。[speculative_server.py](D:/Desktop/PipeSD/cloud/src/speculative_server.py) |
| 方法隔离     | 明确四种方法的上传行为：Vanilla/HSL 在 NAV 时上传完整 draft；EdgeLLM 只在等待 NAV 时 proactive 上传；PipeSD 在 NAV 前和等待期间都执行 DP batch plan。 |
| EdgeLLM 修复 | 阈值更新不再错误使用 PipeSD 的预测窗口，而是使用本次 NAV 的真实 draft 长度；加入置信度快照、数值保护和阈值轨迹，并将正式评测 decay 调为 0.8。[engine.py (line 597)](D:/Desktop/PipeSD/edge/src/engine.py:597) |
| HSL 调参     | HumanEval 默认阈值由 0.99 调整为 0.95；GSM8K 保持 0.7。      |