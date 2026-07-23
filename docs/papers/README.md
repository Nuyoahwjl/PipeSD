# PipeSD 相关论文精选

更新日期：2026-07-23

本目录收录 15 篇与 PipeSD 直接相关、或可以迁移到当前实现中做优化的论文。选择标准不是“尽可能多”，而是每篇都能对应到仓库中的具体状态机、调度器、通信路径、验证策略或实验指标。

目标论文是 [PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](../2605.13319v3.pdf)，不在本目录重复存储。

## 使用说明

- 第一组适合先理解云边协同、等待期并行和动态配置；
- 第二组集中讨论“何时停止 drafting、何时触发验证、验证多少候选”；
- 第三组提供 exact speculative decoding、特征级 drafting 和树形验证基础；
- “可实施优化路线”按当前仓库的改造成本和预期收益排序，而不是按论文影响力排序。

本目录只使用 arXiv、ACL Anthology 等公开来源。PipeSD 论文中的 HSL 与 EdgeLLM 是当前仓库的重要对照方法，但 ACM/IEEE 正式版本没有提供可自动验证的开放 PDF，作者主页也未暴露公开稿。因此没有保存来源不明的转载文件，而是分别选入开放获取的 BiLD 和 PEARL，补足 fallback/rollback 与并行自适应 drafting 两条方法链。HSL、EdgeLLM 与仓库实现的关系仍在下文单独说明。

## PipeSD 方法与当前实现

当前仓库已经超出一个最小复现，核心链路如下。

1. `edge/src/engine.py` 维护边缘端 speculative decoding 状态机。
   - `if_verify()` 实现 fixed-length、single-token、sequence-level、hybrid、entropy 和 probability-gap 条件；
   - PipeSD 的 hybrid 条件在“最后一个 token 置信度低于单 token 阈值”或“累计序列置信度低于序列阈值”时触发 NAV；
   - `_resolve_algorithm_batch_plan()` 让 PipeSD 在 NAV 前和 NAV 等待期使用 DP 批计划；EdgeLLM 只在 NAV 等待期使用移动平均窗口；Vanilla/HSL 在 NAV 时一次上传完整 draft；
   - NAV 等待期间使用独立的 `proactive_sender` 继续生成和上传下一轮候选。
2. `edge/src/merge.py` 实现论文的 token-batch DP 与在线调度器。
   - `dynamic_token_scheduling_dp()` 在固定启动开销、每 token 传输时间和生成时间下最小化流水线完成时间；
   - `PaperDPScheduler` 用历史 draft length 更新调度窗口；
   - `OnlineEnvironmentEstimator` 用运行时通信和生成样本更新 `alpha`、`beta`、`gamma`。
3. `edge/src/comm.py` 与 `edge/src/software_link.py` 实现异步、有序、可限速的上下行链路。
   - 当前请求仍携带每个 draft token 的词表概率向量，弱带宽下通信量很大；
   - 主发送通道和 proactive 通道分离，但每个通道内部有序。
4. `cloud/src/speculative_server.py` 实现 target verification、主动批次缓存和能耗追踪。
   - `proactive_buffers` 按 parent NAV round 缓存提前到达的候选；
   - 只有父轮全部接受、且 target 额外 token 与下一轮首 token 一致时才提升缓存，否则丢弃并回滚；
   - 多任务状态可以并存，但 target model 执行由全局 `model_lock` 串行化，尚未进行跨请求 verification batching。
5. `edge/app/run_edge.py` 实现评测和 PipeSD 阈值贝叶斯优化。
   - BO 在正式评测前离线搜索 `verify_thresh_single` 与 `verify_thresh_multi`；
   - 结果包含 TPT、接受率、draft/accept/reject 长度、回滚率、首 token 延迟、batch trace 和云端能耗。

一个需要特别注意的指标边界：云端返回的能耗仅覆盖 prompt prefill 与 target-model NAV 计算，不包含边缘端 draft、网络传输、NAV 间 GPU idle、模型加载或状态恢复/保存。因此后续引用 ConfigSpec、FlexSpec 的“系统能耗”结论时不能直接与当前 `gpu_power_integral_joules` 等同。

## 仓库模块—论文方向映射

| 仓库位置 | 当前职责 | 最相关论文 | 可扩展方向 |
|---|---|---|---|
| `edge/src/engine.py::if_verify()` | NAV 触发 | BiLD、SpecDec++、SVIP、Speculative Verification、HeteroSpec | 熵/接受概率/上下文难度联合触发 |
| `edge/src/engine.py` proactive 分支 | NAV 等待期继续 drafting | PEARL、SpecEdge、HAT、PicoSpec | pre-verify、post-verify、跨请求流水线 |
| `edge/src/merge.py` | token-batch DP 与环境更新 | PipeSD、FlexSpec、ConfigSpec | 联合优化 batch、draft length、能耗和信道 |
| `edge/src/comm.py` | 概率与 token 上传 | PicoSpec | 稀疏概率传输、分离 rejection sampling |
| `cloud/src/speculative_server.py::verify_tokens()` | exact verification 与纠错采样 | Leviathan、Chen、SpecInfer | 批量 NAV、树形 verifier、压缩协议 |
| `cloud/src/speculative_server.py::model_lock` | 串行 target 执行 | SpecEdge、HAT | pipeline-aware microbatch 与公平调度 |
| `edge/app/run_edge.py::bayes_optimize_thresholds()` | 双阈值 BO | SpecDec++、ConfigSpec、FlexSpec | 多目标与在线策略选择 |
| `verify_stats`、`diagnostics`、energy trace | 实验观测 | ConfigSpec、HeteroSpec | goodput、P95 ITL、端到端 joules/token |

## 一、云边协同与直接基线

### 1. Speculative Decoding with Big Little Decoder（BiLD，NeurIPS 2023）

- 作者：Sehoon Kim、Karttikeya Mangalam、Suhong Moon 等，UC Berkeley。
- 本地 PDF：[2023-bild-big-little-decoder-2302.07863.pdf](2023-bild-big-little-decoder-2302.07863.pdf)
- 官方页面：[arXiv:2302.07863](https://arxiv.org/abs/2302.07863)
- 核心贡献：BiLD 用小模型持续生成，并以 fallback policy 决定何时交给大模型，以 rollback policy 决定大模型需要回退和纠正多长的后缀。它更接近“质量近似的大小模型协作”，不具备标准 speculative sampling 的严格同分布保证，但清晰地建立了置信度触发与回滚之间的权衡。
- 与本仓库的关系：可用来理解 HSL 的 single-token confidence 触发，以及 `edge/src/engine.py` 在 NAV 返回后对 draft cache、`n_tokens` 和输出前缀的修正。PipeSD 的双阈值本质上是在单 token fallback 之外增加累计序列风险。
- 可尝试优化：把当前单一回滚统计扩展为“触发原因—回滚长度”联合分布，分别评估 single、multi、hybrid 条件是否过早或过晚。该改动只涉及诊断字段和离线分析，适合作为低风险消融。

### 2. A Novel Hat-Shaped Device-Cloud Collaborative Inference Framework for Large Language Models（HAT，2025）

- 作者：Zuan Xie、Yang Xu、Hongli Xu、Yunming Liao、Zhiwei Yao。
- 本地 PDF：[2025-hat-device-cloud-2503.18989.pdf](2025-hat-device-cloud-2503.18989.pdf)
- 官方页面：[arXiv:2503.18989](https://arxiv.org/abs/2503.18989)
- 核心贡献：HAT 将 LLM 的首尾层放在设备端、中间层放在云端，以隐藏状态而非原始 token 通信；同时引入 prompt chunking、状态监控和 parallel drafting。它兼顾隐私与并行，但隐藏状态传输比 PipeSD 的 token 协议更重。
- 与本仓库的关系：HAT 的 parallel drafting 与 `proactive_sender`/`proactive_buffers` 高度相似；prompt chunking 则指出当前仓库主要优化 decode、没有系统优化 TTFT/prefill 的缺口。
- 可尝试优化：先不采用 HAT 的层切分，只迁移“prefill chunk + decode request 混合调度”思想。在云端队列中记录 prompt 长度、NAV 长度和 SLA，测量长 prompt 是否阻塞短 NAV。

### 3. SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs（NeurIPS 2025）

- 作者：Jinwoo Park、Seunggeun Cho、Dongsu Han，KAIST。
- 本地 PDF：[2025-specedge-2505.17052.pdf](2025-specedge-2505.17052.pdf)
- 官方页面：[arXiv:2505.17052](https://arxiv.org/abs/2505.17052)
- 核心贡献：SpecEdge 把 drafting 下沉到消费级边缘 GPU，采用 proactive edge drafting 隐藏 RTT 与 verification latency，并以 pipeline-aware scheduling 交错多个请求，提高服务器吞吐。其 draft depth 依据“server verification time ≈ edge drafting time + RTT”动态调整。
- 与本仓库的关系：当前主动轮的有效性检查已经复现了类似的 full-accept + prefix-match 条件；真正缺失的是服务端跨请求 pipeline-aware batching。现有 `model_lock` 使不同任务的 NAV 仍串行执行。
- 可尝试优化：把 `handle_propose_payload()` 产生的 NAV 放入 microbatch 队列，按等待时间、序列长度和 draft depth 组成 target batch；同时保留每任务 cache version。重点观察 aggregate tok/s、P50/P95 ITL、GPU utilization 和公平性。

### 4. FlexSpec: Frozen Drafts Meet Evolving Targets in Edge-Cloud Collaborative LLM Speculative Decoding（2026）

- 作者：Yuchen Li、Rui Kong、Zhonghao Lyu 等。
- 本地 PDF：[2026-flexspec-2601.00644.pdf](2026-flexspec-2601.00644.pdf)
- 官方页面：[arXiv:2601.00644](https://arxiv.org/abs/2601.00644)
- 核心贡献：FlexSpec 用 shared-backbone 让一个固定的边缘 draft 适配不断变化的云端 target，避免频繁同步模型；又用 channel-aware adaptive speculation 根据实时信道状态和设备能耗预算调整 draft length。
- 与本仓库的关系：`SoftwareLink`、`OnlineEnvironmentEstimator` 和 `PaperDPScheduler.update_parameters()` 已具备信道测量基础，但当前主要调整 batch plan，没有把信道、接受率和能耗联合用于 NAV/draft-length 决策。
- 可尝试优化：在现有环境快照中加入 acceptance-rate EWMA 和 edge energy proxy，按网络 profile 选择 `schedule_window`、hybrid 阈值和最大 draft length。先做离线 replay，避免在线策略同时改变太多变量。

### 5. PicoSpec: A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference（2026）

- 作者：Yida Zhang、Zhiyong Gao、Shuaibing Yue、Jie Li、Rui Wang。
- 本地 PDF：[2026-picospec-2603.19133.pdf](2026-picospec-2603.19133.pdf)
- 官方页面：[arXiv:2603.19133](https://arxiv.org/abs/2603.19133)
- 核心贡献：PicoSpec 以异步流水线解决 edge SLM 与 cloud LLM 互相等待，并提出 separate rejection sampling with sparse compression，避免每轮反复传完整词表概率，仅承担一次压缩词表传输成本。
- 与本仓库的关系：当前 `edge/src/engine.py` 把 `current_batch_probs`/`total_speculative_probs` 随 token 一起 msgpack 上传，`cloud/src/speculative_server.py::verify_tokens()` 依赖完整 draft distribution 计算接受比和 residual distribution。这正是弱带宽下的主要可优化对象。
- 可尝试优化：先记录 payload 中 token、概率矩阵和协议字段各自字节数，再实现 PicoSpec 协议原型。必须用同种子对比 accepted tokens、final token 和输出分布，不能只看压缩率。

### 6. ConfigSpec: Profiling-Based Configuration Selection for Distributed Edge–Cloud Speculative LLM Serving（2026）

- 作者：Xiangchen Li、Saeid Ghafouri、Jiakun Fan、Babar Ali、Hans Vandierendonck、Dimitrios S. Nikolopoulos。
- 本地 PDF：[2026-configspec-2604.09722.pdf](2026-configspec-2604.09722.pdf)
- 官方页面：[arXiv:2604.09722](https://arxiv.org/abs/2604.09722)
- 核心贡献：ConfigSpec 联合 profile 边缘 drafting throughput、draft-target alignment、功耗、量化级别和 speculative length，并分别优化 goodput、verification cost efficiency 与 energy efficiency。论文的重要结论是三个目标的最优配置并不相同。
- 与本仓库的关系：仓库已有 Bayesian threshold search、网络 profile、接受率和云端能耗，但配置搜索仍围绕两个阈值，且能耗范围不含边缘端。ConfigSpec 提供了从“阈值调参”扩展到“模型/量化/K/阈值联合选择”的框架。
- 可尝试优化：新增离线 configuration selector，输入 draft model、GGUF 量化、带宽、RTT、K、双阈值，输出 Pareto frontier。至少报告 accepted tok/s、云端 J/100 accepted tokens、端侧估计功耗和回滚开销。

### 7. PEARL: Parallel Speculative Decoding with Adaptive Draft Length（ICLR 2025）

- 作者：Tianyu Liu、Yun Li、Qitan Lv、Kai Liu、Jianchen Zhu、Winston Hu、Xiao Sun。
- 本地 PDF：[2025-pearl-parallel-adaptive-draft-length-2408.11850.pdf](2025-pearl-parallel-adaptive-draft-length-2408.11850.pdf)
- 官方页面：[arXiv:2408.11850](https://arxiv.org/abs/2408.11850)
- 核心贡献：PEARL 通过 pre-verify 在 drafting 期间提前验证首 token，通过 post-verify 在 target verification 期间继续 drafting，从而同时缓解双方等待并形成自适应 draft length。
- 与本仓库的关系：PipeSD 已实现与 post-verify 接近的 NAV 等待期生成，但没有显式 pre-verify。当前 hybrid 触发只有在 edge 累积置信度下降后才发 NAV，困难 token 仍可能产生额外 draft。
- 可尝试优化：增加“一 token 早验证”的实验模式，与现有 proactive 轮并行；先在高 RTT 与低接受率组合中测试。需要严格处理 parent round、cache version 和早返回造成的发送取消。

## 二、动态验证与草稿长度

### 8. SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths（COLM 2025）

- 作者：Kaixuan Huang、Xudong Guo、Mengdi Wang。
- 本地 PDF：[2024-specdec-plus-plus-2405.19715.pdf](2024-specdec-plus-plus-2405.19715.pdf)
- 官方页面：[arXiv:2405.19715](https://arxiv.org/abs/2405.19715)
- 核心贡献：SpecDec++ 将 candidate length 建模为 MDP，并证明最优停止策略具有阈值形式；它训练 acceptance prediction head，预测当前候选至少发生一次拒绝的概率，再决定停止 drafting。
- 与本仓库的关系：PipeSD 的累计 max-prob product 是无需训练的 rejection-risk proxy，SpecDec++ 提供了更直接的 target acceptance 估计。`if_verify()` 可以把 predictor 输出作为新模式，而不改 cloud exact verification。
- 可尝试优化：先用现有 `verify_his`、draft logits、位置和上下文长度训练轻量校准器；与 product threshold 比较 calibration error、TPT、rollback rate。只有在离线增益稳定后再加入在线路径。

### 9. Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation（SVIP，EMNLP 2025）

- 作者：Ziyin Zhang、Jiahao Xu、Tian Liang、Xingyu Chen、Zhiwei He、Rui Wang、Zhaopeng Tu。
- 本地 PDF：[2025-svip-draft-model-knows-when-to-stop-2025.emnlp-main.844.pdf](2025-svip-draft-model-knows-when-to-stop-2025.emnlp-main.844.pdf)
- 官方页面：[ACL Anthology 2025.emnlp-main.844](https://aclanthology.org/2025.emnlp-main.844/)
- 核心贡献：SVIP 观察到 draft entropy 可以近似反映 draft-target discrepancy，以训练自由的熵策略动态决定 draft length，在长上下文和推理任务上替代固定 K。
- 与本仓库的关系：`engine.py` 已有 `_entropy_topk()` 与 `entropy` verify mode，但没有将熵作为 PipeSD hybrid 的正式第三信号，也没有针对长上下文调节阈值。
- 可尝试优化：先把 top-k entropy 加入 trace，不参与决策；验证它对下一轮接受长度和回滚事件的预测能力。若有效，再采用 context-length 分桶的 entropy threshold。

### 10. Speculative Verification: Exploiting Information Gain for Speculative Decoding（ACL Findings 2026）

- 作者：Sungkyun Kim、Jaemin Kim、Dogyung Yoon、Jiho Shin、Junyeol Lee、Jiwon Seo。
- 本地 PDF：[2026-speculative-verification-2509.24328.pdf](2026-speculative-verification-2509.24328.pdf)
- 官方页面：[arXiv:2509.24328](https://arxiv.org/abs/2509.24328)
- 核心贡献：该工作引入一个与 draft 相近的小 companion model，以 companion distribution 提供的信息增益估计 draft-target alignment，并动态调整 verification length，重点减少大 batch 下的无效验证。
- 与本仓库的关系：当前 PipeSD 只利用 draft 自身概率，不知道 draft 与 target 在当前上下文的实际对齐程度；BO 只能找到数据集级静态阈值。
- 可尝试优化：短期不增加 companion model，先用历史 NAV 返回构造在线 alignment estimator；长期再评估 companion 是否能在边缘设备预算内运行。这个方向更适合多请求吞吐而非单请求低延迟。

### 11. HeteroSpec: Leveraging Contextual Heterogeneity for Efficient Speculative Decoding（ACL 2026）

- 作者：Siran Liu、Yang Ye、Qianchao Zhu、Zane Cao、Yongchao He。
- 本地 PDF：[2026-heterospec-2026.acl-long.589.pdf](2026-heterospec-2026.acl-long.589.pdf)
- 官方页面：[ACL Anthology 2026.acl-long.589](https://aclanthology.org/2026.acl-long.589/)
- 核心贡献：HeteroSpec 将不同候选的 verification difficulty 视为异构性，使用轻量 entropy quantifier 分层上下文，并联合调整 speculative depth、pruning threshold 与计算图，减少低价值候选的验证资源。
- 与本仓库的关系：当前云端对收到的线性 draft 统一执行 verification，且双阈值对所有上下文使用相同形式。HeteroSpec 提示应把“触发 NAV”和“NAV 验证多少/如何分配资源”拆成两个决策。
- 可尝试优化：用已有 diagnostics 把轮次按 entropy、draft length、acceptance 划分为 easy/medium/hard 三档，为每档选不同 `schedule_window` 与 threshold；先离线重放，不立即引入树剪枝。

## 三、基础与系统级方法

### 12. Fast Inference from Transformers via Speculative Decoding（ICML 2023）

- 作者：Yaniv Leviathan、Matan Kalman、Yossi Matias。
- 本地 PDF：[2023-fast-inference-speculative-decoding-2211.17192.pdf](2023-fast-inference-speculative-decoding-2211.17192.pdf)
- 官方页面：[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- 核心贡献：提出 exact speculative decoding，通过近似模型给出候选、target 并行验证，并用纠正采样保证输出分布与 target 单独解码相同。
- 与本仓库的关系：这是 `cloud/src/speculative_server.py::verify_tokens()` 接受概率、首拒绝位置与 final token 采样的理论基础，也是验证通信压缩、树形候选或提前上传时不能破坏的正确性基线。
- 可尝试优化：建立固定 seed 的逐轮 trace 对照测试，比较 token、随机数、接受决策、final token 和 cache version，作为后续任何协议改造的“exactness 防线”。

### 13. Accelerating Large Language Model Decoding with Speculative Sampling（2023）

- 作者：Charlie Chen、Sebastian Borgeaud、Geoffrey Irving、Jean-Baptiste Lespiau、Laurent Sifre、John Jumper，DeepMind。
- 本地 PDF：[2023-speculative-sampling-2302.01318.pdf](2023-speculative-sampling-2302.01318.pdf)
- 官方页面：[arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- 核心贡献：独立给出 speculative sampling 与修改后的 rejection sampling，在硬件数值误差范围内保持 target distribution，并展示大模型并行评分短候选的延迟优势。
- 与本仓库的关系：云端当前以 `target_probs / draft_probs` 逐 token 接受，并在拒绝时从 `max(target_probs - draft_probs, 0)` 采样 final token，直接对应论文算法。
- 可尝试优化：为 `verify_tokens()` 增加概率归一化、极小概率和数值稳定性测试，特别覆盖压缩概率、量化 draft 与 top-k 截断场景。

### 14. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty（ICML 2024）

- 作者：Yuhui Li、Fangyun Wei、Chao Zhang、Hongyang Zhang。
- 本地 PDF：[2024-eagle-2401.15077.pdf](2024-eagle-2401.15077.pdf)
- 官方页面：[arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- 核心贡献：EAGLE 不直接在 token 空间训练传统 draft model，而是利用 target 特征并处理 feature uncertainty，以较低 drafting 成本提高接受长度。
- 与本仓库的关系：当前边缘模型是独立 GGUF LLM，优点是部署解耦，缺点是 draft-target alignment 受限。EAGLE 可显著改变接受率，但需要云端特征、额外训练和新的模型格式。
- 可尝试优化：列为 P2。若后续迁移到 PyTorch/vLLM target，再评估 feature drafter；在 llama-cpp-python 双模型架构中直接接入成本过高，不应作为第一阶段优化。

### 15. SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification（ASPLOS 2024）

- 作者：Xupeng Miao、Gabriele Oliaro、Zhihao Zhang 等。
- 本地 PDF：[2024-specinfer-2305.09781.pdf](2024-specinfer-2305.09781.pdf)
- 官方页面：[arXiv:2305.09781](https://arxiv.org/abs/2305.09781)
- 核心贡献：SpecInfer 将多个 speculative model 的候选组织成 token tree，并用 target 一次并行验证整棵树，减少串行 target 调用，同时保持模型质量。
- 与本仓库的关系：当前协议只支持线性 `tokens + probs + index`，云端 KV/cache 恢复也面向单路径。引入树形候选需要重做 payload、attention mask、cache 分支和 accepted path 返回结构。
- 可尝试优化：列为 P2。在改树结构前，先完成 SpecEdge 式跨请求 batching；后者不改变单请求 exact sampling 语义，收益验证和回滚风险更低。

## PipeSD 原始基线补充

### HSL：Hybrid SLM and LLM for Edge-Cloud Collaborative Inference

- 正式页面：[ACM DOI 10.1145/3662006.3662067](https://doi.org/10.1145/3662006.3662067)
- 仓库中的对应模式：`algorithm == "hsl"` 强制 single-token confidence 策略，并在 NAV 时一次上传完整 draft。
- 与 PipeSD 的差别：HSL 只看当前 token 置信度，无法识别“每个 token 都中等可信但累计风险已很高”的序列。
- 本地归档说明：ACM 自动下载被访问保护拦截，OpenAlex/作者主页未提供公开 PDF URL，因此没有保存第三方转载。

### EdgeLLM：Fast On-Device LLM Inference with Speculative Decoding

- 正式页面：[IEEE DOI 10.1109/TMC.2024.3513457](https://doi.org/10.1109/TMC.2024.3513457)
- 仓库中的对应模式：`algorithm == "edgeLLM"` 使用累计序列置信度和动态 `alpha`，只在 NAV pending 期间主动上传，并以移动平均 draft window 控制批大小。
- 与 PipeSD 的差别：PipeSD 在 NAV 前也使用 DP pipeline，并以 single + sequence 双阈值触发；EdgeLLM 基线只有序列级条件。
- 本地归档说明：IEEE、OpenAlex、Semantic Scholar 与三位作者的公开主页均未暴露可验证的作者 PDF，因此以开放的 PEARL 补充并行自适应 draft-length 方法链。

## 推荐阅读顺序

1. Leviathan 与 Chen：先掌握 exact acceptance 和 residual sampling；
2. PipeSD 目标论文：理解双阈值、DP batching 和 proactive upload；
3. BiLD、PEARL：理解 fallback/rollback 与等待期并行；
4. SpecEdge、PicoSpec：分别看服务端调度和通信压缩；
5. ConfigSpec、FlexSpec：扩展到多目标、动态信道和能耗；
6. SpecDec++、SVIP、Speculative Verification、HeteroSpec：逐步增强触发和验证策略；
7. EAGLE、SpecInfer：最后评估需要改模型或 verifier 结构的方向。

## 可实施优化路线

### P0：优先做

1. **PicoSpec 式通信压缩**
   - 入口：`edge/src/engine.py` 的 propose payload、`cloud/src/speculative_server.py::verify_tokens()`；
   - 原因：当前每 token 上传完整词表概率，弱带宽场景最容易被通信体积主导；
   - 第一步：只加 payload component byte trace，建立 token/probability/protocol 占比；
   - 验证：相同 seed 下逐轮接受结果和 final token 一致，再比较 TPT 与上行字节。
2. **SpecEdge 式服务端 pipeline-aware scheduling**
   - 入口：`handle_propose_payload()`、`model_lock`、任务 cache save/restore；
   - 原因：当前多客户端能并发收包，但 target NAV 仍全局串行；
   - 第一步：实现固定最大等待时间的 NAV microbatch 原型；
   - 验证：aggregate tok/s、P50/P95 ITL、GPU utilization、每请求公平性和 cache consistency。
3. **ConfigSpec 式多目标配置选择**
   - 入口：`edge/app/run_edge.py`、CLI 参数、已有结果 JSON；
   - 原因：单一最小 TPT 不代表最低能耗或最低成本；
   - 第一步：离线构造 draft model/量化/K/阈值/带宽的 Pareto 表；
   - 验证：goodput、云端 J/100 accepted tokens、估计端侧能耗和回滚率。

### P1：在 P0 基础上做

1. **动态 NAV 信号**
   - 先记录 SVIP entropy、SpecDec++ acceptance proxy、HeteroSpec context class；
   - 用离线 trace 验证预测性，再决定是否替换双阈值；
   - 避免同时改变阈值、DP 窗口和 proactive 策略，保证消融可解释。
2. **FlexSpec 式 channel/energy-aware draft length**
   - 复用 `SoftwareLink` 与环境估计器；
   - 将网络状态、接受率和能耗预算映射到调度窗口与最大 draft length；
   - 对动态带宽 profile 做稳定性测试，避免频繁重调度。
3. **PEARL 式 pre-verify**
   - 在现有 post-verify/proactive 机制前增加一 token 早验证；
   - 优先测试低接受率、高 RTT 场景；
   - 重点验证旧请求取消、parent round 和 cache version。

### P2：结构性升级

1. EAGLE 特征 drafter：需要额外训练、target 特征接口和不同模型运行时；
2. SpecInfer 树形候选：需要树形 payload、attention mask、分支 KV cache 与 accepted-path 协议；
3. 在完成 P0 的服务端 batching 之前，不建议同时推进这两项。

## 下载来源与校验

| 类别 | 数量 | 来源 |
|---|---:|---|
| arXiv | 13 | `arxiv.org/pdf/<id>` 最新公开版本 |
| ACL Anthology | 2 | EMNLP 2025、ACL 2026 正式 PDF |
| 合计 | 15 | 全部通过 `%PDF-`、页数和首屏文本校验 |

校验结果：15 个文件均可由 `pypdf` 打开，页数范围为 6–27 页，第一页均能提取与标题匹配的文本。
