# PipeSD 相关论文库设计

日期：2026-07-23

## 1. 目标

在 `docs/papers/` 建立一个精选、可追溯、能直接服务于本仓库后续优化的论文库。论文库包含 15 篇相关论文 PDF，以及一份中文索引。索引不仅概述论文内容，还要将论文方法映射到 PipeSD 当前实现，并指出可落地的优化方向。

本任务只整理研究资料和分析，不修改推理代码、实验逻辑或已有结果文件。

## 2. 仓库与论文方法的对应关系

本仓库由边缘端实验驱动程序和云端验证服务组成。

- `edge/src/engine.py`
  - speculative decoding 主状态机；
  - fixed-length、single-token、sequence-level 和 hybrid NAV 触发；
  - NAV 等待期间的主动 drafting、上传、有效性检查和回滚；
  - DP 批调度、在线环境参数更新、诊断指标与结果落盘。
- `edge/src/merge.py`
  - token-batch 动态规划；
  -调度窗口和通信、生成参数的在线更新。
- `edge/src/comm.py` 与 `edge/src/software_link.py`
  - 带宽、启动时延和上下行链路模拟；
  -异步请求队列、取消和重提。
- `edge/app/run_edge.py`
  - HumanEval/GSM8K 评测；
  - PipeSD 双阈值贝叶斯优化；
  -实验协议与结果聚合。
- `cloud/src/speculative_server.py`
  - target model 验证和 rejection sampling；
  -主动批次缓冲、提升与丢弃；
  -多客户端任务状态、全局模型锁和能耗追踪。

论文库围绕上述真实边界组织，而不是泛化地收集所有 speculative decoding 工作。

## 3. 论文选择方案

采用“代码模块映射式精选”，总数控制在 15 篇：

### 3.1 云边协同与直接基线

1. Hybrid SLM and LLM for Edge-Cloud Collaborative Inference（HSL）
2. EdgeLLM: Fast On-Device LLM Inference with Speculative Decoding
3. A Novel Hat-Shaped Device-Cloud Collaborative Inference Framework for Large Language Models（HAT）
4. SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs
5. FlexSpec: Frozen Drafts Meet Evolving Targets in Edge-Cloud Collaborative LLM Speculative Decoding
6. A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference（PicoSpec）
7. ConfigSpec: Profiling-Based Configuration Selection for Distributed Edge-Cloud Speculative LLM Serving

这组论文解释仓库的 HSL/EdgeLLM 基线、主动 drafting、网络流水线、服务端调度，以及动态网络和能耗条件下的配置选择。

### 3.2 动态验证与草稿长度

8. SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths
9. Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation（SVIP）
10. Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding
11. HeteroSpec: Leveraging Contextual Heterogeneity for Efficient Speculative Decoding

这组论文用于扩展 PipeSD 当前的单 token、累计序列概率和双阈值信号，重点考察熵、接受概率预测、验证长度和异构验证开销。

### 3.3 基础与系统级方法

12. Fast Inference from Transformers via Speculative Decoding
13. Accelerating Large Language Model Decoding with Speculative Sampling
14. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
15. SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification

这组论文提供 exact speculative decoding 的理论基础，以及将线性草稿升级为特征级或树形候选时所需的验证器与服务系统参考。

如果正式出版版本无法从官方渠道公开获取，则使用同一论文的作者公开稿。若同一论文不存在可公开获取的作者版本，则用候选池中下一篇与同一仓库模块相关的开放论文替换，使最终 PDF 数量仍为 15；不下载来源不明的转载文件。

## 4. 归档结构

```text
docs/
├── 2605.13319v3.pdf
└── papers/
    ├── README.md
    ├── 2023-fast-inference-speculative-decoding-2211.17192.pdf
    ├── 2023-speculative-sampling-2302.01318.pdf
    └── ...
```

文件命名规则为：

```text
<year>-<short-title>-<stable-id>.pdf
```

- arXiv 论文使用 arXiv ID；
- ACL 论文使用 Anthology ID；
-正式出版且无 arXiv 版本的论文使用 DOI 的末段或公认简称；
-不使用顺序编号，避免增删论文时发生无意义重命名；
-目标论文已位于 `docs/2605.13319v3.pdf`，索引引用该文件，不在 `papers/` 重复存储。

## 5. 中文索引内容

`docs/papers/README.md` 包含：

1. PipeSD 目标论文和仓库实现摘要；
2. 仓库模块到论文方向的快速映射表；
3. 15 篇论文的分层清单；
4. 每篇论文的：
   - 标题、年份、作者或机构；
   - 本地 PDF 链接和官方页面；
   - 核心贡献；
   - 与 PipeSD 的具体关系；
   - 可尝试的优化及预期改造范围；
5. 推荐阅读顺序；
6. 优化路线优先级。

优化路线按以下标准排序：

- P0：与现有架构兼容、预期收益高、可以独立做消融；
- P1：需要增加模型信号、调度器或云端批处理能力；
- P2：需要更换 draft/verification 结构或训练额外模块。

初始推荐优先级为：

- P0：PicoSpec 的概率分布稀疏压缩/分离 rejection sampling；
- P0：SpecEdge 的服务端 pipeline-aware 多请求调度；
- P0：ConfigSpec 的吞吐、能耗、成本多目标配置选择；
- P1：FlexSpec 的信道感知 draft length 与能耗约束；
- P1：SVIP、SpecDec++ 和 HeteroSpec 的动态 NAV 信号；
- P2：EAGLE 或 SpecInfer 的特征级、树形候选。

## 6. 下载与验证

下载时优先使用：

1. arXiv 最新公开版本；
2. ACL Anthology、PMLR、NeurIPS Proceedings 或 OpenReview 正式版本；
3. 作者主页或机构仓库公开版本。

每个 PDF 必须通过以下检查：

- 文件存在且非空；
-文件头是 `%PDF-`，避免将错误页保存为 PDF；
- `pypdf` 可以打开；
-页数大于 0；
-第一页可提取标题相关文本；
-本地文件名、官方来源和索引条目一致。

最终检查还包括：

- `docs/papers/` 中恰有 15 篇目标 PDF；
- README 中 15 个本地链接全部有效；
-没有重复论文或重复目标论文；
-没有 `TBD`、`TODO`、占位链接或无法解释的筛选项；
-不修改用户现有的未跟踪文件。

## 7. 非目标

- 不批量收集 speculative decoding 的全部论文；
-不下载补充材料、幻灯片或代码仓库快照；
-不在本任务中实现论文提出的优化；
-不改写现有 PipeSD 实验结果；
-不把无法验证来源的 PDF 加入仓库。
