# Table 1 场景 1：论文协议复现与测评

本文档对应 arXiv:2605.13319v3。当前实现对照的官方仓库为
`Ghanyunhe/PipeSD`，检查时官方 `main` 为 `dac0f52`。绝对 TPT 依赖硬件；不同于论文的
Intel Core Ultra 9 185H + A800 40 GB 时，应称为“Table 1 场景 1 协议复现”。

## 已对齐的关键口径

- `--evaluation_protocol paper_table1` 按 accepted/output token 总预算停止，默认精确 1000 tokens。
- 运行级 TPT 为 `sum(sample.total_time) / sum(sample.output_length)`，不再平均样本 TPT。
- 样本 token latency 与运行级 latency 分开保存；每条样本不再包含之前样本的数据。
- 场景 1 默认没有人工计算延迟。DP 初值使用 `--initial_generation_gamma`；只有显式传入
  `--enable_compute_emulation --emulated_generation_delay ...` 才会 sleep。
- DP 批计划在窗口之间循环，例如 `[1,3,8] -> [1,3,8,1,3,8,...]`。
- 等待 NAV 时 edge 继续生成，后续批次经独立 HTTP 发送通道进入 cloud 的分轮缓存；cloud
  验证过程不再持有任务缓存锁。NAV、EOS、预算或等待阶段 NAV 会提前 flush。
- 四种方法的上传路径相互隔离：Vanilla/HSL 在触发 NAV 后一次上传完整 draft；EdgeLLM
  在 NAV 前不上传，只有等待 NAV 时才按当前移动平均窗口 `N-hat` proactive 上传；PipeSD
  在 NAV 前和等待期间都执行 DP 批计划。
- proactive batch 带 round/window/batch/index/prefix 元数据；云端仅在上一轮全接受且 target
  extra token 与 edge 预期 token 一致时复用，否则丢弃。
- BO 论文模式为 `(0,1)^2`、GP/Matern、EI、`xi=0.1`、16 次、1 个随机初始点、每候选总计
  20 accepted tokens。旧的“每样本 20 tokens”保留为 `--bo_protocol sample_coverage`。
- alpha/beta 只使用 `/delay` 探针和不触发验证的 propose 响应；NAV 响应耗时不进入通信回归。
- gamma 覆盖一次真实 token step 的 sample、softmax 与 draft eval；网络等待、NAV 和场景 2/3
  的人工 sleep 均在计时区间之外。
- Scenario 1 的上行参数是 2.5 MB/s、下行是 25 MB/s。正式脚本使用
  `--network_shaping_mode os`，必须先用 `tc` 或等价系统工具执行限速。
- 每个正式结果包含 manifest、run id、Git commit/dirty 状态、数据和 draft 模型 SHA-256、
  样本顺序、DP/网络估计、batch trace、验证/接受/rollback/proactive/能耗统计。

## 1. 启动云端

在云端仓库的 `cloud/` 下：

```bash
export GPU_POWER_SAMPLE_INTERVAL=0.005
python -m src.speculative_server
```

边端设置服务地址：

```bash
export PIPE_SD_SERVER_URL=http://CLOUD_HOST:8000
```

若云端和边端不在论文同等网络中，分别限制：edge -> cloud 20 Mbps，cloud -> edge 200 Mbps。
系统级整形必须在正式运行前完成，并用 1--8 token 通信探针检查回归的 alpha、beta、R2。

当前服务为本机 `127.0.0.1:8000` 时，可在 Linux/WSL 网络命名空间中执行：

```bash
sudo -v
bash edge/scripts/setup_tc_loopback_scenario1.sh
# 实验结束后清理：
bash edge/scripts/setup_tc_loopback_scenario1.sh --clear
```

该脚本按 `dport 8000` 将请求限制为 20 Mbps，按 `sport 8000` 将响应限制为 200 Mbps。
若 Python 实际运行在 Windows 主机而不是 WSL 内，WSL 的 `tc` 不会整形 Windows loopback；
此时应把 edge/cloud 放进同一 Linux 网络命名空间，或使用 Windows QoS/容器网络整形。

## 2. 每个数据集独立运行 PipeSD BO

```bash
cd edge
SEED=1234 BO_TOKENS_PER_TRIAL=20 bash scripts/bo_humaneval.sh
SEED=1234 BO_TOKENS_PER_TRIAL=20 bash scripts/bo_gsm8k.sh
```

每个数据集建议用 3 个 seed 重复 BO。每次 BO 会在对应的
`exp/exp__wjl/<dataset>/pipesd/latest_bayes_best.json` 写入稳定配置；正式评测脚本自动读取，
配置缺失时直接失败。论文只说明 EdgeLLM 的初始 `R1` 按实验设置选择，没有给出 BO
流程，因此代码不再为 EdgeLLM 提供 `--bayes_optimize`。正式脚本显式使用原项目设置：
HumanEval `R1=0.92`、GSM8K `R1=0.5`，可用 `EDGELLM_INIT_ALPHA` 覆盖；更新衰减固定为
论文式的 `0.5`，可用 `EDGELLM_FULL_ACCEPT_DECAY` 显式复现实验变体。

## 3. 两个数据集、四种方法正式测评

每次脚本按 Vanilla、HSL、EdgeLLM、PipeSD 顺序各生成一个独立的 1000-token 结果。

```bash
cd edge
SEED=1234 TARGET_OUTPUT_TOKENS=1000 RESULT_TAG=table1_s1_r1 bash scripts/eval_humaneval.sh
SEED=1234 TARGET_OUTPUT_TOKENS=1000 RESULT_TAG=table1_s1_r1 bash scripts/eval_gsm8k.sh
```

至少重复三次，改变 tag 和 seed，并轮换四种方法的执行顺序以减小热状态/时间顺序偏差。
正式参数为：

| 数据集 | Vanilla draft length | HSL R2 | EdgeLLM | PipeSD |
|---|---:|---:|---|---|
| HumanEval | 6 | 0.99 | 初始 R1=0.92，论文动态更新，等待 NAV 继续生成 | 数据集独立 BO 的 R1/R2 |
| GSM8K | 4 | 0.7 | 初始 R1=0.5，论文动态更新，等待 NAV 继续生成 | 数据集独立 BO 的 R1/R2 |

论文场景 1 TPT（ms/token）为：HumanEval `194/155/153/129`，GSM8K
`193/174/169/145`，顺序均为 Vanilla/HSL/EdgeLLM/PipeSD。

## 4. 汇总性能和正确性

```bash
cd edge
python scripts/summarize_table1.py exp/exp__wjl \
  --output table1_summary.json \
  --humaneval-jsonl humaneval_completions.jsonl
```

汇总器输出每方法 TPT 均值/标准差、PipeSD speedup、与论文值的相对误差、tokens、验证频率、
平均 draft length、接受率、rollback rate，以及 GSM8K exact match。HumanEval completion 会导出
为 JSONL；按 HumanEval 官方 evaluator 运行 pass@1。为避免把不同方法混在同一 pass@1 中，
实际评估时按 `method` 和 `run_id` 拆分该 JSONL。

先跑 100-token pilot：

```bash
TARGET_OUTPUT_TOKENS=100 RESULT_TAG=pilot bash scripts/eval_humaneval.sh
TARGET_OUTPUT_TOKENS=100 RESULT_TAG=pilot bash scripts/eval_gsm8k.sh
```

检查 `summary.actual_output_tokens == target_output_tokens`，并人工查看 PipeSD 的 `batch_trace`：
等待 NAV 阶段应出现多 token batch；`num_spec_tokens_generated >= num_spec_tokens_sent`；回滚后
云端 discarded/reused proactive 计数应与 edge 行为一致。通过后再跑 1000 tokens × 3 repeats。
