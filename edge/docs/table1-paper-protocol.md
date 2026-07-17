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
- Scenario 1 的上行参数是 2.5 MB/s、下行是 25 MB/s。`os` 模式使用 tc/QoS；
  无系统权限时可使用共享 FIFO 的 `software` 模式。software 会在 HTTP POST 前模拟上传，
  所有主请求和 proactive 请求共享同一个上行链路，并在响应返回后模拟下行。
- 每个正式结果包含 manifest、run id、Git commit/dirty 状态、数据和 draft 模型 SHA-256、
  样本顺序、DP/网络估计、batch trace、验证/接受/rollback/proactive/能耗统计。

## 1. 启动云端

在云端仓库的 `cloud/` 下：

```bash
export GPU_POWER_SAMPLE_INTERVAL=0.005
python -m src.speculative_server --dataset humaneval
```

HumanEval 完成后停止服务，再用 `python -m src.speculative_server --dataset gsm8k` 重启；两个数据集
使用不同 target model。cloud 端代码固定 `n_gpu_layers=-1`，会把 target model 全部放到 GPU。

边端设置服务地址：

```bash
export PIPE_SD_SERVER_URL=http://CLOUD_HOST:8000
```

若云端和边端不在论文同等网络中，分别限制：edge -> cloud 20 Mbps，cloud -> edge 200 Mbps。
有系统权限时可以使用 `os`；没有权限时使用下述 software 配置，并用 1--8 token 通信探针
检查回归的 alpha、beta、R2：

```bash
export NETWORK_SHAPING_MODE=software
export BANDWIDTH_MBPS=2.5
export DOWNLINK_BANDWIDTH_MBPS=25
export SOFTWARE_UPLINK_STARTUP_MS=25
export SOFTWARE_DOWNLINK_STARTUP_MS=0
```

论文没有报告固定 RTT，25 ms 在这里对应论文通信模型中的每批 startup overhead `alpha`，
不是声称论文 RTT 为 25 ms。如有真实链路测量，应分别用
`SOFTWARE_UPLINK_STARTUP_MS` 和 `SOFTWARE_DOWNLINK_STARTUP_MS` 校准。

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
没有上述权限时不要运行 tc 脚本，保持 `NETWORK_SHAPING_MODE=software` 即可。

software 模式的每条样本结果会保存 `environment_measurements.software_link`、
`primary_sender` 和 `proactive_sender`。应检查两个 sender 的上传最终只汇入同一个
`software_link.totals.uplink`，且每条请求包含 `link_queue_wait_seconds`、
`upload_seconds`、`http_seconds` 和 `download_seconds`。

## 2. 每个数据集独立运行 PipeSD BO

```bash
cd edge
NETWORK_SHAPING_MODE=software SEED=1234 BO_TOKENS_PER_TRIAL=20 bash scripts/bo_humaneval.sh
NETWORK_SHAPING_MODE=software SEED=1234 BO_TOKENS_PER_TRIAL=20 bash scripts/bo_gsm8k.sh
```

每个数据集建议用 3 个 seed 重复 BO。每次 BO 会在对应的
`exp/exp__wjl/<dataset>/pipesd/latest_bayes_best.json` 写入稳定配置；正式评测脚本自动读取，
配置缺失或不含 `shared-fifo-v1` 网络来源时直接失败。因此必须重新运行 BO，不能沿用修复前
software 模式生成的 `latest_bayes_best.json`。论文只说明 EdgeLLM 的初始 `R1` 按实验设置选择，没有给出 BO
流程，因此代码不再为 EdgeLLM 提供 `--bayes_optimize`。正式脚本显式使用原项目设置：
HumanEval `R1=0.92`、GSM8K `R1=0.5`，可用 `EDGELLM_INIT_ALPHA` 覆盖；更新衰减固定为
论文式的 `0.5`，可用 `EDGELLM_FULL_ACCEPT_DECAY` 显式复现实验变体。

## 3. 两个数据集、四种方法正式测评

每次脚本按 Vanilla、HSL、EdgeLLM、PipeSD 顺序各生成一个独立的 1000-token 结果。

```bash
cd edge
NETWORK_SHAPING_MODE=software SEED=1234 TARGET_OUTPUT_TOKENS=1000 RESULT_TAG=table1_s1_r1 bash scripts/eval_humaneval.sh
NETWORK_SHAPING_MODE=software SEED=1234 TARGET_OUTPUT_TOKENS=1000 RESULT_TAG=table1_s1_r1 bash scripts/eval_gsm8k.sh
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
  --network-implementation current_software \
  --result-tag table1_s1_paper \
  --bandwidth-mbps 2.5
```

汇总器针对每个数据集选择同一 result tag、带宽和网络实现下每种算法最新的正式结果，分别输出到
`exp/exp__wjl/<dataset>/comparison/`。输出包含 Markdown、CSV、JSON，以及每种算法独立的
completion JSONL。报告覆盖 TPT、吞吐、P50/P95/P99、TTFT、GPU 能耗、样本波动、NAV、
draft length、接受率、rollback、batch、proactive 复用/废弃、网络流量/排队时间、终止状态、
可比性警告；completion 只保存，不计算 pass@1 或 exact match。由于复现实验和论文使用的机器不同，
汇总报告不直接比较绝对 TPT，也不计算相对论文值的误差。

先跑 100-token pilot：

```bash
TARGET_OUTPUT_TOKENS=100 RESULT_TAG=pilot bash scripts/eval_humaneval.sh
TARGET_OUTPUT_TOKENS=100 RESULT_TAG=pilot bash scripts/eval_gsm8k.sh
```

检查 `summary.actual_output_tokens == target_output_tokens`，并人工查看 PipeSD 的 `batch_trace`：
等待 NAV 阶段应出现多 token batch；`num_spec_tokens_generated >= num_spec_tokens_sent`；回滚后
云端 discarded/reused proactive 计数应与 edge 行为一致。通过后再跑 1000 tokens × 3 repeats。

同时验证 software 链路的共享传输不变量和通信回归：

```bash
python scripts/validate_software_results.py exp/exp__wjl \
  --output software_link_validation.json
```

旧 software 结果没有 `shared-fifo-v1` manifest，会列入 `legacy_skipped`；加 `--strict-legacy`
可以让它们直接导致校验失败。修复前后的结果不会混合汇总。

## 5. Software 模式复现 Scenario 4

论文只给出上下行变化范围和 20 秒间隔，没有公布完整带宽轨迹。为了保证可重复性，必须显式
记录固定轨迹。例如以下上行均位于 10--80 Mbps、下行均位于 150--280 Mbps：

```bash
export NETWORK_SHAPING_MODE=software
export SOFTWARE_BANDWIDTH_PROFILE="1.25:18.75,2.5:25,5:30,10:35"
export SOFTWARE_BANDWIDTH_CHANGE_INTERVAL_S=20
RESULT_TAG=scenario4_trace_a bash scripts/eval_humaneval.sh
RESULT_TAG=scenario4_trace_a bash scripts/eval_gsm8k.sh
```

profile 单位为 MB/s，会循环执行；当前索引和完整轨迹都会写入结果 manifest。由于论文未公开
原始随机带宽序列，该结果应称为“遵循论文范围的 Scenario 4 协议复现”，不能宣称逐点复现。
