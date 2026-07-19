# PipeSD Reproduction Progress Report: Slide Content and Speaker Notes

> **Purpose:** This document is designed for an 18–22 minute progress presentation based on the paper, the current repository, and the available evaluation results.  
> **Structure:** 18 main slides and 4 appendix slides. Every slide includes titled sections for slide content, visual design, and speaker notes.  
> **Paper:** [PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](https://arxiv.org/abs/2605.13319)

## Presentation Scope and Terminology

- This work is a **reproduction of the protocol and core mechanisms**, not a complete reproduction of every result in the paper. The current work primarily reproduces the four-algorithm protocol under a Scenario 1–style setting and adds an extended comparison among edge-only, cloud-only, serial cloud-edge, and PipeSD execution.
- The absolute latency in the current experiments is substantially higher than that reported in the paper. The correct claim is: **the core PipeSD mechanisms have been reproduced, and relative performance gains have been observed in the current environment**. It would be inaccurate to claim that the paper's absolute numbers have been reproduced.
- The Git author name found in the repository is `Nuyoahwjl`, rather than `Nuyoshwjl`. This report uses the actual Git author name and covers eight relevant commits.
- Cloud-only covers a warm target request from preprocessing and prompt prefill through autoregressive decoding and detokenization, but excludes model load and client-cloud transfer. Edge-only uses the smaller draft model, so its output quality is not target-equivalent. Pure modes normalize by committed output tokens, whereas collaborative modes normalize by cloud-accepted draft tokens; the four-mode experiment is therefore an overhead decomposition, not a direct ranking of four equal-quality services under one denominator.
- The current energy metric covers **cloud GPU energy only**. Edge-side energy was not measured with RAPL or an external power meter, so the results do not support conclusions about total end-to-end system energy.

---

## Slide 1 — Title

### Slide Content

**Title:** Reproducing PipeSD for Cloud-Edge Collaborative Speculative Decoding  
**Subtitle:** From Paper Mechanisms and Code Implementation to Four-Algorithm and Four-Mode Evaluations  
**Footer:** Presenter / Date / Project Repository

### Suggested Visual

Use the workflow from the paper as a faded background, or use a thumbnail of the multi-round pipeline generated from the current experiment:

![PipeSD multi-round pipeline](./pipesd-humaneval-trace-multi-round.png)

### Speaker Notes

This presentation reports the current progress in reproducing PipeSD. I will first explain why a cloud-edge pipeline is needed and how dynamic-programming batch scheduling, dual-threshold NAV triggering, and Bayesian optimization address different bottlenecks. I will then map these mechanisms to the current implementation. Finally, I will present two groups of experiments: a comparison among Vanilla, HSL, EdgeLLM, and PipeSD, followed by an overhead decomposition across edge-only, cloud-only, serial cloud-edge, and PipeSD modes. I will also state clearly which parts have been reproduced and which parts remain incomplete.

---

## Slide 2 — Executive Summary of the Current Progress

### Slide Content

**Completed Work**

- An end-to-end PipeSD loop covering edge-side drafting, generation-transmission overlap, cloud-side NAV, accepted-token return, and round-to-round continuation.
- Dynamic-programming token-batch scheduling, dual-threshold triggering, 16-trial Bayesian optimization, online environment estimation, and a shared-FIFO software link.
- A Scenario 1–style, 1,000-token evaluation of four algorithms, with relative PipeSD gains on both tasks.
- An extended comparison among edge-only, cloud-only, serial cloud-edge, and PipeSD modes, together with a trace-based multi-round pipeline visualization.

**Key Results**

- **HumanEval:** PipeSD is **1.249×** faster than Vanilla and reduces cloud GPU energy by **36.4%**.
- **GSM8K:** PipeSD is **1.218×** faster than Vanilla, **1.088×** faster than the best baseline, HSL, and reduces cloud GPU energy by **23.6%**.

**Incomplete Work**

- Complete formal results for Scenarios 2–4, bandwidth sweeps, ablations, BO comparisons, correctness metrics, and confidence intervals from repeated runs.
- Reproduction of the paper's exact hardware, real metropolitan network, and absolute performance numbers.

### Suggested Visual

Use a three-column layout: Completed Work / Key Results / Incomplete Work. Display the key speedup and energy numbers in large type.

### Speaker Notes

The core protocol and evaluation path are now operational. PipeSD is 1.249× faster than Vanilla on HumanEval and 1.218× faster on GSM8K, while reducing measured cloud GPU active-compute energy by 36.4% and 23.6%, respectively. However, “completed” here refers to the availability of the core mechanisms, concurrency control, and an executable evaluation workflow. It does not mean that every table and figure in the paper has been reproduced. The most important remaining limitations are the mismatch in the testbed, incomplete formal experiment coverage, single-run results, and the absence of task-level accuracy evaluation.

---

## Slide 3 — Motivation: Why Conventional Cloud-Edge Speculative Decoding Is Still Slow

### Slide Content

A conventional cloud-edge speculative decoding round usually follows four serial steps:

1. The edge draft model generates a block of draft tokens.
2. The block is uploaded only after generation is complete.
3. The cloud target model performs Non-Autoregressive Verification, or NAV.
4. The edge waits for the verification result before starting the next round.

**Two Main Bottlenecks**

- **Serial generation and communication:** the edge processor is idle during upload, while the network is idle during draft generation.
- **Inflexible fixed-length or single-threshold NAV triggering:** triggering too early repeatedly pays communication startup cost, while triggering too late accumulates low-quality tokens and increases rollback.

### Suggested Visual

Draw two timelines. The conventional timeline should show `Generate → Upload → NAV → Wait`. The PipeSD timeline should show overlapping generation, upload, and speculative continuation blocks.

### Speaker Notes

PipeSD is not merely a system that places a small model at the edge and a large model in the cloud. Its main objective is to keep heterogeneous compute and communication resources busy at the same time. In a conventional workflow, draft generation, network transmission, and cloud verification are mostly serial, so resources alternate between busy and idle states. NAV triggering also creates a trade-off: frequent verification incurs repeated startup overhead, while delayed verification risks producing many tokens that must later be rolled back. PipeSD addresses these two problems separately: dynamic-programming batch scheduling overlaps generation with transmission, while dual-threshold triggering and Bayesian optimization determine when verification should occur.

---

## Slide 4 — Speculative Decoding and the Role of NAV

### Slide Content

Let the edge draft model generate the candidate sequence

\[
D=(d_1,d_2,\ldots,d_n).
\]

The cloud target model checks this sequence with one parallel forward pass:

- It accepts the longest prefix that satisfies the target-model verification rule.
- It produces one additional target token after the accepted prefix.
- If a rejection occurs, all draft tokens after the rejection point become invalid.
- The context of the next round must be based on the cloud-confirmed sequence.

**Critical Promotion Condition**

Edge-side work generated in advance can be promoted to the next formal candidate only if the target model accepts the entire parent draft and its additional target token matches the token assumed by the proactive branch.

### Suggested Visual

Use a simple token example. The edge proposes `A B C D`; the cloud accepts `A B`, rejects `C`, and returns `X`. The confirmed context becomes `A B X`, while `C`, `D`, and their descendants are discarded.

### Speaker Notes

NAV allows the target model to verify a batch of draft tokens in one pass instead of invoking the large model once per output token. The target model also produces one continuation token after the accepted prefix, ensuring that the sequence always makes progress. PipeSD attempts to hide NAV waiting time by continuing to generate and even upload candidates for the next round. These candidates are not yet confirmed. If the cloud accepts only part of the parent draft, or if it accepts the full draft but returns a continuation token different from the one assumed by the edge, the proactive branch has the wrong context and must be discarded. The current implementation explicitly tracks parent-round and parent-token metadata to enforce this condition.

---

## Slide 5 — PipeSD System Architecture

### Slide Content

**Edge Components**

- **Draft Model:** autoregressively produces candidate tokens and confidence values.
- **Transmission Controller:** combines the DP batch scheduler and the dual-threshold NAV trigger.
- **Environment Monitor:** estimates token-generation time \(\gamma\), communication startup cost \(\alpha\), and per-token communication time \(\beta\).
- **Parameter Updater:** uses Bayesian optimization to update the thresholds and recomputes DP batch boundaries when the environment changes.
- **Communication Interface:** uploads draft batches and receives NAV results.

**Cloud Components**

- **Communication API:** receives candidate batches and their round metadata.
- **Target Model:** performs NAV and returns the accepted prefix plus one additional target token.

### Suggested Visual

Draw two large boxes labeled Edge and Cloud, connected with bidirectional arrows. Highlight the Transmission Controller as the core component on the edge side.

### Speaker Notes

Most scheduling decisions are made by the edge-side Transmission Controller. It uses the current generation and communication rates to divide a predicted token window into transmission batches. At the same time, it monitors token confidence and cumulative sequence confidence to determine when NAV should be triggered. The Environment Monitor continually updates timing parameters so that the scheduler does not assume a permanently fixed network. The cloud side is more focused: it receives and assembles batches by round, verifies them with the target model, and returns the confirmed sequence information.

---

## Slide 6 — Core Focus: How One PipeSD Round Runs as a Pipeline

### Slide Content

**Pipeline Stages in One Speculative Round**

1. **Draft Generation:** the edge generates tokens one by one and records their confidence.
2. **Token-Batch Scheduling:** once the current DP boundary is reached, the completed batch is uploaded asynchronously while generation continues.
3. **Dual-Threshold Triggering:** after every token, the system checks both token-level and cumulative confidence; crossing either threshold triggers NAV.
4. **Cloud NAV:** the cloud verifies the uploaded candidates and returns accepted tokens plus one additional target token.
5. **Speculative Continuation:** while waiting for NAV, the edge generates and may upload next-round candidates into an isolated proactive buffer.
6. **Promote or Discard:** proactive data is promoted to the next formal candidate only when the parent draft is fully accepted and the additional token matches; otherwise it is cancelled and discarded.

### Suggested Visual

Use the generated trace as the main visual:

![HumanEval Task 50, first 12 trigger windows](./pipesd-humaneval-trace-multi-round.png)

Use a consistent legend: blue for draft generation, orange for upload batches, red for NAV, green for accepted or promoted work, and gray for discarded proactive work.

### Speaker Notes

This is the central slide of the presentation. PipeSD does not wait for the entire draft block before transmission. It divides the block at DP-selected boundaries, allowing small uploads to overlap with continued generation. After NAV is triggered, the edge also avoids becoming idle: it uses an isolated buffer to speculate on the next round. When the cloud response arrives, the system decides whether that work can be committed. The benefit comes from hiding communication and NAV waiting time, but correctness requires that speculative data remain isolated until validation. This figure is generated from the formal experiment result rather than from synthetic example data.

---

## Slide 7 — Dynamic-Programming Token-Batch Pipeline Scheduling

### Slide Content

The paper models the transmission time of a batch of size \(b\) as

\[
T_{comm}(b)=\alpha+\beta b,
\]

and approximates the generation time of \(b\) tokens as

\[
T_{gen}(b)=\gamma b.
\]

**Scheduling Objective**

Choose batch boundaries within a predicted window \(\hat N\) to minimize the completion time of the final upload.

- A batch can be sent only after all tokens in that batch have been generated and the previous transmission has completed.
- Dynamic programming enumerates the previous partition point with complexity \(O(\hat N^2)\).
- \(\hat N\) starts at 20 and is later updated using the moving average of draft lengths from the most recent 100 rounds.
- If NAV returns early, the active schedule is interrupted and unsent tokens are cleared.

### Suggested Visual

Show a token window `[1…N]` partitioned into batches such as `[1,2] [3,4,5] [6,…]`. Under it, use a Gantt chart to show the overlap between generation and transmission.

### Speaker Notes

Sending every token immediately would repeatedly pay the fixed startup cost alpha. Sending the entire draft as one batch would avoid repeated startup cost but eliminate generation-transmission overlap. The batch size must therefore balance these two costs. The DP scheduler uses the estimated alpha, beta, and gamma values to select the boundaries that minimize the predicted finish time. The implementation also updates the environment model only after a meaningful change, using a 20% gate to avoid excessive rescheduling under small fluctuations. In the formal results, the average PipeSD batch size is 1.670 for HumanEval and 1.371 for GSM8K, showing that a longer draft is actually transmitted through multiple fine-grained pipeline batches.

### Implementation Mapping

- [`edge/src/merge.py`](../edge/src/merge.py): `PaperDPScheduler`, predicted-window updates, and online environment estimation.
- [`edge/src/engine.py`](../edge/src/engine.py): asynchronous sending at batch boundaries, NAV interruption, and pipeline trace recording.

---

## Slide 8 — Dual-Threshold NAV Triggering and Bayesian Optimization

### Slide Content

For the \(n\)-th draft token:

- Token-level confidence: \(P(d_n)\)
- Cumulative sequence confidence: \(C_n=\prod_{i=1}^{n}P(d_i)\)

NAV is triggered when

\[
C_n\le R_1 \quad \text{or}\quad P(d_n)\le R_2.
\]

**Interpretation**

- The token threshold detects a sudden local confidence drop.
- The cumulative threshold detects the growing risk created by several individually plausible tokens.
- Bayesian optimization searches for \((R_1,R_2)\) that minimizes average TPT under the current model, device, and network.
- The current implementation follows the paper's 16-evaluation budget and uses Expected Improvement with `xi=0.1`.

### Suggested Visual

Plot token confidence and cumulative confidence against token index. Add two threshold lines and mark the point where either trigger condition becomes true.

### Speaker Notes

A token-only trigger misses cases in which several tokens each have acceptable confidence but the probability that the entire sequence is correct has already become low. A cumulative-only trigger is less direct when one token experiences a sudden confidence collapse. PipeSD therefore combines the two conditions with an OR rule. Since the best thresholds depend on the task, models, hardware, and network, the paper uses lightweight Bayesian optimization instead of fixed manual constants. The BO mechanism and isolated trials have been implemented, but its experimental reproduction is not yet complete: the final formal runs are not fully linked to the latest BO output, and the paper's BO-versus-Grid-versus-Random comparison has not yet been reproduced.

### Implementation Mapping

- [`edge/src/engine.py`](../edge/src/engine.py): dual-threshold decision logic.
- [`edge/app/run_edge.py`](../edge/app/run_edge.py): BO objective, 16-trial budget, and trial isolation.

---

## Slide 9 — Correctness and Data Lifecycle in a Concurrent Pipeline

### Slide Content

| Cloud Verification Result | Can Proactive Tokens Be Promoted? | Required Action |
|---|---:|---|
| Only part of the parent draft is accepted | No | Roll back tokens after the rejection point and cancel or discard next-round proactive data |
| The full draft is accepted, but the additional target token does not match | No | The proactive context is incorrect; discard it and restart from the target token |
| The full draft is accepted and the additional token matches | Yes | Promote the isolated proactive buffer to the next formal round |

**Current Safety Mechanisms**

- Every batch carries round, parent-round, parent-final-token, and related metadata.
- Proactive and formal batches use separate logical buffers.
- Both channels share one physical `SoftwareLink`, so proactive traffic does not receive fictional extra bandwidth.
- The cloud promotes a buffer only after parent-round verification; otherwise it discards the buffer.

#### Lifecycle of Invalidated Proactive Tokens

When a proactive branch becomes invalid after the NAV result arrives,
its cleanup depends on how far the corresponding tokens have progressed
through the transmission pipeline.

| Token State                                           | Handling                                                     |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| Generated locally but not yet enqueued as a batch     | The tokens exist only in edge memory and are discarded directly. |
| Submitted but still waiting in the transmission queue | The edge performs best-effort cancellation so that the batch is not uploaded. |
| Already being transmitted or buffered at the cloud    | The edge can no longer retract the batch. The cloud validates its parent-round metadata and either retains or discards it. |

The semantic validity rule is identical in all three cases: proactive
tokens can be retained only when the parent draft is fully accepted and
the target model's additional token matches the token assumed by the
proactive branch.

### Suggested Visual

Draw three branches: partial acceptance, full acceptance with mismatch, and full acceptance with match. Only the third branch should lead to a green Commit state.

### Speaker Notes

This is the most common place where an implementation may appear faster while being semantically incorrect. Proactive tokens cannot be promoted unconditionally. The parent round must be fully accepted, and the target model's additional token must equal the context token assumed by the proactive branch. The current edge and cloud implementations both validate this metadata and clear the isolated buffers on failure. Proactive upload is allowed, but it represents unconfirmed work and never bypasses NAV. Adding this lifecycle control was one of the most important correctness improvements over earlier versions of the reproduction.

### Implementation Mapping

- [`edge/src/engine.py`](../edge/src/engine.py): proactive sender, promotion, cancellation, and discard logic.
- [`cloud/src/speculative_server.py`](../cloud/src/speculative_server.py): parent validation and buffer promotion or discard.
- [`edge/src/software_link.py`](../edge/src/software_link.py): shared-FIFO network emulation in both directions.

---

## Slide 10 — End-to-End Mapping from the Paper to the Current Code

### Slide Content

| Paper Component | Current Implementation | Reproduction Status |
|---|---|---|
| Edge draft autoregressive generation | `edge/src/engine.py` | Implemented |
| Token-batch DP scheduler | `edge/src/merge.py` | Aligned with the paper's logic |
| Dual-threshold NAV trigger | `edge/src/engine.py` | Implemented |
| BO parameter updater | `edge/app/run_edge.py` | Implemented; formal-run provenance remains incomplete |
| Environment monitor | `merge.py` and trace/result fields | Implemented; Figure 6 comparison not completed |
| Cloud NAV | `cloud/src/speculative_server.py` | Implemented |
| Dynamic software network | `edge/src/software_link.py` | Implemented |
| Four-mode evaluation | `edge/src/pure_baseline.py` and evaluation scripts | Repository extension; evaluated |

### Suggested Visual

Place the paper architecture on the left and the corresponding source files on the right. Connect each conceptual module to its implementation with arrows.

### Speaker Notes

The core control path is distributed mainly across `engine.py`, `merge.py`, and `speculative_server.py`. `run_edge.py` manages experiment budgets, BO, and result manifests, while `software_link.py` constrains upload and download traffic through a shared physical-link abstraction. The four-mode comparison is not part of the paper's four-algorithm Table 1. It was added to answer a separate systems question: whether the measured benefit comes from the model placement, the network path, or actual pipeline overlap.

---

## Slide 11 — Reproduction Work: Nuyoahwjl Commit Timeline

### Slide Content

| Commit | Main Work | Reproduction Value |
|---|---|---|
| `c3a904d` | Standardized GSM sweep naming and experiment output directories | Improved result organization |
| `4c8a975` | Added results for four algorithms | Established a Table 1–style comparison |
| `4915a13` | Aligned DP scheduling and BO | Reproduced core methods |
| `c4a8b5e` | Isolated BO trials and bootstrapped communication regression | Prevented cross-trial contamination |
| `5a7f3f1` | Enforced exact token and generation budgets | Improved the 1,000-token evaluation protocol |
| `9375fa1` | Aligned true concurrency, proactive metadata, and evaluation rules | Corrected pipeline semantics and comparability |
| `3902f28` | Added a shared-FIFO software network and dynamic bandwidth support | Improved network-model fidelity |
| `f27a203` | Added four-mode evaluation and consolidated reporting | Enabled system-overhead decomposition |

### Suggested Visual

Use a horizontal timeline and group the commits into four phases: algorithm alignment, protocol correctness, network emulation, and extended evaluation.

### Speaker Notes

The repository already contained a basic PipeSD framework, so the work should not be presented as building the entire system from scratch. The contribution of these commits is the alignment, completion, and validation of the paper's protocol. Early commits established four-algorithm results and experiment organization. Later commits aligned DP, BO, trial isolation, and token budgets. The most important subsequent work addressed true concurrency, safe proactive-data promotion, a shared network link, and traceable result manifests. The final commit added the four-mode evaluation. The key achievement is therefore not merely that the code runs, but that important questions about concurrency, correctness, resource sharing, and evaluation consistency are now represented explicitly in the implementation.

---

## Slide 12 — Reproduction and Evaluation on One RTX PRO 6000

### Slide Content

**Deployment and Model Placement: Logical Cloud-Edge on One Host**

| Role | Reproduction Setup | Model and Inference Configuration |
|---|---|---|
| Cloud / target | One **NVIDIA RTX PRO 6000**; local FastAPI service at `127.0.0.1:8000`, one worker | HumanEval: DeepSeek-Coder-6.7B-Instruct Q4_K_M; GSM8K: Llama-2-7B-Chat Q4_K_M; `llama-cpp-python`, `n_gpu_layers=-1`, all target layers on the GPU, `n_ctx=16384` |
| Edge / draft | Same workstation as the cloud process, but the draft model runs explicitly on the **CPU** | HumanEval: DeepSeek-Coder-1.3B-Instruct Q4_K_M; GSM8K: TinyLlama-1.1B-Chat-v1.0 Q4_K_M; `n_gpu_layers=0`, two CPU threads, `n_ctx=16384`, `temperature=0`, `top_k=1` |

“Single-GPU evaluation” therefore means that target-model NAV uses one RTX PRO 6000; the draft model does not share that GPU. This reproduces the logical processes, protocol, and resource placement rather than a physically remote cloud-edge deployment. The paper instead used a Core Ultra 9 185H edge CPU, an A800 40 GB cloud GPU, and a real metropolitan network, so absolute TPT values are not directly comparable ([paper §5.1](https://arxiv.org/pdf/2605.13319#page=6)).

**How Each Paper Module Is Implemented**

| Paper Module | Repository Implementation |
|---|---|
| Draft Model | `edge/src/engine.py` performs autoregressive token-by-token GGUF decoding and derives each token confidence `P(Dn)` from the logits. |
| Dual-Threshold NAV Trigger | `if_verify(..., "hybrid")` computes both the latest token confidence and the product of sequence confidences; NAV fires when `P(Dn) < R2` **or** `product(P(Di)) < R1`. |
| Token-Batch Pipeline Scheduler | `edge/src/merge.py` solves the paper's DP recurrence for the minimum-completion-time batch plan. It starts with `N-hat=20`, `alpha=25 ms`, `beta=0.29/2.5 s/token`, and `gamma=0.036 s/token`, then updates from the latest 100 generation/communication records. Only PipeSD uploads DP batches before NAV and continues doing so while NAV is pending. |
| Communication and Proactive Continuation | Primary and proactive uploads use separate workers but share one `SoftwareLink`. Requests carry round/window/batch/index/prefix metadata. Pending tokens are promoted to the next formal candidate only when the parent round is fully accepted and the target extra token matches the edge's predicted token; otherwise both sides discard the branch and roll back. |
| Cloud NAV | `cloud/src/speculative_server.py` exposes `/init`, `/start`, `/propose`, and `/exit`. The target verifies a whole draft block in one forward pass, accepts with `min(1,p/q)`, samples a correction from the positive target-minus-draft distribution at the first rejection, and produces one extra target token after full acceptance. |
| Environment Adaptation and BO | One-to-eight-token probes fit communication `alpha/beta`; draft `gamma` is measured online; a change above 20% triggers DP replanning. BO implements `(R1,R2) in (0,1)^2`, 16 calls, one random initial point, a Matern GP, EI with `xi=0.1`, and 20 accepted tokens per candidate. However, the formal results on this slide use fixed script values and record an empty `bo_config_path`, so they cannot be presented as thresholds selected by a provenance-linked BO run. |

**Actual Four-Algorithm Parameters**

| Dataset | Vanilla | HSL | EdgeLLM | PipeSD |
|---|---|---|---|---|
| HumanEval | fixed `N=6` | token threshold `R2=0.99` | initial sequence threshold `R1=0.92`, multiply by `0.5` after full acceptance, continue during NAV | `R1(sequence)=0.90`, `R2(token)=0.3514`, DP merge |
| GSM8K | fixed `N=4` | token threshold `R2=0.70` | initial sequence threshold `R1=0.50`, multiply by `0.5` after full acceptance, continue during NAV | `R1(sequence)=0.65`, `R2(token)=0.40`, DP merge |

**How Network Conditions Are Emulated**

- The paper's Scenario 1 rates of 20 Mbps uplink and 200 Mbps downlink are converted to `2.5 MB/s` and `25 MB/s`. The application-level `SoftwareLink` shapes traffic before and after the real loopback HTTP request.
- Transfer service time is `startup + serialized_bytes / bandwidth`: uplink startup is `25 ms`, while downlink startup is `0 ms`. The 25 ms value is the reproduction's per-batch communication startup `alpha`; it is **not a paper-reported RTT**.
- Uplink and downlink use independent FIFOs, providing full duplex. All primary and proactive traffic in the same direction shares one queue, preventing the two upload workers from receiving separate 20 Mbps virtual links. Real loopback HTTP time and target computation remain in end-to-end latency.
- This slide reports a fixed-bandwidth, no-artificial-edge-delay Scenario-1-style run with `enable_compute_emulation=false`. Scenario 2/3 per-token compute delays and the 20-second Scenario 4 dynamic bandwidth profile are not used in these formal results.

**Evaluation Procedure and Measurement Boundaries**

1. Start or restart the cloud service for each dataset so that it loads the correct target; then run Vanilla, HSL, EdgeLLM, and PipeSD from the edge. HumanEval starts at sample 50 and GSM8K at sample 100; each sample is capped at 128 generated tokens.
2. The edge manifest records seed `3407`. Each method keeps consuming samples until the cloud has accepted exactly **1,000 draft tokens**; the stopping rule is not a fixed sample count. The current artifacts use about eight HumanEval samples and eight to nine GSM8K samples.
3. In `comparison`, TPT is `sum(end-to-end time) / sum(cloud-accepted draft tokens)`. Throughput, NAV/100, and GPU J/100 use the same accepted-draft-token normalization; P50/P95/P99 and TTFT describe committed output tokens.
4. The cloud samples GPU board power through NVML every 5 ms. Energy includes prompt prefill and active target NAV compute only, excluding model load, GPU idle between NAVs, the edge CPU, network devices, and whole-system power.
5. Each run saves raw samples, completions, batch/NAV/proactive traces, network queue statistics, a manifest, and SHA-256 values. `edge/scripts/summarize_table1.py` then emits Markdown, CSV, and JSON reports.

### Suggested Visual

Use one left-to-right reproduction path: `CPU draft -> dual threshold / DP -> shared-FIFO software link -> FastAPI -> RTX PRO 6000 target NAV`. Under it, place the four-algorithm parameter table and three prominent labels: “20/200 Mbps,” “alpha=25 ms,” and “1,000 accepted draft tokens.”

### Speaker Notes

This setup does not place both cloud and edge models on the same GPU. The edge draft uses `n_gpu_layers=0` and generates tokens on the CPU, while the cloud target uses `n_gpu_layers=-1` and performs NAV entirely on the single RTX PRO 6000. The two processes run on the same machine and communicate through `127.0.0.1:8000`, so the paper's 20/200 Mbps Scenario 1 link is reproduced with an application-level emulator. The emulator waits for modeled upload time before the HTTP request can reach the cloud and waits for modeled download time before the response reaches the edge. Both upload workers share the same uplink FIFO. This preserves real target computation while making bandwidth and per-batch startup controlled, repeatable variables.

Each task begins with `/init`, which performs target prompt prefill. The edge then drafts one token at a time. Vanilla and HSL do not upload before NAV and send the whole block when triggered. EdgeLLM continues generating only while NAV is pending. PipeSD follows the DP batch plan both before NAV and while NAV is pending. When the cloud receives `should_verify=true`, it performs NAV and returns the accepted length and a target extra token. A pending branch is promoted only when the parent block is fully accepted and the extra token matches; otherwise the branch is discarded and decoding resumes from the confirmed prefix. This gate prevents pipeline overlap from committing tokens that the target has not validated.

For a clean reproduction, start the cloud separately for each dataset and explicitly use the same seed:

```bash
cd cloud
GPU_POWER_SAMPLE_INTERVAL=0.005 python -m src.speculative_server --dataset humaneval --seed 3407
# Stop after HumanEval, then start:
GPU_POWER_SAMPLE_INTERVAL=0.005 python -m src.speculative_server --dataset gsm8k --seed 3407
```

Then, from `edge/`, freeze the link, token budget, and result tag. Run a 100-token pilot first, inspect the accepted-token budget, batch trace, proactive promotion/discard counts, and shared-link totals, and only then run the formal 1,000-token evaluation and aggregation:

```bash
export PIPE_SD_SERVER_URL=http://127.0.0.1:8000
export NETWORK_SHAPING_MODE=software BANDWIDTH_MBPS=2.5 DOWNLINK_BANDWIDTH_MBPS=25
export SOFTWARE_UPLINK_STARTUP_MS=25 SOFTWARE_DOWNLINK_STARTUP_MS=0
SEED=3407 TARGET_OUTPUT_TOKENS=1000 RESULT_TAG=table1_s1_paper bash scripts/eval_humaneval.sh
SEED=3407 TARGET_OUTPUT_TOKENS=1000 RESULT_TAG=table1_s1_paper bash scripts/eval_gsm8k.sh
python scripts/summarize_table1.py exp/exp__wjl --network-implementation current_software --result-tag table1_s1_paper --bandwidth-mbps 2.5
```

The current repository state also needs to be stated explicitly. The HumanEval script still runs all four methods in sequence, but the current `eval_gsm8k.sh` has the first three invocations commented out and therefore runs only PipeSD; the existing GSM8K four-method comparison uses separately saved formal artifacts. A clean rerun must restore those blocks or execute their commands individually. There is only one matching run per method, some artifacts were produced from a dirty worktree, the target-model hash and cloud seed are not recorded in the edge manifest, and the formal PipeSD artifacts do not record a BO configuration path. HumanEval pass@1 and GSM8K exact match have also not been computed. The slide therefore supports a claim about performance and protocol behavior on a controlled single-host emulated link, not a complete reproduction of the paper's absolute numbers or final task accuracy.

### Result Sources

- [HumanEval four-algorithm summary](../edge/exp/exp__wjl__final/humaneval/comparison/table1_scenario1_summary.md)
- [GSM8K four-algorithm summary](../edge/exp/exp__wjl__final/gsm8k/comparison/table1_scenario1_summary.md)

---

## Slide 13 — Four-Algorithm Results: Where the PipeSD Gain Comes From

### Slide Content

#### Performance and Cloud GPU Energy

| Task / Method | TPT (ms/accepted token) | Throughput (accepted token/s) | Cloud GPU Energy (J/100 accepted tokens) |
|---|---:|---:|---:|
| HumanEval Vanilla | 628.705 | 1.591 | 30.060 |
| HumanEval HSL | 773.934 | 1.292 | 29.048 |
| HumanEval EdgeLLM | 1258.404 | 0.795 | 39.091 |
| **HumanEval PipeSD** | **503.444** | **1.986** | **19.121** |
| GSM8K Vanilla | 1062.706 | 0.941 | 45.453 |
| GSM8K HSL | 949.016 | 1.054 | 36.254 |
| GSM8K EdgeLLM | 1498.874 | 0.667 | 54.021 |
| **GSM8K PipeSD** | **872.472** | **1.146** | **34.743** |

#### Pipeline Behavior

| Task / Method | Avg. Draft Length | Total Draft Tokens | Cloud Accept Rate | Total NAVs | Rollback Rate | Promoted | Discard | Discard Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HumanEval Vanilla | 5.881 | 1388 | 72.0% | 236 | 42.8% | 0 | 0 | — |
| HumanEval HSL | 2.955 | 1052 | 95.1% | 356 | 14.3% | 0 | 0 | — |
| HumanEval EdgeLLM | 2.597 | 1496 | 66.8% | 576 | 11.5% | 401 | 354 | 46.9% |
| **HumanEval PipeSD** | **5.279** | **1077** | **92.9%** | **204** | **24.0%** | **528** | **185** | **25.9%** |
| GSM8K Vanilla | 3.797 | 1815 | 55.1% | 478 | 60.9% | 0 | 0 | — |
| GSM8K HSL | 2.998 | 1361 | 73.5% | 454 | 46.5% | 0 | 0 | — |
| GSM8K EdgeLLM | 1.830 | 1464 | 68.3% | 800 | 28.1% | 386 | 694 | 64.3% |
| **GSM8K PipeSD** | **2.995** | **1261** | **79.3%** | **421** | **46.3%** | **443** | **502** | **53.1%** |

> “Total Draft Tokens” counts draft tokens in formal NAV windows. Promoted and Discard count only proactive tokens produced while NAV is pending. Vanilla and HSL generate no proactive tokens, so their Discard Rate is shown as “—”.

### Suggested Visual

Use two grouped TPT bar charts as the main figure. Add three large result cards for the 1.249× HumanEval speedup, the 1.218× GSM8K speedup, and the energy reductions.

### Speaker Notes

On HumanEval, PipeSD reduces TPT from 628.7 ms for Vanilla to 503.4 ms, a 1.249× speedup, while lowering cloud GPU active-compute energy by 36.4%. It uses 204 NAV calls, promotes 528 proactive tokens to the next formal candidates, and discards 185, giving a 25.9% discard rate. On GSM8K, PipeSD reduces TPT from 1062.7 ms to 872.5 ms, a 1.218× speedup, but its discard rate rises to 53.1%, indicating more wasted proactive work on this workload. HSL is the strongest GSM8K baseline, and PipeSD remains 1.088× faster. EdgeLLM is the slowest method on both datasets, which differs from the paper and indicates that hardware, baseline parameters, and network implementation still require further alignment.

#### Metric Definitions

- **Average Draft Length** is the total number of draft tokens in formal NAV windows divided by the total NAV count. **Total Draft Tokens** is the sum of those formal NAV-window lengths and excludes proactive Promoted/Discard, which are reported separately.
- **Cloud Accept Rate** is cloud-accepted draft tokens divided by formally verified draft tokens. **Total NAVs** is the number of verifications actually executed by the cloud.
- **Rollback Rate** is the number of NAV calls that reject at least one draft token divided by the total NAV count. It measures the fraction of NAV rounds with a rollback, not the fraction of rejected tokens.
- **Promoted** counts proactive tokens generated and uploaded while the parent NAV is pending, then promoted to the next formal candidate when the parent draft is fully accepted and the target extra token matches the token assumed by the proactive branch. Promoted does not mean cloud-accepted.
- **Discard** counts proactive tokens removed because the parent draft is not fully accepted, the extra token does not match, or the branch is stale. **Discard Rate** is `Discard / (Promoted + Discard)`. Vanilla and HSL perform no NAV-wait proactive generation, so this rate is undefined and shown as “—”.

For Vanilla, the fixed values are the normal trigger windows: `N=6` on HumanEval and `N=4` on GSM8K. The implementation triggers NAV when the accumulated draft reaches `N`, but it also verifies early when the output-token cap is near, the remaining cloud-accepted-token budget is smaller than `N`, or the draft model emits EOS. The table therefore reports the **mean actual verified length**, which need not equal the configured window. In this run, 225 of 236 HumanEval NAV calls have length 6; the other 11 are short end-of-sample or budget-boundary windows, giving `1388/236=5.881`. On GSM8K, 437 of 478 NAV calls have length 4; 41 are shorter, including 30 triggered early by a draft EOS, giving `1815/478=3.797`. A draft EOS only forces immediate verification—it does not mean the final output ends with EOS, because the cloud may reject it and continue decoding.

---

## Slide 14 — Verifying the Pipeline with a Real Multi-Round Trace

### Slide Content

**Trace Elements and Their Raw-Result Mapping**

- **Draft-generation intervals:** per-token generation timestamps and token indices.
- **Orange transmission blocks:** start time, end time, batch size, and round from `batch_trace`.
- **NAV trigger markers:** trigger condition, trigger timestamp, and current candidate length.
- **Cloud response:** accepted count, additional target token, and NAV latency.
- **Proactive intervals:** next-round generation or upload performed while waiting for NAV.
- **Promoted or discard outcome:** final treatment of proactive work after the parent result returns.

![Multi-round PipeSD trace](./pipesd-humaneval-trace-multi-round.png)

### Suggested Visual

Display the trace nearly full-screen. Animate or highlight one trigger window from left to right, then show how the promotion decision affects the next round.

### Speaker Notes

Aggregate metrics show how much faster the system becomes, but a trace is needed to prove that pipeline overlap actually occurs. This figure shows the first 12 trigger windows for HumanEval Task 50. Draft generation overlaps with multiple small upload batches, and proactive work appears during NAV waiting intervals. Some proactive work is promoted to the next formal candidate, while other work is discarded because the parent round was not fully accepted or because the additional target token did not match. PipeSD therefore does not count every early token as useful work; it hides waiting time only when the strict validation condition permits safe promotion.

### Detailed Trace References

- [Window-by-window multi-round pipeline analysis](./pipesd-humaneval-trace-multi-round-analysis.md)
- [Single-round fields and raw JSON mapping](./pipesd-humaneval-trace-analysis.md)

---

## Slide 15 — Four Deployment Modes: Separating Model, Network, and Pipeline Effects

### Slide Content

| Mode | Model and Execution Path | Network / NAV | Statistical Budget |
|---|---|---|---|
| Pure Cloud | Target model performs complete autoregressive decoding on the single GPU | Warm local request; no client-cloud transfer and no NAV | 1,000 committed output tokens |
| Pure Edge | Draft model performs independent autoregressive decoding on the CPU | No network, cloud, or NAV | 1,000 committed output tokens |
| Serial Edge-Cloud SD | CPU draft is uploaded as one fixed-window block; GPU target performs NAV | 20/200 Mbps software link; generation, upload, and NAV are serial | 1,000 cloud-accepted draft tokens |
| PipeSD | Uses the same draft, target, and software link as the serial mode | DP-batched upload, generation-transfer overlap, and NAV-wait speculation | 1,000 cloud-accepted draft tokens |

| Energy Scope | Included | Excluded |
|---|---|---|
| Pure Cloud | Prompt prefill and complete autoregressive decode | Model load, client transfer, and non-GPU system power |
| Pure Edge | Not measured because RAPL access is unavailable | — |
| Serial Edge-Cloud / PipeSD | Cloud prompt prefill and active target NAV compute | GPU idle between NAVs, edge CPU, network, and whole-system power |

Both datasets use seed `3407`, and every mode records exactly 1,000 benchmark tokens under its own denominator. Only Serial Edge-Cloud SD and PipeSD share the same models, link, normalization token, and cloud-energy boundary.

### Suggested Visual

Draw four horizontal execution paths. Use dashed annotations to show that pure modes bypass network/NAV, and emphasize that Serial Edge-Cloud SD and PipeSD share the same models and link.

### Speaker Notes

The four-mode experiment should not be confused with the paper's four-algorithm comparison. Pure Cloud measures a warm local target-model request, including prefill and complete decode but bypassing client-cloud transport. Pure Edge uses the smaller draft model. These modes provide compute references, not equal-quality service alternatives. The most meaningful comparison is Serial Edge-Cloud SD versus PipeSD: they share the model pair, the 20/200 Mbps software link, the accepted-token budget, and the cloud-energy scope; the main difference is DP batching and pipeline overlap.

### Result Sources

- [HumanEval four-mode summary](../edge/exp/exp__wjl__four__modes__final/humaneval/comparison/four_mode_humaneval.md)
- [GSM8K four-mode summary](../edge/exp/exp__wjl__four__modes__final/gsm8k/comparison/four_mode_gsm8k.md)

---

## Slide 16 — Four-Mode Results: Pipelining Reduces Serial Cloud-Edge Overhead

### Slide Content

#### Latency, Throughput, and Energy

| Task / Mode | TPT (ms/benchmark token) | Throughput (benchmark token/s) | TTFT (ms) | Measured Energy (J/100 benchmark tokens) |
|---|---:|---:|---:|---:|
| HumanEval Pure Cloud* | 4.281 | 233.577 | 36.087 | 176.164 |
| HumanEval Pure Edge† | 45.934 | 21.770 | 444.092 | — |
| HumanEval Serial Edge-Cloud SD | 630.228 | 1.587 | 2596.919 | 30.292 |
| **HumanEval PipeSD** | **503.759** | **1.985** | **2094.993** | **20.026** |
| GSM8K Pure Cloud* | 4.221 | 236.925 | 29.786 | 182.116 |
| GSM8K Pure Edge† | 33.941 | 29.463 | 119.160 | — |
| GSM8K Serial Edge-Cloud SD | 1066.676 | 0.937 | 1748.500 | 45.592 |
| **GSM8K PipeSD** | **876.745** | **1.141** | **1839.752** | **34.930** |

\* Pure Cloud uses committed output tokens and includes prompt prefill plus complete decoding, but excludes client transfer.  
† Pure Edge uses committed output tokens and a non-equivalent draft model; RAPL energy is unavailable. Collaborative modes use cloud-accepted draft tokens, so pure and collaborative absolute values should not be directly ranked.

#### Collaborative-Path Behavior: Like-for-Like Comparison

| Task / Mode | NAV/100 | Draft Length | Cloud Accept Rate | Rollback Rate | Batch | Upload MiB | Uploads |
|---|---:|---:|---:|---:|---:|---:|---:|
| HumanEval Serial Edge-Cloud SD | 23.6 | 5.881 | 72.0% | 42.8% | 5.881 | 384.323 | 266 |
| **HumanEval PipeSD** | **20.4** | **5.279** | **92.9%** | **24.0%** | **1.670** | **401.036** | **823** |
| GSM8K Serial Edge-Cloud SD | 47.8 | 3.797 | 55.1% | 60.9% | 3.797 | 498.593 | 514 |
| **GSM8K PipeSD** | **42.1** | **2.995** | **79.3%** | **46.3%** | **1.371** | **548.189** | **1366** |

**Like-for-Like Result:** Relative to Serial Edge-Cloud SD, PipeSD achieves a **1.251×** speedup and **33.9%** lower cloud active-compute energy on HumanEval, and a **1.217×** speedup with **23.4%** lower energy on GSM8K.

### Suggested Visual

Both layouts place TPT and Energy side by side within each dataset panel. TPT uses a logarithmic axis, and Pure Edge energy is marked N/A. Blue, red, green, and orange identify the four modes, respectively.

![Four-mode performance and energy, horizontal layout](./figures/four_modes_performance_energy_horizontal.png)

![Four-mode performance and energy, vertical layout](./figures/four_modes_performance_energy_vertical.png)

### Speaker Notes

Pure Cloud and Pure Edge have much lower TPT than the collaborative paths, but one bypasses transport under a different token denominator and the other substitutes a smaller model, so they are compute references rather than direct service competitors. In the like-for-like comparison, PipeSD is 1.251× faster on HumanEval and 1.217× faster on GSM8K, while reducing same-scope cloud active-compute energy by 33.9% and 23.4%. PipeSD trades smaller batches and more upload requests for generation-transfer overlap, while improving acceptance and reducing NAV and rollback frequency. HumanEval TTFT falls by about 19.3%, whereas GSM8K TTFT rises by about 5.2%; the pipeline improves overall TPT but does not guarantee a lower first-token delay on every workload.

---

## Slide 17 — Reproduction Status: Completed, Partial, and Missing Components

### Slide Content

| Paper or Project Component | Status | Current Evidence or Missing Work |
|---|---|---|
| Basic cloud-edge draft and NAV loop | Completed | Formal results and generated completions for two tasks |
| DP token-batch pipeline | Completed | Code, batch traces, and multi-round visualization |
| Dual-threshold triggering | Completed | Decision logic and formal run parameters |
| NAV-wait generation and safe promotion | Completed | Parent metadata and promoted/discard statistics |
| Software network and dynamic bandwidth support | Code completed | Shared FIFO and dynamic profile exist; formal Scenario 4 result is missing |
| Bayesian optimization | Partially completed | 16-trial code and output exist; formal-run linkage to the latest BO parameters is incomplete |
| Scenario 1 four-algorithm comparison | Partially completed | Single 1,000-token runs; most manifests are dirty, methods span two commits, and GSM8K sample indices are not fully identical |
| Scenarios 2, 3, and 4 | Not completed | Formal comparative results are missing |
| Bandwidth sweep from Figure 5 | Not completed | No formal 10/20/40/80 Mbps system evaluation |
| Parameter-model validation from Figure 6 | Partially completed | Online estimation exists, but fitted curves and error analysis are missing |
| BO/Grid/Random and fixed-threshold comparisons | Not completed | Tables 3 and 4 have not been reproduced |
| Ablation, scheduling-policy, and overhead studies | Not completed | Formal counterparts of Table 5, Table 6, and appendix comparisons are missing |
| HumanEval and GSM8K correctness | Not completed | Completions exist, but pass@1 and exact match are not reported |
| End-to-end energy and multi-edge clients | Not completed | Only cloud GPU energy is available; multi-client evaluation is missing |
| Automated tests | Partially completed | Cloud: 11/11 passed; Edge: 69/70 passed, with one remaining default-output-directory assertion mismatch |

### Suggested Visual

Use a green, yellow, and gray status matrix. Avoid reducing the work to one vague completion percentage.

### Speaker Notes

The reproduction can be divided into three levels. First, the core protocol loop is complete and supported by trace evidence. Second, the Scenario 1–style evaluation has produced results, but it remains a partial reproduction because of different hardware and model artifacts, single runs, dirty worktrees, and incomplete parameter provenance. Third, the full experimental matrix from the paper—including additional scenarios, bandwidth sweeps, ablations, BO comparisons, and multi-edge tests—has not yet been completed. The automated test suite is also not fully green because one edge-side test still expects an outdated output path. This status breakdown is more accurate and actionable than assigning one overall completion percentage.

---

## Slide 18 — Next Steps and Acceptance Criteria

### Slide Content

**P0 — Strengthen the Existing Evidence**

1. Fix the remaining output-directory test and freeze a clean commit, model revisions and hashes, dependencies, and experiment configurations.
2. Automatically write BO-selected parameters into formal-run manifests and rerun the four algorithms.
3. Use at least three to five seeds per task and report mean ± standard deviation or 95% confidence intervals.
4. Calculate HumanEval pass@1 and GSM8K exact match to verify that acceleration does not reduce output quality.

**P1 — Cover the Paper's Key Experiments**

5. Run Scenarios 2 and 3 and complete the 10/20/40/80 Mbps bandwidth sweep.
6. Define and publish a reproducible Scenario 4 dynamic-bandwidth trace and validate online adaptation.
7. Complete BO versus Grid versus Random, fixed-threshold tests, pipeline and trigger ablations, and DP scheduling-policy comparisons.

**P2 — Strengthen the Testbed and Systems Conclusions**

8. Add edge-side power measurements and report end-to-end energy.
9. Validate absolute performance on available real edge hardware, cloud GPUs, and real network paths.
10. Extend the evaluation to concurrent multi-edge clients and cloud-side queuing.

**Acceptance Criteria**

Reproducible scripts + clean manifests + repeated runs + correctness metrics + raw traces + automatically generated comparison reports.

### Suggested Visual

Use a three-stage roadmap: trustworthy current evidence → paper experiment coverage → broader system validation.

### Speaker Notes

The immediate priority is not simply to add more scenarios. The first goal is to turn the current numbers into auditable evidence by fixing the remaining test, freezing all versions, linking BO parameters to the formal runs, repeating the experiments, and adding correctness evaluation. The second phase expands coverage to the paper's key scenarios, bandwidth conditions, and ablations. The third phase addresses more expensive real-hardware, end-to-end energy, and multi-client tests. The final acceptance criterion should not be merely producing a faster number; another researcher should be able to start from a clean commit and manifest, reproduce the experiment, verify output quality, and trace every aggregate result back to raw events.

---

# Appendix Slides

## Appendix A — Comparison with the Paper's Original Scenario 1 Results

### Paper Results

TPT reported for Scenario 1 in the paper, in ms/token:

| Task | Vanilla | HSL | EdgeLLM | PipeSD | PipeSD vs Vanilla |
|---|---:|---:|---:|---:|---:|
| HumanEval | 194 | 155 | 153 | 129 | 1.50× |
| GSM8K | 193 | 174 | 169 | 145 | 1.33× |

### Current Reproduction Results

| Task | Vanilla | HSL | EdgeLLM | PipeSD | PipeSD vs Vanilla |
|---|---:|---:|---:|---:|---:|
| HumanEval | 628.705 | 773.934 | 1258.404 | 503.444 | 1.249× |
| GSM8K | 1062.706 | 949.016 | 1498.874 | 872.472 | 1.218× |

### Speaker Notes

This table explains why the current work should not be described as reproducing the paper's absolute numbers. The current TPT values are several times higher, and the HSL and EdgeLLM rankings are not fully consistent with the paper. The defensible intermediate conclusions are limited to two points: the core mechanisms operate in the current environment, and PipeSD outperforms Vanilla under the same current protocol. Potential sources of the gap include hardware, inference backend, exact model artifacts, real versus emulated networking, thresholds, sample selection, and implementation details. These factors must be isolated through configuration freezing and targeted ablations.

---

## Appendix B — Formal Result Directories and Raw Field Mapping

### Four-Algorithm Result Directories

- `edge/exp/exp__wjl__final/humaneval/{vanilla,hsl,edgellm,pipesd}/`
- `edge/exp/exp__wjl__final/gsm8k/{vanilla,hsl,edgellm,pipesd}/`
- Aggregated reports are stored in each task's `comparison/table1_scenario1_summary.md`.

### Main Raw Result Fields

- `weighted_tpt_ms`, total elapsed time, and total tokens: primary performance measurements.
- `gpu_energy_per_100_tokens_j`: cloud GPU energy.
- `draft_tokens`, `accepted_tokens`, and `acceptance_rate`: candidate quality and NAV behavior.
- `nav_calls` and `nav_per_100_tokens`: verification frequency.
- `rollback_*`: draft computation invalidated by cloud rejection.
- `batch_trace`: transmission start, end, size, and round for every batch.
- `proactive_reused_*` and `proactive_discarded_*`: raw result fields for NAV-wait work, displayed in the slides as Promoted and Discard.
- Manifest fields: seed, commit, models, network configuration, token budget, and thresholds.

### Four-Mode Result Directories

- `edge/exp/exp__wjl__four__modes__final/humaneval/`
- `edge/exp/exp__wjl__four__modes__final/gsm8k/`
- Aggregated reports are stored under each task's `comparison/four_mode_*.md`.

---

## Appendix C — Paper Testbed and Current Coverage

### Testbed Comparison

| Item | Paper | Current Formal Results |
|---|---|---|
| Edge device | Lenovo ThinkBook 16+, Core Ultra 9 185H, 32 GB, Windows 11 | Current local environment; not the same testbed |
| Cloud device | A800 40 GB, Xeon, Ubuntu 22.04 | Current cloud/service environment; not strictly identical |
| Base network | 20 Mbps uplink, 200 Mbps downlink | Software link at 20/200 Mbps plus 25 ms startup latency |
| Scenario 1 | Laptop | Formal four-algorithm results available |
| Scenario 2 | Emulated 2.5 GHz smartphone | No formal results |
| Scenario 3 | Emulated 1.2 GHz IoT device | No formal results |
| Scenario 4 | Dynamic 10–80 Mbps uplink and 150–280 Mbps downlink, changing every 20 seconds | Dynamic-profile code exists; formal results are missing |
| Statistical budget | 1,000 accepted tokens per method | Exactly 1,000 cloud-accepted draft tokens per method in one run; the precise accepted-token definition still requires alignment with the paper |

### Speaker Notes

The software network reproduces the paper's nominal 20 Mbps uplink and 200 Mbps downlink, but the physical devices and end-to-end deployment are different. The current experiment stops at 1,000 cloud-accepted draft tokens, while the paper states 1,000 accepted tokens without using the repository's newer field distinction. This semantic difference, together with the hardware gap, affects absolute comparability and must remain explicit.

---

## Appendix D — Likely Questions and Suggested Answers

### Question 1 — What happens if the cloud accepts only part of the draft after the edge has already generated and uploaded later tokens?

The later tokens remain isolated speculative or proactive data. Partial acceptance changes the next-round context, so the edge must cancel or discard those tokens. They may already have consumed bandwidth, but they are never committed to the confirmed output sequence.

### Question 2 — Does the cloud always produce one additional token after the accepted prefix?

Yes. Standard speculative decoding requires the target model to provide a continuation token after the accepted prefix. The current cloud NAV path returns this token and uses it to validate the next-round transition.

### Question 3 — What if the cloud accepts the entire draft but its additional token differs from the token assumed by the edge's proactive branch?

The proactive branch still cannot be promoted. Promotion requires both full parent acceptance and an exact additional-token match. Otherwise, the branch is discarded and generation restarts from the target token.

### Question 4 — Why does PipeSD not transmit every token immediately?

Every network request has a fixed startup cost \(\alpha\). Per-token transmission repeatedly pays this cost, while one large transmission eliminates overlap. The DP scheduler uses \(\alpha\), \(\beta\), and \(\gamma\) to find an intermediate batch partition.

### Question 5 — Why are the current measurements much slower than the paper's measurements?

The hardware, inference backend, exact model artifacts, physical network, samples, and parameters are not fully identical. The current numbers support within-environment relative comparisons, but not cross-testbed absolute comparisons.

### Question 6 — Can the current results prove that output quality is preserved?

Not yet. The target-model NAV protocol provides sequence verification, but the current comparison reports do not include HumanEval pass@1 or GSM8K exact match. Adding these quality metrics is a P0 requirement for the next formal evaluation.

---

## Presentation Design Guidelines

### Color and Visual Language

- Use one consistent palette for Slides 3–9: blue for edge generation, orange for upload, red for cloud NAV, green for accepted or promoted work, and gray for discarded work.

### Result Presentation

- On Slides 13 and 16, emphasize TPT and relative speedup. Move dense numerical details to the appendix.
- Keep the cloud-only and edge-only footnotes visible whenever four-mode results are shown, so they are not mistaken for quality-equivalent comparisons.
- Add the following footer to every experimental result slide: task, 1,000 cloud-accepted draft tokens, edge seed 3407, single run, and current software-link configuration.

### Pipeline Explanation

- Display the trace on Slide 14 at nearly full width. Walk through one window from left to right, then explain how the validation result determines cross-round promotion or discard.
