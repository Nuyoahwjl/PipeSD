# PipeSD Reproduction Progress Report: Slide Content and Speaker Notes

> **Purpose:** This document is designed for an 18–22 minute progress presentation based on the paper, the current repository, and the available evaluation results.  
> **Structure:** 18 main slides and 4 appendix slides. Every slide includes titled sections for slide content, visual design, and speaker notes.  
> **Paper:** [PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](https://arxiv.org/abs/2605.13319)

## Presentation Scope and Terminology

- This work is a **reproduction of the protocol and core mechanisms**, not a complete reproduction of every result in the paper. The current work primarily reproduces the four-algorithm protocol under a Scenario 1–style setting and adds an extended comparison among edge-only, cloud-only, serial cloud-edge, and PipeSD execution.
- The absolute latency in the current experiments is substantially higher than that reported in the paper. The correct claim is: **the core PipeSD mechanisms have been reproduced, and relative performance gains have been observed in the current environment**. It would be inaccurate to claim that the paper's absolute numbers have been reproduced.
- The Git author name found in the repository is `Nuyoahwjl`, rather than `Nuyoshwjl`. This report uses the actual Git author name and covers eight relevant commits.
- The cloud-only result measures target-model decoding only and excludes client transfer and prefill, so it is an idealized lower bound. The edge-only mode uses the smaller draft model, so its output quality is not equivalent to that of the target model. The four-mode experiment is therefore mainly an overhead decomposition, not a direct comparison among four services with equal output quality.
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

- **HumanEval:** PipeSD is **1.320×** faster than Vanilla and reduces cloud GPU energy by **22.2%**.
- **GSM8K:** PipeSD is **1.143×** faster than Vanilla, **1.090×** faster than the best baseline, HSL, and reduces cloud GPU energy by **11.7%**.

**Incomplete Work**

- Complete formal results for Scenarios 2–4, bandwidth sweeps, ablations, BO comparisons, correctness metrics, and confidence intervals from repeated runs.
- Reproduction of the paper's exact hardware, real metropolitan network, and absolute performance numbers.

### Suggested Visual

Use a three-column layout: Completed Work / Key Results / Incomplete Work. Display the key speedup and energy numbers in large type.

### Speaker Notes

The core protocol and evaluation path are now operational. PipeSD outperforms serial Vanilla on both tasks, with a stronger gain on HumanEval. However, “completed” here refers to the availability of the core mechanisms, concurrency control, and an executable evaluation workflow. It does not mean that every table and figure in the paper has been reproduced. The most important remaining limitations are the mismatch in the testbed, incomplete formal experiment coverage, single-run results, and the absence of task-level accuracy evaluation.

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

**Critical Reuse Condition**

Edge-side work generated in advance can be reused only if the target model accepts the entire parent draft and its additional target token matches the token assumed by the proactive branch.

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
6. **Commit or Discard:** proactive data is reused only when the parent draft is fully accepted and the additional token matches; otherwise it is cancelled and discarded.

### Suggested Visual

Use the generated trace as the main visual:

![HumanEval Task 50, first 12 trigger windows](./pipesd-humaneval-trace-multi-round.png)

Use a consistent legend: blue for draft generation, orange for upload batches, red for NAV, green for accepted or reused work, and gray for discarded proactive work.

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

Sending every token immediately would repeatedly pay the fixed startup cost alpha. Sending the entire draft as one batch would avoid repeated startup cost but eliminate generation-transmission overlap. The batch size must therefore balance these two costs. The DP scheduler uses the estimated alpha, beta, and gamma values to select the boundaries that minimize the predicted finish time. The implementation also updates the environment model only after a meaningful change, using a 20% gate to avoid excessive rescheduling under small fluctuations. In the formal results, the average PipeSD batch size is 1.725 for HumanEval and 1.354 for GSM8K, showing that a longer draft is actually transmitted through multiple fine-grained pipeline batches.

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

| Cloud Verification Result | Can Proactive Tokens Be Reused? | Required Action |
|---|---:|---|
| Only part of the parent draft is accepted | No | Roll back tokens after the rejection point and cancel or discard next-round proactive data |
| The full draft is accepted, but the additional target token does not match | No | The proactive context is incorrect; discard it and restart from the target token |
| The full draft is accepted and the additional token matches | Yes | Promote the isolated proactive buffer to the next formal round |

**Current Safety Mechanisms**

- Every batch carries round, parent-round, parent-final-token, and related metadata.
- Proactive and formal batches use separate logical buffers.
- Both channels share one physical `SoftwareLink`, so proactive traffic does not receive fictional extra bandwidth.
- The cloud promotes a buffer only after parent-round verification; otherwise it discards the buffer.

### Suggested Visual

Draw three branches: partial acceptance, full acceptance with mismatch, and full acceptance with match. Only the third branch should lead to a green Commit state.

### Speaker Notes

This is the most common place where an implementation may appear faster while being semantically incorrect. Proactive tokens cannot be reused unconditionally. The parent round must be fully accepted, and the target model's additional token must equal the context token assumed by the proactive branch. The current edge and cloud implementations both validate this metadata and clear the isolated buffers on failure. Proactive upload is allowed, but it represents unconfirmed work and never bypasses NAV. Adding this lifecycle control was one of the most important correctness improvements over earlier versions of the reproduction.

### Implementation Mapping

- [`edge/src/engine.py`](../edge/src/engine.py): proactive sender, reuse, cancellation, and discard logic.
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

The repository already contained a basic PipeSD framework, so the work should not be presented as building the entire system from scratch. The contribution of these commits is the alignment, completion, and validation of the paper's protocol. Early commits established four-algorithm results and experiment organization. Later commits aligned DP, BO, trial isolation, and token budgets. The most important subsequent work addressed true concurrency, safe proactive-data reuse, a shared network link, and traceable result manifests. The final commit added the four-mode evaluation. The key achievement is therefore not merely that the code runs, but that important questions about concurrency, correctness, resource sharing, and evaluation consistency are now represented explicitly in the implementation.

---

## Slide 12 — Four-Algorithm Evaluation Design

### Slide Content

**Tasks and Model Pairs**

- **HumanEval:** code generation using the configured small and large DeepSeek-Coder draft/target pair.
- **GSM8K:** mathematical reasoning using the configured TinyLlama/Llama-2 draft/target pair.

**Algorithms**

- **Vanilla:** uses a fixed draft length, then uploads the complete block and triggers NAV.
- **HSL:** triggers NAV based on token-level confidence.
- **EdgeLLM:** uses cumulative sequence confidence and continues generating while waiting for NAV.
- **PipeSD:** combines DP batch pipelining, dual-threshold NAV, and proactive continuation.

**Controlled Evaluation Protocol**

- Random seed 3407 and **1,000 output tokens** for each method.
- Software link with 2.5 MB/s uplink, 25 MB/s downlink, 25 ms startup latency, and shared-FIFO contention.
- Metrics include weighted TPT, throughput, P50/P95 latency, cloud GPU energy per 100 tokens, NAV frequency, acceptance rate, rollback rate, and average batch size.
- Raw results, manifests, and generated completions are written to each algorithm directory and then summarized under `comparison`.

### Suggested Visual

Use an experiment funnel: identical task, model, network, seed, and token budget → four control strategies → one shared metric aggregation pipeline.

### Speaker Notes

The control variable in this experiment is the triggering and pipeline strategy; the other conditions are kept as consistent as possible. A 1,000-output-token budget is used so that TPT and energy can be normalized by token count. Weighted TPT is calculated from total time divided by total output tokens rather than by averaging per-sample averages. The shared FIFO is important because formal and proactive transfers must not each receive an independent full-speed virtual link. One major limitation remains: the generated completions are saved, but HumanEval pass@1 and GSM8K exact match have not yet been calculated. This experiment therefore evaluates performance and system behavior, not final task accuracy.

### Result Sources

- [HumanEval four-algorithm summary](../edge/exp/exp__wjl__final/humaneval/comparison/table1_scenario1_summary.md)
- [GSM8K four-algorithm summary](../edge/exp/exp__wjl__final/gsm8k/comparison/table1_scenario1_summary.md)

---

## Slide 13 — Four-Algorithm Results: Where the PipeSD Gain Comes From

### Slide Content

#### Performance and Cloud GPU Energy

| Task / Method | TPT (ms/token) | Throughput (token/s) | Cloud GPU Energy (J/100 tokens) |
|---|---:|---:|---:|
| HumanEval Vanilla | 516.587 | 1.936 | 4921.003 |
| HumanEval HSL | 552.550 | 1.810 | 5417.605 |
| HumanEval EdgeLLM | 722.554 | 1.384 | 7174.303 |
| **HumanEval PipeSD** | **391.419** | **2.555** | **3827.028** |
| GSM8K Vanilla | 700.350 | 1.428 | 6900.050 |
| GSM8K HSL | 667.692 | 1.498 | 6632.790 |
| GSM8K EdgeLLM | 841.294 | 1.189 | 8552.868 |
| **GSM8K PipeSD** | **612.468** | **1.633** | **6092.901** |

#### Pipeline Behavior

| Task / Method | Average Draft Length | Acceptance Rate | NAV/100 Tokens | Rollback Rate | Average Upload Batch |
|---|---:|---:|---:|---:|---:|
| HumanEval Vanilla | 5.866 | 70.9% | 19.4 | 44.8% | 5.866 |
| HumanEval PipeSD | 5.808 | 93.7% | 15.6 | 20.5% | 1.725 |
| GSM8K PipeSD | 3.141 | 78.2% | 29.0 | 50.0% | 1.354 |

### Suggested Visual

Use two grouped TPT bar charts as the main figure. Add three large result cards for the 1.320× HumanEval speedup, the 1.143× GSM8K speedup, and the energy reductions.

### Speaker Notes

On HumanEval, PipeSD reduces TPT from 516.6 ms for Vanilla to 391.4 ms, corresponding to a 1.32× speedup. Its acceptance rate increases from 70.9% to 93.7%, while NAV frequency and rollback rate both decrease. The average draft length is similar to Vanilla, but the average upload batch is only 1.725 tokens. This confirms that a similarly long candidate sequence is split into small batches and transmitted as a pipeline rather than uploaded as one block. PipeSD is also the fastest method on GSM8K, but the gain is smaller and its acceptance and rollback behavior is less favorable, showing that the model pair and thresholds still need further tuning. HSL and EdgeLLM are slower than Vanilla in parts of the current results, which differs from the paper and indicates that baseline parameters, hardware, or network conditions are not yet fully aligned.

---

## Slide 14 — Verifying the Pipeline with a Real Multi-Round Trace

### Slide Content

**Trace Elements and Their Raw-Result Mapping**

- **Draft-generation intervals:** per-token generation timestamps and token indices.
- **Orange transmission blocks:** start time, end time, batch size, and round from `batch_trace`.
- **NAV trigger markers:** trigger condition, trigger timestamp, and current candidate length.
- **Cloud response:** accepted count, additional target token, and NAV latency.
- **Proactive intervals:** next-round generation or upload performed while waiting for NAV.
- **Reuse or discard outcome:** final treatment of proactive work after the parent result returns.

![Multi-round PipeSD trace](./pipesd-humaneval-trace-multi-round.png)

### Suggested Visual

Display the trace nearly full-screen. Animate or highlight one trigger window from left to right, then show how the reuse decision affects the next round.

### Speaker Notes

Aggregate metrics show how much faster the system becomes, but a trace is needed to prove that pipeline overlap actually occurs. This figure shows the first 12 trigger windows for HumanEval Task 50. Draft generation overlaps with multiple small upload batches, and proactive work appears during NAV waiting intervals. Some proactive work is reused, while other work is discarded because the parent round was not fully accepted or because the additional target token did not match. PipeSD therefore does not count every early token as useful work; it hides waiting time only when the strict validation condition permits safe reuse.

### Detailed Trace References

- [Window-by-window multi-round pipeline analysis](./pipesd-humaneval-trace-multi-round-analysis.md)
- [Single-round fields and raw JSON mapping](./pipesd-humaneval-trace-analysis.md)

---

## Slide 15 — Four-Mode Evaluation Design: A Different Systems Question

### Slide Content

| Mode | Execution Path | Evaluation Purpose |
|---|---|---|
| Edge-only | Draft model decodes locally at the edge | Provides a small-model edge speed reference, but output quality differs |
| Cloud-only | Target model decodes locally in the cloud | Provides a model-only decoding lower bound; transfer and prefill are excluded |
| Serial cloud-edge | Fixed edge draft → upload → cloud NAV | Represents the non-pipelined Vanilla deployment path |
| PipeSD | Overlapped edge generation and upload with dynamic NAV | Measures the gain from pipeline execution over the serial path |

**Controlled Conditions**

The four modes use the same task configuration, random seed 3407, and 1,000-output-token budget. Reported fields include TPT, throughput, TTFT, and available energy measurements.

### Suggested Visual

Draw four horizontal execution paths. Add explicit dashed annotations showing which costs are excluded from cloud-only mode and that edge-only mode uses a different model-quality level.

### Speaker Notes

The four-mode experiment should not be confused with the paper's four-algorithm comparison. The four-algorithm experiment compares control strategies inside the same cloud-edge framework. The four-mode experiment decomposes the deployment path. Cloud-only appears extremely fast because the current measurement covers target-model decoding only and excludes client transfer and prefill. Edge-only is also fast, but it uses the smaller draft model and is therefore not quality-equivalent to target-model inference. The most meaningful comparison is between serial cloud-edge execution and PipeSD because they share the same target-model service path.

### Result Sources

- [HumanEval four-mode summary](../edge/exp/exp__wjl__four__modes/humaneval/comparison/four_mode_humaneval.md)
- [GSM8K four-mode summary](../edge/exp/exp__wjl__four__modes/gsm8k/comparison/four_mode_gsm8k.md)

---

## Slide 16 — Four-Mode Results: Pipelining Reduces Serial Cloud-Edge Overhead

### Slide Content

| Task / Mode | TPT (ms/token) | Throughput (token/s) | TTFT (ms) | Cloud GPU Energy (J/100 tokens) |
|---|---:|---:|---:|---:|
| HumanEval cloud-only* | 4.068 | 245.795 | 6.693 | 168.247 |
| HumanEval edge-only† | 42.670 | 23.436 | 43.901 | N/A |
| HumanEval serial cloud-edge | 518.481 | 1.929 | 2586.206 | 4930.713 |
| **HumanEval PipeSD** | **393.155** | **2.544** | **2117.161** | **3849.947** |
| GSM8K cloud-only* | 3.989 | 250.661 | 6.179 | 171.317 |
| GSM8K edge-only† | 28.223 | 35.432 | 29.431 | N/A |
| GSM8K serial cloud-edge | 698.932 | 1.431 | 1731.888 | 6846.153 |
| **GSM8K PipeSD** | **610.392** | **1.638** | **1833.356** | **6163.528** |

\* Cloud-only excludes network transfer and prefill.  
† Edge-only uses the smaller draft model, so output quality is not equivalent.

**Most Meaningful Comparison:** Relative to serial cloud-edge execution, PipeSD achieves a **1.319×** speedup on HumanEval and a **1.145×** speedup on GSM8K.

### Suggested Visual

Use serial cloud-edge versus PipeSD TPT as the primary bar chart. Show edge-only and cloud-only as gray reference lines rather than equivalent competitors.

### Speaker Notes

The four-mode results independently confirm the relative pipeline gain: approximately 1.319× on HumanEval and 1.145× on GSM8K compared with the serial cloud-edge path. The cloud-only and edge-only values should not be interpreted as direct alternatives with equal service quality and measurement boundaries. Cloud-only excludes important service costs, while edge-only replaces the target model with the smaller draft model. Another useful observation is that PipeSD has slightly worse TTFT than serial execution on GSM8K. The pipeline primarily improves steady-state inter-token latency and does not guarantee lower first-token latency for every workload.

---

## Slide 17 — Reproduction Status: Completed, Partial, and Missing Components

### Slide Content

| Paper or Project Component | Status | Current Evidence or Missing Work |
|---|---|---|
| Basic cloud-edge draft and NAV loop | Completed | Formal results and generated completions for two tasks |
| DP token-batch pipeline | Completed | Code, batch traces, and multi-round visualization |
| Dual-threshold triggering | Completed | Decision logic and formal run parameters |
| NAV-wait generation and safe reuse | Completed | Parent metadata and reuse/discard statistics |
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
| HumanEval | 516.587 | 552.550 | 722.554 | 391.419 | 1.320× |
| GSM8K | 700.350 | 667.692 | 841.294 | 612.468 | 1.143× |

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
- `proactive_reused_*` and `proactive_discarded_*`: final treatment of work generated during NAV waiting.
- Manifest fields: seed, commit, models, network configuration, token budget, and thresholds.

### Four-Mode Result Directories

- `edge/exp/exp__wjl__four__modes/humaneval/`
- `edge/exp/exp__wjl__four__modes/gsm8k/`
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
| Statistical budget | 1,000 accepted tokens per method | 1,000 output tokens in a single run; the counting convention still requires stricter alignment |

### Speaker Notes

The software network reproduces the paper's nominal 20 Mbps uplink and 200 Mbps downlink, but the physical devices and end-to-end deployment are different. The current experiment also counts 1,000 output tokens, whereas the paper describes aggregation over 1,000 accepted tokens. These distinctions affect absolute comparability and should be recorded explicitly in the next version of the evaluation protocol.

---

## Appendix D — Likely Questions and Suggested Answers

### Question 1 — What happens if the cloud accepts only part of the draft after the edge has already generated and uploaded later tokens?

The later tokens remain isolated speculative or proactive data. Partial acceptance changes the next-round context, so the edge must cancel or discard those tokens. They may already have consumed bandwidth, but they are never committed to the confirmed output sequence.

### Question 2 — Does the cloud always produce one additional token after the accepted prefix?

Yes. Standard speculative decoding requires the target model to provide a continuation token after the accepted prefix. The current cloud NAV path returns this token and uses it to validate the next-round transition.

### Question 3 — What if the cloud accepts the entire draft but its additional token differs from the token assumed by the edge's proactive branch?

The proactive branch still cannot be reused. Promotion requires both full parent acceptance and an exact additional-token match. Otherwise, the branch is discarded and generation restarts from the target token.

### Question 4 — Why does PipeSD not transmit every token immediately?

Every network request has a fixed startup cost \(\alpha\). Per-token transmission repeatedly pays this cost, while one large transmission eliminates overlap. The DP scheduler uses \(\alpha\), \(\beta\), and \(\gamma\) to find an intermediate batch partition.

### Question 5 — Why are the current measurements much slower than the paper's measurements?

The hardware, inference backend, exact model artifacts, physical network, samples, and parameters are not fully identical. The current numbers support within-environment relative comparisons, but not cross-testbed absolute comparisons.

### Question 6 — Can the current results prove that output quality is preserved?

Not yet. The target-model NAV protocol provides sequence verification, but the current comparison reports do not include HumanEval pass@1 or GSM8K exact match. Adding these quality metrics is a P0 requirement for the next formal evaluation.

---

## Presentation Design Guidelines

### Color and Visual Language

- Use one consistent palette for Slides 3–9: blue for edge generation, orange for upload, red for cloud NAV, green for accepted or reused work, and gray for discarded work.

### Result Presentation

- On Slides 13 and 16, emphasize TPT and relative speedup. Move dense numerical details to the appendix.
- Keep the cloud-only and edge-only footnotes visible whenever four-mode results are shown, so they are not mistaken for quality-equivalent comparisons.
- Add the following footer to every experimental result slide: task, 1,000 output tokens, seed 3407, single run, and current software-link configuration.

### Pipeline Explanation

- Display the trace on Slide 14 at nearly full width. Walk through one window from left to right, then explain how the validation result determines cross-round reuse or discard.

