# Multi-Client Server Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a one-server multi-client experimental pipeline for PipeSD, using a single target-model weight with per-task saved context, and generate comparable throughput and cloud-energy results against three existing baselines plus two pure-cloud baselines.

**Architecture:** The cloud server keeps one shared `Llama` model instance but stores a separate `LlamaState` per task and restores it before each verification. The edge side gains a multi-client runner and aggregate result logging. Experiments are first tuned on a tiny Humaneval subset, then run at concurrency `1/2/4` with optional `5/6` extensions.

**Tech Stack:** Python, FastAPI, `llama-cpp-python`, `msgpack`, unittest/pytest-style targeted checks, existing `BandwidthSender` and edge/cloud repos.

---

### Task 1: Document And Freeze The Design

**Files:**
- Create: `edge/docs/plans/2026-03-26-multiclient-server-design.md`
- Create: `edge/docs/plans/2026-03-26-multiclient-server-experiments.md`

**Step 1: Write the design and execution plan**

Write the design choices, experiment matrix, metrics, and implementation tasks into the two plan files above.

**Step 2: Verify files exist**

Run: `ls edge/docs/plans/2026-03-26-multiclient-server-*.md`
Expected: both files listed

**Step 3: Commit**

```bash
git add edge/docs/plans/2026-03-26-multiclient-server-design.md edge/docs/plans/2026-03-26-multiclient-server-experiments.md
git commit -m "docs: add multiclient server design and plan"
```

### Task 2: Add A Failing Regression Test For Cloud Task Isolation

**Files:**
- Modify: `cloud/src/speculative_server.py`
- Create: `cloud/tests/test_speculative_server.py`

**Step 1: Write the failing test**

Create a test that simulates:

1. task A initializes prefix and saves state
2. task B initializes a different prefix and saves state
3. server restores task A and verifies again

Expected behavior:

- task A state is restored instead of overwritten by task B
- task A and task B keep different `n_tokens` / state blobs

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest cloud.tests.test_speculative_server`
Expected: FAIL because the current server shares a single mutable context without per-task restore.

**Step 3: Commit**

```bash
git add cloud/tests/test_speculative_server.py
git commit -m "test: cover multiclient task isolation on cloud server"
```

### Task 3: Implement Per-Task `LlamaState` Save/Restore On The Cloud Server

**Files:**
- Modify: `cloud/src/speculative_server.py`
- Test: `cloud/tests/test_speculative_server.py`

**Step 1: Write minimal implementation**

Add task state management to `InferenceTask` and server flow:

- store `LlamaState` on each task after prefix processing
- before handling a task request that touches model state, restore that task state
- after verify, save updated state back to the task
- remove reliance on `shared_model.change_task(task_id)` as the only task switch mechanism

**Step 2: Run the targeted test**

Run: `python3 -m unittest cloud.tests.test_speculative_server`
Expected: PASS

**Step 3: Run a second targeted single-task sanity test**

Run: `python3 -m unittest cloud.tests.test_speculative_server.CloudServerSingleTaskTests`
Expected: PASS

**Step 4: Commit**

```bash
git add cloud/src/speculative_server.py cloud/tests/test_speculative_server.py
git commit -m "feat: isolate cloud task state with llama save and restore"
```

### Task 4: Add Cloud-Side Run-Level Energy And Makespan Accounting

**Files:**
- Modify: `cloud/src/speculative_server.py`
- Modify: `cloud/src/util.py`
- Test: `cloud/tests/test_speculative_server.py`

**Step 1: Write the failing test**

Create a test for a multi-client run record that checks:

- run start is captured once
- run end is captured after the last task exits
- aggregate energy and makespan are emitted

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest cloud.tests.test_speculative_server.CloudServerRunMetricsTests`
Expected: FAIL

**Step 3: Write minimal implementation**

Add run-scoped accounting for:

- concurrency
- start/end timestamps
- aggregate cloud GPU energy
- aggregate completed samples

**Step 4: Run the targeted tests**

Run: `python3 -m unittest cloud.tests.test_speculative_server`
Expected: PASS

**Step 5: Commit**

```bash
git add cloud/src/speculative_server.py cloud/src/util.py cloud/tests/test_speculative_server.py
git commit -m "feat: record cloud run metrics for multiclient experiments"
```

### Task 5: Add A Local Multi-Client Edge Runner

**Files:**
- Modify: `edge/app/run_edge.py`
- Modify: `edge/src/util.py`
- Create: `edge/tests/test_multiclient_runner.py`

**Step 1: Write the failing test**

Create a test that checks:

- samples are partitioned across `N` clients
- different-sample mode and same-sample mode both work
- runner emits aggregate metadata for one run

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest edge.tests.test_multiclient_runner`
Expected: FAIL

**Step 3: Write minimal implementation**

Add CLI arguments for:

- `--num_clients`
- `--workload_mode`
- `--pilot_samples`
- optional run id / tag

Add orchestration that launches multiple client workers against the same server.

**Step 4: Run the targeted tests**

Run: `python3 -m unittest edge.tests.test_multiclient_runner`
Expected: PASS

**Step 5: Commit**

```bash
git add edge/app/run_edge.py edge/src/util.py edge/tests/test_multiclient_runner.py
git commit -m "feat: add multiclient edge runner"
```

### Task 6: Implement Pure-Cloud Big-Only Baseline

**Files:**
- Modify: `cloud/src/speculative_server.py`
- Modify: `edge/app/run_edge.py`
- Create: `edge/tests/test_pure_cloud_baselines.py`

**Step 1: Write the failing test**

Create a test that asserts the system can trigger a pure-cloud big-only path and log results using the same run metadata format.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest edge.tests.test_pure_cloud_baselines`
Expected: FAIL

**Step 3: Write minimal implementation**

Add a pure-cloud big-only execution path with:

- no edge draft generation
- server-only generation and energy accounting
- compatible result schema

**Step 4: Run the targeted tests**

Run: `python3 -m unittest edge.tests.test_pure_cloud_baselines`
Expected: PASS

**Step 5: Commit**

```bash
git add cloud/src/speculative_server.py edge/app/run_edge.py edge/tests/test_pure_cloud_baselines.py
git commit -m "feat: add pure-cloud big-only baseline"
```

### Task 7: Implement Pure-Cloud Cloud-Speculative Baseline

**Files:**
- Modify: `cloud/src/speculative_server.py`
- Modify: `edge/app/run_edge.py`
- Test: `edge/tests/test_pure_cloud_baselines.py`

**Step 1: Write the failing test**

Add a test that checks:

- cloud-side draft + target path is selectable
- results remain comparable with cloud-edge runs

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest edge.tests.test_pure_cloud_baselines`
Expected: FAIL

**Step 3: Write minimal implementation**

Add a cloud-speculative path that runs both draft and target on the cloud server and reports cloud energy in the same format.

**Step 4: Run the targeted tests**

Run: `python3 -m unittest edge.tests.test_pure_cloud_baselines`
Expected: PASS

**Step 5: Commit**

```bash
git add cloud/src/speculative_server.py edge/app/run_edge.py edge/tests/test_pure_cloud_baselines.py
git commit -m "feat: add pure-cloud speculative baseline"
```

### Task 8: Add Tuning And Aggregation Scripts

**Files:**
- Create: `edge/scripts/run_multiclient_pilot.py`
- Create: `edge/scripts/summarize_multiclient_results.py`
- Create: `edge/tests/test_multiclient_scripts.py`

**Step 1: Write the failing test**

Create tests that cover:

- pilot parameter grid generation
- aggregate summary for throughput and cloud energy

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest edge.tests.test_multiclient_scripts`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- a small sweep over `default_token_compute`, `C`, `bandwidth_MBps`
- result aggregation for `tokens/s`, `samples/s`, energy totals, energy per token, energy per sample

**Step 4: Run the targeted tests**

Run: `python3 -m unittest edge.tests.test_multiclient_scripts`
Expected: PASS

**Step 5: Commit**

```bash
git add edge/scripts/run_multiclient_pilot.py edge/scripts/summarize_multiclient_results.py edge/tests/test_multiclient_scripts.py
git commit -m "feat: add multiclient pilot and summarization scripts"
```

### Task 9: Run A Verified Pilot

**Files:**
- Modify: `edge/docs/rebuttal/2026-03-26-nav-diagnostics-rebuttal.md`
- Create: `edge/exp/...` runtime artifacts only

**Step 1: Run the pilot**

Run the first verified pilot on Humaneval with:

- concurrency `1 / 2 / 4`
- different-sample workload
- `8-16` total samples per method

**Step 2: Verify output artifacts**

Run: `find edge/exp -type f | tail`
Expected: new run outputs exist for the pilot

**Step 3: Summarize the first outcome**

Capture:

- whether PipeSD still beats the three edge baselines
- whether cloud-edge beats either pure-cloud baseline on throughput and cloud energy
- whether `5 / 6` clients are necessary

**Step 4: Commit**

```bash
git add edge/docs/rebuttal/2026-03-26-nav-diagnostics-rebuttal.md
git commit -m "docs: record first multiclient pilot findings"
```
