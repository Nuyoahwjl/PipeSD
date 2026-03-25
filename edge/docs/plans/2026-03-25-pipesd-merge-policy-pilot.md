# PipeSD Merge Policy Pilot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a configurable PipeSD merge policy, keep result names distinct across variants, and run a 5-task pilot comparing `dp`, `immediate`, and `no_early`.

**Architecture:** Thread a new CLI argument from parsing into `Decoding`, centralize merge-plan selection in one helper, and encode the chosen policy in PipeSD result filenames so pilot outputs do not collide. Add a small shell driver that runs the three variants back-to-back on the same 5 tasks.

**Tech Stack:** Python, unittest/pytest, bash

---

### Task 1: Add failing tests for merge policy and naming

**Files:**
- Modify: `tests/test_run_edge.py`

**Step 1: Write the failing test**

Add tests that expect:
- `DummyDecoding(..., merge_policy="immediate")._resolve_merge_plan()` returns `[1] * 40`
- `DummyDecoding(..., merge_policy="no_early").exp2path("2.5")` contains `merge=no_early`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_run_edge.py -q`
Expected: FAIL because `_resolve_merge_plan` does not exist and PipeSD result naming does not encode merge policy.

### Task 2: Implement merge policy plumbing and naming

**Files:**
- Modify: `src/util.py`
- Modify: `src/engine.py`

**Step 1: Add CLI/config support**

Add `--merge_policy` with choices `dp`, `immediate`, `no_early`, default `dp`.

**Step 2: Implement minimal code**

- Store `merge_policy` in `Decoding`
- Initialize threshold fields in `__init__`
- Add `_resolve_merge_plan()` helper
- Use helper in `edge_process_draft_model()`
- Include `merge=...` in PipeSD output filenames
- Persist `merge_policy` in each JSON result row

**Step 3: Run targeted tests**

Run: `python3 -m pytest tests/test_run_edge.py -q`
Expected: PASS

### Task 3: Add a dedicated 5-task pilot runner

**Files:**
- Add: `scripts/pilot_pipesd_merge_policies.sh`

**Step 1: Create the script**

The script should:
- Default to `python3`
- Run `app/run_edge.py` three times with identical PipeSD hybrid settings
- Change only `--merge_policy`
- Default to tasks `0..4`

**Step 2: Verify script syntax**

Run: `bash -n scripts/pilot_pipesd_merge_policies.sh`
Expected: PASS

### Task 4: Run verification and the pilot

**Files:**
- Read: result JSON files under `exp/exp__gsm/<dataset>/pipesd/`

**Step 1: Run focused verification**

Run: `python3 -m pytest tests/test_run_edge.py -q`
Expected: PASS

**Step 2: Execute pilot**

Run: `bash scripts/pilot_pipesd_merge_policies.sh`
Expected: three 5-task runs for `dp`, `immediate`, and `no_early`

**Step 3: Summarize output**

Read the produced JSON files and compare latency-related metrics across the three merge policies.
