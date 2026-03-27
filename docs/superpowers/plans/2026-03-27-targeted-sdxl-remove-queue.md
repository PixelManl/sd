# Targeted SDXL Remove Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default long-running SDXL remove sweep with a small structure-constrained queue that avoids accidental 10-hour parameter scans.

**Architecture:** Keep the change at the queue-builder layer so existing queue execution stays intact. Update the queue manifest contract to emit explicit `mask_mode`-driven tasks, then extend command generation so those new task fields are passed through to `run_sdxl_oriented_batch.py`.

**Tech Stack:** Python, `unittest`, existing SDXL queue builder/runner scripts

---

### Task 1: Lock the new queue contract in tests

**Files:**
- Modify: `tests/test_build_sdxl_experiment_queue.py`
- Modify: `tests/test_run_sdxl_experiment_queue.py`

- [ ] **Step 1: Write failing tests for the new targeted queue shape**

Add assertions for:
- a reduced task count
- no `batch20` / `batch50` / `fullrun` tasks
- explicit `mask_mode` coverage for `oriented` and `root_contact`
- explicit `geometry_prior` coverage for `axis` and `envelope`
- command passthrough for `--mask-mode`

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m uv run python -m unittest discover -s tests -p "test_build_sdxl_experiment_queue.py" -v`

Run: `python -m uv run python -m unittest discover -s tests -p "test_run_sdxl_experiment_queue.py" -v`

Expected: failures showing the current queue still emits the long sweep contract and does not forward new mask arguments.

### Task 2: Replace the default queue with a targeted structure-constrained queue

**Files:**
- Modify: `bolt/generate/scripts/build_sdxl_experiment_queue.py`

- [ ] **Step 1: Update the task payload contract**

Add task fields needed by the smaller queue, at minimum:
- `mask_mode`
- `geometry_prior`
- any explicit occupancy / strength values kept in the targeted probe

- [ ] **Step 2: Rewrite the default queue definition**

Emit a small queue built around:
- `oriented` and `root_contact`
- `axis` and `envelope`
- a narrow set of occupancies / strengths only
- no medium/full-batch expansion tasks

- [ ] **Step 3: Keep queue package output compatible**

Preserve manifest + bash + PowerShell package generation so existing execution flow still works.

### Task 3: Forward the new queue arguments through the runner

**Files:**
- Modify: `bolt/generate/scripts/run_sdxl_experiment_queue.py`
- Test: `tests/test_run_sdxl_experiment_queue.py`

- [ ] **Step 1: Pass new task fields into subprocess commands**

Ensure the queue runner forwards the targeted queue arguments such as `--mask-mode`.

- [ ] **Step 2: Keep backward compatibility**

Use optional field handling so older manifests without new keys still build commands cleanly.

### Task 4: Verify focused regression coverage

**Files:**
- Test: `tests/test_build_sdxl_experiment_queue.py`
- Test: `tests/test_run_sdxl_experiment_queue.py`

- [ ] **Step 1: Run focused tests**

Run: `python -m uv run python -m unittest discover -s tests -p "test_build_sdxl_experiment_queue.py" -v`

Run: `python -m uv run python -m unittest discover -s tests -p "test_run_sdxl_experiment_queue.py" -v`

Expected: all tests pass.

- [ ] **Step 2: Run the broader generate-adjacent regression if needed**

Run: `python -m uv run python -m unittest discover -s tests -p "test_run_sdxl_nut_mainline.py" -v`

Expected: pass, confirming the queue contract still matches the downstream script shape.
