# Post-SAM2 Queue Trigger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically start the prepared SDXL experiment queue after a SAM2 job has finished, without requiring manual intervention at wake-up time.

**Architecture:** Add a lightweight watcher script beside the existing queue builder and queue runner. The watcher should wait on an explicit trigger source, then invoke the existing queue runner and write a separate watcher journal so launch timing is auditable.

**Tech Stack:** Python, argparse, JSON, subprocess, unittest

---

### Task 1: Add watcher trigger tests

**Files:**
- Create: `tests/test_wait_for_sam2_then_run_queue.py`
- Create: `bolt/generate/scripts/wait_for_sam2_then_run_queue.py`

- [ ] **Step 1: Write failing tests for marker, PID, and command-pattern trigger modes**
- [ ] **Step 2: Run `python -m uv run python -m unittest tests.test_wait_for_sam2_then_run_queue -v` and confirm failure**
- [ ] **Step 3: Implement minimal trigger polling helpers**
- [ ] **Step 4: Re-run `python -m uv run python -m unittest tests.test_wait_for_sam2_then_run_queue -v` and confirm pass**

### Task 2: Launch the existing queue runner safely

**Files:**
- Modify: `bolt/generate/scripts/wait_for_sam2_then_run_queue.py`
- Test: `tests/test_wait_for_sam2_then_run_queue.py`

- [ ] **Step 1: Add queue command construction that reuses `run_sdxl_experiment_queue.py`**
- [ ] **Step 2: Support dry-run mode so launch automation can be validated without consuming GPU**
- [ ] **Step 3: Write a watcher log with trigger metadata, planned command, and final status**

### Task 3: Prepare a private launcher package

**Files:**
- Create: private review launcher under `sd_private_review/...`

- [ ] **Step 1: Point the watcher at the already-generated queue manifest**
- [ ] **Step 2: Emit a ready-to-run PowerShell launcher without starting the queue**
- [ ] **Step 3: Summarize how to use marker/PID/pattern trigger modes on the real machine**
