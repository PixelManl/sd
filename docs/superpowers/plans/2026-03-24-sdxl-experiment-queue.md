# SDXL Experiment Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a ready-to-run queued experiment package for the bolt SDXL mainline so long GPU runs can start later without manual command assembly.

**Architecture:** Add a lightweight queue builder that emits a machine-readable task manifest plus shell wrappers. The queue will schedule staged experiments for geometry prior and adaptive ROI, but it will not execute them during build time.

**Tech Stack:** Python, argparse, JSON, unittest

---

### Task 1: Add queue generation tests

**Files:**
- Create: `tests/test_build_sdxl_experiment_queue.py`
- Create: `bolt/generate/scripts/build_sdxl_experiment_queue.py`

- [ ] **Step 1: Write failing tests for queue task structure and output files**
- [ ] **Step 2: Run `pytest tests/test_build_sdxl_experiment_queue.py -q` and confirm failure**
- [ ] **Step 3: Implement minimal queue builder helpers**
- [ ] **Step 4: Re-run `pytest tests/test_build_sdxl_experiment_queue.py -q` and confirm pass**

### Task 2: Emit shell-ready queue assets

**Files:**
- Modify: `bolt/generate/scripts/build_sdxl_experiment_queue.py`

- [ ] **Step 1: Add staged experiment groups (geometry prior, occupancy ladder, batch validation)**
- [ ] **Step 2: Emit `queue_manifest.json`, `run_queue.sh`, and `run_queue.ps1`**
- [ ] **Step 3: Keep outputs data-path only and avoid touching private assets during build**

### Task 3: Verify generated queue package

**Files:**
- Test: `tests/test_build_sdxl_experiment_queue.py`

- [ ] **Step 1: Run the focused queue test**
- [ ] **Step 2: Generate one local sample queue package**
- [ ] **Step 3: Summarize expected run order and approximate runtime budget**
