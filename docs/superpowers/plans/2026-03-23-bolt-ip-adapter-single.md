# Bolt IP-Adapter Single-Image Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal single-image bolt IP-Adapter path on top of the existing SDXL ROI inpainting script.

**Architecture:** Extend the existing `run_sdxl_oriented_batch.py` entrypoint instead of creating a parallel pipeline. Keep the original SDXL path unchanged when IP-Adapter options are absent, and add one-image filtering so the same script can serve single-image debugging and future batch reuse.

**Tech Stack:** Python, diffusers SDXL inpainting, IP-Adapter, unittest, existing bolt ROI/mask assets

---

### Task 1: Add test coverage for single-image selection and IP-Adapter validation

**Files:**
- Create: `tests/test_run_sdxl_oriented_batch.py`
- Modify: `bolt/generate/scripts/run_sdxl_oriented_batch.py`

- [ ] **Step 1: Write failing tests for record filtering and IP-Adapter validation**
- [ ] **Step 2: Run the targeted test file and confirm failure**
- [ ] **Step 3: Implement the minimal helpers in the script**
- [ ] **Step 4: Re-run the targeted test file and confirm pass**

### Task 2: Extend the existing ROI script with optional IP-Adapter support

**Files:**
- Modify: `bolt/generate/scripts/run_sdxl_oriented_batch.py`

- [ ] **Step 1: Add CLI flags for `--image-name`, `--reference-image`, and IP-Adapter settings**
- [ ] **Step 2: Keep baseline behavior unchanged when IP-Adapter is not enabled**
- [ ] **Step 3: Load the reference image and pass `ip_adapter_image` only when configured**
- [ ] **Step 4: Write the output manifest with the new run metadata**

### Task 3: Verify locally and run one real server sample

**Files:**
- Modify: `bolt/generate/scripts/run_sdxl_oriented_batch.py`
- Test: `tests/test_run_sdxl_oriented_batch.py`

- [ ] **Step 1: Run local targeted tests**
- [ ] **Step 2: Sync the updated script to the server**
- [ ] **Step 3: Run one real bolt sample with local-path IP-Adapter assets**
- [ ] **Step 4: Pull the result back to local review space**
