# Adaptive ROI Geometry Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight adaptive ROI and geometric-prior preprocessing path for the SDXL bolt inpainting mainline without changing default behavior.

**Architecture:** Keep `run_sdxl_oriented_batch.py` as the main entrypoint, but add two optional preprocessing layers before SDXL inference: ROI normalization from mask occupancy and coarse geometry seeding from the exposed stud mask. Default execution remains identical when the new flags are not enabled.

**Tech Stack:** Python, NumPy, OpenCV, PIL, diffusers SDXL inpainting, unittest

---

### Task 1: Add adaptive ROI helper coverage

**Files:**
- Create: `tests/test_adaptive_roi.py`
- Create: `bolt/generate/adaptive_roi.py`

- [ ] **Step 1: Write failing tests for square ROI sizing from mask occupancy**
- [ ] **Step 2: Run `pytest tests/test_adaptive_roi.py -q` and confirm failure**
- [ ] **Step 3: Implement minimal helper functions**
- [ ] **Step 4: Re-run `pytest tests/test_adaptive_roi.py -q` and confirm pass**

### Task 2: Add geometry prior helper coverage

**Files:**
- Create: `tests/test_geometry_prior.py`
- Create: `bolt/generate/geometry_prior.py`
- Modify: `bolt/generate/mask_geometry.py`

- [ ] **Step 1: Write failing tests for axis normalization and envelope/tail masks**
- [ ] **Step 2: Run `pytest tests/test_geometry_prior.py -q` and confirm failure**
- [ ] **Step 3: Implement the minimal geometry-prior dataclass and builders**
- [ ] **Step 4: Re-run `pytest tests/test_geometry_prior.py -q` and confirm pass**

### Task 3: Wire optional preprocessing into the SDXL script

**Files:**
- Modify: `bolt/generate/scripts/run_sdxl_oriented_batch.py`
- Modify: `tests/test_run_sdxl_oriented_batch.py`

- [ ] **Step 1: Add failing tests for new CLI/helper behavior**
- [ ] **Step 2: Run `pytest tests/test_run_sdxl_oriented_batch.py -q` and confirm failure**
- [ ] **Step 3: Add optional adaptive ROI and geometry prior arguments with safe defaults**
- [ ] **Step 4: Save debug artifacts and keep baseline behavior unchanged when disabled**
- [ ] **Step 5: Re-run `pytest tests/test_run_sdxl_oriented_batch.py -q` and confirm pass**

### Task 4: Verify focused regression

**Files:**
- Test: `tests/test_adaptive_roi.py`
- Test: `tests/test_geometry_prior.py`
- Test: `tests/test_run_sdxl_oriented_batch.py`
- Test: `tests/test_generate_mask_geometry.py`
- Test: `tests/test_distance_ladder.py`

- [ ] **Step 1: Run the focused regression suite**
- [ ] **Step 2: Confirm default script behavior remains intact**
- [ ] **Step 3: Summarize what is implemented now versus what still needs real-image validation**
