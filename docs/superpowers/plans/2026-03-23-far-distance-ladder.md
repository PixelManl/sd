# Far Distance Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable SDXL ladder runner that fixes ROI scale to the current `far` setting and sweeps only edit-mask expansion distance.

**Architecture:** Keep pure geometry and mask-building logic in a small helper module under `bolt/generate/`, then add one CLI runner under `bolt/generate/scripts/` that loads an existing batch manifest, rebuilds per-variant crop and edit masks, calls SDXL inpainting, and writes a comparison manifest. Tests cover the pure helper logic so the experiment runner stays thin.

**Tech Stack:** Python, unittest, NumPy, OpenCV, diffusers, torch, PIL

---

### Task 1: Add pure helper coverage for distance ladder geometry

**Files:**
- Create: `tests/test_distance_ladder.py`
- Create: `bolt/generate/distance_ladder.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Add the SDXL distance-ladder CLI

**Files:**
- Create: `bolt/generate/scripts/run_sdxl_distance_ladder.py`
- Modify: `bolt/generate/README.md`

- [ ] **Step 1: Write the failing test for CLI-adjacent helpers if needed**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run targeted tests**

### Task 3: Verify the experiment path

**Files:**
- Use: `tests/test_distance_ladder.py`
- Use: `tests/test_generate_mask_geometry.py`

- [ ] **Step 1: Run local unit tests**
- [ ] **Step 2: Dry-check the runner arguments**
- [ ] **Step 3: Copy the script to the server and run a 4-variant, 5-image batch**
- [ ] **Step 4: Pull results back to a new private review directory**
