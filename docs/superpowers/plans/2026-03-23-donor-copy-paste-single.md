# Donor Copy-Paste Single-Image Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal single-image donor copy-paste experiment for the bolt target using an existing healthy SAM donor patch.

**Architecture:** Create a small geometry helper that reads donor RGB plus alpha, fits the donor content box into a target ROI box, and composites it with a feathered alpha mask. Keep it standalone so we can evaluate example-based repair without changing the SDXL pipeline.

**Tech Stack:** Python, Pillow, NumPy, unittest

---

### Task 1: Add helper tests

**Files:**
- Create: `tests/test_donor_paste.py`
- Create: `bolt/generate/donor_paste.py`

- [ ] **Step 1: Write failing tests for donor bbox extraction and placement**
- [ ] **Step 2: Run the targeted test file and confirm failure**
- [ ] **Step 3: Implement the minimal helper functions**
- [ ] **Step 4: Re-run the targeted test file and confirm pass**

### Task 2: Add single-image experiment entrypoint

**Files:**
- Create: `bolt/generate/scripts/run_donor_copy_paste_single.py`
- Create: `bolt/generate/donor_paste.py`

- [ ] **Step 1: Wire the helper into a CLI script**
- [ ] **Step 2: Save ROI and full-image outputs plus run metadata**
- [ ] **Step 3: Execute one real experiment on `1766374553.644332`**
