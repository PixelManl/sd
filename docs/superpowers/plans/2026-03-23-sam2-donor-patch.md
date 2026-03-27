# SAM2 Donor Patch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export reusable donor RGB patches and alpha masks from healthy SAM2 bolt assets.

**Architecture:** Read one healthy SAM2 metadata record, resolve the local source image and core mask paths, crop a tight donor patch around the nonzero mask region, and write RGB plus alpha outputs for downstream reference-guided or copy-paste experiments.

**Tech Stack:** Python, Pillow, unittest, existing local healthy SAM2 assets

---

### Task 1: Add donor crop tests

**Files:**
- Create: `tests/test_export_good_bolt_donor_patch.py`
- Create: `bolt/scripts/export_good_bolt_donor_patch.py`

- [ ] **Step 1: Write failing tests for local path resolution and donor crop export**
- [ ] **Step 2: Run the targeted test file and confirm failure**
- [ ] **Step 3: Implement the minimal donor export helpers**
- [ ] **Step 4: Re-run the targeted test file and confirm pass**

### Task 2: Verify donor export and use one clean asset

**Files:**
- Create: `bolt/scripts/export_good_bolt_donor_patch.py`
- Test: `tests/test_export_good_bolt_donor_patch.py`

- [ ] **Step 1: Run local targeted tests**
- [ ] **Step 2: Export one donor patch from a clean healthy SAM asset**
- [ ] **Step 3: Use the donor patch in a single downstream bolt experiment**
