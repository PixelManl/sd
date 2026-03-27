# PowerPaint V2 Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local-only PowerPaint V2 batch workflow skeleton that can process ordered target records, rewrite VOC/COCO annotations, and stay ready for a future real backend.

**Architecture:** Introduce a manifest parser, placeholder backend adapter, annotation rewrite helpers, and a sequential batch runner. The first delivery proves task orchestration and annotation consistency without depending on a real PowerPaint V2 install.

**Tech Stack:** Python, argparse, JSON, XML, Pillow, unittest

---

### Task 1: Add annotation rewrite tests

**Files:**
- Create: `tests/test_powerpaint_v2_annotations.py`
- Create: `bolt/generate/powerpaint_v2_annotations.py`

- [ ] **Step 1: Write failing tests for VOC object removal and COCO annotation removal**
- [ ] **Step 2: Run `python -m uv run python -m unittest discover -s tests -p "test_powerpaint_v2_annotations.py" -v` and confirm failure**
- [ ] **Step 3: Implement minimal annotation rewrite helpers**
- [ ] **Step 4: Re-run `python -m uv run python -m unittest discover -s tests -p "test_powerpaint_v2_annotations.py" -v` and confirm pass**

### Task 2: Add manifest and placeholder backend tests

**Files:**
- Create: `tests/test_run_powerpaint_v2_batch.py`
- Create: `bolt/generate/powerpaint_v2_manifest.py`
- Create: `bolt/generate/powerpaint_v2_backend.py`
- Create: `bolt/generate/scripts/run_powerpaint_v2_batch.py`

- [ ] **Step 1: Write failing tests for manifest normalization, dry-run output, and sequential same-image processing**
- [ ] **Step 2: Run `python -m uv run python -m unittest discover -s tests -p "test_run_powerpaint_v2_batch.py" -v` and confirm failure**
- [ ] **Step 3: Implement the placeholder backend and batch runner with per-record summaries**
- [ ] **Step 4: Re-run `python -m uv run python -m unittest discover -s tests -p "test_run_powerpaint_v2_batch.py" -v` and confirm pass**

### Task 3: Document the workflow

**Files:**
- Create: `bolt/docs/powerpaint_v2_batch_workflow.md`
- Modify: `bolt/docs/README.md`
- Modify: `docs/project_test_registry.md`

- [ ] **Step 1: Document manifest shape, output layout, dry-run behavior, and future real-backend hookup**
- [ ] **Step 2: Add the new workflow link to the bolt docs portal**
- [ ] **Step 3: Register the focused PowerPaint V2 tests in the project test registry**

### Task 4: Verify the skeleton end-to-end

**Files:**
- Test: `tests/test_powerpaint_v2_annotations.py`
- Test: `tests/test_run_powerpaint_v2_batch.py`

- [ ] **Step 1: Run the new focused unittest files**
- [ ] **Step 2: Run one local dry-run invocation of `bolt/generate/scripts/run_powerpaint_v2_batch.py` against a temporary fixture**
- [ ] **Step 3: Summarize what is implemented now versus what still requires a real PowerPaint V2 install**
