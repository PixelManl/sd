# SDXL Inpainting Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional SDXL inpainting backend to the existing demo without breaking the current SD1.5 ControlNet path.

**Architecture:** Extend boot config with a pipeline selector, keep the current default path unchanged, and branch the runtime loader inside `demo/generate_defect.py`. Validate the new behavior with focused unittest coverage and update README usage examples.

**Tech Stack:** Python, unittest, diffusers, torch

---

### Task 1: Add failing tests for pipeline selection

**Files:**
- Modify: `tests/test_demo_boot.py`

- [ ] Add a test for the default pipeline remaining `sd15-controlnet`
- [ ] Add a test for selecting `sdxl-inpaint` via CLI
- [ ] Add a dry-run assertion that SDXL mode is reported in stdout
- [ ] Run the targeted tests and confirm they fail for the missing behavior

### Task 2: Extend boot config for SDXL

**Files:**
- Modify: `demo/project_boot.py`
- Test: `tests/test_demo_boot.py`

- [ ] Add `pipeline_kind` to `DemoConfig`
- [ ] Add CLI and env parsing for pipeline selection
- [ ] Keep SD1.5 defaults unchanged
- [ ] Add SDXL default base model selection when `pipeline_kind=sdxl-inpaint`
- [ ] Run targeted tests and confirm they pass

### Task 3: Branch runtime loading in the demo entrypoint

**Files:**
- Modify: `demo/generate_defect.py`
- Test: `tests/test_demo_boot.py`

- [ ] Keep the current SD1.5 ControlNet branch unchanged
- [ ] Add an SDXL inpainting branch using a diffusers inpainting pipeline
- [ ] Keep dry-run model-free
- [ ] Preserve the existing output path behavior
- [ ] Run targeted tests again

### Task 4: Update README usage

**Files:**
- Modify: `README.md`

- [ ] Document the new pipeline selector
- [ ] Add one SDXL dry-run example
- [ ] Make clear that SD1.5 ControlNet remains the default

### Task 5: Verify end state

**Files:**
- Verify modified files only

- [ ] Run `python -m unittest discover -s tests -p "test_demo_boot.py" -v`
- [ ] Run `python -m py_compile demo/project_boot.py demo/generate_defect.py`
- [ ] Review `git status --short`
