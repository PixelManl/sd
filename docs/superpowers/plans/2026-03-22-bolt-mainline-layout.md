# Bolt Mainline Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate the repository into a demo track and a bolt mainline track, with bolt work organized by task stage.

**Architecture:** Keep the repository root as the shared project shell, move demo runtime code into `demo/`, move the missing-fastener pipeline into `bolt/`, and introduce stage-oriented directories under `bolt/` for dataset, mask, generate, detect, and package work. Preserve current minimal verification by updating tests and docs instead of rewriting runtime logic.

**Tech Stack:** Python, uv, unittest, git

---

### Task 1: Create the new top-level layout

**Files:**
- Create: `demo/__init__.py`
- Create: `bolt/README.md`
- Create: `bolt/dataset/README.md`
- Create: `bolt/mask/README.md`
- Create: `bolt/generate/README.md`
- Create: `bolt/detect/README.md`
- Create: `bolt/package/README.md`
- Create: `bolt/docs/README.md`
- Modify: `.gitignore`

- [ ] Add the new tracked folder skeleton for `demo/` and `bolt/`
- [ ] Add ignore rules for private mainline assets under `bolt/`
- [ ] Verify private asset paths remain out of git

### Task 2: Move demo runtime into `demo/`

**Files:**
- Move: `generate_defect.py -> demo/generate_defect.py`
- Move: `project_boot.py -> demo/project_boot.py`
- Move: `batch_augment.py -> demo/batch_augment.py`
- Move: `work_stream.py -> demo/work_stream.py`
- Move: `test_generate.py -> demo/test_generate.py`

- [ ] Move the demo-oriented runtime and legacy demo scripts into `demo/`
- [ ] Update imports and repo-root resolution
- [ ] Keep `demo/generate_defect.py --dry-run` working

### Task 3: Move bolt pipeline into `bolt/`

**Files:**
- Move: `scripts/ -> bolt/scripts/`

- [ ] Move the missing-fastener pipeline under `bolt/scripts/`
- [ ] Keep internal script behavior unchanged unless path fixes are required

### Task 4: Update tests and documentation

**Files:**
- Move: `tests/test_project_boot.py -> tests/test_demo_boot.py`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `.context/index.md`
- Modify: `.context/memory.md`
- Modify: `docs/repo_tree.md`

- [ ] Update tests to target `demo/project_boot.py` and `demo/generate_defect.py`
- [ ] Rewrite repo docs to present `bolt/` as the mainline and `demo/` as a secondary smoke path
- [ ] Keep privacy and git-boundary rules aligned with the new layout

### Task 5: Verify, commit, and push

**Files:**
- Verify modified files only

- [ ] Run `python -m unittest discover -s tests -p "test_*.py" -v`
- [ ] Run `python -m uv run python demo/generate_defect.py --dry-run`
- [ ] Inspect `git status --short`
- [ ] Commit with a layout-focused message
- [ ] Push `main` to origin
