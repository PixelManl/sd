# Bolt Three-Line Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the bolt-missing project around one detection mainline and two support lines: healthy-data acquisition and SAM2 assetization.

**Architecture:** Keep box-based detection as the competition-facing path, run healthy-data collection as a negative-sample and mother-image pipeline, and run SAM2 as a derived-asset pipeline that produces reusable local masks rather than replacing official labels.

**Tech Stack:** Python, uv, box annotations, YOLO-style detector baseline, SAM2, JSON metadata, private local datasets

---

### Task 1: Freeze the private dataset contract

**Files:**
- Modify: `bolt/dataset/README.md`
- Create: `bolt/docs/dataset_contract.md`
- Create: `bolt/docs/dataset_review_checklist.md`

- [ ] Define the canonical private layout under `bolt/dataset/` for `raw/`, `annotations/`, and `derived/`
- [ ] Define the canonical single class name as `missing_fastener`
- [ ] Define required metadata fields for each image and annotation source
- [ ] Document split rules that avoid scene leakage
- [ ] Document a minimal human review checklist for official box labels

### Task 2: Build the detection-ready dataset baseline

**Files:**
- Create: `bolt/detect/README.md`
- Create: `bolt/detect/configs/baseline.yaml`
- Create: `bolt/detect/scripts/prepare_detection_dataset.py`
- Create: `bolt/detect/scripts/report_bbox_stats.py`
- Create: `bolt/detect/scripts/train_baseline.py`
- Create: `bolt/detect/scripts/eval_baseline.py`
- Create: `bolt/detect/scripts/infer_baseline.py`

- [ ] Normalize the current box dataset into one detector-ready format
- [ ] Emit train/val/test manifests with stable IDs
- [ ] Compute bbox size, aspect ratio, and image-relative scale statistics
- [ ] Choose a first lightweight single-class detector baseline
- [ ] Run a first baseline and save metrics plus representative FP/FN examples
- [ ] Record whether tile or patch inference is required

### Task 3: Start the healthy-data screening line

**Files:**
- Modify: `bolt/dataset/README.md`
- Create: `bolt/docs/healthy_data_strategy.md`
- Create: `bolt/scripts/screen_healthy_candidates.py`
- Create: `bolt/scripts/build_healthy_manifest.py`

- [ ] Define what counts as a valid healthy candidate image
- [ ] Separate raw healthy candidates from accepted healthy assets
- [ ] Record source provenance and screening decisions
- [ ] Export a healthy manifest with acceptance status and notes
- [ ] Build a small hard-negative set for detector error analysis

### Task 4: Pilot the SAM2 asset line

**Files:**
- Modify: `bolt/mask/README.md`
- Create: `bolt/docs/sam2_asset_contract.md`
- Create: `bolt/mask/scripts/run_sam2_pilot.py`
- Create: `bolt/mask/scripts/review_sam2_assets.py`
- Create: `bolt/mask/scripts/export_sam2_manifest.py`

- [ ] Define the asset contract for `core_mask`, `edit_mask`, `overlay`, and `metadata`
- [ ] Define QA states such as `usable`, `fix_prompt`, `manual_fix`, and `reject`
- [ ] Run a small pilot batch on representative defect cases
- [ ] Export mask metadata including source box, padded ROI, derived box, area ratio, operator, and QA state
- [ ] Record pilot usable-rate and failure modes before scaling

### Task 5: Connect the three lines

**Files:**
- Create: `bolt/docs/line_intersections.md`
- Modify: `bolt/README.md`

- [ ] Document what the detection line requests from healthy-data and SAM2
- [ ] Document what healthy-data provides back to detection and SAM2
- [ ] Document what SAM2 provides back to detection and later generation work
- [ ] Define which outputs are competition-facing and which are internal enhanced assets

### Task 6: Verify and hand off

**Files:**
- Verify created and modified files only

- [ ] Verify the docs are consistent with the competition-facing box supervision path
- [ ] Verify private paths are still ignored by git
- [ ] Verify each line has a concrete first executable action
- [ ] Review `git status --short`
- [ ] Hand execution to subagents or inline execution with the same three-line split
