# Detect Stage

Detection is the final delivery mainline for the bolt-missing task.
This directory now holds a runnable baseline contract for a box-supervised
detector, with local dataset materialization and Ultralytics-oriented entrypoints.

The current repository state is intentionally conservative:

- Official supervision is rectangle boxes.
- This pass defines the baseline path, file contracts, and CLIs.
- Runtime outputs such as `bolt/detect/runs/` remain local-only artifacts.

## Baseline Goal

The immediate goal is to make detection work legible and repeatable before
locking into a framework. The recommended first baseline is a single-class
detector for `missing_fastener`.

The path is:

1. Normalize existing box annotations into a detection-friendly dataset view.
2. Inspect box quality and scale distribution before training.
3. Train a minimal single-class baseline.
4. Evaluate false positives and false negatives on a held-out split.
5. Run inference on images or folders for quick error review.

For now, the scripts in `bolt/detect/scripts/` focus on:

- argparse-based entrypoints
- path validation
- config and dataset contract documentation
- structured dry-run output
- TODO anchors for later framework integration

## Suggested Data Contract

The detector entry now accepts either:

- a Pascal VOC XML directory
- a COCO JSON annotation file

and materializes a YOLO-style dataset view for the first baseline.

Recommended local-only inputs:

- image root: `data/bolt/detect/images/`
- box annotation file: `data/bolt/detect/annotations/instances.json`
- prepared detection root: `data/bolt/detect/prepared/baseline/`

Expected class set for the first pass:

- `missing_fastener`

The current scripts are intentionally narrow:

- `prepare_detection_dataset.py` validates paths, supports COCO/VOC, attaches
  optional metadata, applies group-aware split isolation, and materializes a
  YOLO-style dataset view.
- `report_bbox_stats.py` reads COCO-like box JSON and reports box-scale stats.
- `train_baseline.py` resolves config/paths and prints a training plan.
- `eval_baseline.py` resolves config/paths and prints an evaluation plan.
- `infer_baseline.py` resolves config/paths and prints an inference plan.

If `ultralytics` is available locally, the train/eval/infer scripts can already
execute a YOLO baseline. Runtime artifacts still stay local-only.

## Config Template

Start from:

- `bolt/detect/configs/baseline.yaml`

The config is a baseline contract only. It documents expected paths and key
options such as:

- dataset roots and annotation files
- class names
- split ratios
- model/backend placeholders
- training, evaluation, and inference knobs

The scripts accept explicit CLI overrides so the baseline remains usable even if
`PyYAML` is not installed.

## Example Commands

Prepare dataset skeleton:

```powershell
python -m uv run python bolt/detect/scripts/prepare_detection_dataset.py `
  --images-dir data/bolt/detect/images `
  --annotations data/bolt/detect/annotations/instances.json `
  --metadata data/bolt/detect/annotations/sample_metadata.json `
  --output-dir data/bolt/detect/prepared/baseline `
  --dry-run
```

Recommended metadata fields for leak-safe splitting:

- `sample_id`
- `scene_id`
- `capture_group_id`

Report box stats:

```powershell
python -m uv run python bolt/detect/scripts/report_bbox_stats.py `
  --annotations data/bolt/detect/annotations/instances.json
```

Plan baseline training:

```powershell
python -m uv run python bolt/detect/scripts/train_baseline.py `
  --config bolt/detect/configs/baseline.yaml `
  --dataset-root data/bolt/detect/prepared/baseline `
  --dry-run
```

Plan baseline evaluation:

```powershell
python -m uv run python bolt/detect/scripts/eval_baseline.py `
  --config bolt/detect/configs/baseline.yaml `
  --dataset-root data/bolt/detect/prepared/baseline `
  --split val `
  --dry-run
```

Plan baseline inference:

```powershell
python -m uv run python bolt/detect/scripts/infer_baseline.py `
  --config bolt/detect/configs/baseline.yaml `
  --input data/bolt/detect/images `
  --dry-run
```

## What Is Deliberately Missing

This baseline still does not commit to:

- scene-aware split policy beyond the metadata you provide
- checkpoint selection policy
- training hyperparameter sweep logic
- result visualization standards

Those pieces should be added only after:

1. the box dataset contract is stable
2. group leakage is checked
3. the first YOLO baseline has been run once end-to-end

## Local Output Conventions

Suggested local-only output roots:

- `bolt/detect/runs/baseline/` for training/evaluation outputs
- `bolt/detect/runs/infer/` for prediction previews

Do not commit those outputs.
