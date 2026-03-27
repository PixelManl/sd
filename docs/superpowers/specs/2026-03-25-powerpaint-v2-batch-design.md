# PowerPaint V2 Batch Design

## Goal

Add a local-only batch workflow for healthy-to-defect conversion that can later call a real `PowerPaint V2` backend to remove nuts/fasteners from marked regions and keep annotations in sync.

## Scope

This change is limited to the bolt generate path:

- batch manifest loading
- sequential target processing for `K` marked regions
- placeholder backend integration point for future `PowerPaint V2`
- VOC / COCO annotation rewrite after each successful target edit
- local-only workflow documentation and focused tests

It does not ship a real `PowerPaint V2` model, installer, or third-party runtime.

## Design

Add a new batch entrypoint that consumes a manifest of per-target edit records. One image with `K` marked targets is represented as `K` ordered records so the workflow can update the image and annotation state after each successful edit.

The runtime is split into four focused units:

- `powerpaint_v2_manifest`: validates and normalizes batch records
- `powerpaint_v2_backend`: provides a placeholder backend contract and future real backend hook
- `powerpaint_v2_annotations`: rewrites VOC XML and COCO JSON without touching source annotations
- `run_powerpaint_v2_batch.py`: orchestrates sequential processing, failure isolation, output layout, and summary export

The first backend mode is a placeholder implementation. It does not remove hardware; it only copies the current image forward so the task graph, output paths, annotation updates, and recovery behavior can be verified before a real `PowerPaint V2` runtime exists locally.

## Input Contract

The batch manifest supports either:

- a JSON object with a top-level `records` list
- a raw JSON list of records

Each record represents one target and must include enough information to:

- resolve the current image
- resolve the source annotation file
- identify the target to remove from annotations
- name the per-step output assets

Minimum fields:

- `image_path`
- `annotation_format` as `voc` or `coco`
- `annotation_path`
- `target_id`
- `output_stem`

Target resolution fields:

- VOC: `bbox` plus optional `class_name`, or `object_index`
- COCO: `annotation_id`, or `image_id` plus `bbox`, with optional `category_id` / `class_name`

## Output Contract

Each batch run writes only local-private outputs under a caller-provided output root:

- `images/`
- `annotations/`
- `manifest_results.json`

Each processed record emits:

- `status`
- `backend_mode`
- `source_image`
- `edited_image`
- `annotation_before`
- `annotation_after`
- `target_id`
- `error_message`

Source images and source annotations are never overwritten.

## Failure Handling

- Records run sequentially in manifest order.
- A failure on one record does not abort the entire batch.
- Annotation rewrite only happens after a backend step succeeds.
- When multiple records belong to the same source image, later records consume the latest successful image and annotation outputs from earlier records.
- `--dry-run` validates inputs and emits a full execution plan without calling any backend or mutating assets.

## Constraints

- Keep the feature local-only and out of Git-tracked asset paths.
- Do not expand the defect scope beyond the current missing-fastener mainline.
- Do not require a real `PowerPaint V2` installation for dry-run or unit tests.
- Keep the real backend integration isolated so later installation does not require rewriting the batch runner.

## Verification

Minimum verification for this change:

- focused annotation rewrite tests for VOC and COCO
- focused batch sequencing test for repeated edits on the same image
- dry-run test for manifest validation and summary export
- focused unittest pass for the new workflow
