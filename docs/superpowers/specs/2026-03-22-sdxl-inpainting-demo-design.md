# SDXL Inpainting Demo Design

## Goal

Keep the current SD1.5 ControlNet inpainting demo intact while adding a second selectable backend for SDXL inpainting.

## Scope

This change is limited to the demo boot and runtime path:

- `demo/project_boot.py`
- `demo/generate_defect.py`
- `tests/test_demo_boot.py`
- `README.md`

It does not redesign the bolt mainline or replace the current default pipeline.

## Design

Add a new config field named `pipeline_kind` with two supported values:

- `sd15-controlnet`
- `sdxl-inpaint`

The existing behavior remains the default:

- default pipeline: `sd15-controlnet`
- default base model for that pipeline: `runwayml/stable-diffusion-inpainting`
- default controlnet model: `lllyasviel/control_v11p_sd15_canny`

When `pipeline_kind` is `sdxl-inpaint`, the demo will:

- use an SDXL inpainting checkpoint as the base model
- skip ControlNet loading
- run the same image + mask generation entrypoint

The first SDXL default will be:

- `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`

## Constraints

- Dry-run must still work without loading any model.
- Existing tests for the default path must keep passing.
- The old CLI must remain valid.
- The new path should be enabled by CLI or environment variable, not by replacing defaults.

## Verification

Minimum verification for this change:

- config parsing tests for default and SDXL selection
- dry-run output test for SDXL mode
- targeted `unittest` pass
