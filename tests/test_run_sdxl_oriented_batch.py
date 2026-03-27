from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt.generate.scripts.run_sdxl_oriented_batch import (
    apply_geometry_prior_to_roi,
    build_control_image_from_roi,
    build_roi_blend_mask,
    build_roi_inpaint_mask,
    build_pipeline,
    filter_records,
    resolve_effective_crop_box,
    resolve_manifest_payload,
    resolve_prompt_config,
    validate_controlnet_config,
    validate_ip_adapter_config,
)


class RunSdxlOrientedBatchTests(unittest.TestCase):
    def test_build_pipeline_loads_lora_when_requested(self) -> None:
        from unittest.mock import MagicMock, patch

        pipe = MagicMock(name="pipe")
        pipe.to.return_value = pipe

        with patch(
            "bolt.generate.scripts.run_sdxl_oriented_batch.StableDiffusionXLInpaintPipeline.from_pretrained",
            return_value=pipe,
        ) as load_pipe:
            result = build_pipeline(
                "base-model",
                lora_path="/tmp/nut_semantic_lora.safetensors",
                lora_scale=0.7,
                lora_adapter_name="nut_semantic_lora",
            )

        self.assertIs(result, pipe)
        load_pipe.assert_called_once()
        pipe.load_lora_weights.assert_called_once_with(
            "/tmp/nut_semantic_lora.safetensors",
            adapter_name="nut_semantic_lora",
        )
        pipe.set_adapters.assert_called_once_with(
            ["nut_semantic_lora"],
            adapter_weights=[0.7],
        )

    def test_filter_records_supports_exact_name_and_stem(self) -> None:
        records = [
            {"image": "1766374553.644332.jpg"},
            {"image": "1766374537.309335.jpg"},
        ]

        selected_by_name = filter_records(records, image_name="1766374553.644332.jpg", limit=0)
        selected_by_stem = filter_records(records, image_name="1766374537.309335", limit=0)

        self.assertEqual(selected_by_name, [{"image": "1766374553.644332.jpg"}])
        self.assertEqual(selected_by_stem, [{"image": "1766374537.309335.jpg"}])

    def test_filter_records_supports_multiple_image_names(self) -> None:
        records = [
            {"image": "a.jpg"},
            {"image": "b.jpg"},
            {"image": "c.jpg"},
        ]

        selected = filter_records(records, image_name=["a.jpg", "c"], limit=0)

        self.assertEqual(selected, [{"image": "a.jpg"}, {"image": "c.jpg"}])

    def test_filter_records_applies_limit_after_selection(self) -> None:
        records = [
            {"image": "a.jpg"},
            {"image": "b.jpg"},
            {"image": "c.jpg"},
        ]

        selected = filter_records(records, image_name=None, limit=2)

        self.assertEqual(selected, [{"image": "a.jpg"}, {"image": "b.jpg"}])

    def test_validate_ip_adapter_config_requires_weight_and_reference_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "weight"):
            validate_ip_adapter_config(
                ip_adapter_repo="/tmp/ip",
                ip_adapter_weight_name="",
                reference_image="ref.png",
            )
        with self.assertRaisesRegex(ValueError, "reference"):
            validate_ip_adapter_config(
                ip_adapter_repo="/tmp/ip",
                ip_adapter_weight_name="ip-adapter.bin",
                reference_image="",
            )

    def test_validate_ip_adapter_config_returns_false_when_disabled(self) -> None:
        self.assertFalse(
            validate_ip_adapter_config(
                ip_adapter_repo="",
                ip_adapter_weight_name="",
                reference_image="",
            )
        )

    def test_validate_ip_adapter_config_returns_true_when_complete(self) -> None:
        self.assertTrue(
            validate_ip_adapter_config(
                ip_adapter_repo="/tmp/ip",
                ip_adapter_weight_name="ip-adapter.bin",
                reference_image="ref.png",
            )
        )

    def test_validate_controlnet_config_returns_false_when_disabled(self) -> None:
        self.assertFalse(
            validate_controlnet_config(
                controlnet_model="",
                controlnet_conditioning_scale=0.8,
                control_image_source="canny",
            )
        )

    def test_validate_controlnet_config_rejects_non_positive_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "conditioning scale"):
            validate_controlnet_config(
                controlnet_model="diffusers/controlnet-canny-sdxl-1.0",
                controlnet_conditioning_scale=0.0,
                control_image_source="canny",
            )

    def test_validate_controlnet_config_rejects_unknown_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "control image source"):
            validate_controlnet_config(
                controlnet_model="diffusers/controlnet-canny-sdxl-1.0",
                controlnet_conditioning_scale=0.8,
                control_image_source="depth",
            )

    def test_validate_controlnet_config_returns_true_when_complete(self) -> None:
        self.assertTrue(
            validate_controlnet_config(
                controlnet_model="diffusers/controlnet-canny-sdxl-1.0",
                controlnet_conditioning_scale=0.8,
                control_image_source="canny",
            )
        )

    def test_resolve_effective_crop_box_keeps_manifest_crop_when_adaptive_disabled(self) -> None:
        crop_box = resolve_effective_crop_box(
            {"crop_box": [100, 120, 220, 260]},
            core_mask=None,
            image_width=512,
            image_height=512,
            adaptive_target_occupancy=0.0,
            adaptive_min_side=0,
        )

        self.assertEqual(crop_box, [100, 120, 220, 260])

    def test_resolve_effective_crop_box_can_fallback_to_xml_box(self) -> None:
        crop_box = resolve_effective_crop_box(
            {"xml_box": [160, 180, 220, 260]},
            core_mask=None,
            image_width=512,
            image_height=512,
            adaptive_target_occupancy=0.25,
            adaptive_min_side=128,
        )

        self.assertEqual(crop_box, [30, 60, 350, 380])

    def test_resolve_effective_crop_box_can_expand_from_mask_occupancy(self) -> None:
        import numpy as np

        core_mask = np.zeros((400, 400), dtype=np.uint8)
        core_mask[160:240, 190:210] = 255

        crop_box = resolve_effective_crop_box(
            {"crop_box": [120, 120, 260, 260]},
            core_mask=core_mask,
            image_width=400,
            image_height=400,
            adaptive_target_occupancy=0.25,
            adaptive_min_side=128,
            adaptive_root_bias=0.0,
        )

        self.assertEqual(crop_box, [40, 40, 360, 360])

    def test_resolve_effective_crop_box_can_bias_toward_root_side(self) -> None:
        import numpy as np

        core_mask = np.zeros((400, 400), dtype=np.uint8)
        core_mask[160:240, 190:210] = 255

        crop_box = resolve_effective_crop_box(
            {"crop_box": [120, 120, 260, 260]},
            core_mask=core_mask,
            image_width=400,
            image_height=400,
            adaptive_target_occupancy=0.25,
            adaptive_min_side=128,
            adaptive_root_bias=1.0,
        )

        self.assertEqual(crop_box, [40, 0, 360, 320])

    def test_apply_geometry_prior_to_roi_is_noop_when_disabled(self) -> None:
        import numpy as np

        roi_rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
        roi_core_mask = np.zeros((64, 64), dtype=np.uint8)
        roi_core_mask[18:46, 28:36] = 255

        seeded, debug = apply_geometry_prior_to_roi(
            roi_rgb,
            roi_core_mask,
            geometry_prior_mode="none",
            geometry_prior_strength=0.55,
        )

        self.assertEqual(int(seeded[30, 32, 0]), 100)
        self.assertEqual(debug, {})

    def test_apply_geometry_prior_to_roi_returns_seeded_roi_and_debug_masks(self) -> None:
        import numpy as np

        roi_rgb = np.full((96, 96, 3), 100, dtype=np.uint8)
        roi_core_mask = np.zeros((96, 96), dtype=np.uint8)
        roi_core_mask[18:78, 44:52] = 255

        seeded, debug = apply_geometry_prior_to_roi(
            roi_rgb,
            roi_core_mask,
            geometry_prior_mode="envelope",
            geometry_prior_strength=0.6,
        )

        self.assertGreater(int(seeded[34, 48, 0]), 100)
        self.assertIn("prior_axis", debug)
        self.assertIn("prior_envelope", debug)
        self.assertIn("prior_preserve_tail", debug)

    def test_build_roi_inpaint_mask_supports_none_full_mode(self) -> None:
        import numpy as np

        roi_core_mask = np.zeros((128, 128), dtype=np.uint8)
        roi_core_mask[44:96, 58:70] = 255

        oriented_mask, oriented_debug = build_roi_inpaint_mask(
            roi_core_mask,
            mask_mode="oriented",
            axial_scale=0.72,
            transverse_scale=1.85,
            min_axial_radius=10.0,
            min_transverse_radius=12.0,
            contact_bias=0.0,
            dilate_ratio=0.26,
            dilate_min_pad=24,
            blur_ksize=0,
        )
        none_full_mask, none_full_debug = build_roi_inpaint_mask(
            roi_core_mask,
            mask_mode="none_full",
            axial_scale=0.72,
            transverse_scale=1.85,
            min_axial_radius=10.0,
            min_transverse_radius=12.0,
            contact_bias=0.0,
            dilate_ratio=0.26,
            dilate_min_pad=24,
            blur_ksize=0,
        )

        self.assertGreater(int(none_full_mask.sum()), int(oriented_mask.sum()))
        self.assertEqual(int((none_full_mask[roi_core_mask > 0] == 255).all()), 1)
        self.assertIn("expanded_mask_box", none_full_debug)
        self.assertNotIn("expanded_mask_box", oriented_debug)

    def test_build_roi_inpaint_mask_supports_root_contact_mode(self) -> None:
        import numpy as np

        roi_core_mask = np.zeros((128, 128), dtype=np.uint8)
        roi_core_mask[28:100, 58:70] = 255

        oriented_mask, _ = build_roi_inpaint_mask(
            roi_core_mask,
            mask_mode="oriented",
            axial_scale=0.72,
            transverse_scale=1.85,
            min_axial_radius=10.0,
            min_transverse_radius=12.0,
            contact_bias=0.0,
            dilate_ratio=0.26,
            dilate_min_pad=24,
            blur_ksize=0,
        )
        root_contact_mask, root_contact_debug = build_roi_inpaint_mask(
            roi_core_mask,
            mask_mode="root_contact",
            axial_scale=0.50,
            transverse_scale=2.45,
            min_axial_radius=10.0,
            min_transverse_radius=12.0,
            contact_bias=0.35,
            dilate_ratio=0.26,
            dilate_min_pad=24,
            blur_ksize=0,
        )

        self.assertGreater(int(root_contact_mask.sum()), 0)
        oriented_ys = np.nonzero(oriented_mask > 0)[0]
        root_contact_ys = np.nonzero(root_contact_mask > 0)[0]
        self.assertGreater(oriented_ys.size, 0)
        self.assertGreater(root_contact_ys.size, 0)
        self.assertLess(float(root_contact_ys.mean()), float(oriented_ys.mean()))
        self.assertEqual(root_contact_debug, {})

    def test_build_roi_blend_mask_can_use_harder_core_dilate_mask(self) -> None:
        import numpy as np

        roi_core_mask = np.zeros((128, 128), dtype=np.uint8)
        roi_core_mask[44:96, 58:70] = 255
        inpaint_mask, _ = build_roi_inpaint_mask(
            roi_core_mask,
            mask_mode="none_full",
            axial_scale=0.72,
            transverse_scale=1.85,
            min_axial_radius=10.0,
            min_transverse_radius=12.0,
            contact_bias=0.0,
            dilate_ratio=0.26,
            dilate_min_pad=24,
            blur_ksize=21,
        )

        blend_mask = build_roi_blend_mask(
            roi_core_mask,
            inpaint_mask,
            source="core_dilate",
            dilate_ratio=0.12,
            min_pad=8,
            blur_ksize=0,
        )

        self.assertEqual(blend_mask.shape, roi_core_mask.shape)
        self.assertEqual(int((blend_mask[roi_core_mask > 0] == 255).all()), 1)
        self.assertLess(int(blend_mask.sum()), int(inpaint_mask.sum()))

    def test_build_roi_blend_mask_can_reuse_inpaint_mask(self) -> None:
        import numpy as np

        roi_core_mask = np.zeros((64, 64), dtype=np.uint8)
        roi_core_mask[18:46, 28:36] = 255
        inpaint_mask = np.zeros((64, 64), dtype=np.uint8)
        inpaint_mask[10:54, 20:44] = 255

        blend_mask = build_roi_blend_mask(
            roi_core_mask,
            inpaint_mask,
            source="inpaint",
            dilate_ratio=0.12,
            min_pad=8,
            blur_ksize=0,
        )

        self.assertTrue((blend_mask == inpaint_mask).all())

    def test_build_control_image_from_roi_returns_rgb_edges(self) -> None:
        import numpy as np

        roi_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        roi_rgb[:, :32] = 30
        roi_rgb[:, 32:] = 220

        control_image = build_control_image_from_roi(
            roi_rgb,
            source="canny",
            target_size=128,
            canny_low_threshold=64,
            canny_high_threshold=192,
        )

        self.assertEqual(control_image.size, (128, 128))
        control_np = np.asarray(control_image)
        self.assertEqual(control_np.shape, (128, 128, 3))
        self.assertGreater(int(control_np.sum()), 0)

    def test_build_pipeline_uses_controlnet_variant_when_requested(self) -> None:
        from unittest.mock import MagicMock, patch

        controlnet = MagicMock(name="controlnet")
        pipe = MagicMock(name="pipe")
        pipe.to.return_value = pipe

        with (
            patch(
                "bolt.generate.scripts.run_sdxl_oriented_batch.ControlNetModel.from_pretrained",
                return_value=controlnet,
            ) as load_controlnet,
            patch(
                "bolt.generate.scripts.run_sdxl_oriented_batch.StableDiffusionXLControlNetInpaintPipeline.from_pretrained",
                return_value=pipe,
            ) as load_pipe,
        ):
            result = build_pipeline(
                "base-model",
                controlnet_model="controlnet-model",
                controlnet_conditioning_scale=0.8,
            )

        self.assertIs(result, pipe)
        load_controlnet.assert_called_once()
        load_pipe.assert_called_once()
        _, kwargs = load_pipe.call_args
        self.assertIs(kwargs["controlnet"], controlnet)

    def test_resolve_manifest_payload_supports_legacy_record_list(self) -> None:
        payload = [
            {
                "image": "a.jpg",
                "xml_box": [10, 20, 30, 50],
            }
        ]

        resolved = resolve_manifest_payload(payload)

        self.assertIn("prompt", resolved)
        self.assertIn("negative_prompt", resolved)
        self.assertEqual(resolved["records"], payload)

    def test_resolve_prompt_config_prefers_cli_overrides(self) -> None:
        manifest = {
            "prompt": "manifest prompt",
            "negative_prompt": "manifest negative",
            "records": [],
        }

        prompt, negative_prompt = resolve_prompt_config(
            manifest,
            prompt_override="cli prompt",
            negative_prompt_override="cli negative",
        )

        self.assertEqual(prompt, "cli prompt")
        self.assertEqual(negative_prompt, "cli negative")


if __name__ == "__main__":
    unittest.main()
