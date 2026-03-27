from __future__ import annotations

import importlib.util
import unittest
from argparse import Namespace
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "generate" / "scripts" / "run_sdxl_nut_mainline.py"
    spec = importlib.util.spec_from_file_location("run_sdxl_nut_mainline", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunSdxlNutMainlineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_build_command_forwards_lora_flags_when_enabled(self) -> None:
        args = Namespace(
            batch_manifest=Path("manifest.json"),
            image_dir=Path("images"),
            core_mask_dir=Path("masks"),
            output_dir=Path("out"),
            image_name=[],
            limit=0,
            steps=30,
            guidance_scale=6.0,
            strength=0.92,
            target_size=1024,
            seed_base=700,
            adaptive_target_occupancy=0.20,
            adaptive_min_side=320,
            adaptive_root_bias=0.20,
            mask_mode="none_full",
            mask_dilate_ratio=0.34,
            mask_min_pad=28,
            blur_ksize=21,
            blend_mask_source="core_dilate",
            blend_mask_dilate_ratio=0.10,
            blend_mask_min_pad=8,
            blend_mask_blur_ksize=0,
            geometry_prior="none",
            geometry_prior_strength=0.55,
            controlnet_model="",
            controlnet_conditioning_scale=0.8,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            control_image_source="canny",
            control_canny_low_threshold=100,
            control_canny_high_threshold=200,
            base_model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            prompt="prompt",
            negative_prompt="negative",
            lora_path="/tmp/nut_semantic_lora.safetensors",
            lora_scale=0.75,
            lora_adapter_name="nut_semantic_lora",
            dry_run=True,
            execute=False,
        )

        command = self.module.build_command(args)

        self.assertIn("--lora-path", command)
        self.assertIn("/tmp/nut_semantic_lora.safetensors", command)
        self.assertIn("--lora-scale", command)
        self.assertIn("0.75", command)
        self.assertIn("--lora-adapter-name", command)
        self.assertIn("nut_semantic_lora", command)

    def test_build_plan_exposes_lora_configuration(self) -> None:
        args = Namespace(
            batch_manifest=Path("manifest.json"),
            image_dir=Path("images"),
            core_mask_dir=Path("masks"),
            output_dir=Path("out"),
            image_name=[],
            limit=0,
            steps=30,
            guidance_scale=6.0,
            strength=0.92,
            target_size=1024,
            seed_base=700,
            adaptive_target_occupancy=0.20,
            adaptive_min_side=320,
            adaptive_root_bias=0.20,
            mask_mode="none_full",
            mask_dilate_ratio=0.34,
            mask_min_pad=28,
            blur_ksize=21,
            blend_mask_source="core_dilate",
            blend_mask_dilate_ratio=0.10,
            blend_mask_min_pad=8,
            blend_mask_blur_ksize=0,
            geometry_prior="none",
            geometry_prior_strength=0.55,
            controlnet_model="",
            controlnet_conditioning_scale=0.8,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            control_image_source="canny",
            control_canny_low_threshold=100,
            control_canny_high_threshold=200,
            base_model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            prompt="prompt",
            negative_prompt="negative",
            lora_path="/tmp/nut_semantic_lora.safetensors",
            lora_scale=0.75,
            lora_adapter_name="nut_semantic_lora",
            dry_run=True,
            execute=False,
        )

        plan = self.module.build_plan(args)

        self.assertEqual(plan["lora_path"], "/tmp/nut_semantic_lora.safetensors")
        self.assertEqual(plan["lora_scale"], 0.75)
        self.assertEqual(plan["lora_adapter_name"], "nut_semantic_lora")


if __name__ == "__main__":
    unittest.main()
