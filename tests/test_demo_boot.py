import subprocess
import sys
import unittest
from pathlib import Path

import torch

from demo.generate_defect import build_sdxl_load_kwargs
from demo.project_boot import build_demo_config


class BuildDemoConfigTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_uses_repo_relative_defaults(self):
        config = build_demo_config([], env={}, repo_root=self.repo_root)

        self.assertEqual(config.input_dir, self.repo_root / "data" / "sg" / "inputs")
        self.assertEqual(config.output_dir, self.repo_root / "data" / "sg" / "outputs")
        self.assertEqual(config.image_path.name, "drone_healthy_facade.jpg")
        self.assertEqual(config.mask_path.name, "crack_position_mask.png")
        self.assertEqual(config.pipeline_kind, "sd15-controlnet")
        self.assertEqual(config.base_model, "runwayml/stable-diffusion-inpainting")
        self.assertEqual(config.controlnet_model, "lllyasviel/control_v11p_sd15_canny")
        self.assertEqual(config.ip_adapter_repo, "")
        self.assertEqual(config.ip_adapter_weight_name, "")
        self.assertEqual(config.ip_adapter_scale, 0.6)
        self.assertEqual(config.reference_image_path.name, "reference_subject.jpg")
        self.assertFalse(config.dry_run)

    def test_prefers_cli_and_env_over_defaults(self):
        env = {
            "SD_BASE_MODEL": "env/base",
            "SD_CONTROLNET_MODEL": "env/control",
        }

        config = build_demo_config(
            [
                "--base-model",
                "cli/base",
                "--controlnet-model",
                "cli/control",
                "--output-dir",
                "custom/out",
                "--dry-run",
            ],
            env=env,
            repo_root=self.repo_root,
        )

        self.assertEqual(config.base_model, "cli/base")
        self.assertEqual(config.controlnet_model, "cli/control")
        self.assertEqual(config.output_dir, self.repo_root / "custom" / "out")
        self.assertTrue(config.dry_run)

    def test_uses_env_models_when_cli_not_set(self):
        env = {
            "SD_BASE_MODEL": "env/base",
            "SD_CONTROLNET_MODEL": "env/control",
        }

        config = build_demo_config([], env=env, repo_root=self.repo_root)

        self.assertEqual(config.base_model, "env/base")
        self.assertEqual(config.controlnet_model, "env/control")

    def test_supports_selecting_sdxl_pipeline(self):
        config = build_demo_config(
            [
                "--pipeline-kind",
                "sdxl-inpaint",
                "--dry-run",
            ],
            env={},
            repo_root=self.repo_root,
        )

        self.assertEqual(config.pipeline_kind, "sdxl-inpaint")
        self.assertEqual(
            config.base_model,
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        )
        self.assertEqual(config.controlnet_model, "")
        self.assertTrue(config.dry_run)

    def test_supports_optional_ip_adapter_configuration(self):
        env = {
            "SD_IP_ADAPTER_REPO": "env/ip-adapter",
            "SD_IP_ADAPTER_WEIGHT_NAME": "env.safetensors",
            "SD_IP_ADAPTER_SCALE": "0.75",
        }

        config = build_demo_config(
            [
                "--pipeline-kind",
                "sdxl-inpaint",
                "--ip-adapter-repo",
                "cli/ip-adapter",
                "--ip-adapter-weight-name",
                "cli.safetensors",
                "--ip-adapter-scale",
                "0.9",
                "--reference-image-name",
                "healthy_bolt_roi.png",
                "--dry-run",
            ],
            env=env,
            repo_root=self.repo_root,
        )

        self.assertEqual(config.ip_adapter_repo, "cli/ip-adapter")
        self.assertEqual(config.ip_adapter_weight_name, "cli.safetensors")
        self.assertEqual(config.ip_adapter_scale, 0.9)
        self.assertEqual(config.reference_image_path.name, "healthy_bolt_roi.png")

    def test_generate_defect_supports_dry_run_without_model_loading(self):
        result = subprocess.run(
            [sys.executable, "demo/generate_defect.py", "--dry-run"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("dry-run", result.stdout.lower())
        self.assertIn("runwayml/stable-diffusion-inpainting", result.stdout)

    def test_generate_defect_supports_sdxl_dry_run(self):
        result = subprocess.run(
            [
                sys.executable,
                "demo/generate_defect.py",
                "--pipeline-kind",
                "sdxl-inpaint",
                "--dry-run",
            ],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("sdxl-inpaint", result.stdout)
        self.assertIn(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            result.stdout,
        )

    def test_generate_defect_reports_ip_adapter_in_dry_run(self):
        result = subprocess.run(
            [
                sys.executable,
                "demo/generate_defect.py",
                "--pipeline-kind",
                "sdxl-inpaint",
                "--ip-adapter-repo",
                "h94/IP-Adapter",
                "--ip-adapter-weight-name",
                "ip-adapter_sdxl.bin",
                "--reference-image-name",
                "healthy_bolt_roi.png",
                "--dry-run",
            ],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("h94/IP-Adapter", result.stdout)
        self.assertIn("healthy_bolt_roi.png", result.stdout)

    def test_sdxl_loader_uses_fp16_safetensors_on_cuda(self):
        kwargs = build_sdxl_load_kwargs("cuda", torch.float16)
        self.assertEqual(kwargs["torch_dtype"], torch.float16)
        self.assertEqual(kwargs["variant"], "fp16")
        self.assertTrue(kwargs["use_safetensors"])


if __name__ == "__main__":
    unittest.main()
