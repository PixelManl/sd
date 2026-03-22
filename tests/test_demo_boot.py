import subprocess
import sys
import unittest
from pathlib import Path

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
        self.assertEqual(config.base_model, "runwayml/stable-diffusion-inpainting")
        self.assertEqual(config.controlnet_model, "lllyasviel/control_v11p_sd15_canny")
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


if __name__ == "__main__":
    unittest.main()
