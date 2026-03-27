import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "mask" / "scripts" / "run_sam2_pilot.py"
    spec = importlib.util.spec_from_file_location("run_sam2_pilot", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_backend_module(path: Path) -> None:
    path.write_text(
        """
from PIL import Image


def predictor(source_image, asset_context):
    with Image.open(source_image) as image:
        width, height = image.size

    core = Image.new("L", (width, height), 0)
    core.putpixel((1, 1), 255)

    edit = Image.new("L", (width, height), 0)
    for x in range(1, min(width, 3)):
        for y in range(1, min(height, 3)):
            edit.putpixel((x, y), 255)

    return {
        "core_mask": core,
        "edit_mask": edit,
        "tool_name": "test-backend",
        "tool_version": "0.1",
        "qa_state": "draft",
    }
""".strip(),
        encoding="utf-8",
    )


class RunSam2PilotTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.repo_root = Path(__file__).resolve().parents[1]

    def create_image(self, path: Path, color: tuple[int, int, int] = (120, 120, 120)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color).save(path)

    def test_build_plan_preserves_plan_only_behavior_without_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            output_root = tmp_path / "private_assets"
            self.create_image(input_dir / "sample_a.jpg")
            self.create_image(input_dir / "sample_b.png")
            (input_dir / "ignore.txt").write_text("not an image", encoding="utf-8")

            plan = self.module.build_plan(
                input_dir=input_dir,
                output_root=output_root,
                pilot_run_id="pilot-20260322-01",
                defect_type="missing_fastener",
                limit=1,
                recursive=False,
                init_layout=False,
            )

            self.assertEqual(plan["status"], "placeholder")
            self.assertEqual(plan["mode"], "sam2-pilot-plan")
            self.assertFalse(plan["sam2_dependency_required"])
            self.assertEqual(plan["source_image_count"], 2)
            self.assertEqual(plan["selected_image_count"], 1)
            self.assertEqual(len(plan["selected_images"]), 1)
            self.assertEqual(plan["planned_outputs"]["assets_dir"], str(output_root / "assets"))

    def test_cli_plan_mode_runs_without_backend_or_sam2(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            output_root = tmp_path / "private_assets"
            self.create_image(input_dir / "sample_a.jpg")

            result = subprocess.run(
                [
                    sys.executable,
                    "bolt/mask/scripts/run_sam2_pilot.py",
                    "--input-dir",
                    str(input_dir),
                    "--output-root",
                    str(output_root),
                    "--pilot-run-id",
                    "pilot-20260322-01",
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["mode"], "sam2-pilot-plan")
            self.assertFalse(payload["sam2_dependency_required"])
            self.assertEqual(payload["selected_image_count"], 1)

    def test_resolve_backend_accepts_python_file_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            backend_path = tmp_path / "sam2_backend.py"
            write_backend_module(backend_path)

            backend = self.module.resolve_backend(f"{backend_path}:predictor")

            self.assertTrue(callable(backend))

    def test_coerce_mask_image_accepts_array_like_mask(self):
        mask = self.module.coerce_mask_image(
            [
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            (4, 4),
            "core_mask",
        )

        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.size, (4, 4))
        self.assertEqual(mask.getpixel((1, 0)), 255)
        self.assertEqual(mask.getpixel((0, 0)), 0)

    def test_execute_plan_materializes_masks_overlay_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            output_root = tmp_path / "private_assets"
            source_image = input_dir / "sample_a.jpg"
            self.create_image(source_image)

            backend_path = tmp_path / "sam2_backend.py"
            write_backend_module(backend_path)
            backend = self.module.resolve_backend(f"{backend_path}:predictor")

            plan = self.module.build_plan(
                input_dir=input_dir,
                output_root=output_root,
                pilot_run_id="pilot-20260322-01",
                defect_type="missing_fastener",
                limit=10,
                recursive=False,
                init_layout=False,
            )

            summary = self.module.execute_plan(
                plan=plan,
                backend=backend,
                backend_label=f"{backend_path}:predictor",
            )

            self.assertEqual(summary["status"], "executed")
            self.assertEqual(summary["mode"], "sam2-pilot-execute")
            self.assertEqual(summary["generated_asset_count"], 1)
            self.assertEqual(len(summary["generated_assets"]), 1)

            metadata_path = Path(summary["generated_assets"][0]["metadata_path"])
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["contract_version"], "sam2_asset_contract/v1")
            self.assertEqual(payload["asset_line"], "sam2_pilot")
            self.assertEqual(payload["qa_state"], "draft")
            self.assertEqual(payload["tool_name"], "test-backend")

            core_mask_path = Path(payload["core_mask_path"])
            edit_mask_path = Path(payload["edit_mask_path"])
            overlay_path = Path(payload["overlay_path"])

            self.assertTrue(core_mask_path.exists())
            self.assertTrue(edit_mask_path.exists())
            self.assertTrue(overlay_path.exists())

            with Image.open(core_mask_path) as core_mask:
                self.assertEqual(core_mask.mode, "L")
                self.assertEqual(core_mask.size, (4, 4))

            with Image.open(edit_mask_path) as edit_mask:
                self.assertEqual(edit_mask.getpixel((1, 1)), 255)
                self.assertEqual(edit_mask.getpixel((0, 0)), 0)


if __name__ == "__main__":
    unittest.main()
