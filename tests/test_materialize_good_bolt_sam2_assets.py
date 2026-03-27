import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "mask" / "scripts" / "materialize_good_bolt_sam2_assets.py"
    spec = importlib.util.spec_from_file_location("materialize_good_bolt_sam2_assets", module_path)
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
    edit = Image.new("L", (width, height), 0)
    x1, y1, x2, y2 = asset_context["box_xyxy"]
    for x in range(x1, x2):
        for y in range(y1, y2):
            core.putpixel((x, y), 255)
    for x in range(max(0, x1 - 1), min(width, x2 + 1)):
        for y in range(max(0, y1 - 1), min(height, y2 + 1)):
            edit.putpixel((x, y), 255)
    return {
        "core_mask": core,
        "edit_mask": edit,
        "tool_name": "test-good-bolt-sam2",
        "tool_version": "0.1",
        "qa_state": "candidate",
    }
""".strip(),
        encoding="utf-8",
    )


class MaterializeGoodBoltSam2AssetsTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def create_workspace(self, root: Path) -> Path:
        workspace = root / "good_bolt_assets"
        (workspace / "incoming" / "images").mkdir(parents=True, exist_ok=True)
        (workspace / "dino" / "boxes_json").mkdir(parents=True, exist_ok=True)
        (workspace / "manual_labels" / "healthy_labelme_json").mkdir(parents=True, exist_ok=True)
        for name in ("core_masks", "edit_masks", "overlays", "metadata"):
            (workspace / "sam2" / name).mkdir(parents=True, exist_ok=True)

        Image.new("RGB", (24, 24), (120, 120, 120)).save(
            workspace / "incoming" / "images" / "sample_a.jpg"
        )
        (workspace / "dino" / "boxes_json" / "sample_a.json").write_text(
            json.dumps(
                {
                    "image": "sample_a.jpg",
                    "boxes": [
                        {
                            "box_xyxy": [5, 6, 14, 18],
                            "score": 0.95,
                            "label": "healthy_bolt",
                            "review_state": "approved",
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return workspace

    def write_labelme_annotation(self, workspace: Path) -> Path:
        labelme_path = (
            workspace / "manual_labels" / "healthy_labelme_json" / "sample_a.json"
        )
        labelme_path.write_text(
            json.dumps(
                {
                    "version": "5.11.4",
                    "flags": {},
                    "shapes": [
                        {
                            "label": "GG",
                            "points": [[4, 5], [14, 18]],
                            "group_id": None,
                            "description": "",
                            "shape_type": "rectangle",
                            "flags": {},
                            "mask": None,
                        },
                        {
                            "label": "GG",
                            "points": [[15, 3], [21, 10]],
                            "group_id": None,
                            "description": "",
                            "shape_type": "rectangle",
                            "flags": {},
                            "mask": None,
                        },
                    ],
                    "imagePath": "sample_a.jpg",
                    "imageData": None,
                    "imageHeight": 24,
                    "imageWidth": 24,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return labelme_path

    def write_backend_module_with_empty_masks(self, path: Path) -> None:
        path.write_text(
            """
from PIL import Image


def predictor(source_image, asset_context):
    with Image.open(source_image) as image:
        width, height = image.size
    return {
        "core_mask": Image.new("L", (width, height), 0),
        "edit_mask": Image.new("L", (width, height), 0),
    }
""".strip(),
            encoding="utf-8",
        )

    def test_build_asset_plan_collects_reviewed_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.create_workspace(Path(tmp))

            plan = self.module.build_asset_plan(workspace)

            self.assertEqual(plan["asset_count"], 1)
            self.assertEqual(plan["records"][0]["image_name"], "sample_a.jpg")
            self.assertEqual(plan["records"][0]["box_xyxy"], [5, 6, 14, 18])

    def test_build_asset_plan_collects_labelme_rectangles(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.create_workspace(Path(tmp))
            labelme_dir = workspace / "manual_labels" / "healthy_labelme_json"
            self.write_labelme_annotation(workspace)

            plan = self.module.build_asset_plan(workspace, labelme_dir=labelme_dir)

            self.assertEqual(plan["asset_count"], 2)
            self.assertEqual(plan["records"][0]["image_name"], "sample_a.jpg")
            self.assertEqual(plan["records"][0]["box_xyxy"], [4, 5, 14, 18])
            self.assertEqual(plan["records"][0]["review_state"], "approved")
            self.assertEqual(plan["records"][0]["target_class"], "healthy_bolt")
            self.assertIn("manual", plan["records"][0]["roi_id"])

    def test_build_asset_plan_supports_external_image_dir_for_labelme(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = self.create_workspace(tmp_path)
            labelme_dir = workspace / "manual_labels" / "healthy_labelme_json"
            self.write_labelme_annotation(workspace)
            external_images = tmp_path / "server_defect_images"
            external_images.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 24), (110, 110, 110)).save(external_images / "sample_a.jpg")

            for path in (workspace / "incoming" / "images").glob("*.jpg"):
                path.unlink()

            plan = self.module.build_asset_plan(
                workspace,
                labelme_dir=labelme_dir,
                image_dir=external_images,
            )

            self.assertEqual(plan["asset_count"], 2)
            self.assertEqual(plan["records"][0]["source_image"], str(external_images / "sample_a.jpg"))

    def test_build_asset_plan_accepts_labelme_imagepath_with_parent_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.create_workspace(Path(tmp))
            labelme_dir = workspace / "manual_labels" / "healthy_labelme_json"
            labelme_path = self.write_labelme_annotation(workspace)
            payload = json.loads(labelme_path.read_text(encoding="utf-8"))
            payload["imagePath"] = "nested/sample_a.jpg"
            labelme_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            plan = self.module.build_asset_plan(workspace, labelme_dir=labelme_dir)

            self.assertEqual(plan["asset_count"], 2)

    def test_build_asset_plan_rejects_missing_labelme_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.create_workspace(Path(tmp))

            with self.assertRaises(ValueError):
                self.module.build_asset_plan(
                    workspace,
                    labelme_dir=workspace / "manual_labels" / "missing_dir",
                )

    def test_build_asset_plan_rejects_missing_image_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.create_workspace(Path(tmp))
            labelme_dir = workspace / "manual_labels" / "healthy_labelme_json"
            self.write_labelme_annotation(workspace)

            with self.assertRaises(ValueError):
                self.module.build_asset_plan(
                    workspace,
                    labelme_dir=labelme_dir,
                    image_dir=workspace / "incoming" / "missing_images",
                )

    def test_execute_asset_plan_materializes_masks_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = self.create_workspace(tmp_path)
            backend_path = tmp_path / "good_bolt_backend.py"
            write_backend_module(backend_path)

            plan = self.module.build_asset_plan(workspace)
            backend = self.module.resolve_backend(f"{backend_path}:predictor")
            summary = self.module.execute_asset_plan(plan, workspace, backend)

            self.assertEqual(summary["generated_asset_count"], 1)
            metadata_path = Path(summary["generated_assets"][0]["metadata_path"])
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["target_class"], "healthy_bolt")
            self.assertEqual(payload["qa_state"], "draft")
            self.assertEqual(payload["defect_type"], "healthy_bolt")
            self.assertTrue(payload["pilot_run_id"])
            self.assertTrue(Path(payload["core_mask_path"]).exists())
            self.assertTrue(Path(payload["edit_mask_path"]).exists())
            self.assertTrue(Path(payload["overlay_path"]).exists())

    def test_execute_asset_plan_rejects_duplicate_output_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = self.create_workspace(tmp_path)
            source_image = workspace / "incoming" / "images" / "sample_a.jpg"

            plan = {
                "records": [
                    {
                        "roi_id": "a/b",
                        "image_name": "sample_a.jpg",
                        "source_image": str(source_image),
                        "box_xyxy": [4, 5, 14, 18],
                        "target_class": "healthy_bolt",
                    },
                    {
                        "roi_id": "a:b",
                        "image_name": "sample_a.jpg",
                        "source_image": str(source_image),
                        "box_xyxy": [5, 6, 12, 16],
                        "target_class": "healthy_bolt",
                    },
                ]
            }

            def backend(source_image, asset_context):
                with Image.open(source_image) as image:
                    width, height = image.size
                core = Image.new("L", (width, height), 0)
                edit = Image.new("L", (width, height), 0)
                x1, y1, x2, y2 = asset_context["box_xyxy"]
                for x in range(x1, x2):
                    for y in range(y1, y2):
                        core.putpixel((x, y), 255)
                        edit.putpixel((x, y), 255)
                return {"core_mask": core, "edit_mask": edit}

            with self.assertRaises(ValueError):
                self.module.execute_asset_plan(plan, workspace, backend)

    def test_execute_asset_plan_rejects_empty_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = self.create_workspace(tmp_path)
            backend_path = tmp_path / "good_bolt_empty_backend.py"
            self.write_backend_module_with_empty_masks(backend_path)

            plan = self.module.build_asset_plan(workspace)
            backend = self.module.resolve_backend(f"{backend_path}:predictor")

            with self.assertRaises(ValueError):
                self.module.execute_asset_plan(plan, workspace, backend)

    def test_main_returns_error_for_missing_image_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.create_workspace(Path(tmp))
            argv = [
                "materialize_good_bolt_sam2_assets.py",
                "--workspace",
                str(workspace),
                "--image-dir",
                str(workspace / "incoming" / "missing_images"),
            ]

            with mock.patch.object(sys, "argv", argv):
                exit_code = self.module.main()

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
