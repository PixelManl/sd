import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "generate" / "scripts" / "prepare_lora_labelme_roi_crops.py"
    spec = importlib.util.spec_from_file_location("prepare_lora_labelme_roi_crops", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepareLoraLabelmeRoiCropsTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_collect_labelme_records_filters_gn_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            labelme_dir = root / "labelme"
            image_dir.mkdir(parents=True, exist_ok=True)
            labelme_dir.mkdir(parents=True, exist_ok=True)

            Image.new("RGB", (200, 160), color=(120, 120, 120)).save(image_dir / "sample_a.jpg")
            payload = {
                "imagePath": "sample_a.jpg",
                "imageWidth": 200,
                "imageHeight": 160,
                "shapes": [
                    {
                        "label": "GG",
                        "shape_type": "rectangle",
                        "points": [[10, 20], [40, 60]],
                    },
                    {
                        "label": "GN",
                        "shape_type": "rectangle",
                        "points": [[50, 30], [90, 100]],
                    },
                ],
            }
            (labelme_dir / "sample_a.json").write_text(json.dumps(payload), encoding="utf-8")

            records = self.module.collect_labelme_records(
                labelme_dir=labelme_dir,
                image_dir=image_dir,
                include_labels=["GN"],
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["manual_label"], "GN")
            self.assertEqual(records[0]["box_xyxy"], [50, 30, 90, 100])
            self.assertEqual(records[0]["sample_id"], "sample_a-manual-002")

    def test_materialize_roi_crops_writes_images_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            labelme_dir = root / "labelme"
            output_dir = root / "output"
            image_dir.mkdir(parents=True, exist_ok=True)
            labelme_dir.mkdir(parents=True, exist_ok=True)

            Image.new("RGB", (220, 180), color=(100, 110, 120)).save(image_dir / "sample_a.jpg")
            payload = {
                "imagePath": "nested/sample_a.jpg",
                "imageWidth": 220,
                "imageHeight": 180,
                "shapes": [
                    {
                        "label": "GN",
                        "shape_type": "rectangle",
                        "points": [[80, 50], [110, 120]],
                    }
                ],
            }
            (labelme_dir / "sample_a.json").write_text(json.dumps(payload), encoding="utf-8")

            records = self.module.collect_labelme_records(
                labelme_dir=labelme_dir,
                image_dir=image_dir,
                include_labels=["GN"],
            )
            summary = self.module.materialize_roi_crops(
                records=records,
                output_dir=output_dir,
                expand_ratio=0.5,
                min_side=96,
                limit=0,
            )

            self.assertEqual(len(summary), 1)
            crop_path = output_dir / "images" / "sample_a-manual-001.png"
            self.assertTrue(crop_path.exists())
            with Image.open(crop_path) as crop:
                self.assertEqual(crop.size[0], crop.size[1])
                self.assertGreaterEqual(crop.size[0], 96)

            manifest_path = output_dir / "manifests" / "roi_crops.json"
            self.assertTrue(manifest_path.exists())
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["record_count"], 1)
            self.assertEqual(payload["records"][0]["manual_label"], "GN")
            self.assertEqual(payload["records"][0]["image_relpath"], "images/sample_a-manual-001.png")


if __name__ == "__main__":
    unittest.main()
