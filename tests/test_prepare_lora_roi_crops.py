import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "generate" / "scripts" / "prepare_lora_roi_crops.py"
    spec = importlib.util.spec_from_file_location("prepare_lora_roi_crops", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepareLoraRoiCropsTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_apply_crop_jitter_keeps_box_in_bounds(self):
        crop_box = self.module.apply_crop_jitter(
            [100, 80, 300, 280],
            image_width=400,
            image_height=360,
            jitter_ratio=0.08,
            seed=11,
        )

        x1, y1, x2, y2 = crop_box
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 400)
        self.assertLessEqual(y2, 360)
        self.assertEqual(x2 - x1, 200)
        self.assertEqual(y2 - y1, 200)

    def test_materialize_roi_crops_uses_manifest_and_core_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            mask_dir = root / "masks"
            output_dir = root / "output"
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            Image.new("RGB", (400, 300), color=(120, 130, 140)).save(image_dir / "sample_a.jpg")
            mask = Image.new("L", (400, 300), color=0)
            for x in range(180, 220):
                for y in range(100, 220):
                    mask.putpixel((x, y), 255)
            mask.save(mask_dir / "sample_a_mask.png")

            manifest = {
                "prompt": "unused",
                "negative_prompt": "unused",
                "records": [
                    {
                        "image": "sample_a.jpg",
                        "xml_box": [185, 110, 215, 210],
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            records = self.module.materialize_roi_crops(
                manifest_path=manifest_path,
                image_dir=image_dir,
                core_mask_dir=mask_dir,
                output_dir=output_dir,
                image_names=[],
                limit=0,
                adaptive_target_occupancy=0.20,
                adaptive_min_side=160,
                adaptive_root_bias=0.20,
                jitter_ratio=0.0,
                seed=7,
            )

            self.assertEqual(len(records), 1)
            crop_image = output_dir / "images" / "sample_a.png"
            self.assertTrue(crop_image.exists())
            crop_meta = json.loads((output_dir / "manifests" / "roi_crops.json").read_text(encoding="utf-8"))
            self.assertEqual(len(crop_meta["records"]), 1)
            self.assertEqual(crop_meta["records"][0]["image_relpath"], "images/sample_a.png")


if __name__ == "__main__":
    unittest.main()
