import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "scripts" / "export_good_bolt_donor_patch.py"
    spec = importlib.util.spec_from_file_location("export_good_bolt_donor_patch", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExportGoodBoltDonorPatchTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def create_workspace(self, root: Path) -> Path:
        workspace = root / "good_bolt_assets"
        (workspace / "incoming" / "images").mkdir(parents=True, exist_ok=True)
        (workspace / "sam2" / "core_masks").mkdir(parents=True, exist_ok=True)
        (workspace / "sam2" / "metadata").mkdir(parents=True, exist_ok=True)

        image = Image.new("RGB", (40, 40), (100, 100, 100))
        for x in range(14, 22):
            for y in range(10, 30):
                image.putpixel((x, y), (180, 120, 80))
        image.save(workspace / "incoming" / "images" / "sample_a.jpg")

        mask = Image.new("L", (40, 40), 0)
        for x in range(14, 22):
            for y in range(10, 30):
                mask.putpixel((x, y), 255)
        mask.save(workspace / "sam2" / "core_masks" / "sample-asset.png")

        metadata = {
            "asset_id": "sample-asset",
            "image_name": "sample_a.jpg",
            "source_image": "/root/sd-test/defect_images/sample_a.jpg",
            "core_mask_path": "/root/sd-main/data/bolt_parallel/good_bolt_assets/sam2/core_masks/sample-asset.png",
        }
        (workspace / "sam2" / "metadata" / "sample-asset.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return workspace

    def test_export_donor_patch_writes_rgb_alpha_and_rgba(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.create_workspace(Path(tmp))
            output_dir = Path(tmp) / "out"

            summary = self.module.export_donor_patch(
                workspace,
                "sample-asset",
                output_dir,
                padding=2,
            )

            rgb_path = Path(summary["donor_rgb_path"])
            alpha_path = Path(summary["donor_alpha_path"])
            rgba_path = Path(summary["donor_rgba_path"])
            self.assertTrue(rgb_path.exists())
            self.assertTrue(alpha_path.exists())
            self.assertTrue(rgba_path.exists())
            self.assertEqual(summary["crop_box"], [12, 8, 24, 32])

            with Image.open(rgb_path) as rgb:
                self.assertEqual(rgb.size, (12, 24))
            with Image.open(alpha_path) as alpha:
                self.assertEqual(alpha.size, (12, 24))
                self.assertEqual(alpha.getbbox(), (2, 2, 10, 22))


if __name__ == "__main__":
    unittest.main()
