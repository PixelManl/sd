import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "scripts" / "export_good_bolt_roi_bank.py"
    spec = importlib.util.spec_from_file_location("export_good_bolt_roi_bank", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExportGoodBoltRoiBankTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def create_workspace(self, root: Path) -> Path:
        workspace = root / "good_bolt_assets"
        (workspace / "incoming" / "images").mkdir(parents=True, exist_ok=True)
        (workspace / "exports" / "healthy_roi_bank").mkdir(parents=True, exist_ok=True)
        (workspace / "manifests").mkdir(parents=True, exist_ok=True)

        Image.new("RGB", (40, 40), (120, 120, 120)).save(
            workspace / "incoming" / "images" / "sample_a.jpg"
        )
        records = [
            {
                "image_name": "sample_a.jpg",
                "source_image": str(workspace / "incoming" / "images" / "sample_a.jpg"),
                "roi_id": "good-bolt-0001",
                "target_class": "healthy_bolt",
                "box_xyxy": [10, 12, 18, 26],
                "qa_state": "usable",
            },
            {
                "image_name": "sample_a.jpg",
                "source_image": str(workspace / "incoming" / "images" / "sample_a.jpg"),
                "roi_id": "good-bolt-0002",
                "target_class": "healthy_bolt",
                "box_xyxy": [2, 2, 6, 6],
                "qa_state": "reject",
            },
        ]
        (workspace / "manifests" / "good_bolt_assets_manifest.jsonl").write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
            encoding="utf-8",
        )
        return workspace

    def test_export_usable_records_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.create_workspace(Path(tmp))

            summary = self.module.export_roi_bank(workspace, padding=4)

            self.assertEqual(summary["exported_count"], 1)
            exported_path = Path(summary["records"][0]["roi_path"])
            self.assertTrue(exported_path.exists())
            with Image.open(exported_path) as image:
                self.assertGreaterEqual(image.size[0], 8)
                self.assertGreaterEqual(image.size[1], 14)


if __name__ == "__main__":
    unittest.main()
